"""Training script"""
import os
import time
import numpy as np
import torch
from transformers import BertTokenizer
from tqdm import tqdm

from lib.datasets import image_caption
from lib.vse import VSEModel
from lib.evaluation import i2t, t2i, g2i, g2t, AverageMeter, LogCollector, encode_data, compute_sim
from lib.evaluation import i2t_cond_goal, t2i_cond_goal
import evaluation

import logging
import tensorboard_logger as tb_logger

import multiprocessing
from utils.score import caption_scores

import arguments
import json


def main():
    # Hyper Parameters
    parser = arguments.get_argument_parser()
    opt = parser.parse_args()

    # To avoid the dataloader bug
    # modify the multi-threaded tensor mode file_system
    # (the default mode is file_descriptor, limited by the number of open files）
    torch.multiprocessing.set_sharing_strategy('file_system')

    # For reproducibility - refer https://pytorch.org/docs/stable/notes/randomness.html
    # These five lines control all the major sources of randomness.
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.benchmark = False

    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    logger = logging.getLogger(__name__)
    logger.info(opt)

    # Load Tokenizer and Vocabulary
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    train_loader, val_loader = image_caption.get_loaders(
        opt.data_path, opt.data_name, tokenizer, opt.batch_size, opt.workers, opt)

    model = VSEModel(opt)

    lr_schedules = [opt.lr_update, ]

    # optionally resume from a checkpoint
    start_epoch = 0
    if opt.resume:
        if os.path.isfile(opt.resume):
            logger.info("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            if not model.is_data_parallel:
                model.make_data_parallel()
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another training
            model.Eiters = checkpoint['Eiters']
            logger.info("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                        .format(opt.resume, start_epoch, best_rsum))
            # validate(opt, val_loader, model)
            if opt.reset_start_epoch:
                start_epoch = 0
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.resume))

    if not model.is_data_parallel:
        model.make_data_parallel()

    # Train the Model
    best_rsum = 0
    for epoch in range(start_epoch, opt.num_epochs):
        logger.info(opt.logger_name)
        logger.info(opt.model_name)

        adjust_learning_rate(opt, model.optimizer, epoch, lr_schedules)

        if epoch >= opt.vse_mean_warmup_epochs:
            opt.max_violation = True
            model.set_max_violation(opt.max_violation)

        # Set up the all warm-up options
        if opt.precomp_enc_type == 'backbone':
            if epoch < opt.embedding_warmup_epochs:
                model.freeze_backbone()
                logger.info('All backbone weights are frozen, only train the embedding layers')
            else:
                model.unfreeze_backbone(3)

            if epoch < opt.embedding_warmup_epochs:
                logger.info('Warm up the embedding layers')
            elif epoch < opt.embedding_warmup_epochs + opt.backbone_warmup_epochs:
                model.unfreeze_backbone(3)  # only train the last block of resnet backbone
            elif epoch < opt.embedding_warmup_epochs + opt.backbone_warmup_epochs * 2:
                model.unfreeze_backbone(2)
            elif epoch < opt.embedding_warmup_epochs + opt.backbone_warmup_epochs * 3:
                model.unfreeze_backbone(1)
            else:
                model.unfreeze_backbone(0)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if not os.path.exists(opt.model_name):
            os.mkdir(opt.model_name)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, filename='checkpoint.pth'.format(epoch), prefix=opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    logger = logging.getLogger(__name__)
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    logger.info('image encoder trainable parameters: {}'.format(count_params(model.img_enc)))
    logger.info('caption encoder trainable parameters: {}'.format(count_params(model.cap_enc)))
    logger.info('rnn encoder trainable parameters: {}'.format(count_params(model.rnn_enc)))
    logger.info('caption decoder trainable parameters: {}'.format(count_params(model.captioner)))
    logger.info('multimodal encoder trainable parameters: {}'.format(count_params(model.multimodal_enc)))

    num_loader_iter = len(train_loader.dataset) // train_loader.batch_size + 1

    end = time.time()
    # opt.viz = True
    for i, train_data in enumerate(train_loader):
        # switch to train mode
        model.train_start()

        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        if opt.precomp_enc_type == 'basic':
            images, img_lengths, cap_targets, cap_lengths, goal_tokens, goal_sentences, ids, image_ids, goal_ids, batch_ids = train_data
            model.train_emb(images, cap_targets, cap_lengths, goal_tokens, goal_sentences, batch_ids, image_lengths=img_lengths)
        else:
            images, cap_targets, cap_lengths, caption_map, image_map, goal_tokens, goal_sentences, ids, image_ids, goal_ids, batch_ids = train_data
            if epoch == opt.embedding_warmup_epochs:
                warmup_alpha = float(i) / num_loader_iter
                model.train_emb(images, cap_targets, cap_lengths, caption_map, image_map, goal_tokens, goal_sentences, batch_ids, warmup_alpha=warmup_alpha)
            else:
                model.train_emb(images, cap_targets, cap_lengths, caption_map, image_map, goal_tokens, goal_sentences, batch_ids)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # logger.info log info
        if model.Eiters % opt.log_step == 0:
            if opt.precomp_enc_type == 'backbone' and epoch == opt.embedding_warmup_epochs:
                logging.info('Current epoch-{}, the first epoch for training backbone, warmup alpha {}'.format(epoch,
                                                                                                               warmup_alpha))
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                '{e_log}\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                    .format(
                    epoch, i, len(train_loader.dataset) // train_loader.batch_size + 1, batch_time=batch_time,
                    data_time=data_time, e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)


def validate(opt, val_loader, model):
    logger = logging.getLogger(__name__)
    tokenizer_pool = multiprocessing.Pool()
    model.val_start()
    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        predictions, ground_truth, attention = encode_data(
            model, val_loader, opt.log_step, logging.info, backbone=opt.precomp_enc_type == 'backbone')

    start = time.time()

    goal_id_to_index_dict = dict()
    for index in range(len(predictions)):
        goal_id_to_index_dict[predictions[index]["goal_id"]] = index
    # construct cap_gens and cap_gts
    cap_gens = list()
    cap_gts = list()
    for pred in predictions:
        pred_goal_id = pred['goal_id']
        pred_caption = pred["caption"]
        cap_gens.append(pred_caption)

        gt_sentences = [ground_truth[goal_id_to_index_dict[pred_goal_id]]["caption"]]
        cap_gts.append(gt_sentences)

    cap_gens_token, cap_gts_token = tokenizer_pool.map(evaluation.PTBTokenizer.tokenize, [cap_gens, cap_gts])

    scores, all_scores = caption_scores(cap_gens_token, cap_gts_token)

    results = {"predictions": predictions, "ground_truth": ground_truth, "scores": scores}
    with open(os.path.join(opt.model_name, "validation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    end = time.time()
    logger.info("calculate evaluation time: {}".format(end - start))

    METEOR = scores["METEOR"]
    ROUGE = scores["ROUGE"]
    CIDEr = scores["CIDEr"]
    SPICE = scores["SPICE"]
    BLEU1 = scores["BLEU"][0]
    BLEU2 = scores["BLEU"][1]
    BLEU3 = scores["BLEU"][2]
    BLEU4 = scores["BLEU"][3]

    logging.info("BLEU1: %.3f," % BLEU1)
    logging.info("BLEU2: %.3f," % BLEU2)
    logging.info("BLEU3: %.3f," % BLEU3)
    logging.info("BLEU4: %.3f," % BLEU4)
    logging.info("METEOR: %.3f," % METEOR)
    logging.info("ROUGE: %.3f," % ROUGE)
    logging.info("CIDEr: %.3f," % CIDEr)
    logging.info("SPICE: %.3f," % SPICE)

    currscore = CIDEr
    logger.info('Current currscore is {}'.format(currscore))

    # record metrics in tensorboard
    tb_logger.log_value('metrics/BLEU1', BLEU1, step=model.Eiters)
    tb_logger.log_value('metrics/BLEU2', BLEU2, step=model.Eiters)
    tb_logger.log_value('metrics/BLEU3', BLEU3, step=model.Eiters)
    tb_logger.log_value('metrics/BLEU4', BLEU4, step=model.Eiters)
    tb_logger.log_value('metrics/METEOR', METEOR, step=model.Eiters)
    tb_logger.log_value('metrics/ROUGE', ROUGE, step=model.Eiters)
    tb_logger.log_value('metrics/CIDEr', CIDEr, step=model.Eiters)
    tb_logger.log_value('metrics/SPICE', SPICE, step=model.Eiters)

    return currscore


def save_checkpoint(state, is_best, filename='checkpoint.pth', prefix=''):
    logger = logging.getLogger(__name__)
    tries = 15

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                torch.save(state, prefix + 'model_best.pth')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        logger.info('model save {} failed, remaining {} trials'.format(filename, tries))
        if not tries:
            raise error


def adjust_learning_rate(opt, optimizer, epoch, lr_schedules):
    logger = logging.getLogger(__name__)
    """Sets the learning rate to the initial LR
       decayed by 10 every opt.lr_update epochs"""
    if epoch in lr_schedules:
        logger.info('Current epoch num is {}, decrease all lr by 10'.format(epoch, ))
        for param_group in optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = old_lr * 0.1
            param_group['lr'] = new_lr
            logger.info('new lr {}'.format(new_lr))


def count_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


if __name__ == '__main__':
    main()
