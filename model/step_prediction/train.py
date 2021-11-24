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
from lib.evaluation import single_retrieval
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
    logger.info('goal encoder trainable parameters: {}'.format(count_params(model.goal_enc)))
    logger.info('rnn encoder trainable parameters: {}'.format(count_params(model.rnn_enc)))
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
            images, image_numbers, img_lengths, cap_targets, cap_lengths, goal_tokens, goal_sentences, dependency_type, \
            ids, image_ids, goal_ids, batch_ids, dataset_idx, mutliple_choice_candidates = train_data
            model.train_emb(images, cap_targets, cap_lengths, goal_tokens, goal_sentences, batch_ids, image_lengths=img_lengths)
        else:
            images, image_numbers, cap_targets, cap_lengths, caption_map, image_map, goal_targets, goal_lengths, dependency_type,\
            ids, image_ind, goal_ids, batch_ids, prediction_ids, dataset_idx, mutliple_choice_candidates = train_data
            if epoch == opt.embedding_warmup_epochs:
                warmup_alpha = float(i) / num_loader_iter
                model.train_emb(images, image_numbers, cap_targets, cap_lengths, caption_map, image_map, goal_targets, goal_lengths,
                                dependency_type, batch_ids, prediction_ids, warmup_alpha=warmup_alpha)
            else:
                model.train_emb(images, image_numbers, cap_targets, cap_lengths, caption_map, image_map, goal_targets, goal_lengths,
                                dependency_type, batch_ids, prediction_ids)

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
    model.val_start()
    with torch.no_grad():
        # compute the encoding for all the validation images and captions
        all_img_embs, all_cap_embs, all_rnn_embs, all_goal_ids, all_mutliple_choice_candidates, attention, cur_attention \
            = encode_data(
            model, val_loader, opt.log_step, logging.info, backbone=opt.precomp_enc_type == 'backbone')

    goal_ids2_index = {}
    for index in range(len(all_goal_ids)):
        goal_ids2_index[all_goal_ids[index]] = index

    start = time.time()

    candidate_matrix = np.zeros((all_img_embs.shape[0], len(all_mutliple_choice_candidates[0])))

    for index in range(len(all_goal_ids)):
        for inner_idx in range(100):
            candidate_matrix[index, inner_idx] = goal_ids2_index[all_mutliple_choice_candidates[index][inner_idx]]
    #
    # candidate_matrix = np.zeros((all_img_embs.shape[0], 50))
    # for index in range(50):
    #     candidate_matrix[index] = np.arange(50)

    (ir1, ir5, ir10, imedr, imeanr, imrr) = single_retrieval(all_rnn_embs, all_img_embs, candidate_matrix)
    (cr1, cr5, cr10, cmedr, cmeanr, cmrr) = single_retrieval(all_rnn_embs, all_cap_embs, candidate_matrix)
    rsum = ir1 + ir5 + ir10 + cr1 + cr5 + cr10

    end = time.time()
    logger.info("calculate evaluation time: {}".format(end - start))

    logging.info("imrr: %.4f," % imrr)
    logging.info("ir1: %.3f," % ir1)
    logging.info("ir5: %.3f," % ir5)
    logging.info("ir10: %.3f," % ir10)
    logging.info("imeanr: %.3f," % imeanr)
    logging.info("imedr: %.3f," % imedr)

    logging.info("cmrr: %.4f," % cmrr)
    logging.info("cr1: %.3f," % cr1)
    logging.info("cr5: %.3f," % cr5)
    logging.info("cr10: %.3f," % cr10)
    logging.info("cmeanr: %.3f," % cmeanr)
    logging.info("cmedr: %.3f," % cmedr)

    logging.info("rsum: %.3f," % rsum)

    currscore = rsum
    logger.info('Current currscore is {}'.format(currscore))


    # record metrics in tensorboard
    tb_logger.log_value('metrics/ir1', ir1, step=model.Eiters)
    tb_logger.log_value('metrics/ir5', ir5, step=model.Eiters)
    tb_logger.log_value('metrics/ir10', ir10, step=model.Eiters)
    tb_logger.log_value('metrics/imedr', imedr, step=model.Eiters)
    tb_logger.log_value('metrics/imeanr', imeanr, step=model.Eiters)
    tb_logger.log_value('metrics/imrr', imrr, step=model.Eiters)

    tb_logger.log_value('metrics/cr1', cr1, step=model.Eiters)
    tb_logger.log_value('metrics/cr5', cr5, step=model.Eiters)
    tb_logger.log_value('metrics/cr10', cr10, step=model.Eiters)
    tb_logger.log_value('metrics/cmedr', cmedr, step=model.Eiters)
    tb_logger.log_value('metrics/cmeanr', cmeanr, step=model.Eiters)
    tb_logger.log_value('metrics/cmrr', cmrr, step=model.Eiters)

    tb_logger.log_value('metrics/rsum', rsum, step=model.Eiters)

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
    # --data_path / media / xianyu / CVPR_2022 / data / wikihow - -data_name
    # wikihow - -logger_name
    # runs / debug_wikihow_wsl_grid_bert_baseline / log - -model_name
    # runs / debug_wikihow_wsl_grid_bert_baseline - -num_epochs = 25 - -lr_update = 15 - -batch_size = 6 - -learning_rate = 5e-4 - -precomp_enc_type
    # backbone - -workers
    # 8 - -backbone_source
    # wsl - -vse_mean_warmup_epochs
    # 1 - -backbone_warmup_epochs
    # 0 - -embedding_warmup_epochs
    # 1 - -optim
    # adam - -backbone_lr_factor
    # 0.01 - -log_step
    # 20 - -input_scale_factor
    # 1.0 - -backbone_path / media / xianyu / CVPR_2022 / data / weights / original_updown_backbone.pth - -embedding_size
    # 300 - -aggregate_mode
    # avg