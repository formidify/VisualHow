"""Evaluation"""
from __future__ import print_function
import logging
import time
import torch
import numpy as np

from collections import OrderedDict
from transformers import BertTokenizer

from lib.datasets import image_caption, image_caption_stepwise, image_caption_choice
from lib.vse import VSEModel

import evaluation

import logging
import tensorboard_logger as tb_logger

import multiprocessing
# from utils.score import caption_scores


import multiprocessing

import os
import json

from tqdm import tqdm

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=logger.info, backbone=False):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    all_rnn_embs = None
    all_img_embs = None
    all_cap_embs = None
    all_goal_ids = []
    all_mutliple_choice_candidates = []

    image_att_ground_truth = {}
    caption_att_ground_truth = {}
    image_att_predictions = {}
    caption_att_predictions = {}

    cur_image_att_ground_truth = {}
    cur_caption_att_ground_truth = {}
    cur_image_att_predictions = {}
    cur_caption_att_predictions = {}



    for i, data_i in enumerate(data_loader):
        # make sure val logger is used
        if not backbone:
            images, image_numbers, captions, cap_lengths, caption_map, image_map, \
            goal_targets, goal_lengths, dependency_type, ids, image_ind, goal_ids, batch_ids, prediction_ids, \
            dataset_idx, mutliple_choice_candidates = data_i
        else:
            images, image_numbers, captions, cap_lengths, caption_map, image_map, \
            goal_targets, goal_lengths, dependency_type, ids, image_ind, goal_ids, batch_ids, prediction_ids, \
            dataset_idx, mutliple_choice_candidates = data_i

        model.logger = val_logger

        all_mutliple_choice_candidates.extend(mutliple_choice_candidates)


        # compute the embeddings
        if not backbone:
            img_emb, img_pool_weights, img_pool_masks, cap_emb, cap_pool_weights, goal_emb, goal_pool_weights = \
                model.forward_emb(images, captions, cap_lengths, goal_targets, goal_lengths, image_lengths=image_lengths)
            # img_emb, img_pool_weights, img_pool_masks, cap_emb, cap_pool_weights = model.forward_emb(images, cap_targets, cap_lengths, image_lengths=img_lengths)
        else:
            img_emb, img_pool_weights, cap_emb, cap_pool_weights, \
            goal_emb, goal_pool_weights, origin_image_pool_weights, image_pool_weight_masks = \
                model.forward_emb(images, captions, cap_lengths, goal_targets, goal_lengths)
            # img_emb, img_pool_weights, img_pool_masks, cap_emb, cap_pool_weights = model.forward_emb(images, cap_targets, cap_lengths)

        # cat the feat
        if torch.cuda.is_available():
            prediction_ids_tensor = torch.tensor(prediction_ids).cuda()
            batch_ids_tensor = torch.tensor(batch_ids).cuda()
            prediction_ids_tensor = torch.tensor(prediction_ids).cuda()
            image_numbers = torch.tensor(image_numbers).cuda()
        else:
            prediction_ids_tensor = torch.tensor(prediction_ids)
            batch_ids_tensor = torch.tensor(batch_ids)
            prediction_ids_tensor = torch.tensor(prediction_ids).cuda()
            image_numbers = torch.tensor(image_numbers)

        cat_feat = torch.cat([goal_emb, img_emb, cap_emb], dim=1)
        cat_feat = cat_feat[prediction_ids_tensor < 0]
        rnn_out = model.rnn_enc(cat_feat, image_numbers - 1, dependency_type)

        next_img_emb = img_emb[prediction_ids_tensor >= 0]
        next_cap_emb = cap_emb[prediction_ids_tensor >= 0]


        if torch.cuda.is_available():
            batch_ids = torch.tensor(batch_ids, dtype=torch.long).cuda()
        else:
            batch_ids = torch.tensor(batch_ids, dtype=torch.long)

        goal_ids = [goal_ids[torch.nonzero(batch_ids==i)[0][0]] for i in range(next_cap_emb.shape[0])]
        all_goal_ids.extend(goal_ids)

        if all_img_embs is None:
            if img_emb.dim() == 3:
                all_img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                all_img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            all_cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
            all_rnn_embs = np.zeros((len(data_loader.dataset), rnn_out.size(1)))
            all_cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        all_img_embs[dataset_idx] = next_img_emb.data.cpu().numpy().copy()
        all_cap_embs[dataset_idx, :] = next_cap_emb.data.cpu().numpy().copy()
        all_rnn_embs[dataset_idx, :] = rnn_out.data.cpu().numpy().copy()


        image_maps = image_map.squeeze(1).numpy()
        caption_maps = caption_map.numpy()
        img_pool_maps = img_pool_weights.cpu().numpy()
        cap_pool_maps = cap_pool_weights.cpu().numpy()
        for ii in range(img_emb.shape[0]):
            image_id = image_ind[ii]
            image_att_ground_truth[image_id] = image_maps[ii].tolist()
            caption_att_ground_truth[image_id] = caption_maps[ii].tolist()
            image_att_predictions[image_id] = img_pool_maps[ii].tolist()
            caption_att_predictions[image_id] = cap_pool_maps[ii].tolist()

            if prediction_ids_tensor[ii] > -1:
                cur_image_att_ground_truth[image_id] = image_maps[ii].tolist()
                cur_caption_att_ground_truth[image_id] = caption_maps[ii].tolist()
                cur_image_att_predictions[image_id] = img_pool_maps[ii].tolist()
                cur_caption_att_predictions[image_id] = cap_pool_maps[ii].tolist()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader.dataset) // data_loader.batch_size + 1, batch_time=batch_time,
                e_log=str(model.logger)))
        del images, captions
    return all_img_embs, all_cap_embs, all_rnn_embs, all_goal_ids, all_mutliple_choice_candidates, \
           (image_att_ground_truth, caption_att_ground_truth, image_att_predictions, caption_att_predictions), \
           (cur_image_att_ground_truth, cur_caption_att_ground_truth, cur_image_att_predictions, cur_caption_att_predictions)

def encode_stepwise_data(model, data_loader, log_step=10, logging=logger.info, backbone=False):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    all_correct_indicator = []

    img_recall_rate = []
    cap_recall_rate = []

    for i, data_i in enumerate(data_loader):
        # make sure val logger is used
        if not backbone:
            images, image_numbers, captions, cap_lengths, caption_map, image_map, \
            goal_targets, goal_lengths, dependency_type, ids, image_ind, goal_ids, batch_ids, prediction_ids, \
            dataset_idx, correct_indicator = data_i
        else:
            images, image_numbers, captions, cap_lengths, caption_map, image_map, \
            goal_targets, goal_lengths, dependency_type, ids, image_ind, goal_ids, batch_ids, prediction_ids, \
            dataset_idx, correct_indicator = data_i

        model.logger = val_logger

        all_correct_indicator.extend(correct_indicator)


        # compute the embeddings
        if not backbone:
            img_emb, img_pool_weights, img_pool_masks, cap_emb, cap_pool_weights, goal_emb, goal_pool_weights = \
                model.forward_emb(images, captions, cap_lengths, goal_targets, goal_lengths, image_lengths=image_lengths)
            # img_emb, img_pool_weights, img_pool_masks, cap_emb, cap_pool_weights = model.forward_emb(images, cap_targets, cap_lengths, image_lengths=img_lengths)
        else:
            img_emb, img_pool_weights, cap_emb, cap_pool_weights, \
            goal_emb, goal_pool_weights, origin_image_pool_weights, image_pool_weight_masks = \
                model.forward_emb(images, captions, cap_lengths, goal_targets, goal_lengths)
            # img_emb, img_pool_weights, img_pool_masks, cap_emb, cap_pool_weights = model.forward_emb(images, cap_targets, cap_lengths)

        # cat the feat
        if torch.cuda.is_available():
            prediction_ids_tensor = torch.tensor(prediction_ids).cuda()
            batch_ids_tensor = torch.tensor(batch_ids).cuda()
            prediction_ids_tensor = torch.tensor(prediction_ids).cuda()
            image_numbers = torch.tensor(image_numbers).cuda()
            correct_indicator = torch.tensor(correct_indicator).cuda()
        else:
            prediction_ids_tensor = torch.tensor(prediction_ids)
            batch_ids_tensor = torch.tensor(batch_ids)
            prediction_ids_tensor = torch.tensor(prediction_ids)
            image_numbers = torch.tensor(image_numbers)
            correct_indicator = torch.tensor(correct_indicator)
        dataset_idx = np.array(dataset_idx, dtype=np.int64)

        img_selected_indicator = batch_ids_tensor.new_zeros(correct_indicator.shape)
        cap_selected_indicator = batch_ids_tensor.new_zeros(correct_indicator.shape)

        batch_size = correct_indicator.shape[0]
        candicate_num = batch_ids_tensor.shape[0] // batch_size

        img_remaining_indicator = batch_ids_tensor.new_ones((batch_size, batch_ids_tensor.shape[0] // batch_size))
        img_remaining_indicator[:, 0] = 0
        cap_remaining_indicator = batch_ids_tensor.new_ones((batch_size, batch_ids_tensor.shape[0] // batch_size))
        cap_remaining_indicator[:, 0] = 0

        img_selected_order = batch_ids_tensor.new_zeros((batch_size, batch_ids_tensor.shape[0] // batch_size)) - 1
        img_selected_order[:, 0] = 0
        cap_selected_order = batch_ids_tensor.new_zeros((batch_size, batch_ids_tensor.shape[0] // batch_size)) - 1
        cap_selected_order[:, 0] = 0

        sequence_len = correct_indicator.sum(-1)
        decode_length = correct_indicator.sum(-1).max()
        for decode_i in range(decode_length):
            selected_goal_embs = [goal_emb[img_selected_order.view(-1) == _] for _ in range(decode_i + 1)]
            selected_img_embs = [img_emb[img_selected_order.view(-1) == _]for _ in range(decode_i + 1)]
            selected_cap_embs = [cap_emb[cap_selected_order.view(-1) == _] for _ in range(decode_i + 1)]
            selected_goal_embs = torch.cat(selected_goal_embs, 0)
            selected_img_embs = torch.cat(selected_img_embs, 0)
            selected_cap_embs = torch.cat(selected_cap_embs, 0)


            cat_feat = torch.cat([selected_goal_embs, selected_img_embs, selected_cap_embs], dim=1)
            # dependency_type_rnn = dependency_type[img_remaining_indicator == 0]
            # rnn_out = model.rnn_enc(cat_feat, (img_remaining_indicator==0).sum(-1, keepdim=True), dependency_type_rnn)
            rnn_out = model.rnn_enc(cat_feat, (img_selected_order >= 0).sum(-1), None)

            img_scores = torch.matmul(rnn_out.view(batch_size, 1, -1),
                                      img_emb.view(batch_size, candicate_num, -1).transpose(1, 2)).squeeze()
            cap_scores = torch.matmul(rnn_out.view(batch_size, 1, -1),
                                      cap_emb.view(batch_size, candicate_num, -1).transpose(1, 2)).squeeze()

            img_scores[img_remaining_indicator == 0] = -1
            cap_scores[cap_remaining_indicator == 0] = -1

            selected_img_idx = img_scores.argmax(-1) - 1
            selected_cap_idx = cap_scores.argmax(-1) - 1

            # img_selected_indicator.scatter_(1, selected_img_idx.unsqueeze(-1), 1)
            # cap_selected_indicator.scatter_(1, selected_cap_idx.unsqueeze(-1), 1)

            img_remaining_indicator.scatter_(1, (selected_img_idx + 1).unsqueeze(-1), 0)
            cap_remaining_indicator.scatter_(1, (selected_cap_idx + 1).unsqueeze(-1), 0)

            img_selected_order.scatter_(1, (selected_img_idx + 1).unsqueeze(-1), decode_i + 1)
            cap_selected_order.scatter_(1, (selected_cap_idx + 1).unsqueeze(-1), decode_i + 1)

            # filter the selected_order
            for index in range(batch_size):
                img_selected_order[index, img_selected_order[index] > sequence_len[index]] = -1
                cap_selected_order[index, cap_selected_order[index] > sequence_len[index]] = -1

        # filter the selected_order
        for index in range(batch_size):
            # img_selected_order[index, img_selected_order[index] > sequence_len[index]] = -1
            img_selected_indicator[index, img_selected_order[index, 1:] > 0] = 1

            # cap_selected_order[index, cap_selected_order[index] > sequence_len[index]] = -1
            cap_selected_indicator[index, cap_selected_order[index, 1:] > 0] = 1



        img_recall = (img_selected_indicator * correct_indicator).sum(-1) / ((img_selected_indicator + correct_indicator) > 0).sum(-1)
        cap_recall = (cap_selected_indicator * correct_indicator).sum(-1) / ((cap_selected_indicator + correct_indicator) > 0).sum(-1)

        img_recall_rate.append(img_recall)
        cap_recall_rate.append(cap_recall)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # if i % log_step == 0:
        if True:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                .format(
                i, len(data_loader.dataset) // data_loader.batch_size + 1, batch_time=batch_time,
                e_log=str(model.logger)))
        del images, captions

    img_recall_rate = torch.cat(img_recall_rate).cpu().numpy()
    cap_recall_rate = torch.cat(cap_recall_rate).cpu().numpy()
    return img_recall_rate, cap_recall_rate
    # return np.array(img_recall_rate, dtype=np.float32), np.array(cap_recall_rate, dtype=np.float32)

def eval_ensemble(results_paths, fold5=False):
    all_sims = []
    all_npts = []
    for sim_path in results_paths:
        results = np.load(sim_path, allow_pickle=True).tolist()
        npts = results['npts']
        sims = results['sims']
        all_npts.append(npts)
        all_sims.append(sims)
    all_npts = np.array(all_npts)
    all_sims = np.array(all_sims)
    assert np.all(all_npts == all_npts[0])
    npts = int(all_npts[0])
    sims = all_sims.mean(axis=0)

    if not fold5:
        r, rt = i2t(npts, sims, return_ranks=True)
        ri, rti = t2i(npts, sims, return_ranks=True)
        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
        logger.info("rsum: %.1f" % rsum)
        logger.info("Average i2t Recall: %.1f" % ar)
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
        logger.info("Average t2i Recall: %.1f" % ari)
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    else:
        npts = npts // 5
        results = []
        all_sims = sims.copy()
        for i in range(5):
            sims = all_sims[i * npts: (i + 1) * npts, i * npts * 5: (i + 1) * npts * 5]
            r, rt0 = i2t(npts, sims, return_ranks=True)
            logger.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(npts, sims, return_ranks=True)
            logger.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            logger.info("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]
        logger.info("-----------------------------------")
        logger.info("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        logger.info("rsum: %.1f" % (mean_metrics[12]))
        logger.info("Average i2t Recall: %.1f" % mean_metrics[10])
        logger.info("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[:5])
        logger.info("Average t2i Recall: %.1f" % mean_metrics[11])
        logger.info("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                    mean_metrics[5:10])


def evalstepwise(model_path, data_path=None, split='dev', fold5=False, save_path=None, cxc=False, result_path=None):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.workers = 5
    opt.batch_size = 8


    # load vocabulary used by the model
    tokenizer_pool = multiprocessing.Pool()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    opt.backbone_path = '/tmp/data/weights/original_updown_backbone.pth'
    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = VSEModel(opt)

    model.make_data_parallel()
    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    logger.info('Loading dataset')
    data_loader = image_caption_stepwise.get_test_loader(split, opt.data_name, tokenizer,
                                                         opt.batch_size, opt.workers, opt)

    logger.info('Computing results...')
    with torch.no_grad():
        if opt.precomp_enc_type == 'basic':
            all_img_embs, all_cap_embs, all_rnn_embs, all_goal_ids, all_mutliple_choice_candidates, attention, cur_attention = encode_data(model, data_loader)
        else:
            img_recall_rate, cap_recall_rate = encode_stepwise_data(model, data_loader, backbone=True)

    start = time.time()

    end = time.time()
    logger.info("calculate evaluation time: {}".format(end - start))
    logging.info("img_recall: %.4f," % float(img_recall_rate.mean()))
    logging.info("cap_recall: %.4f," % float(cap_recall_rate.mean()))

    scores = {}
    scores["img_recall"] = float(img_recall_rate.mean())
    scores["cap_recall"] = float(img_recall_rate.mean())

    end = time.time()
    logger.info("calculate evaluation time: {}".format(end - start))

def evalrank(model_path, data_path=None, split='dev', fold5=False, save_path=None, cxc=False, result_path=None):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.workers = 5
    opt.batch_size = 8

    logger.info(opt)
    if not hasattr(opt, 'caption_loss'):
        opt.caption_loss = False

    if not hasattr(opt, 'use_dependency_info'):
        opt.use_dependency_info = False

    # load vocabulary used by the model
    tokenizer_pool = multiprocessing.Pool()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    opt.backbone_path = '/tmp/data/weights/original_updown_backbone.pth'
    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = VSEModel(opt)

    model.make_data_parallel()
    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    logger.info('Loading dataset')
    data_loader = image_caption.get_test_loader(split, opt.data_name, tokenizer,
                                                opt.batch_size, opt.workers, opt)

    logger.info('Computing results...')
    with torch.no_grad():
        if opt.precomp_enc_type == 'basic':
            all_img_embs, all_cap_embs, all_rnn_embs, all_goal_ids, all_mutliple_choice_candidates, attention, cur_attention = encode_data(model, data_loader)
        else:
            all_img_embs, all_cap_embs, all_rnn_embs, all_goal_ids, all_mutliple_choice_candidates, attention, cur_attention = encode_data(model, data_loader, backbone=True)

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

    (ir1, ir5, ir10, imedr, imeanr, imrr), (iranks, itop1) = single_retrieval(all_rnn_embs, all_img_embs, candidate_matrix, True)
    (cr1, cr5, cr10, cmedr, cmeanr, cmrr), (cranks, ctop1) = single_retrieval(all_rnn_embs, all_cap_embs, candidate_matrix, True)
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

    logging.info("cmrr: %.4f," % cmrr)

    logging.info("rsum: %.3f," % rsum)

    currscore = rsum
    logger.info('Current currscore is {}'.format(currscore))

    scores = {}
    scores["imrr"] = imrr
    scores["ir1"] = ir1
    scores["ir5"] = ir5
    scores["ir10"] = ir10
    scores["imeanr"] = imeanr
    scores["imedr"] = imedr

    scores["cmrr"] = cmrr
    scores["cr1"] = cr1
    scores["cr5"] = cr5
    scores["cr10"] = cr10
    scores["cmeanr"] = cmeanr
    scores["cmedr"] = cmedr

    scores["rsum"] = rsum

    all_eval_score = {}
    for i, goal_id in enumerate(all_goal_ids):
        score_dict = {}
        score_dict["iranks"] = float(iranks[i])
        score_dict["itop1"] = float(itop1[i])
        score_dict["cranks"] = float(cranks[i])
        score_dict["ctop1"] = float(ctop1[i])
        all_eval_score[goal_id] = score_dict

    results = {"scores": scores}
    with open(os.path.join(opt.model_name, "test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    results = {"all_eval_score": all_eval_score}
    with open(os.path.join(opt.model_name, "test_sample_scores.json"), 'w') as f:
        json.dump(results, f, indent=2)

    image_att_ground_truth, caption_att_ground_truth, image_att_predictions, caption_att_predictions = attention
    if not os.path.exists(os.path.join(opt.model_name, "attention")):
        os.makedirs(os.path.join(opt.model_name, "attention"))

    with open(os.path.join(opt.model_name, "attention/image_att_ground_truth.json"), 'w') as f:
        json.dump(image_att_ground_truth, f, indent=2)
    with open(os.path.join(opt.model_name, "attention/caption_att_ground_truth.json"), 'w') as f:
        json.dump(caption_att_ground_truth, f, indent=2)
    with open(os.path.join(opt.model_name, "attention/image_att_predictions.json"), 'w') as f:
        json.dump(image_att_predictions, f, indent=2)
    with open(os.path.join(opt.model_name, "attention/caption_att_predictions.json"), 'w') as f:
        json.dump(caption_att_predictions, f, indent=2)

    cur_image_att_ground_truth, cur_caption_att_ground_truth, cur_image_att_predictions, cur_caption_att_predictions = cur_attention
    if not os.path.exists(os.path.join(opt.model_name, "cur_attention")):
        os.makedirs(os.path.join(opt.model_name, "cur_attention"))

    with open(os.path.join(opt.model_name, "cur_attention/image_att_ground_truth.json"), 'w') as f:
        json.dump(cur_image_att_ground_truth, f, indent=2)
    with open(os.path.join(opt.model_name, "cur_attention/caption_att_ground_truth.json"), 'w') as f:
        json.dump(cur_caption_att_ground_truth, f, indent=2)
    with open(os.path.join(opt.model_name, "cur_attention/image_att_predictions.json"), 'w') as f:
        json.dump(cur_image_att_predictions, f, indent=2)
    with open(os.path.join(opt.model_name, "cur_attention/caption_att_predictions.json"), 'w') as f:
        json.dump(cur_caption_att_predictions, f, indent=2)

    end = time.time()
    logger.info("calculate evaluation time: {}".format(end - start))


def evalchoice(model_path, data_path=None, split='dev', fold5=False, save_path=None, cxc=False, result_path=None):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.workers = 5
    opt.batch_size = 8

    logger.info(opt)
    if not hasattr(opt, 'caption_loss'):
        opt.caption_loss = False

    if not hasattr(opt, 'use_dependency_info'):
        opt.use_dependency_info = False

    # load vocabulary used by the model
    tokenizer_pool = multiprocessing.Pool()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    vocab = tokenizer.vocab
    opt.vocab_size = len(vocab)

    opt.backbone_path = '/tmp/data/weights/original_updown_backbone.pth'
    if data_path is not None:
        opt.data_path = data_path

    # construct model
    model = VSEModel(opt)

    model.make_data_parallel()
    # load model state
    model.load_state_dict(checkpoint['model'])
    model.val_start()

    logger.info('Loading dataset')
    data_loader = image_caption_choice.get_test_loader(split, opt.data_name, tokenizer,
                                                opt.batch_size, opt.workers, opt)

    logger.info('Computing results...')
    with torch.no_grad():
        if opt.precomp_enc_type == 'basic':
            all_img_embs, all_cap_embs, all_rnn_embs, all_goal_ids, all_mutliple_choice_candidates, attention, cur_attention = encode_data(model, data_loader)
        else:
            all_img_embs, all_cap_embs, all_rnn_embs, all_goal_ids, all_mutliple_choice_candidates, attention, cur_attention = encode_data(model, data_loader, backbone=True)

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

    (ir1, ir5, ir10, imedr, imeanr, imrr), (iranks, itop1) = single_retrieval(all_rnn_embs, all_img_embs, candidate_matrix, True)
    (cr1, cr5, cr10, cmedr, cmeanr, cmrr), (cranks, ctop1) = single_retrieval(all_rnn_embs, all_cap_embs, candidate_matrix, True)
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

    logging.info("cmrr: %.4f," % cmrr)

    logging.info("rsum: %.3f," % rsum)

    currscore = rsum
    logger.info('Current currscore is {}'.format(currscore))

    scores = {}
    scores["imrr"] = imrr
    scores["ir1"] = ir1
    scores["ir5"] = ir5
    scores["ir10"] = ir10
    scores["imeanr"] = imeanr
    scores["imedr"] = imedr

    scores["cmrr"] = cmrr
    scores["cr1"] = cr1
    scores["cr5"] = cr5
    scores["cr10"] = cr10
    scores["cmeanr"] = cmeanr
    scores["cmedr"] = cmedr

    scores["rsum"] = rsum

    all_eval_score = {}
    for i, goal_id in enumerate(all_goal_ids):
        score_dict = {}
        score_dict["iranks"] = float(iranks[i])
        score_dict["itop1"] = float(itop1[i])
        score_dict["cranks"] = float(cranks[i])
        score_dict["ctop1"] = float(ctop1[i])
        all_eval_score[goal_id] = score_dict

    results = {"scores": scores}
    with open(os.path.join(opt.model_name, "test_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    results = {"all_eval_score": all_eval_score}
    with open(os.path.join(opt.model_name, "test_sample_scores.json"), 'w') as f:
        json.dump(results, f, indent=2)

    image_att_ground_truth, caption_att_ground_truth, image_att_predictions, caption_att_predictions = attention
    if not os.path.exists(os.path.join(opt.model_name, "attention")):
        os.makedirs(os.path.join(opt.model_name, "attention"))

    with open(os.path.join(opt.model_name, "attention/image_att_ground_truth.json"), 'w') as f:
        json.dump(image_att_ground_truth, f, indent=2)
    with open(os.path.join(opt.model_name, "attention/caption_att_ground_truth.json"), 'w') as f:
        json.dump(caption_att_ground_truth, f, indent=2)
    with open(os.path.join(opt.model_name, "attention/image_att_predictions.json"), 'w') as f:
        json.dump(image_att_predictions, f, indent=2)
    with open(os.path.join(opt.model_name, "attention/caption_att_predictions.json"), 'w') as f:
        json.dump(caption_att_predictions, f, indent=2)

    cur_image_att_ground_truth, cur_caption_att_ground_truth, cur_image_att_predictions, cur_caption_att_predictions = cur_attention
    if not os.path.exists(os.path.join(opt.model_name, "cur_attention")):
        os.makedirs(os.path.join(opt.model_name, "cur_attention"))

    with open(os.path.join(opt.model_name, "cur_attention/image_att_ground_truth.json"), 'w') as f:
        json.dump(cur_image_att_ground_truth, f, indent=2)
    with open(os.path.join(opt.model_name, "cur_attention/caption_att_ground_truth.json"), 'w') as f:
        json.dump(cur_caption_att_ground_truth, f, indent=2)
    with open(os.path.join(opt.model_name, "cur_attention/image_att_predictions.json"), 'w') as f:
        json.dump(cur_image_att_predictions, f, indent=2)
    with open(os.path.join(opt.model_name, "cur_attention/caption_att_predictions.json"), 'w') as f:
        json.dump(cur_caption_att_predictions, f, indent=2)

    end = time.time()
    logger.info("calculate evaluation time: {}".format(end - start))

def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities

# def compute_triplet_sim(images, captions, goals):
#     similarities = np.zeros((images.shape[0], captions.shape[0], goals.shape[0]), dtype=np.float32)
#     images_copy = images[:, np.newaxis, :]
#     captions_copy = captions[np.newaxis, :, :]
#     img_cap_emb = images_copy * captions_copy
#     for goal_index in tqdm(range(goals.shape[0])):
#         similarities[:, :, goal_index] = (img_cap_emb * goals[goal_index][np.newaxis, np.newaxis, :]).sum(-1)
#     return similarities

def i2t(npts, sims, return_ranks=False, mode='coco'):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r25 = 100.0 * len(np.where(ranks < 25)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, r25, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r25, medr, meanr)


def t2i(npts, sims, return_ranks=False, mode='coco'):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]

    if mode == 'coco':
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        if mode == 'coco':
            for i in range(5):
                inds = np.argsort(sims[5 * index + i])[::-1]
                ranks[5 * index + i] = np.where(inds == index)[0][0]
                top1[5 * index + i] = inds[0]
        else:
            inds = np.argsort(sims[index])[::-1]
            ranks[index] = np.where(inds == index)[0][0]
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r25 = 100.0 * len(np.where(ranks < 25)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, r25, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r25, medr, meanr)

def g2i(npts, sims, return_ranks=False, mode='coco'):
    """
    Goals->Image (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    all_ranks = np.zeros((npts, npts))
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]
        all_ranks[index] = inds

    # Compute metrics
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r25 = 100.0 * len(np.where(ranks < 25)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r10, r25, r50, r100, medr, meanr), (ranks, top1, all_ranks)
    else:
        return (r10, r25, r50, r100, medr, meanr)


def g2t(npts, sims, return_ranks=False, mode='coco'):
    """
    Goals->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    all_ranks = np.zeros((npts, npts))
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        if mode == 'coco':
            rank = 1e20
            for i in range(5 * index, 5 * index + 5, 1):
                tmp = np.where(inds == i)[0][0]
                if tmp < rank:
                    rank = tmp
            ranks[index] = rank
            top1[index] = inds[0]
        else:
            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]
        all_ranks[index] = inds

    # Compute metrics
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r25 = 100.0 * len(np.where(ranks < 25)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r10, r25, r50, r100, medr, meanr), (ranks, top1, all_ranks)
    else:
        return (r10, r25, r50, r100, medr, meanr)


def i2t_cond_goal(npts, ic_sims, ig_sims, cg_sims, goal_idxs, return_ranks=False, mode='coco'):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    goal_npts = goal_idxs.max()

    for goal_pt in range(goal_npts):
        satisfied_ids = np.argwhere(goal_idxs == goal_pt)
        # sims = ic_sims + np.expand_dims(ig_sims[:, goal_pt], axis=1) + np.expand_dims(cg_sims[:, goal_pt], axis=0)
        # sims = ic_sims
        sims = ic_sims + np.expand_dims(cg_sims[:, goal_pt], axis=0)
        for idx in range(satisfied_ids.shape[0]):
            index = satisfied_ids[idx, 0]
            inds = np.argsort(sims[index])[::-1]
            if mode == 'coco':
                rank = 1e20
                for i in range(5 * index, 5 * index + 5, 1):
                    tmp = np.where(inds == i)[0][0]
                    if tmp < rank:
                        rank = tmp
                ranks[index] = rank
                top1[index] = inds[0]
            else:
                rank = np.where(inds == index)[0][0]
                ranks[index] = rank
                top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r25 = 100.0 * len(np.where(ranks < 25)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    if return_ranks:
        return (r1, r5, r10, r25, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r25, medr, meanr)

def t2i_cond_goal(npts, ic_sims, ig_sims, cg_sims, goal_idxs, return_ranks=False, mode='coco'):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    # npts = images.shape[0]

    if mode == 'coco':
        ranks = np.zeros(5 * npts)
        top1 = np.zeros(5 * npts)
    else:
        ranks = np.zeros(npts)
        top1 = np.zeros(npts)
    goal_npts = goal_idxs.max()

    for goal_pt in range(goal_npts):
        satisfied_ids = np.argwhere(goal_idxs == goal_pt)
        # sims = ic_sims + np.expand_dims(ig_sims[:, goal_pt], axis=1) + np.expand_dims(cg_sims[:, goal_pt], axis=0)
        # sims = ic_sims
        sims = ic_sims + np.expand_dims(ig_sims[:, goal_pt], axis=1)
        sims = sims.T
        for idx in range(satisfied_ids.shape[0]):
            index = satisfied_ids[idx, 0]
            inds = np.argsort(sims[index])[::-1]

            rank = np.where(inds == index)[0][0]
            ranks[index] = rank
            top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r25 = 100.0 * len(np.where(ranks < 25)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, r25, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, r25, medr, meanr)

"""
    CxC related evaluation.
"""

def eval_cxc(images, captions, data_path):
    import os
    import json
    cxc_annot_base = os.path.join(data_path, 'cxc_annots')
    img_id_path = os.path.join(cxc_annot_base, 'testall_ids.txt')
    cap_id_path = os.path.join(cxc_annot_base, 'testall_capids.txt')

    images = images[::5, :]

    with open(img_id_path) as f:
        img_ids = f.readlines()
    with open(cap_id_path) as f:
        cap_ids = f.readlines()

    img_ids = [img_id.strip() for i, img_id in enumerate(img_ids) if i % 5 == 0]
    cap_ids = [cap_id.strip() for cap_id in cap_ids]

    with open(os.path.join(cxc_annot_base, 'cxc_it.json')) as f_it:
        cxc_it = json.load(f_it)
    with open(os.path.join(cxc_annot_base, 'cxc_i2i.json')) as f_i2i:
        cxc_i2i = json.load(f_i2i)
    with open(os.path.join(cxc_annot_base, 'cxc_t2t.json')) as f_t2t:
        cxc_t2t = json.load(f_t2t)

    sims = compute_sim(images, captions)
    t2i_recalls = cxc_inter(sims.T, img_ids, cap_ids, cxc_it['t2i'])
    i2t_recalls = cxc_inter(sims, cap_ids, img_ids, cxc_it['i2t'])
    logger.info('T2I R@1: {}, R@5: {}, R@10: {}'.format(*t2i_recalls))
    logger.info('I2T R@1: {}, R@5: {}, R@10: {}'.format(*i2t_recalls))

    i2i_recalls = cxc_intra(images, img_ids, cxc_i2i)
    t2t_recalls = cxc_intra(captions, cap_ids, cxc_t2t, text=True)
    logger.info('I2I R@1: {}, R@5: {}, R@10: {}'.format(*i2i_recalls))
    logger.info('T2T R@1: {}, R@5: {}, R@10: {}'.format(*t2t_recalls))


def cxc_inter(sims, data_ids, query_ids, annot):
    ranks = list()
    for idx, query_id in enumerate(query_ids):
        if query_id not in annot:
            raise ValueError('unexpected query id {}'.format(query_id))
        pos_data_ids = annot[query_id]
        pos_data_ids = [pos_data_id for pos_data_id in pos_data_ids if str(pos_data_id[0]) in data_ids]
        pos_data_indices = [data_ids.index(str(pos_data_id[0])) for pos_data_id in pos_data_ids]
        rank = 1e20
        inds = np.argsort(sims[idx])[::-1]
        for pos_data_idx in pos_data_indices:
            tmp = np.where(inds == pos_data_idx)[0][0]
            if tmp < rank:
                rank = tmp
        ranks.append(rank)
    ranks = np.array(ranks)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return (r1, r5, r10)


def cxc_intra(embs, data_ids, annot, text=False):
    pos_thresh = 3.0 if text else 2.5 # threshold for positive pairs according to the CxC paper

    sims = compute_sim(embs, embs)
    np.fill_diagonal(sims, 0)

    ranks = list()
    for idx, data_id in enumerate(data_ids):
        sim_items = annot[data_id]
        pos_items = [item for item in sim_items if item[1] >= pos_thresh]
        rank = 1e20
        inds = np.argsort(sims[idx])[::-1]
        if text:
            coco_pos = list(range(idx // 5 * 5, (idx // 5 + 1) * 5))
            coco_pos.remove(idx)
            pos_indices = coco_pos
            pos_indices.extend([data_ids.index(str(pos_item[0])) for pos_item in pos_items])
        else:
            pos_indices = [data_ids.index(str(pos_item[0])) for pos_item in pos_items]
            if len(pos_indices) == 0:  # skip it since there is positive example in the annotation
                continue
        for pos_idx in pos_indices:
            tmp = np.where(inds == pos_idx)[0][0]
            if tmp < rank:
                rank = tmp
        ranks.append(rank)

    ranks = np.array(ranks)
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    return (r1, r5, r10)


def single_retrieval(rnn_feat, single_modality, candidate_matrix, return_ranks=False):

    npts = rnn_feat.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    sims = np.matmul(rnn_feat, single_modality.T)


    for index in range(npts):
        inds = np.argsort(sims[index][candidate_matrix[index].astype(np.int64).tolist()])[::-1]
        ranks[index] = np.where(inds == 0)[0][0]
        top1[index] = inds[0]

    mrr = (1 / (ranks + 1)).mean()

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr, mrr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr, mrr)