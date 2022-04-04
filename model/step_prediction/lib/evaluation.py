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


import logging
import tensorboard_logger as tb_logger

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

def encode_stepwise_data(model, data_loader, log_step=10, logging=logger.info, backbone=False):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    data_num = len(data_loader.dataset)
    # num * 10 * candidate
    all_image_score = np.zeros((data_num, 10, 20), dtype=np.float32)
    all_caption_score = np.zeros((data_num, 10, 20), dtype=np.float32)
    all_image_selected_sequence = np.zeros((data_num, 10), dtype=np.int64) - 1
    all_caption_selected_sequence = np.zeros((data_num, 10), dtype=np.int64) - 1
    all_dependency_type = np.zeros((data_num), dtype=np.int64)
    # np array to keep all the embeddings
    all_choice_step_indexes = []
    all_dependency_graph = []

    start_index = 0

    for i, data_i in enumerate(data_loader):
        # make sure val logger is used
        if not backbone:
            images, image_numbers, captions, cap_lengths, caption_map, image_map, \
            goal_targets, goal_lengths, dependency_type, ids, image_ind, goal_ids, batch_ids, \
            dataset_idx, dependency_index_info, choice_step_indexes = data_i
        else:
            images, image_numbers, captions, cap_lengths, caption_map, image_map, \
            goal_targets, goal_lengths, dependency_type, ids, image_ind, goal_ids, batch_ids, \
            dataset_idx, dependency_index_info, choice_step_indexes = data_i

        model.logger = val_logger

        all_choice_step_indexes.extend(choice_step_indexes)
        all_dependency_graph.extend(dependency_index_info)


        # compute the embeddings
        if not backbone:
            img_emb, img_pool_weights, img_pool_masks, cap_emb, cap_pool_weights, goal_emb, goal_pool_weights = \
                model.forward_emb(images, captions, cap_lengths, goal_targets, goal_lengths, image_lengths=image_lengths)
        else:
            base_features, feat_lengths, feature_index, cap_emb, goal_emb = \
                model.forward_emb(images, captions, cap_lengths, goal_targets, goal_lengths)

        # cat the feat
        if torch.cuda.is_available():
            batch_ids_tensor = torch.tensor(batch_ids).cuda()
            image_numbers = torch.tensor(image_numbers).cuda()
            choice_step_indexes = torch.tensor(choice_step_indexes).cuda()
            cap_lengths = torch.tensor(cap_lengths).cuda()
            goal_lengths = torch.tensor(goal_lengths).cuda()
        else:
            batch_ids_tensor = torch.tensor(batch_ids)
            image_numbers = torch.tensor(image_numbers)
            choice_step_indexes = torch.tensor(choice_step_indexes)
            cap_lengths = torch.tensor(cap_lengths)
            goal_lengths = torch.tensor(goal_lengths)
        dataset_idx = np.array(dataset_idx, dtype=np.int64)

        img_emb, img_pool_weights, cap_emb, cap_pool_weights, \
        goal_emb, goal_pool_weights, origin_image_pool_weights, image_pool_weight_masks = \
            model.multimodal_enc(base_features, feat_lengths, feature_index,
                                 cap_emb, cap_lengths,
                                 goal_emb, goal_lengths)

        img_selected_indicator = batch_ids_tensor.new_zeros(choice_step_indexes.shape)
        cap_selected_indicator = batch_ids_tensor.new_zeros(choice_step_indexes.shape)

        batch_size = choice_step_indexes.shape[0]
        candicate_num = batch_ids_tensor.shape[0] // batch_size

        img_remaining_indicator = batch_ids_tensor.new_ones((batch_size, candicate_num))
        img_remaining_indicator[:, 0] = 0
        cap_remaining_indicator = batch_ids_tensor.new_ones((batch_size, candicate_num))
        cap_remaining_indicator[:, 0] = 0

        img_selected_order = batch_ids_tensor.new_zeros((batch_size, candicate_num)) - 1
        img_selected_order[:, 0] = 0
        cap_selected_order = batch_ids_tensor.new_zeros((batch_size, candicate_num)) - 1
        cap_selected_order[:, 0] = 0

        sequence_len = (choice_step_indexes >= 0).sum(-1)
        decode_length = (choice_step_indexes >= 0).sum(-1).max()
        end_index = start_index + batch_size
        all_dependency_type[start_index: end_index] = dependency_type.cpu().numpy()
        for decode_i in range(decode_length):
            selected_goal_embs = [goal_emb[img_selected_order.view(-1) == _] for _ in range(decode_i + 1)]
            selected_img_embs = [img_emb[img_selected_order.view(-1) == _] for _ in range(decode_i + 1)]
            selected_cap_embs = [cap_emb[cap_selected_order.view(-1) == _] for _ in range(decode_i + 1)]
            selected_goal_embs = torch.cat(selected_goal_embs, 0)
            selected_img_embs = torch.cat(selected_img_embs, 0)
            selected_cap_embs = torch.cat(selected_cap_embs, 0)

            cat_feat = torch.cat([selected_goal_embs, selected_img_embs, selected_cap_embs], dim=1)
            # dependency_type_rnn = dependency_type[img_remaining_indicator == 0]
            # rnn_out = model.rnn_enc(cat_feat, (img_remaining_indicator==0).sum(-1, keepdim=True), dependency_type_rnn)
            rnn_out = model.rnn_enc(cat_feat, (img_selected_order >= 0).sum(-1), None)

            img_scores = torch.matmul(rnn_out.view(batch_size, 1, -1),
                                      img_emb.view(batch_size, candicate_num, -1).transpose(1, 2)).squeeze(1)
            cap_scores = torch.matmul(rnn_out.view(batch_size, 1, -1),
                                      cap_emb.view(batch_size, candicate_num, -1).transpose(1, 2)).squeeze(1)

            img_scores[img_remaining_indicator == 0] = -10000
            cap_scores[cap_remaining_indicator == 0] = -10000

            all_image_score[start_index: end_index, decode_i] = img_scores[:, 1:].cpu().numpy()
            all_caption_score[start_index: end_index, decode_i] = cap_scores[:, 1:].cpu().numpy()

            selected_img_idx = img_scores.argmax(-1) - 1
            selected_cap_idx = cap_scores.argmax(-1) - 1

            all_image_selected_sequence[start_index: end_index, decode_i] = selected_img_idx.cpu().numpy()
            all_caption_selected_sequence[start_index: end_index, decode_i] = selected_cap_idx.cpu().numpy()

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
            img_selected_indicator[index, img_selected_order[index, ] > 0] = 1

            # cap_selected_order[index, cap_selected_order[index] > sequence_len[index]] = -1
            cap_selected_indicator[index, cap_selected_order[index, ] > 0] = 1

        start_index = end_index

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
        del images, captions, goal_targets
    image_result = score_function(all_image_score, all_image_selected_sequence, all_dependency_type, all_dependency_graph)
    caption_result = score_function(all_caption_score, all_caption_selected_sequence, all_dependency_type, all_dependency_graph)
    return image_result, caption_result

def score_gt_function(all_score, all_dependency_type):
    ranks = []
    dependency = []
    for index in range(all_score.shape[0]):
        for jj in range(all_score.shape[1]):
            if all_score[index][jj] > -1:
                ranks.append(all_score[index][jj])
                dependency.append(all_dependency_type[index])

    ranks = np.array(ranks, dtype=np.float32)
    dependency = np.array(dependency, dtype=np.int64)

    result = {}
    for ii, dep_type in enumerate(["sequential", "parallel", "other"]):
        specific_ranks = ranks[dependency==ii]
        mrr = (1 / (specific_ranks + 1)).mean()

        # Compute metrics
        r1 = 100.0 * len(np.where(specific_ranks < 1)[0]) / len(specific_ranks)
        r3 = 100.0 * len(np.where(specific_ranks < 3)[0]) / len(specific_ranks)
        r5 = 100.0 * len(np.where(specific_ranks < 5)[0]) / len(specific_ranks)
        medr = np.floor(np.median(specific_ranks)) + 1
        meanr = specific_ranks.mean() + 1

        result[dep_type] = (mrr, r1, r3, r5, meanr, medr)

    mrr = (1 / (ranks + 1)).mean()

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    result["all"] = (mrr, r1, r3, r5, meanr, medr)

    return result

def encode_previous_gt_stepwise_data(model, data_loader, log_step=10, logging=logger.info, backbone=False):
    """Encode all images and captions loadable by `data_loader`
    this method use the ground truth as the previous result
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # for attention analyze
    image_att_ground_truth = {}
    caption_att_ground_truth = {}
    image_att_predictions = {}
    caption_att_predictions = {}

    # all the eval score
    all_eval_score = {}

    data_num = len(data_loader.dataset)
    all_image_ranks = np.zeros((data_num, 10), dtype=np.float32) - 1
    all_caption_ranks = np.zeros((data_num, 10), dtype=np.float32) - 1
    # num * 10 * candidate
    all_image_score = np.zeros((data_num, 10, 20), dtype=np.float32)
    all_caption_score = np.zeros((data_num, 10, 20), dtype=np.float32)
    all_image_selected_sequence = np.zeros((data_num, 10), dtype=np.int64) - 1
    all_caption_selected_sequence = np.zeros((data_num, 10), dtype=np.int64) - 1
    all_dependency_type = np.zeros((data_num), dtype=np.int64)
    # np array to keep all the embeddings
    all_choice_step_indexes = []
    all_dependency_graph = []

    start_index = 0

    for i, data_i in enumerate(data_loader):
        # make sure val logger is used
        if not backbone:
            images, image_numbers, captions, cap_lengths, caption_map, image_map, \
            goal_targets, goal_lengths, dependency_type, ids, image_ind, goal_ids, batch_ids, \
            dataset_idx, dependency_index_info, choice_step_indexes = data_i
        else:
            images, image_numbers, captions, cap_lengths, caption_map, image_map, \
            goal_targets, goal_lengths, dependency_type, ids, image_ind, goal_ids, batch_ids, \
            dataset_idx, dependency_index_info, choice_step_indexes = data_i

        model.logger = val_logger

        all_choice_step_indexes.extend(choice_step_indexes)
        all_dependency_graph.extend(dependency_index_info)


        # compute the embeddings
        if not backbone:
            img_emb, img_pool_weights, img_pool_masks, cap_emb, cap_pool_weights, goal_emb, goal_pool_weights = \
                model.forward_emb(images, captions, cap_lengths, goal_targets, goal_lengths, image_lengths=image_lengths)
        else:
            base_features, feat_lengths, feature_index, cap_emb, goal_emb = \
                model.forward_emb(images, captions, cap_lengths, goal_targets, goal_lengths)

        # cat the feat
        if torch.cuda.is_available():
            batch_ids_tensor = torch.tensor(batch_ids).cuda()
            image_numbers = torch.tensor(image_numbers).cuda()
            choice_step_indexes = torch.tensor(choice_step_indexes).cuda()
            cap_lengths = torch.tensor(cap_lengths).cuda()
            goal_lengths = torch.tensor(goal_lengths).cuda()
        else:
            batch_ids_tensor = torch.tensor(batch_ids)
            image_numbers = torch.tensor(image_numbers)
            choice_step_indexes = torch.tensor(choice_step_indexes)
            cap_lengths = torch.tensor(cap_lengths)
            goal_lengths = torch.tensor(goal_lengths)
        dataset_idx = np.array(dataset_idx, dtype=np.int64)

        img_emb, img_pool_weights, cap_emb, cap_pool_weights, \
        goal_emb, goal_pool_weights, origin_image_pool_weights, image_pool_weight_masks, \
        cap_emb_mask, goal_emb_mask = \
            model.multimodal_enc(base_features, feat_lengths, feature_index,
                                 cap_emb, cap_lengths,
                                 goal_emb, goal_lengths)

        image_maps = image_map.squeeze(1).numpy()
        caption_maps = caption_map.numpy()
        img_pool_maps = img_pool_weights.cpu().numpy()
        cap_pool_maps = cap_pool_weights.cpu().numpy()
        flatten_choice_step_indexes = choice_step_indexes.view(-1)
        for ii in range(img_emb.shape[0]):
            image_id = image_ind[ii]
            if flatten_choice_step_indexes[ii] >= 0:
                image_att_ground_truth[image_id] = image_maps[ii].tolist()
                caption_att_ground_truth[image_id] = caption_maps[ii].tolist()
                image_att_predictions[image_id] = img_pool_maps[ii].tolist()
                caption_att_predictions[image_id] = cap_pool_maps[ii].tolist()

        img_selected_indicator = batch_ids_tensor.new_zeros(choice_step_indexes.shape)
        cap_selected_indicator = batch_ids_tensor.new_zeros(choice_step_indexes.shape)

        batch_size = choice_step_indexes.shape[0]
        candicate_num = batch_ids_tensor.shape[0] // batch_size

        img_remaining_indicator = batch_ids_tensor.new_ones((batch_size, candicate_num))
        img_remaining_indicator[:, 0] = 0
        cap_remaining_indicator = batch_ids_tensor.new_ones((batch_size, candicate_num))
        cap_remaining_indicator[:, 0] = 0

        img_selected_order = batch_ids_tensor.new_zeros((batch_size, candicate_num)) - 1
        img_selected_order[:, 0] = 0
        cap_selected_order = batch_ids_tensor.new_zeros((batch_size, candicate_num)) - 1
        cap_selected_order[:, 0] = 0

        sequence_len = (choice_step_indexes >= 0).sum(-1)
        decode_length = (choice_step_indexes >= 0).sum(-1).max()
        end_index = start_index + batch_size
        all_dependency_type[start_index: end_index] = dependency_type.cpu().numpy()
        for decode_i in range(decode_length):
            selected_goal_embs = [goal_emb[img_selected_order.view(-1) == _] for _ in range(decode_i + 1)]
            selected_img_embs = [img_emb[img_selected_order.view(-1) == _] for _ in range(decode_i + 1)]
            selected_cap_embs = [cap_emb[cap_selected_order.view(-1) == _] for _ in range(decode_i + 1)]

            try:
                selected_goal_embs = torch.stack(selected_goal_embs, 1).view(-1, goal_emb.shape[-1])
                selected_img_embs = torch.stack(selected_img_embs, 1).view(-1, img_emb.shape[-1])
                selected_cap_embs = torch.stack(selected_cap_embs, 1).view(-1, cap_emb.shape[-1])
            except:
                for _ in selected_goal_embs:
                    print(_.shape)
                print(dataset_idx)
                print(choice_step_indexes)
                print(img_selected_order)
                print(cap_selected_order)

            cat_feat = torch.cat([selected_goal_embs, selected_img_embs, selected_cap_embs], dim=1)
            # dependency_type_rnn = dependency_type[img_remaining_indicator == 0]
            # rnn_out = model.rnn_enc(cat_feat, (img_remaining_indicator==0).sum(-1, keepdim=True), dependency_type_rnn)
            rnn_out = model.rnn_enc(cat_feat, (img_selected_order >= 0).sum(-1), None)

            img_scores = torch.matmul(rnn_out.view(batch_size, 1, -1),
                                      img_emb.view(batch_size, candicate_num, -1).transpose(1, 2)).squeeze(1)
            cap_scores = torch.matmul(rnn_out.view(batch_size, 1, -1),
                                      cap_emb.view(batch_size, candicate_num, -1).transpose(1, 2)).squeeze(1)

            img_scores[img_remaining_indicator == 0] = -10000
            cap_scores[cap_remaining_indicator == 0] = -10000

            all_image_score[start_index: end_index, decode_i] = img_scores[:, 1:].cpu().numpy()
            all_caption_score[start_index: end_index, decode_i] = cap_scores[:, 1:].cpu().numpy()

            selected_img_idx = []
            selected_cap_idx = []
            for batch_id in range(len(dependency_index_info)):
                curr_dependency_index_info = dependency_index_info[batch_id]
                previous_selected_index = all_image_selected_sequence[start_index + batch_id]
                previous_selected_index = previous_selected_index[previous_selected_index >= 0].tolist()
                previous_selected_index = set(previous_selected_index)
                current_possible_selection = []
                for ii in range(len(curr_dependency_index_info)):
                    if len(set(curr_dependency_index_info[ii]) - previous_selected_index) == 0:
                        if ii not in previous_selected_index:
                            current_possible_selection.append(ii)
                curr_image_sort_idx = torch.argsort(img_scores[batch_id, 1:], descending=True)
                for ii in range(curr_image_sort_idx.shape[0]):
                    if curr_image_sort_idx[ii] in current_possible_selection:
                        selected_img_idx.append(curr_image_sort_idx[ii])
                        all_image_ranks[start_index + batch_id, decode_i] = ii
                        break
                    if ii == curr_image_sort_idx.shape[0] - 1:
                        for jj in range(curr_image_sort_idx.shape[0]):
                            if curr_image_sort_idx[jj] not in previous_selected_index:
                                selected_img_idx.append(curr_image_sort_idx[jj])
                                all_image_ranks[start_index + batch_id, decode_i] = -1
                                break
                curr_cap_sort_idx = torch.argsort(cap_scores[batch_id, 1:], descending=True)
                for ii in range(curr_cap_sort_idx.shape[0]):
                    if curr_cap_sort_idx[ii] in current_possible_selection:
                        selected_cap_idx.append(curr_cap_sort_idx[ii])
                        all_caption_ranks[start_index + batch_id, decode_i] = ii
                        break
                    if ii == curr_cap_sort_idx.shape[0] - 1:
                        for jj in range(curr_cap_sort_idx.shape[0]):
                            if curr_cap_sort_idx[jj] not in previous_selected_index:
                                selected_cap_idx.append(curr_cap_sort_idx[jj])
                                all_caption_ranks[start_index + batch_id, decode_i] = -1
                                break

            selected_img_idx = torch.stack(selected_img_idx)
            selected_cap_idx = torch.stack(selected_cap_idx)

            # force image cap are the same
            selected_cap_idx = selected_img_idx

            all_image_selected_sequence[start_index: end_index, decode_i] = selected_img_idx.cpu().numpy()
            all_caption_selected_sequence[start_index: end_index, decode_i] = selected_cap_idx.cpu().numpy()

            img_remaining_indicator.scatter_(1, (selected_img_idx + 1).unsqueeze(-1), 0)
            cap_remaining_indicator.scatter_(1, (selected_cap_idx + 1).unsqueeze(-1), 0)

            img_selected_order.scatter_(1, (selected_img_idx + 1).unsqueeze(-1), decode_i + 1)
            cap_selected_order.scatter_(1, (selected_cap_idx + 1).unsqueeze(-1), decode_i + 1)

        # filter the selected_order
        for index in range(batch_size):
            img_selected_indicator[index, img_selected_order[index, ] > 0] = 1

            cap_selected_indicator[index, cap_selected_order[index, ] > 0] = 1

        batch_image_ranks = all_image_ranks[start_index:end_index, :]
        batch_caption_ranks = all_caption_ranks[start_index:end_index, :]
        for cur_index in range(batch_size):
            for order in range(1, decode_length+1):
                img_index = torch.where(img_selected_order[cur_index] == order)
                img_id = image_ind[cur_index*candicate_num+int(img_index[0])]
                score_dict = {}
                score_dict["iranks"] = float(batch_image_ranks[cur_index, order-1])
                score_dict["cranks"] = float(batch_caption_ranks[cur_index, order-1])
                if score_dict["iranks"] != -1 or score_dict["cranks"] != -1:
                    all_eval_score[img_id] = score_dict

        start_index = end_index

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
        del images, captions, goal_targets
    image_result = score_gt_function(all_image_ranks, all_dependency_type)
    caption_result = score_gt_function(all_caption_ranks, all_dependency_type)
    attention_maps = (image_att_ground_truth, caption_att_ground_truth, image_att_predictions, caption_att_predictions)
    return image_result, caption_result, attention_maps, all_eval_score

def score_function(all_score, all_selected_sequence, all_dependency_type, dependency_info):
    ranks = []
    dependency = []
    for index in range(len(dependency_info)):
        dependency_type = all_dependency_type[index]
        score = all_score[index]
        dep_info = dependency_info[index]
        sample_len = len(dep_info)
        selected_correct_step = set([])
        dep_info_set = [set(_) for _ in dep_info]
        for step_index in range(sample_len):
            step_score = score[step_index]
            score_inds = np.argsort(step_score)[::-1]
            possible_step = set()
            for ii in range(sample_len):
                if ii not in selected_correct_step and len(set(dep_info_set[ii]) - selected_correct_step) == 0:
                    possible_step.add(ii)
            possible_step = list(possible_step)
            cur_indicator = np.zeros_like(score_inds)
            for value in possible_step:
                cur_indicator += (score_inds == value)
            rank = np.where(cur_indicator == 1)[0][0]
            # if score_inds[0] < sample_len:
            if score_inds[0] in possible_step:
                selected_correct_step.add(score_inds[0])
            ranks.append(rank)
            dependency.append(dependency_type)

    ranks = np.array(ranks, dtype=np.float32)
    dependency = np.array(dependency, dtype=np.int64)

    result = {}
    for ii, dep_type in enumerate(["sequential", "parallel", "other"]):
        specific_ranks = ranks[dependency==ii]
        mrr = (1 / (specific_ranks + 1)).mean()

        # Compute metrics
        r1 = 100.0 * len(np.where(specific_ranks < 1)[0]) / len(specific_ranks)
        r3 = 100.0 * len(np.where(specific_ranks < 3)[0]) / len(specific_ranks)
        r5 = 100.0 * len(np.where(specific_ranks < 5)[0]) / len(specific_ranks)
        medr = np.floor(np.median(specific_ranks)) + 1
        meanr = specific_ranks.mean() + 1

        result[dep_type] = (mrr, r1, r3, r5, meanr, medr)

    mrr = (1 / (ranks + 1)).mean()

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r3 = 100.0 * len(np.where(ranks < 3)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    result["all"] = (mrr, r1, r3, r5, meanr, medr)

    return result

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
    opt.batch_size = 4


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
            pass
        else:
            image_scores, caption_scores, attention_maps, all_eval_score = encode_previous_gt_stepwise_data(model, data_loader, backbone=True)

    start = time.time()

    image_att_ground_truth, caption_att_ground_truth, image_att_predictions, caption_att_predictions = attention_maps
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

    results = {"all_eval_score": all_eval_score}
    with open(os.path.join(opt.model_name, "test_sample_scores.json"), 'w') as f:
        json.dump(results, f, indent=2)

    end = time.time()
    (imrr, ir1, ir3, ir5, imeanr, imedr) = image_scores["all"]
    (cmrr, cr1, cr3, cr5, cmeanr, cmedr) = caption_scores["all"]
    rsum = ir1 + ir3 + ir5 + cr1 + cr3 + cr5

    scores = {}
    scores["imrr"] = float(imrr)
    scores["ir1"] = float(ir1)
    scores["ir3"] = float(ir3)
    scores["ir5"] = float(ir5)
    scores["imeanr"] = float(imeanr)
    scores["imedr"] = float(imedr)

    scores["cmrr"] = float(cmrr)
    scores["cr1"] = float(cr1)
    scores["cr3"] = float(cr3)
    scores["cr5"] = float(cr5)
    scores["cmeanr"] = float(cmeanr)
    scores["cmedr"] = float(cmedr)

    scores["rsum"] = float(rsum)

    results = {"scores": scores}
    with open(os.path.join(opt.model_name, "test_evaluation_results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(image_scores)
    logger.info(caption_scores)

    end = time.time()
    logger.info("calculate evaluation time: {}".format(end - start))

    logging.info("imrr: %.4f," % imrr)
    logging.info("ir1: %.3f," % ir1)
    logging.info("ir3: %.3f," % ir3)
    logging.info("ir5: %.3f," % ir5)
    logging.info("imeanr: %.3f," % imeanr)
    logging.info("imedr: %.3f," % imedr)

    logging.info("cmrr: %.4f," % cmrr)
    logging.info("cr1: %.3f," % cr1)
    logging.info("cr3: %.3f," % cr3)
    logging.info("cr5: %.3f," % cr5)
    logging.info("cmeanr: %.3f," % cmeanr)
    logging.info("cmedr: %.3f," % cmedr)

    logging.info("rsum: %.3f," % rsum)

    end = time.time()
    logger.info("calculate evaluation time: {}".format(end - start))

def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities