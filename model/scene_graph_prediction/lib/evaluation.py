"""Evaluation"""
from __future__ import print_function
import logging
import time
import torch
import numpy as np

from collections import OrderedDict
from transformers import BertTokenizer

from lib.datasets import image_caption
from lib.vse import VSEModel

from sklearn import metrics

import os
import json

from tqdm import tqdm

logger = logging.getLogger(__name__)

def sigmoid(x):
    z = 1/(1 + np.exp(-x))
    return z

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
    pred_results = []
    ground_truth = []
    dep_types = []

    for i, data_i in enumerate(data_loader):
        # make sure val logger is used
        if not backbone:
            images, image_lengths, captions, lengths, ids = data_i
        else:
            images, cap_targets, cap_lengths, goal_targets, goal_lengths, caption_map, image_map, \
            dependency_mats, ids, goal_ids, batch_ids, dep_type = data_i

        model.logger = val_logger

        dep_types.extend(dep_type)

        # compute the embeddings
        image_lengths = None
        if not backbone:
            img_emb, cap_emb = model.forward_emb(images, captions, lengths, image_lengths=image_lengths)
        else:
            img_emb, img_pool_weights, cap_emb, cap_pool_weights, \
            goal_emb, goal_pool_weights, origin_image_pool_weights, image_pool_weight_masks = \
                model.forward_emb(images, cap_targets, cap_lengths, goal_targets, goal_lengths, image_lengths=image_lengths)

            # cat the emb
            if model.opt.use_image == True and model.opt.use_caption == True:
                overall_emb = torch.cat((img_emb, cap_emb, goal_emb), dim=1)
                overall_goal_emb = torch.cat((goal_emb, goal_emb, goal_emb), dim=1)
            elif model.opt.use_image == False and model.opt.use_caption == True:
                overall_emb = torch.cat((img_emb * 0, cap_emb, goal_emb), dim=1)
                overall_goal_emb = torch.cat((goal_emb * 0, goal_emb, goal_emb), dim=1)
            elif model.opt.use_image == True and model.opt.use_caption == False:
                overall_emb = torch.cat((img_emb, cap_emb * 0, goal_emb), dim=1)
                overall_goal_emb = torch.cat((goal_emb, goal_emb * 0, goal_emb), dim=1)
            pred_dependency_list = []
            dependency_scores_list = []

            if torch.cuda.is_available():
                dependency_mats = [_.cuda() for _ in dependency_mats]

            for id_value in range(batch_ids.max() + 1):
                cur_emb = overall_emb[batch_ids == id_value]
                cur_goal_emb = overall_goal_emb[batch_ids == id_value]
                task_step_embs = torch.cat([cur_goal_emb, cur_emb], dim=1).unsqueeze(0)

                row_emb = cur_emb.unsqueeze(1).repeat(1, cur_emb.shape[0], 1)
                col_emb = cur_emb.unsqueeze(0).repeat(cur_emb.shape[0], 1, 1)
                merge_emb = torch.cat([row_emb, col_emb], dim=2)

                merge_embs = torch.cat([task_step_embs, merge_emb], dim=0)
                merge_embs = merge_embs.view(-1, merge_embs.shape[-1])

                pred_dependency_list.append(model.dependency_module(merge_embs))
                dependency_scores_list.append(dependency_mats[id_value].view(-1))

        # measure accuracy and record loss
        model.forward_loss(torch.cat(pred_dependency_list, dim=0), torch.cat(dependency_scores_list, dim=0))
        # cache embeddings
        pred_dependency_list = [_.data.squeeze().cpu().numpy().copy() for _ in pred_dependency_list]
        dependency_scores_list = [_.data.cpu().numpy().copy() for _ in dependency_scores_list]
        pred_results.extend(pred_dependency_list)
        ground_truth.extend(dependency_scores_list)

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
        del images, cap_targets, goal_targets
    auc_list = []
    for index in range(len(ground_truth)):
        fpr, tpr, thresholds = metrics.roc_curve(ground_truth[index], sigmoid(pred_results[index]), pos_label=1)
        auc_value = metrics.auc(fpr, tpr)
        auc_list.append(auc_value)
    pred_results = [sigmoid(_) for _ in pred_results]
    # calculate IoU
    IoU_dict = {_: [] for _ in ["0.05", "0.1", "0.15", "0.2", "0.25", "0.3",
                                "0.35", "0.4", "0.45", "0.5", "0.55", "0.6",
                                "0.65", "0.7", "0.75", "0.8", "0.85", "0.9", "0.95"]}
    for index in range(len(ground_truth)):
        cur_gt = ground_truth[index]
        cur_pred = pred_results[index]
        for key, value in IoU_dict.items():
            cur_pred_hard = cur_pred >= float(key)
            iou = (cur_gt * cur_pred_hard).sum() / ((cur_gt + cur_pred_hard) > 0).sum()
            value.append(iou)

    IoU_result = {}
    for key, value in IoU_dict.items():
        IoU_result[key] = float(np.array(value).mean())

    return auc_list, IoU_result, pred_results, ground_truth, dep_types


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
    opt.batch_size = 32

    if not hasattr(opt, 'use_image'):
        opt.use_image = True
    if not hasattr(opt, 'use_caption'):
        opt.use_caption = True

    logger.info(opt)
    if not hasattr(opt, 'caption_loss'):
        opt.caption_loss = False

    # load vocabulary used by the model
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
            img_embs, cap_embs = encode_data(model, data_loader)
        else:
            auc_list, IoU_result, pred_results, ground_truth, dep_types = encode_data(model, data_loader, backbone=True)
    auc = np.array(auc_list)
    dep_types = np.array(dep_types)
    sequential_auc = auc[dep_types == "sequential"]
    parallel_auc = auc[dep_types == "parallel"]
    others_auc = auc[dep_types == "others"]
    auc_value = float(np.array(auc_list).mean())
    IoU_025_value = IoU_result["0.25"]
    IoU_05_value = IoU_result["0.5"]
    IoU_075_value = IoU_result["0.75"]
    sequential_auc_value = float(sequential_auc.mean())
    parallel_auc_value = float(parallel_auc.mean())
    others_auc_value = float(others_auc.mean())
    logging.info("AUC: %.3f" % (auc_value))
    logging.info("IoU 0.25: %.3f" % (IoU_025_value))
    logging.info("IoU 0.5: %.3f" % (IoU_05_value))
    logging.info("IoU 0.75: %.3f" % (IoU_075_value))
    logging.info("Sequential AUC: %.3f" % (sequential_auc_value))
    logging.info("Parallel AUC: %.3f" % (parallel_auc_value))
    logging.info("Others AUC: %.3f" % (others_auc_value))

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
