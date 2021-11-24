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

import evaluation

import logging
import tensorboard_logger as tb_logger

import multiprocessing
from utils.score import caption_scores


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

    predictions = []
    ground_truth = []

    image_att_ground_truth = {}
    caption_att_ground_truth = {}
    image_att_predictions = {}
    caption_att_predictions = {}

    for i, data_i in enumerate(data_loader):
        # make sure val logger is used
        if not backbone:
            images, img_lengths, cap_targets, cap_lengths, goal_tokens, goal_sentences, ids, image_ids, goal_ids, batch_ids = data_i
        else:
            images, cap_targets, cap_lengths, caption_map, image_map, goal_tokens, goal_sentences, ids, image_ids, goal_ids, batch_ids = data_i

        model.logger = val_logger

        # compute the embeddings
        if not backbone:
            img_emb, img_pool_weights, img_pool_masks, cap_emb, cap_pool_weights = model.forward_emb(images, cap_targets, cap_lengths, image_lengths=img_lengths)
        else:
            img_emb, img_pool_weights, cap_emb, cap_pool_weights, \
            origin_image_pool_weights, image_pool_weight_masks = model.forward_emb(images, cap_targets, cap_lengths)

        # cat the feat
        cat_feat = torch.cat([img_emb, cap_emb], dim=1)

        feats, feat_masks = model.construct_feat(cat_feat, batch_ids)
        rnn_feats = model.rnn_enc(feats)

        if torch.cuda.is_available():
            goal_tokens = goal_tokens.cuda()
        output_dict = model.captioner(rnn_feats)

        if torch.cuda.is_available():
            batch_ids = torch.tensor(batch_ids, dtype=torch.long).cuda()
        else:
            batch_ids = torch.tensor(batch_ids, dtype=torch.long)

        for i in range(output_dict["predictions"].shape[0]):
            instance_predictions = output_dict["predictions"][i, :]

            # De-tokenize caption tokens and trim until first "@@BOUNDARY@@".
            caption = [
                model.vocabulary.get_token_from_index(p.item()) for p in instance_predictions
            ]
            eos_occurences = [
                j for j in range(len(caption)) if caption[j] == "@@BOUNDARY@@"
            ]
            caption = (
                caption[: eos_occurences[0]] if len(eos_occurences) > 0 else caption
            )
            predictions.append(
                {"goal_id": goal_ids[torch.nonzero(batch_ids==i)[0][0]], "caption": " ".join(caption)}
            )
            ground_truth.append(
                {"goal_id": goal_ids[torch.nonzero(batch_ids==i)[0][0]], "caption": goal_sentences[i]}
            )

        image_maps = image_map.squeeze(1).numpy()
        caption_maps = caption_map.numpy()
        img_pool_maps = img_pool_weights.cpu().numpy()
        cap_pool_maps = cap_pool_weights.cpu().numpy()
        for i in range(output_dict["predictions"].shape[0]):
            image_id = image_ids[torch.nonzero(batch_ids==i)[0][0]]
            image_att_ground_truth[image_id] = image_maps[i].tolist()
            caption_att_ground_truth[image_id] = caption_maps[i].tolist()
            image_att_predictions[image_id] = img_pool_maps[i].tolist()
            caption_att_predictions[image_id] = cap_pool_maps[i].tolist()

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
        del images, cap_targets
    return predictions, ground_truth, \
           (image_att_ground_truth, caption_att_ground_truth, image_att_predictions, caption_att_predictions)


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
    opt.batch_size = 8

    logger.info(opt)
    if not hasattr(opt, 'caption_loss'):
        opt.caption_loss = False

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
            predictions, ground_truth, attention = encode_data(model, data_loader)
        else:
            predictions, ground_truth, attention = encode_data(model, data_loader, backbone=True)
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

    all_eval_score = {}
    for i, prediction in enumerate(predictions):
        goal_id = prediction["goal_id"]
        score_dict = {}
        score_dict["BLEU1"] = float(all_scores["BLEU"][0][i])
        score_dict["BLEU2"] = float(all_scores["BLEU"][1][i])
        score_dict["BLEU3"] = float(all_scores["BLEU"][2][i])
        score_dict["BLEU4"] = float(all_scores["BLEU"][3][i])
        score_dict["METEOR"] = float(all_scores["METEOR"][i])
        score_dict["ROUGE"] = float(all_scores["ROUGE"][i])
        score_dict["CIDEr"] = float(all_scores["CIDEr"][i])
        score_dict["SPICE"] = float(all_scores["SPICE"][i]['All']['f'])
        all_eval_score[goal_id] = score_dict

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

    end = time.time()
    logger.info("calculate evaluation time: {}".format(end - start))


def compute_sim(images, captions):
    similarities = np.matmul(images, np.matrix.transpose(captions))
    return similarities



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
