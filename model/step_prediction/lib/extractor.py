"""Evaluation"""
from __future__ import print_function
import logging
import time
import torch
import numpy as np
import os

from collections import OrderedDict
from transformers import BertTokenizer

from lib.datasets import image_caption
from lib.vse import VSEModel

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


def encode_data(model, data_loader, save_feature_path, log_step=10, logging=logger.info, backbone=False):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    goal_embs = None

    # image names
    image_names = data_loader.dataset.images
    goal_names = data_loader.dataset.goals

    if not os.path.exists(os.path.join(save_feature_path, "embedding")):
        os.makedirs(os.path.join(save_feature_path, "embedding"))

    if not os.path.exists(os.path.join(save_feature_path, "embedding", "img_embs")):
        os.makedirs(os.path.join(save_feature_path, "embedding", "img_embs"))

    if not os.path.exists(os.path.join(save_feature_path, "embedding", "cap_embs")):
        os.makedirs(os.path.join(save_feature_path, "embedding", "cap_embs"))

    if not os.path.exists(os.path.join(save_feature_path, "embedding", "goal_embs")):
        os.makedirs(os.path.join(save_feature_path, "embedding", "goal_embs"))

    for i, data_i in enumerate(data_loader):
        # make sure val logger is used
        if not backbone:
            images, image_lengths, captions, lengths, ids = data_i
        else:
            images, cap_targets, cap_lengths, goal_targets, goal_lengths, ids, goal_ids = data_i
        model.logger = val_logger

        # compute the embeddings
        if not backbone:
            img_emb, cap_emb = model.forward_emb(images, captions, lengths, image_lengths=image_lengths)
        else:
            img_emb, cap_emb, goal_emb = model.forward_emb(images, cap_targets, cap_lengths, goal_targets, goal_lengths)

        img_emb = img_emb.data.cpu().numpy().copy()
        cap_emb = cap_emb.data.cpu().numpy().copy()
        goal_emb = goal_emb.data.cpu().numpy().copy()

        for index in range(len(ids)):
            image_name = image_names[ids[index]]
            goal_name = goal_names[ids[index]]
            task_folder = image_name.split("_")[0]
            img_idx_emb = np.reshape(img_emb[index], (1, 1024))
            cap_idx_emb = np.reshape(cap_emb[index], (1, 1024))
            goal_idx_emb = np.reshape(goal_emb[index], (1, 1024))

            if not os.path.exists(os.path.join(save_feature_path, "embedding", "img_embs", task_folder)):
                os.makedirs(os.path.join(save_feature_path, "embedding", "img_embs", task_folder))

            if not os.path.exists(os.path.join(save_feature_path, "embedding", "cap_embs", task_folder)):
                os.makedirs(os.path.join(save_feature_path, "embedding", "cap_embs", task_folder))

            if not os.path.exists(os.path.join(save_feature_path, "embedding", "goal_embs", task_folder)):
                os.makedirs(os.path.join(save_feature_path, "embedding", "goal_embs", task_folder))

            np.save(os.path.join(save_feature_path, "embedding", "img_embs", task_folder, image_name), img_idx_emb)
            np.save(os.path.join(save_feature_path, "embedding", "cap_embs", task_folder, image_name), cap_idx_emb)
            np.save(os.path.join(save_feature_path, "embedding", "goal_embs", task_folder, goal_name), goal_idx_emb)

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

def extract_feature(model_path, data_path=None, save_feature_path=None, split='dev', fold5=False, save_path=None, cxc=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    opt.workers = 5
    opt.batch_size = 128

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
            encode_data(model, data_loader, save_feature_path)
        else:
            encode_data(model, data_loader, save_feature_path, backbone=True)

