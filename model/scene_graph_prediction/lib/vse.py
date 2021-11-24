"""VSE model"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder, get_dependency_module, get_multimodal_encoder
from lib.loss import ContrastiveLoss
from lib.att_loss import attention_loss, attention_loss_bce

import logging

logger = logging.getLogger(__name__)

class VSEModel(object):
    """
        The standard VSE model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = get_image_encoder(opt.data_name, opt.img_dim, opt.embed_size, opt,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm)
        self.cap_enc = get_text_encoder(opt.embed_size, aggregate_mode=opt.aggregate_mode,
                                        attention_loss=opt.attention_loss, no_txtnorm=opt.no_txtnorm)
        self.goal_enc = get_text_encoder(opt.embed_size, aggregate_mode=opt.aggregate_mode,
                                         attention_loss=opt.attention_loss, no_txtnorm=opt.no_txtnorm)
        self.multimodal_enc = get_multimodal_encoder(opt)
        self.dependency_module = get_dependency_module(opt.embed_size * 3)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.cap_enc.cuda()
            self.goal_enc.cuda()
            self.multimodal_enc.cuda()
            self.dependency_module.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = nn.BCELoss(reduction="sum")

        params = list(self.cap_enc.parameters())
        params += list(self.goal_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.multimodal_enc.parameters())
        params += list(self.dependency_module.parameters())

        self.params = params
        self.opt = opt

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                all_text_params = list(self.cap_enc.parameters()) + list(self.goal_enc.parameters())
                bert_params = list(self.cap_enc.bert.parameters()) + list(self.goal_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                ],
                    lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))
        else:
            if self.opt.optim == 'adam':
                all_text_params = list(self.cap_enc.parameters()) + list(self.goal_enc.parameters())
                bert_params = list(self.cap_enc.bert.parameters()) + list(self.goal_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.backbone.top.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.img_enc.backbone.base.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    # {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                    {'params': self.multimodal_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.dependency_module.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD([
                    {'params': self.cap_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.goal_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.img_enc.backbone.parameters(), 'lr': opt.learning_rate * opt.backbone_lr_factor,
                     'weight_decay': decay_factor},
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, momentum=0.9, nesterov=True)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.cap_enc.state_dict(),
                      self.goal_enc.state_dict(), self.multimodal_enc.state_dict(),
                      self.dependency_module.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.cap_enc.load_state_dict(state_dict[1], strict=False)
        self.goal_enc.load_state_dict(state_dict[2], strict=False)
        self.multimodal_enc.load_state_dict(state_dict[3], strict=False)
        self.dependency_module.load_state_dict(state_dict[4], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.cap_enc.train()
        self.goal_enc.train()
        self.multimodal_enc.train()
        self.dependency_module.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.cap_enc.eval()
        self.goal_enc.eval()
        self.multimodal_enc.eval()
        self.dependency_module.eval()

    def freeze_backbone(self):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.freeze_backbone()
            else:
                self.img_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.unfreeze_backbone(fixed_blocks)
            else:
                self.img_enc.unfreeze_backbone(fixed_blocks)

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.cap_enc = nn.DataParallel(self.cap_enc)
        self.goal_enc = nn.DataParallel(self.goal_enc)
        self.multimodal_enc = nn.DataParallel(self.multimodal_enc)
        self.dependency_module = nn.DataParallel(self.dependency_module)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, images, captions, cap_lengths, goal_targets, goal_lengths, image_lengths=None):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if self.opt.precomp_enc_type == 'basic':
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                goal_targets = goal_targets.cuda()
                image_lengths = image_lengths.cuda()
            base_features, feat_lengths, feature_index = self.img_enc(images, image_lengths)
        else:
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                goal_targets = goal_targets.cuda()

            base_features, feat_lengths, feature_index = self.img_enc(images)


        cap_lengths = torch.Tensor(cap_lengths).cuda()
        cap_emb = self.cap_enc(captions, cap_lengths)

        goal_lengths = torch.Tensor(goal_lengths).cuda()
        goal_emb = self.goal_enc(goal_targets, goal_lengths)

        image_pooled_features, image_pool_weights, cap_pooled_features, cap_pool_weights, \
        goal_pooled_features, goal_pool_weights, origin_image_pool_weights, image_pool_weight_masks = \
            self.multimodal_enc(base_features, feat_lengths, feature_index,
                            cap_emb, cap_lengths,
                            goal_emb, goal_lengths)


        return image_pooled_features, image_pool_weights, cap_pooled_features, cap_pool_weights, \
               goal_pooled_features, goal_pool_weights, origin_image_pool_weights, image_pool_weight_masks

    def forward_loss(self, pred_dependency, dependency_scores):
        """Compute the loss given pairs of image and caption embeddings
        """
        dep_loss = self.criterion(torch.sigmoid(pred_dependency), dependency_scores.unsqueeze(-1))
        loss = dep_loss
        self.logger.update('Le_DEP', dep_loss.data.item(), pred_dependency.size(0))
        self.logger.update('Le', loss.data.item(), pred_dependency.size(0))
        return loss

    def train_emb(self, images, captions, cap_lengths, goals, goal_lengths, goal_ids, batch_ids,
                  caption_map, image_map, dependency_mats, image_lengths=None, warmup_alpha=None):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, img_pool_weights, cap_emb, cap_pool_weights, \
        goal_emb, goal_pool_weights, origin_image_pool_weights, image_pool_weight_masks =\
            self.forward_emb(images, captions, cap_lengths, goals, goal_lengths, image_lengths=image_lengths)

        # cat the emb
        overall_emb = torch.cat((img_emb, cap_emb, goal_emb), dim=1)
        overall_goal_emb = torch.cat((goal_emb, goal_emb, goal_emb), dim=1)

        merge_emb_list = []
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
            merge_emb_list.append(merge_embs)
            dependency_scores_list.append(dependency_mats[id_value].view(-1))

        merge_embs = torch.cat(merge_emb_list, dim=0)
        dependency_scores = torch.cat(dependency_scores_list, dim=0)

        pred_dependency = self.dependency_module(merge_embs)

        # measure accuracy and record loss
        self.optimizer.zero_grad()

        if self.opt.aggregate_mode == "att" and self.opt.attention_loss != "None":
            # attention supervision
            if torch.cuda.is_available():
                image_map = image_map.cuda()
                caption_map = caption_map.cuda()
            if self.opt.attention_loss == "ce":
                # normalize
                image_map[image_pool_weight_masks.view(-1, 1, 8, 8) == 0] = 0
                image_map = (image_map + 1e-10) / (image_map + 1e-10).sum(-1, keepdim=True).sum(-2, keepdim=True)
                caption_map = (caption_map + 1e-10) / (caption_map + 1e-10).sum(-1, keepdim=True)
                v_att_loss = attention_loss(img_pool_weights, image_map, image_pool_weight_masks)
                l_att_loss = attention_loss(cap_pool_weights, caption_map)
            elif self.opt.attention_loss == "bce":
                image_map[image_pool_weight_masks.view(-1, 1, 8, 8) == 0] = 0
                v_att_loss = attention_loss_bce(img_pool_weights, image_map, image_pool_weight_masks)
                l_att_loss = attention_loss_bce(cap_pool_weights, caption_map)
            att_loss = v_att_loss + l_att_loss
            self.logger.update('Att_loss', att_loss.data.item(), image_map.size(0))
            loss = self.forward_loss(pred_dependency, dependency_scores) + self.opt.attention_loss_weight * att_loss
        else:
            loss = self.forward_loss(pred_dependency, dependency_scores)

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        # compute gradient and update
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

