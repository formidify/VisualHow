"""VSE model"""
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_
from allennlp.data import Vocabulary

from lib.encoders import get_image_encoder, get_text_encoder, get_rnn_encoder, get_captioner, get_multimodal_encoder
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
        self.rnn_enc = get_rnn_encoder(opt)
        self.captioner = get_captioner(opt)
        self.multimodal_enc = get_multimodal_encoder(opt)
        self.vocabulary = Vocabulary.from_files("./data/vocabulary")
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.cap_enc.cuda()
            self.rnn_enc.cuda()
            self.captioner.cuda()
            self.multimodal_enc.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)

        params = list(self.cap_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.rnn_enc.parameters())
        params += list(self.captioner.parameters())
        params += list(self.multimodal_enc.parameters())

        self.params = params
        self.opt = opt

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                all_text_params = list(self.cap_enc.parameters())
                bert_params = list(self.cap_enc.bert.parameters())
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
                all_text_params = list(self.cap_enc.parameters())
                bert_params = list(self.cap_enc.bert.parameters())
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
                    {'params': self.rnn_enc.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.captioner.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.multimodal_enc.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    # {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD([
                    {'params': self.cap_enc.parameters(), 'lr': opt.learning_rate},
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
                      self.rnn_enc.state_dict(), self.captioner.state_dict(),
                      self.multimodal_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.cap_enc.load_state_dict(state_dict[1], strict=False)
        self.rnn_enc.load_state_dict(state_dict[2], strict=False)
        self.captioner.load_state_dict(state_dict[3], strict=False)
        self.multimodal_enc.load_state_dict(state_dict[4], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.cap_enc.train()
        self.rnn_enc.train()
        self.captioner.train()
        self.multimodal_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.cap_enc.eval()
        self.rnn_enc.eval()
        self.captioner.eval()
        self.multimodal_enc.eval()

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
        self.rnn_enc = nn.DataParallel(self.rnn_enc)
        self.captioner = nn.DataParallel(self.captioner)
        self.multimodal_enc = nn.DataParallel(self.multimodal_enc)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, images, captions, cap_lengths, image_lengths=None):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if self.opt.precomp_enc_type == 'basic':
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                image_lengths = image_lengths.cuda()
            base_features, feat_lengths, feature_index = self.img_enc(images, image_lengths)
        else:
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()

            base_features, feat_lengths, feature_index = self.img_enc(images)


        cap_lengths = torch.Tensor(cap_lengths).cuda()
        cap_emb = self.cap_enc(captions, cap_lengths)

        image_pooled_features, image_pool_weights, cap_pooled_features, cap_pool_weights, \
        origin_image_pool_weights, image_pool_weight_masks, cap_emb_mask = \
            self.multimodal_enc(base_features, feat_lengths, feature_index,
                            cap_emb, cap_lengths)

        return image_pooled_features, image_pool_weights, cap_pooled_features, cap_pool_weights, \
               origin_image_pool_weights, image_pool_weight_masks, cap_emb_mask

    def forward_loss(self, output_dict):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = output_dict["loss"].sum()
        self.logger.update('Le', loss.data.item(), output_dict["loss"].size(0))
        return loss

    def construct_feat(self, cat_feat, batch_ids):
        if torch.cuda.is_available():
            batch_ids = torch.tensor(batch_ids, dtype=torch.long).cuda()
        else:
            batch_ids = torch.tensor(batch_ids, dtype=torch.long)
        batch_size = int(batch_ids.max()) + 1
        feat_list = []
        feat_masks = cat_feat.new_zeros(cat_feat.shape[0], 10)
        for index in range(batch_size):
            feat = cat_feat[batch_ids == index]
            feat_len = feat.shape[0]
            zero_padding = feat.new_zeros(10 - feat_len, feat.shape[1])
            feat_list.append(torch.cat([feat, zero_padding], dim=0))
            feat_masks[index, :feat_len] = 1
        feats = torch.stack(feat_list, dim=0)

        return feats, feat_masks

    def train_emb(self, images, captions, cap_lengths, caption_map, image_map,
                  goal_tokens, goal_sentences, batch_ids, image_lengths=None, warmup_alpha=None):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, img_pool_weights, cap_emb, cap_pool_weights, \
        origin_image_pool_weights, image_pool_weight_masks, cap_emb_mask\
            = self.forward_emb(images, captions, cap_lengths, image_lengths=image_lengths)

        # cat the feat
        cat_feat = torch.cat([img_emb, cap_emb], dim=1)

        feats, feat_masks = self.construct_feat(cat_feat, batch_ids)
        rnn_feats = self.rnn_enc(feats)

        if torch.cuda.is_available():
            goal_tokens = goal_tokens.cuda()
        output_dict = self.captioner(rnn_feats, goal_tokens)


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
                image_map = image_map.view(image_map.shape[0], -1)
                image_map[image_pool_weight_masks > 0] += 1e-10
                image_map = image_map / image_map.sum(-1, keepdim=True)
                caption_map[cap_emb_mask > 0] += 1e-10
                caption_map = caption_map / caption_map.sum(-1, keepdim=True)
                if not self.opt.no_image:
                    v_att_loss = attention_loss(origin_image_pool_weights, image_map, image_pool_weight_masks)
                else:
                    v_att_loss = image_map.sum() * 0
                if not self.opt.no_caption:
                    l_att_loss = attention_loss(cap_pool_weights, caption_map, cap_emb_mask)
                else:
                    l_att_loss = caption_map.sum() * 0
            elif self.opt.attention_loss == "bce":
                if not self.opt.no_image:
                    v_att_loss = attention_loss_bce(img_pool_weights, image_map)
                else:
                    v_att_loss = image_map.sum() * 0
                if not self.opt.no_caption:
                    l_att_loss = attention_loss_bce(cap_pool_weights, caption_map)
                else:
                    l_att_loss = caption_map.sum() * 0
            att_loss = v_att_loss + l_att_loss
            self.logger.update('V_Att_loss', v_att_loss.data.item(), output_dict["loss"].size(0))
            self.logger.update('L_Att_loss', l_att_loss.data.item(), output_dict["loss"].size(0))
            self.logger.update('Att_loss', att_loss.data.item(), output_dict["loss"].size(0))

            loss = self.forward_loss(output_dict) + self.opt.attention_loss_weight * att_loss
        else:
            loss = self.forward_loss(output_dict)

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        # compute gradient and update
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()

