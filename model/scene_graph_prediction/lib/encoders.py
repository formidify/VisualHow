"""VSE modules"""

import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from allennlp.data import Vocabulary

from transformers import BertModel

from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.aggr.gpo import GPO
from lib.modules.mlp import MLP
from lib.modules.aggr.attention import ImageAttention, GoalAttention, CaptionAttention
# from lib.modules.captioner.updown_captioner import UpDownCaptioner
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import logging

logger = logging.getLogger(__name__)

def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def maxk_pool1d_var(x, dim, k, lengths):
    results = list()
    lengths = list(lengths.cpu().numpy())
    lengths = [int(x) for x in lengths]
    for idx, length in enumerate(lengths):
        k = min(k, length)
        max_k_i = maxk(x[idx, :length, :], dim - 1, k).mean(dim - 1)
        results.append(max_k_i)
    results = torch.stack(results, dim=0)
    return results


def maxk_pool1d(x, dim, k):
    max_k = maxk(x, dim, k)
    return max_k.mean(dim)


def maxk(x, dim, k):
    index = x.topk(k, dim=dim)[1]
    return x.gather(dim, index)


def get_text_encoder(embed_size, aggregate_mode='gpo', attention_loss='ce', no_txtnorm=False):
    return EncoderText(embed_size, aggregate_mode=aggregate_mode, attention_loss=attention_loss, no_txtnorm=no_txtnorm)

def get_multimodal_encoder(opt):
    return MultiModalEncoder(opt)
    # return EncoderText(embed_size, aggregate_mode=aggregate_mode, attention_loss=attention_loss, no_txtnorm=no_txtnorm)

def get_rnn_encoder(opt):
    return EncoderRNN(opt)


def get_dependency_module(embed_size):
    return DependencyModule(embed_size)

def get_image_encoder(data_name, img_dim, embed_size, opt, precomp_enc_type='basic',
                      backbone_source=None, backbone_path=None, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImageAggr(
            img_dim, embed_size, precomp_enc_type, opt.aggregate_mode, opt.attention_loss, no_imgnorm)
    elif precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, opt.aggregate_mode,
                                   opt.attention_loss, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc


class EncoderImageAggr(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', aggregate_mode='gpo', attention_loss='ce', no_imgnorm=False):
        super(EncoderImageAggr, self).__init__()
        self.embed_size = embed_size
        self.attention_loss = attention_loss
        self.no_imgnorm = no_imgnorm
        self.fc = nn.Linear(img_dim, embed_size)
        self.precomp_enc_type = precomp_enc_type
        self.aggregate_mode = aggregate_mode
        if self.precomp_enc_type == 'basic':
            self.mlp = MLP(img_dim, embed_size // 2, embed_size, 2)
        if self.aggregate_mode == 'att':
            self.att = ImageAttention(embed_size, embed_size, embed_size // 2, self.attention_loss)
        else:
            raise ("It only support attention mode")
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images, goal_cls_feature, image_lengths):
        """Extract image feature vectors."""
        features = self.fc(images)
        if self.precomp_enc_type == 'basic':
            # When using pre-extracted region features, add an extra MLP for the embedding transformation
            features = self.mlp(images) + features

        image_features_mask = features.new_ones(features.shape[0], features.shape[1])
        for index in range(image_lengths.shape[0]):
            image_features_mask[index, int(image_lengths[index]):] = 0
        pool_weights = self.att(features, goal_cls_feature, image_features_mask)
        if self.attention_loss == 'bce':
            features = (features * pool_weights.unsqueeze(-1)).sum(1) / image_lengths.unsqueeze(-1)
        elif self.attention_loss == 'ce':
            features = (features * pool_weights.unsqueeze(-1)).sum(1)
        else:
            features = (features * pool_weights.unsqueeze(-1)).sum(1)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features, pool_weights

class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', aggregate_mode='gpo',
                 attention_loss='ce', no_imgnorm=False):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.backbone_freezed = False

    def forward(self, images):
        """Extract image feature vectors."""
        base_features = self.backbone(images)
        feature_index = np.zeros((base_features.shape[0], 64))
        if self.training:
            # Size Augmentation during training, randomly drop grids
            base_length = base_features.size(1)
            features = []
            feat_lengths = []
            rand_list_1 = np.random.rand(base_features.size(0), base_features.size(1))
            rand_list_2 = np.random.rand(base_features.size(0))
            for i in range(base_features.size(0)):
                if rand_list_2[i] > 0.2:
                    feat_i = base_features[i][np.where(rand_list_1[i] > 0.20 * rand_list_2[i])]
                    selected_index = np.array(np.where(rand_list_1[i] > 0.20 * rand_list_2[i]))
                    feature_index[i, :selected_index.shape[1]] = selected_index.squeeze()
                    no_selected_index = np.array(np.where(rand_list_1[i] <= 0.20 * rand_list_2[i]))
                    feature_index[i, selected_index.shape[1]:] = no_selected_index.squeeze()
                    len_i = len(feat_i)
                    pads_i = torch.zeros(base_length - len_i, base_features.size(-1)).to(base_features.device)
                    feat_i = torch.cat([feat_i, pads_i], dim=0)
                else:
                    feat_i = base_features[i]
                    len_i = base_length
                    feature_index[i] = np.arange(len_i)
                feat_lengths.append(len_i)
                features.append(feat_i)
            base_features = torch.stack(features, dim=0)
            # base_features = base_features[:, :max(feat_lengths), :]
            feat_lengths = torch.tensor(feat_lengths).to(base_features.device)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)
            feature_index[:] = np.arange(base_features.size(1))

        return base_features, feat_lengths, feature_index

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


# Language Model with BERT
class EncoderText(nn.Module):
    def __init__(self, embed_size, aggregate_mode='gpo', attention_loss='ce', no_txtnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.attention_loss = attention_loss
        self.aggregate_mode = aggregate_mode
        self.no_txtnorm = no_txtnorm

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(768, embed_size)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)

        return cap_emb


# Model with multimodal encoder
class MultiModalEncoder(nn.Module):
    def __init__(self, opt):
        super(MultiModalEncoder, self).__init__()
        self.opt = opt
        self.img_dim = opt.img_dim
        self.embed_size = opt.embed_size
        self.precomp_enc_type = opt.precomp_enc_type
        self.aggregate_mode = opt.aggregate_mode
        self.attention_loss = opt.attention_loss
        self.no_imgnorm = opt.no_imgnorm
        self.no_txtnorm = opt.no_txtnorm

        if self.aggregate_mode == 'att':
            self.image_encoder = EncoderImageAggr(self.img_dim, self.embed_size, self.precomp_enc_type,
                                                  self.aggregate_mode,
                                                  self.attention_loss, self.no_imgnorm)
            self.caption_attention_encoder = CaptionAttention(self.embed_size, self.embed_size, self.embed_size // 2, self.attention_loss)
            self.goal_attention_encoder = GoalAttention(self.embed_size, self.img_dim, self.embed_size, self.embed_size // 2, self.attention_loss)
        else:
            raise ("It only support attention mode")

    def forward(self, image_features, image_lengths, feature_index, cap_emb, cap_len, goal_emb, goal_len):

        image_features_mask = image_features.new_ones(image_features.shape[0], image_features.shape[1])
        for index in range(image_lengths.shape[0]):
            image_features_mask[index, int(image_lengths[index]):] = 0

        cap_emb_mask = cap_emb.new_ones(cap_emb.shape[0], cap_emb.shape[1])
        for index in range(cap_len.shape[0]):
            cap_emb_mask[index, int(cap_len[index]):] = 0

        goal_emb_mask = goal_emb.new_ones(goal_emb.shape[0], goal_emb.shape[1])
        for index in range(goal_len.shape[0]):
            goal_emb_mask[index, int(goal_len[index]):] = 0

        image_avg_feature = (image_features * image_features_mask.unsqueeze(-1)).sum(1) / image_lengths.unsqueeze(-1)
        cap_cls_feature = cap_emb[:, 0]
        goal_cls_feature = goal_emb[:, 0]

        cap_pool_weights = self.caption_attention_encoder(cap_emb, goal_cls_feature, cap_emb_mask)
        goal_pool_weights = self.goal_attention_encoder(goal_emb, image_avg_feature, cap_cls_feature, goal_emb_mask)
        img_emb, image_pool_weights = self.image_encoder(image_features, goal_cls_feature, image_lengths)

        # Caption
        if self.attention_loss == 'bce':
            cap_pooled_features = (cap_emb * cap_pool_weights.unsqueeze(-1)).sum(1) / cap_len.unsqueeze(-1)
        elif self.attention_loss == 'ce':
            cap_pooled_features = (cap_emb * cap_pool_weights.unsqueeze(-1)).sum(1)
        else:
            cap_pooled_features = (cap_emb * cap_pool_weights.unsqueeze(-1)).sum(1)

        if not self.no_txtnorm:
            cap_pooled_features = l2norm(cap_pooled_features, dim=-1)

        # goal
        # pooled_features = (cap_emb * pool_weights.unsqueeze(-1)).sum(1)
        if self.attention_loss == 'bce':
            goal_pooled_features = (goal_emb * goal_pool_weights.unsqueeze(-1)).sum(1) / goal_len.unsqueeze(-1)
        elif self.attention_loss == 'ce':
            goal_pooled_features = (goal_emb * goal_pool_weights.unsqueeze(-1)).sum(1)
        else:
            goal_pooled_features = (goal_emb * goal_pool_weights.unsqueeze(-1)).sum(1)

        if not self.no_txtnorm:
            goal_pooled_features = l2norm(goal_pooled_features, dim=-1)

        feature_index = feature_index.astype(np.int64)
        origin_image_pool_weights = torch.zeros(image_pool_weights.shape[0], image_pool_weights.shape[1], dtype=torch.float32).to(img_emb.device)
        pool_weight_masks = torch.zeros_like(image_pool_weights)
        for index_1 in range(image_pool_weights.shape[0]):
            for index_2 in range(image_pool_weights.shape[1]):
                origin_image_pool_weights[index_1, feature_index[index_1, index_2]] = image_pool_weights[index_1, index_2]
                if index_2 < image_lengths[index_1]:
                    pool_weight_masks[index_1, feature_index[index_1, index_2]] = 1
        origin_image_pool_weights = origin_image_pool_weights.view(-1, 64)
        image_pool_weight_masks = pool_weight_masks.view(-1, 64)

        return img_emb, image_pool_weights, cap_pooled_features, cap_pool_weights, \
               goal_pooled_features, goal_pool_weights, origin_image_pool_weights, image_pool_weight_masks, \
               cap_emb_mask, goal_emb_mask



# Model with RNN
class EncoderRNN(nn.Module):
    def __init__(self, opt):
        super(EncoderRNN, self).__init__()
        self.opt = opt
        self.rnn_type = opt.rnn_type
        self.dropout = opt.rnn_dropout
        self.embed_dim_all = opt.embed_size * 3
        self.hidden_dim = opt.embed_size
        self.num_layers = opt.num_layers
        self.embed_dim = opt.embed_size
        self.no_rnnnorm = opt.no_rnnnorm
        self.use_dependency_info = opt.use_dependency_info

        self.embedding = nn.Embedding(3, self.hidden_dim // 2)
        self.transformer = nn.Linear(self.embed_dim_all, self.hidden_dim)

        self.hin_dropout_layer = nn.Dropout(self.dropout)
        if self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=self.hidden_dim, hidden_size=self.hidden_dim // 2,
                              dropout=self.dropout, batch_first=True, bidirectional=True)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=self.hidden_dim, hidden_size=self.hidden_dim // 2,
                               dropout=self.dropout, batch_first=True, bidirectional=True)
        else:
            raise Exception("RNN type is not supported: {}".format(self.rnn_type))

        # residual part
        self.project_layer = nn.Linear(self.hidden_dim // 2, self.embed_dim)
        self.relu = nn.ReLU()

    def init_hidden(self, input, batch_size, bi, dim):
        # the first parameter from the class
        times = 2 if bi else 1
        if self.rnn_type == 'gru':
            return input.new_zeros(self.num_layers * times, batch_size, dim)
        else:
            return input.new_zeros(self.num_layers * times, batch_size, dim), \
                   input.new_zeros(self.num_layers * times, batch_size, dim)

    def forward(self, input, length, dependency_type):
        batch_size, seq_length = input.size(0), input.size(1)

        transformed_input = self.transformer(input)
        max_length = length.max()
        transformed_input_list = []
        start_idx = 0
        for index in range(length.shape[0]):
            cur_len = length[index]
            cur_feat = transformed_input[start_idx : start_idx+cur_len]
            padding = cur_feat.new_zeros((max_length-cur_len, transformed_input.shape[1]))
            transformed_input_list.append(torch.cat([cur_feat, padding], dim=0))
            start_idx += cur_len
        transformed_input = torch.stack(transformed_input_list)
        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(transformed_input, length.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2

        selected_out_emb = []
        for index in range(length.shape[0]):
            selected_out_emb.append(out_emb[index, length[index] - 1])
        selected_out_emb = torch.stack(selected_out_emb)
        # selected_out_emb = torch.index_select(out_emb, 1, length-1)

        # residual layer
        if self.use_dependency_info:
            dep_embedding = self.embedding(dependency_type)
            out = self.project_layer(selected_out_emb + dep_embedding)
        else:
            out = self.project_layer(selected_out_emb)

        # out = self.relu(out)  # (batch_size, 10, embed_dim)

        if not self.no_rnnnorm:
            out = l2norm(out, dim=-1)

        return out

# Dependency Model
class DependencyModule(nn.Module):
    def __init__(self, embed_size):
        super(DependencyModule, self).__init__()
        self.embed_size = embed_size
        self.dependency_predict = nn.Sequential(nn.Linear(embed_size * 2, embed_size), nn.ReLU(), nn.Linear(embed_size, 1))

    def forward(self, x):
        """Handles variable size captions
        """
        dependency_prediction = self.dependency_predict(x)

        return dependency_prediction