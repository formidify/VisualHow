"""COCO dataset loader"""
import torch
import torch.utils.data as data
import os
import os.path as osp
import numpy as np
from imageio import imread
import random
import json
import cv2
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from allennlp.data import Vocabulary
import html
import torch.nn.functional as F

import logging

logger = logging.getLogger(__name__)


class RawImageDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_name, data_split, tokenzier, opt, train):
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name
        self.tokenizer = tokenzier
        self.max_goal_length = opt.max_goal_length

        loc_info = osp.join(data_path, "final_result_w_split_20028_raw.json")
        # get wikihow data information
        with open(loc_info) as f:
            data = json.load(f)
        self.data = []
        for key, value in data.items():
            if value["split"] == data_split:
                value["goal_id"] = key
                self.data.append(value)

        self.vocabulary = Vocabulary.from_files("./data/vocabulary")

        # construct data structure
        self.captions = []
        self.tasks = []
        self.methods = []
        self.task_tokens = []
        self.method_tokens = []
        self.image_paths = []
        self.hit_ids = []
        self.caption_attention_index = []
        self.bbox_info = []
        # self.goals = []
        for value in self.data:
            self.captions.append(value["step_list"])
            self.tasks.append(value["task_title"])
            self.methods.append(value["method_title"])
            self.image_paths.append(["/".join(_.split("/")[-2:]) for _ in value["image_url"]])
            self.hit_ids.append(value["post_id"] + "_" + value["method_idx"])
            self.caption_attention_index.append(value["step_to_object_selected_index_position_result"])
            self.bbox_info.append(value["step_to_object_bbox_result"])

        if "wikihow" in data_name:
            self.image_base = osp.join(data_path, 'images')

        # List of punctuations taken from pycocoevalcap - these are ignored during evaluation.
        PUNCTUATIONS = [
            "''", "'", "``", "`", "(", ")", "{", "}",
            ".", "?", "!", ",", ":", "-", "--", "...", ";"
        ]
        # fmt: on

        # print(f"Tokenizing captions from {captions_jsonpath}...")
        for task_item, method_item in tqdm(zip(self.tasks, self.methods)):

            task = task_item.lower().strip()
            method = method_item.lower().strip()
            task_tokens = word_tokenize(task)
            method_tokens = word_tokenize(method)
            task_tokens = [ct for ct in task_tokens if ct not in PUNCTUATIONS]
            method_tokens = [ct for ct in method_tokens if ct not in PUNCTUATIONS]

            self.task_tokens.append(task_tokens)
            self.method_tokens.append(method_tokens)

        assert len(self.captions) == len(self.tasks) and len(self.captions) == len(self.methods), \
            "The lengths of captions, tasks and methods are not equal!"

        # Set related parameters according to the pre-trained backbone **
        assert 'backbone' in opt.precomp_enc_type

        self.backbone_source = opt.backbone_source
        self.base_target_size = 256
        self.crop_ratio = 0.875
        self.train_scale_rate = 1
        if hasattr(opt, 'input_scale_factor') and opt.input_scale_factor != 1:
            self.base_target_size = int(self.base_target_size * opt.input_scale_factor)
            logger.info('Input images are scaled by factor {}'.format(opt.input_scale_factor))
        if 'detector' in self.backbone_source:
            self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        else:
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]

        self.length = len(self.data)

        self.image_num = 0
        self.image_path_to_idx = {}
        counter = 0
        for value in self.image_paths:
            self.image_num += len(value)
            for image_path in value:
                self.image_path_to_idx[image_path] = counter
                counter += 1

        self.cap_num = self.image_num


        if data_split == 'dev':
            self.length = 5000


    def __getitem__(self, index):
        goal_index = self.hit_ids[index]
        caption = self.captions[index]
        cap_att_indexes = self.caption_attention_index[index]
        targets, g_attentions, ng_attentions = [], [], []
        all_caption_attentions = []
        image_index = []
        image_ids = []
        grounded_word2idx_list = []
        for idx in range(len(caption)):
            # get the caption attention index
            cap_att_index = cap_att_indexes[idx]

            caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption[idx])
            cat_tokens = "".join(caption_tokens)
            cat_tokens_start_end_pt = []
            start_pt = 0
            end_pt = 0
            for _ in caption_tokens:
                end_pt += len(_)
                cat_tokens_start_end_pt.append((start_pt, end_pt))
                start_pt = end_pt

            grounded_attention_index = cap_att_index[0]
            non_grounded_attention_index = cap_att_index[1]

            grounded_attention = [0] * len(caption_tokens)
            non_grounded_attention = [0] * len(caption_tokens)

            grounded_word2idx = {}
            for ii, (start_pos, end_pos) in enumerate(grounded_attention_index):
                selected_words = caption[idx][start_pos:end_pos]
                grounded_word2idx[selected_words.lower()] = ii + 1
                selected_word_tokens = self.tokenizer.basic_tokenizer.tokenize(selected_words)
                cat_selected_word_tokens = "".join(selected_word_tokens)
                cat_start_pos = cat_tokens.find(cat_selected_word_tokens)
                cat_end_pos = cat_start_pos + len(cat_selected_word_tokens)
                flag = 0
                for i, (cap_start_pos, cap_end_pos) in enumerate(cat_tokens_start_end_pt):
                    if cap_start_pos == cat_start_pos:
                        flag = 1
                    if flag == 1:
                        grounded_attention[i] = ii + 1
                    if cap_end_pos == cat_end_pos:
                        flag = 0
            grounded_word2idx_list.append(grounded_word2idx)
            for start_pos, end_pos in non_grounded_attention_index:
                selected_words = caption[idx][start_pos:end_pos]
                selected_word_tokens = self.tokenizer.basic_tokenizer.tokenize(selected_words)
                cat_selected_word_tokens = "".join(selected_word_tokens)
                cat_start_pos = cat_tokens.find(cat_selected_word_tokens)
                cat_end_pos = cat_start_pos + len(cat_selected_word_tokens)
                flag = 0
                for i, (cap_start_pos, cap_end_pos) in enumerate(cat_tokens_start_end_pt):
                    if cap_start_pos == cat_start_pos:
                        flag = 1
                    if flag == 1:
                        non_grounded_attention[i] = flag
                    if cap_end_pos == cat_end_pos:
                        flag = 0

            # Convert caption (string) to word ids (with Size Augmentation at training time).
            target, g_att, ng_att = \
                process_caption(self.tokenizer, caption_tokens, grounded_attention, non_grounded_attention, self.train)
            targets.append(target)
            g_attentions.append(g_att)
            ng_attentions.append(ng_att)

            all_caption_attention = g_att + ng_att
            all_caption_attention = (all_caption_attention > 0).float()
            all_caption_attentions.append(all_caption_attention)


        image_paths = self.image_paths[index]
        bbox_info = self.bbox_info[index]
        images = []
        g_att_maps = []
        att_maps = []
        all_image_att_maps = []
        for idx in range(len(image_paths)):
            image_ids.append(image_paths[idx].split("/")[-1])
            image_index.append(self.image_path_to_idx[image_paths[idx]])
            image_path = os.path.join(self.image_base, image_paths[idx])
            im_in = np.array(imread(image_path))
            im_g_att_in = np.zeros((im_in.shape[0], im_in.shape[1], g_attentions[idx].shape[0]), dtype=np.float32)
            im_att_in = np.zeros((im_in.shape[0], im_in.shape[1], 1), dtype=np.float32)
            curr_bbox_info = bbox_info[idx]
            g_curr_bbox_info = curr_bbox_info[0]
            grounded_word2idx = grounded_word2idx_list[idx]
            for objs_bbox_info in g_curr_bbox_info:
                if objs_bbox_info is not None:
                    for obj_bbox_info in objs_bbox_info:
                        left = obj_bbox_info["left"]
                        top = obj_bbox_info["top"]
                        width = obj_bbox_info["width"]
                        height = obj_bbox_info["height"]
                        object_label = obj_bbox_info["label"]
                        # if grounded_word2idx.get(html.unescape(object_label)) is None:
                        #     a=1
                        wordidx = grounded_word2idx[html.unescape(object_label)]
                        im_g_att_in[top:top + height, left:left + width, g_attentions[idx] == wordidx] = 1
            important_curr_bbox_info = curr_bbox_info[2]
            for objs_bbox_info in important_curr_bbox_info:
                if objs_bbox_info is not None:
                    for obj_bbox_info in objs_bbox_info:
                        left = obj_bbox_info["left"]
                        top = obj_bbox_info["top"]
                        width = obj_bbox_info["width"]
                        height = obj_bbox_info["height"]
                        im_att_in[top:top + height, left:left + width] = 1
            processed_image, processed_im_g_att_in, processed_im_att_in = \
                self._process_image(im_in, im_g_att_in, im_att_in)
            image = torch.Tensor(processed_image)
            image = image.permute(2, 0, 1)
            g_att_map = torch.Tensor(processed_im_g_att_in)
            g_att_map = g_att_map.permute(2, 0, 1)
            att_map = torch.Tensor(processed_im_att_in[:, :, np.newaxis])
            att_map = att_map.permute(2, 0, 1)
            images.append(image)
            g_att_map = g_att_map.sum(0, keepdim=True)
            att_map = att_map.sum(0, keepdim=True)
            g_att_map = (g_att_map > 0).float()
            att_map = (att_map > 0).float()
            g_att_maps.append(g_att_map)
            att_maps.append(att_map)

            all_image_att_map = g_att_map + att_map
            all_image_att_map = (all_image_att_map > 0).float()
            all_image_att_map = F.interpolate(all_image_att_map.unsqueeze(1), size=(8, 8)).squeeze(1)
            all_image_att_maps.append(all_image_att_map)

        task_tokens = self.task_tokens[index]
        method_tokens = self.method_tokens[index]
        # goal_sentences = ' '.join(task_tokens + ["@@SEPARATE@@"] + method_tokens)
        goal_sentences = ' '.join(task_tokens + method_tokens)
        # Tokenize goal.
        task_tokens = [self.vocabulary.get_token_index(c) for c in task_tokens]
        method_tokens = [self.vocabulary.get_token_index(c) for c in method_tokens]

        # goal_tokens = task_tokens + [self.vocabulary.get_token_index("@@SEPARATE@@")] + method_tokens
        goal_tokens = task_tokens + method_tokens
        # Pad upto max_caption_length.
        goal_tokens = goal_tokens[: self.max_goal_length]
        goal_tokens.extend(
            [self.vocabulary.get_token_index("@@UNKNOWN@@")]
            * (self.max_goal_length - len(goal_tokens))
        )

        if self.opt.no_caption:
            targets = [_*0 for _ in targets]
        if self.opt.no_image:
            images = [_*0 for _ in images]
        return images, targets, all_caption_attentions, all_image_att_maps, goal_tokens, goal_sentences, image_index, image_ids, goal_index

    def __len__(self):
        return self.length

    def _process_image(self, im_in, im_g_att_in, im_att_in):
        """
            Converts an image into a network input, with pre-processing including re-scaling, padding, etc, and data
        augmentation.
        """
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        if 'detector' in self.backbone_source:
            im_in = im_in[:, :, ::-1]
        im = im_in.astype(np.float32, copy=True)

        if self.train:
            target_size = self.base_target_size * self.train_scale_rate
        else:
            target_size = self.base_target_size

        # 2. Random crop when in training mode, elsewise just skip
        if self.train:
            crop_ratio = np.random.random() * 0.4 + 0.6
            crop_size_h = int(im.shape[0] * crop_ratio)
            crop_size_w = int(im.shape[1] * crop_ratio)
            processed_im, x_start, y_start = self._crop(im, crop_size_h, crop_size_w, random=True)
            processed_im_g_att_in = im_g_att_in[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]
            processed_im_att_in = im_att_in[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]
        else:
            processed_im = im
            processed_im_g_att_in = im_g_att_in
            processed_im_att_in = im_att_in

        # 3. Resize to the target resolution
        im_shape = processed_im.shape
        im_scale_x = float(target_size) / im_shape[1]
        im_scale_y = float(target_size) / im_shape[0]
        processed_im = cv2.resize(processed_im, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)
        processed_im_g_att_in = cv2.resize(processed_im_g_att_in, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)
        processed_im_att_in = cv2.resize(processed_im_att_in, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)

        if self.train:
            if np.random.random() > 0.5:
                processed_im = self._hori_flip(processed_im)
                processed_im_g_att_in = self._hori_flip(processed_im_g_att_in)
                processed_im_att_in = self._hori_flip(processed_im_att_in)

        # Normalization
        if 'detector' in self.backbone_source:
            processed_im = self._detector_norm(processed_im)
        else:
            processed_im = self._imagenet_norm(processed_im)

        # _im_show(processed_im)
        # _im_show(processed_im_g_att_in)
        # _im_show(processed_im_att_in)

        return processed_im, processed_im_g_att_in, processed_im_att_in

    def _imagenet_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in = im_in / 255
        for i in range(im_in.shape[-1]):
            im_in[:, :, i] = (im_in[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        return im_in

    def _detector_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in -= self.pixel_means
        return im_in

    @staticmethod
    def _crop(im, crop_size_h, crop_size_w, random):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size_w == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size_w, size=1)[0]
            if h - crop_size_h == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size_h, size=1)[0]
        else:
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2

        cropped_im = im[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]

        return cropped_im, x_start, y_start

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im

def _im_show(image):
    import matplotlib.pyplot as plt
    plt.imshow(image)
    plt.show()


class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed captions and image features for COCO or Flickr
    """

    def __init__(self, data_path, data_name, data_split, tokenizer, opt, train):
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.max_goal_length = opt.max_goal_length

        loc_info = osp.join(data_path, "final_result_w_split_15395.json")
        # get wikihow data information
        with open(loc_info) as f:
            data = json.load(f)
        self.data = []
        for key, value in data.items():
            if value["split"] == data_split:
                value["goal_id"] = key
                self.data.append(value)

        self.vocabulary = Vocabulary.from_files("./data/vocabulary")

        # construct data structure
        self.captions = []
        self.tasks = []
        self.methods = []
        self.task_tokens = []
        self.method_tokens = []
        self.image_paths = []
        # self.goals = []
        for value in self.data:
            self.captions.append(value["step_list"])
            self.tasks.append(value["task_title"])
            self.methods.append(value["method_title"])
            self.image_paths.append([_.split("/")[-1].split(".")[0] + ".npz" for _ in value["image_url"]])

        if "wikihow" in data_name:
            self.image_base = osp.join(data_path, 'region_data/data_att')

        # List of punctuations taken from pycocoevalcap - these are ignored during evaluation.
        PUNCTUATIONS = [
            "''", "'", "``", "`", "(", ")", "{", "}",
            ".", "?", "!", ",", ":", "-", "--", "...", ";"
        ]
        # fmt: on

        # print(f"Tokenizing captions from {captions_jsonpath}...")
        for task_item, method_item in tqdm(zip(self.tasks, self.methods)):
            task = task_item.lower().strip()
            method = method_item.lower().strip()
            task_tokens = word_tokenize(task)
            method_tokens = word_tokenize(method)
            task_tokens = [ct for ct in task_tokens if ct not in PUNCTUATIONS]
            method_tokens = [ct for ct in method_tokens if ct not in PUNCTUATIONS]

            self.task_tokens.append(task_tokens)
            self.method_tokens.append(method_tokens)

        assert len(self.captions) == len(self.tasks) and len(self.captions) == len(self.methods), \
            "The lengths of captions, tasks and methods are not equal!"

        self.length = len(self.data)

        self.image_num = 0
        self.image_path_to_idx = {}
        counter = 0
        for value in self.image_paths:
            self.image_num += len(value)
            for image_path in value:
                self.image_path_to_idx[image_path] = counter
                counter += 1

        self.cap_num = self.image_num

        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        goal_index = index
        caption = self.captions[index]
        image_index = []
        targets = []
        for idx in range(len(caption)):
            # get the caption attention index
            caption_tokens = self.tokenizer.basic_tokenizer.tokenize(caption[idx])

            # Convert caption (string) to word ids (with Size Augmentation at training time).
            target = process_caption(self.tokenizer, caption_tokens, self.train)
            targets.append(target)


        image_paths = self.image_paths[index]
        images = []
        for idx in range(len(image_paths)):
            image_index.append(self.image_path_to_idx[image_paths[idx]])
            image_path = os.path.join(self.image_base, image_paths[idx])
            image = np.load(image_path)["feat"]
            if self.train:  # Size augmentation for region feature
                num_features = image.shape[0]
                rand_list = np.random.rand(num_features)
                image = image[np.where(rand_list > 0.20)]
            images.append(image)

        task_tokens = self.task_tokens[index]
        method_tokens = self.method_tokens[index]
        # goal_sentences = ' '.join(task_tokens + ["@@SEPARATE@@"] + method_tokens)
        goal_sentences = ' '.join(task_tokens + method_tokens)
        # Tokenize goal.
        task_tokens = [self.vocabulary.get_token_index(c) for c in task_tokens]
        method_tokens = [self.vocabulary.get_token_index(c) for c in method_tokens]

        # goal_tokens = task_tokens + [self.vocabulary.get_token_index("@@SEPARATE@@")] + method_tokens
        goal_tokens = task_tokens + method_tokens
        # Pad upto max_caption_length.
        goal_tokens = goal_tokens[: self.max_goal_length]
        goal_tokens.extend(
            [self.vocabulary.get_token_index("@@UNKNOWN@@")]
            * (self.max_goal_length - len(goal_tokens))
        )

        return images, targets, goal_tokens, goal_sentences, image_index, goal_index

    def __len__(self):
        return self.length


def process_caption(tokenizer, tokens, grounded_attention, non_grounded_attention, train=True):
    output_tokens = []
    deleted_idx = []
    target_grounded_attention = []
    target_non_grounded_attention = []

    for i, token in enumerate(tokens):
        sub_tokens = tokenizer.wordpiece_tokenizer.tokenize(token)
        prob = random.random()

        if prob < 0.20 and train:  # mask/remove the tokens only during training
            prob /= 0.20

            # 50% randomly change token to mask token
            if prob < 0.5:
                for sub_token in sub_tokens:
                    output_tokens.append("[MASK]")
                    target_grounded_attention.append(0)
                    target_non_grounded_attention.append(0)
            # 10% randomly change token to random token
            elif prob < 0.6:
                for sub_token in sub_tokens:
                    output_tokens.append(random.choice(list(tokenizer.vocab.keys())))
                    # -> rest 10% randomly keep current token
                    target_grounded_attention.append(0)
                    target_non_grounded_attention.append(0)
            else:
                for sub_token in sub_tokens:
                    output_tokens.append(sub_token)
                    deleted_idx.append(len(output_tokens) - 1)
                    target_grounded_attention.append(grounded_attention[i])
                    target_non_grounded_attention.append(non_grounded_attention[i])
        else:
            for sub_token in sub_tokens:
                # no masking token (will be ignored by loss function later)
                output_tokens.append(sub_token)
                target_grounded_attention.append(grounded_attention[i])
                target_non_grounded_attention.append(non_grounded_attention[i])

    if len(deleted_idx) != 0:
        output_tokens = [output_tokens[i] for i in range(len(output_tokens)) if i not in deleted_idx]
        target_grounded_attention = [target_grounded_attention[i] for i in range(len(target_grounded_attention)) if
                                     i not in deleted_idx]
        target_non_grounded_attention = [target_non_grounded_attention[i] for i in
                                         range(len(target_non_grounded_attention)) if
                                         i not in deleted_idx]

    output_tokens = ['[CLS]'] + output_tokens + ['[SEP]']
    target_grounded_attention = [0] + target_grounded_attention + [0]
    target_non_grounded_attention = [0] + target_non_grounded_attention + [0]
    target = tokenizer.convert_tokens_to_ids(output_tokens)
    target = torch.Tensor(target)
    target_grounded_attention = torch.Tensor(target_grounded_attention)
    target_non_grounded_attention = torch.Tensor(target_non_grounded_attention)
    return target, target_grounded_attention, target_non_grounded_attention

def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # images, captions, goals, ids, goal_ids = zip(*data)
    # images, targets, goal_tokens, goal_sentences, image_index, goal_index = zip(*data)
    images, targets, all_caption_attentions, all_image_att_maps, goal_tokens, goal_sentences, image_index, image_ids, goal_index\
        = zip(*data)
    # temporally we do not use region feature
    if len(images[0][0].shape) == 2:  # region feature
        # Sort a data list by caption length
        # Merge images (convert tuple of 3D tensor to 4D tensor)
        # images = torch.stack(images, 0)
        ids = []
        goal_ids = []
        batch_ids = []
        for value in image_index:
            ids.extend(value)
        counter = 0
        for id_value in range(len(images)):
            goal_ids.extend([goal_index[id_value]] * len(images[id_value]))
            batch_ids.extend([counter] * len(images[id_value]))
            counter += 1

        # Merge images (convert tuple of 3D tensor to 4D tensor)
        image_list = []
        for value in images:
            image_list.extend(value)
        img_lengths = [_.shape[0] for _ in image_list]
        images = np.zeros((len(img_lengths), max(img_lengths), image_list[0].shape[1]), dtype=np.float32)
        for i, img in enumerate(image_list):
            end = img_lengths[i]
            images[i, :end] = img[:end]
        img_lengths = torch.Tensor(img_lengths)
        images = torch.Tensor(images)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        caption_list = []
        for value in targets:
            caption_list.extend(value)
        cap_lengths = [len(cap) for cap in caption_list]
        cap_targets = torch.zeros(len(caption_list), max(cap_lengths)).long()
        for i, cap in enumerate(caption_list):
            end = cap_lengths[i]
            cap_targets[i, :end] = cap[:end]

        # Merget goals (convert tuple of 1D tensor to 2D tensor)
        goal_tokens = torch.tensor(goal_tokens, dtype=torch.long)

        return images, img_lengths, cap_targets, cap_lengths, goal_tokens, goal_sentences, ids, goal_ids, batch_ids

    else:  # raw input image
        ids = []
        goal_ids = []
        batch_ids = []
        image_ind = []
        for value in image_index:
            ids.extend(value)
        for value in image_ids:
            image_ind.extend(value)
        counter = 0
        for id_value in range(len(images)):
            goal_ids.extend([goal_index[id_value]] * len(images[id_value]))
            batch_ids.extend([counter] * len(images[id_value]))
            counter += 1


        # Merge images (convert tuple of 3D tensor to 4D tensor)
        image_list = []
        for value in images:
            image_list.extend(value)
        images = torch.stack(image_list, 0)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        caption_list = []
        for value in targets:
            caption_list.extend(value)
        cap_lengths = [len(cap) for cap in caption_list]
        cap_targets = torch.zeros(len(caption_list), max(cap_lengths)).long()
        for i, cap in enumerate(caption_list):
            end = cap_lengths[i]
            cap_targets[i, :end] = cap[:end]


        # Merge g_att_maps (convert tuple of 3D tensor to 4D tensor)
        caption_map_list = []
        for value in all_caption_attentions:
            caption_map_list.extend(value)
        caption_map = torch.zeros(len(caption_list), max(cap_lengths)).float()
        for i, att_map in enumerate(caption_map_list):
            end = cap_lengths[i]
            caption_map[i, :end] = att_map

        image_map_list = []
        for value in all_image_att_maps:
            image_map_list.extend(value)
        image_map = torch.stack(image_map_list)

        # Merget goals (convert tuple of 1D tensor to 2D tensor)
        goal_tokens = torch.tensor(goal_tokens, dtype=torch.long)

        return images, cap_targets, cap_lengths, caption_map, image_map, goal_tokens, goal_sentences, ids, image_ind, goal_ids, batch_ids



def get_loader(data_path, data_name, data_split, tokenizer, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False
    if opt.precomp_enc_type == 'basic':
        dset = PrecompRegionDataset(data_path, data_name, data_split, tokenizer, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
    else:
        dset = RawImageDataset(data_path, data_name, data_split, tokenizer, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn)
    return data_loader


def get_loaders(data_path, data_name, tokenizer, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, True, workers)
    val_loader = get_loader(data_path, data_name, 'val', tokenizer, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, tokenizer, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, tokenizer, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, tokenizer, opt,
                             batch_size, False, workers, train=False)
    return test_loader
