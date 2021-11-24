# encoding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# reload(sys)
# sys.setdefaultencoding('utf8')
import json
import h5py
import os
import os.path
import numpy as np
import random
import logging
import misc.utils as utils
import cv2
from imageio import imread

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import eval_utils
import torch.nn.functional as F


class VISTDataset(Dataset):
    def __init__(self, opt):
        self.mode = 'train'  # by default
        self.opt = opt

        self.task = opt.task  # option: 'story_telling', 'image_captioning'
        self.data_type = {
            'whole_story': False,
            'split_story': True,
            'caption': False,
            'title': True
        }
        with open(os.path.join(opt.data_dir, "final_result_w_split_20028_raw.json")) as f:
            self.data_info = json.load(f)

        # open the hdf5 file
        print('DataLoader loading story h5 file: ', opt.story_h5)
        self.story_h5 = h5py.File(opt.story_h5, 'r', driver='core')['story']
        print("story's max sentence length is ", self.story_h5.shape[1])

        print('DataLoader loading story h5 file: ', opt.full_story_h5)
        self.full_story_h5 = h5py.File(opt.full_story_h5, 'r', driver='core')['story']
        print("full story's max sentence length is ", self.full_story_h5.shape[1])

        if opt.use_title:
            print('DataLoader loading title h5 file: ', opt.title_h5)
            self.title_h5 = h5py.File(opt.title_h5, 'r', driver='core')['title']
            print("title's max sentence length is ", self.title_h5.shape[1])

        # print('DataLoader loading description h5 file: ', opt.desc_h5)
        # self.desc_h5 = h5py.File(opt.desc_h5, 'r', driver='core')['story']
        # print("caption's max sentence length is ", self.desc_h5.shape[1])

        print('DataLoader loading story_line json file: ', opt.story_line_json)
        self.story_line = json.load(open(opt.story_line_json))

        self.id2word = self.story_line['id2words']
        print("vocab[0] = ", self.id2word['0'])
        print("vocab[1] = ", self.id2word['1'])
        self.word2id = self.story_line['words2id']
        self.vocab_size = len(self.id2word)
        print('vocab size is ', self.vocab_size)

        self.story_ids = {'train': [], 'val': [], 'test': []}
        self.description_ids = {'train': [], 'val': [], 'test': []}
        self.story_ids['train'] = self.story_line['train'].keys()
        self.story_ids['val'] = self.story_line['val'].keys()
        self.story_ids['test'] = self.story_line['test'].keys()
        # self.description_ids['train'] = self.story_line['image2caption']['train'].keys()
        # self.description_ids['val'] = self.story_line['image2caption']['val'].keys()
        # self.description_ids['test'] = self.story_line['image2caption']['test'].keys()

        print('There are {} training data, {} validation data, and {} test data'.format(len(self.story_ids['train']),
                                                                                        len(self.story_ids['val']),
                                                                                        len(self.story_ids['test'])))

        ref_dir = 'data/reference'
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)

        # write reference files for storytelling
        for split in ['val', 'test']:
            reference = {}
            for story in self.story_line[split].values():
                if story['task_id'] not in reference:
                    reference[story['task_id']] = [story['origin_text']]
                else:
                    reference[story['task_id']].append(story['origin_text'])
            with open(os.path.join(ref_dir, '{}_reference.json'.format(split)), 'w') as f:
                json.dump(reference, f, indent=2)

        self.base_target_size = 256
        self.crop_ratio = 0.875
        self.train_scale_rate = 1
        self.backbone_source = self.opt.backbone_source
        if hasattr(opt, 'input_scale_factor') and opt.input_scale_factor != 1:
            self.base_target_size = int(self.base_target_size * opt.input_scale_factor)
            print('Input images are scaled by factor {}'.format(opt.input_scale_factor))
        if 'detector' in self.backbone_source:
            self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        else:
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]

        # write reference files for captioning
        # for split in ['val', 'test']:
        #     reference = {}
        #     for flickr_id, story in self.story_line['image2caption_original'][split].iteritems():
        #         reference[flickr_id] = story
        #     with open(os.path.join(ref_dir, '{}_desc_reference.json'.format(split)), 'w') as f:
        #         json.dump(reference, f)

    def __getitem__(self, index):
        if self.task == 'story_telling':
            story_id = self.story_ids[self.mode][index]
            story = self.story_line[self.mode][story_id]
            if self.mode == "train":
                used_length = min(self.opt.max_story_size, story['length'])
            else:
                used_length = story['length']
            bbox_info = self.data_info[story_id]['step_to_object_bbox_result']
            # load feature
            # feature_folder = story['task_id'].split("_")[0]
            feature_fc = np.zeros((used_length, self.base_target_size, self.base_target_size, 3), dtype='float32')
            feature_conv = np.zeros((used_length, self.opt.conv_feat_size), dtype='float32')
            image_attention = np.zeros((used_length, 1, 8, 8), dtype='float32')
            for i in xrange(used_length):
                # load fc feature
                # fc_path = os.path.join(self.opt.data_dir, 'resnet_features/fc', feature_folder,
                #                        '{}.npy'.format(story['img_id'][i]))
                # feature_fc[i] = np.load(fc_path)
                # if self.opt.use_conv:
                #     conv_path = os.path.join(self.opt.data_dir, 'resnet_features/conv', feature_folder,
                #                              '{}.npz'.format(story['img_id'][i]))
                #     feature_conv[i] = np.load(conv_path)['arr_0'].flatten()
                image_path = os.path.join(self.opt.data_dir, 'images/{}/{}.jpg'.
                                          format(story['img_id'][i].split("_")[0], story['img_id'][i]))
                im_in = np.array(imread(image_path))

                im_g_att_in = np.zeros((im_in.shape[0], im_in.shape[1], 1), dtype=np.float32)
                im_att_in = np.zeros((im_in.shape[0], im_in.shape[1], 1), dtype=np.float32)
                curr_bbox_info = bbox_info[i]
                g_curr_bbox_info = curr_bbox_info[0]
                for objs_bbox_info in g_curr_bbox_info:
                    if objs_bbox_info is not None:
                        for obj_bbox_info in objs_bbox_info:
                            left = obj_bbox_info["left"]
                            top = obj_bbox_info["top"]
                            width = obj_bbox_info["width"]
                            height = obj_bbox_info["height"]
                            im_g_att_in[top:top + height, left:left + width] = 1
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
                    self._process_image_att(im_in, im_g_att_in, im_att_in)
                image_att_map = ((processed_im_g_att_in + processed_im_att_in) > 0).astype(np.float32)
                # image_att_map = cv2.resize(image_att_map, dsize=(8, 8), interpolation=cv2.INTER_NEAREST)
                # image_att_map = F.interpolate(image_att_map.unsqueeze(1), size=(8, 8)).squeeze(1)
                # processed_image = self._process_image(im_in)
                feature_fc[i] = processed_image
                image_attention[i, 0] = image_att_map[:, :]

            sample = {'feature_fc': feature_fc, "image_attention": image_attention}
            if self.opt.use_conv:
                sample['feature_conv'] = feature_conv

            sample['length'] = used_length

            # load story
            if self.data_type['whole_story']:
                whole_story = self.full_story_h5[story['whole_text_index']]
                sample['whole_story'] = np.int64(whole_story)

            if self.data_type['split_story']:
                split_story = self.story_h5[story['text_index']][:used_length]
                sample['split_story'] = np.int64(split_story)

            # load title
            if self.data_type['title']:
                title = self.title_h5[story['whole_text_index']][:self.opt.max_title_size]
                sample['title'] = np.int64(title)

            # load caption
            if self.data_type['caption']:
                caption = []
                for flickr_id in story['flickr_id']:
                    if flickr_id in self.story_line['image2caption'][self.mode]:
                        descriptions = self.story_line['image2caption'][self.mode][flickr_id]['caption']
                        random_idx = np.random.choice(len(descriptions), 1)[0]
                        caption.append(self.desc_h5[descriptions[random_idx]])
                    else:
                        caption.append(np.zeros((self.desc_h5.shape[1],), dtype='int64'))
                sample['caption'] = np.asarray(caption, 'int64')
            sample['index'] = np.int64(index)

            return sample

        elif self.task == "image_captioning":
            flickr_id = self.description_ids[self.mode][index]
            descriptions = self.story_line['image2caption'][self.mode][flickr_id]['caption']
            random_idx = np.random.choice(len(descriptions), 1)[0]
            description = descriptions[random_idx]

            fc_path = os.path.join(self.opt.data_dir, 'resnet_features/fc', self.mode,
                                   '{}{}.npy'.format(self.opt.prefix, flickr_id))
            conv_path = os.path.join(self.opt.data_dir, 'resnet_features/conv', self.mode, '{}.npy'.format(flickr_id))
            feature_fc = np.load(fc_path)
            feature_conv = np.load(conv_path).flatten()

            sample = {'feature_fc': feature_fc, 'feature_conv': feature_conv}
            target = np.int64(self.desc_h5[description])
            sample['whole_story'] = target
            sample['mask'] = np.zeros_like(target, dtype='float32')
            nonzero_num = (target != 0).sum() + 1
            sample['mask'][:nonzero_num] = 1
            sample['index'] = np.int64(index)

            return sample

    def _process_image_att(self, im_in, im_g_att_in, im_att_in):
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
        processed_im_g_att_in = cv2.resize(processed_im_g_att_in, dsize=(8, 8),
                                  interpolation=cv2.INTER_NEAREST)
        processed_im_att_in = cv2.resize(processed_im_att_in, dsize=(8, 8),
                                  interpolation=cv2.INTER_NEAREST)

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

    def _process_image(self, im_in):
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

        if self.mode == "train":
            target_size = self.base_target_size * self.train_scale_rate
        else:
            target_size = self.base_target_size

        # 2. Random crop when in training mode, elsewise just skip
        if self.mode == "train":
            crop_ratio = np.random.random() * 0.4 + 0.6
            crop_size_h = int(im.shape[0] * crop_ratio)
            crop_size_w = int(im.shape[1] * crop_ratio)
            processed_im = self._crop(im, crop_size_h, crop_size_w, random=True)
        else:
            processed_im = im

        # 3. Resize to the target resolution
        im_shape = processed_im.shape
        im_scale_x = float(target_size) / im_shape[1]
        im_scale_y = float(target_size) / im_shape[0]
        processed_im = cv2.resize(processed_im, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)

        if self.mode == "train":
            if np.random.random() > 0.5:
                processed_im = self._hori_flip(processed_im)

        # Normalization
        if 'detector' in self.backbone_source:
            processed_im = self._detector_norm(processed_im)
        else:
            processed_im = self._imagenet_norm(processed_im)

        return processed_im

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

    def collate_func(self, batch):
        index_batch = []
        split_story_batch = []
        feature_fc_batch = []
        title_batch = []
        image_attention_batch = []
        length_collection = [tmp["length"] for tmp in batch]
        max_length = max(length_collection)

        for sample in batch:
            tmp_index, tmp_split_story, tmp_feature_fc, tmp_title, tmp_attention \
                = sample["index"], sample["split_story"], sample["feature_fc"], sample["title"], sample["image_attention"]

            reshape_split_story = np.zeros((max_length, tmp_split_story.shape[1]), dtype=np.int64)
            reshape_split_story[: tmp_split_story.shape[0]] = tmp_split_story
            reshape_feature_fc = np.zeros((max_length, tmp_feature_fc.shape[1], tmp_feature_fc.shape[2], 3), dtype=np.float32)
            reshape_feature_fc[: tmp_feature_fc.shape[0]] = tmp_feature_fc
            reshape_image_attention = np.zeros((max_length, 1, 8, 8), dtype=np.float32)
            reshape_image_attention[: tmp_attention.shape[0]] = tmp_attention

            index_batch.append(tmp_index)
            split_story_batch.append(reshape_split_story)
            feature_fc_batch.append(reshape_feature_fc)
            title_batch.append(tmp_title)
            image_attention_batch.append(reshape_image_attention)

        data = dict()
        data["index"] = np.stack(index_batch)
        data["split_story"] = np.stack(split_story_batch)
        data["feature_fc"] = np.stack(feature_fc_batch)
        data["title"] = np.stack(title_batch)
        data["image_attention"] = np.stack(image_attention_batch)

        data = {k: torch.from_numpy(v) if type(v) is np.ndarray else v for k, v in
                data.items()}

        data["feature_fc"] = data["feature_fc"].permute(0, 1, 4, 2, 3)

        return data

    def __len__(self):
        if self.task == 'story_telling':
            return len(self.story_ids[self.mode])
        elif self.task == 'image_captioning':
            return len(self.description_ids[self.mode])
        else:
            raise Exception("{} task is not proper for this dataset.".format(self.task))

    def train(self):
        self.mode = 'train'

    def val(self):
        self.mode = 'val'

    def test(self):
        self.mode = 'test'

    def set_option(self, data_type=None):
        if self.task == 'story_telling':
            if data_type is not None:
                self.data_type = data_type
        else:
            pass
            # logging.error("{} task is not proper for this dataset.".format(self.task))
            # raise Exception("{} task is not proper for this dataset.".format(self.task))

    def get_GT(self, index):
        if self.task == 'story_telling':
            story_id = self.story_ids[self.mode][index]
            story = self.story_line[self.mode][story_id]
            return story['origin_text']
        elif self.task == 'image_captioning':
            raise Exception("To be implemented.")
        else:
            raise Exception("{} task is not proper for this dataset.".format(self.task))

    def get_id(self, index):
        if self.task == 'story_telling':
            story_id = self.story_ids[self.mode][index]
            return self.story_line[self.mode][story_id]['task_id'], self.story_line[self.mode][story_id]['img_id']
        else:
            return self.description_ids[self.mode][index]

    def get_all_id(self, index):
        story_id = self.story_ids[self.mode][index]
        return self.story_line[self.mode][story_id]['task_id'], self.story_line[self.mode][story_id]['img_id']

    def get_vocab_size(self):
        return self.vocab_size

    def get_vocab(self):
        return self.id2word

    def get_word2id(self):
        return self.word2id

    def get_whole_story_length(self):
        return self.full_story_h5.shape[1]

    def get_story_length(self):
        return self.story_h5.shape[1]

    def get_title_length(self):
        return self.title_h5.shape[1]

    def get_caption_length(self):
        return self.desc_h5.shape[1]


if __name__ == "__main__":
    import sys
    import os
    import opts
    import time

    start = time.time()

    opt = opts.parse_opt()
    dataset = VISTDataset(opt)

    print("dataset finished: ", time.time() - start)
    start = time.time()

    dataset.train()
    train_loader = DataLoader(dataset, batch_size=64, shuffle=opt.shuffle, num_workers=8, collate_fn=dataset.collate_func)

    print("dataloader finished: ", time.time() - start)

    dataset.train()  # train() mode has to be called before using train_loader
    for iter, batch in enumerate(train_loader):
        print("enumerate: ", time.time() - start)
        print(iter)
        print(batch["feature_fc"].size())
