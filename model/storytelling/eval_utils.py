# encoding=utf8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

# reload(sys)
# sys.setdefaultencoding('utf8')

import os
import json
import hashlib
import pandas as pd
import time
from vist_eval.album_eval import AlbumEvaluator
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader
import misc.utils as utils


class CocoResFormat:
    def __init__(self):
        self.res = []
        self.caption_dict = {}

    def read_multiple_files(self, filelist, hash_img_name):
        for filename in filelist:
            # print('In file %s\n' % filename)
            self.read_file(filename, hash_img_name)

    def read_file(self, filename, hash_img_name):
        count = 0
        with open(filename, 'r') as opfd:
            for line in opfd:
                count += 1
                id_sent = line.split('\t')
                if len(id_sent) > 2:
                    id_sent = id_sent[-2:]
                assert len(id_sent) == 2
                sent = id_sent[1].strip()

                if hash_img_name:
                    img_id = int(int(hashlib.sha256(id_sent[0].encode('utf8')).hexdigest(),
                                     16) % sys.maxsize)
                else:
                    img_id = id_sent[0]
                imgid_sent = {}

                if img_id in self.caption_dict:
                    print(img_id)
                    assert self.caption_dict[img_id] == sent
                else:
                    self.caption_dict[img_id] = sent
                    imgid_sent['image_id'] = img_id
                    imgid_sent['caption'] = sent
                    self.res.append(imgid_sent)

    def dump_json(self, outfile):
        with open(outfile, 'w') as fd:
            json.dump(self.res, fd, ensure_ascii=False, sort_keys=True,
                      indent=2, separators=(',', ': '))


class Evaluator:
    def __init__(self, opt, mode='val'):
        if opt.task == 'story_telling' or opt.task == 'story_telling_with_caption':
            ref_json_path = "data/reference/{}_reference.json".format(mode)
        else:
            ref_json_path = "data/reference/{}_desc_reference.json".format(mode)
        self.reference = json.load(open(ref_json_path))
        print("loading file {}".format(ref_json_path))
        self.save_dir = os.path.join(opt.checkpoint_path, opt.id)
        self.prediction_file = os.path.join(self.save_dir, 'prediction_{}'.format(mode))
        self.prediction_by_step_file = os.path.join(self.save_dir, 'prediction_by_step_{}.json'.format(mode))
        self.eval = AlbumEvaluator()

    def measure(self):
        json_prediction_file = '{}.json'.format(self.prediction_file)
        predictions = {}
        with open(self.prediction_file) as f:
            for line in f:
                vid, seq = line.strip().split('\t')
                if vid not in predictions:
                    predictions[vid] = [seq]
        self.eval.evaluate(self.reference, predictions)
        with open(json_prediction_file, 'w') as f:
            json.dump(predictions, f, indent=2)

        detail_json_prediction_file = '{}_detail.json'.format(self.prediction_file)
        detail_predictions = {"album_to_Gts": self.eval.album_to_Gts,
                              "album_to_Res": self.eval.album_to_Res,
                              "album_to_eval": self.eval.album_to_eval,
                              "eval_albums": self.eval.eval_albums,
                              "eval_overall": self.eval.eval_overall
                              }
        with open(detail_json_prediction_file, 'w') as f:
            json.dump(detail_predictions, f, indent=2)
        return self.eval.eval_overall

    def eval_story(self, model, crit, dataset, loader, opt, side_model=None):
        # Make sure in the evaluation mode
        logging.info("Evaluating...")
        start = time.time()
        model.eval()
        dataset.val()

        loss_sum = 0
        loss_evals = 0
        predictions = {}

        prediction_txt = open(self.prediction_file, 'w')  # open the file to store the predictions

        count = 0
        for iter, batch in enumerate(loader):
            iter_start = time.time()

            feature_fc = Variable(batch['feature_fc'], volatile=True).cuda()
            target = Variable(batch['split_story'], volatile=True).cuda()
            conv_feature = Variable(batch['feature_conv'], volatile=True).cuda() if 'feature_conv' in batch else None
            title = Variable(batch['title'], volatile=True).cuda()

            count += feature_fc.size(0)

            if side_model is not None:
                story, _ = side_model.predict(feature_fc.view(-1, feature_fc.shape[2]), 1)
                story = Variable(story).cuda()
                if conv_feature is not None:
                    output = model(feature_fc, target, story, conv_feature)
                else:
                    output = model(feature_fc, target, story)
            else:
                if conv_feature is not None:
                    output = model(feature_fc, target, conv_feature)
                else:
                    output, _ = model(feature_fc, target, title)

            loss = crit(output, target).data[0]
            loss_sum += loss
            loss_evals += 1

            # forward the model to also get generated samples for each video
            if side_model is not None:
                if conv_feature is not None:
                    results, _ = model.predict(feature_fc, story, conv_feature, beam_size=opt.beam_size)
                else:
                    results, _ = model.predict(feature_fc, conv_feature, beam_size=opt.beam_size)
            else:
                if conv_feature is not None:
                    results, _ = model.predict(feature_fc, conv_feature, beam_size=opt.beam_size)
                else:
                    results, _ = model.predict(feature_fc, title, beam_size=opt.beam_size)
            stories = utils.decode_story(dataset.get_vocab(), results)

            indexes = batch['index'].numpy()
            for j, story in enumerate(stories):
                vid, _ = dataset.get_id(indexes[j])
                if vid not in predictions:  # only predict one story for an album
                    # write into txt file for evaluate metrics like Cider
                    prediction_txt.write('{}\t {}\n'.format(vid, story))
                    # save into predictions
                    predictions[vid] = story

            logging.info("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(iter,
                                                                                len(loader),
                                                                                iter * 100.0 / len(loader),
                                                                                time.time() - iter_start))

        prediction_txt.close()
        metrics = self.measure()  # compute all the language metrics

        # Switch back to training mode
        model.train()
        dataset.train()
        logging.info("Evaluation finished. Evaluated {} samples. Time used: {}".format(count, time.time() - start))
        return loss_sum / loss_evals, predictions, metrics

    def test_story(self, model, dataset, loader, opt):
        logging.info("Evaluating...")
        start = time.time()
        model.eval()
        dataset.test()

        predictions = {}
        prediction_txt = open(self.prediction_file, 'w')  # open the file to store the predictions
        predictions_by_steps = {}

        for iter, batch in enumerate(loader):
            iter_start = time.time()

            feature_fc = Variable(batch['feature_fc'], volatile=True).cuda()
            feature_conv = Variable(batch['feature_conv'], volatile=True).cuda() if 'feature_conv' in batch else None
            title = Variable(batch['title'], volatile=True).cuda()
            if feature_conv is not None:
                results, _ = model.predict(feature_fc, feature_conv, beam_size=opt.beam_size)
            else:
                if opt.use_title:
                    results, _ = model.predict(feature_fc, title, beam_size=opt.beam_size)
                else:
                    results, _ = model.predict(feature_fc, beam_size=opt.beam_size)

            sents = utils.decode_story(dataset.get_vocab(), results)
            sents_by_step = utils.decode_story_by_step(dataset.get_vocab(), results)

            indexes = batch['index'].numpy()
            for j, story in enumerate(sents):
                vid, _ = dataset.get_id(indexes[j])
                if vid not in predictions:  # only predict one story for an album
                    # write into txt file for evaluate metrics like Cider
                    prediction_txt.write('{}\t {}\n'.format(vid, story))
                    # save into predictions
                    predictions[vid] = story

            for j, story in enumerate(sents_by_step):
                vid, _ = dataset.get_id(indexes[j])
                if vid not in predictions_by_steps:  # only predict one story for an album
                    # save into predictions_by_steps
                    predictions_by_steps[vid] = story

            print("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(iter,
                                                                         len(loader),
                                                                         iter * 100.0 / len(loader),
                                                                         time.time() - iter_start))

        prediction_txt.close()
        metrics = self.measure()  # compute all the language metrics

        with open(self.prediction_by_step_file, 'w') as f:
            json.dump(predictions_by_steps, f, indent=2)

        json.dump(metrics, open(os.path.join(self.save_dir, 'test_scores.json'), 'w'))
        # Switch back to training mode
        print("Test finished. Time used: {}".format(time.time() - start))
        return predictions, metrics

    def test_upper_bound(self, model, dataset, loader, opt):
        logging.info("Evaluating...")
        start = time.time()
        model.eval()
        dataset.test()

        predictions = {}
        prediction_txt = open(self.prediction_file, 'w')  # open the file to store the predictions

        for iter, batch in enumerate(loader):
            iter_start = time.time()
            results = batch['split_story'].cuda()
            sents = utils.decode_story(dataset.get_vocab(), results)
            sents_by_step = utils.decode_story_by_step(dataset.get_vocab(), results)

            indexes = batch['index'].numpy()
            for j, story in enumerate(sents):
                vid, _ = dataset.get_id(indexes[j])
                if vid not in predictions:  # only predict one story for an album
                    # write into txt file for evaluate metrics like Cider
                    prediction_txt.write('{}\t {}\n'.format(vid, story))
                    # save into predictions
                    predictions[vid] = story

            print("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(iter,
                                                                         len(loader),
                                                                         iter * 100.0 / len(loader),
                                                                         time.time() - iter_start))

        prediction_txt.close()
        metrics = self.measure()  # compute all the language metrics

        json.dump(metrics, open(os.path.join(self.save_dir, 'test_scores.json'), 'w'))
        # Switch back to training mode
        print("Test finished. Time used: {}".format(time.time() - start))
        return predictions, metrics

    def test_challange(self, model, dataset, loader, opt, side_model=None):
        # Make sure in the evaluation mode
        logging.info("Evaluating...")
        start = time.time()
        model.eval()
        dataset.test()

        predictions = {"team_name": "", "evaluation_info": {"additional_description": ""}, "output_stories": []}

        prediction_txt = open(self.prediction_file, 'w')  # open the file to store the predictions

        count = 0
        finished_flickr_ids = []
        for iter, batch in enumerate(loader):
            iter_start = time.time()

            feature_fc = Variable(batch['feature_fc'], volatile=True).cuda()
            conv_feature = Variable(batch['feature_conv'], volatile=True).cuda() if 'feature_conv' in batch else None
            count += feature_fc.size(0)
            if conv_feature is not None:
                results, _ = model.predict(feature_fc, conv_feature, beam_size=opt.beam_size)
            else:
                results, _ = model.predict(feature_fc, beam_size=opt.beam_size)
            stories = utils.decode_story(dataset.get_vocab(), results)

            indexes = batch['index'].numpy()
            for j, story in enumerate(stories):
                album_id, flickr_id = dataset.get_all_id(indexes[j])
                concat_flickr_id = "-".join(flickr_id)
                if concat_flickr_id not in finished_flickr_ids:
                    # if vid not in predictions:  # only predict one story for an album
                    # write into txt file for evaluate metrics like Cider
                    prediction_txt.write('{}\t {}\n'.format(album_id, story))
                    # save into predictions
                    predictions['output_stories'].append(
                        {'album_id': album_id, 'photo_sequence': flickr_id, 'story_text_normalized': story})
                    finished_flickr_ids.append(concat_flickr_id)

            logging.info("Evaluate iter {}/{}  {:04.2f}%. Time used: {}".format(iter,
                                                                                len(loader),
                                                                                iter * 100.0 / len(loader),
                                                                                time.time() - iter_start))

        prediction_txt.close()
        json_prediction_file = os.path.join(self.save_dir, 'challenge.json')
        with open(json_prediction_file, 'w') as f:
            json.dump(predictions, f)

        logging.info("Evaluation finished. Evaluated {} samples. Time used: {}".format(count, time.time() - start))
        return predictions
