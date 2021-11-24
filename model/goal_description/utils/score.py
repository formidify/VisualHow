import json
from tqdm import tqdm
import numpy as np
import copy
import re

from evaluation.bleu import Bleu
from evaluation.meteor import Meteor
from evaluation.rouge import Rouge
from evaluation.cider import Cider
from evaluation.spice import Spice
from evaluation.tokenizer import PTBTokenizer

def compute_scores(gts, gen, all_gts):
    metrics = (Bleu(), Meteor(), Rouge(), Cider(all_gts), Spice())
    # metrics = (Bleu(), Meteor(), Rouge(), Cider(all_gts))
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores

def map_of_annotations_set(json_file_name):
    f_josn = open(json_file_name)
    data = json.load(f_josn)
    annos = data['annotations']
    id_to_index = dict()
    for index in range(len(annos)):
        anno = annos[index]
        id_to_index.setdefault(anno['image_id'], []).append(index)
    return id_to_index, annos


def split_sent(sent):
  sent = sent.lower()
  sent = re.sub('[^A-Za-z0-9\s]+','', sent)
  return sent.split()


def caption_scores(cap_gens_token, cap_gts_token):
    scores, all_scores = compute_scores(cap_gts_token, cap_gens_token, cap_gts_token)
    return scores, all_scores
