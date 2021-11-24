import json
import os.path as osp
from collections import OrderedDict
import numpy as np
from tqdm import tqdm


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in tqdm(entries):
        vals = entry.split(' ')
        word = vals[0]
        vals = list(map(float, vals[1:]))
        word2emb[word] = np.array(vals)
        words = list(idx2word.values())
    for idx, word in tqdm(enumerate(words)):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb

with open("story_line.json") as f:
    whole_task = json.load(f, object_pairs_hook=OrderedDict)

emb_dim = 300
glove_file = '/media/xianyu/ICCV_2021/vqa_data/vqa2/glove/glove.6B.%dd.txt' % emb_dim
weights, word2emb = create_glove_embedding_init(whole_task["id2words"], glove_file)
np.save('embedding.npy', weights)