import json
import os.path as osp
import matplotlib.pyplot as plt
from collections import OrderedDict

base_path = "../../../ViST/data/origin_data/sis"
train_data = json.load(open(osp.join(base_path, "train.story-in-sequence.json")))
val_data = json.load(open(osp.join(base_path, "val.story-in-sequence.json")))
test_data = json.load(open(osp.join(base_path, "test.story-in-sequence.json")))

prefix = ["train", "val", "test"]
whole_album2im = {}
for i, data in enumerate([train_data, val_data, test_data]):
    album2im = {}
    for im in data['images']:
        if im['album_id'] not in album2im:
            album2im[im['album_id']] = [im['id']]
        else:
            if im['id'] not in album2im[im['album_id']]:
                album2im[im['album_id']].append(im['id'])
    whole_album2im[prefix[i]] = album2im

whole_album = {}
story_lines = {}
whole_lines = {}
story_line_count = 0
whole_line_count = 0
for i, data in enumerate([train_data, val_data, test_data]):
    album_mapping = {}
    for annot_new in data["annotations"]:
        annot = annot_new[0]
        assert len(annot_new) == 1
        text = annot['text'].encode('utf8')
        if annot['story_id'] not in album_mapping:
            album_mapping[annot['story_id']] = {"text_index": [story_line_count], "flickr_id": [annot['photo_flickr_id']], "length": 1,
                                                "album_id": annot['album_id'], "album_flickr_id": whole_album2im[prefix[i]][annot['album_id']],
                                                "whole_text_index": whole_line_count, "origin_text": text}
            story_lines[annot['story_id']] = [{"index": story_line_count, "text": text.split()}]
            whole_lines[annot['story_id']] = {"index": whole_line_count, "text": text.split()}
            whole_line_count +=1
        else:
            album_mapping[annot['story_id']]["text_index"].append(story_line_count)
            album_mapping[annot['story_id']]["flickr_id"].append(annot['photo_flickr_id'])
            album_mapping[annot['story_id']]["length"] += 1
            story_lines[annot['story_id']].append({"index": story_line_count, "text": text.split()})
            whole_lines[annot['story_id']]["text"].extend(text.split())
            album_mapping[annot['story_id']]["origin_text"] += " " + text
        story_line_count += 1
    whole_album[prefix[i]] = album_mapping

new_story_lines = []
for l in story_lines.values():
    for li in l:
        new_story_lines.append(li)
story_lines = new_story_lines
whole_lines = whole_lines.values()

story_lines = [r['text'] for r in sorted(story_lines, key=lambda thing: thing['index'])]
whole_lines = [r['text'] for r in sorted(whole_lines, key=lambda thing: thing['index'])]

print len(story_lines)
print len(whole_lines)

from collections import Counter
import numpy
cnt = Counter()
for l in story_lines:
    words = l
    for w in words:
        cnt[w] += 1
words2id = {}
idx = 2
for k, v in cnt.most_common():
    if v > 5:
        words2id[k] = idx
        idx += 1
words2id["<EOS>"] = 0
words2id["<UNK>"] = 1
id2words = {v:k for k,v in words2id.iteritems()}
print len(id2words)

whole_album["words2id"] = words2id
whole_album["id2words"] = {v:k for k,v in words2id.iteritems()}

id_story_lines = []
for l in story_lines:
    s = [words2id[w] if w in words2id else 1 for w in l]
    id_story_lines.append(s)

id_whole_lines = []
for l in whole_lines:
    s = [words2id[w] if w in words2id else 1 for w in l]
    id_whole_lines.append(s)

new_id_whole_lines = []
specify_longest = 105
for i in range(len(id_whole_lines)):
    cur_len = len(id_whole_lines[i])
    if cur_len < specify_longest:
        new_id_whole_lines.append(id_whole_lines[i] + [0] * (specify_longest - cur_len))
    else:
        new_id_whole_lines.append(id_whole_lines[i][:specify_longest-1] + [0])

data = numpy.asarray(new_id_whole_lines)
import h5py
f = h5py.File("full_story.h5", "w")
f.create_dataset("story", data=data)
f.close()

new_id_story_lines = []
specify_longest = 30
for i in range(len(id_story_lines)):
    cur_len = len(id_story_lines[i])
    if cur_len < specify_longest:
        new_id_story_lines.append(id_story_lines[i] + [0] * (specify_longest - cur_len))
    else:
        new_id_story_lines.append(id_story_lines[i][:specify_longest-1] + [0])

data = numpy.asarray(new_id_story_lines, "int32")
import h5py
f = h5py.File("story.h5", "w")
f.create_dataset("story", data=data)
f.close()
