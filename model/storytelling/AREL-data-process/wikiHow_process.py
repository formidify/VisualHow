import json
import os.path as osp
import matplotlib.pyplot as plt
from collections import OrderedDict
from tqdm import tqdm

word_num = 5

def replace_special_characters(title):
	# replace special characters in title with proper strings
	title = title.replace('\xC3\xA2', 'a')
	title = title.replace('\xC3\xA9', 'e')
	title = title.replace('\xC3\xA8', 'e')
	title = title.replace('\xC3\x89', 'E')
	title = title.replace('\xC3\xAD', 'i')
	title = title.replace('\xC3\xB3', 'o')
	title = title.replace('\xC3\xB6', 'o')
	title = title.replace('\xC3\xB1', 'n')
	title = title.replace('\xC5\x9B', 's')
	title = title.replace('\xE2\x82\x82', '2')
	title = title.replace('\xE2\x80\x90', '-')
	title = title.replace('\xE2\x80\x93', '-')
	title = title.replace('\x27', '\'')
	title = title.replace('\xE2\x80\x98', '\'')
	title = title.replace('\xE2\x80\x99', '\'')
	title = title.replace('(\xE2\x88\x9E)', '') # infinity symbol - get rid of
	title = title.replace('\xE2\x84\xA2', '') # TM symbol - get rid of
	title = title.replace('\xC2\xAE', '') # R symbol - get rid of
	title = title.replace('(\xED\x8C\xA5\xEB\xB9\x99\xEC\x88\x98)', '') # korean words - get rid of
	title = title.replace('(\xC2\xB0C)', '') # celsius symbol - get rid of
	title = title.replace('(\xC2\xB0F)', '') # fahrenheit symbol - get rid of
	title = title.replace('\x22', '"')
	title = title.replace('\xE2\x80\x9C', '"')
	title = title.replace('\xE2\x80\x9D', '"')
	title = title.replace('\x26', '&')
	title = title.replace('\x2B', '+')
	title = title.replace('\x25', '%')
	title = title.replace('\x3D', '=')
	title = title.replace('\x3F', '?')
	title = title.replace('\x5E', '^')
	title = title.replace('\x24', '$')

	return title

def transform_sentence(sent, with_full_stop=True):
    sent = sent.encode('utf8')
    sent = sent.lower()
    sent = replace_special_characters(sent)
    sent = sent.strip().strip(".").strip()
    if with_full_stop:
        sent = sent.replace(',', '').replace('"', ''). \
                   replace('(', '').replace(')', '').replace('!', ''). \
                   replace('?', '').replace('\'s', ' \'s') + ' .'
        # sent = sent.replace(',', '').replace('"', '').\
        #            replace('(', '').replace(')', '').replace('!', '.').\
        #            replace('?', '.') + ' .'
    else:
        sent = sent.replace(',', '').replace('"', ''). \
            replace('(', '').replace(')', '').replace('!', ''). \
            replace('?', '').replace('\'s', ' \'s')
    sent = sent.split(" ")
    for index in range(len(sent)):
        try:
            sent[index].encode('ASCII')
        except:
            sent[index] = "<UNK>"
    sent = " ".join(sent)
    # try:
    #     a = sent.encode('ascii')
    # except:
    #     exception_list.setdefault(task_id, []).append(sent)
    return sent

info_file = "/media/CVPR_2022/data/wikihow/final_result_w_split_20028_raw.json"
data_info = json.load(open(info_file), object_pairs_hook=OrderedDict)
whole_task2im = {"train": {}, "val": {}, "test": {}}


for task_id, task in tqdm(data_info.items()):
    image_urls = task["image_url"]
    visual_file_list = []
    for image_url in image_urls:
        file_name = image_url.split("/")[-1].split(".")[0]
        visual_file_list.append(file_name)
    whole_task2im[task["split"]][task_id] = visual_file_list

whole_task = {}
step_lines = {}
whole_lines = {}
title_lines = {}
step_line_count = 0
whole_line_count = 0

train_task_mapping = {}
val_task_mapping = {}
test_task_mapping = {}

exception_list = {}
training_index = []
for task_id, task in tqdm(data_info.items()):
    # post_id = task["post_id"]
    task_mapping = {}
    step_list = []
    for annotation_item in task["step_list"]:
        step_list.append(annotation_item)
    transformed_step_list = []
    for sent in step_list:
        sent = transform_sentence(sent)
        transformed_step_list.append(sent)

    method_title = task["method_title"]
    transformed_method_title = transform_sentence(method_title, with_full_stop=False)


    task_mapping = {"text_index": range(step_line_count, step_line_count + len(transformed_step_list)),
                    "task_id": task_id,
                    "img_id": whole_task2im[task["split"]][task_id],
                    "length": len(transformed_step_list),
                    "whole_text_index": whole_line_count,
                    "origin_text": " ".join(transformed_step_list),
                    "method_title": transformed_method_title}
    if task["split"] == "train":
        train_task_mapping[task_id] = task_mapping
    elif task["split"] == "val":
        val_task_mapping[task_id] = task_mapping
    elif task["split"] == "test":
        test_task_mapping[task_id] = task_mapping
    step_lines[task_id] = [{"index": step_line_count + index, "text": transformed_step_list[index].split()}
                           for index in range(len(transformed_step_list))]
    whole_lines[task_id] = {"index": whole_line_count, "text": " ".join(transformed_step_list).split()}
    title_lines[task_id] = {"index": whole_line_count, "text": transformed_method_title.split()}
    step_line_count += len(transformed_step_list)
    whole_line_count += 1

whole_task["train"] = train_task_mapping
whole_task["val"] = val_task_mapping
whole_task["test"] = test_task_mapping

new_step_lines = []
for l in step_lines.values():
    for li in l:
        new_step_lines.append(li)
step_lines = new_step_lines
whole_lines = whole_lines.values()
title_lines = title_lines.values()

step_lines = [r['text'] for r in sorted(step_lines, key=lambda thing: thing['index'])]
whole_lines = [r['text'] for r in sorted(whole_lines, key=lambda thing: thing['index'])]
title_lines = [r['text'] for r in sorted(title_lines, key=lambda thing: thing['index'])]

print len(step_lines)
print len(whole_lines)
print len(title_lines)


from collections import Counter
import numpy
cnt = Counter()
for l in step_lines:
    words = l
    for w in words:
        cnt[w] += 1
for l in title_lines:
    words = l
    for w in words:
        cnt[w] += 1
words2id = {}
idx = 2
for k, v in cnt.most_common():
    if v > word_num and k != "<UNK>":
        words2id[k] = idx
        idx += 1
words2id["<EOS>"] = 0
words2id["<UNK>"] = 1
id2words = {v:k for k,v in words2id.iteritems()}
print len(id2words)

whole_task["words2id"] = words2id
whole_task["id2words"] = {v:k for k,v in words2id.iteritems()}

id_step_lines = []
for l in step_lines:
    s = [words2id[w] if w in words2id else 1 for w in l]
    id_step_lines.append(s)

id_whole_lines = []
for l in whole_lines:
    s = [words2id[w] if w in words2id else 1 for w in l]
    id_whole_lines.append(s)

id_title_lines = []
for l in title_lines:
    s = [words2id[w] if w in words2id else 1 for w in l]
    id_title_lines.append(s)

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
specify_longest = 16
for i in range(len(id_step_lines)):
    cur_len = len(id_step_lines[i])
    if cur_len < specify_longest:
        new_id_story_lines.append(id_step_lines[i] + [0] * (specify_longest - cur_len))
    else:
        new_id_story_lines.append(id_step_lines[i][:specify_longest-1] + [0])

data = numpy.asarray(new_id_story_lines, "int32")
import h5py
f = h5py.File("story.h5", "w")
f.create_dataset("story", data=data)
f.close()

new_id_title_lines = []
specify_longest = 10
for i in range(len(id_title_lines)):
    cur_len = len(id_title_lines[i])
    if cur_len < specify_longest:
        new_id_title_lines.append(id_title_lines[i] + [0] * (specify_longest - cur_len))
    else:
        new_id_title_lines.append(id_title_lines[i][:specify_longest-1] + [0])

data = numpy.asarray(new_id_title_lines, "int32")
import h5py
f = h5py.File("title.h5", "w")
f.create_dataset("title", data=data)
f.close()


with open("story_line.json", 'w') as f:
    json.dump(whole_task, f)