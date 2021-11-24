import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np
import random
from collections import OrderedDict
import argparse

'''
In this code, I remove all the 
'''


def main(params):
    seed = params["seed"]
    np.random.seed(seed)
    random.seed(seed)

    with open(params["input_info"]) as f:
        database_input_info = json.load(f, object_pairs_hook=OrderedDict)

    output_dir = params["output_dir"]
    info_file = os.path.join(output_dir, "split_amt_json_30000_sample_base_category.json")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    database_category_to_info = {}
    for key, value in database_input_info.items():
        category = value["category"][0]
        database_category_to_info.setdefault(category, []).append(value)

    train_num = 0
    val_num = 0
    test_num = 0
    used_sample_num = 0
    train_proportion = params["train_proportion"]
    val_proportion = params["val_proportion"]
    test_proportion = 1 - train_proportion - val_proportion
    for key, value in database_category_to_info.items():
        random.shuffle(value)
        length = len(value)
        print("the number of {} posts is {}".format(key, length))
        used_sample_num += length
        cur_val_num = int(round(used_sample_num * val_proportion)) - val_num
        cur_test_num = int(round(used_sample_num * test_proportion)) - test_num
        cur_train_num = used_sample_num - int(round(used_sample_num * val_proportion)) \
                        - int(round(used_sample_num * test_proportion))\
                        - train_num
        train_sets, val_sets, test_sets = \
            value[: cur_train_num], \
            value[cur_train_num: cur_val_num + cur_train_num],\
            value[cur_val_num + cur_train_num:]
        train_num += len(train_sets)
        val_num += len(val_sets)
        test_num += len(test_sets)
        print("the number of train/val/test is {}/{}/{}".format(train_num, val_num, test_num))

        for list_content, split_value in zip([train_sets, val_sets, test_sets], ["train", "val", "test"]):
            for list_content_value in list_content:
                list_content_value["split"] = split_value

    with open(info_file, "w") as f:
        json.dump(database_input_info, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_info', default='/srv/xianyu/CVPR_2022/storytelling/wikiHow/data/filtered_sample_images/amt_json_30000_sample.json',
                        help='input image info json file')
    parser.add_argument('--output_dir', default='/srv/xianyu/CVPR_2022/storytelling/wikiHow/data/filtered_sample_images', help='output json file')

    parser.add_argument('--train_proportion', default=0.8, type=float, help='proportion of train data')
    parser.add_argument('--val_proportion', default=0.1, type=float, help='proportion of val data')

    parser.add_argument('--seed', default=0, type=int, help='control the random seed')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)