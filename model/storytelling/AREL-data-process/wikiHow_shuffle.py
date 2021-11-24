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
    info_file = os.path.join(output_dir, "split_amt_json_30000_sample.json")
    info_json = {}
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    info_keys = list(database_input_info.items())
    random.shuffle(info_keys)
    info_json = dict(info_keys)

    info_keys = list(info_json.keys())
    length = len(info_keys)
    print("the number of filtered posts is {}".format(length))
    train_keys, val_keys, test_keys = info_keys[: int(length * params["train_proportion"])], \
                                      info_keys[int(length * params["train_proportion"]):
                                                int(length * (params["train_proportion"] + params["val_proportion"]))], \
                                      info_keys[int(length * (params["train_proportion"] + params["val_proportion"])):]

    for key in train_keys:
        info_json[key]["split"] = "train"
    for key in val_keys:
        info_json[key]["split"] = "val"
    for key in test_keys:
        info_json[key]["split"] = "test"

    with open(info_file, "w") as f:
        json.dump(info_json, f, indent=2)


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