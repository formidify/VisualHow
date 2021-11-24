import os
import json
from tqdm import tqdm
import numpy as np
import random
from collections import OrderedDict
import argparse
import html
import spacy
from topological_sort import getAllTopologicalOrders, Graph
import itertools
import copy


def main(params):
    seed = params["seed"]
    np.random.seed(seed)
    random.seed(seed)

    nlp = spacy.load("en_core_web_md")  # make sure to use larger package!

    input_json = params["input_json"]
    output_file = params["output_file"]

    with open(input_json) as f:
        data_info = json.load(f)
    with open(input_json) as f:
        data_info_save = json.load(f)

    validation_ids = []
    test_ids = []

    validation_cat_split = {}
    test_cat_split = {}
    for key, value in tqdm(data_info.items()):

        if value["dependency_type"] == "sequential":
            topological_graph = [[_ for _ in range(len(value["step_list"]))]]
            value["topological_graph"] = topological_graph
        if value["dependency_type"] == "parallel":
            topological_index = [_ for _ in range(len(value["step_list"]))]
            topological_graph = list(itertools.permutations(topological_index, len(topological_index)))
            value["topological_graph"] = topological_graph

        if value["split"] == "val":
            validation_ids.append(key)
            validation_cat_split.setdefault(data_info[key]["category"][0], []).append(key)

        if value["split"] == "test":
            test_ids.append(key)
            test_cat_split.setdefault(data_info[key]["category"][0], []).append(key)

    # for val
    validation_goals = {}
    for key in validation_ids:
        task = data_info[key]["task_title"]
        method = data_info[key]["method_title"]
        validation_goals[key] = task + " " + method

    sent_vectors = np.zeros((len(validation_ids), 300), dtype=np.float32)
    for index, key in enumerate(validation_ids):
        doc = nlp(validation_goals[key])
        sent_vectors[index] = doc.vector

    similarity = np.matmul(sent_vectors, sent_vectors.transpose())

    validation_goals_selection = {}
    for index in tqdm(range(len(validation_ids))):
        select_goal = []
        select_goal.append(validation_ids[index])

        # add 10 hard goal
        similarity[index, index] = -10
        n = 10
        ranked = np.argsort(similarity[index])
        largest_indices = ranked[::-1][:n]

        for value in largest_indices:
            select_goal.append(validation_ids[value])

        category = data_info[validation_ids[index]]["category"][0]
        category_goal_ids = validation_cat_split[category]
        if len(set(category_goal_ids) | (set(select_goal))) < 100:
            for ii in range(len(category_goal_ids)):
                if category_goal_ids[ii] not in select_goal:
                    select_goal.append(category_goal_ids[ii])
            while True:
                if len(select_goal) == 100:
                    break
                rand_value = np.random.randint(len(validation_ids))
                if validation_ids[rand_value] not in select_goal:
                    select_goal.append(validation_ids[rand_value])
        else:
            while True:
                rand_value = np.random.randint(len(category_goal_ids))
                if category_goal_ids[rand_value] not in select_goal:
                    select_goal.append(category_goal_ids[rand_value])
                if len(select_goal) == 100:
                    break

        validation_goals_selection[validation_ids[index]] = select_goal

    for index in tqdm(range(len(validation_ids))):
        id_value = validation_ids[index]
        all_topological_graph = data_info[id_value]["topological_graph"]
        selected_topological_graph = all_topological_graph[random.randint(0, len(all_topological_graph) - 1)]
        current_step_index = random.randint(1, len(selected_topological_graph) - 1)

        previous_steps_index = [selected_topological_graph[idx] for idx in range(current_step_index)]
        current_step_index = selected_topological_graph[current_step_index]
        data_info_save[id_value]["previous_step_index"] = previous_steps_index
        data_info_save[id_value]["current_step_index"] = current_step_index
        data_info_save[id_value]["multiple_choice_candidates"] = validation_goals_selection[id_value]

    # for test
    test_goals = {}
    for key in test_ids:
        task = data_info[key]["task_title"]
        method = data_info[key]["method_title"]
        test_goals[key] = task + " " + method

    sent_vectors = np.zeros((len(test_ids), 300), dtype=np.float32)
    for index, key in enumerate(test_ids):
        doc = nlp(test_goals[key])
        sent_vectors[index] = doc.vector

    similarity = np.matmul(sent_vectors, sent_vectors.transpose())

    test_goals_selection = {}
    for index in tqdm(range(len(test_ids))):
        select_goal = []
        select_goal.append(test_ids[index])

        # add 10 hard goal
        similarity[index, index] = -10
        n = 10
        ranked = np.argsort(similarity[index])
        largest_indices = ranked[::-1][:n]

        for value in largest_indices:
            select_goal.append(test_ids[value])

        category = data_info[test_ids[index]]["category"][0]
        category_goal_ids = test_cat_split[category]
        if len(set(category_goal_ids) | (set(select_goal))) < 100:
            for ii in range(len(category_goal_ids)):
                if category_goal_ids[ii] not in select_goal:
                    select_goal.append(category_goal_ids[ii])
            while True:
                if len(select_goal) == 100:
                    break
                rand_value = np.random.randint(len(test_ids))
                if test_ids[rand_value] not in select_goal:
                    select_goal.append(test_ids[rand_value])
        else:
            while True:
                rand_value = np.random.randint(len(category_goal_ids))
                if category_goal_ids[rand_value] not in select_goal:
                    select_goal.append(category_goal_ids[rand_value])
                if len(select_goal) == 100:
                    break

        test_goals_selection[test_ids[index]] = select_goal

    for index in tqdm(range(len(test_ids))):
        id_value = test_ids[index]
        all_topological_graph = data_info[id_value]["topological_graph"]
        selected_topological_graph = all_topological_graph[random.randint(0, len(all_topological_graph) - 1)]
        current_step_index = random.randint(1, len(selected_topological_graph) - 1)

        previous_steps_index = [selected_topological_graph[idx] for idx in range(current_step_index)]
        current_step_index = selected_topological_graph[current_step_index]
        data_info_save[id_value]["previous_step_index"] = previous_steps_index
        data_info_save[id_value]["current_step_index"] = current_step_index
        data_info_save[id_value]["multiple_choice_candidates"] = test_goals_selection[id_value]

    with open(output_file, "w") as f:
        json.dump(data_info_save, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default="../data/final_result_w_split_20028_TG.json", help='json file')
    parser.add_argument('--output_file', default="../data/final_result_w_split_20028_step_question_new.json",
                        help='output file name')

    parser.add_argument('--seed', default=0, type=int, help='control the random seed')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)