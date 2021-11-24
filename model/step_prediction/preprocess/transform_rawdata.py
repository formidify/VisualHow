import os
import json
from tqdm import tqdm
import numpy as np
import random
from collections import OrderedDict
import argparse
import html
from topological_sort import getAllTopologicalOrders, Graph

def main(params):
    seed = params["seed"]
    np.random.seed(seed)
    random.seed(seed)

    input_json = params["input_json"]
    input_file = params["input_file"]
    output_file = params["output_file"]

    with open(input_json) as f:
        data_info = json.load(f)

    results = {}
    with open(input_file) as f:
        for line in tqdm(f.readlines()):
            y = json.loads(line)
            y_output = y
            key_value = y_output["post_id"] + "_" + y_output["method_idx"]
            results[key_value] = {**y_output, **data_info[key_value]}

    # change the html -> string
    for key, value in results.items():
        step_num = len(value["step_to_object_selected_result"])
        for index in range(step_num):
            curr_obj = value["step_to_object_selected_result"][index]
            grounding_obj = curr_obj[0]
            grounding_obj = [html.unescape(_) for _ in grounding_obj]

            non_grounding_obj = curr_obj[1]
            non_grounding_obj = [html.unescape(_) for _ in non_grounding_obj]

            value["step_to_object_selected_result"][index][0] = grounding_obj
            value["step_to_object_selected_result"][index][1] = non_grounding_obj

    for key, value in tqdm(results.items()):
        if isinstance(value["step_to_dependency_index_result"], list):
            pass
        else:
            value["step_to_dependency_index_result"] = json.loads(value["step_to_dependency_index_result"])
        # if len(value["step_to_dependency_index_result"]) != len(value["step_list"]):
        #     print(value["step_to_dependency_index_result"])
        #     print(value["step_list"])
        #     print(value["post_id"])
        #     print(value["method_idx"])
        #     print(len(value["step_list"]))

        if value["dependency_type"] == "others":
            dependency_result = value["step_to_dependency_index_result"]
            # Number of nodes in the graph
            N = len(dependency_result)
            # List of graph edges as per above diagram
            # edges = [(5, 2), (5, 0), (4, 0), (4, 1), (2, 3), (3, 1)]
            edges = []
            for index in range(N):
                dependency_nodes = dependency_result[index]
                for dependency_node in dependency_nodes:
                    edges.append((dependency_node, index))

            # create a graph from edges
            graph = Graph(edges, N)
            graph_results = getAllTopologicalOrders(graph)

            value["topological_graph"] = graph_results


    counter = 0
    for key, value in tqdm(results.items()):
        if value.get("step_to_object_selected_index_position_result") is None:
            counter += 1
            step_to_object_selected_index_position_result = []
            step_num = len(value["step_to_object_selected_result"])
            step_list = value["step_list"]
            for index in range(step_num):
                cur_result = [[], []]
                cur_step = step_list[index]
                curr_obj = value["step_to_object_selected_result"][index]
                grounding_obj = curr_obj[0]
                for obj in grounding_obj:
                    start_pos = cur_step.find(obj)
                    end_pos = start_pos + len(obj)
                    if start_pos == -1:
                        print("error")
                    cur_result[0].append([start_pos, end_pos])
                    # for this case we only indexing the first word appearing in the sentence
                non_grounding_obj = curr_obj[1]
                for obj in non_grounding_obj:
                    start_pos = cur_step.find(obj)
                    end_pos = start_pos + len(obj)
                    if start_pos == -1:
                        print("error")
                    cur_result[1].append([start_pos, end_pos])
                    # for this case we only indexing the first word appearing in the sentence
                step_to_object_selected_index_position_result.append(cur_result)
            value["step_to_object_selected_index_position_result"] = step_to_object_selected_index_position_result

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_json', default="../data/amt_json_95627_all.json", help='json file')
    parser.add_argument('--input_file', default="../data/final_result_w_split_15395.txt", help='txt file')
    parser.add_argument('--output_file', default="../data/final_result_w_split_15395_TG.json", help='output file name')

    parser.add_argument('--seed', default=0, type=int, help='control the random seed')

    args = parser.parse_args()
    params = vars(args)  # convert to ordinary dict
    print('parsed input parameters:')
    print(json.dumps(params, indent=2))
    main(params)