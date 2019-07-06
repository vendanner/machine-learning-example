#-*-coding:utf-8-*-
"""
准备数据
"""

import os
import numpy as np
import scipy.sparse

def get_mat_from_graph(graph):
    """
    从graph 得到转移矩阵 M
    :param graph:
    :return:
        m:转移矩阵
        keys：[];所有point
        keys_index：{};所有point 的index
    """
    keys = list(graph.keys())
    total = len(keys)
    keys_index = {}
    # 设定 point 的index
    for i in range(total):
        keys_index[keys[i]] = i

    # M 是稀疏矩阵，本例中scipy 来表示
    row = []
    col = []
    data = []
    for out_point in graph:
        weight = round(1 / len(graph[out_point]), 3)
        row_index = keys_index[out_point]
        for inner_point in graph[out_point]:
            col_index = keys_index[inner_point]
            row.append(row_index)
            col.append(col_index)
            data.append(weight)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    # 转移矩阵稀疏表示
    m = scipy.sparse.coo_matrix((data, (row, col)),shape = (total,total))
    return m,keys,keys_index

def get_graph_info(file_path,score_thr = 4.0):
    """
    准备图数据，有个准则：打分少于4 不参与
    :param file_path:
    :return: dict,{userid:{itemA:1,itemB:1},item:{userA:1}}
    """
    if not os.path.exists(file_path):
        print(file_path," not exists")
        return {}
    first_flag = 0
    graph = {}
    with open(file_path) as f:
        for line in f.readlines():
            if first_flag == 0:
                first_flag += 1
                continue
            words = line.strip().split(",")
            if len(words) < 3:
                continue
            user_id,item_id,rating = words[0],"item_"+words[1],float(words[2])
            if rating < score_thr:
                # 小于4 不参与
                continue
            if user_id not in graph:
                graph[user_id] = {}
            graph[user_id][item_id] = 1
            if item_id not in graph:
                graph[item_id] = {}
            graph[item_id][user_id] = 1

    return graph

def get_movies_info(file_path):
    """
    获取 movies 信息
    :param file_path:
    :return: dict,{itemid:[title,genre]}
    """
    if not os.path.exists(file_path):
        print(file_path+" not exists")
        return {}

    first_flag = 0
    item_info = {}
    with open(file_path) as f:
        for line in f.readlines():
            if first_flag == 0:
                first_flag += 1
                continue
            words = line.strip().split(",")
            if len(words) < 3:
                continue
            elif len(words) == 3:
                item_id,title,genres = words[0],words[1],words[2]
            else:
                item_id = words[0]
                genres = words[-1]
                title = ','.join(words[1:-1])
            item_info[item_id] = [title,genres]

    return item_info