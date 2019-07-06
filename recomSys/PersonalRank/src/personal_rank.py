#-*-coding:utf-8-*-
"""
personal rank 算法
"""

import ready
import time
import numpy as np
import scipy.sparse
import scipy.sparse.linalg

def personal_rank_train(graph,user_id,alpha,iter_num,recom_num=10):
    """
    给用户推荐 recom_num movies
    每个用户推荐商品都要重新计算知道收敛，非常耗时
    :param graph:
    :param user_id:
    :param alpha:
    :param iter_num:
    :param recom_num:
    :return:dict,{item:pr}
    """
    rank = {point:0 for point in graph}
    rank[user_id] = 1
    recom_result = {}
    for index in range(iter_num):
        tmp_rank = {point:0 for point in graph}
        # personal_rank 公式,稍微变换
        for out_point,out_items in graph.items():
            for inner_point,_ in out_items.items():
                tmp_rank[inner_point] += round(alpha * (rank[out_point]/len(out_items)),4)
                if inner_point == user_id:
                    tmp_rank[inner_point] += round(1 - alpha,4)
        if tmp_rank == rank:
            # 收敛，直接退出
            print(index+" out")
            break
        rank = tmp_rank
    # 查找最大 rank
    for item in sorted(rank.items(),key=lambda item:item[1],reverse = True):
        if len(item[0].split('_')) < 2:
            # 只推荐 item
            continue
        if item[0] in graph[user_id]:
            # 已消费的不推荐
            continue
        recom_result[item[0]] = item[1]
        recom_num -= 1
        if recom_num ==0:
            break

    return recom_result

def personal_rank_train_by_mat(graph,user_id,alpha,iter_num,recom_num=10):
    """
    矩阵式求解，加快推荐过程
    :param graph:
    :param user_id:
    :param alpha:
    :param iter_num:
    :param recom_num:
    :return:dict,{item:pr}
    """
    m,points,points_index = ready.get_mat_from_graph(graph)
    if user not in points_index:
        return {}
    m_all_points = get_all_point_pr(m,points,alpha)
    user_index = points_index[user_id]
    user_m = [0 for i in range(len(points))]
    user_m[user_index] = 1
    user_m = np.array(user_m)
    # 求解线性方程来替换求矩阵的逆
    # gress 可以解Ax=b 方程，但b必须是n*1;
    #gree 返回2个值，x 和status=是否成功解出方程
    res = scipy.sparse.linalg.gmres(m_all_points,user_m,tol =1e-8)[0]

    scores = {}
    for i in range(len(points)):
        point = points[i]
        if len(point.split('_')) < 2:
            continue
        # 已消费不推荐
        if point in graph[user_id]:
            continue
        scores[point] = round(res[i],3)
    return {item[0]:item[1] for item in (sorted(scores.items(),key=lambda  item:item[1],reverse=True)[:recom_num])}

def get_all_point_pr(m,points,alpha):
    """
    得到(E - alpha*m_mat.T) 值
    :param m: 转移矩阵
    :param points: points
    :param alpha:
    :return:矩阵，包含所有顶点的pr
    """
    total = len(points)
    rows = []
    cols = []
    data = []
    for i in range(total):
        rows.append(i)
        cols.append(i)
        data.append(1)
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.array(data)
    E = scipy.sparse.coo_matrix((data,(rows,cols)),shape=(total,total))
    return E.tocsr() - alpha*m.tocsr().transpose()

if __name__ == "__main__":
    print("begin")
    graph_info = ready.get_graph_info('../data/ratings.txt')
    # print(graph_info["1"])
    movies_info = ready.get_movies_info('../data/movies.txt')
    # print(movies_info["1"])
    user = "1"
    alpha = 0.8

    times = time.time()
    # recom_result = personal_rank_train(graph_info,user,alpha,100)
    # 矩阵比之前快50倍
    recom_result = personal_rank_train_by_mat(graph_info, user, alpha, 100)
    print("times: "+str(time.time() - times))
    print(recom_result)