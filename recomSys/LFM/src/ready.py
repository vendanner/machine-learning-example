# -*-coding:utf-8-*-

"""
准备数据
"""

import os

def get_item_info(file_path):
    """
    获取电影信息
    :param file_path:
    :return: dict key itemID,value [title,genres]
    """
    if not os.path.exists(file_path):
        print(file_path ," not exists")
        return {}
    item_info = {}
    first_flag = 0
    fp = open(file_path)
    for line in fp:
        # 第一行是列信息，去除
        if first_flag == 0:
            first_flag =1
            continue
        words = line.strip().split(",")
        if len(words) < 3:
            continue
        elif len(words) == 3:
            item_id,title,genres = words[0],words[1],words[2]
        else :
            item_id = words[0]
            genres = words[-1]
            title = ','.join(words[1:-1])
        item_info[item_id] = [title,genres]
    fp.close()

    return item_info

def get_ave_score(file_path):
    """
    计算电影平均打分值
    :param file_path:
    :return: dict key itemid ,value score
    """
    if not os.path.exists(file_path):
        print(file_path," not exists")
        return {}
    item_ave_info = {}
    score_info = {}
    first_flag = 0
    fp = open(file_path)
    for line in fp:
        # 第一行是列信息，去除
        if first_flag == 0:
            first_flag = 1
            continue
        words = line.strip().split(",")
        if len(words) < 3:
            continue
        item_id,score = words[1],float(words[2])
        if item_id not in item_ave_info:
            # [打分个数，打分总值]
            item_ave_info[item_id] = [0,0]
        item_ave_info[item_id][0] += 1
        item_ave_info[item_id][1] += score
    fp.close()

    for item_id in item_ave_info:
        score_info[item_id] = round(item_ave_info[item_id][1]/item_ave_info[item_id][0],3)

    return score_info

def get_train_data(file_path,score_thr =4.0):
    """
    准备训练数据，为了均衡正负样本，这里保证userid 的正样本和负样本数量相同
    :param ffile_path:
    :return:[(userid,itemid,score)]
    """
    score_info = get_ave_score(file_path)
    if score_info =={}:
        return []
    first_flag = 0
    pos = {}
    neg = {}
    train_data =[]
    with open(file_path) as f:
        for line in f.readlines():
            if first_flag == 0:
                first_flag += 1
                continue
            words = line.strip().split(",")
            if len(words) <3:
                continue
            user_id,item_id,rating = words[0],words[1],float(words[2])
            if user_id not in pos:
                pos[user_id] = []
            if user_id not in neg:
                neg[user_id] = []
            # score < 4.0 当作负样本
            if rating < score_thr:
                score = score_info.get(item_id,0)
                neg[user_id].append((item_id,score))
            else:
                pos[user_id].append((item_id,1))

    for user_id in pos:
        # 必须保证有负样本，才加入训练
        num = min(len(pos[user_id]),len(neg.get(user_id,[])))
        if num > 0:
            train_data += [(user_id,item[0],item[1])for item in pos[user_id]][:num]
        else:
            continue
        train_data += [(user_id,info[0],0)for info in (sorted(neg[user_id],key=lambda item:item[1],reverse = True)[:num])]
    return train_data