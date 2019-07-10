# -*-coding:utf-8-*-

import os


def get_ave_score(file_path):
    """
    获取 item 平均打分
    :param file_path:
    :return: a dict,key itemid,value ave_score
    """
    if not os.path.exists(file_path):
        print(file_path + " no exist")
        return {}

    first_flag = 0
    sum_score = {}
    with open(file_path) as f:
        for line in f.readlines():
            if first_flag == 0:
                first_flag += 1
                continue
            words = line.strip().split(',')
            if len(words) < 4:
                continue
            _,item_id,score,_ = words
            if item_id not in sum_score:
                sum_score.setdefault(item_id,[0,0])
            sum_score[item_id][0] += float(score)
            sum_score[item_id][1] += 1

    ave_score = {}
    for item_id in sum_score:
        ave_score[item_id] = round(sum_score[item_id][0]/sum_score[item_id][1],3)

    return ave_score


def get_item_cate(ave_score,file_path,item_num = 100):
    """
    获取cate_item dict，其包含对应 cate 的item
    :param ave_score:
    :param file_path: item_info
    :param item_num: 每个 cate 下保存多少个 item
    :return:
        dict:{item_id : {cate:ratio}}
        dict:{cat1:[item_id1,item_id2...]}
    """
    if not os.path.exists(file_path):
        print(file_path+" no exist")
        return {},{}

    first_flag = 0
    item_cate_ratio = {}
    record = {}
    cate_item = {}
    with open(file_path) as f:
        for line in f.readlines():
            if first_flag == 0:
                first_flag += 1
                continue
            words = line.strip().split(',')
            if len(words) < 3:
                continue
            item_id = words[0]
            cates = words[-1].split('|')
            ratio = round(1 / len(cates), 3)
            if item_id not in item_cate_ratio:
                item_cate_ratio[item_id] = {}
            for cate in cates:
                item_cate_ratio[item_id][cate] = ratio

    for item_id in item_cate_ratio:
        for cate in item_cate_ratio[item_id]:
            if cate not in record:
                record[cate] = {}
            score = ave_score.get(item_id,0)
            # cate 下 item 打分值
            record[cate][item_id] = score

    for cate in record:
        cate_item[cate] = [arr[0] for arr in sorted(record[cate].items(),key=lambda item:item[1],reverse=True)[:item_num]]

    return item_cate_ratio,cate_item
