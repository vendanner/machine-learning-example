# -*-coding:utf-8-*-
"""
用word2vec 方法训练出item 的向量表示，然后直接计算 item 相似度
1、构造用户观看序列
2、训练数据直接塞入gensim 训练item vec
3、item vec 计算相似度
"""

import os
from gensim.models.word2vec import Word2Vec


def get_item_info(file_path):
    """
    获取item 类别信息
    :param file_path:
    :return: {item_id:info}
    """
    if not os.path.exists(file_path):
        print(file_path+"not exists")
        return {}

    first_flag = 0
    item_info = {}
    with open(file_path) as f:
        for line in f.readlines():
            if first_flag == 0:
                first_flag += 1
                continue
            words = line.strip().split(',')
            if len(words) < 3:
                continue
            item_id,info = words[0],words[1]+words[2]
            if item_id not in item_info:
                item_info[item_id] = info

    return item_info

def get_train_data(input_file,output_file,score_thr = 4.0):
    """
    将input_file 评分文件内容，格式化成用户浏览喜欢视频记录:user_id:item1,item2...
    但评分小于 score_thr ，不参与计算
    :param input_file:
    :param output_file:
    :return train_data
    """
    if not os.path.exists(input_file):
        print(input_file +"is not exists")
        return
    first_flag = 0
    record = {}
    with open(input_file) as f:
        for line in f.readlines():
            if first_flag == 0:
                first_flag += 1
                continue
            words = line.strip().split(',')
            if len(words) < 4:
                continue
            user_id,item_id,score = words[0],words[1],float(words[2])
            if score < score_thr:
                continue
            if user_id not in record:
                record[user_id] = []
            record[user_id].append(item_id)
    train_data = []
    with open(output_file,'w') as f:
        for user_id in record:
            if len(record[user_id]) < 3:
                continue
            train_data.append(record[user_id])
            f.write(" ".join(record[user_id])+"\n" )

    return train_data

def get_item_vec(train_data,vec_file,model_file):
    """
    训练item 向量
    :param train_data:
    :param vec_file:
    :param model_file:
    :return:item2vec_model
    """
    n_dim = 128
    max_iter = 50
    item2vec_model = Word2Vec(size=n_dim)
    item2vec_model.build_vocab(train_data)
    item2vec_model.train(train_data,total_examples=item2vec_model.corpus_count,epochs = max_iter)
    item2vec_model.save(model_file)

    vec = {}
    for items in train_data:
        for item_id in items:
            if item_id not in vec:
                try:
                    vec[item_id] = item2vec_model[item_id]
                except KeyError:
                    continue

    with open(vec_file,'w') as f:
        for item_id in vec:
            f.write(item_id+" "+" ".join([str(ele) for ele in vec[item_id]])+"\n" )

    return vec

def recommend(item_id,model_file,item_info):
    """
    推荐相似 item
    :param item_id:
    :param model_file:
    :param item_info:
    :return:
    """
    if item_id not in item_info:
        print('error')
        return
    item2vec_mode = Word2Vec.load(model_file)
    lists = item2vec_mode.most_similar([item_id])
    print(lists)
    print(item_id+item_info[item_id]+" similar item:")
    for res in lists:
        if res[0] in item_info:
            print(item_info[res[0]])


if __name__ == "__main__":
    print("begin")
    item_info = get_item_info("../data/movies.txt")
    train_data = get_train_data("../data/ratings.txt","../data/train_data.txt")
    vec = get_item_vec(train_data,'../data/train_vec.txt',"../model/w2v_model.pkl")
    print(vec["1"])
    item_id = '1'
    recommend(item_id,"../model/w2v_model.pkl",item_info)