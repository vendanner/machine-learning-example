#-*-coding:utf-8-*-

"""
lfm model
"""
import numpy as np
import operator
import ready

def lfm_train(train_data,F,alpha,beta,step):
    """
    训练模型，得到user_vec,item_vec
    :param train_data:
    :param F:embedding 维度
    :param alpha:梯度下降速率
    :param beta:正则化超参
    :param step:训练周期
    :return:
    """
    user_vec ={}
    item_vec = {}
    for count in range(step):
        for data in train_data:
            user_id,item_id,label = data
            if user_id not in user_vec:
                user_vec[user_id] = init_model(F)
            if item_id not in item_vec:
                item_vec[item_id] = init_model(F)
            delta = label - predict(user_vec[user_id],item_vec[item_id])
            # 梯度下降
            # for f in range(F):
            #     user_vec[user_id][f] += alpha*(delta*item_vec[item_id][f] - beta*user_vec[user_id][f])
            #     item_vec[item_id][f] += alpha*(delta*user_vec[user_id][f] - beta*item_vec[item_id][f])
            # 矩阵
            user_vec[user_id] += alpha * (delta * item_vec[item_id] - beta * user_vec[user_id])
            item_vec[item_id] += alpha * (delta * user_vec[user_id] - beta * item_vec[item_id])
        # 速率逐渐变慢
        # alpha = alpha*0.9

    return user_vec,item_vec
def recom_result(user_vec,item_vec,user_id,K = 50):
    """
    给用户推荐topK
    :param user_vec:
    :param item_vec:
    :param user_id:
    :param K:
    :return:
    """
    if user_id not in user_vec:
        return []
    record = {}
    for item_id in item_vec:
        score = predict(user_vec[user_id],item_vec[item_id])
        record[item_id] = score

    return [(item[0],round(item[1],3)) for item in (sorted(record.items(),key=lambda info:info[1],reverse = True)[:K])]
def show_topK_movies(item_info,result):
    """
    展示推荐结果
    :param item_info:
    :param result:
    :return:
    """
    for item in result:
        print(item)
        print(item_info[item[0]])
        print("***************************************")

def predict(user_vec,item_vec):
    """
    预测，由于label 是0，1，这里用余弦值
    :param user_vec:
    :param item_vec:
    :return:
    """
    return np.dot(user_vec,item_vec)/(np.linalg.norm(user_vec)*np.linalg.norm(item_vec))

def init_model(F):
    """
    随机初始化
    :param F:
    :return:
    """
    return np.random.rand(F)

if __name__ =="__main__":
    print("begin")
    item_info = ready.get_item_info('../data/movies.txt')
    # print(item_info["10"])
    # score_info = ready.get_ave_score('../data/ratings.txt')
    # print(score_info["23"])
    train_data = ready.get_train_data('../data/ratings.txt')
    print(len(train_data))
    user_vec,item_vec = lfm_train(train_data,50,0.01,0.01,100)
    for user_id in user_vec:
        user_id="5"
        print(user_id)
        result = recom_result(user_vec,item_vec,user_id)
        # 可以和原始数据集比对，高分影集大部分都被我们推荐
        show_topK_movies(item_info,result)
        break
