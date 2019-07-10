#-*-coding:utf-8-*-

"""
lr:
    lr预测后，我们是根据概率值从高到低排序
    auc 含义是指随机找正样本和负样本，lr预测出正样本的概率大于负样本的概率  的概率
    即 auc 指排序后正样本排在负样本之上的概率，故auc 可以较好的评估 rank 模型
"""

import sys
import numpy as np
import sklearn.linear_model
import sklearn.externals.joblib
import math
import util
import ready

def predict_by_lr_ceof(test_feature, lr_coef):
    """
    predict by lr_coef
    """
    sigmoid_func = np.frompyfunc(sigmoid, 1, 1)
    return sigmoid_func(np.dot(test_feature, lr_coef))


def sigmoid(x):
    """
    sigmoid function
    """
    return 1/(1+math.exp(-x))

def predict_by_lr_model(data,model):
    """

    :param data:
    :param model:
    :return:
    """
    result_list = []
    prob_list = model.predict_proba(data)
    for index in range(len(prob_list)):
        result_list.append(prob_list[index][1])
    return result_list

def show_auc(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of  test data
    auc = (sum(pos_index)-pos_num(pos_num + 1)/2)/pos_num*neg_num
    """
    total_list = []
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        label = test_label[index]
        total_list.append((label, predict_score))
    sorted_total_list = sorted(total_list, key = lambda ele:ele[1])
    neg_num = 0
    pos_num = 0
    count = 1
    total_pos_index = 0
    for zuhe in sorted_total_list:
        label, predict_score = zuhe
        if label == 0:
            neg_num += 1
        else:
            pos_num += 1
            total_pos_index += count
        count += 1
    # auc 公式
    auc_score = (total_pos_index - (pos_num)*(pos_num + 1)/2) / (pos_num*neg_num)
    print("auc:{0:.5f}".format(auc_score))


def show_accuary(predict_list, test_label):
    """
    Args:
        predict_list: model predict score list
        test_label: label of test data
    """
    score_thr = 0.5
    right_num = 0
    for index in range(len(predict_list)):
        predict_score = predict_list[index]
        if predict_score >= score_thr:
            predict_label = 1
        else:
            predict_label = 0
        if predict_label == test_label[index]:
            right_num += 1
    total_num = len(predict_list)
    accuary_score = right_num/total_num
    print("accuary:{0:.5f}".format(accuary_score))

def eval_model(test_data,label,model,predict_func):
    """
    准确率和AUC 评估模型
    :param test_data:
    :param label:
    :param lr_model:
    :param predict_by_lr_model:
    :return:
    """
    predict_label = predict_func(test_data,model)
    show_auc(predict_label,label)
    show_accuary(predict_label,label)


def train_model(train_file, model_coef, model_file, feature_num_file):
    """
    模型训练
    :param train_file:
    :param model_coef:
    :param model_file:
    :param feature_num_file:
    :return:
    """
    total_feature_num = util.get_feature_num(feature_num_file)
    label = np.genfromtxt(train_file,dtype=np.int32,delimiter =',',usecols = [-1])
    train_data = np.genfromtxt(train_file,dtype=np.int32,delimiter =',',usecols = np.arange(total_feature_num).tolist())
    #构建模型
    # LogisticRegressionCV 使用交叉验证来优化正则化系数C
    # Cs=[1,10,100] 正则化参数，tol=0.0001 两次迭代之间，公差 少于tol 则提前结束
    lr_model = sklearn.linear_model.LogisticRegressionCV(Cs=[1],penalty='l2',tol=0.0001,max_iter = 500,cv=5).fit(train_data,label)
    print(lr_model.scores_[1])
    scores = lr_model.scores_[1]
    print('diff:{}'.format(','.join([str(ele) for ele in scores.mean(axis=0)])))
    print("Accuracy:{0} (+-{1:2f})".format(scores.mean(), scores.std()*2))

    # auc 指标
    lr_model = sklearn.linear_model.LogisticRegressionCV(Cs=[1], penalty='l2', tol=0.0001, max_iter=500, cv=5,scoring='roc_auc').fit(train_data, label)
    print(lr_model.scores_[1])
    scores = lr_model.scores_[1]
    print('diff:{}'.format(','.join([str(ele) for ele in scores.mean(axis=0)])))
    print("auc:{0} (+-{1:2f})".format(scores.mean(), scores.std()*2))

    # 保存模型
    # 权重
    coef = lr_model.coef_[0]
    util.save_data(','.join([str(ele) for ele in coef]),model_coef)
    sklearn.externals.joblib.dump(lr_model,model_file)

def predict(test_file, model_coef, model_file, feature_num_file):
    """
    预测评估模型
    :param train_file:
    :param model_coef:
    :param model_file:
    :param feature_num_file:
    :return:
    """
    total_feature_num = util.get_feature_num(feature_num_file)
    label = np.genfromtxt(test_file,dtype=np.int32,delimiter =',',usecols = [-1])
    test_data = np.genfromtxt(test_file,dtype=np.int32,delimiter =',',usecols = np.arange(total_feature_num).tolist())
    lr_ceof = np.genfromtxt(model_coef, dtype=np.float32, delimiter=",")
    lr_model = sklearn.externals.joblib.load(model_file)

    print("model eval:")
    eval_model(test_data,label,lr_model,predict_by_lr_model)
    print("coef eval:")
    eval_model(test_data, label, lr_ceof, predict_by_lr_ceof)

if __name__ == "__main__":
    print('')
    # df = ready.train_data('../data/train.txt','../data/test.txt','../data/train_data.txt','../data/train_test.txt','../data/feature.txt')
    # train_model('../data/train_data.txt','../model/model_coef','../model/model_file','../data/feature.txt')
    predict('../data/train_test.txt','../model/model_coef','../model/model_file','../data/feature.txt')