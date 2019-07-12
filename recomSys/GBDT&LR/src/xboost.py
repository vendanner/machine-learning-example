#-*-coding:utf-8-*-
"""
gbdt&lr:
    1、gbdt 得到样本的叶节点序号
    2、叶节点序号 one-hot 后当离散特征进入LR
"""

import os
import sys
import math
import numpy as np
import sklearn.linear_model
import xgboost as xgb
import scipy.sparse as sp
import util
import ready

def get_mix_model_tree_info():
    """
    tree info of mix model
    """
    tree_depth = 4
    tree_num = 10
    step_size = 0.3
    result = (tree_depth, tree_num, step_size)
    return result

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

def eval_model(test_data,label,model,predict_func,lr_coef = None):
    """
    准确率和AUC 评估模型
    :param test_data:
    :param label:
    :param lr_model:
    :param predict_by_lr_model:
    :return:
    """
    if lr_coef is None:
        predict_label = predict_func(test_data, model)
    else:
        predict_label = predict_func(test_data, model, lr_coef)
    show_auc(predict_label,label)
    show_accuary(predict_label,label)

def train_xgboost_model_core(train_mat,tree_deep,tree_num,learning_rate):
    """
    xgboost 训练，返回xgboost 模型
    :param train_mat:训练数据
    :param tree_deep:树深度
    :param tree_num:集成树的数量
    :param learning_rate:alpha
    :return:模型
    """
    # objective 是模型的目标：回归、分类...
    # silent =0 打印信息，1不打印
    para_dict = {'max_depth':tree_deep,"eta":learning_rate,"objective":"reg:linear","silent": 1}
    best = xgb.train(para_dict,train_mat,tree_num)
    return best

def grid_search(train_mat):
    """
    网格搜寻最佳超参,输出模型auc
    :param train_mat: 训练数据
    :return:
    """
    grid_list = []
    tree_depth_list = [4, 5, 6]
    tree_num_list = [10, 50, 100]
    learning_rate_list = [0.3, 0.5, 0.7]
    for ele_deep in tree_depth_list:
        for ele_num in tree_num_list:
            for ele_rate in learning_rate_list:
                grid_list.append((ele_deep,ele_num,ele_rate))
    for ele in grid_list:
        deep,num,learning_rate = ele
        param_dict = {'max_depth':deep,"eta":learning_rate,"objective":"reg:linear","silent": 0}
        history = xgb.cv(param_dict,train_mat,num,nfold=5,metrics='auc')
        # train-auc-mean  train-auc-std  test-auc-mean  test-auc-std
        auc_score = history.loc[num - 1, ['test-auc-mean']].values[0]
        print("tree_depth:{},tree_num:{}, learning_rate:{}, auc:{}".format \
              (deep, num, learning_rate, auc_score))

def get_gbdt_and_lr_feature(tree_leaf,tree_num,tree_deepth):
    """
    将gbdt输出的叶序号成组合特征
    本例不是直接将序号one-hot，而是
    :param tree_leaf:叶序号
    :param tree_num:树的数量
    :param tree_deepth:树的深度
    :return:稀疏矩阵
    """
    # total_node_num = 2**(tree_deepth + 1) - 1
    # leaf_num = 2**(tree_deepth)
    # not_leaf_num = total_node_num - leaf_num
    # # 稀疏矩阵存储
    # col_num = []
    # row_num = []
    # data = []
    # row_index = 0
    # for one_leaf in tree_leaf:
    #     col_index = 0
    #     for leaf_index in one_leaf:
    #         leaf_num_adjust = leaf_index - not_leaf_num
    #         leaf_num_adjust = leaf_num_adjust if leaf_num_adjust >= 0 else 0
    #         col_num.append(leaf_num_adjust + col_index)
    #         row_num.append(row_index)
    #         data.append(1)
    #         col_index += leaf_num
    #     row_index += 1
    # 还是直接one-hot auc 高
    total_node_num = 2**(tree_deepth + 1) - 1
    # 稀疏矩阵存储
    col_num = []
    row_num = []
    data = []
    row_index = 0
    for one_leaf in tree_leaf:
        col_index = 0
        for leaf_index in one_leaf:
            col_num.append(leaf_index + col_index)
            row_num.append(row_index)
            data.append(1)
            col_index += total_node_num
        row_index += 1
    return sp.coo_matrix((data,(row_num,col_num)),shape=(len(tree_leaf),tree_num*total_node_num))

def predict_by_tree(test_feature, tree_model):
    """
    predict by gbdt model
    """
    predict_list = tree_model.predict(xgb.DMatrix(test_feature))
    return predict_list


def predict_by_lr_gbdt(test_feature, tree_model, lr_coef):
    """

    :param test_feature:
    :param tree_model:
    :param lr_coef:
    :return:
    """
    tree_leaf = tree_model.predict(xgb.DMatrix(test_feature), pred_leaf = True)
    tree_deep, tree_num, _ = get_mix_model_tree_info()
    feature_list = get_gbdt_and_lr_feature(tree_leaf,tree_num,tree_deep)
    # lr 模型
    result_list = np.dot(sp.csr_matrix(lr_coef),feature_list.tocsc().T).toarray()[0]
    sigmoid_ufunc = np.frompyfunc(sigmoid, 1, 1)
    return sigmoid_ufunc(result_list)


def sigmoid(x):
    """
    sigmoid function
    """
    return 1/(1+math.exp(-x))

def train_gbdt_model(train_file,feature_file,tree_model_file):
    """
    训练 xboost 模型
    :param train_file: 训练数据
    :param feature_file: 特征数
    :param tree_model_file: 模型保存路径
    :return:
    """
    train_data , label = ready.get_train_data(train_file,feature_file)
    # DMatrix xgboost 内部数据结构
    train_mat = xgb.DMatrix(train_data,label = label)

    # 网络搜索寻找最优超参
    # grid_search(train_mat)
    tree_num = 10
    tree_deep = 4
    learning_rate =0.3
    best = train_xgboost_model_core(train_mat,tree_deep,tree_num,learning_rate)
    best.save_model(tree_model_file)

def train_tree_and_lr_model(train_file,feature_file,tree_mix_model_file,lr_mix_model_file):
    """
    训练gbdt&LR 模型
    :param train_file: 训练数据
    :param feature_file: 特征数
    :param tree_mix_model_file:
    :param lr_mix_model_file:
    :return:
    """
    train_data , label = ready.get_train_data(train_file,feature_file)
    train_mat = xgb.DMatrix(train_data,label = label)
    tree_deepth ,tree_num,learning_rate = get_mix_model_tree_info()
    best = train_xgboost_model_core(train_mat,tree_deepth,tree_num,learning_rate)
    best.save_model(tree_mix_model_file)

    # tree_leaf 输出每个用户在每颗树的叶序号
    tree_leaf = best.predict(train_mat,pred_leaf=True)
    total_feature_list = get_gbdt_and_lr_feature(tree_leaf,tree_num,tree_deepth)

    # auc 指标
    lr_model = sklearn.linear_model.LogisticRegressionCV(Cs=[1], penalty='l2', tol=0.0001, max_iter=500, cv=5,scoring='roc_auc').fit(total_feature_list, label)
    print(lr_model.scores_[1])
    scores = lr_model.scores_[1]
    print('diff:{}'.format(','.join([str(ele) for ele in scores.mean(axis=0)])))
    print("auc:{0} (+-{1:2f})".format(scores.mean(), scores.std()*2))

    # 保存模型
    # 权重
    coef = lr_model.coef_[0]
    util.save_data(','.join([str(ele) for ele in coef]),lr_mix_model_file)


def gbdt_predict(train_file,model_file,feature_file):
    """
    xgboost 预测
    :param train_file:
    :param model_file:
    :param feature_file:
    :return:
    """
    test_data,label = ready.get_train_data(train_file,feature_file)
    model = xgb.Booster(model_file=model_file)
    eval_model(test_data,label,model,predict_by_tree)

def lr_gbdt_predict(train_file,gbdt_model_file,lr_model_file,feature_file):
    """
    lr_gbdt 模型预测
    :param train_file:
    :param gbdt_model_file:
    :param lr_model_file:
    :param feature_file:
    :return:
    """
    test_data,label = ready.get_train_data(train_file,feature_file)
    gbdt_model = xgb.Booster(model_file=gbdt_model_file)
    lr_coef = np.genfromtxt(lr_model_file,dtype=np.float32,delimiter=',')
    eval_model(test_data,label,gbdt_model,predict_by_lr_gbdt,lr_coef)

if __name__ == "__main__":
    print('')
    # df = ready.train_data('../data/train.txt','../data/test.txt','../data/train_data.txt','../data/train_test.txt','../data/feature.txt')
    # train_gbdt_model("../data/train_data.txt", "../data/feature.txt", "../model/xgb.model")
    train_tree_and_lr_model("../data/train_data.txt", "../data/feature.txt", "../model/xgb_mix_model","../model/lr_coef_mix_model")

    # gbdt_predict('../data/train_test.txt','../model/xgb.model','../data/feature.txt')
    lr_gbdt_predict('../data/train_test.txt','../model/xgb_mix_model','../model/lr_coef_mix_model','../data/feature.txt')