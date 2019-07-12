#-*-coding:utf-8-*-

import os
import sys
import numpy as np
import pandas as pd
import util

"""
特征工程：
    1、标签值 one-hot
    2、连续值后续GBDT自动组合
"""

def get_train_data(train_file,feature_file):
    """
    从文件里组装出训练数据
    :param train_file:
    :param feature_file:
    :return:训练特征，训练标签
    """
    feature_num = util.get_feature_num(feature_file)
    train_label = np.genfromtxt(train_file,dtype=np.int32,delimiter=',',usecols=-1)
    feature_list = np.arange(feature_num).tolist()
    train_data = np.genfromtxt(train_file,dtype=np.int32,delimiter=',',usecols=feature_list)
    return train_data,train_label

def read_csv_data(file_path):
    """
    读取csv 文件
    :param file_path:
    :return:
    """
    if not os.path.exists(file_path):
        print(file_path+" no exists")
        return None

    # 第二列 fnlwgt 不参与建模
    use_list = np.arange(15).tolist()
    use_list.remove(2)
    # 这几列为 int ，其他strig 默认为 object
    dtype_dict = {"age": np.int32,
                  "education-num": np.int32,
                  "capital - gain": np.int32,
                  "capital - loss": np.int32,
                  "hours - per - week": np.int32}
    # 第一行为列名，分割符为','，数据缺失值表示为'?'
    # 直接删除有缺失值的样本
    df = pd.read_csv(file_path,sep=',',header = 0,dtype = dtype_dict,na_values ='?',usecols = use_list).dropna(axis=0,how='any')
    return df

def trans_label(x):
    """

    :param x:
    :return:
    """
    if x == "<=50K":
        return "0"
    if x == ">50K":
        return "1"
    return "0"

def get_label(label_feature_str,df):
    """
    label 从string 转换成 int
    :param label_feature_str: 标签列名
    :param df:
    :return: df，标签转换为数值后的df
    """
    df.loc[:,label_feature_str] = df.loc[:,label_feature_str].apply(trans_label)

def one_hot(x,index_dict):
    """
    序号变one-hot
    3 -> 0001
    5 -> 000001
    :param x:
    :param index_dict:
    :return: 本来直接返回one-hot，但其本身值是string，
            故这里返回'0001'
    """
    one_hot_list = [0]*len(index_dict)
    if x in index_dict:
        one_hot_list[index_dict[x]] = 1
    # test 异常值用众数值来补
    return ','.join([str(ele) for ele in one_hot_list])

def con_to_feature(x,desc_list):
    """
    获取值所在区间 index，直接返回one-hot
    :param x:
    :param desc_list:
    :return:2 -> '0,1,0'
    """
    size = len(desc_list) - 1
    result = [0] * size
    for i in range(size):
        if x >= desc_list[i] and  x <= desc_list[i+1]:
            result[i] = 1
            break
    return ','.join([str(ele) for ele in result])

def combine(feature_one_val, feature_two_val):
    """
    len(feature_one_val)  = 3
    len(feature_two_val) = 4
    新的特征维度 = 3*4
    :param feature_one_val:
    :param feature_two_val:
    :return: 新的特征值
    """
    one_list = feature_one_val.split(',')
    tow_list = feature_two_val.split(',')
    one_size = len(one_list)
    two_size = len(tow_list)
    result = [0] *(one_size * two_size)
    try:
        one_one = one_list.index('1')
    except:
        one_one = 0
    try:
        one_two = tow_list.index('1')
    except:
        one_two = 0
    result[one_one*two_size+one_two] = 1
    return ','.join([str(ele) for ele in result])

def process_dis_feature(feature_str,train_df,test_df):
    """
    标签特征变换成 one-hot
    1、统计特每个取值的数量
    2、按数量从高到低排序，给每个取值设置index
    3、按index on-hot
    :param feature_str:特征名称
    :param train_df:
    :param test_df:
    :return:int -> one-hot 长度
    """
    value_dict = train_df.loc[:,feature_str].value_counts().to_dict()
    index_dict = {}
    index = 0
    for item in sorted(value_dict.items(),key=lambda item:item[1],reverse=True):
        index_dict[item[0]] = index
        index += 1

    train_df.loc[:, feature_str] = train_df.loc[:,feature_str].apply(one_hot,args=(index_dict,))
    test_df.loc[:, feature_str] = test_df.loc[:, feature_str].apply(one_hot, args=(index_dict,))
    return len(index_dict)

def process_con_feature(feature_str,train_df,test_df):
    """
    连续值特征变换，本例将连续值也进行 one-hot 处理，test异常值用最小值来补
    1、得到特征值:min,25%,50%,75%,max，四个区间
    2、得到当前值所在的区间 index
    3、index 转为 one-hot
    :param feature_str:
    :param train_df:
    :param test_df:
    :return:
    """
    desc_dict = train_df.loc[:,feature_str].describe().to_dict()
    key_list = ["min", "25%", "50%", "75%", "max"]
    desc_list = [0] * len(key_list)
    for index in range(len(desc_list)):
        key = key_list[index]
        if key not in desc_dict:
            print("process_con_feature desc error")
            sys.exit()
        else:
            desc_list[index] = desc_dict[key]

    train_df.loc[:,feature_str] = train_df.loc[:,feature_str].apply(con_to_feature,args = (desc_list,))
    test_df.loc[:, feature_str] = test_df.loc[:, feature_str].apply(con_to_feature, args=(desc_list,))
    return len(desc_list) - 1

def combine_feature(feature_one, feature_two, new_feature, train_df, test_df, feature_num_dict):
    """
    将feature_one、feature_two 以笛卡儿积方式组合成新的特征 new_feature
    :param feature_one:
    :param feature_two:
    :param new_feature:
    :param train_df:
    :param test_df:
    :param feature_num_dict:
    :return:
    """
    train_df[new_feature] = train_df.apply(lambda row:combine(row[feature_one],row[feature_two]),axis=1)
    test_df[new_feature] = test_df.apply(lambda row: combine(row[feature_one], row[feature_two]), axis=1)

    if feature_one not in feature_num_dict:
        print(feature_one + "is not in feature_num_dict")
        sys.exit()
    if feature_two not in feature_num_dict:
        print(feature_two + "is not in feature_num_dict")
        sys.exit()

    return feature_num_dict[feature_one] * feature_num_dict[feature_two]

def train_data(train_file_path,test_file_path,out_train_path,out_test_path,features_path):
    """
    准备模型训练和预测数据
    :param train_file_path:
    :param test_file_path:
    :param features_path:
    :param out_train_path:
    :param out_test_path:
    :return:
    """
    # 读取数据
    train_df =  read_csv_data(train_file_path)
    test_df = read_csv_data(test_file_path)
    label_feature_str = "label"
    # 标签特征
    dis_feature_list = ["workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "sex", "native-country"]
    # 数值变量既连续特征
    con_feature_list = ["age","education-num","capital-gain","capital-loss","hours-per-week"]
    #
    index_list = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']

    get_label(label_feature_str, train_df)
    get_label(label_feature_str, test_df)

    # 开始构造特征
    dis_feature_num = 0
    con_feature_num = 0
    # 标签特征变换
    for feature_str in dis_feature_list:
        tmp = process_dis_feature(feature_str,train_df,test_df)
        dis_feature_num += tmp
    # 连续值特征不处理待后续GBDT自动组合
    for feature_str in con_feature_list:
        con_feature_num += 1

    util.save_df_data(train_df,out_train_path)
    util.save_df_data(test_df, out_test_path)
    util.save_data("feature_num="+str(dis_feature_num+con_feature_num),features_path)

    return train_df

if __name__ == "__main__":
    print("begin")
    pd.set_option('display.max_columns',1000)
    df = train_data('../data/train.txt','../data/test.txt','../data/train_data.txt','../data/train_test.txt','../data/feature.txt')
    print(df.head())
    # print(df.dtypes)