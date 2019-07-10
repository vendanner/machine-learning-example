#-*-coding:utf-8-*-

import os

def save_data(data,path):
    """

    :param data:
    :param path:
    :return:
    """
    f = open(path,'w')
    f.write(data)
    f.close()

def save_df_data(df,path):
    """
    将df 数据保存到文件
    :param df:
    :param path:
    :return:
    """
    f = open(path,'w')
    for index in df.index:
        content = ','.join([str(ele) for ele in df.loc[index].values])
        f.write(content+'\n')
    f.close()

def get_feature_num(feature_num_file):
    """
    读取特征维度
    :param feature_num_file:
    :return: int
    """
    if not os.path.exists(feature_num_file):
        return 0
    else:
        fp = open(feature_num_file)
        for line in fp:
            item = line.strip().split("=")
            if item[0] == "feature_num":
                return int(item[1])
        return 0

