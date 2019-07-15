# -*-coding:utf-8-*-

"""
收入预测：Adult Census Income(https://www.kaggle.com/uciml/adult-census-income)
    原始特征：'age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship',
                  'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country'
        离散特征："workclass", "education", "marital-status", "occupation",
                        "relationship", "race", "sex", "native-country"
        连续特征："age","education-num","capital-gain","capital-loss","hours-per-week"
    特征工程：
        连续特征：直接进DNN；离散后进LR
        离散特征：hash后进LR，接着 Embedding 进DNN
    模型：
        sigmod(LR+DNN)
"""

import os
import tensorflow as tf


def get_auc(predict_list, test_label):
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
    auc_score = (total_pos_index - (pos_num)*(pos_num + 1)/2) / (pos_num*neg_num)
    print("auc:{0:.5f}".format(auc_score))

def get_test_label(test_file):
    """
    get label of  test_file
    """
    if not os.path.exists(test_file):
        return []
    fp = open(test_file)
    linenum = 0
    test_label_list = []
    for line in fp:
        if linenum == 0:
            linenum+= 1
            continue
        if "?" in line.strip():
            continue
        item = line.strip().split(",")
        label_str = item[-1]
        if label_str==">50K":
            test_label_list.append(1)
        elif label_str == "<=50K":
            test_label_list.append(0)
        else:
            print(label_str)
            print("error")
    fp.close()
    return test_label_list

def input_fn(data_file, re_time, shuffle, batch_num, predict):
    """
    Args:
        data_file:input data , train_data, test_data
        re_time:time to repeat the data file
        shuffle: shuffle or not [true or false]
        batch_num:
        predict: train or test [true or false]
    Return:
        train_feature, train_label or test_feature
    """
    _CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''],
                        [0], [0], [0], [''], ['']]

    _CSV_COLUMNS = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num',
        'marital-status', 'occupation', 'relationship', 'race', 'gender',
        'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
        'label'
     ]

    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        classes = tf.equal(labels, '>50K')
        return features, classes

    def parse_csv_predict(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('label')
        return features

    data_set = tf.data.TextLineDataset(data_file).skip(1).filter(lambda line: tf.not_equal(tf.strings.regex_full_match(line, ".*\?.*"),True))
    if shuffle:
        data_set = data_set.shuffle(buffer_size=30000)
    if predict:
        data_set = data_set.map(parse_csv_predict, num_parallel_calls=5)
    else:
        data_set = data_set.map(parse_csv, num_parallel_calls=5)

    # 训练集重复次数
    data_set = data_set.repeat(re_time)
    # 设置  minibatch 大小
    data_set = data_set.batch(batch_num)
    return data_set

def get_feature_column():
    """
    获取LR 、DNN模型的特征名
    :return: LR 、DNN模型的特征名
    """
    age = tf.feature_column.numeric_column("age")
    education_num = tf.feature_column.numeric_column("education-num")
    capital_gain = tf.feature_column.numeric_column("capital-gain")
    capital_loss = tf.feature_column.numeric_column("capital-loss")
    hours_per_weak = tf.feature_column.numeric_column("hours-per-week")

    work_class = tf.feature_column.categorical_column_with_hash_bucket("workclass",hash_bucket_size=512)
    education = tf.feature_column.categorical_column_with_hash_bucket("education",hash_bucket_size =512)
    marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital-status",hash_bucket_size=512)
    occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",hash_bucket_size=512)
    realationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size = 512)
    race = tf.feature_column.categorical_column_with_hash_bucket("race",hash_bucket_size=512)
    sex = tf.feature_column.categorical_column_with_hash_bucket("sex",hash_bucket_size=512)
    native_country = tf.feature_column.categorical_column_with_hash_bucket("relationship",hash_bucket_size = 512)

    age_bucket = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
    gain_bucket = tf.feature_column.bucketized_column(capital_gain, boundaries=[0, 1000, 2000, 3000, 10000])
    loss_bucket = tf.feature_column.bucketized_column(capital_loss, boundaries=[0, 1000, 2000, 3000, 5000])

    cross_columns = [
        tf.feature_column.crossed_column([age_bucket,gain_bucket],hash_bucket_size=36),
        tf.feature_column.crossed_column([gain_bucket,loss_bucket],hash_bucket_size=16)
    ]
    base_columns = [work_class, education, marital_status, occupation, realationship, age_bucket, gain_bucket, loss_bucket,]
    wide_columns = base_columns + cross_columns
    deep_columns =[
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_weak,
        tf.feature_column.embedding_column(work_class,9),
        tf.feature_column.embedding_column(education,9),
        tf.feature_column.embedding_column(marital_status,9),
        tf.feature_column.embedding_column(occupation,9),
        tf.feature_column.embedding_column(realationship,9),
    ]
    return wide_columns, deep_columns

def build_model_estimator(wide_columns, deep_columns,model_folder):
    """
    构建模型、输入格式
    :param wide_columns: LR 特征
    :param deep_columns: DNN 特征
    :param model_folder: 模型保存路径
    :return:model_es,serving_input_fn
    """
    model_es = tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_folder,
        linear_feature_columns=wide_columns,
        linear_optimizer = tf.train.FtrlOptimizer(0.1,l2_regularization_strength=1.0),
        dnn_feature_columns=deep_columns,
        dnn_optimizer=tf.train.ProximalAdagradOptimizer(learning_rate =0.1,l1_regularization_strength =0.001,
                                                        l2_regularization_strength=0.001),
        # 定义网络结构
        dnn_hidden_units=[128,64,32,16]
    )
    feature_column = wide_columns + deep_columns
    feature_spec = tf.feature_column.make_parse_example_spec(feature_column)
    serving_input_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    return model_es,serving_input_fn

def train_wd_model(model_es, train_file, test_file, model_export_folder, serving_input_fn):
    """
    训练模型
    :param model_es:
    :param train_file:
    :param test_file:
    :param model_export_folder:
    :param serving_input_fn:
    :return:
    """
    total_run = 6
    for index in range(total_run):
        # input_fn 是数据输入函数
        model_es.train(input_fn = lambda: input_fn(train_file, 10,True, 100,False))
        print(model_es.evaluate(input_fn=lambda: input_fn(test_file, 1, False, 100, False)))
    model_es.export_savedmodel(model_export_folder,serving_input_fn)


def eval_model_performance(model_es, test_file):
    """
    评估模型
    :param model_es:
    :param test_file:
    :return:
    """
    label = get_test_label(test_file)
    result = model_es.predict(input_fn = lambda: input_fn(test_file, 1,False,100,False))
    predict_list = []
    for res in result:
        if "probabilities" in res:
            predict_list.append(res["probabilities"][1])
    get_auc(predict_list, label)


if __name__ == "__main__":
    print("begin")
    wide_columns, deep_columns = get_feature_column()
    model_es,serving_input_fn = build_model_estimator(wide_columns, deep_columns,"../model/wd")
    train_wd_model(model_es,"../data/income_train.txt","../data/income_test.txt","../model/wd_export",serving_input_fn)
    eval_model_performance(model_es, "../data/income_test.txt")