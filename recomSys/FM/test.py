import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm

np.set_printoptions(threshold=np.inf)

def batcher(X_, y_=None, batch_size=-1):
    n_samples = X_.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = X_[i:upper_bound]
        ret_y = None
        if y_ is not None:
            ret_y = y_[i:i + batch_size]
            yield (ret_x, ret_y)

# 电影评分数据
# 列名
# laod data with pandas
cols = ['user','item','rating','timestamp']

train = pd.read_csv('data/ua.base',delimiter='\t',names = cols)
test = pd.read_csv('data/ua.test',delimiter='\t',names = cols)
y_train = train['rating'].values
y_test = test['rating'].values

train_users = train["user"].values
train_items = train["item"].values
test_users = test["user"].values
test_items = test["item"].values
num_users = np.max(train_users)+1
num_items = np.max(train_items)+1
train_users_one_hot = tf.one_hot(indices=train_users,depth=num_users,axis =1)
train_items_one_hot = tf.one_hot(indices=train_items,depth=num_items,axis =1)
test_users_one_hot = tf.one_hot(indices=test_users,depth=num_users,axis =1)
test_items_one_hot = tf.one_hot(indices=test_items,depth=num_items,axis =1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train = sess.run(tf.concat([train_users_one_hot,train_items_one_hot],1))
    test = sess.run(tf.concat([test_users_one_hot,test_items_one_hot],1))

print(train.shape)
print(test.shape)

x_train = train
x_test = test
n,p = x_train.shape
# Embedding 维度
k = 10

x = tf.placeholder('float',[None,p])

y = tf.placeholder('float',[None,1])

w0 = tf.Variable(tf.zeros([1]))
w = tf.Variable(tf.zeros([p]))

# 每个电影的 Embedding
v = tf.Variable(tf.random_normal([k,p],mean=0,stddev=0.01))

# FM 线性部分：特征各自计算
linear_terms = tf.add(w0,tf.reduce_sum(tf.multiply(w,x),1,keep_dims=True)) # n * 1
# FM 中特征组合部分
pair_interactions = 0.5 * tf.reduce_sum(
    tf.subtract(
        tf.pow(tf.matmul(x,tf.transpose(v)),2),
        tf.matmul(tf.pow(x,2),tf.transpose(tf.pow(v,2)))
    ),axis = 1 , keep_dims=True)

# 模型输出的值
y_hat = tf.add(linear_terms,pair_interactions)
# L2 正则
lambda_w = tf.constant(0.001,name='lambda_w')
lambda_v = tf.constant(0.001,name='lambda_v')
l2_norm = tf.reduce_sum(
    tf.add(
        tf.multiply(lambda_w,tf.pow(w,2)),
        tf.multiply(lambda_v,tf.pow(v,2))
    )
)
# 定义损失函数(增加 L2 正则)
error = tf.reduce_mean(tf.square(y-y_hat))
loss = tf.add(error,l2_norm)

train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

epochs = 10
batch_size = 1000

# Launch the graph
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for epoch in tqdm(range(epochs), unit='epoch'):
        perm = np.random.permutation(x_train.shape[0])
        # iterate over batches
        for bX, bY in batcher(x_train[perm], y_train[perm], batch_size):
            _,t = sess.run([train_op,loss], feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)})

            print(t)

    errors = []
    for bX, bY in batcher(x_test, y_test):
        errors.append(sess.run(error, feed_dict={x: bX.reshape(-1, p), y: bY.reshape(-1, 1)}))
        print(errors)
    RMSE = np.sqrt(np.array(errors).mean())
    print (RMSE)