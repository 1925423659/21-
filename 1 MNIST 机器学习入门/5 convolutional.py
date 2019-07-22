import tensorflow
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# 读入数据
mnist = read_data_sets('dataset/mnist', one_hot=True)
# x 是训练图像的占位符
x = tensorflow.placeholder(tensorflow.float32, [None, 784])
# y_ 是训练图像标签的占位符
y = tensorflow.placeholder(tensorflow.float32, [None, 10])

# 第一层卷积层
# tensorflow.nn.conv2d 和 tensorflow.nn.max_pool 计算的尺寸跟 strides 有关
# 尺寸的计算方法是：ceil(size / strides_size)
w_conv_1 = tensorflow.Variable(tensorflow.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv_1 = tensorflow.Variable(tensorflow.constant(0.1, shape=[32]))
# 将图片从 784 维向量重新还原为 28*28 的矩阵图片
x_reshape = tensorflow.reshape(x, [-1, 28, 28, 1])
# 获得 28*28 的矩阵
features_conv_1 = tensorflow.nn.conv2d(x_reshape, w_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1
h_conv_1 = tensorflow.nn.relu(features_conv_1)
# 获得 14*14 的矩阵
h_pool_1 = tensorflow.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第二层卷积层
w_conv_2 = tensorflow.Variable(tensorflow.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv_2 = tensorflow.Variable(tensorflow.constant(0.1, shape=[64]))
# 获得 14*14 的矩阵
features_conv_2 = tensorflow.nn.conv2d(h_pool_1, w_conv_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_2
h_conv_2 = tensorflow.nn.relu(features_conv_2)
# 获得 7*7 的矩阵
h_pool_2 = tensorflow.nn.max_pool(h_conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 全连接层
w_fc_1 = tensorflow.Variable(tensorflow.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc_1 = tensorflow.Variable(tensorflow.constant(0.1, shape=[1024]))
h_pool_2_reshape = tensorflow.reshape(h_pool_2, [-1, 7 * 7 * 64])
features_fc_1 = tensorflow.matmul(h_pool_2_reshape, w_fc_1) + b_fc_1
h_fc_1 = tensorflow.nn.relu(features_fc_1)
# 使用 Dropout，keep_prob 是占位符，训练时为 0.5，测试时为 1
keep_prob = tensorflow.placeholder(tensorflow.float32)
h_fc_1_dropout = tensorflow.nn.dropout(h_fc_1, keep_prob)

# 把 1024 维的向量转换成 10 维，对应 10 个类别
w_fc_2 = tensorflow.Variable(tensorflow.truncated_normal([1024, 10], stddev=0.1))
b_fc_2 = tensorflow.Variable(tensorflow.constant(0.1, shape=[10]))
y_ = tensorflow.matmul(h_fc_1_dropout, w_fc_2) + b_fc_2

#### 训练
# 计算交叉熵
cross_entropy = tensorflow.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
cross_entropy = tensorflow.reduce_mean(cross_entropy)

train_step = tensorflow.train.AdamOptimizer(0.0001).minimize(cross_entropy)

#### 测试
# 定义测试的准确率
predict = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1))
cast = tensorflow.cast(predict, tensorflow.float32)
accuracy = tensorflow.reduce_mean(cast)

# 创建session，对变量初始化
session = tensorflow.InteractiveSession()
session.run(tensorflow.global_variables_initializer())

# 训练 20000 步
for i in range(20000):
    batch_x, batch_y = mnist.train.next_batch(50)

    session.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})

    # 每 100 步报告一次在验证集上的准确率
    if i % 100 == 0:
        result = session.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, result))
    

# 训练结束后报告在测试集上的准确率
result = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob: 1.0})
print('test accuracy %g' % result)