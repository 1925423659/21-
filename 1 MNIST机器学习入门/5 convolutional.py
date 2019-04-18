import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据
mnist = input_data.read_data_sets('mnist_data', one_hot=True)
# x 是训练图像的占位符
x = tensorflow.placeholder(tensorflow.float32, [None, 784])
# y_ 是训练图像标签的占位符
y_ = tensorflow.placeholder(tensorflow.float32, [None, 10])
# 将图片从 784 维向量重新还原为 28*28 的矩阵图片
x_matrix = tensorflow.reshape(x, [-1, 28, 28, 1])

# 第一层卷积层
# tensorflow.nn.conv2d 和 tensorflow.nn.max_pool 计算的尺寸跟 strides 有关
# 尺寸的计算方法是：ceil(size / strides_size)
w_conv_1 = tensorflow.Variable(tensorflow.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv_1 = tensorflow.Variable(tensorflow.constant(0.1, shape=[32]))
# 获得 28*28 的矩阵
features_conv_1 = tensorflow.nn.conv2d(x_matrix, w_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1
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
h_pool_2_reshape = tensorflow.reshape(h_pool_2, [-1, 7 * 7 * 64])
w_fc_1 = tensorflow.Variable(tensorflow.truncated_normal([7 * 7 * 64, 1024], stddev=0.1))
b_fc_1 = tensorflow.Variable(tensorflow.constant(0.1, shape=[1024]))
features_fc_1 = tensorflow.matmul(h_pool_2_reshape, w_fc_1) + b_fc_1
h_fc_1 = tensorflow.nn.relu(features_fc_1)
# 使用 Dropout，keep_prob 是占位符，训练时为 0.5，测试时为 1
keep_prob = tensorflow.placeholder(tensorflow.float32)
h_fc_1_drop = tensorflow.nn.dropout(h_fc_1, keep_prob)

# 把 1024 维的向量转换成 10 维，对应 10 个类别
w_fc_2 = tensorflow.Variable(tensorflow.truncated_normal([1024, 10], stddev=0.1))
b_fc_2 = tensorflow.Variable(tensorflow.constant(0.1, shape=[10]))
y = tensorflow.matmul(h_fc_1_drop, w_fc_2) + b_fc_2

# 计算交叉熵
cross_entropy = tensorflow.reduce_mean(tensorflow.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tensorflow.train.AdamOptimizer(0.0001).minimize(cross_entropy)

# 定义测试的准确率
correct_prediction = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

# 创建session，对变量初始化
sess = tensorflow.InteractiveSession()
sess.run(tensorflow.global_variables_initializer())

# 训练 20000 步
for i in range(20000):
    batch = mnist.train.next_batch(50)
    # 每 100 步报告一次在验证集上的准确率
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# 训练结束后报告在测试集上的准确率
print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))