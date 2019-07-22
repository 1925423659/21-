import tensorflow
import numpy

# x 是训练图像的占位符
x = tensorflow.constant(numpy.arange(784, dtype=float).reshape(1, 784))
# 将图片从 784 维向量重新还原为 28*28 的矩阵图片
x_matrix = tensorflow.reshape(x, [-1, 28, 28, 1])

# 第一层卷积层
w_conv_1 = tensorflow.constant(numpy.ones((5, 5, 1, 32)))
features_conv_1 = tensorflow.nn.conv2d(x_matrix, w_conv_1, strides=[1, 1, 1, 1], padding='SAME')
h_conv_1 = tensorflow.nn.relu(features_conv_1)
h_pool_1 = tensorflow.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

sess = tensorflow.InteractiveSession()
print(sess.run(h_conv_1).shape)
print(sess.run(h_pool_1).shape)