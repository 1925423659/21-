# 从 tensorflow.examples.tutorials.mnist 引入模块
# 这是 TensorFlow 为了教学 MNIST 而提前编制的程序
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# 从 mnist_data 文件夹中读取 MNIST 数据
# 如果数据不存在，会自动执行下载
mnist = read_data_sets('dataset/mnist', one_hot=True)

# 查看训练数据的大小
print(mnist.train.images.shape)         # (55000, 784)
print(mnist.train.labels.shape)         # (55000, 10)

# 查看验证数据的大小
print(mnist.validation.images.shape)    # (5000, 784)
print(mnist.validation.labels.shape)    # (5000, 10)

# 查看测试数据的大小
print(mnist.test.images.shape)          # (10000, 784)
print(mnist.test.labels.shape)          # (10000, 10)

# 查看第1张图片的特征
print(mnist.train.images[0])

# 查看第1张图片的标签
print(mnist.train.labels[0])
