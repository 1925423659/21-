from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import scipy.misc
import os
import numpy

# 读取 MNIST 数据集
mnist = read_data_sets('dataset/mnist', one_hot=True)

# 把原始图片保存在 mnist_data 文件夹下
# 如果没有这个文件夹，会自动创建
save_dir = 'dataset/image_label'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 前 20 张训练图片的 label
for i in range(20):
    # 得到独热表示
    one_hot_label = mnist.train.labels[i]
    # 通过 numpy.argmax，可以直接获得原始的 label
    label = numpy.argmax(one_hot_label)

    image_matrix = mnist.train.images[i]
    # MNIST 图片是 784 维的向量
    # 将其还原为 28*28 维的图像
    image_matrix = image_matrix.reshape(28, 28)
    # 保存的文件名
    filename = save_dir + '/' + 'mnist_train_%d_%d.jpg' % (i, label)
    # 先用 scipy.misc.toimage 转换为图片
    # 再调用 save 保存
    scipy.misc.toimage(image_matrix, cmin=0, cmax=1).save(filename)