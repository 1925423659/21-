from tensorflow.examples.tutorials.mnist import input_data
import scipy.misc
import os

# 读取 MNIST 数据集
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# 把原始图片保存在 mnist_data 文件夹下
# 如果没有这个文件夹，会自动创建
save_dir = 'mnist_data/raw'
if os.path.exists(save_dir) is False:
    os.makedirs(save_dir)

# 保存前20张图片
for i in range(20):
    image_matrix = mnist.train.images[i]
    # MNIST 图片是 784 维的向量
    # 将其还原为 28*28 维的图像
    image_matrix = image_matrix.reshape(28, 28)
    # 保存的文件名
    filename = save_dir + '/' + 'mnist_train_%d.jpg' % i
    # 先用 scipy.misc.toimage 转换为图片
    # 再调用 save 保存
    scipy.misc.toimage(image_matrix, cmin=0, cmax=1).save(filename)