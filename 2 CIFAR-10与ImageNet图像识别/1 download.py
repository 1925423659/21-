# 引入当前目录中的 cifar10 模块
import cifar10
# 引入 tensorflow
import tensorflow

# tensorflow.app.flags.FLAGS 是 tensorflow 内部的一个全局变量存储器
# 用于命令行参数的处理
FLAGS = tensorflow.app.flags.FLAGS
# 在 cifar10 模块中预先定义了 tensorflow.app.flags.FLAGS.data_dir 为 CIFAR-10 的数据路径
FLAGS.data_dir = 'cifar10_data'

# 如果不存在数据文件，执行下载
cifar10.maybe_download_and_extract()