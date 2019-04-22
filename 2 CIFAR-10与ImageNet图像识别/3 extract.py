import tensorflow
import os
import scipy.misc
# 导入当前目录的 cifar10_input
# 这个模块负责读入 cifar10 数据
import cifar10_input

# 创建文件夹 cifar10_data/raw
if not os.path.exists('cifar10_data/raw'):
    os.makedirs('cifar10_data/raw')

# 创建会话 session
with tensorflow.Session() as sess:
    # cifar10_data/cifar-10-batches-bin 是下载数据的文件夹位置
    # filenames 一共 5 个
    # 从 data_batch_1.bin 到 data_batch_5.bin
    # 读入的都是训练图像
    filenames = [os.path.join('cifar10_data/cifar-10-batches-bin', 'data_batch_%d.bin' % i) for i in range(1, 6)]
    # 判断文件是否存在
    for f in filenames:
        if not tensorflow.gfile.Exists(f):
            raise ValueError('Failed to find file: ' + f)
    # 将文件名的 list 包装成 tensorflow 中 queue 的形式
    filename_queue = tensorflow.train.string_input_producer(filenames)
    # cifar10_input.read_cifar10 是从 queue 中读取文件的函数
    # 返回的结果 read_input 的属性 uint8image 是图像的 tensor
    read_input = cifar10_input.read_cifar10(filename_queue)
    # 将图片转换为实数形式
    # 返回的 reshaped_image 是一张图片的 tensor
    # 应当这样理解 reshaped_image：每次使用 sess.run(reshaped_image)，就会取出一张图片
    reshaped_image = tensorflow.cast(read_input.uint8image, tensorflow.float32)

    # 这一步 start_queue_runner 很重要
    # 之前有 filename_queue = tensorflow.train.string_input_producer(filenames)
    # 这个 queue 必须通过 start_queue_runners 才能启动
    # 若缺少 start_queue_runners，程序将不能执行
    threads = tensorflow.train.start_queue_runners(sess=sess)
    # 对变量初始化
    sess.run(tensorflow.global_variables_initializer())
    # 保存 30 张图片
    for i in range(30):
        # 每次 sess.run(reshaped_image)，都会取出一张图片
        image_array = sess.run(reshaped_image)
        # 将图片保存
        scipy.misc.toimage(image_array).save('cifar10_data/raw/%d.jpg' % i)