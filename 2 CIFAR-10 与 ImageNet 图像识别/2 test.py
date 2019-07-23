import tensorflow
import os

if not os.path.exists('dataset/test/read'):
    os.makedirs('dataset/test/read')

# 新建 session
with tensorflow.Session() as sess:
    # 读 3 张图片 A.jpg、B.jpg、C.jpg
    filenames = ['dataset/test/image/A.jpg', 'dataset/test/image/B.jpg', 'dataset/test/image/C.jpg']
    # string_input_producer 会产生一个文件名队列
    filename_queue = tensorflow.train.string_input_producer(filenames, shuffle=False, num_epochs=5)
    # reader 从文件名队列中读数据
    # 对应的方法是 reader.read
    reader = tensorflow.WholeFileReader()
    key, value = reader.read(filename_queue)
    # tensorflow.train.string_input_producer 定义了一个 epoch 变量
    # 需进行初始化
    tensorflow.local_variables_initializer().run()
    # 使用 start_queue_runners 之后，才会开始填充队列
    threads = tensorflow.train.start_queue_runners(sess=sess)
    i = 0
    while True:
        i += 1
        # 获取图片数据并保存
        image = sess.run(value)
        with open('dataset/test/read/test_%d.jpg' % i, 'wb') as file:
            file.write(image)