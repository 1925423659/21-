import tensorflow
import numpy
import scipy.misc

# 创建图和会话
graph = tensorflow.Graph()
sess = tensorflow.InteractiveSession(graph=graph)

# tensorflow_inception_graph.pb 文件中
# 存储了 inception 的网络结构和对应的数据
# 导入
model_fn = 'tensorflow_inception_graph.pb'
with tensorflow.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tensorflow.GraphDef()
    graph_def.ParseFromString(f.read())

# 定义输入的图像
t_input = tensorflow.placeholder(numpy.float32, name='input')
imagenet_mean = 117

# 输入图像需要经过处理才能送入网络中
# expand_dims 是加一维
# 从 [height, width, channel] 变成 [1, height, width, channel]
# t_input - imagenet_mean 是减去一个均值
t_preprocessed = tensorflow.expand_dims(t_input - imagenet_mean, 0)
tensorflow.import_graph_def(graph_def, {'input': t_preprocessed})

def resize_ratio(image, ratio):
    min = image.min()
    max = image.max()
    image = (image - min) / (max - min) * 255
    image = numpy.float32(scipy.misc.imresize(image, ratio))
    image = image / 255 * (max - min) + min
    return image

def calc_grad_tiled(image, t_grad, tile_size=512):
    # 每次只对 tile_size * tile_size 大小的图像计算梯度，避免内存问题
    height, width = image.shape[:2]
    # image_shift：先在行上做整体移动，再在列上做整体移动
    # 防止在 tile 的边缘产生边缘效应
    sx, sy = numpy.random.randint(tile_size, size=2)
    image_shift = numpy.roll(numpy.roll(image, sx, 1), sy, 0)
    grad = numpy.zeros_like(image)
    # y，x 是开始位置的像素
    for y in range(0, max(height - tile_size // 2, tile_size), tile_size):
        for x in range(0, max(width - tile_size // 2, tile_size), tile_size):
            # 每次对 sub 计算梯度。sub 的大小是 tile_size * tile_size
            sub = image_shift[y:y + tile_size, x:x + tile_size]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + tile_size, x:x + tile_size] = g
    # 使用 numpy.roll 移动回去
    return numpy.roll(numpy.roll(grad, -sx, 1), -sy, 0)

def render_multiscale(t_obj, image, iter_n=10, step=1, octave_n=3, octave_scale=1.4):
    # t_score 是优化目标，是 t_obj 的平均值
    # 结合调用处看，是 layer_output[:, :, :, channel] 的平均值
    t_score = tensorflow.reduce_mean(t_obj)
    # 计算 t_score 对 t_input 的梯度
    t_grad = tensorflow.gradients(t_score, t_input)[0]

    image_copy = image.copy()
    for octave in range(octave_n):
        if octave > 0:
            # 每次将图片放大 octave_scale 倍
            # 共放大 octave_n - 1 次
            image_copy = resize_ratio(image_copy, octave_scale)
        for i in range(iter_n):
            # 调用 calc_grad_tiled 计算任意大小图像的梯度
            g = calc_grad_tiled(image_copy, t_grad)
            g /= g.std() + 1e-8
            image_copy += g * step
            print('.', end=' ')

    # 保存图片
    scipy.misc.toimage(image_copy).save('multiscale.jpg')
    print('image saved: multiscale.jpg')

name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
image_noise = numpy.random.uniform(size=(224, 224, 3)) + 100
layer_output = graph.get_tensor_by_name('import/%s:0' % name)
render_multiscale(layer_output[:, :, :, channel], image_noise, iter_n=20)