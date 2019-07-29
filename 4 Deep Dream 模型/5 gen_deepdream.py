

import tensorflow
import numpy
import scipy.misc
import PIL.Image

# 创建图和会话
graph = tensorflow.Graph()
sess = tensorflow.InteractiveSession(graph=graph)

# tensorflow_inception_graph.pb 文件中
# 存储了 inception 的网络结构和对应的数据
# 导入
with tensorflow.gfile.FastGFile('dataset/graph/tensorflow_inception_graph.pb', mode='rb') as f:
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

def resize(image, hw):
    min = image.min()
    max = image.max()
    image = (image - min) / (max - min) * 255
    image = numpy.float32(scipy.misc.imresize(image, hw))
    image = image / 255 * (max - min) + min
    return image

def calc_grad_tiled(image, t_grad, tile_size=512):
    height, width = image.shape[:2]
    sx, sy = numpy.random.randint(tile_size, size=2)
    # 先在行上做整体移动，再在列上做整体移动
    image_shift = numpy.roll(numpy.roll(image, sx, 1), sy, 0)
    grad = numpy.zeros_like(image)
    for y in range(0, max(height - tile_size // 2, tile_size), tile_size):
        for x in range(0, max(width - tile_size // 2, tile_size), tile_size):
            sub = image_shift[y:y + tile_size, x:x + tile_size]
            g = sess.run(t_grad, {t_input: sub})
            grad[y:y + tile_size, x:x + tile_size] = g
    return numpy.roll(numpy.roll(grad, -sx, 1), -sy, 0)

def render_deepdream(t_obj, image, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
    t_score = tensorflow.reduce_mean(t_obj)
    t_grad = tensorflow.gradients(t_score, t_input)[0]

    image_copy = image.copy()
    # 同样将图像进行金字塔分解
    # 此时提取高频、低频的方法比较简单
    # 直接缩放就可以
    octaves = []
    for i in range(octave_n - 1):
        hw = image_copy.shape[:2]
        lo = resize(image_copy, numpy.int32(numpy.float32(hw) / octave_scale))
        hi = image_copy - resize(lo, hw)
        image_copy = lo
        octaves.append(hi)
    
    # 先生成低频的图像，再依次放大并加上高频
    for octave in range(octave_n):
        if octave > 0:
            hi = octaves[-octave]
            image_copy = resize(image_copy, hi.shape[:2]) + hi
        for i in range(iter_n):
            g = calc_grad_tiled(image_copy, t_grad)
            image_copy += g * (step / (numpy.abs(g).mean() + 1e-7))
            print('.', end=' ')
    
    image_copy = image_copy.clip(0, 255)
    scipy.misc.toimage(image_copy).save('dataset/generate/deepdream.jpg')

image = PIL.Image.open('dataset/image/test.jpg')
image = numpy.float32(image)

# name = 'mixed4d_3x3_bottleneck_pre_relu'
# channel = 139
# layer_output = graph.get_tensor_by_name('import/%s:0' % name)
# render_deepdream(layer_output[:, :, :, channel], image)

name = 'mixed4c'
layer_output = graph.get_tensor_by_name('import/%s:0' % name)
render_deepdream(tensorflow.square(layer_output), image)