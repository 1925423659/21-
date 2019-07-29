import tensorflow
import numpy
import scipy.misc
from functools import partial

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

def resize_ratio(image, ratio):
    min = image.min()
    max = image.max()
    image = (image - min) / (max - min) * 255
    image = numpy.float32(scipy.misc.imresize(image, ratio))
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

k = numpy.float32([1, 4, 6, 4, 1])
# 相乘得到 5*5 的矩阵
k = numpy.outer(k, k)
k5x5 = k[:, :, None, None] / k.sum() * numpy.eye(3, dtype=numpy.float32)

# 将图像分为低频和高频成分
def lap_split(image):
    with tensorflow.name_scope('split'):
        # 做过一次卷积相当于一次“平滑”，因此 lo 为低频成分
        lo = tensorflow.nn.conv2d(image, k5x5, [1, 2, 2, 1], 'SAME')
        # 低频成分缩放到原始图像一样大小得到 lo_
        # 再用原始图像 image 减去 lo_，得到高频成分 hi
        lo_ = tensorflow.nn.conv2d_transpose(lo, k5x5 * 4, tensorflow.shape(image), [1, 2, 2, 1])
        hi = image - lo_
    return lo, hi

# 将图像分成 n 层拉普拉斯金字塔
def lap_split_n(image, n):
    levels = []
    for i in range(n):
        # 调用 lap_split 将图像分为低频和高频部分
        # 高频部分保存到 levels 中
        # 低频部分再继续分解
        image, hi = lap_split(image)
        levels.append(hi)
    levels.append(image)
    # 倒序
    return levels[::-1]

# 将拉普拉斯金字塔还原到原始图像
def lap_merge(levels):
    image = levels[0]
    for hi in levels[1:]:
        with tensorflow.name_scope('merge'):
            image = tensorflow.nn.conv2d_transpose(image, k5x5 * 4, tensorflow.shape(hi), [1, 2, 2, 1]) + hi
    return image

# 对 image 做标准化
def normalize_std(image, eps=1e-10):
    with tensorflow.name_scope('normalize'):
        square = tensorflow.square(image)
        mean = tensorflow.reduce_mean(square)
        std = tensorflow.sqrt(mean)
        return image / tensorflow.maximum(std, eps)

# 拉普拉斯金字塔标准化
def lap_normalize(image, scale_n=4):
    image = tensorflow.expand_dims(image, 0)
    levels = lap_split_n(image, scale_n)
    # 每一层都做一次 normalize_std
    levels = list(map(normalize_std, levels))
    out = lap_merge(levels)
    return out[0, :, :, :]

def tffunc(*argtypes):
    placeholders = list(map(tensorflow.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

def render_lapnorm(t_obj, image, iter_n=10, step=1, octave_n=3, octave_scale=1.4, lap_n=4):
    # 同样定义目标和梯度
    t_score = tensorflow.reduce_mean(t_obj)
    t_grad = tensorflow.gradients(t_score, t_input)[0]
    # 将 lap_normalize 转换为正常函数
    lap_norm_func = tffunc(numpy.float32)(partial(lap_normalize, scale_n=lap_n))

    image_copy = image.copy()
    for octave in range(octave_n):
        if octave > 0:
            image_copy = resize_ratio(image_copy, octave_scale)
        for i in range(iter_n):
            g = calc_grad_tiled(image_copy, t_grad)
            # 唯一的区别在于使用 lap_norm_func 将 g 标准化
            g = lap_norm_func(g)
            image_copy += g * step
            print('.', end=' ')
    # 保存图片
    scipy.misc.toimage(image_copy).save('dataset/generate/lapnorm.jpg')
    print('image saved: lapnorm.jpg')

name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
image_noise = numpy.random.uniform(size=(224, 224, 3)) + 100
layer_output = graph.get_tensor_by_name('import/%s:0' % name)
render_lapnorm(layer_output[:, :, :, channel], image_noise, iter_n=20)