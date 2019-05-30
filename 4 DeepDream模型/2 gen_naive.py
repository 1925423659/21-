import tensorflow
import numpy
import scipy.misc

# 创建图和会话
graph = tensorflow.Graph()
sess = tensorflow.InteractiveSession(graph=graph)

# https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
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

def render_naive(t_obj, image, iter_n=20, step=1):
    # t_score 是优化目标，是 t_obj 的平均值
    # 结合调用处看，是 layer_output[:, :, :, channel] 的平均值
    t_score = tensorflow.reduce_mean(t_obj)
    # 计算 t_score 对 t_input 的梯度
    t_grad = tensorflow.gradients(t_score, t_input)[0]

    # 创建新图
    image_copy = image.copy()
    for i in range(iter_n):
        # 在 sess 中计算梯度，及当前的 score
        g, score = sess.run([t_grad, t_score], {t_input: image_copy})
        # 对应 image 应用梯度
        # step 可以看作“学习率”
        g /= g.std() + 1e-8
        image_copy += g * step
        print('score(mean)=%f' % score)
    
    # 保存图片
    scipy.misc.toimage(image_copy, 'naive.jpg')
    print('image saved: naive.jpg')

# 定义卷积层、通道数，取出对应的 Tensor
name = 'mixed4d_3x3_bottleneck_pre_relu'
channel = 139
layer_output = graph.get_tensor_by_name('import/%s:0' % name)

# 定义原始的图像噪声
image_noise = numpy.random.uniform(size=(224, 224, 3)) + 100
# 调用 render_naive 函数渲染
render_naive(layer_output[:, :, :, channel], image_noise, iter_n=20)
