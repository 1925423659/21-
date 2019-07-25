import tensorflow
import numpy

# 创建图和会话
graph = tensorflow.Graph()
sess = tensorflow.InteractiveSession(graph=graph)

# https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
# tensorflow_inception_graph.pb 文件中
# 存储了 inception 的网络结构和对应的数据
# 导入
with tensorflow.gfile.FastGFile('dataset/graph/tensorflow_inception_graph.pb', 'rb') as f:
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

# 找到所有卷积层
layers = [op.name for op in graph.get_operations() if op.type == 'Conv2D' and 'import/' in op.name]

# 输出卷积层层数
print('Number of layers', len(layers))
print(layers)

# 输出 mixed4d_3x3_bottleneck_pre_relu 的形状
name = 'mixed4d_3x3_bottleneck_pre_relu'
print('shape of %s: %s' % (name, str(graph.get_tensor_by_name('import/' + name + ':0').get_shape())))