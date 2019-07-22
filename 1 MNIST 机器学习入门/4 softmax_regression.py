import tensorflow
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

# 读入 MNIST 数据
mnist = read_data_sets('dataset/mnist', one_hot=True)

# x 是占位符（placeholder），代表待识别的图片
x = tensorflow.placeholder(tensorflow.float32, [None, 784])
# y 是实际的图像标签，同样以占位符表示
y = tensorflow.placeholder(tensorflow.float32, [None, 10])
# w 是 softmax 模型的参数，将一个 784 维的输入转换为一个 10 维的输出
# 模型的参数用 tensorflow.Variable 表示
w = tensorflow.Variable(tensorflow.zeros([784, 10]))
# b 是 softmax 模型的参数，叫做“偏置项”（bias）
b = tensorflow.Variable(tensorflow.zeros([10]))
# y_ 表示模型的输出
y_ = tensorflow.nn.softmax(tensorflow.matmul(x, w) + b)

#### 训练
# 根据 y 和 y_ 构造交叉熵损失
sum = -tensorflow.reduce_sum(y * tensorflow.log(y_))
cross_entropy = tensorflow.reduce_mean(sum)

# 有了损失，用梯度下降法针对模型的参数（w 和 b）进行优化
train_step = tensorflow.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#### 测试
# 正确的预测结果
predict = tensorflow.equal(tensorflow.argmax(y_, 1), tensorflow.argmax(y, 1))
# 计算预测准确率
cast = tensorflow.cast(predict, tensorflow.float32)
accuracy = tensorflow.reduce_mean(cast)

# 创建 session
# 只有在 session 中才能运行优化步骤 train_step
session = tensorflow.InteractiveSession()
# 运行前必须要初始化所有变量，分配内存
session.run(tensorflow.global_variables_initializer())

# 进行 1000 步梯度下降
for _ in range(1000):
    # 在 mnist.train 中取 100 个训练数据
    # batch_xs 是形状为 (100, 784) 的图像数据
    # batch_ys 是形状为 (100, 10) 的实际标签
    # batch_xs, batch_ys 对应两个占位符 x 和 y_
    batch_x, batch_y = mnist.train.next_batch(100)
    # 在 session 中运行 train_step，运行时要传入占位符的值
    session.run(train_step, feed_dict={x: batch_x, y: batch_y})

# 获取最终模型的准确率
result = session.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
print(result)