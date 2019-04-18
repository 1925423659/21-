import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

# 读入 MNIST 数据
mnist = input_data.read_data_sets('mnist_data', one_hot=True)

# x 是占位符（placeholder），代表待识别的图片
x = tensorflow.placeholder(tensorflow.float32, [None, 784])
# w 是 softmax 模型的参数，将一个 784 维的输入转换为一个 10 维的输出
# 模型的参数用 tensorflow.Variable 表示
w = tensorflow.Variable(tensorflow.zeros([784, 10]))
# b 是 softmax 模型的参数，叫做“偏置项”（bias）
b = tensorflow.Variable(tensorflow.zeros([10]))
# y 表示模型的输出
y = tensorflow.nn.softmax(tensorflow.matmul(x, w) + b)
# y_label 是实际的图像标签，以占位符表示
y_label = tensorflow.placeholder(tensorflow.float32, [None, 10])

# 交叉熵损失
input_tensor = -tensorflow.reduce_sum(y_label * tensorflow.log(y))
cross_entropy = tensorflow.reduce_mean(input_tensor)

# 有了损失，用梯度下降法针对模型的参数（w 和 b）进行优化
train_step = tensorflow.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# 创建 session
# 只有在 session 中才能运行优化步骤 train_step
sess = tensorflow.InteractiveSession()
# 运行前必须要初始化所有变量，分配内存
tensorflow.global_variables_initializer().run()

# 进行 1000 步梯度下降
for _ in range(1000):
    # 在 mnist.train 中取 100 个训练数据
    # batch_xs 是形状为 (100, 784) 的图像数据
    # batch_ys 是形状为 (100, 10) 的实际标签
    # batch_xs, batch_ys 对应两个占位符 x 和 y
    batch_xs, batch_ys = mnist.train.next_batch(100)
    # 在 session 中运行 train_step，运行时要传入占位符的值
    sess.run(train_step, feed_dict={x: batch_xs, y_label: batch_ys})

# 正确的预测结果
correct_prediction = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_label, 1))
# 计算预测准确率
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))
# 获取最终模型的准确率
result = sess.run(accuracy, feed_dict={x: mnist.test.images, y_label: mnist.test.labels})
print(result)