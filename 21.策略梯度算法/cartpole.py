## 初始化
import numpy
import tensorflow
import gym

# gym环境
env = gym.make('CartPole-v0')

# 参数
D = 4                       # 输入层神经元个数
H = 10                      # 隐藏层神经元个数

batch_size = 5              # 一个batch中有5个episode，即5次游戏
learning_rate = 0.01        # 学习率
gamma = 0.99                # 奖励折扣率

## 定义策略网络
# 定义policy网络
# 输入观察值，输出右移的概率
observations = tensorflow.placeholder(tensorflow.float32, [None, D], name='input_x')
W1 = tensorflow.get_variable('W1', shape=[D, H], initializer=tensorflow.contrib.layers.xavier_initializer())
layer1 = tensorflow.nn.relu(tensorflow.matmul(observations, W1))
W2 = tensorflow.get_variable('W2', shape=[H, 1], initializer=tensorflow.contrib.layers.xavier_initializer())
score = tensorflow.matmul(layer1, W2)
probability = tensorflow.nn.sigmoid(score)
# 定义和训练和loss有关的变量
tvars = tensorflow.trainable_variables()
input_y = tensorflow.placeholder(tensorflow.float32, [None, 1], name='input_y')
advantages = tensorflow.placeholder(tensorflow.float32, name='reward_signal')
# 定义loss函数
loglik = tensorflow.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tensorflow.reduce_mean(loglik * advantages)
newGrads = tensorflow.gradients(loss, tvars)
# 优化器和梯度
adam = tensorflow.train.AdamOptimizer(learning_rate=learning_rate)
W1Grad = tensorflow.placeholder(tensorflow.float32, name='batch_grad1')
W2Grad = tensorflow.placeholder(tensorflow.float32, name='batch_grad2')
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))

def discounted_rewards(r):
    discounted_r = numpy.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

## 训练
xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tensorflow.global_variables_initializer()
# 开始训练
with tensorflow.Session() as sess:
    rendering = False
    sess.run(init)
    # observation是环境的初始观察量（输入神经网络的值）
    observation = env.reset()
    # gradBuffer会存储梯度，此处进行初始化
    gradBuffer = sess.run(tvars)
    for ix, grad in enumerate(gradBuffer):
        gradBuffer[ix] = grad * 0

    while episode_number <= total_episodes:
        # 当一个batch内的平均奖励达到180以上时，显示游戏窗口
        if reward_sum / batch_size > 180 or rendering is True:
            env.render()
            rendering = True
        # 输入神经网络的值
        x = numpy.reshape(observation, [1, D])
        # action = 1 表示向右移
        # action = 0 表示向左移
        # tfprob 为网络输入向右走的概率
        tfprob = sess.run(probability, feed_dict={observations: x})
        # numpy.random.uniform 为0～1之间的随机数
        # 当随机数小于 tfprob 时，采取右移策略，反之左移
        action = 1 if numpy.random.uniform() < tfprob else 0
        # xs 记录每一步的观察量，ys 记录每一步采取的策略
        xs.append(x)
        y = 1 if action == 0 else 0
        ys.append(y)
        # 执行 action
        observation, reward, done, info = env.step(action)
        reward_sum += reward
        # drs 记录每一步的 reward
        drs.append(reward)

        # 一局游戏结束
        if done:
            episode_number += 1
            # 将 xs、ys、drs 从 list 变成 numpy 数组形式
            epx = numpy.vstack(xs)
            epy = numpy.vstack(ys)
            epr = numpy.vstack(drs)
            tfp = tfps
            xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
            # 对 epr 计算期望奖励
            discounted_epr = discounted_rewards(epr)
            # 对期望奖励做归一化
            discounted_epr -= numpy.mean(discounted_epr)
            discounted_epr //= numpy.std(discounted_epr)
            # 将梯度保存到 gradBuffer 中
            tGrad = sess.run(newGrads, feed_dict={observations: epx, input_y: epy, advantages: discounted_epr})
            for ix, grad in enumerate(tGrad):
                gradBuffer[ix] += grad

            # 每玩 batch_size 局游戏，将 gradBuffer 中的梯度真正更新到 policy 网络中
            if episode_number % batch_size == 0:
                sess.run(updateGrads, feed_dict={W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
                for ix, grad in enumerate(gradBuffer):
                    gradBuffer[ix] = grad * 0
                # 打印一些信息
                print('Episode: %d ~ %d Average reward: %f.' % (episode_number - batch_size + 1, episode_number, reward_sum // batch_size))
                # 当在 batch_size 游戏中平均能拿到 200 的奖励，停止训练
                if reward_sum // batch_size >= 200:
                    print('Task solved in', episode_number, 'episodes!')
                    break
                reward_sum = 0
            observation = env.reset()
