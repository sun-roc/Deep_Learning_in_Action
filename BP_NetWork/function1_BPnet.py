import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# 神经网络结构
class BPNet:
    # 初始化，构造函数
    def __init__(self, layers, learning_rate=0.1,act = "sigmoid"):
        self.W = [] 
        self.layers = layers
        self.learning_rate = learning_rate # 学习率
        self.act = act #激活函数
        # 初始化Weight
        # 这里-2 表示除了最后由隐藏层到输出层的权值
        for i in np.arange(0, len(layers) - 2): 
            # 输入层到隐藏层及隐藏层内部的Weight
            # 这里+1是表示这些层中都添加了偏置项bias所以输入的维度会+1改变
            weight = np.random.rand(layers[i] + 1, layers[i + 1] + 1) 
            self.W.append(weight)
        #添加最后一层的权重,由隐藏层到输出层
        self.W.append(np.random.rand(layers[-2] + 1, layers[-1]))
    #激活函数
    def act_func(self,x):
        if (self.act == "sigmoid"):
            return 1.0 / (1 + np.exp(-x))
        elif(self.act == "tanh"):
            return np.tanh(x)
    #激活函数的导数
    def act_derivative(self,x):
        if (self.act == "sigmoid"):
            return x * (1 - x)
        elif(self.act == "tanh"):
            return 1 - np.tanh(x) * np.tanh(x)  # tanh函数的导数
    # 均方误差函数
    def MSE_loss(self, inputs, targetValue):
        targetValue = np.atleast_2d(targetValue)
        predictions = self.test(inputs)
        loss = 0.5 * np.sum((predictions - targetValue) ** 2)
        return loss
    # 链式求导,反向传播
    def back_propagation(self, x, y):
        # np.atleast_2d()函数用于将输入视为至少具有两个维度的数组
        forward_list = [np.atleast_2d(x)] 
        # 计算这个w矩阵下整个神经网络的输出
        for layer in np.arange(0, len(self.W)): #２
            net = forward_list[layer].dot(self.W[layer])
            out = self.act_func(net)
            forward_list.append(out)
        error = forward_list[-1] - y 
        D = [error * self.act_derivative(forward_list[-1])]
        for layer in np.arange(len(forward_list)- 2, 0, -1):
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.act_derivative(forward_list[layer])
            D.append(delta)
        D = D[::-1] 
        # 更新权值W
        for layer in np.arange(0, len(self.W)):
            self.W[layer] += -self.learning_rate * forward_list[layer].T.dot(D[layer])
        # 训练方法 epochs迭代次数
    def train(self, inputs, y, epochs=100000):
        #表示对输入数据添加偏置项,np.c_按行连接数组
        inputs = np.c_[inputs, np.ones((inputs.shape[0]))] 
        losses = []
        time = 0
        # 根据每一层网络进行反向传播，然后更新W
        for epoch in np.arange(0, epochs):
        # while (True):
            for (x, target) in zip(inputs, y):
                self.back_propagation(x, target) # 更新weights
            # 显示训练结果,计算误差
            if epoch == 0 or (epoch + 1) % 1000 == 0:
                loss = self.MSE_loss(inputs, y)
                losses.append(loss)
                print("epoch={}, loss={:.3f}".format(epoch + 1, loss))
            if (loss<=0.005):
                time = epoch
                break
        return losses,time
    # 测试
    def test(self, inputs):
        predict = np.atleast_2d(inputs)
        if(len(predict[-1])<len(self.W[0])):
            predict = np.c_[predict, np.ones((predict.shape[0]))]
        # 正常的前向传播得到预测的输出值
        for layer in np.arange(0, len(self.W)):
            predict = self.act_func(np.dot(predict, self.W[layer])) 
        return predict
data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])
net = BPNet([2, 2, 1], learning_rate=0.5,act = "sigmoid")
losses,time = net.train(data, y)
# 测试并输出结果
for (x, target) in zip(data, y):
    outputs = net.test(x)[0][0]
    label = 1 if outputs > 0.5 else 0
    print("data={}, 实际值={}, 预测值大小={:.2f}, 预测判别为={}"
          .format(x, target[0], outputs, label))
print("训练权重\n", net.W)
plt.figure()
plt.plot(np.arange(0, len(losses)), losses)
plt.xlabel("epoch")
plt.ylabel("loss")
plt.show()


