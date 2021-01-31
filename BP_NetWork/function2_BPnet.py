import torch
import torch.nn as nn
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

neurons = 30
# 搭建网络
BPNet = nn.Sequential(
    nn.Linear(1, neurons),
    nn.Tanh(),
    nn.Linear(neurons, 1),
).to(device)
# 设置优化器
optimzer = torch.optim.SGD(BPNet.parameters(),lr=0.1)
loss_func = nn.MSELoss()
#数据
X = np.linspace(-math.pi/2,math.pi/2,50)
X = np.reshape(X,(50,1))
Y = 1/np.sin(X) + 1/np.cos(X)
for i in range(len(Y)):
    if(abs(Y[i])>100):
        Y[i] = 10
x = torch.tensor(X).float().to(device)
y = torch.tensor(Y).float().to(device)
losses = []
optimzer.zero_grad()  # 清除梯度
for epoch in range(20001):
    out = BPNet(x)
    loss = loss_func(out, y)  # 计算误差
    optimzer.zero_grad()  # 清除梯度
    loss.backward()
    optimzer.step()
    if(epoch%1000 == 0):
        print(epoch/1000,loss)
        losses.append(loss.to("cpu").detach().numpy())
# 测试结果
yTest = BPNet(x).to("cpu").detach().numpy()
print(yTest)
plt.figure()
plt.plot(X, yTest, color='green')
plt.title('Curve')
plt.plot(X, Y,color = "red")
plt.figure()
plt.plot(np.arange(0, len(losses)), losses)
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


