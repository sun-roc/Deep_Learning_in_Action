import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch import optim
import time
# 训练设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# batch的大小
batch_size = 100

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
            # 卷积层C1
            nn.Conv2d(in_channels=1, out_channels=6,
                      kernel_size=5, padding=2),  # 32,32 to 28*28
            nn.Sigmoid(),
            # 池化层S2
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28*28 to 14*14
            # 卷积层C3
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),  # 10*10
            nn.Sigmoid(),
            # 池化层S4
            nn.MaxPool2d(kernel_size=2, stride=2),  # 10*10 to 5*5
        )
        self.fullconnect = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 10)
        )

    def forward(self, inputs):
        output = self.conv(inputs)
        # 将高维向量flatten
        output = self.fullconnect(output.view(inputs.shape[0], -1))
        return output


# 实例化
# net = LeNet()
#加载之前训练好的模型
net = torch.load("net.pth")
net.to(device)
# 打印模型
print(net)
# 对输入变量的操作
transform = transforms.Compose([
    transforms.ToTensor(),  # 转为Tensor
    transforms.Normalize((0.5,), (0.5,)),  # 归一化
])
# 返回值为mnist类
train_dataset = torchvision.datasets.MNIST(
    root='./mnist', train=True, transform=transform, download=False)
test_dataset = torchvision.datasets.MNIST(
    root='./mnist', train=False, transform=transform, download=False)

# 数据加载器,返回list[0]为100,1,28,28 list[1] 100
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

def main():
    # 交叉熵损失函数
    criterion = nn.CrossEntropyLoss()  
    # 优化器
    optimizer = optim.Adam(net.parameters(), lr=0.01)  
    running_loss = 0.0  # 初始化loss
    loss_list = []
    for epoch in range(5):
        start = time.time()
        running_loss = 0.0 
        for i, (inputs, labels) in enumerate(trainloader, 0):
            net.train()
            # 输入数据
            inputs = inputs.to(device)
            labels = labels.to(device)
            # 梯度清零
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # 更新参数
            optimizer.step()
            running_loss += loss.item()
        loss_list.append(loss.item())
        print("loss:",running_loss,"\n")
        
        net.eval()
        with torch.no_grad():
            # 测试集不需要计算梯度，放在torch.no_grad()节省计算资源
            # 总共正确的数量
            total_correct = 0
            #总共的数量
            total_num = 0
            for inputs, labels in trainloader:
                # 输入数据
                inputs = inputs.to(device)
                labels = labels.to(device)
                #输出数据
                outputs = net(inputs)
                #输出的数据选取概率最大的值
                pred = outputs.argmax(dim=1)
                #eq函数比较是否相等返回相等的和
                total_correct += torch.eq(labels, pred).float().sum()
                total_num += inputs.size(0)
            acc = total_correct/total_num
            print('测试集正确率为 :', (acc*100),"\n")
        end = time.time()
        print("训练时间:",end-start,"\n")
main()
#保存模型
torch.save(net, "net.pth") 