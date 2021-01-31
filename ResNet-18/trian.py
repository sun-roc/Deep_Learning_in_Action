import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
from ResNet import ResNet
import matplotlib.pyplot as plt

# 训练设备选择
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = "train_out.txt"

def train():
    batch_size = 100
    # 训练集
    cifar_train = datasets.CIFAR10(
        root='cifar',
        train=True,
        transform=transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]))
    cifar_train = DataLoader(cifar_train,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

    # 测试集
    cifar_test = datasets.CIFAR10(
        root='cifar',
        train=False,
        transform=transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ]))
    cifar_test = DataLoader(cifar_test,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0)

    # 定义模型-ResNet
    model = ResNet()
    model = torch.load("net.pth")
    model.to(device)

    # 定义损失函数和优化方式
    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), 0.001)

    # 训练网络
    for epoch in range(5):
        f = open(path , "a+")
        model.train()  # 训练模式
        loss_sum = []
        for batchidx, (data, label) in enumerate(cifar_train):

            data = data.to(device)
            label = label.to(device)
            predict = model(data)
            loss = criteon(predict, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum.append(loss.item())
            if (batchidx % 20 == 0):
                print(batchidx, loss.item())

                
        print("Epoch_num: ", epoch, ' training-mean-loss:', np.mean(loss_sum),file = f)

        model.eval()  # 测试模式
        with torch.no_grad():

            total_correct = 0  # 预测正确的个数
            total_num = 0
            for data, label in cifar_test:
                data = data.to(device)
                label = label.to(device)
                predict = model(data)

                pred = predict.argmax(dim=1)
                correct = torch.eq(pred, label).float().sum().item()
                total_correct += correct
                total_num += data.size(0)
            acc = total_correct / total_num
            print("Epoch_num: ", epoch, 'test_acc:', acc,file = f)
            torch.save(model, "net.pth") 
            f.close()

if __name__ == '__main__':
    train()
