import torch
import torch.nn as nn

#残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #仅第一层卷积stride = 2,另图片大小减半
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        # 批归一化
        self.bn1 = nn.BatchNorm2d(out_channel)
        #激活函数 ReLU
        # inplace对从上层网络nn.Conv2d中传递下来的tensor直接进行修改，这样能够节省运算内存
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=out_channel,
                               out_channels=out_channel,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_classes=10):
        super(ResNet, self).__init__()
        # 控制padding stride 让输入图片大小减半
        # 224 to 112
        self.in_channel = 64
        self.conv1 = nn.Conv2d(in_channels=3,
                               out_channels=64,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        #112 to 56
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #maxpool已经减半了
        self.layer1 = self.make_layer(block, 64)
        self.layer2 = self.make_layer(block, 128, stride=2)
        self.layer3 = self.make_layer(block, 256, stride=2)
        self.layer4 = self.make_layer(block, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        #初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, channel, stride=1):
        downsample = None
        #不是第一层时要加入下采样项
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel,
                          channel,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(channel))
        bk1 = block(self.in_channel,
                    channel,
                    downsample= downsample,
                    stride=stride)
        self.in_channel = channel
        bk2 = block(self.in_channel, channel)
        bk = nn.Sequential(bk1,bk2)
        return bk

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x



if __name__ == '__main__':
    net = ResNet()
    print(net)
