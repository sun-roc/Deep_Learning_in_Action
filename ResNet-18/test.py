import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch import nn, optim
import numpy as np
from ResNet import ResNet
import matplotlib.pyplot as plt
import torchvision.models as models

from PIL import Image
batch_size = 500
name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# model = ResNet()
# model = torch.load("net.pth")
transform=transforms.Compose([
            transforms.Resize((112,112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
for  i in range(2):
    im = Image.open('{}.jpg'.format(i))
    plt.imshow(im)
    plt.show()
    im = transform(im)
    im=im.type(torch.cuda.FloatTensor)
    im = torch.reshape(im,(1,3,112,112))
    predict = model(im)
    pred = predict.argmax(dim=1)
    print(name[pred])
