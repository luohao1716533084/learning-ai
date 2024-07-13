import torch

import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(root='./data_nn_conv2d',
                                       train=False,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset,batch_size=64)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(196608, 10)

    def forward(self, x):
        out = self.linear(x)
        return out

test_demo = Net()


for data in dataloader:
    imgs, targets = data
    print("imgs.shape: ", imgs.shape)
    outputs = torch.reshape(imgs,(1,1,1,-1))
    print("outputs.shape: ", outputs.shape)
    outputs = test_demo(outputs)

    print("linear: ")
    print(outputs.shape)
    print("=" * 10)
