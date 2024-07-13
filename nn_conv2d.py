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

class test_demo(nn.Module):

    def __init__(self, *args, **kwargs) -> None:
        super(test_demo, self).__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(3, 6, 3,1,0)


    def forward(self, x):
        x = self.conv1(x)
        return x


test_demo = test_demo()
print(test_demo)

writer = SummaryWriter('./nn_conv2d')

step = 10
for data in dataloader:
    imgs, targets = data
    outputs = test_demo(imgs)

    writer.add_images('imgs', imgs, step)

    outputs = torch.reshape(outputs, (-1,3,30,30))

    writer.add_images('outputs', outputs, step)

    step += 1

