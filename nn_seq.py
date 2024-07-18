import torch
from torch import nn

from torch.nn import Conv2d, Sequential, MaxPool2d, Linear
from torch.utils.tensorboard import SummaryWriter


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.conv1 = Conv2d(3, 32, kernel_size=3,  padding=2)
        # self.maxpool1 = nn.MaxPool2d(2)
        # self.conv2 = Conv2d(32, 32, kernel_size=5, padding=2)
        # self.maxpool2 = nn.MaxPool2d(2)
        # self.conv3 = Conv2d(32, 64, kernel_size=5, padding=2)
        # self.maxpool3 = nn.MaxPool2d(2)
        # self.flatten = nn.Flatten()
        # self.linear1 = nn.Linear(64 * 4 * 4, 64)
        # self.linear2 = nn.Linear(64, 10)

        self.model1 = Sequential(
            Conv2d(3, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, kernel_size=5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, kernel_size=5, padding=2),
            MaxPool2d(2),
            nn.Flatten(),
            Linear(64 * 4 * 4, 64),
            Linear(64, 10)

        )

    def forward(self, x):
        # x = self.conv1(x)
        # x = self.maxpool1(x)
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        # x = self.flatten(x)
        # x = self.linear1(x)
        # x = self.linear2(x)
        x = self.model1(x)
        return x


test_demo = Net()
print(test_demo)

inputs = torch.ones((64,3,32,32))
print(inputs)

outputs = test_demo(inputs)
print(outputs)
print(outputs.shape)

writer = SummaryWriter(log_dir='./logs_seq')
writer.add_graph(test_demo, inputs)
writer.close()
