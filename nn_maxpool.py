import torch
import torchvision
from torch.nn import MaxPool2d
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader


inputs = torch.tensor([[1, 2, 0, 3, 1],
                       [0, 1, 2, 3, 1],
                       [1, 2, 1, 0, 0],
                       [5, 2, 3, 1, 1],
                       [2, 1, 0, 1, 1]])

print("inputs of type ", type(inputs))
print("inputs.shape: ", inputs.shape)


inputs = torch.reshape(inputs, (-1, 1, 5, 5))

print("inputs reshape: ", inputs.shape)


"""
dataset = torchvision.datasets.CIFAR10(root='./data_nn_conv2d',
                                       train=False,
                                       download=True,
                                       transform=torchvision.transforms.ToTensor())

dataloader = DataLoader(dataset, batch_size=4)

"""


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, x):
        output = self.maxpool1(x)
        return output


net = Net()
outputs = net(inputs)
print("outputs of type ", type(outputs))

print("outputs.shape: ")
print(outputs.shape)

print('=' * 10)
print("outputs:", outputs)
print("outputs of type:", type(outputs))


""""
writer = SummaryWriter('logs/nn_maxpool')

step = 0
for data in dataloader:
    imgs, targets = data
    writer.add_images('input', imgs, step)
    output = net(imgs)
    writer.add_images("output", output, step)
    step = step + 1

"""

