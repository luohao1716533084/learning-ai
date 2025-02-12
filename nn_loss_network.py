import torch
import torchvision
from torch import nn

from torch.nn import Conv2d, Sequential, MaxPool2d, Linear
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


dataset = torchvision.datasets.CIFAR10(root='./data',
                                       train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

dataloader = DataLoader(dataset, batch_size=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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
        x = self.model1(x)
        return x


loss = nn.CrossEntropyLoss()
test_demo = Net()
optimizer = torch.optim.SGD(test_demo.parameters(), lr=0.001)

for epoch in range(20):
    running_loss = 0.0
    for data in dataloader:
        imgs, targets = data
        outputs = test_demo(imgs)
        res = loss(outputs, targets)
        # print("targets: ", targets)
        # 重置梯度
        optimizer.zero_grad()
        # 获取梯度 grad
        res.backward()
        optimizer.step()
        # print("ok")
        running_loss = running_loss + res
    print(running_loss)




# writer = SummaryWriter(log_dir='./logs_seq')
# writer.add_graph(test_demo, inputs)
# writer.close()
