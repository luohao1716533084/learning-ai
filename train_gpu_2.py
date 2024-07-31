import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.model(x)
        return x


# from model import *
train_data = torchvision.datasets.CIFAR10(root='./data',
                                          train=True,
                                          transform=torchvision.transforms.ToTensor(),
                                          download=True)


test_data = torchvision.datasets.CIFAR10(root='./data',
                                         train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=True)



train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练数据集的长度: {}".format(train_data_size))
print("训练数据集的长度: {}".format(test_data_size))


# 利用 dataloader 来加载数据集
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)


# 创建网络模型
test_demo = Net()
test_demo.to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss()

loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(test_demo.parameters(), lr=learning_rate)

# 设置续联的网络的参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epochs = 30

writer = SummaryWriter('./logs_train')
start_time = time.time()

for i in range(epochs):
    print("-------第{}轮训练开始----".format(i+1))

    # 训练
    test_demo.train()
    for data in train_dataloader:
        imgs, labels = data
        imgs, labels = imgs.to(device), labels.to(device)
        # 预测的输出
        outputs = test_demo(imgs)
        # labels 真实值
        loss = loss_fn(outputs, labels)

        # 优化器调优
        # grad清零
        optimizer.zero_grad()
        # 反向传播
        loss.backward()
        optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(end_time-start_time)
            print("训练次数:{}, loss{} ".format(total_train_step, loss.item()))
            writer.add_scalar('train_loss', loss.item(), total_train_step)

    # 测试步骤开始
    test_demo.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs_no, labels_no = data
            imgs_no, labels_no = imgs_no.to(device), labels_no.to(device)
            outputs_no = test_demo(imgs_no)
            loss_no = loss_fn(outputs_no, labels_no)
            total_test_loss += loss_no.item()
            accuracy = (outputs_no.argmax(1) == labels_no).sum()
            total_accuracy += accuracy
    print("整体测试集上的loss：{}".format(total_test_loss))
    print("整体正确率:{}".format(total_accuracy/test_data_size))
    writer.add_scalar('test_loss', total_test_loss, total_test_step)
    writer.add_scalar('test_accuracy', total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(test_demo, 'test_demo{}.pth'.format(i))
    print("模型已保存")
writer.close()


