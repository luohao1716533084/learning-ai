import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10(root='./data',
                                           train=False,
                                           download=True,
                                           transform=torchvision.transforms.ToTensor()
                                           )


test_loader = DataLoader(dataset=test_data,
                         batch_size=64,
                         shuffle=True,
                         num_workers=0,
                         drop_last=False)


# 测试数据集中的第一张图片及target
img, label = test_data[0]
print("img: ", img.shape)
print("label: ", label)
print(label)

print("*" * 10)



writer = SummaryWriter(log_dir='./dataloader')
step = 0

for i in test_loader:
    imgs, labels = i
    print("imgs: ", imgs.shape)
    # print("imgs.shape: ", imgs.shape)
    # print("labels.shape: ", labels.shape)
    # print("labels", labels)
    # print("*" * 10)
    writer.add_images('test_dataloader', imgs, step)
    step += 1


# # 迭代 test_loader
# for batch_idx, (imgs, labels) in enumerate(test_loader):
#     for img_idx in range(imgs.size(0)):
#         img = imgs[img_idx]
#         writer.add_image(f'test_dataloader/batch_{batch_idx}_img_{img_idx}', img, step)
#         step += 1

writer.close()


