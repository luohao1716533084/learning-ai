import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import transforms

dataset_transforms = transforms.Compose([
    transforms.ToTensor()
])


# 会自动解压./data下 的 tar.gz文件，不需要手动进行解压
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=dataset_transforms)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=dataset_transforms)

# print(test_set[0])
#
# print('train_set.classes: ', train_set.classes)
#
# img, target = train_set[0]
#
# print(img)
# print('target: ', target)
# print(test_set.classes[target])
#
# img.show()
#
# img_1, target_1 = train_set[1]
#
# print(img_1)
# print('target_1: ', target_1)
# print(test_set.classes[target_1])
#
# img_1.show()
#


print(test_set[0])

writer = SummaryWriter(log_dir='p10')
for i in range(10):
    image, label = train_set[i]
    writer.add_image('image', image, i)

writer.close()
