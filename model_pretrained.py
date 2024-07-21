
import torch
import torchvision.models as models
import torchvision
from torch import nn

dataset = torchvision.datasets.CIFAR10(root='./data',
                                       train=True,
                                       transform=torchvision.transforms.ToTensor(),
                                       download=True)

# 加载预训练的VGG16模型
# vgg16 = models.get_model('vgg16', weights=models.VGG16_Weights.DEFAULT)

vgg16_false = models.get_model('vgg16', weights=models.VGG16_Weights.DEFAULT)
vgg16_true = models.get_model('vgg16', weights=models.VGG16_Weights.DEFAULT)

# 修改模型
# 为classifier 添加 linear()
vgg16_true.classifier.add_module('(7)', nn.Linear(1000, 10))
print(vgg16_true)