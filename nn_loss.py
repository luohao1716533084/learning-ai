
import torch
from torch import nn

from torch.nn import L1Loss

inputs = torch.tensor([1,2,3], dtype=torch.float32)
targets = torch.tensor([1,2,5], dtype=torch.float32)

inputs = torch.reshape(inputs, (1, 1, 1, 3))
targets = torch.reshape(targets, (1, 1, 1, 3))

# loss = L1Loss()
loss = L1Loss(reduction='sum')
result = loss(inputs, targets)

print(result)


from torch import nn

x = torch.tensor([0.1, 0.2, 0.3])
y = torch.tensor([1])


# 这里我们将 x 重新调整为二维张量 (1, 3)，表示有一个样本，每个样本有三个类别的得分。
# 这是因为 CrossEntropyLoss 期望输入的 logits 是一个二维张量，形状为 (N, C)，其中 N 是样本数量，C 是类别数量。
x = torch.reshape(x, (1, 3))
print(x.shape)
print(x)
loss_cross = nn.CrossEntropyLoss()
result_loss = loss_cross(x, y)
print(result_loss)
