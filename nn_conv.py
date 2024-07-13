import torch
import torch.nn.functional as F


inputs = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]])

kernel = torch.tensor([[1, 2, 1],
                       [0, 1, 0],
                       [2, 1, 0]])


inputs = torch.reshape(inputs, (1, 1, 5, 5))
kernel = torch.reshape(kernel, (1, 1, 3, 3))

print(inputs.shape)
print("inputs: ", inputs)

print(kernel.shape)
print("kernel: ", kernel)

outputs_1 = F.conv2d(inputs, kernel, stride=1)
print("*" * 10)
print(outputs_1.shape)
print(outputs_1)

outputs_2 = F.conv2d(inputs, kernel, stride=2)
print("*" * 10)
print(outputs_2.shape)
print(outputs_2)

outputs_3 = F.conv2d(inputs, kernel, stride=1, padding=1)
print("*" * 10)
print(outputs_3.shape)
print(outputs_3)


