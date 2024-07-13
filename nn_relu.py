import torch

inputs = torch.tensor([[1, 0.5],
                       [-1, 3]])

print(inputs)

outputs = torch.reshape(inputs, (-1, 1, 2, 2))

print(outputs.shape)
print(outputs)