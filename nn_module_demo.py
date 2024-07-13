import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        outputs = inputs + 1
        return outputs


test_model = NeuralNetwork()
x = torch.tensor(1.0)
outputs = test_model(x)
print(outputs)

