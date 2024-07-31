import torch
from PIL import Image
from torch import nn
from torchvision import transforms

img_path = "./imgs/img12.jpeg"

image = Image.open(img_path)


transform = transforms.Compose([transforms.Resize((32, 32)),
                                transforms.ToTensor()])


image = transform(image)
print("shape: ")
print(image.shape)
print("size: ")
print(image.size())
print("numel: ")
print(image.numel())

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


model = torch.load("test_demo99.pth", map_location=torch.device('cpu'))
print(model)
image = torch.reshape(image, (1, 3, 32, 32))
print("reshape: ")
print(image.shape)
model.eval()
with torch.no_grad():
    outputs = model(image)

print(outputs)
print(outputs.argmax(1))
