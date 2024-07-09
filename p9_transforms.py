from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image


# img_path = 'train/ants_image/0013035.jpg'
img_path = 'train/ants_image/6240329_72c01e663e.jpg'
img = Image.open(img_path)

writer = SummaryWriter(log_dir='logs')

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# print(tensor_img)

writer.add_image("tensor", tensor_img, 1)

writer.close()



