import torch
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np


writer = SummaryWriter('logs')
img_path = "train/ants_image/0013035.jpg"
img_PIL = Image.open(img_path)
img_array = np.array(img_PIL)


writer.add_image("test_image", img_array, 1, dataformats="HWC")

for i in range(100):
    writer.add_scalar('y=3x', 4*i, i)

writer.close()
