from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


writer = SummaryWriter('logs')

img = Image.open('train/ants_image/45472593_bfd624f8dc.jpg')

print(img)

trans_to_tensor = transforms.ToTensor()
img_tensor = trans_to_tensor(img)

writer.add_image("toTensor", img_tensor)
trans_norm = transforms.Normalize([0.485, 0.456, 0.406], [0.5, 0.5, 0.5])

img_norm = trans_norm(img_tensor)

writer.add_image("norm", img_norm)

print(img.size)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)

img_resize = trans_to_tensor(img_resize)
writer.add_image("resize", img_resize, 0)

trans_resize_2 = transforms.Resize(512)

trans_compose = transforms.Compose([trans_resize_2, trans_to_tensor])
img_resize_2 = trans_compose(img)
writer.add_image("compose", img_resize_2, 1)

writer.close()

