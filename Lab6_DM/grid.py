import torch
from torchvision.utils import make_grid, save_image
import os
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# base_dir = "./images_ep500"
base_dir = "./images_ep500_improvUnet_GN"

to_tensor = transforms.ToTensor()

for img_dir in os.listdir(base_dir):
    imgs = []
    for img_name in os.listdir(os.path.join(base_dir,img_dir)):
        img = Image.open(os.path.join(base_dir, img_dir, img_name))
        imgs.append(to_tensor(img))

    imgs = torch.stack(imgs)
    grid = make_grid(imgs, nrow=8, normalize=False)
    save_image(grid, fp=f"{img_dir}_grid.png")

