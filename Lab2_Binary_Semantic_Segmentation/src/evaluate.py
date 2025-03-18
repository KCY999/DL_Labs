import torch
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision


from utils import writer, dice_score_same_size
from models.unet import UNet
# from models.resnet34_unet import 
import oxford_pet

def evaluate(net, criterion, val_dataloader):
    net.eval()
    val_running_loss = 0
    with torch.no_grad():
        for sample in tqdm(val_dataloader):
            imgs = sample['image'].float().cuda()
            masks = sample['mask'].squeeze(1).long().cuda()

            preds = net(imgs)
            # print(preds.shape, masks.shape)
            loss = criterion(preds, masks)
            val_running_loss += loss.item()

    return val_running_loss


if __name__ == "__main__":

    pass
