import torch
import os
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import torchvision


from utils import new_writer, dice_score_same_size, batch_avg_dice
from models.unet import UNet
# from models.resnet34_unet import 
import oxford_pet

def evaluate(net, criterion, val_dataloader, net_type):
    net.eval()
    val_running_loss = 0
    avg_dice = 0
    with torch.no_grad():
        for sample in tqdm(val_dataloader):
            imgs = sample['image'].float().cuda()
            if net_type == "unet":
                masks = sample['mask'].squeeze(1).long().cuda()
            else:
                masks = sample['mask'].float().cuda()


            preds = net(imgs)
            # print(preds.shape, masks.shape)
            loss = criterion(preds, masks)
            val_running_loss += loss.item()
            avg_dice += batch_avg_dice(preds, masks, type=net_type)

    return val_running_loss, avg_dice


if __name__ == "__main__":

    pass
