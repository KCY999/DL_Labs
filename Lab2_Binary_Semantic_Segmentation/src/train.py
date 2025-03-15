import argparse
import os
import sys
src_dir = os.path.abspath("src")
sys.path.append(src_dir)
from models.unet import UNet
# from models.resnet34_unet import 
import oxford_pet

import torch
from torch.utils.data import DataLoader






# train U-Net
def train_unet(args):
    print(args)
    
    train_dataset = oxford_pet.SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet", mode='train')
    val_dataset = oxford_pet.SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet", mode='train')






    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    


