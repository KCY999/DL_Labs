import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time

import os
import sys

src_dir = os.path.abspath("src")
sys.path.append(src_dir)
from models.unet import UNet
# from models.resnet34_unet import 
import oxford_pet


# train U-Net
def train_unet(args):
    print(args)
    data_path = args.data_path
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs


    batch_size = 15
    learning_rate = 1e-5
    epochs = 1



    train_dataset = oxford_pet.SimpleOxfordPetDataset(root=data_path, mode='train')
    val_dataset = oxford_pet.SimpleOxfordPetDataset(root=data_path, mode='valid')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    unn_model = UNet().cuda()
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(unn_model.parameters(), lr=learning_rate)

    
    for epoch in tqdm(range(epochs)):
        unn_model.train()
        train_running_loss = 0
        # use val_loader to test
        for sample in tqdm(train_dataloader):
            imgs = sample['image'].float().cuda()
            masks = sample['mask'].squeeze(1).long().cuda()

            optimizer.zero_grad()

            preds = unn_model(imgs)
            
            loss = criterion(preds, masks)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        unn_model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for sample in tqdm(val_dataloader):
                imgs = sample['image'].float().cuda()
                masks = sample['mask'].squeeze(1).long().cuda()


                preds = unn_model(imgs)
                loss = criterion(preds, masks)
                val_running_loss += loss.item()


        print(f"\n Epoch {epoch+1}, avg train_running_loss: {train_running_loss / len(train_dataloader)} \n",
              f"Epoch {epoch+1}, avg val_running_loss: {val_running_loss / len(val_dataloader)} \n" )

    
    torch.save(unn_model.state_dict(), f"./saved_models/unet_weights_{epochs}_{learning_rate}.pth")

    

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="./dataset/oxford-iiit-pet",  help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=15, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train_unet(args)


