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
from models.resnet34_unet import ResNet34_UNet
import oxford_pet
from evaluate import evaluate
from utils import new_writer

writer = new_writer()


# train U-Net
def train_unet(data_path="./dataset/oxford-iiit-pet", batch_size=15, epochs=5, lr=1e-5, n_aug=0):

    # train_dataset = oxford_pet.SimpleOxfordPetDataset(mode='train')
    # val_dataset = oxford_pet.SimpleOxfordPetDataset(mode='valid')


    train_dataset = oxford_pet.load_dataset(data_path=data_path, mode='train', n_aug=n_aug)
    val_dataset = oxford_pet.load_dataset(data_path=data_path, mode='valid')


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    

    unn_model = UNet().cuda()
    criterion = nn.CrossEntropyLoss() 
    optimizer = optim.Adam(unn_model.parameters(), lr=lr)

    
    for epoch in tqdm(range(epochs)):
        unn_model.train()
        train_running_loss = 0
        # use val_loader to test
        for sample in tqdm(train_dataloader):
            optimizer.zero_grad()

            imgs = sample['image'].float().cuda()
            # print(imgs.shape)
            masks = sample['mask'].squeeze(1).long().cuda()
            # print(masks.shape)


            preds = unn_model(imgs)
            
            loss = criterion(preds, masks)
            train_running_loss += loss.item()

            loss.backward()
            optimizer.step()
        
        val_running_loss = evaluate(net=unn_model, criterion=criterion, val_dataloader=val_dataloader)

        train_loss =  train_running_loss / len(train_dataloader)
        val_loss = val_running_loss / len(val_dataloader)
        print(f"\n Epoch {epoch+1}, avg train_running_loss: {train_loss} \n",
              f"Epoch {epoch+1}, avg val_running_loss: {val_loss} \n" )
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

    
    torch.save(unn_model.state_dict(), f"./saved_models/unet_weights_{epochs}_{lr}_{n_aug}.pth")

# train resnet34_unet
def train_resnet34_unet(data_path="./dataset/oxford-iiit-pet", batch_size=10, epochs=5, lr=1e-5, n_aug=0):

    # train_dataset = oxford_pet.SimpleOxfordPetDataset(root=data_path, mode='train')
    # val_dataset = oxford_pet.SimpleOxfordPetDataset(root=data_path, mode='valid')
    
    train_dataset = oxford_pet.load_dataset(data_path=data_path, mode='train', n_aug=n_aug)
    val_dataset = oxford_pet.load_dataset(data_path=data_path, mode='valid')


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    resnet34_unet = ResNet34_UNet().cuda()

    optimizer = optim.Adam(resnet34_unet.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in tqdm(range(epochs)):
        train_running_loss = 0
        for sample in tqdm(train_dataloader):
            optimizer.zero_grad()

            resnet34_unet.train()

            imgs = sample['image'].float().cuda()
            # masks = sample['mask'].squeeze(1).long().cuda()
            masks = sample['mask'].float().cuda()

            preds = resnet34_unet(imgs)

            # print(preds.shape, masks.shape)
            loss = criterion(preds, masks)
            train_running_loss += loss.item()


            loss.backward()
            optimizer.step()

        
        resnet34_unet.eval()
        val_running_loss = 0
        with torch.no_grad():
            for sample in tqdm(val_dataloader):
                imgs = sample['image'].float().cuda()
                masks = sample['mask'].float().cuda()

                preds = resnet34_unet(imgs)

                loss = criterion(preds, masks)
                val_running_loss += loss.item()

        train_loss =  train_running_loss / len(train_dataloader)
        val_loss = val_running_loss / len(val_dataloader)
        print(f"\n Epoch {epoch+1}, avg train_running_loss: {train_loss} \n",
              f"Epoch {epoch+1}, avg val_running_loss: {val_loss} \n" )
        
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

    
    torch.save(resnet34_unet.state_dict(), f"./saved_models/resnet34_unet_weights_{epochs}_{lr}_{n_aug}.pth")

            


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, default="./dataset/oxford-iiit-pet",  help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=15, help='batch size')
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5, help='learning rate')

    return parser.parse_args()
 
if __name__ == "__main__":
    # args = get_args()

    train_unet(epochs=3, batch_size=10, lr=1e-5, n_aug=0)
    # train_resnet34_unet(epochs=10, batch_size=15, lr=1e-5, n_aug=2)
    # train_resnet34_unet(epochs=10, batch_size=15, lr=1e-5, n_aug=0)



