import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import time
import copy
import os
import sys

src_dir = os.path.abspath("src")
sys.path.append(src_dir)
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
import oxford_pet
from evaluate import evaluate
from utils import new_writer, batch_avg_dice


class EarlyStop(object):
    def __init__(self, limit=7):
        self.end = False
        self.limit = limit
        self.lowest_loss = 1 
        self.stocking_count = 0
        self.best_model_wts = None

    
    def check(self, curr_loss, curr_model_wt):
        if (curr_loss <  self.lowest_loss):
            self.stocking_count = 0
            self.lowest_loss = curr_loss
            self.best_model_wts = copy.deepcopy(curr_model_wt)
        else:
            self.stocking_count += 1
            if self.stocking_count >= self.limit:
                self.end = True
        print(f"\nstocking_count = {self.stocking_count}\n")



def training(model_type ,data_path="./dataset/oxford-iiit-pet", batch_size=10, epochs=5, lr=1e-5, aug_type="combine", n_aug=0):
    
    # model = unet  or  resnet34_unet
    if model_type == "unet":
        model = UNet().cuda()
        criterion = nn.CrossEntropyLoss() 
    elif model_type == "resnet34_unet":
        model = ResNet34_UNet().cuda()
        criterion = nn.BCEWithLogitsLoss()
    else:
        assert False, "unknown model type"


    if aug_type == "combine":
        train_dataset = oxford_pet.load_dataset(data_path=data_path, mode='train', n_aug=n_aug)
    elif aug_type == "flip":
        train_dataset = oxford_pet.load_fliped_augmented_train_dataset(data_path=data_path, n_aug=n_aug)
    else:
        assert False, "unknown aug_type of train_dataset"
    val_dataset = oxford_pet.load_dataset(data_path=data_path, mode='valid')


    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    print(f"model_type = {model_type}")
    print(f"epochs = {epochs}")
    print(f"lr = {lr}")
    print(f"aug_type = {aug_type}")
    print(f"n_aug = {n_aug}\n")

    writer = new_writer(type="fit", model_name=f"{model_type}_{epochs}_{lr}_{aug_type}_{n_aug}")


    # optimizer = optim.Adam(model.parameters(), lr=lr)
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    early_stop = EarlyStop()

    for epoch in tqdm(range(epochs)):
        if early_stop.end:
            # epochs = epoch
            break

        train_running_loss = 0
        train_running_acc = 0
        for sample in tqdm(train_dataloader):
            optimizer.zero_grad()

            model.train()

            imgs = sample['image'].float().cuda()
            masks = sample['mask'].float().cuda() if model_type=="resnet34_unet" else sample['mask'].squeeze(1).long().cuda()


            preds = model(imgs)

            # print(preds.shape, masks.shape)
            loss = criterion(preds, masks)
            train_running_loss += loss.item()
            
            train_running_acc += batch_avg_dice(preds, masks, type=model_type)


            loss.backward()
            optimizer.step()

        
        val_running_loss, val_running_acc = evaluate(net=model, criterion=criterion, val_dataloader=val_dataloader, net_type=model_type)

        avg_train_loss =  train_running_loss / len(train_dataloader)
        avg_val_loss = val_running_loss / len(val_dataloader)

        early_stop.check(avg_val_loss, model.state_dict())

        print(f"\n Epoch {epoch+1}, avg_train_loss: {avg_train_loss} \n",
              f"Epoch {epoch+1}, avg_val_loss: {avg_val_loss}" )
        writer.add_scalars("Loss", {"Train": avg_train_loss, "Validation": avg_val_loss}, epoch)

        avg_train_acc = train_running_acc / len(train_dataloader)
        avg_val_acc = val_running_acc / len(val_dataloader)
        print(f" Epoch {epoch+1}, avg_train_acc: {avg_train_acc} \n",
              f"Epoch {epoch+1}, avg_val_acc: {avg_val_acc} \n" ) 
        writer.add_scalars("Acc", {"Train": avg_train_acc, "Validation": avg_val_acc}, epoch)
    
    torch.save(early_stop.best_model_wts, f"./saved_models/{model_type}_weights_{epochs}_{lr}_{aug_type}_{n_aug}.pth")


            
 
if __name__ == "__main__":




    # training("unet", epochs=60, batch_size=15, lr=1e-3, aug_type="combine", n_aug=2)
    # training("resnet34_unet", epochs=60, batch_size=15, lr=1e-3, aug_type="combine", n_aug=2)


    # training("unet", epochs=60, batch_size=15, lr=1e-4, aug_type="combine", n_aug=1)





    training("unet", epochs=60, batch_size=15, lr=1e-4, aug_type="combine", n_aug=2)





    # training("resnet34_unet", epochs=60, batch_size=15, lr=1e-3, aug_type="combine", n_aug=2)

    # training("resnet34_unet", epochs=60, batch_size=15, lr=1e-3, aug_type="flip", n_aug=2)
    # training("resnet34_unet", epochs=60, batch_size=15, lr=1e-3, aug_type="flip", n_aug=2)



    # training("unet", epochs=60, batch_size=15, lr=1e-3, aug_type="flip", n_aug=1)
    # training("resnet34_unet", epochs=60, batch_size=15, lr=1e-3, aug_type="flip", n_aug=1)

    # training("unet", epochs=2, batch_size=15, lr=1e-5, aug_type="combine", n_aug=0)
    # training("resnet34_unet", epochs=3, batch_size=15, lr=1e-5, aug_type="flip", n_aug=2)

    # training("unet", epochs=60, batch_size=15, lr=1e-5, aug_type="combine", n_aug=0)


    # training("unet", epochs=60, batch_size=15, lr=1e-5, aug_type="combine", n_aug=3)
    # training("resnet34_unet", epochs=60, batch_size=15, lr=1e-5, aug_type="combine", n_aug=3)


