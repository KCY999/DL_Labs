import argparse
import torch
import os
import sys
import tqdm
from torch.utils.data import DataLoader
import torchvision

from utils import new_writer, dice_score_same_size, batch_avg_dice
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
import oxford_pet

def infer_UNet(model_name, batch_size=15):
    writer = new_writer(type="img")

    test_dataset = oxford_pet.SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet", mode='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = UNet().cuda()
    model.load_state_dict(torch.load(f"./saved_models/{model_name}"))

    model.eval()
    s = 0

    avg_dice_score = 0 
    for sample in test_dataloader:
        # sample_size = sample['image'].shape[0]

        imgs = sample['image'].float().cuda()
        masks = sample['mask'].long().cuda()
        # print(masks)
        writer.add_image('gt_mask', torchvision.utils.make_grid(masks), s)
        # print(masks.shape)
        preds = model(imgs)
        # print(preds)
        # print(preds.shape)
        avg_batch_dice_socre = batch_avg_dice(preds, masks)
        print(f"Batch {s}: avg_dice_score  =  {avg_batch_dice_socre} \n")
        avg_dice_score += avg_batch_dice_socre

        binary_preds = torch.argmax(preds, dim=1)
        binary_preds = binary_preds.unsqueeze(1)
        # print(binary_preds)
        # print(binary_preds.shape)
        writer.add_image('pred', torchvision.utils.make_grid(binary_preds), s)        
        s += 1

    avg_dice_score /= s
    print(f"Whole, avg_dice_score = {avg_dice_score}")


def infer_ResNet34_UNet(model_name, batch_size=15):
    writer = new_writer(type="img")

    test_dataset = oxford_pet.SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet", mode='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = ResNet34_UNet().cuda()
    model.load_state_dict(torch.load(f"./saved_models/{model_name}"))

    sigmoid = torch.nn.Sigmoid()

    model.eval()
    s = 0
    avg_dice_score = 0 
    for sample in test_dataloader:
        # sample_size = sample['image'].shape[0]

        imgs = sample['image'].float().cuda()
        masks = sample['mask'].float().cuda()
        # print(masks)

        writer.add_image(f'gt_mask', torchvision.utils.make_grid(masks), s)
        # print(masks.shape)
        preds = model(imgs)
        avg_batch_dice_socre = batch_avg_dice(preds, masks, type="resnet34_unet")
        print(f"Batch {s}: avg_dice_score  =  {avg_batch_dice_socre} \n")

        # print(preds)
        # print(preds.shape)
        binary_preds = (sigmoid(preds) > 0.5).float()
        # print(binary_preds)
        # print(binary_preds.shape)
        writer.add_image(f'pred', torchvision.utils.make_grid(binary_preds), s)
        

        avg_dice_score += avg_batch_dice_socre

        s += 1

    avg_dice_score /= s
    print(f"Whole, avg_dice_score = {avg_dice_score}")



def infer(model_name, model_type, batch_size=15):
    writer = new_writer(type="img")

    test_dataset = oxford_pet.SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet", mode='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    if model_type == 'unet':
        model = UNet().cuda()
    elif model_type == 'resnet34_unet':
        model = ResNet34_UNet().cuda()
    else:
        assert False, "unknown model type"
    
    model.load_state_dict(torch.load(f"./saved_models/{model_name}"))

    model.eval()
    
    s = 0

    avg_dice_score = 0 
    for sample in test_dataloader:
        # sample_size = sample['image'].shape[0]

        imgs = sample['image'].float().cuda()
        masks = sample['mask'].long().cuda() if model_type == 'unet' else sample['mask'].float().cuda()
        # print(masks)
        writer.add_image('gt_mask', torchvision.utils.make_grid(masks), s)
        # print(masks.shape)
        preds = model(imgs)
        # print(preds)
        # print(preds.shape)

        avg_batch_dice_socre = batch_avg_dice(preds, masks, type=model_type)
        print(f"Batch {s}: avg_dice_score  =  {avg_batch_dice_socre} \n")
        avg_dice_score += avg_batch_dice_socre


        # visulize preds
        if model_type == 'unet':
            binary_preds = torch.argmax(preds, dim=1)
            binary_preds = binary_preds.unsqueeze(1)
        else:
            binary_preds = (torch.sigmoid(preds) > 0.5).float()

        # print(binary_preds)
        # print(binary_preds.shape)
        writer.add_image('pred', torchvision.utils.make_grid(binary_preds), s)        
        s += 1

    avg_dice_score /= s
    print(f"Whole, avg_dice_score = {avg_dice_score}")




if __name__ == '__main__':

    # infer_UNet("unet_weights_60_1e-05_flip_2.pth")
    # infer_ResNet34_UNet("resnet34_unet_weights_60_1e-05_flip_2.pth")

    
    # infer("unet_weights_50_1e-05_combine_1.pth", model_type='unet')
    infer("resnet34_unet_weights_50_1e-05_flip_1.pth", model_type='resnet34_unet')