import argparse
import torch
import os
import sys
import tqdm
from torch.utils.data import DataLoader
import torchvision

from utils import writer, dice_score_same_size
from models.unet import UNet
from models.resnet34_unet import ResNet34_UNet
import oxford_pet

def infer_UNet(model_name, batch_size=15):


    test_dataset = oxford_pet.SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet", mode='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = UNet().cuda()
    model.load_state_dict(torch.load(f"./saved_models/{model_name}"))

    model.eval()
    s = 0

    avg_dice_score = 0 
    for sample in test_dataloader:
        sample_size = sample['image'].shape[0]

        imgs = sample['image'].float().cuda()
        masks = sample['mask'].long().cuda()
        # print(masks)
        writer.add_image('gt_mask', torchvision.utils.make_grid(masks), s)
        # print(masks.shape)
        preds = model(imgs)
        # print(preds)
        # print(preds.shape)
        binary_preds = torch.argmax(preds, dim=1)
        binary_preds = binary_preds.unsqueeze(1)
        # print(binary_preds)
        # print(binary_preds.shape)
        writer.add_image('pred', torchvision.utils.make_grid(binary_preds), s)
        
        avg_batch_dice_socre = 0
        for i in range(sample_size):
            ds = dice_score_same_size(binary_preds[i][0], masks[i][0])
            avg_batch_dice_socre += ds / sample_size
            print(f"Batch {s}, instance {i}:  dice score  =  {ds}")
        
        print(f"Batch {s}: avg_dice_score  =  {avg_batch_dice_socre} \n")
        avg_dice_score += avg_batch_dice_socre


        s += 1
        # if s > 15:
        #     break
        
    print(f"Whole, avg_dice_score = {avg_dice_score / s}")


def infer_ResNet34_UNet(model_name, batch_size=15):

    test_dataset = oxford_pet.SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet", mode='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    model = ResNet34_UNet().cuda()
    model.load_state_dict(torch.load(f"./saved_models/{model_name}"))

    sigmoid = torch.nn.Sigmoid()

    model.eval()
    s = 0
    avg_dice_score = 0 
    for sample in test_dataloader:
        sample_size = sample['image'].shape[0]

        imgs = sample['image'].float().cuda()
        masks = sample['mask'].long().cuda()
        # print(masks)
        writer.add_image('gt_mask', torchvision.utils.make_grid(masks), s)
        # print(masks.shape)
        preds = model(imgs)
        # print(preds)
        # print(preds.shape)
        binary_preds = (sigmoid(preds) > 0.5).float()
        # print(binary_preds)
        # print(binary_preds.shape)
        writer.add_image('pred', torchvision.utils.make_grid(binary_preds), s)
        
        avg_batch_dice_socre = 0
        for i in range(sample_size):
            ds = dice_score_same_size(binary_preds[i][0], masks[i][0])
            avg_batch_dice_socre += ds / sample_size
            print(f"Batch {s}, instance {i}:  dice score  =  {ds}")
        
        print(f"Batch {s}: avg_dice_score  =  {avg_batch_dice_socre} \n")
        avg_dice_score += avg_batch_dice_socre

        s += 1
        # if s > 15:
        #     break
        
    print(f"Whole, avg_dice_score = {avg_dice_score / s}")

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', default='MODEL.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    infer_UNet("unet_weights_20_1e-05_0.pth")
    # infer_ResNet34_UNet("resnet34_unet_weights_10_1e-05_0.pth")