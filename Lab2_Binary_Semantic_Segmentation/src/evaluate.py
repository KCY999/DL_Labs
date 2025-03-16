import torch
import os
import sys
import tqdm
from torch.utils.data import DataLoader
import torchvision


from utils import writer, dice_score_same_size
from models.unet import UNet
# from models.resnet34_unet import 
import oxford_pet

def evaluate(net, data, device):
    # implement the evaluation function here

    assert False, "Not implemented yet!"


if __name__ == "__main__":
    device = "cuda"
    model_name = "unet_weights_1_1e-05.pth"
    batch_size = 10
    
    test_dataset = oxford_pet.SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet", mode='test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size)

    unn_model = UNet().to(device)
    unn_model.load_state_dict(torch.load(f"./saved_models/{model_name}", map_location=device))

    unn_model.eval()
    s = 0

    avg_dice_score = 0 
    for sample in test_dataloader:
        sample_size = sample['image'].shape[0]

        imgs = sample['image'].float().cuda()
        masks = sample['mask'].float().cuda()
        # print(masks)
        writer.add_image('gt_mask', torchvision.utils.make_grid(masks), s)
        # print(masks.shape)
        preds = unn_model(imgs)
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
            avg_batch_dice_socre += ds / batch_size
            print(f"Batch {s}, instance {i}:  dice score  =  {ds}")
        
        print(f"Batch {s}: avg_dice_score  =  {avg_batch_dice_socre}")
        avg_dice_score += avg_batch_dice_socre


        s += 1
        # if s > 15:
        #     break
        
    print(f"Whole, avg_dice_score = {avg_dice_score / s}")
