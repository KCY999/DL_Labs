import torch
from torch.utils.tensorboard import SummaryWriter
import datetime


def new_writer(type="fit", model_name = None):
    name = model_name + "_"  if model_name else ""
    log_dir = f"logs/{type}/" + name + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir)
    return writer


def dice_score_same_size(pred_mask, gt_mask, epsilon=1e-6):
    size = pred_mask.shape[0] * pred_mask.shape[1]
    diff = pred_mask - gt_mask
    zero_region = (diff == 0)
    # print(zero_region.shape)
    # print(zero_region)
    zero_count = zero_region.sum()
    # print(zero_count)
    # print(zero_region.shape)

    dice = ( zero_count + epsilon) / (size + epsilon)
    return dice.item()


def batch_avg_dice(pred_masks, gt_masks, type="unet", epsilon=1e-6):
    n = pred_masks.shape[0]
    size = pred_masks.shape[-1]**2

    # print(pred_masks.shape, gt_masks.shape)

    if type == "unet":
        binary_preds = torch.argmax(pred_masks, dim=1)
        gt_masks = gt_masks.squeeze(1)
    else:
        sigmoid = torch.nn.Sigmoid()
        binary_preds = (sigmoid(pred_masks) > 0.5).float().squeeze(1)
        gt_masks = gt_masks.squeeze(1)

    # print(binary_preds)
    diff = binary_preds - gt_masks
    zero_sum = (diff == 0).sum()
    # print(zero_sum)
    avg_dice = (zero_sum + epsilon) / ( n * size + epsilon)

    return avg_dice.item()

def batch_avg_dice_without_back(pred_masks, gt_masks, type="unet", epsilon=1e-6):
    n = pred_masks.shape[0]

    if type == "unet":
        binary_preds = torch.argmax(pred_masks, dim=1)  
        gt_masks = gt_masks.squeeze(1)                 
    else:
        sigmoid = torch.nn.Sigmoid()
        binary_preds = (sigmoid(pred_masks) > 0.5).float().squeeze(1)  
        gt_masks = gt_masks.squeeze(1)

    dices = []
    for i in range(n):
        pred = binary_preds[i]
        gt = gt_masks[i]

        intersection = (pred * gt).sum()
        pred_sum = pred.sum()
        gt_sum = gt.sum()
        dice = (2 * intersection + epsilon) / (pred_sum + gt_sum + epsilon)

        dices.append(dice)

    avg_dice = torch.stack(dices).mean()
    return avg_dice.item()






if __name__ == "__main__":
    # a = torch.rand([256, 256])
    # b = torch.rand([256, 256])
    # print(dice_score_same_size(a, a))
    # print(dice_score_same_size(a, b))

    c = torch.rand([3, 1, 256, 256])
    d = torch.rand([3, 1, 256, 256])
    print(batch_avg_dice(c, c, type='resnet34_unet'))
    print(batch_avg_dice(c, d, type='resnet34_unet'))
    


