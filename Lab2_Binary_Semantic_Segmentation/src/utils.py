import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

writer = SummaryWriter('logs/tmp')


def new_writer():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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


def dice_score(pred_mask, gt_mask):
    
    return 

if __name__ == "__main__":
    a = torch.rand([256, 256])
    b = torch.rand([256, 256])

    # print(dice_score_same_size(a, b))
    print(dice_score_same_size(a, a))


