import torch
import torch.nn as nn

from models import unet 




# Block of residual
class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.is_down_sampling = (in_channels != out_channels)
        if self.is_down_sampling:
            conv1_stride = 2
            self.rec_identity = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(self.out_channels)
            )
        else:
            conv1_stride = 1
            self.rec_identity = None

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=conv1_stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)


    def forward(self, x):
        identity = x
        if self.is_down_sampling and self.rec_identity is not None:
            identity = self.rec_identity(identity)
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x += identity
        x = self.relu(x)

        return x






class unet_up_sampling_in_res34_unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(unet_up_sampling_in_res34_unet, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.conv = unet.DoubleConv(out_channel, out_channel)

 
    def forward(self, x, crop):
        # print("x:", x.shape)
        # print("crop:", crop.shape)
        x = torch.cat([x, crop], dim=1)
        # print('cated:', x.shape)
        x = self.up_conv(x)

        return self.conv(x)


class ResNet34_UNet(nn.Module):
    def __init__(self):
        super(ResNet34_UNet, self).__init__()
        
        # halves the size:  (S_in - k + 1 + 2P) / stride. for 2P = k -1, stride = 2:   S_out = S_in / 2 => P = (7 - 1) / 2 = 3
        self.res_conv = nn.Conv2d(3, 64, kernel_size=7, padding=3, stride=2)
        self.res_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res_down_sampling_1 = self._make_resnet34_down_sampling(64, 64, 3)
        self.res_down_sampling_2 = self._make_resnet34_down_sampling(64, 128, 4)
        self.res_down_sampling_3 = self._make_resnet34_down_sampling(128, 256, 6)
        self.res_down_sampling_4 = self._make_resnet34_down_sampling(256, 512, 3)
        
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # nn.Softmax2d()
        )

        # self.unet_up_sampling_1 = unet_up_sampling_in_res34_unet(512, 256)
        self.unet_up_sampling_1 = unet_up_sampling_in_res34_unet(256+512, 32)
        self.scaling_down3 = nn.Conv2d(256, 512, kernel_size=1)
        self.unet_up_sampling_2 = unet_up_sampling_in_res34_unet(32+512, 32)
        self.unet_up_sampling_3 = unet_up_sampling_in_res34_unet(32+128, 32)
        self.unet_up_sampling_4 = unet_up_sampling_in_res34_unet(32+64, 32)
        
        self.last_up_sampling = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        
        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)


    def forward(self, x):
        x = self.res_conv(x)
        x = self.res_pool(x)

        down_1 = self.res_down_sampling_1(x)
        # print("down_1:", down_1.shape)

        down_2 = self.res_down_sampling_2(down_1)
        # print("down_2:", down_2.shape)

        down_3 = self.res_down_sampling_3(down_2)
        # print("down_3:", down_3.shape)

        down_4 = self.res_down_sampling_4(down_3)
        # print("down_4:", down_4.shape)


        b = self.bridge(down_4)
        # print(b)
        # print("b:", b.shape)


        up_1 = self.unet_up_sampling_1(b, down_4)
        # print("up_1", up_1.shape)
        
        down_3 = self.scaling_down3(down_3)
        up_2 = self.unet_up_sampling_2(up_1, down_3)
        # print("up_2", up_2.shape)

        up_3 = self.unet_up_sampling_3(up_2, down_2)
        # print("up_3",up_3.shape)

        up_4 = self.unet_up_sampling_4(up_3, down_1)
        # print("up_4", up_4.shape)

        up_5 = self.last_up_sampling(up_4)

        out = self.out_conv(up_5)
        # print("out", out.shape)

        return out
        

    def _make_resnet34_down_sampling(self, in_channels, out_channels, n_block):
        blocks = []
        blocks.append(Block(in_channels, out_channels))

        for _ in range(1, n_block):
            blocks.append(Block(out_channels, out_channels))

        return nn.Sequential(*blocks)





if __name__ == "__main__":
    import os
    import sys
    src = os.path.abspath("src")
    sys.path.append(src)

    import oxford_pet
    from torch.utils.data import DataLoader
    import torch.nn.functional as F

    b_size = 10
    dataset = oxford_pet.SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet")
    datloader = DataLoader(dataset=dataset, batch_size=b_size)

    # a = torch.rand([10, 3, 256, 256]).cuda()
    # print(a.shape)
    resnet34_unet = ResNet34_UNet().cuda()

    from utils import dice_score_same_size, writer
    import torchvision
    

    s = 0
    for data in datloader:
        imgs = data['image'].float().cuda()
        masks = data['mask'].float().cuda()
        

        preds = resnet34_unet(imgs)
        print(preds)
        print(preds.shape)


        prids_binary = (F.sigmoid(preds) > 0.5).float()

        writer.add_image('Original Image', torchvision.utils.make_grid(imgs, normalize=True), s)
        writer.add_image('Predicted Mask', torchvision.utils.make_grid(prids_binary, normalize=True), s)
        writer.add_image('Ground Truth Mask', torchvision.utils.make_grid(masks, normalize=True), s)


        for i in range(b_size):
            ds = dice_score_same_size(preds, masks)
            print(ds)

        s += 1
        if s > 5:
            break







