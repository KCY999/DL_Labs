import torch
import torch.nn as nn

PADDING = 1
IMAGE_CHANNEL = 3
N_CLASSES = 2

"""
Ref:
    U-Net Architecture: paper of U-Net , https://www.youtube.com/watch?v=HS3Q_90hnDg
    Batch Normalization in CNN: https://medium.com/biased-algorithms/batch-normalization-in-cnn-81c0bd832c63

"""

class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleConv, self).__init__()
        self.d_conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=PADDING),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=PADDING),
            # nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.d_conv(x)
    
class DownSampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSampling, self).__init__()
        self.conv = DoubleConv(in_channel, out_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        crop = self.conv(x)
        p = self.pool(crop)
        return crop, p

class UpSampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UpSampling, self).__init__()
        self.up_conv = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channel, out_channel)
    
    def forward(self, x, crop):
        x = self.up_conv(x)
        # print(x.shape)
        # print(crop.shape)
        x = torch.cat([x, crop], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down_sampling_1 = DownSampling(IMAGE_CHANNEL, 64)
        self.down_sampling_2 = DownSampling(64, 128)
        self.down_sampling_3 = DownSampling(128, 256)
        self.down_sampling_4 = DownSampling(256, 512)
        
        self.bottom_conv = DoubleConv(512, 1024)

        self.up_sampling_1 = UpSampling(1024, 512)
        self.up_sampling_2 = UpSampling(512, 256)
        self.up_sampling_3 = UpSampling(256, 128)
        self.up_sampling_4 = UpSampling(128, 64)

        self.out_conv = nn.Conv2d(64, N_CLASSES, kernel_size=1)

    def forward(self, input):
        # endcode
        down_1, p1 = self.down_sampling_1(input)
        down_2, p2 = self.down_sampling_2(p1)
        down_3, p3 = self.down_sampling_3(p2)
        down_4, p4 = self.down_sampling_4(p3)


        # bottom
        bottom = self.bottom_conv(p4)
        # print(bottom.shape)

        # decode
        # print(down_4.shape)
        up_1 = self.up_sampling_1(bottom, down_4)
        up_2 = self.up_sampling_2(up_1, down_3)
        up_3 = self.up_sampling_3(up_2, down_2)
        up_4 = self.up_sampling_4(up_3, down_1)

        output = self.out_conv(up_4)

        return output


if __name__ == "__main__":
    import os
    import sys
    src = os.path.abspath("src")
    sys.path.append(src)
    
    import oxford_pet
    from torch.utils.data import DataLoader

    dataset = oxford_pet.SimpleOxfordPetDataset(root="./dataset/oxford-iiit-pet")
    datloader = DataLoader(dataset=dataset, batch_size=1)

    unn = UNet().cuda()

    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter('logs')


    import torchvision


    for data in datloader:
        print(data)
        imgs = data['image'].float().cuda()
        masks = data['mask'].float().cuda()  


        output = unn(imgs)


        # output = torch.softmax(output, dim=1)  
        # output = torch.argmax(output, dim=1, keepdim=True).float()
        print(output)

        writer.add_image('Original Image', torchvision.utils.make_grid(imgs, normalize=True))
        writer.add_image('Predicted Mask', torchvision.utils.make_grid(output, normalize=True))
        writer.add_image('Ground Truth Mask', torchvision.utils.make_grid(masks, normalize=True))

        print(f"Input shape: {imgs.shape}, Output shape: {output.shape}")

        break


