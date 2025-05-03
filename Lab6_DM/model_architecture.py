import torch
import torch.nn as nn



class ConvBlockCond(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

        self.cond_proj = nn.Linear(cond_dim, out_ch)

    def forward(self, x, cond_embed):
        B, C, H, W = x.shape
        cond = self.cond_proj(cond_embed).view(B, -1, 1, 1) 
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x + cond)

        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x + cond)
        return x



class UNetCond(nn.Module):
    def __init__(self, cond_dim=24, time_dim=128):
        super().__init__()
        # Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.ReLU(),
            nn.Linear(time_dim, time_dim)
        )
        

        # Encoder blocks
        self.enc1 = ConvBlockCond(3, 64, cond_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlockCond(64, 128, cond_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlockCond(128, 256, cond_dim)
        self.pool3 = nn.MaxPool2d(2)

        # Bottleneck
        self.time_to_bot = nn.Linear(time_dim, 256) 
        self.bot = ConvBlockCond(256, 512, cond_dim)

        # Decoder
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = ConvBlockCond(512, 256, cond_dim)

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = ConvBlockCond(256, 128, cond_dim)

        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = ConvBlockCond(128, 64, cond_dim)

        self.out = nn.Conv2d(64, 3, 1)

    def forward(self, x, t, cond):
        # Embed time + condition
        t_embed = self.time_mlp(t.unsqueeze(-1).float())  # (B, D)

        # Encoder
        x1 = self.enc1(x, cond)    #  64x64
        p1 = self.pool1(x1)

        x2 = self.enc2(p1, cond)   #  32x32
        p2 = self.pool2(x2)

        x3 = self.enc3(p2, cond)   #  16x16
        p3 = self.pool3(x3)

        # Bottleneck 
        temb_proj = self.time_to_bot(t_embed).view(t_embed.size(0), 256, 1, 1)
        b = self.bot(p3 + temb_proj, cond)

        # Decoder
        u3 = self.up3(b)
        u3 = self.dec3(torch.cat([u3, x3], dim=1), cond)

        u2 = self.up2(u3)
        u2 = self.dec2(torch.cat([u2, x2], dim=1), cond)

        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, x1], dim=1), cond)

        return self.out(u1)



# ===================================================
# imporved ver full cond. enbedding + time embedding

class ConvblockCondTime(nn.Module):
    def __init__(self, in_ch, out_ch, group_normalize=True, num_groups=8):
        super().__init__()
        
        self.time_embedding = nn.Sequential(
            nn.Linear(1, 128),
            nn.SiLU(inplace=True),
            nn.Linear(128, 128)
        )
        
        self.time_mlp = nn.Sequential(
            nn.Linear(128, out_ch),
            nn.SiLU(inplace=True),
            nn.Linear(out_ch, out_ch)
        )

        self.cond_mlp = nn.Sequential(
            nn.Linear(24, out_ch),
            nn.SiLU(inplace=True),
            nn.Linear(out_ch, out_ch)
        )

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        if group_normalize:        
            self.norm1 = nn.GroupNorm(num_groups, out_ch)
        else:
            self.norm1 = nn.BatchNorm2d(out_ch)

        self.act1 = nn.SiLU(inplace=True)

        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        if group_normalize:        
            self.norm2 = nn.GroupNorm(num_groups, out_ch)
        else:
            self.norm2 = nn.BatchNorm2d(out_ch)

        self.act2 = nn.SiLU(inplace=True)

    def forward(self, x, t, cond):
        B = x.shape[0]

        t_embed = self.time_embedding(t.unsqueeze(-1).float())
        t_proj = self.time_mlp(t_embed).view(B, -1, 1, 1)

        cond_proj = self.cond_mlp(cond).view(B, -1, 1, 1)
        
        # print(t.shape)
        # print(cond.shape)

        x = self.conv1(x)
        x = x + t_proj + cond_proj
        x = self.norm1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = x + t_proj + cond_proj
        x = self.norm2(x)
        x = self.act2(x)

        return x




class UnetCondTime(nn.Module):
    def __init__(self, group_normalize=False):
        super().__init__()
        
        self.enc1 = ConvblockCondTime(3, 64, group_normalize, num_groups=8)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2 = ConvblockCondTime(64, 128, group_normalize, num_groups=16)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.enc3 = ConvblockCondTime(128, 256, group_normalize, num_groups=32)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        

        self.bot = ConvblockCondTime(256, 512, group_normalize, num_groups=64)


        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvblockCondTime(512, 256, group_normalize, num_groups=32)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvblockCondTime(256, 128, group_normalize, num_groups=16)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvblockCondTime(128, 64, group_normalize, num_groups=8)


        self.output = nn.Conv2d(64, 3, kernel_size=1)


    def forward(self, x, t, cond):
        down1 = self.enc1(x, t, cond)
        p1 = self.pool1(down1)

        down2 = self.enc2(p1, t, cond)
        p2 = self.pool2(down2)

        down3 = self.enc3(p2, t, cond)
        p3 = self.pool3(down3)

        b = self.bot(p3, t, cond)

        up3 = self.up3(b)
        up3 = self.dec3(torch.cat([up3, down3], dim=1), t, cond)  # skip connection
        
        up2 = self.up2(up3)
        up2 = self.dec2(torch.cat([up2, down2], dim=1), t, cond)

        up1 = self.up1(up2)
        up1 = self.dec1(torch.cat([up1, down1], dim=1), t, cond)


        return self.output(up1)



