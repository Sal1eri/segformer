import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.transformer import TransformerEncoder
import config

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSample, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # 计算填充大小，确保x1和x2尺寸匹配
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        
        # 拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TransUNet(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_channels=3, num_classes=1,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4.,
                 start_channels=64, bilinear=True):
        super(TransUNet, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim

        # U-Net 编码器部分
        self.inc = ConvBlock(in_channels, start_channels)
        self.down1 = DownSample(start_channels, start_channels * 2)
        self.down2 = DownSample(start_channels * 2, start_channels * 4)
        self.down3 = DownSample(start_channels * 4, start_channels * 8)
        self.down4 = DownSample(start_channels * 8, start_channels * 16)

        # Transformer编码器
        self.transformer = TransformerEncoder(
            img_size=img_size // 16,
            patch_size=1,
            in_channels=start_channels * 16,
            embed_dim=embed_dim,
            depth=depth,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio
        )

        # 投影回原始通道数
        self.conv_transformer = nn.Conv2d(embed_dim, start_channels * 16, kernel_size=1)

        # 修正上採樣層的輸入通道數
        self.up1 = UpSample(start_channels * (16 * 2 + 8), start_channels * 8, bilinear)  # 32+8=40
        self.up2 = UpSample(start_channels * (8 + 4), start_channels * 4, bilinear)  # 8+4=12
        self.up3 = UpSample(start_channels * (4 + 2), start_channels * 2, bilinear)  # 4+2=6
        self.up4 = UpSample(start_channels * (2 + 1), start_channels, bilinear)  # 2+1=3

        # 最終輸出層
        self.outc = nn.Conv2d(start_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # 編碼器路徑
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Transformer處理
        x_transformer = self.transformer(x5)
        x_transformer = rearrange(x_transformer, 'b (h w) c -> b c h w', h=x5.shape[2], w=x5.shape[3])
        x_transformer = self.conv_transformer(x_transformer)

        # 拼接特徵並上採樣
        x = torch.cat([x5, x_transformer], dim=1)  # 通道數: start_channels*32
        x = self.up1(x, x4)  # 輸入通道數: 32+8=40
        x = self.up2(x, x3)  # 輸入通道數: 8+4=12
        x = self.up3(x, x2)  # 輸入通道數: 4+2=6
        x = self.up4(x, x1)  # 輸入通道數: 2+1=3

        logits = self.outc(x)
        return logits