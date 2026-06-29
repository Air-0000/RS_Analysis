"""
增强版 Tiny U-Net — 建筑语义分割

改进:
  1. SCConv (CVPR 2020): 空间-通道重建卷积替换标准 Conv-BN-ReLU
  2. CARAFE (ICCV 2019): 内容感知上采样替换双线性
  3. CoordAtt (CVPR 2021): 坐标注意力插入跳跃连接

参数量: ~2.1M (原版 ~2M，仅增加 ~100K)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .enhanced_modules import SCConv, CARAFE, CoordAtt


class SCConvBlock(nn.Module):
    """双卷积块 + SCConv 特征重建"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.scconv = SCConv(out_channels)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.scconv(x)          # ✨ SCConv 特征重建
        x = self.conv2(x)
        return x


class EnhancedTinyUNet(nn.Module):
    """增强版 Tiny U-Net — 建筑语义分割"""

    def __init__(self, in_channels=3, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        # === 编码器 ===
        self.encoder1 = SCConvBlock(in_channels, features[0])
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = SCConvBlock(features[0], features[1])
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = SCConvBlock(features[1], features[2])
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = SCConvBlock(features[2], features[3])

        # === 瓶颈 ===
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features[3], features[3] * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(features[3] * 2),
            nn.ReLU(inplace=True),
        )

        # === 解码器 ===
        # CoordAtt 插入在每次跳跃连接之后
        self.coordatt4 = CoordAtt(features[3] + features[3] * 2)
        self.upconv4 = nn.Sequential(
            nn.Conv2d(features[3] + features[3] * 2, features[3], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[3]),
            nn.ReLU(inplace=True),
        )
        self.up4 = CARAFE(features[3])  # ✨ CARAFE 上采样

        self.coordatt3 = CoordAtt(features[2] + features[3])
        self.upconv3 = nn.Sequential(
            nn.Conv2d(features[2] + features[3], features[2], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[2]),
            nn.ReLU(inplace=True),
        )
        self.up3 = CARAFE(features[2])

        self.coordatt2 = CoordAtt(features[1] + features[2])
        self.upconv2 = nn.Sequential(
            nn.Conv2d(features[1] + features[2], features[1], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[1]),
            nn.ReLU(inplace=True),
        )
        self.up2 = CARAFE(features[1])

        self.coordatt1 = CoordAtt(features[0] + features[1])
        self.upconv1 = nn.Sequential(
            nn.Conv2d(features[0] + features[1], features[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
        )

        # === 输出 ===
        self.out = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        # 编码
        e1 = self.encoder1(x)       # [B, 16, H, W]
        e2 = self.encoder2(self.pool1(e1))  # [B, 32, H/2, W/2]
        e3 = self.encoder3(self.pool2(e2))  # [B, 64, H/4, W/4]
        e4 = self.encoder4(self.pool3(e3))  # [B, 128, H/8, W/8]

        # 瓶颈
        b = self.bottleneck(e4)     # [B, 256, H/8, W/8]

        # 解码
        d4 = self.upconv4(self.coordatt4(torch.cat([e4, b], dim=1)))  # ✨ CoordAtt + concat
        d4 = self.up4(d4)           # ✨ CARAFE: H/8 → H/4

        d3 = self.upconv3(self.coordatt3(torch.cat([e3, d4], dim=1)))
        d3 = self.up3(d3)           # ✨ CARAFE: H/4 → H/2

        d2 = self.upconv2(self.coordatt2(torch.cat([e2, d3], dim=1)))
        d2 = self.up2(d2)           # ✨ CARAFE: H/2 → H

        d1 = self.upconv1(self.coordatt1(torch.cat([e1, d2], dim=1)))

        return self.out(d1)         # logits

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


def build_enhanced_unet(num_classes=1):
    model = EnhancedTinyUNet(out_channels=num_classes)
    params = model.get_param_count()
    return model


if __name__ == '__main__':
    m = build_enhanced_unet()
    x = torch.randn(2, 3, 256, 256)
    o = m(x)
    print(f"输入: {list(x.shape)} → 输出: {list(o.shape)}")
