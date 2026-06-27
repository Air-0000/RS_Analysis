"""
Tiny U-Net 遥感图像语义分割

轻量级 U-Net，支持二类（建筑分割）和多类分割。
编码器: 16 → 32 → 64 → 128
参数量: ~0.8M
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, base_ch=16):
        """in_channels: 输入通道, num_classes: 输出类别, base_ch: 基础通道"""

        super().__init__()
        c = base_ch
        self.num_classes = num_classes

        # 编码器
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)

        # 瓶颈
        self.bottleneck = DoubleConv(c * 8, c * 16)

        # 解码器
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec4 = DoubleConv(c * 16 + c * 8, c * 8)
        self.dec3 = DoubleConv(c * 8 + c * 4, c * 4)
        self.dec2 = DoubleConv(c * 4 + c * 2, c * 2)
        self.dec1 = DoubleConv(c * 2 + c, c)

        # 输出
        self.out = nn.Conv2d(c, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], 1))
        return self.out(d1)  # logits

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


def build_unet(num_classes=1):
    model = TinyUNet(num_classes=num_classes)
    params = model.get_param_count()
    print(f"[Tiny U-Net] 参数量: {params['total']:,} ({params['trainable']:,} 可训练)")
    return model
