"""
FC-Siam-diff 变化检测模型 (标准版)

论文: Daudt et al. "Fully Convolutional Siamese Networks for Change Detection" (ICIP 2018)

架构:
  共享编码器 (48→96→192 通道 + 下采样)
  → 特征差
  → 解码器 + skip connection (U-Net 风格)
  → 变化图输出

参数量: ~625K (通道减半轻量版)
"""

import torch
import torch.nn as nn


class SiamDiffChangeDetector(nn.Module):
    """FC-Siam-diff 标准版"""

    def __init__(self):
        super().__init__()
        c = 48
        # 编码器
        self.e1 = nn.Sequential(
            nn.Conv2d(3, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.p1 = nn.MaxPool2d(2)
        self.e2 = nn.Sequential(
            nn.Conv2d(c, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(inplace=True),
            nn.Conv2d(c*2, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(inplace=True),
        )
        self.p2 = nn.MaxPool2d(2)
        self.e3 = nn.Sequential(
            nn.Conv2d(c*2, c*4, 3, padding=1), nn.BatchNorm2d(c*4), nn.ReLU(inplace=True),
        )
        # 解码器
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.d2 = nn.Sequential(
            nn.Conv2d(c*6, c*2, 3, padding=1), nn.BatchNorm2d(c*2), nn.ReLU(inplace=True),
        )
        self.d1 = nn.Sequential(
            nn.Conv2d(c*3, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(c, 1, 1)

    def forward(self, img_A, img_B, return_logits=False):
        def encode(x):
            f1 = self.e1(x)
            f2 = self.e2(self.p1(f1))
            f3 = self.e3(self.p2(f2))
            return f1, f2, f3

        a1, a2, a3 = encode(img_A)
        b1, b2, b3 = encode(img_B)

        x = self.d2(torch.cat([self.up(torch.abs(a3 - b3)), torch.abs(a2 - b2)], 1))
        x = self.d1(torch.cat([self.up(x), torch.abs(a1 - b1)], 1))
        out = self.out(x)

        if return_logits:
            return out
        return torch.sigmoid(out)

    def get_param_count(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}


def build_cnn_model():
    model = SiamDiffChangeDetector()
    params = model.get_param_count()
    print(f"[SiamDiff Std] 参数量: {params['total']:,} ({params['trainable']:,} 可训练)")
    return model


if __name__ == '__main__':
    m = build_cnn_model()
    x = torch.randn(2, 3, 256, 256)
    y = torch.randn(2, 3, 256, 256)
    o = m(x, y)
    print(f"输出: {list(o.shape)}")
