"""
增强模块库 — 桌面「模型改进」资源移植

SCConv (CVPR 2020 SCNet):  Spatial and Channel Reconstruction Conv
CARAFE (ICCV 2019):      Content-Aware ReAssembly of Features
CoordAtt (CVPR 2021):    Coordinate Attention for Mobile Networks
FocalLoss (RetinaNet):   解决类别不平衡的 Focal Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# SCConv — Spatial and Channel Reconstruction Convolution
# ============================================================
# 来源: 桌面 模型改进/ultralytics-main/ultralytics/nn/extra_modules/block.py
# 论文: CVPR 2020 SCNet
# 核心: 通过 avgpool + sigmoid 门控实现空间-通道特征重建
class SCConv(nn.Module):
    """SCConv: 空间-通道重建卷积（轻量深度可分离版）

    标准卷积 → 深度可分离卷积 (3×3 DW + 1×1 PW)
    参数量: O(ch) vs O(ch²), 适配大通道场景
    """
    def __init__(self, channels, pooling_r=4):
        super().__init__()
        # 深度可分离卷积: 3×3 depthwise + 1×1 pointwise
        def dw_conv(c):
            return nn.Sequential(
                nn.Conv2d(c, c, 3, padding=1, groups=c, bias=False),
                nn.Conv2d(c, c, 1, bias=False),
            )

        self.k2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
            dw_conv(channels),
            nn.BatchNorm2d(channels),
        )
        self.k3 = nn.Sequential(
            dw_conv(channels),
            nn.BatchNorm2d(channels),
        )
        self.k4 = nn.Sequential(
            dw_conv(channels),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        identity = x
        k2_out = F.interpolate(self.k2(x), size=x.shape[2:], mode='nearest')
        gate = torch.sigmoid(identity + k2_out)
        out = self.k3(x) * gate
        out = self.k4(out)
        return out


# ============================================================
# CARAFE — Content-Aware ReAssembly of Features
# ============================================================
# 来源: 桌面 模型改进/ultralytics-main/ultralytics/nn/extra_modules/block.py
# 论文: ICCV 2019
# 核心: 内容感知的重组上采样（优于双线性/双三次）
class CARAFE(nn.Module):
    """CARAFE: 内容感知上采样，输入输出通道数相同"""
    def __init__(self, channels, scale=2, k_up=5, k_enc=3, c_mid=64):
        super().__init__()
        self.scale = scale
        self.comp = nn.Sequential(
            nn.Conv2d(channels, c_mid, 1, bias=False),
            nn.BatchNorm2d(c_mid),
            nn.ReLU(inplace=True),
        )
        self.enc = nn.Conv2d(c_mid, (scale * k_up) ** 2, k_enc, padding=k_enc // 2, bias=False)
        self.pix_shf = nn.PixelShuffle(scale)
        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        # 生成上采样核权重
        W = self.comp(X)                     # b, m, h, w
        W = self.enc(W)                      # b, (scale*k_up)^2, h, w
        W = self.pix_shf(W)                  # b, k_up^2, h_, w_
        W = torch.softmax(W, dim=1)          # b, k_up^2, h_, w_  ← 核权重归一化

        # 内容感知重组
        X = self.upsmp(X)                    # b, c, h_, w_
        X = self.unfold(X)                   # b, c*k_up^2, h_*w_
        X = X.view(b, c, -1, h_, w_)         # b, c, k_up^2, h_, w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b, c, h_, w_
        return X


# ============================================================
# CoordAtt — Coordinate Attention
# ============================================================
# 来源: 桌面 模型改进/ultralytics-main/ultralytics/nn/extra_modules/attention.py
# 论文: CVPR 2021
# 核心: 将位置编码为坐标注意力权重，增强空间定位
class CoordAtt(nn.Module):
    """坐标注意力 — 编码位置信息的通道注意力"""
    def __init__(self, channels, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, channels // reduction)
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mip)
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1, bias=False)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1, bias=False)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = F.relu(y, inplace=True)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        return identity * a_w * a_h


# ============================================================
# Focal Loss
# ============================================================
# 来源: 桌面 模型改进/ultralytics-main/ultralytics/utils/loss.py
# 论文: RetinaNet, Lin et al. ICCV 2017
# 核心: 降低 easy negative 权重，聚焦难例
class FocalLoss(nn.Module):
    """Focal Loss — 解决类别不平衡"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, target):
        """
        pred: logits [B, 1, H, W]
        target: binary mask [B, H, W] or [B, 1, H, W]
        """
        if target.dim() == 3:
            target = target.unsqueeze(1).float()
        loss = F.binary_cross_entropy_with_logits(pred, target.float(), reduction='none')
        pred_prob = pred.sigmoid()
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if self.alpha > 0:
            alpha_factor = target * self.alpha + (1 - target) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean()
