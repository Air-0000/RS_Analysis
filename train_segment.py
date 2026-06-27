"""
增强版 Tiny U-Net 语义分割训练脚本

改进:
  - 模型: SCConv + CARAFE + CoordAtt (EnhancedTinyUNet)
  - 损失: Dice + BCE 组合损失
"""

import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

PROJ_ROOT = str(Path(__file__).parent)
sys.path.insert(0, PROJ_ROOT)

from models.unet_tiny import build_enhanced_unet
from utils.seg_dataset import create_dataloaders


class DiceBCELoss(nn.Module):
    """Dice + BCE 组合损失，直接优化 mIoU"""
    def __init__(self, weight_dice=0.5, weight_bce=0.5):
        super().__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, mask):
        # BCE 部分
        bce_loss = self.bce(logits, mask)

        # Dice 部分
        prob = torch.sigmoid(logits)
        smooth = 1.0
        intersect = (prob * mask).sum(dim=(2, 3))
        union = prob.sum(dim=(2, 3)) + mask.sum(dim=(2, 3))
        dice = (2 * intersect + smooth) / (union + smooth)
        dice_loss = 1 - dice.mean()

        return self.weight_bce * bce_loss + self.weight_dice * dice_loss


def train():
    import argparse
    p = argparse.ArgumentParser(description='增强版 Tiny U-Net 二值语义分割训练')
    p.add_argument('--data_root', default='data/building_dataset')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--exp_name', default=None)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    exp = args.exp_name or f"enhanced_unet_{datetime.now().strftime('%m%d_%H%M')}"
    out_dir = Path(f"outputs/{exp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader = create_dataloaders(
        args.data_root, args.batch_size, args.img_size, num_workers=0
    )

    model = build_enhanced_unet(num_classes=1).to(device)
    criterion = DiceBCELoss(weight_dice=0.5, weight_bce=0.5)  # ✨ Dice+BCE 组合
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f" 增强版 Tiny U-Net 训练 (SCConv + CARAFE + CoordAtt)")
    print(f"{'='*50}")
    print(f" 参数: {params:,}")
    print(f" 损失: Dice(0.5) + BCE(0.5)")
    print(f" Epochs: {args.epochs}")
    print(f" Batch: {args.batch_size}")
    print(f" 设备: {device}")
    print(f"{'='*50}\n")

    best_iou = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')

        for batch in pbar:
            img = batch['image'].to(device)
            mask = batch['mask'].to(device)

            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, mask.unsqueeze(1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(train_loader)

        # 验证
        model.eval()
        ious = []
        with torch.no_grad():
            for batch in val_loader:
                img = batch['image'].to(device)
                mask = batch['mask'].to(device)
                pred = (torch.sigmoid(model(img)) > 0.5).float()

                inter = ((pred.squeeze(1) + mask) == 2).sum().item()
                union = ((pred.squeeze(1) + mask) >= 1).sum().item()
                ious.append(inter / (union + 1e-7))

        miou = sum(ious) / len(ious)
        lr_now = optimizer.param_groups[0]['lr']
        print(f" Loss: {avg_loss:.4f} | Val mIoU: {miou:.4f} | LR: {lr_now:.2e}")

        if miou > best_iou:
            best_iou = miou
            torch.save(model.state_dict(), out_dir / 'best.pth')
            torch.save(model.state_dict(), Path('outputs') / 'best_unet.pth')
            print(f" ✓ 保存最佳模型 (mIoU={miou:.4f})")

        scheduler.step()

    print(f"\n{'='*50}")
    print(f" 训练完成! 最佳 mIoU: {best_iou:.4f}")
    print(f" 模型: {out_dir / 'best.pth'}")
    print(f"{'='*50}")


if __name__ == '__main__':
    train()
