"""
增强版 FC-Siam-diff 变化检测训练脚本

改进:
  - 模型: SCConv + CoordAtt + CARAFE (EnhancedSiamDiff)
  - 损失: Focal Loss (gamma=2.0, alpha=0.25) 替代 BCE
"""

import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJ_ROOT = str(Path(__file__).parent)
sys.path.insert(0, PROJ_ROOT)

from models.cnn_cd import build_enhanced_cd_model
from models.enhanced_modules import FocalLoss
from utils.dataset import create_dataloaders
from utils.metrics import MetricsTracker


def get_device():
    if torch.cuda.is_available():
        d = torch.device('cuda')
        print(f"[Device] CUDA: {torch.cuda.get_device_name(0)}")
    else:
        d = torch.device('cpu')
        print("[Device] CPU")
    return d


def train():
    import argparse
    p = argparse.ArgumentParser(description='增强版 FC-Siam-diff 变化检测训练')
    p.add_argument('--data_root', default='data/LEVIR-CD256')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--eval_threshold', type=float, default=0.3)
    p.add_argument('--exp_name', default=None)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', default=None, help='从 best.pth 续训')
    p.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma')
    p.add_argument('--focal_alpha', type=float, default=0.25, help='Focal Loss alpha')
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = get_device()

    exp = args.exp_name or f"enhanced_siamdiff_{datetime.now().strftime('%m%d_%H%M')}"
    out_dir = Path(f"outputs/{exp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / 'logs'))

    train_loader, val_loader = create_dataloaders(
        args.data_root, args.batch_size, num_workers=0
    )

    model = build_enhanced_cd_model().to(device)
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt)
        print(f" 续训: {args.resume}")

    # ✨ Focal Loss 替代 BCE — 解决变化像素仅 8-15% 的不平衡
    criterion = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*50}")
    print(f" 增强版 FC-Siam-diff 训练")
    print(f"{'='*50}")
    print(f" 参数: {params:,}")
    print(f" 损失: Focal Loss (gamma={args.focal_gamma}, alpha={args.focal_alpha})")
    print(f" Epochs: {args.epochs}")
    print(f" Batch: {args.batch_size}")
    print(f" LR: {args.lr}")
    print(f" 评估阈值: {args.eval_threshold}")
    print(f" 设备: {device}")
    print(f" 输出: {out_dir}")
    print(f"{'='*50}\n")

    best_f1 = 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        # 训练
        model.train()
        tm = MetricsTracker()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}')

        for batch in pbar:
            img_A = batch['img_A'].to(device)
            img_B = batch['img_B'].to(device)
            label = batch['label'].to(device)

            optimizer.zero_grad()
            logits = model(img_A, img_B, return_logits=True)
            loss = criterion(logits, label.unsqueeze(1).float())
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            pred = torch.sigmoid(logits.detach())
            tm.update(pred, label, threshold=args.eval_threshold)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'F1': f'{tm.get_metrics()["F1"]:.4f}'})

        train_metrics = tm.get_metrics()
        writer.add_scalar('Train/Loss', loss.item(), epoch)
        writer.add_scalar('Train/F1', train_metrics['F1'], epoch)

        # 验证
        model.eval()
        vm = MetricsTracker()
        with torch.no_grad():
            for batch in val_loader:
                img_A = batch['img_A'].to(device)
                img_B = batch['img_B'].to(device)
                label = batch['label'].to(device)
                pred = model(img_A, img_B)
                vm.update(pred, label, threshold=args.eval_threshold)

        val_metrics = vm.get_metrics()
        writer.add_scalar('Val/F1', val_metrics['F1'], epoch)
        writer.add_scalar('Val/IoU', val_metrics['IoU'], epoch)

        lr_now = optimizer.param_groups[0]['lr']
        print(f" Train F1: {train_metrics['F1']:.4f} | Val F1: {val_metrics['F1']:.4f} "
              f"IoU: {val_metrics['IoU']:.4f} | Best: {max(best_f1, val_metrics['F1']):.4f} | "
              f"LR: {lr_now:.2e}")

        if val_metrics['F1'] > best_f1:
            best_f1 = val_metrics['F1']
            torch.save(model.state_dict(), out_dir / 'best.pth')
            torch.save(model.state_dict(), Path('outputs') / 'best_siamdiff.pth')
            print(f" ✓ 保存最佳模型 (F1={best_f1:.4f})")

        scheduler.step()

    print(f"\n{'='*50}")
    print(f" 训练完成! 最佳 Val F1: {best_f1:.4f}")
    print(f" 模型: {out_dir / 'best.pth'}")
    print(f" TensorBoard: tensorboard --logdir {out_dir / 'logs'}")
    print(f"{'='*50}")


if __name__ == '__main__':
    train()
