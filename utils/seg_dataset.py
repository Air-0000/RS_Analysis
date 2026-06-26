"""
通用二值语义分割数据加载器

适用于:
  WHU Building Dataset: data/WHU_Building/
  Massachusetts Buildings: data/Massachusetts_Buildings/
  或其他 image/mask 结构的二类分割数据集

数据集结构:
  data_root/
  ├── train/
  │   ├── images/  (*.png/*.jpg)
  │   └── masks/   (*.png, 0=背景, 1=目标)
  └── val/
      ├── images/
      └── masks/
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class BinarySegDataset(Dataset):
    """通用二值语义分割数据集 (适配 WHU / Massachusetts / 等)

    支持目录名:
      images/ 或 image/  (输入图像)
      masks/  或 label/  (标注掩码)
    支持格式:
      .png .jpg .tif .tiff
    """

    def __init__(self, data_root, split='train', img_size=256):
        self.data_root = Path(data_root)
        self.split = split
        self.img_size = img_size

        # 尝试多种常见目录名
        img_dirs = [
            self.data_root / split / 'images',
            self.data_root / split / 'image',
            self.data_root / split / 'A',
        ]
        mask_dirs = [
            self.data_root / split / 'masks',
            self.data_root / split / 'mask',
            self.data_root / split / 'label',
            self.data_root / split / 'labels',
            self.data_root / split / 'OUT',
        ]

        img_dir = next((d for d in img_dirs if d.exists()), None)
        mask_dir = next((d for d in mask_dirs if d.exists()), None)

        assert img_dir is not None, f"未找到图目录: {[str(d) for d in img_dirs]}"
        assert mask_dir is not None, f"未找到标注目录: {[str(d) for d in mask_dirs]}"

        self.images = sorted([f for f in img_dir.iterdir()
                              if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')])
        self.masks = sorted([f for f in mask_dir.iterdir()
                             if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff')])

        assert len(self.images) > 0, f"未找到图像: {img_dir}"
        assert len(self.images) == len(self.masks), \
            f"图像和标注数量不匹配: {len(self.images)} vs {len(self.masks)}"
        print(f"[Dataset] {split}: {len(self.images)} 对")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # 图像
        img = cv2.imread(str(self.images[idx]))
        if img is None:
            # 尝试 PIL (处理 TIFF 等格式)
            from PIL import Image
            img = np.array(Image.open(self.images[idx]).convert('RGB'))
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 标注 (二值)
        mask = cv2.imread(str(self.masks[idx]), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            from PIL import Image
            mask = np.array(Image.open(self.masks[idx]).convert('L'))
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 0).astype(np.float32)

        # 归一化
        img = img.astype(np.float32) / 255.0
        img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]

        return {
            'image': torch.from_numpy(img).permute(2, 0, 1).float(),
            'mask': torch.from_numpy(mask).float(),  # (H, W), 0/1
        }


def create_dataloaders(data_root, batch_size=8, img_size=256, num_workers=0):
    train_dataset = BinarySegDataset(data_root, 'train', img_size)
    val_dataset = BinarySegDataset(data_root, 'val', img_size)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
