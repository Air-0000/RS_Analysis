"""
LEVIR-CD 变化检测数据集加载器

数据集结构:
    LEVIR-CD256/
    ├── A/          # T1 时刻图像 (3×256×256)
    ├── B/          # T2 时刻图像 (3×256×256)
    ├── label/      # 变化标注 (1=变化, 0=不变)
    └── list/
        ├── train.txt
        ├── val.txt
        └── test.txt
"""

import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ChangeDetectionDataset(Dataset):
    """遥感变化检测数据集 (LEVIR-CD / DSIFN-CD 格式)"""

    def __init__(self, data_root, split='train', transform=None):
        """data_root: 根目录, split: 训练/验证/测试, transform: 数据增强"""

        self.data_root = data_root
        self.split = split
        self.transform = transform or self._get_default_transform()

        # 读取文件列表
        list_path = os.path.join(data_root, 'list', f'{split}.txt')
        with open(list_path, 'r') as f:
            self.names = [line.strip() for line in f.readlines() if line.strip()]

        self.img_dir_A = os.path.join(data_root, 'A')
        self.img_dir_B = os.path.join(data_root, 'B')
        self.label_dir = os.path.join(data_root, 'label')

        print(f"[Dataset] {split}: {len(self.names)} 对图像")

    def _get_default_transform(self):
        """默认变换：归一化 + ToTensor"""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        # 文件名不含扩展名，补上 .png
        name_png = name if name.endswith('.png') else name + '.png'

        # T1 图像
        img_A = np.array(Image.open(os.path.join(self.img_dir_A, name_png)).convert('RGB'))
        img_A = cv2.resize(img_A, (256, 256))

        # T2 图像
        img_B = np.array(Image.open(os.path.join(self.img_dir_B, name_png)).convert('RGB'))
        img_B = cv2.resize(img_B, (256, 256))

        # 变化标注
        label = np.array(Image.open(os.path.join(self.label_dir, name_png)).convert('L'))
        label = cv2.resize(label, (256, 256), interpolation=cv2.INTER_NEAREST)
        label = (label > 0).astype(np.float32)

        # 数据增强（同时对两张图做相同变换）
        if self.transform:
            augmented = self.transform(image=img_A, image_B=img_B, mask=label)
            img_A = augmented['image']
            img_B = augmented['image_B']
            label = augmented['mask'].float()

        return {
            'img_A': img_A,
            'img_B': img_B,
            'label': label,
            'name': name,
        }


def get_training_transform(img_size=256):
    """训练数据增强"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'image_B': 'image'})


def get_val_transform(img_size=256):
    """验证/测试数据增强"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], additional_targets={'image_B': 'image'})


def create_dataloaders(data_root, batch_size=16, num_workers=4):
    """创建数据加载器"""
    train_dataset = ChangeDetectionDataset(
        data_root, split='train',
        transform=get_training_transform()
    )
    val_dataset = ChangeDetectionDataset(
        data_root, split='val',
        transform=get_val_transform()
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
