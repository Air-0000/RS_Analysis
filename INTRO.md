# 🛰️ 遥感图像分析平台

> **《人工智能基础B》课程大作业**

---

## 一、项目概述

### 1.1 项目简介

本系统是一个基于深度学习的遥感图像分析平台，提供两个核心功能：

| 功能 | 说明 | 用户场景 |
|:----|:-----|:---------|
| **📷 语义分割** | 从单张遥感图像中提取建筑区域 | 城市规划、土地利用调查 |
| **🔄 变化检测** | 对比两时相图像，标出建筑变化区域 | 违章建筑监测、灾害评估 |

系统配备交互式 Web 界面，用户上传图像即可在线推理，无需编写代码。

### 1.2 技术栈

| 技术 | 用途 |
|:----|:------|
| **PyTorch** | 深度学习框架 |
| **Streamlit** | Web 交互界面 |
| **OpenCV + PIL** | 图像处理 |
| **Albumentations** | 数据增强 |
| **TensorBoard** | 训练过程监控 |
| **CUDA 13.0** | GPU 加速 |

---

## 二、模型详解

### 2.1 建筑分割 — Tiny U-Net

**架构设计：**

Tiny U-Net 是一个轻量化的 U-Net 变体，采用对称的编码器-解码器结构。

```
编码器:
  输入: 256×256×3 (RGB)
  ↓ Conv(3→16) + BN + ReLU
  ↓ Conv(16→16) + BN + ReLU        → 256×256×16  (skip e1)
  ↓ MaxPool(2×2)
  ↓ Conv(16→32) + BN + ReLU
  ↓ Conv(32→32) + BN + ReLU        → 128×128×32  (skip e2)
  ↓ MaxPool(2×2)
  ↓ Conv(32→64) + BN + ReLU
  ↓ Conv(64→64) + BN + ReLU        → 64×64×64    (skip e3)
  ↓ MaxPool(2×2)
  ↓ Conv(64→128) + BN + ReLU
  ↓ Conv(128→128) + BN + ReLU      → 32×32×128   (skip e4)
  ↓ MaxPool(2×2)

瓶颈:
  Conv(128→256) + BN + ReLU × 2    → 16×16×256

解码器:
  ↑ Upsample(2×) + 拼接 e4
  ↓ Conv(256+128→128) × 2          → 32×32×128
  ↑ Upsample(2×) + 拼接 e3
  ↓ Conv(128+64→64) × 2            → 64×64×64
  ↑ Upsample(2×) + 拼接 e2
  ↓ Conv(64+32→32) × 2             → 128×128×32
  ↑ Upsample(2×) + 拼接 e1
  ↓ Conv(32+16→16) × 2             → 256×256×16
  ↓ Conv(16→1)                     → 256×256×1 (建筑概率图)
```

**关键参数：**

| 参数 | 值 |
|:----|:---|
| 参数量 | 1,965,569 (~2M) |
| 输入尺寸 | 256×256 RGB |
| 输出尺寸 | 256×256 单通道概率图 |
| 损失函数 | BCEWithLogitsLoss |
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-4) |
| 学习率策略 | CosineAnnealingLR |
| 训练 epoch | 50 |

### 2.2 变化检测 — FC-Siam-diff

**架构设计：**

FC-Siam-diff（Fully Convolutional Siamese-difference）来自 Daudt et al. (ICIP 2018)，核心思想是 T1 和 T2 图像通过共享权重的编码器提取特征，在特征空间做差，再解码重建变化图。

```
T1 图像 → [共享编码器 Conv×3] → feature₁ ─┐
                                              ├→ |feature₁ - feature₂|
T2 图像 → [共享编码器 Conv×3] → feature₂ ─┘         ↓
                                            [解码器 + Skip Connection]
                                                     ↓
                                              变化概率图 (256×256)
```

**编码器细节：**

```
Conv(3→48) + BN + ReLU
Conv(48→48) + BN + ReLU           256×256×48
↓ MaxPool(2×2)
Conv(48→96) + BN + ReLU
Conv(96→96) + BN + ReLU           128×128×96
↓ MaxPool(2×2)
Conv(96→192) + BN + ReLU          64×64×192
```

**解码器细节：**

```
↑ Upsample(2×) + 拼接编码器第2层特征差
Conv(192+96→96) + BN + ReLU       128×128×96
↑ Upsample(2×) + 拼接编码器第1层特征差
Conv(96+48→48) + BN + ReLU        256×256×48
Conv(48→1) 1×1                    256×256×1 (变化概率)
```

**关键参数：**

| 参数 | 值 |
|:----|:---|
| 参数量 | 625,297 |
| 输入 | T1 (256×256×3) + T2 (256×256×3) |
| 输出 | 256×256 变化概率图 |
| 损失函数 | BCEWithLogitsLoss |
| 优化器 | AdamW (lr=1e-3) |
| 训练 epoch | 30 |

---

## 三、数据集

### 3.1 WHU Building Dataset（建筑分割）

| 属性 | 值 |
|:----|:-----|
| 来源 | 武汉大学 |
| 图像 | 8189 张 512×512 卫星图像 |
| 标注 | 建筑/非建筑 二值掩码 |
| 分辨率 | 0.3-0.5m/pixel |
| 训练/验证 | 4736 / 1036 张 |
| 下载 | Google Drive（约 450MB） |

**标注示例：**

```
原图 (RGB)               标注 (二值)
┌──────────────┐        ┌──────────────┐
│   🏠 🏠 🏠    │        │   ██  ██     │
│  🏠    🏠     │   →    │  ██   ██     │
│ 🏠 🏠 🏠 🏠   │        │ ██ ██ ██ ██  │
└──────────────┘        └──────────────┘
```

### 3.2 LEVIR-CD（变化检测）

| 属性 | 值 |
|:----|:-----|
| 来源 | 武汉大学 LEVIR 实验室 |
| 图像 | 637 对 1024×1024 卫星图像 |
| 标注 | 建筑变化/不变 二值掩码 |
| 分辨率 | 0.5m/pixel |
| 地域 | 美国德州 20 个城市 |
| 训练/验证/测试 | 445 / 64 / 128 |
| 下载 | Google Drive（约 2.5GB） |

---

## 四、实验结果

### 4.1 建筑分割 — WHU Building

| epoch | 训练 Loss | 验证 mIoU |
|:----:|:---------:|:---------:|
| 10 | 0.0712 | 0.7521 |
| 20 | 0.0548 | 0.8136 |
| 30 | 0.0493 | 0.8524 |
| 40 | 0.0468 | 0.8691 |
| 50 | 0.0435 | **0.8774** |

**参数量对比：**

| 模型 | 参数量 | 预训练 | mIoU |
|:----|:-----:|:------:|:----:|
| DeepLabV3+ (ResNet50) | 40M | ✅ ImageNet | ~0.89 |
| U-Net (ResNet34) | 24M | ✅ ImageNet | ~0.88 |
| U-Net (VGG16) | 24M | ✅ ImageNet | ~0.87 |
| **Tiny U-Net (本系统)** | **2M** | **❌ 无** | **0.877** |

→ 仅用 1/10 的参数量，无预训练，达到接近 SOTA 的性能。

### 4.2 变化检测 — LEVIR-CD

| epoch | 训练 F1 | 验证 F1 | 验证 IoU |
|:----:|:-------:|:-------:|:--------:|
| 5 | 0.427 | 0.330 | 0.198 |
| 10 | 0.547 | 0.349 | 0.213 |
| 15 | 0.627 | 0.367 | 0.224 |
| 20 | 0.681 | 0.370 | 0.228 |
| 28 | 0.801 | **0.651** | 0.477 |

→ FC-Siam-diff 标准版本约 0.73（1M 参数），本系统为轻量版（0.625M），性能达到原版的约 90%。

---

## 五、Web 界面使用说明

### 5.1 启动

```bash
bash run.sh
# 浏览器访问 http://localhost:8501
```

### 5.2 功能介绍

**📷 语义分割模式：**

1. 上传一张遥感/航拍图像
2. 系统自动提取建筑区域
3. 显示：原图 vs 分割覆盖图 + 面积占比统计
4. 可调整分割阈值控制敏感度

**🔄 变化检测模式：**

1. 上传同一地点不同时间的两张图像
2. 系统自动标出变化区域
3. 显示：T1/T2/差异图 + 变化概率图 + 变化覆盖图
4. 可调整检测阈值控制敏感度

### 5.3 样例测试

系统内置样例数据：
- 分割样例：WHU 数据集（高/中建筑密度各一组）
- 检测样例：LEVIR-CD 双时相对比

---

## 六、项目结构

```
RS_Analysis/                  # 项目根目录
├── app.py                    # Streamlit 双模式界面
├── train.py                  # 变化检测训练脚本
├── train_seg.py              # 语义分割训练脚本
├── setup.sh                  # 一键环境配置
├── run.sh                    # 一键启动脚本
│
├── models/                   # 模型定义
│   ├── __init__.py
│   ├── cnn_cd.py             # FC-Siam-diff 变化检测
│   └── unet_tiny.py          # Tiny U-Net 语义分割
│
├── utils/                    # 工具模块
│   ├── __init__.py
│   ├── dataset.py            # LEVIR-CD 数据加载
│   ├── seg_dataset.py        # WHU 等分割数据加载
│   └── metrics.py            # 评估指标 (IoU, F1, etc.)
│
├── data/                     # 数据集（需手动下载）
│   ├── LEVIR-CD256/          # 变化检测数据
│   └── WHU_Building/         # 建筑分割数据
│
├── outputs/                  # 预训练权重
│   ├── best_siamdiff.pth     # 变化检测 (F1=0.65, 2.5MB)
│   └── best_unet.pth         # 建筑分割 (mIoU=0.877, 7.6MB)
│
├── samples/                  # 样例图片
│
├── requirements.txt          # Python 依赖
├── README.md                 # 项目快速介绍
└── INTRO.md                  # 本文档（详细介绍）
```

---

## 七、环境配置

### 7.1 自动配置

```bash
bash setup.sh
```

### 7.2 手动配置

```bash
# 创建虚拟环境（可选）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 下载数据集（详见下方说明）
# WHU Building: https://drive.google.com/...
# LEVIR-CD: https://drive.google.com/...
```

### 7.3 依赖清单

```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.24.0
opencv-python>=4.8.0
albumentations>=1.3.0
streamlit>=1.28.0
matplotlib>=3.7.0
Pillow>=10.0.0
tqdm>=4.65.0
tensorboard>=2.13.0
pandas>=2.0.0
```

---

## 八、常见问题

**Q: 训练脚本运行报错 `ModuleNotFoundError`？**
A: 先运行 `bash setup.sh` 安装依赖。

**Q: 显存不足（OOM）？**
A: 降低 batch_size，如 `--batch_size 8` 或 `--batch_size 4`。

**Q: 界面无法访问？**
A: 检查端口是否被占用：`netstat -ano | grep 8501`。

**Q: 分割/检测效果不理想？**
A: 调整阈值参数，或增加训练 epoch。

---

## 九、参考文献

1. Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015)
2. Daudt et al. "Fully Convolutional Siamese Networks for Change Detection" (ICIP 2018)
3. Chen & Shi. "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection" (Remote Sensing, 2020)
4. WHU Building Dataset. Wuhan University. http://gpcv.whu.edu.cn/data/
5. Eshraghian et al. "Training Spiking Neural Networks Using Lessons From Deep Learning" (Proceedings of the IEEE, 2023)
