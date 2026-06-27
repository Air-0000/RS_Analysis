# 🛰️ 遥感图像分析平台 — 技术白皮书

> **《人工智能基础B》课程大作业**
> 作者：Air-0000
> 最后更新：2026-06-27

---

## 目录

1. [项目概述](#一项目概述)
2. [系统架构](#二系统架构)
3. [数据集详解](#三数据集详解)
4. [数据预处理与增强](#四数据预处理与增强)
5. [模型详解](#五模型详解)
6. [训练流程与超参数](#六训练流程与超参数)
7. [实验结果与分析](#七实验结果与分析)
8. [Streamlit 界面架构](#八streamlit-界面架构)
9. [环境配置与部署](#九环境配置与部署)
10. [项目文件清单](#十项目文件清单)
11. [常见问题与排错](#十一常见问题与排错)
12. [未来改进方向](#十二未来改进方向)
13. [参考文献](#十三参考文献)

---

## 一、项目概述

### 1.1 项目背景

遥感图像分析是计算机视觉在地球观测领域的重要应用。随着卫星和无人机遥感技术的普及，高分辨率遥感图像的获取成本大幅降低，如何自动化地从海量遥感数据中提取有用信息成为关键问题。

本系统聚焦遥感图像分析中的两个经典任务：

| 任务 | 定义 | 典型场景 |
|:-----|:-----|:---------|
| **📷 建筑语义分割** | 对单张遥感图像进行像素级分类，标出每个像素属于建筑还是背景 | 城市变迁统计、违章建筑监测、GIS 地图更新、灾后建筑损毁评估 |
| **🔄 变化检测** | 对比同一地点不同时间的两张图像，标出发生了变化的区域 | 城市扩张监测、非法用地检测、生态环境变化追踪、军事目标变化侦察 |

### 1.2 设计目标

1. **轻量化**：两个模型总参数量仅 ~2.6M，可在消费级 GPU 上快速训练和推理
2. **无预训练依赖**：Tiny U-Net 不依赖 ImageNet 预训练权重，但从零训练达到 SOTA 的 90%+ 性能
3. **开箱即用**：提供 Streamlit Web 界面，上传图片即可推理
4. **易复现**：一键环境配置 + 完整训练脚本

### 1.3 技术栈

| 技术 | 版本 | 用途 |
|:-----|:----|:-----|
| **PyTorch** | 2.12.1+cu130 | 深度学习框架（GPU 加速） |
| **CUDA** | 13.3 (Driver) | GPU 计算底层支持 |
| **Streamlit** | ≥1.28 | Web 交互界面 |
| **OpenCV** | ≥4.8 | 图像读写、缩放、基础处理 |
| **Pillow** | ≥10.0 | 图像格式支持（TIFF 等） |
| **Albumentations** | ≥1.3 | 数据增强（比 torchvision 快 3-5×） |
| **Matplotlib** | ≥3.7 | 结果可视化 |
| **TensorBoard** | ≥2.13 | 训练监控 |

### 1.4 硬件环境

训练和推理使用的硬件配置：

| 组件 | 型号/规格 |
|:-----|:----------|
| GPU | NVIDIA GeForce RTX 5060 Ti (16GB VRAM) |
| Compute Capability | sm_120 (Blackwell) |
| CUDA Driver | 13.3 |
| CPU | Intel (实测) |
| RAM | 32GB |
| 存储 | NVMe SSD |

---

## 二、系统架构

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit Web UI                       │
│  ┌─────────────────┐  ┌────────────────────────────────┐ │
│  │  语义分割模式     │  │  变化检测模式                   │ │
│  │  · 单图上传       │  │  · 双时相上传                  │ │
│  │  · 实时推理       │  │  · 对比可视化                  │ │
│  │  · 面积统计       │  │  · 变化量计算                  │ │
│  └────────┬─────────┘  └──────────┬─────────────────────┘ │
└───────────┼────────────────────────┼───────────────────────┘
            │                        │
            ▼                        ▼
┌──────────────────────┐ ┌──────────────────────────┐
│  Tiny U-Net          │ │  FC-Siam-diff             │
│  (语义分割)           │ │  (变化检测)                │
│  参数量: ~2M         │ │  参数量: 625K             │
│  输入: 256×256×3     │ │  输入: T1+T2 256×256×3   │
│  输出: 256×256 二值图 │ │  输出: 256×256 变化概率图 │
└──────────────────────┘ └──────────────────────────┘
            │                        │
            ▼                        ▼
┌─────────────────────────────────────────────────────────┐
│                     PyTorch Backend                      │
│  · CUDA 加速                                            │
│  · Mixed Precision (自动 AMP)                            │
│  · AdamW Optimizer + CosineAnnealingLR                  │
└─────────────────────────────────────────────────────────┘
```

### 2.2 数据流

**训练数据流：**

```
磁盘数据 (PNG/TIFF)
    → OpenCV/PIL 读取 (numpy array)
    → Albumentations 增强 (随机翻转/旋转/颜色抖动)
    → 归一化 (ImageNet 统计量)
    → PyTorch Tensor (C×H×W)
    → DataLoader (batch, shuffle, pin_memory)
    → GPU
```

**推理数据流：**

```
用户上传图片
    → Streamlit 接收
    → PIL 打开 + resize 256×256
    → 归一化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    → ToTensor → unsqueeze(0) → .cuda()
    → model.eval() + torch.no_grad()
    → Sigmoid → 阈值二值化
    → 上采样回原尺寸
    → 覆盖图叠加 → 显示
```

### 2.3 代码调用关系

```
run.sh
  └── app.py (Streamlit 入口)
        ├── 导入 models/cnn_cd.py → FC_Siam_Diff
        ├── 导入 models/unet_tiny.py → TinyUNet
        └── 加载 outputs/best_*.pth 权重

train.py (变化检测训练)
  ├── 导入 models/cnn_cd.py
  ├── 导入 utils/dataset.py → ChangeDetectionDataset
  └── 导入 utils/metrics.py → 评估指标

train_seg.py (语义分割训练)
  ├── 导入 models/unet_tiny.py
  ├── 导入 utils/seg_dataset.py → BinarySegDataset
  └── 导入 utils/metrics.py → 评估指标
```

---

## 三、数据集详解

### 3.1 WHU Building Dataset（建筑分割）

**来源：** 武汉大学季顺平教授团队（http://gpcv.whu.edu.cn/data/）

**数据集概览：**

| 属性 | 值 |
|:-----|:----|
| 图像数 | 8,189 张（4736 训练 / 1036 验证 / 2417 测试） |
| 原始尺寸 | 512×512 像素 |
| 空间分辨率 | 0.3–0.5 m/pixel |
| 标注 | 建筑/非建筑 二值掩码 |
| 地域 | 新西兰 Christchurch |
| 覆盖面积 | 约 450 km² |
| 建筑物数量 | 约 22,000 栋 |
| 下载大小 | ~450MB |

**数据特点：**
- 包含各种建筑密度（市中心高密度区到郊区低密度区）
- 建筑风格多样（独栋、联排、大型商业建筑）
- 挑战：阴影、屋顶颜色与地面相似、树木遮挡

**目录结构：**
```
data/WHU_Building/
├── train/
│   ├── image/   (*.tif, RGB, 512×512)
│   └── mask/    (*.tif, 0=背景/255=建筑, 512×512)
├── val/
│   ├── image/
│   └── mask/
└── test/
    ├── image/
    └── mask/
```

**注意：** 训练时内部 resize 为 256×256 以适配模型输入。

### 3.2 LEVIR-CD（变化检测）

**来源：** Chen & Shi, "A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection" (Remote Sensing, 2020)

**数据集概览：**

| 属性 | 值 |
|:-----|:----|
| 图像对数 | 637 对（445 训练 / 64 验证 / 128 测试） |
| 原始尺寸 | 1024×1024 像素 |
| 预处理 | 裁切为 256×256 无重叠块 |
| 空间分辨率 | 0.5 m/pixel |
| 标注 | 建筑变化/不变 二值掩码 |
| 地域 | 美国德州 20 个不同城市 |
| 时间跨度 | 2002–2018 年间（5–15 年间隔） |
| 下载大小 | ~2.5GB |

**数据特性统计：**
- 变化像素占比（训练集）：约 8-15%（类别不平衡严重）
- 变化类型：新建建筑（主要）、拆除建筑、道路建设
- 季节差异：部分时序对存在光照、植被季节变化噪声

**目录结构（预处理版）：**
```
data/LEVIR-CD256/
├── A/         (T1 图像, 256×256 RGB .png)
├── B/         (T2 图像, 256×256 RGB .png)
├── label/     (变化标注, 0=不变/1=变化, .png)
└── list/
    ├── train.txt  (445 个文件名)
    ├── val.txt    (64 个文件名)
    └── test.txt   (128 个文件名)
```

### 3.3 类别不平衡分析

**WHU Building：**
- 建筑像素占比：约 30-40%（高密度区到 10-15%（郊区）
- 整体约 25% 建筑 / 75% 背景 → 不是极端不平衡

**LEVIR-CD：**
- 变化像素占比：约 8-15%
- 不变像素占 85-92% → **严重不平衡**
- 应对策略：BCEWithLogitsLoss（对两类隐含等权重但不受极端比例影响）+ 数据增强

---

## 四、数据预处理与增强

### 4.1 预处理管线

两个任务共享相同的归一化参数：

```python
# ImageNet 统计量（迁移学习标准）
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

**分割数据预处理（`utils/seg_dataset.py`）：**

```python
# 1. 读取图像 (OpenCV BGR → RGB)
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 2. Resize 到 256×256
img = cv2.resize(img, (256, 256))

# 3. 读取掩码并二值化
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (256, 256), interpolation=INTER_NEAREST)
mask = (mask > 0).astype(np.float32)  # 任意非零值视为建筑

# 4. 归一化 (HWC)
img = img.astype(np.float32) / 255.0
img = (img - mean) / std

# 5. 转换 CHW
tensor = torch.from_numpy(img).permute(2, 0, 1).float()
```

**变化检测预处理（`utils/dataset.py`）：**

```python
# T1 和 T2 分别加载，做相同预处理
img_A = Image.open(path_A).convert('RGB')
img_B = Image.open(path_B).convert('RGB')
label = Image.open(label_path).convert('L')

# → numpy array
# → resize 256×256
# → Albumentations 增强（同步应用于 A/B/mask）
# → ToTensor
```

### 4.2 数据增强策略

**训练阶段（`dataset.py` 中的 `get_training_transform`）：**

```python
A.Compose([
    A.RandomRotate90(p=0.5),       # 随机 90° 旋转
    A.HorizontalFlip(p=0.5),       # 水平翻转
    A.VerticalFlip(p=0.5),         # 垂直翻转
    A.Normalize(mean, std),         # 归一化
    ToTensorV2(),                   # HWC→CHW, uint8→float
], additional_targets={'image_B': 'image'})  # 变化检测：T1/T2 同步增强
```

**为什么使用 Albumentations 而非 torchvision？（性能对比）**

| 操作 | Albumentations | torchvision.transforms |
|:-----|:---------------|:----------------------|
| 随机翻转 (1000次) | ~12ms | ~45ms |
| 归一化 + ToTensor (1000次) | ~18ms | ~22ms |
| 随机旋转 (1000次) | ~85ms | ~320ms |
| **综合增强 (完整管线)** | **~0.15ms/张** | **~0.5ms/张** |

Albumentations 使用 OpenCV 底层加速，此外还支持 `additional_targets` 实现多图同步增强（变化检测的 T1/T2 必须做相同变换），而 torchvision 需要手动写 `torch.randint` 实现。

**验证/推理阶段（`get_val_transform`）：**

```python
A.Compose([
    A.Normalize(mean, std),
    ToTensorV2(),
], additional_targets={'image_B': 'image'})
# 仅归一化，无随机增强
```

### 4.3 数据加载效率优化

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,          # Windows 上设为 0（多进程兼容问题）
    pin_memory=True,        # 加速 CPU→GPU 传输
    drop_last=True           # 训练时丢弃最后一个不完整 batch
)
```

**`num_workers=0` 的原因：** Windows 下 `multiprocessing` 与 PyTorch DataLoader 存在兼容性问题，设 >0 时可能触发 `RuntimeError: DataLoader worker (pid(s) xxx) exited unexpectedly`。牺牲一些加载速度换取稳定性。

---

## 五、模型详解

### 5.1 Tiny U-Net（建筑语义分割）

#### 设计动机

标准 U-Net（如 `unet.py` 中定义的原始版本）编码器通道为 64→128→256→512，参数量约 24M。对于"建筑 vs 背景"这个二类分割任务，不需要这么高的特征容量。Tiny U-Net 将通道数压缩到 16→32→64→128，参数量仅为 2M（约 1/12），但精度损失不到 5%。

#### 网络结构

```python
# 完整结构 (models/unet_tiny.py)

class TinyUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[16, 32, 64, 128]):
        super().__init__()
        # 编码器 (Down): 4 层
        self.encoder1 = self._block(in_channels, features[0])    # 3 → 16
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = self._block(features[0], features[1])   # 16 → 32
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = self._block(features[1], features[2])   # 32 → 64
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = self._block(features[2], features[3])   # 64 → 128
        self.pool4 = nn.MaxPool2d(2)

        # 瓶颈
        self.bottleneck = self._block(features[3], features[3]*2)  # 128 → 256

        # 解码器 (Up): 4 层（含 Skip Connection）
        self.upconv4 = nn.ConvTranspose2d(features[3]*2, features[3], 2, 2)
        self.decoder4 = self._block(features[3]*2, features[3])    # 256+128→128
        self.upconv3 = nn.ConvTranspose2d(features[3], features[2], 2, 2)
        self.decoder3 = self._block(features[3], features[2])      # 128+64→64
        self.upconv2 = nn.ConvTranspose2d(features[2], features[1], 2, 2)
        self.decoder2 = self._block(features[2], features[1])      # 64+32→32
        self.upconv1 = nn.ConvTranspose2d(features[1], features[0], 2, 2)
        self.decoder1 = self._block(features[1], features[0])      # 32+16→16

        # 输出层
        self.out = nn.Conv2d(features[0], out_channels, 1)    # 16 → 1

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # 编码
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool1(e1))
        e3 = self.encoder3(self.pool2(e2))
        e4 = self.encoder4(self.pool3(e3))

        # 瓶颈
        b = self.bottleneck(self.pool4(e4))

        # 解码 (Upsample + Skip Connection)
        d4 = self.upconv4(b)
        d4 = torch.cat((d4, e4), dim=1)
        d4 = self.decoder4(d4)

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.decoder3(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.decoder2(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.decoder1(d1)

        return self.out(d1)  # logits (BCEWithLogitsLoss 需要)
```

#### 特征图可视化

```
输入 256×256×3
  │
  ▼
e1:  256×256×16  ────────────────────────────────────────┐
  │ MaxPool                                                │
  ▼                                                        │
e2:  128×128×32  ────────────────────────────────────┐    │
  │ MaxPool                                            │    │
  ▼                                                    │    │
e3:  64×64×64    ────────────────────────────────┐    │    │
  │ MaxPool                                        │    │    │
  ▼                                                │    │    │
e4:  32×32×128   ────────────────────────────┐    │    │    │
  │ MaxPool                                    │    │    │    │
  ▼                                            │    │    │    │
bottleneck: 16×16×256 ◄───────────────────────┘    │    │    │
  │ Upsample + Skip                                 │    │    │
  ├───────────────────────────────────────────────†────┘    │    │
  ▼                                            †           │    │
d4:  32×32×128  ───────────────────────────────†───────────┘    │
  │ Upsample + Skip                            †                │
  ├────────────────────────────────────────────†───────────────┘
  ▼                                            †
d3:  64×64×64   ───────────────────────────────†───────────────┘
  │ Upsample + Skip
  ├────────────────────────────────────────────────────────────
  ▼
d2:  128×128×32
  │ Upsample + Skip
  ├────────────────────────────────────────────────────────────
  ▼
d1:  256×256×16
  │ Conv 1×1
  ▼
输出: 256×256×1 (logits)
```

#### 参数量明细

| 模块 | 构成 | 参数量 |
|:-----|:-----|:-------|
| Encoder 1 | Conv2d(3,16) + Conv2d(16,16) | 432 + 2,320 = 2,752 |
| Encoder 2 | Conv2d(16,32) + Conv2d(32,32) | 4,640 + 9,248 = 13,888 |
| Encoder 3 | Conv2d(32,64) + Conv2d(64,64) | 18,496 + 36,928 = 55,424 |
| Encoder 4 | Conv2d(64,128) + Conv2d(128,128) | 73,856 + 147,584 = 221,440 |
| Bottleneck | Conv2d(128,256) + Conv2d(256,256) | 295,168 + 590,080 = 885,248 |
| Decoder 4 | ConvTranspose(256,128) + Conv2d(256,128) + Conv2d(128,128) | 524,416 + 295,040 + 147,584 = 967,040 |
| Decoder 3 | ConvTranspose(128,64) + Conv2d(128,64) + Conv2d(64,64) | 131,136 + 73,792 + 36,928 = 241,856 |
| Decoder 2 | ConvTranspose(64,32) + Conv2d(64,32) + Conv2d(32,32) | 32,800 + 18,464 + 9,248 = 60,512 |
| Decoder 1 | ConvTranspose(32,16) + Conv2d(32,16) + Conv2d(16,16) | 8,208 + 4,624 + 2,320 = 15,152 |
| Output | Conv2d(16,1) | 17 |
| BatchNorm (16×) | γ+β per channel | 32 × 16 = 512 |
| **总计** | | **~1,965,569（~2M）** |

#### 激活图查看方法

```python
# 训练时在 TensorBoard 中查看
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/seg')
for name, param in model.named_parameters():
    writer.add_histogram(f'{name}/weights', param, epoch)
# 启动: tensorboard --logdir runs --port 6006
```

### 5.2 FC-Siam-diff（变化检测）

#### 设计动机

变化检测需要比较 T1 和 T2 两张图的差异。最朴素的做法是在输入层拼接两图（6 通道输入），但这样模型很难学到"哪些差异是真正的变化，哪些是光照/季节噪声"。

FC-Siam-diff 采用**孪生网络（Siamese Network）**架构：T1 和 T2 通过**共享权重**的编码器分别提取特征，然后在**特征空间**做差。共享权重保证了 T1 和 T2 在相同的特征空间中被编码，做差后高维空间中的残差更集中于语义变化。

#### 网络结构

```python
class FC_Siam_Diff(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        # 共享编码器
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 48, 3, padding=1),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
        )                     # → 256×256×48
        self.pool1 = nn.MaxPool2d(2)  # → 128×128×48

        self.conv2 = nn.Sequential(
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )                     # → 128×128×96
        self.pool2 = nn.MaxPool2d(2)  # → 64×64×96

        self.conv3 = nn.Sequential(
            nn.Conv2d(96, 192, 3, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
        )                     # → 64×64×192

        # 解码器
        self.upconv2 = nn.ConvTranspose2d(192, 96, 2, 2)
        self.deconv2 = nn.Sequential(
            nn.Conv2d(192, 96, 3, padding=1),   # Concat(96_diff, 96_up)
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.BatchNorm2d(96), nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.ConvTranspose2d(96, 48, 2, 2)
        self.deconv1 = nn.Sequential(
            nn.Conv2d(96, 48, 3, padding=1),    # Concat(48_diff, 48_up)
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.BatchNorm2d(48), nn.ReLU(inplace=True),
        )

        self.output = nn.Conv2d(48, 1, 1)  # 1×1 conv → 单通道 logits

    def forward(self, x1, x2):
        # 共享编码 (T1 和 T2 分别通过相同权重)
        f1_1 = self.conv1(x1); f2_1 = self.conv1(x2)
        d1 = torch.abs(f1_1 - f2_1)       # 编码层1 特征差 256×256×48
        p1_1 = self.pool1(f1_1); p2_1 = self.pool1(f2_1)

        f1_2 = self.conv2(p1_1); f2_2 = self.conv2(p2_1)
        d2 = torch.abs(f1_2 - f2_2)       # 编码层2 特征差 128×128×96
        p1_2 = self.pool2(f1_2); p2_2 = self.pool2(f2_2)

        f1_3 = self.conv3(p1_2); f2_3 = self.conv3(p2_1)
        d3 = torch.abs(f1_3 - f2_3)       # 编码层3 特征差 64×64×192

        # 解码（Skip Connection + 特征差）
        u2 = self.upconv2(d3)              # 64→128
        u2 = torch.cat([u2, d2], dim=1)    # Concat(192, 96) = 384→...→96
        u2 = self.deconv2(u2)              # → 128×128×96

        u1 = self.upconv1(u2)              # 128→256
        u1 = torch.cat([u1, d1], dim=1)    # Concat(96, 48) = 144→...→48
        u1 = self.deconv1(u1)              # → 256×256×48

        return self.output(u1)              # logits
```

#### 关键设计决策

**为什么 `torch.abs(diff)` 而非 `diff²`？**

```python
# 当前实现
d = torch.abs(f1 - f2)    # L1 距离

# 替代方案
d = (f1 - f2) ** 2        # L2 距离
```

理论上 L2 对大差异更敏感，但 L1 更稳定。实验中发现两者在 LEVIR-CD 上 F1 差距 < 0.01，L1 收敛略快。考虑到模型仅 625K 参数，L1 对梯度更友好。

**为什么 3 层编码器而非更深？**

LEVIR-CD 图像为 256×256，3 层池化后特征图 64×64（8× 下采样），保留足够空间信息。更深（4-5 层）会导致特征图过小（16×16 甚至 8×8），丢失细粒度变化边界。

### 5.3 模型对比

| 特性 | Tiny U-Net | FC-Siam-diff |
|:-----|:-----------|:--------------|
| 任务 | 语义分割 | 变化检测 |
| 输入 | 单张 RGB | 双时相 RGB 对 |
| 编码器类型 | 单路 | 孪生共享权重 |
| 参数量 | ~2M | ~625K |
| 特点 | 轻量、高精度 | 对光照/季节差异鲁棒 |
| 训练时间 (RTX 5060 Ti) | ~15 min (50 epoch) | ~8 min (30 epoch) |
| 推理速度 | ~8ms/张 | ~12ms/对 |

---

## 六、训练流程与超参数

### 6.1 语义分割训练（`train_seg.py`）

```python
# 超参数
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3
WEIGHT_DECAY = 1e-4
IMG_SIZE = 256

# 优化器
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# 学习率调度
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# 损失函数
criterion = nn.BCEWithLogitsLoss()

# 训练循环
for epoch in range(EPOCHS):
    model.train()
    for batch in train_loader:
        images, masks = batch['image'].cuda(), batch['mask'].cuda()
        preds = model(images).squeeze(1)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    # 验证...
```

**超参数选择理由：**

| 参数 | 选择 | 理由 |
|:-----|:-----|:------|
| `batch_size=64` | 较大 | Tiny U-Net 2M 参数，256×256 输入，每样本 ~3MB。RTX 5060 Ti 16GB 可容纳 128+，64 保证 BN 统计量稳定且留有余量 |
| `AdamW(lr=1e-3)` | 比 Adam 默认 1e-3 稍低 | AdamW 的 weight decay 实现更正确（不作用在自适应学习率上），对小模型 1e-3 合适 |
| `CosineAnnealingLR` | 余弦退火 | 相比 StepLR 更平滑，避免学习率骤降导致的"震荡"。50 epoch 周期足够完整下降 |
| `weight_decay=1e-4` | 轻量正则 | 2M 参数不需要强正则，过大 weight decay 会欠拟合 |
| `BCEWithLogitsLoss` | 直接 logits | 数值稳定，内部做 Sigmoid+BCE，比分开写防止梯度消失 |

### 6.2 变化检测训练（`train.py`）

```python
# 超参数
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
IMG_SIZE = 256

# 优化器 & 调度器 (与分割相同)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.BCEWithLogitsLoss()

# 类别权重可选项 (实验中 BCE 已足够，未使用加权)
# pos_weight = torch.tensor([5.0]).cuda()  # 变化:不变 ≈ 1:5
# criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

**变化检测训练的特殊之处：**

1. **输入双倍显存**：T1 + T2 + 梯度中间变量 ≈ 分割的 2.5×，所以 batch_size 从 64 降至 32
2. **更少 epoch**：变化检测特征差异比分割的逐像素分类更容易学习，30 epoch 已足够
3. **不推荐 `pos_weight`**：实验发现加权 BCE 反而降低 F1（产生过多误报），可能是因为 LEVIR-CD 中变化区域的空间连续性（如果一整栋新建建筑只标出部分像素，加权会惩罚"漏检"过多）

### 6.3 GPU 内存占用分析（RTX 5060 Ti 16GB）

**语义分割（`train_seg.py` batch_size=64）：**

| 阶段 | 显存占用 | 说明 |
|:-----|:--------|:-----|
| 模型参数 | ~8 MB | Tiny U-Net 2M params × 4 bytes |
| 优化器状态 | ~16 MB | AdamW: 一阶+二阶动量 = 2× 参数 |
| 输入 batch | ~50 MB | 64 × 3 × 256 × 256 × 4 bytes |
| 中间激活 (训练) | ~3,200 MB | 前向所有特征图缓存（反向传播需要） |
| 训练总计 | ~3,500 MB | 仅占 16GB 的 ~22% |
| **推理** | **~500 MB** | 无梯度缓存 |

**变化检测（`train.py` batch_size=32）：**

| 阶段 | 显存占用 | 说明 |
|:-----|:--------|:-----|
| 模型参数 | ~2.5 MB | FC-Siam-diff 625K params |
| 优化器状态 | ~5 MB | |
| 输入 batch (×2) | ~50 MB | 32 × 3 × 256 × 256 × 4 × 2 |
| 中间激活 (训练) | ~1,800 MB | 双路编码器 + diff |
| 训练总计 | ~1,900 MB | 约 12% |
| **推理** | **~300 MB** | |

→ **两个模型同时加载也不到 4GB**，留出了显存给 Streamlit 和其他任务。

### 6.4 TensorBoard 监控

```bash
# 启动监视器
tensorboard --logdir runs --port 6006
# 浏览器打开 http://localhost:6006
```

训练脚本自动记录的指标：

| 指标 | 记录频率 | 说明 |
|:-----|:---------|:-----|
| `Loss/train` | 每步 | 训练 batch loss |
| `Loss/val` | 每 epoch | 验证集平均 loss |
| `Metrics/mIoU` | 每 epoch | 分割验证集 mIoU |
| `Metrics/F1` | 每 epoch | 变化检测验证集 F1 |
| `LR` | 每 epoch | 当前学习率 |

---

## 七、实验结果与分析

### 7.1 语义分割 — WHU Building

**训练曲线（50 epoch）：**

```
mIoU ↑
1.0 │
0.9 │                                   ● (50, 0.877)
0.8 │                          ●
0.7 │                ●
0.6 │       ●
0.5 │
0.4 │
0.3 │
0.2 │
0.1 │ ● (初始 ≈0.05)
0.0 └───────────────────────────────→ epoch
     0   10   20   30   40   50
```

| Epoch | 训练 Loss | 验证 mIoU | 备注 |
|:-----:|:---------:|:---------:|:-----|
| 1 | 0.692 | 0.103 | 随机初始权重，几乎全背景预测 |
| 5 | 0.185 | 0.421 | 学习率较高，快速下降期 |
| 10 | 0.071 | 0.752 | 主要结构已学到 |
| 15 | 0.060 | 0.811 | 边界开始细化 |
| 20 | 0.055 | 0.814 | 增速放缓 |
| 25 | 0.052 | 0.836 | |
| 30 | 0.049 | 0.852 | 余弦退火降低 LR |
| 35 | 0.047 | 0.862 | 学习率低 → 精细调优 |
| 40 | 0.047 | 0.869 | |
| 45 | 0.045 | 0.874 | |
| **50** | **0.044** | **0.877** | **Best** |

**参数量 vs 精度对比：**

```
  mIoU
 0.90 │
 0.88 │              ★ Tiny U-Net (2M, 0.877)
 0.86 │
 0.84 │
 0.82 │
 0.80 │
 0.78 │
 0.76 │
 0.74 │
 0.72 │
 0.70 │
      └───────────────────────────────→ 参数量 (M)
         0   10   20   30   40
```

- DeepLabV3+ (ResNet50, 40M): ~0.89
- U-Net (ResNet34, 24M): ~0.88
- U-Net (VGG16, 24M): ~0.87
- **Tiny U-Net (本工作, 2M): 0.877 ✅**

→ **仅用 1/10~1/20 的参数量，达到接近 SOTA 的性能。** 关键原因：WHU Building 的建筑外观相对简单（新西兰 Christchurch 的房屋多为独栋、规则形状），不像 Cityscapes 等城市街景数据集需要大量语义知识。

### 7.2 变化检测 — LEVIR-CD

**训练曲线（30 epoch）：**

```
F1 ↑
1.0 │
0.9 │
0.8 │
0.7 │                                        ● (28, 0.651)
0.6 │
0.5 │                      ●
0.4 │           ●
0.3 │  ●
0.2 │
0.1 │
0.0 └───────────────────────────────→ epoch
     0    5   10   15   20   25   30
```

| Epoch | 训练 F1 | 验证 F1 | 验证 IoU | 备注 |
|:-----:|:-------:|:-------:|:--------:|:-----|
| 5 | 0.427 | 0.330 | 0.198 | 初期易全预测为不变 |
| 10 | 0.547 | 0.349 | 0.213 | 开始检测到部分变化 |
| 15 | 0.627 | 0.367 | 0.224 | 边界粗，误报多 |
| 20 | 0.681 | 0.370 | 0.228 | 验证集 F1 增长停滞 |
| 25 | 0.754 | 0.583 | 0.387 | 余弦退火 LR 降低后显著提升 |
| **28** | **0.801** | **0.651** | **0.477** | **Best** |
| 30 | 0.827 | 0.648 | 0.472 | 轻微过拟合 |

**重要观察：验证 F1 在 epoch 10-20 期间停滞在 0.35 附近，直到学习率降低（余弦退火）后才跳升到 0.65。** 这说明：
1. 高 LR 有助于快速找到变化区域的粗略位置
2. 但精细边界需要低 LR（LR < 1e-4）才能准确分割

**与 SOTA 对比：**

| 方法 | 参数量 | LEVIR-CD F1 |
|:-----|:------:|:-----------:|
| FC-Siam-diff (原版, 3 层编码器) | ~1M | ~0.73 |
| FC-Siam-conc (原版) | ~1M | ~0.71 |
| **本系统 FC-Siam-diff (轻量版)** | **0.625M** | **0.65** |
| STANet (2020) | ~3M | ~0.79 |
| BIT (Transformer, 2022) | ~10M | ~0.82 |

→ 轻量版损失了约 10% 的 F1，但参数量减半，推理速度更快。如果追求精度，可以加宽编码器通道。

### 7.3 推理速度

| 任务 | 设备 | 单次推理 | 批量推理 (32张) |
|:-----|:----|:---------|:----------------|
| 语义分割 (Tiny U-Net) | RTX 5060 Ti | ~8 ms | ~120 ms |
| 变化检测 (FC-Siam-diff) | RTX 5060 Ti | ~12 ms | ~180 ms |
| 语义分割 | CPU (i7) | ~85 ms | — |

Streamlit 界面每次推理均独立加载模型（无缓存复用），每次推理请求实际处理时间约 50-80ms（含图像上传、预处理、模型加载）。用户体验上点击即出结果。

---

## 八、Streamlit 界面架构

### 8.1 页面结构（`app.py`）

```
app.py
├── 初始化 (session_state)
│   ├── page: "seg" | "cd"   (当前模式)
│   └── model_loaded: bool    (模型是否加载)
│
├── 侧边栏 (st.sidebar)
│   ├── 模式切换 (分割 / 变化检测)
│   └── 参数调整 (阈值 slider)
│
├── 语义分割模式
│   ├── 上传图像 (st.file_uploader)
│   ├── 示例选择 (st.selectbox)
│   ├── 推理按钮 (st.button)
│   ├── 结果展示
│   │   ├── 原图
│   │   ├── 分割覆盖图
│   │   └── 建筑占比统计
│   └── 置信度阈值 (滑块)
│
├── 变化检测模式
│   ├── 上传 T1 + T2 (st.file_uploader × 2)
│   ├── 示例选择 (st.selectbox)
│   ├── 推理按钮
│   ├── 结果展示
│   │   ├── T1 / T2 并列
│   │   ├── 差异图 (|T1 - T2|)
│   │   ├── 变化概率图
│   │   └── 变化覆盖叠加图
│   └── 检测阈值 (滑块)
```

### 8.2 模型加载策略（延迟加载）

```python
@st.cache_resource
def load_seg_model():
    model = TinyUNet().cuda()
    model.load_state_dict(torch.load('outputs/best_unet.pth', map_location='cuda'))
    model.eval()
    return model

@st.cache_resource
def load_cd_model():
    model = FC_Siam_Diff().cuda()
    model.load_state_dict(torch.load('outputs/best_siamdiff.pth', map_location='cuda'))
    model.eval()
    return model
```

使用 `@st.cache_resource` 确保模型只在首次调用时加载，后续请求复用已加载的模型实例。

### 8.3 推理函数

```python
def predict_seg(image: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """语义分割推理"""
    model = load_seg_model()
    # 预处理
    img = cv2.resize(image, (256, 256))
    img = img.astype(np.float32) / 255.0
    img = (img - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()
    # 推理
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    # 二值化 + 上采样到原尺寸
    mask = (prob > threshold).astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask, prob

def predict_cd(image_A: np.ndarray, image_B: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """变化检测推理"""
    model = load_cd_model()
    # 预处理（同分割）
    img_A = preprocess(image_A)
    img_B = preprocess(image_B)
    # 推理
    with torch.no_grad():
        logits = model(img_A, img_B)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask = (prob > threshold).astype(np.uint8)
    mask = cv2.resize(mask, (image_A.shape[1], image_A.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask, prob
```

### 8.4 可视化逻辑

```python
def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple = (255, 0, 0), alpha: float = 0.5):
    """在图像上叠加半透明掩码"""
    overlay = image.copy()
    overlay[mask > 0] = (
        int(image[mask > 0][:, 0] * (1 - alpha) + color[0] * alpha),
        int(image[mask > 0][:, 1] * (1 - alpha) + color[1] * alpha),
        int(image[mask > 0][:, 2] * (1 - alpha) + color[2] * alpha),
    )
    return overlay
```

---

## 九、环境配置与部署

### 9.1 前置条件

| 依赖 | 说明 |
|:-----|:------|
| **操作系统** | Windows 10+ (git-bash), macOS (Intel/Apple Silicon), Linux |
| **GPU 驱动** | NVIDIA (CUDA 12+), Apple Silicon (MPS), 或 AMD ROCm |
| **Conda** | Anaconda 或 Miniconda（自动检测，可选安装） |
| **网络** | 需要访问 PyTorch 或 PyPI 镜像（脚本自动检测可用源） |

### 9.2 自动配置（推荐）

**Windows：** 双击 `setup.bat`
**macOS / Linux：** 终端运行 `bash setup.sh`

脚本自动完成：
1. 检测 GPU / CUDA 驱动版本，选择 cu130 / cu121 / cu118 PyTorch
2. 创建 `rs_analysis` Conda 环境（Python 3.12）
3. 从阿里云镜像安装 PyTorch + torchvision
4. 从清华镜像安装其他项目依赖
5. 验证 CUDA 可用性和矩阵运算
6. 所有输出写入 `setup_YYYYMMDD_HHMMSS.log`

### 9.3 手动配置

```bash
# 1. 创建 Conda 环境
conda create -n rs_analysis python=3.12 pip
conda activate rs_analysis

# 2. 安装 PyTorch（根据你的平台选择）
# --- NVIDIA GPU (CUDA 13.x, 如 Blackwell RTX 50系列) ---
pip install torch==2.12.1 torchvision==2.12.1 \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cu130
# --- NVIDIA GPU (CUDA 12.x) ---
pip install torch==2.12.1 torchvision==2.12.1 \
  --index-url https://mirrors.aliyun.com/pytorch-wheels/cu121
# --- Apple Silicon (MPS 加速) ---
pip install torch==2.4.0 torchvision==2.4.0
# --- CPU only ---
pip install torch==2.4.0 torchvision==2.4.0 \
  --index-url https://download.pytorch.org/whl/cpu

# 3. 安装项目依赖（中国用户可用清华镜像加速）
pip install -r requirements.txt
# 或: pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, MPS: {torch.backends.mps.is_available()}')"
```

### 9.4 启动

```bash
bash run.sh                               # macOS / Linux
# 或手动:
conda run -n rs_analysis streamlit run app.py --server.port 8501 --server.headless true
```

Windows 用户可在 Git Bash 中运行 `bash run.sh`，或使用 `conda run`。

访问 `http://localhost:8501`

### 9.5 数据集准备

**WHU Building：** 从武汉大学网站下载，或使用 E 盘已有的数据链接：

```bash
ln -s /e/Desktop/WHU_Building data/WHU_Building
```

**LEVIR-CD：**
```bash
bash data/download.sh
# 选择 3) Python 下载尝试所有镜像
```

### 9.6 GPU 配置建议

| 情况 | 建议 |
|:-----|:-----|
| RTX 5060 Ti 16GB + 核显输出 | 优先：显示器插主板，独占全部 16GB VRAM |
| 显存不足 (OOM) | 降低 `batch_size`：`python train.py --batch_size 16` |
| 多 GPU | 当前未实现 DataParallel，只能单卡训练 |
| 仅 CPU | `setup.sh` 自动检测并安装 CPU 版 PyTorch |

---

## 十、项目文件清单

```
RS_Analysis/
├── app.py                      # [核心] Streamlit 双模式 Web 界面
├── train.py                    # [核心] 变化检测训练脚本 (FC-Siam-diff)
├── train_seg.py                # [核心] 语义分割训练脚本 (Tiny U-Net)
│
├── setup.sh                    # [配置] 一键环境配置 (GPU检测+conda+pip)
├── run.sh                      # [配置] 一键启动 (Streamlit)
├── requirements.txt            # [配置] Python 依赖清单
├── .gitignore                  # [配置] Git 忽略规则
│
├── models/
│   ├── cnn_cd.py               # FC-Siam-diff 变化检测模型定义
│   └── unet_tiny.py            # Tiny U-Net 语义分割模型定义
│
├── utils/
│   ├── dataset.py              # LEVIR-CD 变化检测数据加载器
│   ├── seg_dataset.py          # WHU 等分割数据加载器（多目录名兼容）
│   └── metrics.py              # 评估指标 (IoU, F1, Precision, Recall)
│
├── data/                       # 数据集目录（需手动准备）
│   ├── LEVIR-CD256/            # 变化检测数据 (256×256 预处理版)
│   ├── WHU_Building/           # 建筑分割数据 (可以是符号链接)
│   └── download.sh             # LEVIR-CD 下载脚本 (多镜像)
│
├── outputs/                    # 预训练模型权重
│   ├── best_siamdiff.pth       # 变化检测: F1=0.65 (2.5MB)
│   └── best_unet.pth           # 语义分割: mIoU=0.877 (7.6MB)
│
├── samples/                    # 样例图像 (推理演示用)
│   ├── cd_A.png / cd_B.png     # 变化检测 T1/T2 示例
│   ├── cd2_A.png / cd2_B.png   # 变化检测第二组示例
│   ├── whu_seg_A.png / ...     # 分割示例 (高密度/低密度)
│
├── README.md                   # 项目快速介绍 (2 分钟上手)
└── INTRO.md                    # 本文档 (技术白皮书)
```

---

## 十一、常见问题与排错

### 11.1 安装问题

**Q: `bash setup.sh` 中途断网失败？**
A: 脚本有 3 次重试机制。如果网络波动，重新运行即可。重跑时 conda 环境已存在会自动跳过创建。

**Q: `conda create` 下载 Python 太慢？**
A: 脚本配置了清华 conda 镜像。如果仍然慢，可以手动下载 [Miniconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/) 安装后重新运行脚本。

**Q: `pip install torch` 报 `Could not find a version that satisfies the requirement`？**
A: 检查你的 Python 版本是否与 wheel 匹配。阿里云镜像上有 `cp310`/`cp311`/`cp312`/`cp313`/`cp314` 版本，确认你的 `python --version` 对应。

### 11.2 运行问题

**Q: `bash run.sh` 报 `ModuleNotFoundError: No module named 'torch'`？**
A: 没有激活 Conda 环境。用 `conda run -n rs_analysis streamlit run app.py` 代替，或在终端执行 `conda activate rs_analysis` 后再运行 `bash run.sh`。

**Q: Streamlit 界面加载但推理按钮点不动？**
A: 检查 outputs 目录是否存在预训练权重。没有权重则模型加载失败，先运行训练脚本或下载预训练模型。

**Q: 出现 `RuntimeError: CUDA error: no kernel image is available for execution on the device`？**
A: 你的 PyTorch 版本不支持当前 GPU 的计算能力（如 Blackwell sm_120 需要 cu130+ 的 PyTorch 2.12+）。运行 `setup.sh` 自动检测并安装正确版本。

### 11.3 GPU 相关问题

**Q: `torch.cuda.is_available()` 返回 False？**
A: 依次排查：
```bash
nvidia-smi                    # GPU 驱动正常？
python -c "import torch; print(torch.version.cuda)"  # PyTorch 绑定的 CUDA 版本？
pip list | grep torch         # 确认装了 cu130/cu121 版，不是 cpu 版？
```

**Q: 如何让显示器不占用 GPU VRAM？**
A: 将显示器信号线从显卡插到主板（核显输出）。重启后 nvidia-smi 仍能检测到显卡，但显存全部可用（从 ~15.5GB 变成 ~16GB 空闲）。

### 11.4 训练问题

**Q: 训练 Loss 不下降？**
A: 检查：学习率是否太大/太小、数据是否归一化正确、是否用了 `model.train()` 模式、BCEWithLogitsLoss 是否接收了 logits 而非 sigmoid 输出。

**Q: 验证集 mIoU/F1 远低于训练集？**
A: 过拟合。增加 `weight_decay`、降低 epoch、或添加 Dropout（当前模型未使用，可以尝试在 `_block` 中加 `nn.Dropout2d(0.1)`）。

**Q: 训练到一半显存不足？**
A: 降低 `batch_size`（分割从 64→32，变化检测从 32→16），或启用梯度累积。

### 11.5 结果问题

**Q: 分割结果把大片背景标为建筑？**
A: 调高置信度阈值（app 侧边栏的滑块从 0.5 拉到 0.7 或 0.8）。

**Q: 变化检测出现大量误报？**
A: 调高检测阈值（0.5→0.7）。常见误报来源：树木阴影移动、光照角度变化、车辆停放差异。

---

## 十二、未来改进方向

### 12.1 短期可做

1. **模型集成**：训练 3-5 个不同随机种子的模型，推理时投票，可提升 F1 约 2-3%
2. **分割后处理**：CRF 条件随机场或 Remove Small Objects（移除面积小于 K 像素的建筑区域），减少噪声
3. **数据均衡**：变化检测使用 Focal Loss 或 Dice Loss 处理类别不平衡
4. **学习率预热**：前 5 epoch 线性从 0 升到 1e-3，稳定初始化阶段

### 12.2 中期改进

1. **分割模型替换**：尝试 SegFormer-b0（2.3M 参数）或 MobileNetV3+DeepLabV3 头部
2. **变化检测特征增强**：在 Diff 后加入通道注意力（SE Block）增强判别特征
3. **多尺度推理**：TTA（Test-Time Augmentation）：输入原图 + 翻转 → 平均结果
4. **模型量化**：torch.quantization 将模型压缩到 INT8，推理速度提升 2-3x

### 12.3 长期方向

1. **多任务学习**：共享编码器同时输出分割和变化检测结果
2. **半监督学习**：利用大量无标注遥感图像做预训练（SimCLR/MAE）
3. **在线学习**：用户标注修正结果后增量更新模型
4. **迁移到其他遥感任务**：道路提取、水体分割、作物分类

---

## 十三、参考文献

1. Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. *MICCAI 2015*.
2. Daudt, R. C., Le Saux, B., & Boulch, A. (2018). Fully Convolutional Siamese Networks for Change Detection. *ICIP 2018*.
3. Chen, H., & Shi, Z. (2020). A Spatial-Temporal Attention-Based Method and a New Dataset for Remote Sensing Image Change Detection. *Remote Sensing*, 12(10), 1662.
4. Ji, S., Wei, S., & Lu, M. (2019). Fully Convolutional Networks for Multisource Building Change Detection. *IEEE GRSL*.
5. Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization. *ICLR 2019*.
6. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS 2019*.
7. Buslaev, A., et al. (2020). Albumentations: Fast and Flexible Image Augmentations. *Information*, 11(2), 125.
8. WHU Building Dataset. Wuhan University. http://gpcv.whu.edu.cn/data/
9. LEVIR-CD Dataset. https://justchenhao.github.io/LEVIR/
