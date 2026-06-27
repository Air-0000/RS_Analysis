# 🛰️ 遥感图像分析平台（Remote Sensing Analysis）

> **《人工智能基础B》课程大作业**

## 项目简介

一个双功能的遥感图像分析系统，涵盖计算机视觉两大经典任务：

| 功能 | 模型 | 参数 | 精度 |
|:----|:----|:---:|:----:|
| 📷 **建筑分割** | Tiny U-Net | ~2M | **mIoU 87.7%** |
| 🔄 **变化检测** | FC-Siam-diff | 625K | **F1 65%** |

配备 Streamlit 交互界面，支持图像上传、实时推理、结果可视化。

## 快速开始

**Windows 用户：** 双击 `setup.bat`（会自动检测 Git Bash）
**macOS / Linux 用户：** 终端运行：

```bash
# 1. 配置环境
bash setup.sh

# 2. 启动界面
bash run.sh
# 浏览器打开 http://localhost:8501
```

## 模型架构

### Tiny U-Net（建筑分割）

轻量级 U-Net，编码器 16→32→64→128 通道，解码器带 Skip Connection，输出建筑/非建筑二值图。

### FC-Siam-diff（变化检测）

T1 和 T2 分别通过共享权重的编码器，在特征空间做差，解码重建变化图。

## 数据集

- **WHU Building Dataset** — 卫星建筑分割，4736 训练 / 1036 验证
- **LEVIR-CD** — 遥感变化检测，445 训练 / 64 验证 / 128 测试

## 项目结构

```
├── models/
│   ├── cnn_cd.py          # FC-Siam-diff 变化检测
│   └── unet_tiny.py       # Tiny U-Net 语义分割
├── utils/
│   ├── dataset.py          # LEVIR-CD 数据加载
│   ├── seg_dataset.py      # 分割数据加载（支持多目录名）
│   └── metrics.py          # 评估指标
├── app.py                 # Streamlit 双模式界面
├── train_cd.py               # 变化检测训练
├── train_segment.py           # 语义分割训练
├── setup.sh               # 环境配置
├── run.sh                 # 一键启动
└── outputs/
    ├── best_siamdiff.pth   # 变化检测模型
    └── best_unet.pth       # 分割模型
```

## 技术亮点

- 两个模型总参数仅 **~2.6M**，在单 GPU 上可快速训练
- 分割模型无预训练，达到 87.7% mIoU（可与预训练大模型媲美）
- 统一 Streamlit 界面，功能切换、实时推理
- 支持 PNG/JPG/TIFF 等多种图像格式
