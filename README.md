# 🛰️ RS_Analysis — 轻量级遥感图像智能分析系统

> **Remote Sensing Image Analysis: Building Segmentation & Change Detection**  
> 基于 PyTorch 与 Streamlit 的双功能遥感分析平台，参数总量仅 2.6M，兼顾边缘部署实时性与检测精度。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12%2B-ee4c2c.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B.svg)](https://streamlit.io/)

---

## 📌 核心功能与性能指标

系统提供两大高空遥感与航拍图像核心分析能力：

| 任务 | 核心模型 | 参数量 | 评测数据集 | 核心性能指标 |
| :--- | :--- | :---: | :--- | :--- |
| 📷 **建筑物语义分割** | **Tiny U-Net** | ~2.0M | WHU Building Dataset | **mIoU: 87.7%** / F1: 93.4% / Acc: 97.8% |
| 🔄 **双时相变化检测** | **FC-Siam-diff** | ~625K | LEVIR-CD Benchmark | **F1: 65.0%** / Precision: 68.2% / Recall: 62.1% |

---

## 🏗️ 模型架构特点

### 1. 建筑语义分割 (Tiny U-Net)
* 4 层编码-解码对称架构，轻量化特征通道（`[32, 64, 128, 256]`）；
* 跨层拼接跳跃连接（Skip-Connection），精准保留遥感高分辨率影像中的建筑物边缘几何纹理；
* 结合 BCE + Dice Loss 混合损失函数，显著缓解高空航拍场景中建筑物与复杂背景样本不均衡的问题。

### 2. 遥感变化检测 (FC-Siam-diff)
* **全卷积孪生网络（Fully Convolutional Siamese）**：前后双时相图像共享编码器权重，提取多尺度时序空间表征；
* **差分融合解码器**：在多尺度特征图上执行绝对差分运算（Absolute Difference），捕捉建筑物新建、拆除与地表变化边界；
* 极小参数量（625KB），支持无缝集成至无人机或边缘计算设备。

---

## 🖥️ Streamlit 交互系统

提供开箱即用的现代化 Web 交互界面：
* **单图语义分割工作台**：上传遥感正射影像，实时渲染建筑物二值掩码与半透明热力叠加图，支持导出分割统计结果；
* **双时相变化检测工作台**：同时上传 Time-1（前期）与 Time-2（后期）航拍瓦片，动态滑块调整变化判定门限，实时对比差异区域。

---

## 🚀 快速启动

### 自动化一键启动
* **Linux / macOS**:
  ```bash
  bash setup.sh    # 自动探测 CUDA 环境并安装适配 PyTorch 与依赖
  bash run.sh      # 启动 Streamlit Web 界面 (http://localhost:8501)
  ```
* **Windows**:
  双击根目录下的 `setup.bat` 即可自动拉起环境并运行。

### 手动安装与训练
```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Web 界面
streamlit run app.py

# 训练建筑物语义分割模型
python train_segment.py --epochs 50 --batch_size 16 --lr 1e-3

# 训练双时相变化检测模型
python train_cd.py --epochs 60 --batch_size 8 --lr 5e-4
```

---

## 📂 仓库结构

```
├── app.py                  # Streamlit 交互式 Web 分析应用入口
├── train_segment.py        # 建筑物语义分割训练与验证脚本
├── train_cd.py             # 双时相变化检测训练与评测脚本
├── models/
│   ├── unet_tiny.py        # Tiny U-Net 轻量分割网络架构
│   └── cnn_cd.py           # FC-Siam-diff 孪生差分变化检测网络
├── utils/
│   ├── dataset.py          # LEVIR-CD 双时相数据集管道
│   ├── seg_dataset.py      # WHU 分割数据集处理与几何增强
│   └── metrics.py          # mIoU / Dice / F1 / Precision / Recall 评估
├── samples/                # 示例输入测试遥感影像
├── INTRO.md                # 完整技术背景与消融实验详解
├── REPORT.md               # 课题技术与设计实验报告
└── REPORT.tex              # 学术报告 LaTeX 源码
```

---

*详细设计推导、参数选择论证与各组消融实验请查阅 [INTRO.md](INTRO.md) 与 [REPORT.md](REPORT.md)。*
