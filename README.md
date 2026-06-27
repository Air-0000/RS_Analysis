# 🛰️ RS_Analysis — 遥感图像分析

> 《人工智能基础B》课程作业 · 双功能遥感分析系统

一句话：上传图片，跑分割或变化检测。两个模型加一起 2.6M 参数。

| 功能 | 模型 | 参数 | 精度 |
|:----|:----|:---:|:----:|
| 📷 **建筑分割** | Tiny U-Net | ~2M | mIoU 87.7% |
| 🔄 **变化检测** | FC-Siam-diff | 625K | F1 65% |

---

## 快速开始

**Windows：** 双击 `setup.bat`（会自己找 Git Bash）
**macOS / Linux：** 终端里跑

```bash
bash setup.sh    # 装环境（自动检测 GPU，选对应 PyTorch）
bash run.sh      # 启动界面 → http://localhost:8501
```

## 项目结构

```
├── app.py                  # Streamlit 界面
├── train_cd.py             # 变化检测训练
├── train_segment.py        # 语义分割训练
├── models/
│   ├── cnn_cd.py           # FC-Siam-diff
│   └── unet_tiny.py        # Tiny U-Net
├── utils/
│   ├── dataset.py          # LEVIR-CD 数据加载
│   ├── seg_dataset.py      # 分割数据加载
│   └── metrics.py          # 评估指标
├── setup.sh                # 环境配置
├── setup.bat               # Windows 双击入口
└── run.sh                  # 一键启动
```

详细的模型结构、训练日志、实验数据、架构图 → **[INTRO.md](INTRO.md)**
