#!/bin/bash
# 环境一键配置脚本
# Usage: bash setup.sh

set -e

echo "========================================="
echo "  🛰️ 遥感图像分析平台 - 环境配置"
echo "========================================="
echo ""

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 检测 Python
PYTHON=""
for p in python3 python; do command -v "$p" &>/dev/null && { PYTHON="$p"; break; }; done
if [ -z "$PYTHON" ]; then
    [ -f "/d/environment/anaconda3/python.exe" ] && PYTHON="/d/environment/anaconda3/python.exe"
fi
if [ -z "$PYTHON" ]; then
    echo "❌ 未找到 Python，请安装 Python 3.10+"
    exit 1
fi
echo "✅ Python: $($PYTHON --version)"

# 安装依赖
echo ""
echo "📦 安装依赖..."
$PYTHON -m pip install -r requirements.txt -q
echo "✅ 依赖安装完成"

# 检查数据集
echo ""
echo "📂 检查数据集..."
HAS_CD=0; HAS_SEG=0
[ -d "data/LEVIR-CD256" ] && HAS_CD=1 && echo "  ✅ 变化检测: LEVIR-CD256"
[ -d "data/WHU_Building" ] && HAS_SEG=1 && echo "  ✅ 语义分割: WHU_Building"
[ $HAS_CD -eq 0 ] && echo "  ⚠️  变化检测数据集缺失 (data/LEVIR-CD256/)"
[ $HAS_SEG -eq 0 ] && echo "  ⚠️  语义分割数据集缺失 (data/WHU_Building/)"

# 检查模型
echo ""
echo "🧠 检查已训练模型..."
[ -f "outputs/best_siamdiff.pth" ] && echo "  ✅ 变化检测: best_siamdiff.pth" || echo "  ⚠️  未找到 (运行 python train.py)"
[ -f "outputs/best_unet.pth" ] && echo "  ✅ 语义分割: best_unet.pth" || echo "  ⚠️  未找到 (运行 python train_seg.py)"

echo ""
echo "========================================="
echo "  🚀 启动命令: bash run.sh"
echo "========================================="
