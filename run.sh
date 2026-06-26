#!/bin/bash
# 遥感图像分析平台 - 一键启动

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# 找 Python
PYTHON=""
for p in python3 python; do command -v "$p" &>/dev/null && { PYTHON="$p"; break; }; done
[ -z "$PYTHON" ] && [ -f "/d/environment/anaconda3/python.exe" ] && PYTHON="/d/environment/anaconda3/python.exe"
[ -z "$PYTHON" ] && echo "❌ 未找到 Python" && exit 1

echo "========================================="
echo "  🛰️ 遥感图像分析平台"
echo "========================================="
echo ""
echo "  功能:"
echo "  📷 语义分割 - 识别地物类别"
echo "  🔄 变化检测 - 标出变化区域"
echo ""

# 检查模型
HAS_CD=0; HAS_SEG=0
[ -f "outputs/best_siamdiff.pth" ] && HAS_CD=1
[ -f "outputs/best_unet.pth" ] && HAS_SEG=1

echo "📦 模型状态:"
[ $HAS_CD -eq 1 ] && echo "  ✅ 变化检测: best_siamdiff.pth" || echo "  ❌ 变化检测: 未训练 (python train.py)"
[ $HAS_SEG -eq 1 ] && echo "  ✅ 语义分割: best_unet.pth" || echo "  ❌ 语义分割: 未训练 (python train_seg.py)"
echo ""

echo "🚀 启动界面..."
NO_ALBUMENTATIONS_UPDATE=1 $PYTHON -m streamlit run app.py --server.port 8501 --server.headless true
