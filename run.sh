#!/bin/bash
# 遥感图像分析平台 — 一键启动
# 自动检测 base / conda 环境，优先使用 base
#
# Usage: bash run.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════"
echo "  🛰️  遥感图像分析平台"
echo "  功能:  语义分割  |  变化检测"
echo "═══════════════════════════════════════════════"

# 检测可用 Python 环境：先 base 后 conda
PY_RUN=""
# 尝试 Anaconda base
[ -f "/d/environment/anaconda3/python.exe" ] && \
    KMP_DUPLICATE_LIB_OK=TRUE /d/environment/anaconda3/python.exe -c "import torch" 2>/dev/null && \
    PY_RUN="/d/environment/anaconda3/python.exe -m streamlit"
# 尝试 conda rs_analysis
if [ -z "$PY_RUN" ] && conda env list 2>/dev/null | grep -q "^rs_analysis[[:space:]]"; then
    PY_RUN="conda run -n rs_analysis --no-capture-output streamlit"
fi

if [ -z "$PY_RUN" ]; then
    echo "❌ 未找到可用 Python 环境（需 torch）"
    echo "   请先运行: bash setup.sh"
    exit 1
fi

# 检查模型
echo ""
echo "📦 模型状态:"
[ -f "outputs/best_siamdiff.pth" ] \
    && echo "  ✅ 变化检测: best_siamdiff.pth" \
    || echo "  ⚠️  变化检测: 未训练 (python train_cd.py)"
[ -f "outputs/best_unet.pth" ] \
    && echo "  ✅ 语义分割: best_unet.pth" \
    || echo "  ⚠️  语义分割: 未训练 (python train_segment.py)"

echo ""
echo "🚀 启动界面..."
exec $PY_RUN run app.py --server.port 8501 --server.headless true
