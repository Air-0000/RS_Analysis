#!/bin/bash
# 遥感图像分析平台 — 一键启动
# 使用 conda run 自动调用 rs_analysis 环境
#
# Usage: bash run.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════"
echo "  🛰️  遥感图像分析平台"
echo "  功能:  语义分割  |  变化检测"
echo "═══════════════════════════════════════════════"

# 检查 Conda 环境
ENV_NAME="rs_analysis"
if ! conda env list 2>/dev/null | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "❌ 环境 '$ENV_NAME' 不存在，请先运行: bash setup.sh"
    exit 1
fi

# 检查模型
echo ""
echo "📦 模型状态:"
CONDA_RUN="conda run -n $ENV_NAME --no-capture-output"
[ -f "outputs/best_siamdiff.pth" ] \
    && echo "  ✅ 变化检测: best_siamdiff.pth" \
    || echo "  ⚠️  变化检测: 未训练 (conda run -n $ENV_NAME python train_cd.py)"
[ -f "outputs/best_unet.pth" ] \
    && echo "  ✅ 语义分割: best_unet.pth" \
    || echo "  ⚠️  语义分割: 未训练 (conda run -n $ENV_NAME python train_segment.py)"

echo ""
echo "🚀 启动界面..."
exec $CONDA_RUN streamlit run app.py --server.port 8501 --server.headless true
