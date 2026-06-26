#!/bin/bash
# 遥感图像分析平台 — 环境一键配置
# 自动检测 GPU/CUDA，选择匹配的 PyTorch 版本
# 所有输出写入 setup_<时间戳>.log
#
# Usage: bash setup.sh

set -euo pipefail

PIP_TIMEOUT=120

# ── 日志 ──────────────────────────────────────────────
LOG_FILE="setup_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo "═══════════════════════════════════════════════"
echo "  🛰️  遥感图像分析平台 — 环境配置"
echo "  日志: $LOG_FILE"
echo "═══════════════════════════════════════════════"
echo ""

# ── 硬件检测 ──────────────────────────────────────
echo "🔍 检测硬件..."
GPU_NAME=""; CUDA_MAJOR=""; TORCH_INDEX=""; TORCH_VERSION=""

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    CUDA_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
    CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "")
    
    if [ -n "$GPU_NAME" ]; then
        echo "  GPU: $GPU_NAME"
        echo "  CUDA Driver: $CUDA_VER"
        echo "  Compute Capability: $CC_RAW"
        
        CUDA_MAJOR=$(echo "$CUDA_VER" | sed 's/\..*//')
        
        if [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
            TORCH_INDEX="https://mirrors.aliyun.com/pytorch-wheels/cu130"
            TORCH_VERSION="2.12.1"
            echo "  → 选用: PyTorch ${TORCH_VERSION}+cu130"
        elif [ "$CUDA_MAJOR" -ge 12 ] 2>/dev/null; then
            TORCH_INDEX="https://mirrors.aliyun.com/pytorch-wheels/cu121"
            TORCH_VERSION="2.12.1"
            echo "  → 选用: PyTorch ${TORCH_VERSION}+cu121"
        else
            echo "  ⚠️  CUDA $CUDA_VER 较旧，尝试 cu118"
            TORCH_INDEX="https://mirrors.aliyun.com/pytorch-wheels/cu118"
            TORCH_VERSION="2.4.0"
        fi
    else
        echo "  ⚠️  nvidia-smi 检测失败"
    fi
fi

if [ -z "$TORCH_INDEX" ]; then
    echo "  未检测到 NVIDIA GPU，使用 CPU-only 版本"
    TORCH_INDEX="https://download.pytorch.org/whl/cpu"
    TORCH_VERSION="2.4.0"
fi
echo ""

# ── Conda ──────────────────────────────────────────
echo "📦 检测 Conda..."
CONDA_BASE=""
if command -v conda &>/dev/null; then
    CONDA_BASE=$(conda info --base 2>/dev/null || echo "")
fi

if [ -z "$CONDA_BASE" ]; then
    echo "  ⚠️  未找到 Conda，安装 Miniconda..."
    INSTALLER="/tmp/Miniconda3-latest-Windows-x86_64.exe"
    curl -fsSL -o "$INSTALLER" \
      "https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    bash "$INSTALLER" /S /D="$HOME/miniconda3"
    export PATH="$HOME/miniconda3/Scripts:$HOME/miniconda3:$PATH"
    CONDA_BASE="$HOME/miniconda3"
    echo "  ✅ Miniconda 已安装到 $CONDA_BASE"
else
    echo "  ✅ Conda: $CONDA_BASE"
fi
echo ""

ENV_NAME="rs_analysis"

# ── 创建环境（只用 pip，避过 conda 镜像问题）────────
echo "🔧 创建 Conda 环境: $ENV_NAME"
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "  环境已存在，跳过创建"
else
    # 用 conda 只装 python+pip，不走额外 channel
    for i in 1 2 3; do
        echo "  尝试 ($i/3)..."
        if conda create -y -n "$ENV_NAME" python=3.12 pip \
            --channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ 2>/dev/null; then
            echo "  ✅ 创建成功"
            break
        fi
        [ "$i" -eq 3 ] && echo "  ❌ 创建失败，请检查网络后重试" && exit 1
        echo "  网络超时，5 秒后重试..."
        sleep 5
    done
fi
echo ""

CONDA_RUN="conda run -n $ENV_NAME --no-capture-output"

# ── 安装 PyTorch ──────────────────────────────────
echo "📥 安装 PyTorch $TORCH_VERSION..."
TV_VERSION=$(echo "$TORCH_VERSION" | sed 's/\.[0-9]*$//')
for i in 1 2 3; do
    echo "  尝试 ($i/3)..."
    if $CONDA_RUN pip install \
        --index-url "$TORCH_INDEX" \
        --trusted-host mirrors.aliyun.com \
        --timeout "$PIP_TIMEOUT" \
        torch=="$TORCH_VERSION" torchvision=="$TV_VERSION" \
        --quiet 2>&1 | tail -3; then
        break
    fi
    [ "$i" -eq 3 ] && echo "  ⚠️  PyTorch 安装失败，跳过（可手动安装）" && break
    sleep 3
done
echo ""

# ── 安装其他依赖 ──────────────────────────────────
echo "📦 安装项目依赖..."
for i in 1 2 3; do
    echo "  尝试 ($i/3)..."
    if $CONDA_RUN pip install \
        -r requirements.txt \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --trusted-host pypi.tuna.tsinghua.edu.cn \
        --timeout "$PIP_TIMEOUT" \
        --quiet 2>&1 | tail -3; then
        echo "  ✅ 项目依赖安装完成"
        break
    fi
    [ "$i" -eq 3 ] && echo "  ⚠️  部分依赖安装失败，跳过（可手动安装）" && break
    sleep 3
done
echo ""

# ── 验证 GPU ──────────────────────────────────────
echo "🧪 验证安装..."
$CONDA_RUN python -c "
import torch, sys
print(f'PyTorch: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    x = torch.randn(100, 100).cuda()
    y = x @ x.T
    print(f'CUDA 矩阵运算: ✅ ({(y.sum().item()):.2f})')
else:
    print('⚠️  GPU 不可用，使用 CPU')
" 2>&1 || echo "  ⚠️  验证失败（可能 torch 未正确安装）"

echo ""
echo "───────────────────────────────────────────────"
echo "  🚀 启动: bash run.sh"
echo "  变化检测训练: conda run -n $ENV_NAME python train.py"
echo "  语义分割训练: conda run -n $ENV_NAME python train_seg.py"
echo "  查看日志: cat $LOG_FILE"
echo "───────────────────────────────────────────────"
