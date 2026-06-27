#!/bin/bash
# 遥感图像分析平台 — 环境一键配置
# 自动检测 GPU/CUDA，选择匹配的 PyTorch 版本
# 自动选择可用镜像源（中国优先，全球降级）
# 所有输出写入 setup_<时间戳>.log
#
# Usage: bash setup.sh

set -euo pipefail

PIP_TIMEOUT=120
ENV_NAME="rs_analysis"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

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

# ── 镜像源检测 ────────────────────────────────────
# 国内优先走阿里云/清华，连不上则降级到官方源
echo "🌐 检测镜像源..."

PIP_INDEX="https://pypi.org/simple"
PIP_TRUSTED=""
TORCH_FALLBACK_INDEX="https://download.pytorch.org/whl"

# 测试清华 PyPI
if curl -sI --max-time 5 "https://pypi.tuna.tsinghua.edu.cn/simple" >/dev/null 2>&1; then
    PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
    PIP_TRUSTED="--trusted-host pypi.tuna.tsinghua.edu.cn"
    echo "  📦 PyPI 镜像: 清华 (Tsinghua)"
else
    echo "  📦 PyPI 源: 官方 (pypi.org)"
fi

# 测试阿里云 PyTorch
TORCH_INDEX=""
if curl -sI --max-time 5 "https://mirrors.aliyun.com/pytorch-wheels/cu130/" >/dev/null 2>&1; then
    TORCH_MIRROR="aliyun"
    echo "  🔥 PyTorch 镜像: 阿里云"
else
    TORCH_MIRROR="official"
    echo "  🔥 PyTorch 源: 官方 (download.pytorch.org)"
fi

# 测试清华 Conda
CONDA_CHANNEL=""
if curl -sI --max-time 5 "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/" >/dev/null 2>&1; then
    CONDA_CHANNEL="--channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/"
    echo "  📥 Conda 镜像: 清华"
fi
echo ""

# ── 硬件检测 ──────────────────────────────────────
echo "🔍 检测硬件..."
GPU_NAME=""; CUDA_MAJOR=""; TORCH_INDEX_URL=""; TORCH_VERSION=""

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    
    if [ -n "$GPU_NAME" ]; then
        # 从 nvidia-smi 顶部信息中提取 CUDA UMD 版本
        # 输出: "CUDA UMD Version: 13.3"
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA UMD Version" | sed 's/.*CUDA UMD Version: *//' | head -1 || echo "")
        CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "")
        
        echo "  GPU: $GPU_NAME"
        [ -n "$CUDA_VER" ] && echo "  CUDA: $CUDA_VER"
        [ -n "$CC_RAW" ] && echo "  Compute Capability: $CC_RAW"
        
        if [ -n "$CUDA_VER" ]; then
            CUDA_MAJOR=$(echo "$CUDA_VER" | sed 's/\..*//')
        else
            # 降级：从 driver version 反推（driver 525+ = CUDA 12, 610+ = CUDA 13）
            DRV_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
            DRV_MAJOR=$(echo "$DRV_VER" | sed 's/\..*//')
            [ "$DRV_MAJOR" -ge 610 ] 2>/dev/null && CUDA_MAJOR=13 || true
            [ "$DRV_MAJOR" -ge 525 ] 2>/dev/null && [ "$DRV_MAJOR" -lt 610 ] 2>/dev/null && CUDA_MAJOR=12 || true
        fi
        
        # 选 PyTorch 版本
        if [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
            if [ "$TORCH_MIRROR" = "aliyun" ]; then
                TORCH_INDEX_URL="https://mirrors.aliyun.com/pytorch-wheels/cu130"
            else
                TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cu130"
            fi
            TORCH_VERSION="2.12.1"
            echo "  → 选用: PyTorch ${TORCH_VERSION}+cu130"
        elif [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 12 ] 2>/dev/null; then
            if [ "$TORCH_MIRROR" = "aliyun" ]; then
                TORCH_INDEX_URL="https://mirrors.aliyun.com/pytorch-wheels/cu121"
            else
                TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cu121"
            fi
            TORCH_VERSION="2.12.1"
            echo "  → 选用: PyTorch ${TORCH_VERSION}+cu121"
        else
            echo "  ⚠️  CUDA 版本较旧，尝试 cu118"
            if [ "$TORCH_MIRROR" = "aliyun" ]; then
                TORCH_INDEX_URL="https://mirrors.aliyun.com/pytorch-wheels/cu118"
            else
                TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cu118"
            fi
            TORCH_VERSION="2.4.0"
        fi
    else
        echo "  ⚠️  nvidia-smi 检测失败（可能无权限或无驱动）"
    fi
else
    echo "  未找到 nvidia-smi（可能无 NVIDIA GPU）"
fi

if [ -z "$TORCH_INDEX_URL" ]; then
    echo "  → 使用 CPU-only 版本"
    TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cpu"
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
    INSTALLER="$TMP/Miniconda3-latest-Windows-x86_64.exe"
    [ -z "$TMP" ] && INSTALLER="$HOME/Miniconda3-latest-Windows-x86_64.exe"
    
    curl -fsSL -o "$INSTALLER" "$MINICONDA_URL"
    echo "  运行安装程序 (静默安装)..."
    
    # Windows 下 .exe 不能用 bash 执行，用 cmd 或 start
    if [ -f "$INSTALLER" ]; then
        cmd //c "$INSTALLER" /S "/D=$HOME/miniconda3" 2>/dev/null || \
        start /wait "" "$INSTALLER" /S "/D=$HOME/miniconda3" 2>/dev/null || \
        "$INSTALLER" /S "/D=$HOME/miniconda3"
        rm -f "$INSTALLER"
    fi
    
    # 添加到 PATH
    export PATH="$HOME/miniconda3/Scripts:$HOME/miniconda3:$PATH"
    CONDA_BASE="$HOME/miniconda3"
    echo "  ✅ Miniconda 已安装到 $CONDA_BASE"
else
    echo "  ✅ Conda: $CONDA_BASE"
fi
echo ""

# ── 创建环境 ──────────────────────────────────────
echo "🔧 创建 Conda 环境: $ENV_NAME"
if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
    echo "  环境已存在，跳过创建"
else
    for i in 1 2 3; do
        echo "  尝试 ($i/3)..."
        # 有清华镜像就用，否则用默认
        if [ -n "$CONDA_CHANNEL" ]; then
            conda create -y -n "$ENV_NAME" python=3.12 pip $CONDA_CHANNEL 2>/dev/null && break
        else
            conda create -y -n "$ENV_NAME" python=3.12 pip 2>/dev/null && break
        fi
        [ "$i" -eq 3 ] && echo "  ❌ 创建失败，请检查网络后重试" && exit 1
        echo "  网络超时，5 秒后重试..."
        sleep 5
    done
    echo "  ✅ 创建成功"
fi
echo ""

CONDA_RUN="conda run -n $ENV_NAME --no-capture-output"
$CONDA_RUN pip cache purge 2>/dev/null || true

# ── 安装 PyTorch ──────────────────────────────────
echo "📥 安装 PyTorch $TORCH_VERSION..."
TV_VERSION=$(echo "$TORCH_VERSION" | sed 's/\.[0-9]*$//')

# 构造 pip extra args
PIP_EXTRA="--timeout $PIP_TIMEOUT --quiet"
if [ "$TORCH_MIRROR" = "aliyun" ]; then
    PIP_EXTRA="$PIP_EXTRA --trusted-host mirrors.aliyun.com"
fi

for i in 1 2 3; do
    echo "  尝试 ($i/3)..."
    if $CONDA_RUN pip install \
        --index-url "$TORCH_INDEX_URL" \
        $PIP_EXTRA \
        torch=="$TORCH_VERSION" torchvision=="$TV_VERSION" 2>&1 | tail -3; then
        echo "  ✅ PyTorch 安装完成"
        break
    fi
    [ "$i" -eq 3 ] && echo "  ⚠️  PyTorch 安装失败，跳过（可手动安装）" && break
    echo "  3 秒后重试..."
    sleep 3
done
echo ""

# ── 安装其他依赖 ──────────────────────────────────
echo "📦 安装项目依赖..."
for i in 1 2 3; do
    echo "  尝试 ($i/3)..."
    if $CONDA_RUN pip install \
        -r requirements.txt \
        -i "$PIP_INDEX" \
        $PIP_TRUSTED \
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
