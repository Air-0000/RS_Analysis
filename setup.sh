#!/bin/bash
# 遥感图像分析平台 — 环境一键配置（跨平台版）
# 支持: Windows (git-bash), macOS (Intel/Apple Silicon), Linux
# 自动检测 GPU/CUDA/MPS，选择匹配的 PyTorch 版本
# 自动选择可用镜像源（中国优先，全球降级）
# 所有输出写入 setup_<时间戳>.log
#
# Usage: bash setup.sh

set -euo pipefail

PIP_TIMEOUT=120
ENV_NAME="rs_analysis"
MINICONDA_BASE="https://repo.anaconda.com/miniconda"

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

# ── 操作系统检测 ──────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS" in
    Linux*)      OS_NAME="linux"   ;;
    Darwin*)     OS_NAME="macos"   ;;
    MINGW*|MSYS*) OS_NAME="windows" ;;
    *) echo "❌ 不支持的系统: $OS (请使用 bash/git-bash)"; exit 1 ;;
esac
echo "💻 系统: $OS_NAME ($ARCH)"
echo ""

# ── 镜像源检测 ────────────────────────────────────
echo "🌐 检测镜像源..."
PIP_INDEX="https://pypi.org/simple"
PIP_TRUSTED=""
TORCH_FALLBACK_INDEX="https://download.pytorch.org/whl"
TORCH_MIRROR="official"

if curl -sI --max-time 5 "https://pypi.tuna.tsinghua.edu.cn/simple" >/dev/null 2>&1; then
    PIP_INDEX="https://pypi.tuna.tsinghua.edu.cn/simple"
    PIP_TRUSTED="--trusted-host pypi.tuna.tsinghua.edu.cn"
    echo "  📦 PyPI 镜像: 清华"
fi
if curl -sI --max-time 5 "https://mirrors.aliyun.com/pytorch-wheels/cu130/" >/dev/null 2>&1; then
    TORCH_MIRROR="aliyun"
    echo "  🔥 PyTorch 镜像: 阿里云"
else
    echo "  🔥 PyTorch 源: 官方"
fi
CONDA_CHANNEL=""
if curl -sI --max-time 5 "https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/" >/dev/null 2>&1; then
    CONDA_CHANNEL="--channel https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/"
    echo "  📥 Conda 镜像: 清华"
fi
echo ""

# ── 硬件与 PyTorch 版本检测 ──────────────────────
echo "🔍 检测硬件..."
GPU_NAME=""; TORCH_INDEX_URL=""; TORCH_VERSION=""; HAS_CUDA=0; HAS_MPS=0

detect_nvidia() {
    if command -v nvidia-smi &>/dev/null; then
        local name
        name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
        [ -n "$name" ] && GPU_NAME="$name" && HAS_CUDA=1 && return 0
    fi
    return 1
}

detect_apple_mps() {
    # Apple Silicon 且有 Python 时检测 MPS
    if [ "$OS_NAME" = "macos" ] && [ "$ARCH" = "arm64" ]; then
        # 检查是否已有 torch 安装
        if python3 -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null | grep -q True; then
            GPU_NAME="Apple Silicon (MPS)"
            HAS_MPS=1
            return 0
        fi
    fi
    return 1
}

detect_amd_rocm() {
    if command -v rocm-smi &>/dev/null; then
        local name
        name=$(rocm-smi --showproductname 2>/dev/null | grep "GPU" | head -1 || echo "")
        [ -n "$name" ] && GPU_NAME="AMD GPU (ROCm)" && HAS_CUDA=1 && return 0
    fi
    return 1
}

case "$OS_NAME" in
    windows|linux)
        if detect_nvidia; then
            # 提取 CUDA 版本
            CUDA_VER=$(nvidia-smi 2>/dev/null | grep -i "CUDA Version" | sed 's/.*CUDA Version: *//' | cut -d' ' -f1 | head -1 || echo "")
            if [ -z "$CUDA_VER" ]; then
                CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA UMD Version" | sed 's/.*CUDA UMD Version: *//' | cut -d' ' -f1 | head -1 || echo "")
            fi
            CC_RAW=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 || echo "")
            
            echo "  GPU: $GPU_NAME"
            [ -n "$CUDA_VER" ] && echo "  CUDA: $CUDA_VER"
            [ -n "$CC_RAW" ] && echo "  Compute Capability: $CC_RAW"
            
            CUDA_MAJOR=$(echo "$CUDA_VER" | sed 's/\..*//')
            
            if [ -z "$CUDA_MAJOR" ]; then
                # 降级从驱动版本反推
                DRV_VER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || echo "")
                DRV_MAJOR=$(echo "$DRV_VER" | sed 's/\..*//')
                [ -n "$DRV_MAJOR" ] && [ "$DRV_MAJOR" -ge 610 ] 2>/dev/null && CUDA_MAJOR=13
                [ -n "$DRV_MAJOR" ] && [ "$DRV_MAJOR" -ge 525 ] 2>/dev/null && [ "$DRV_MAJOR" -lt 610 ] 2>/dev/null && CUDA_MAJOR=12
            fi
            
            if [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 13 ] 2>/dev/null; then
                CUDA_SUFFIX="cu130"
                [ "$TORCH_MIRROR" = "aliyun" ] && CUDA_SUFFIX="+cu130" || CUDA_SUFFIX=""
                [ "$TORCH_MIRROR" = "aliyun" ] && TORCH_INDEX_URL="https://mirrors.aliyun.com/pytorch-wheels/cu130" \
                    || TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cu130"
                TORCH_VERSION="2.12.1${CUDA_SUFFIX}"
                TV_VERSION="0.27.1${CUDA_SUFFIX}"
                echo "  → 选用: PyTorch ${TORCH_VERSION}"
            elif [ -n "$CUDA_MAJOR" ] && [ "$CUDA_MAJOR" -ge 12 ] 2>/dev/null; then
                [ "$TORCH_MIRROR" = "aliyun" ] && CUDA_SUFFIX="+cu121" || CUDA_SUFFIX=""
                [ "$TORCH_MIRROR" = "aliyun" ] && TORCH_INDEX_URL="https://mirrors.aliyun.com/pytorch-wheels/cu121" \
                    || TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cu121"
                TORCH_VERSION="2.12.1${CUDA_SUFFIX}"
                TV_VERSION="0.27.1${CUDA_SUFFIX}"
                echo "  → 选用: PyTorch ${TORCH_VERSION}"
            else
                [ "$TORCH_MIRROR" = "aliyun" ] && CUDA_SUFFIX="+cu118" || CUDA_SUFFIX=""
                [ "$TORCH_MIRROR" = "aliyun" ] && TORCH_INDEX_URL="https://mirrors.aliyun.com/pytorch-wheels/cu118" \
                    || TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cu118"
                TORCH_VERSION="2.4.0${CUDA_SUFFIX}"
                TV_VERSION="0.19.0${CUDA_SUFFIX}"
            fi
        elif [ "$OS_NAME" = "linux" ] && detect_amd_rocm; then
            echo "  GPU: $GPU_NAME"
            TORCH_INDEX_URL="https://download.pytorch.org/whl/rocm6.2"
            TORCH_VERSION="2.4.0"
            TV_VERSION="0.19.0"
            echo "  → 选用: PyTorch ${TORCH_VERSION}+ROCm"
        else
            echo "  未检测到 NVIDIA/AMD GPU → CPU-only"
        fi
        ;;
    macos)
        if detect_apple_mps; then
            echo "  GPU: $GPU_NAME (MPS 加速)"
            TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cpu"
            TORCH_VERSION="2.4.0"
            TV_VERSION="0.19.0"
            echo "  → 选用: PyTorch ${TORCH_VERSION} (MPS)"
        elif detect_nvidia; then
            echo "  GPU: $GPU_NAME (Intel Mac, 不推荐)"
            echo "  ⚠️  macOS 上 CUDA 支持已废弃，使用 CPU 版"
            TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cpu"
            TORCH_VERSION="2.4.0"
            TV_VERSION="0.19.0"
        else
            echo "  未检测到 GPU → CPU-only"
            TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cpu"
            TORCH_VERSION="2.4.0"
            TV_VERSION="0.19.0"
        fi
        ;;
esac

if [ -z "$TORCH_INDEX_URL" ]; then
    TORCH_INDEX_URL="$TORCH_FALLBACK_INDEX/cpu"
    TORCH_VERSION="2.4.0"
    TV_VERSION="0.19.0"
    echo "  → 使用 CPU-only 版本"
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
    
    # 选择安装器
    case "$OS_NAME" in
        windows)
            INSTALLER="$TMP/Miniconda3-latest-Windows-x86_64.exe"
            [ -z "${TMP:-}" ] && INSTALLER="$HOME/Miniconda3-latest-Windows-x86_64.exe"
            MINICONDA_URL="$MINICONDA_BASE/Miniconda3-latest-Windows-x86_64.exe"
            curl -# -fSL -o "$INSTALLER" "$MINICONDA_URL"
            echo "  运行安装程序..."
            cmd //c "$INSTALLER" /S "/D=$HOME/miniconda3" 2>/dev/null || \
                "$INSTALLER" /S "/D=$HOME/miniconda3"
            rm -f "$INSTALLER"
            export PATH="$HOME/miniconda3/Scripts:$HOME/miniconda3:$PATH"
            ;;
        macos)
            if [ "$ARCH" = "arm64" ]; then
                INSTALLER="/tmp/Miniconda3-latest-MacOSX-arm64.sh"
                MINICONDA_URL="$MINICONDA_BASE/Miniconda3-latest-MacOSX-arm64.sh"
            else
                INSTALLER="/tmp/Miniconda3-latest-MacOSX-x86_64.sh"
                MINICONDA_URL="$MINICONDA_BASE/Miniconda3-latest-MacOSX-x86_64.sh"
            fi
            curl -# -fSL -o "$INSTALLER" "$MINICONDA_URL"
            bash "$INSTALLER" -b -p "$HOME/miniconda3"
            rm -f "$INSTALLER"
            export PATH="$HOME/miniconda3/bin:$PATH"
            ;;
        linux)
            INSTALLER="/tmp/Miniconda3-latest-Linux-x86_64.sh"
            MINICONDA_URL="$MINICONDA_BASE/Miniconda3-latest-Linux-x86_64.sh"
            curl -# -fSL -o "$INSTALLER" "$MINICONDA_URL"
            bash "$INSTALLER" -b -p "$HOME/miniconda3"
            rm -f "$INSTALLER"
            export PATH="$HOME/miniconda3/bin:$PATH"
            ;;
    esac
    
    CONDA_BASE="$HOME/miniconda3"
    echo "  ✅ Miniconda 已安装到 $CONDA_BASE"
else
    echo "  ✅ Conda: $CONDA_BASE"
fi
echo ""

# ── Conda 初始化 (确保 conda run 可用) ────────────
# conda 4.14+ 的 conda run 需要初始化 shell 集成
if [ "$OS_NAME" != "windows" ]; then
    conda init bash 2>/dev/null || true
fi

echo ""

# ── 检测 base 环境是否已满足 ─────────────────
echo "🔍 检查 base 环境..."
BASE_PY=""
for p in python3 python; do
    command -v "$p" &>/dev/null && BASE_PY="$p" && break
done
[ -z "$BASE_PY" ] && [ -f "/d/environment/anaconda3/python.exe" ] && BASE_PY="/d/environment/anaconda3/python.exe"

USE_BASE=0
if [ -n "$BASE_PY" ]; then
    HAS_TORCH=$($BASE_PY -c "import torch; print('ok')" 2>/dev/null || echo "")
    if [ -n "$HAS_TORCH" ]; then
        # 检查核心包是否能导入
        CAN_IMPORT=$($BASE_PY -c "
pkgs = ['numpy', 'cv2', 'PIL', 'streamlit', 'tqdm', 'matplotlib', 'albumentations', 'tensorboard', 'pandas']
missing = [p for p in pkgs if __import__(p) is None]
try:
    for p in pkgs: __import__(p)
    print('ok')
except: print('missing')
" 2>/dev/null || echo "missing")
        if [ "$CAN_IMPORT" = "ok" ]; then
            echo "  ✅ base 环境已满足全部依赖"
            USE_BASE=1
        else
            echo "  ⚠️  base 环境缺包，将创建独立环境"
        fi
    else
        echo "  ⚠️  base 环境无 PyTorch，将创建独立环境"
    fi
fi

if [ "$USE_BASE" -eq 1 ]; then
    echo "  使用: $BASE_PY"
    PY_RUN="$BASE_PY"
else
    echo "  base 环境不满足，创建独立环境..."
    ENV_NAME="rs_analysis"
    if conda env list | grep -q "^${ENV_NAME}[[:space:]]"; then
        echo "  环境已存在，跳过创建"
    else
        for i in 1 2 3; do
            echo "  尝试 ($i/3)..."
            OPTS=""
            [ -n "$CONDA_CHANNEL" ] && OPTS="$CONDA_CHANNEL"
            if conda create -y -n "$ENV_NAME" python=3.12 pip $OPTS 2>/dev/null; then
                echo "  ✅ 创建成功"
                break
            fi
            [ "$i" -eq 3 ] && echo "  ❌ 创建失败，请检查网络后重试" && exit 1
            sleep 5
        done
    fi
    PY_RUN="conda run -n $ENV_NAME --no-capture-output"
fi
echo ""

# 检查是否需要装 PyTorch
NEED_TORCH=0
$PY_RUN -c "import torch; print('ok')" 2>/dev/null || NEED_TORCH=1

# ── 安装 PyTorch（按需）──────────────────────────
[ "$NEED_TORCH" -eq 0 ] && echo "📥 PyTorch 已存在，跳过安装 ($($PY_RUN -c "import torch; print(torch.__version__)" 2>/dev/null || echo '?'))" && skip_torch=1
if [ "${skip_torch:-0}" -eq 0 ]; then
echo "📥 安装 PyTorch $TORCH_VERSION..."
PIP_EXTRA="--timeout $PIP_TIMEOUT"
[ "$TORCH_MIRROR" = "aliyun" ] && PIP_EXTRA="$PIP_EXTRA --trusted-host mirrors.aliyun.com"

for i in 1 2 3; do
    echo "  尝试 ($i/3)..."
    if [ "$TORCH_MIRROR" = "aliyun" ] && [ -n "$CUDA_SUFFIX" ]; then
        # 阿里云镜像：先用 curl 下载 wheel（断点续传），再本地安装
        WHEEL_DIR="$PROJECT_DIR/.torch_wheels"
        mkdir -p "$WHEEL_DIR"
        # 阿里云 wheel 文件名中 +cuXXX 需要 URL 编码为 %2BcuXXX
        WHL_CUDA_SUFFIX="%2Bcu130"
        echo "  下载 torch ..."
        curl -# -fSL --retry 3 --retry-delay 5 \
            "https://mirrors.aliyun.com/pytorch-wheels/cu130/torch-2.12.1${WHL_CUDA_SUFFIX}-cp312-cp312-win_amd64.whl" \
            -o "$WHEEL_DIR/torch.whl" 2>&1 || { rm -rf "$WHEEL_DIR"; continue; }
        echo "  下载 torchvision ..."
        curl -# -fSL --retry 3 --retry-delay 5 \
            "https://mirrors.aliyun.com/pytorch-wheels/cu130/torchvision-0.27.1${WHL_CUDA_SUFFIX}-cp312-cp312-win_amd64.whl" \
            -o "$WHEEL_DIR/torchvision.whl" 2>&1 || { rm -rf "$WHEEL_DIR"; continue; }
        if $PY_RUN pip install "$WHEEL_DIR/torch.whl" "$WHEEL_DIR/torchvision.whl" $PIP_EXTRA; then
            echo "  ✅ PyTorch 安装完成"
            rm -rf "$WHEEL_DIR"
            break
        fi
        rm -rf "$WHEEL_DIR"
    else
        # 官方源：直接 pip 走 index
        if $PY_RUN pip install \
            --index-url "$TORCH_INDEX_URL" \
            $PIP_EXTRA \
            torch=="$TORCH_VERSION" torchvision=="$TV_VERSION"; then
            echo "  ✅ PyTorch 安装完成"
            break
        fi
    fi
    [ "$i" -eq 3 ] && echo "  ⚠️  PyTorch 安装失败，跳过（可手动安装）" && break
    sleep 3
done
fi  # end skip_torch
echo ""

# ── 安装其他依赖 ──────────────────────────────────
echo "📦 安装项目依赖..."
for i in 1 2 3; do
    echo "  尝试 ($i/3)..."
    if $PY_RUN pip install \
        -r requirements.txt \
        -i "$PIP_INDEX" \
        $PIP_TRUSTED \
        --timeout "$PIP_TIMEOUT"; then
        echo "  ✅ 项目依赖安装完成"
        break
    fi
    [ "$i" -eq 3 ] && echo "  ⚠️  部分依赖安装失败，跳过（可手动安装）" && break
    sleep 3
done
echo ""

# ── 验证 GPU ──────────────────────────────────────
echo "🧪 验证安装..."
VERIFY_SCRIPT="$PROJECT_DIR/.verify_env.py"
cat > "$VERIFY_SCRIPT" << 'PYEOF'
import torch, sys
print(f'PyTorch: {torch.__version__}')

if torch.cuda.is_available():
    print(f'CUDA 可用: True')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
    x = torch.randn(100, 100).cuda()
    y = x @ x.T
    print(f'CUDA 矩阵运算: OK ({y.sum().item():.2f})')
elif torch.backends.mps.is_available():
    print(f'MPS 可用: True (Apple Silicon)')
    print('MPS 部分算子支持有限，训练推荐 CPU')
else:
    print('GPU: 未检测到 (使用 CPU)')
PYEOF
$PY_RUN python "$VERIFY_SCRIPT" 2>&1 || echo "  ⚠️  验证失败（可能 torch 未正确安装）"
rm -f "$VERIFY_SCRIPT"

echo ""
echo "───────────────────────────────────────────────"
echo "  🚀 启动: bash run.sh"
echo "  变化检测训练: conda run -n $ENV_NAME python train_cd.py"
echo "  语义分割训练: conda run -n $ENV_NAME python train_segment.py"
echo "  查看日志: cat $LOG_FILE"
echo "───────────────────────────────────────────────"
