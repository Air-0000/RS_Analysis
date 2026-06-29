"""
遥感图像分析平台 — Streamlit 交互界面

功能:
  📷 语义分割 — 上传单张图像，识别建筑/目标区域
  🔄 变化检测 — 上传两时相图像，检测建筑变化
"""

import sys
from pathlib import Path

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
HERMES_DIR = str(Path.home() / 'AppData/Local/hermes/hermes-agent')
sys.path = [p for p in sys.path if p != '' and p != HERMES_DIR]

from models.cnn_cd import build_enhanced_cd_model as build_cnn_model
from models.unet_tiny import build_enhanced_unet as build_unet

st.set_page_config(page_title="遥感图像分析平台", page_icon="🛰️", layout="wide")


@st.cache_resource
def load_segmenter():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_unet(num_classes=1).to(device)
    path = Path('outputs') / 'best_unet.pth'
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        st.sidebar.success("✓ 分割模型已加载")
    else:
        st.sidebar.info("分割模型未训练，使用随机初始化")
    model.eval()
    return model, device


@st.cache_resource
def load_change_detector():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_cnn_model().to(device)
    path = Path('outputs') / 'best_siamdiff.pth'
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        st.sidebar.success("✓ 变化检测模型已加载 (F1=0.65)")
    model.eval()
    return model, device


@torch.no_grad()
def run_segmentation(model, img, device):
    img_r = cv2.resize(img, (256, 256))
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    x = (img_r.astype(np.float32) / 255.0 - mean) / std
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)
    pred = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return pred, img_r


@torch.no_grad()
def run_change_detection(model, img_A, img_B, device):
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    def norm(x): return (x / 255.0 - mean) / std
    A = torch.from_numpy(norm(img_A)).permute(2, 0, 1).unsqueeze(0).float().to(device)
    B = torch.from_numpy(norm(img_B)).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return model(A, B)[0, 0].cpu().numpy()


# ======== 模式选择 ========
st.sidebar.title("🛰️ 遥感分析")
mode = st.sidebar.radio("选择功能", ["📷 语义分割", "🔄 变化检测"], index=0)

# ======== 语义分割 ========
if mode == "📷 语义分割":
    st.title("📷 建筑/目标分割")
    st.markdown("上传遥感图像，自动提取建筑或目标区域")

    seg_model, seg_device = load_segmenter()
    threshold = st.sidebar.slider("分割阈值", 0.1, 0.9, 0.5, 0.05)

    file = st.sidebar.file_uploader("上传图像", type=['png', 'jpg', 'jpeg'])
    use_sample = st.sidebar.checkbox("使用样例")

    if use_sample and not file:
        sample_dir = Path(__file__).parent / 'samples'
        samples = sorted(list(sample_dir.glob('whu_seg*_A.*')))
        if samples:
            idx = st.sidebar.selectbox("样例", range(len(samples)),
                                       format_func=lambda i: f"建筑密度{['高','中'][i]}" if i < 2 else f"样例{i+1}")
            img = cv2.imread(str(samples[idx]))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            st.info("无样例"); st.stop()
    elif file:
        img = np.array(Image.open(file))
    else:
        st.info("上传图像或勾选「使用样例」"); st.stop()

    with st.spinner("分析中..."):
        pred_mask, img_resized = run_segmentation(seg_model, img, seg_device)

    cols = st.columns(2)
    with cols[0]:
        st.markdown('<p class="section-title">📷 原图</p>', unsafe_allow_html=True)
        st.image(img_resized, use_container_width=True, channels="RGB")
    with cols[1]:
        st.markdown('<p class="section-title">🎯 分割结果</p>', unsafe_allow_html=True)
        # 绿色叠加
        overlay = img_resized.copy()
        overlay[pred_mask > threshold] = [0, 255, 0]
        blended = cv2.addWeighted(img_resized, 0.6, overlay, 0.4, 0)
        st.image(blended, use_container_width=True, channels="RGB")

    ratio = (pred_mask > threshold).mean() * 100
    st.markdown(f'<p class="metric-value">{ratio:.1f}%</p><p class="metric-label">目标面积占比</p>',
                unsafe_allow_html=True)

    with st.expander("📋 模型"):
        st.markdown("Tiny U-Net | 参数: ~0.8M | 设备: " + seg_device.upper())

# ======== 变化检测 ========
else:
    st.title("🔄 遥感变化检测")
    st.markdown("上传两时相图像，自动标出变化区域")

    cd_model, cd_device = load_change_detector()

    with st.sidebar:
        fA = st.file_uploader("T1", type=['png', 'jpg', 'jpeg'], key='t1')
        fB = st.file_uploader("T2", type=['png', 'jpg', 'jpeg'], key='t2')
        use_sample = st.checkbox("使用样例")
        threshold = st.slider("检测阈值", 0.1, 0.9, 0.3, 0.05)

    if use_sample:
        sample_dir = Path(__file__).parent / 'samples'
        samples_A = sorted(list(sample_dir.glob('cd*_A.*')))
        if samples_A:
            idx = st.sidebar.selectbox("样例", range(len(samples_A)),
                                       format_func=lambda i: f"样例{i+1}", key='cd_s')
            img_A = cv2.imread(str(samples_A[idx]))
            img_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2RGB)
            base = samples_A[idx].stem
            sb = sample_dir / f"{base[:-2]}_B.{samples_A[idx].suffix[1:]}"
            if not sb.exists(): sb = sample_dir / f"{base[:-2]}_B.png"
            img_B = cv2.imread(str(sb))
            img_B = cv2.cvtColor(img_B, cv2.COLOR_BGR2RGB)
        else:
            st.info("无样例"); st.stop()
    elif fA and fB:
        img_A = np.array(Image.open(fA))
        img_B = np.array(Image.open(fB))
    else:
        st.info("上传 T1/T2 或勾选「使用样例」"); st.stop()

    img_A = cv2.resize(img_A, (256, 256))
    img_B = cv2.resize(img_B, (256, 256))

    with st.spinner("推理中..."):
        pred = run_change_detection(cd_model, img_A, img_B, cd_device)

    cols = st.columns(3)
    with cols[0]:
        st.markdown('<p class="section-title">📷 T1</p>', unsafe_allow_html=True)
        st.image(img_A, use_container_width=True, channels="RGB")
    with cols[1]:
        st.markdown('<p class="section-title">📷 T2</p>', unsafe_allow_html=True)
        st.image(img_B, use_container_width=True, channels="RGB")
    with cols[2]:
        st.markdown('<p class="section-title">🔍 差异</p>', unsafe_allow_html=True)
        diff = np.abs(img_A.astype(float) - img_B.astype(float)).astype(np.uint8)
        st.image(diff, use_container_width=True)

    bin_m = (pred > threshold).astype(float)
    overlay = img_A.copy()
    overlay[bin_m == 1] = [255, 0, 0]
    blended = cv2.addWeighted(img_A, 0.6, overlay, 0.4, 0)

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 🔥 变化概率"); st.image(pred, use_container_width=True, clamp=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with r2:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 🎯 变化区域"); st.image(blended, use_container_width=True, channels="RGB")
        st.markdown('</div>', unsafe_allow_html=True)
    with r3:
        st.markdown('<div class="result-card">', unsafe_allow_html=True)
        st.markdown("### 📈 统计")
        st.markdown(f'<p class="metric-value">{(pred>threshold).mean()*100:.1f}%</p>'
                    f'<p class="metric-label">变化面积</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{pred[pred>threshold].mean():.2f}</p>'
                    f'<p class="metric-label">平均置信度</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{pred.max():.2f}</p>'
                    f'<p class="metric-label">最高置信度</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 模型"):
        st.markdown("FC-Siam-diff | 625K 参数 | Val F1=0.65 | 设备: " + cd_device.upper())
