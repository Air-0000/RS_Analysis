"""
RS_Analysis CLI 推理脚本
支持:
  1. 遥感图像建筑语义分割 (Tiny U-Net)
  2. 双时相遥感图像变化检测 (FC-Siam-diff)
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import cv2
import torch

from models.unet_tiny import build_unet
from models.cnn_cd import build_cnn_model

NORM_MEAN = np.array([0.485, 0.456, 0.406])
NORM_STD = np.array([0.229, 0.224, 0.225])


def run_segmentation(img_path, weights_path=None, output_path=None, device='cpu', threshold=0.5):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {img_path}")

    h_orig, w_orig = img.shape[:2]
    img_r = cv2.resize(img, (256, 256))
    x = (img_r.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
    tensor = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float().to(device)

    model = build_unet(num_classes=1).to(device)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded segmentation weights from {weights_path}")
    else:
        print("Notice: No trained weights provided, running forward pass with model architecture.")

    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()

    mask = (prob > threshold).astype(np.uint8) * 255
    mask_orig = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Color overlay
    color_mask = np.zeros_like(img)
    color_mask[:, :, 2] = mask_orig  # Red highlight for buildings
    overlay = cv2.addWeighted(img, 0.7, color_mask, 0.3, 0)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), overlay)
        print(f"Saved segmentation visualization to {out_p}")

    return prob, mask_orig


def run_change_detection(img_a_path, img_b_path, weights_path=None, output_path=None, device='cpu', threshold=0.5):
    img_A = cv2.imread(str(img_a_path))
    img_B = cv2.imread(str(img_b_path))
    if img_A is None or img_B is None:
        raise FileNotFoundError(f"Cannot load input images: {img_a_path}, {img_b_path}")

    h_orig, w_orig = img_A.shape[:2]
    A_r = cv2.resize(img_A, (256, 256))
    B_r = cv2.resize(img_B, (256, 256))

    norm_A = (A_r.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
    norm_B = (B_r.astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD

    tensor_A = torch.from_numpy(norm_A).permute(2, 0, 1).unsqueeze(0).float().to(device)
    tensor_B = torch.from_numpy(norm_B).permute(2, 0, 1).unsqueeze(0).float().to(device)

    model = build_cnn_model().to(device)
    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
        print(f"Loaded change detection weights from {weights_path}")
    else:
        print("Notice: No trained weights provided, running forward pass with model architecture.")

    model.eval()
    with torch.no_grad():
        prob = model(tensor_A, tensor_B)[0, 0].cpu().numpy()

    mask = (prob > threshold).astype(np.uint8) * 255
    mask_orig = cv2.resize(mask, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)

    # Highlight changes in green
    color_mask = np.zeros_like(img_B)
    color_mask[:, :, 1] = mask_orig
    overlay = cv2.addWeighted(img_B, 0.7, color_mask, 0.3, 0)

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), overlay)
        print(f"Saved change detection visualization to {out_p}")

    return prob, mask_orig


def main():
    parser = argparse.ArgumentParser(description="RS_Analysis CLI Inference Tool")
    parser.add_argument('--task', choices=['segment', 'cd', 'demo'], default='demo',
                        help="Task: 'segment' (building segmentation), 'cd' (change detection), or 'demo' (run both on samples)")
    parser.add_argument('--image', type=str, default='samples/whu_seg_A.png', help="Input image for segmentation")
    parser.add_argument('--img_a', type=str, default='samples/cd_A.png', help="Time 1 image for change detection")
    parser.add_argument('--img_b', type=str, default='samples/cd_B.png', help="Time 2 image for change detection")
    parser.add_argument('--weights', type=str, default=None, help="Path to trained weights .pth file")
    parser.add_argument('--output', type=str, default=None, help="Output image file path")
    parser.add_argument('--threshold', type=float, default=0.5, help="Detection threshold (0.0 - 1.0)")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    print(f"=== RS_Analysis Inference [Device: {args.device}] ===")

    if args.task == 'segment':
        run_segmentation(args.image, args.weights, args.output, args.device, args.threshold)
    elif args.task == 'cd':
        run_change_detection(args.img_a, args.img_b, args.weights, args.output, args.device, args.threshold)
    elif args.task == 'demo':
        print("\n--- Running Segmentation Demo on samples/whu_seg_A.png ---")
        out_seg = args.output or 'outputs/infer/seg_demo.png'
        run_segmentation('samples/whu_seg_A.png', args.weights, out_seg, args.device, args.threshold)

        print("\n--- Running Change Detection Demo on samples/cd_A.png & cd_B.png ---")
        out_cd = 'outputs/infer/cd_demo.png'
        run_change_detection('samples/cd_A.png', 'samples/cd_B.png', args.weights, out_cd, args.device, args.threshold)
        print("\nDemo completed successfully! Check outputs/infer/ for visual results.")


if __name__ == '__main__':
    main()
