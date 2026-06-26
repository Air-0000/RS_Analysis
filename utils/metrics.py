"""
变化检测评估指标

支持: IoU, F1, Precision, Recall, Overall Accuracy
"""

import torch
import numpy as np


def compute_metrics(pred, target, threshold=0.5):
    """
    计算变化检测指标

    Args:
        pred: (B, 1, H, W) 概率图
        target: (B, H, W) 二值标注
        threshold: 二值化阈值

    Returns:
        dict: 各项指标
    """
    pred = (pred.squeeze(1) > threshold).float()
    target = target.float()

    # 展平
    pred = pred.view(-1)
    target = target.view(-1)

    # 混淆矩阵
    tp = ((pred == 1) & (target == 1)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()
    tn = ((pred == 0) & (target == 0)).sum().float()

    # 计算指标
    eps = 1e-7
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    accuracy = (tp + tn) / (tp + fp + fn + tn + eps)

    return {
        'IoU': iou.item(),
        'F1': f1.item(),
        'Precision': precision.item(),
        'Recall': recall.item(),
        'Accuracy': accuracy.item(),
    }


class MetricsTracker:
    """训练/验证指标跟踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.tn = 0

    def update(self, pred, target, threshold=0.5):
        pred = (pred.squeeze(1) > threshold).float()
        target = target.float()

        self.tp += ((pred == 1) & (target == 1)).sum().item()
        self.fp += ((pred == 1) & (target == 0)).sum().item()
        self.fn += ((pred == 0) & (target == 1)).sum().item()
        self.tn += ((pred == 0) & (target == 0)).sum().item()

    def get_metrics(self):
        eps = 1e-7
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn

        return {
            'IoU': tp / (tp + fp + fn + eps),
            'F1': 2 * tp / (2 * tp + fp + fn + eps),
            'Precision': tp / (tp + fp + eps),
            'Recall': tp / (tp + fn + eps),
            'Accuracy': (tp + tn) / (tp + fp + fn + tn + eps),
            'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn,
        }

    def summary(self):
        m = self.get_metrics()
        return (f"IoU: {m['IoU']:.4f} | F1: {m['F1']:.4f} | "
                f"Prec: {m['Precision']:.4f} | Rec: {m['Recall']:.4f} | "
                f"Acc: {m['Accuracy']:.4f}")
