"""
Evaluation Metrics for Stroke Lesion Segmentation
===================================================
OPTİMİZASYONLAR:
  1. HD95 validation'da her batch'te değil, her 5 epoch'ta hesaplanır
     → scipy distance_transform_edt çok yavaş, validation hızı 3-4x artar
  2. Tensor işlemleri numpy'a geçmeden önce batch seviyesinde yapılır
  3. compute_metrics_batch: hd95_freq parametresi ile kontrollü hesaplama
"""

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt


def dice_score(pred, target, smooth=1e-6):
    """Dice Similarity Coefficient (DSC)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = (pred > 0.5).astype(np.float32).flatten()
    target = target.astype(np.float32).flatten()
    inter = (pred * target).sum()
    return (2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, smooth=1e-6):
    """Intersection over Union (Jaccard Index)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    pred = (pred > 0.5).astype(np.float32).flatten()
    target = target.astype(np.float32).flatten()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return (inter + smooth) / (union + smooth)


def hausdorff_distance_95(pred, target):
    """95th percentile Hausdorff Distance (HD95)."""
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    pred = (pred > 0.5).astype(np.bool_).squeeze()
    target = target.astype(np.bool_).squeeze()

    if not pred.any() and not target.any():
        return 0.0
    if not pred.any() or not target.any():
        return float(np.sqrt(np.sum(np.array(pred.shape) ** 2)))

    pred_dt = distance_transform_edt(~pred)
    target_dt = distance_transform_edt(~target)
    all_dist = np.concatenate([pred_dt[target], target_dt[pred]])
    return float(np.percentile(all_dist, 95))


def compute_metrics_batch(preds, targets, compute_hd95=True):
    """
    Batch için tüm metrikleri hesapla.

    Args:
        compute_hd95: False ise HD95 hesaplanmaz (hızlı validation için).
                      train_local.py içinde her 5 epoch'ta bir True yapılır.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    dsc_list, iou_list, hd95_list = [], [], []

    for i in range(preds.shape[0]):
        p, t = preds[i], targets[i]
        dsc_list.append(dice_score(p, t))
        iou_list.append(iou_score(p, t))
        if compute_hd95 and t.sum() > 0:
            hd95_list.append(hausdorff_distance_95(p, t))

    return {
        'dsc': float(np.mean(dsc_list)) if dsc_list else 0.0,
        'iou': float(np.mean(iou_list)) if iou_list else 0.0,
        'hd95': float(np.mean(hd95_list)) if hd95_list else 0.0,
    }