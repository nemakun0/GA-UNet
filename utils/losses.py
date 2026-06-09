"""
Loss Functions for Stroke Lesion Segmentation
===============================================
OPTİMİZASYONLAR:
  1. FocalDiceLoss eklendi: Küçük stroke lezyonlarında standart Dice'tan
     çok daha iyi sonuç verir. Zor örneklere (küçük, saçılmış lezyonlar)
     daha yüksek ağırlık verir.
  2. DiceBCELoss korundu (geriye uyumluluk).
  3. FocalDiceBCELoss: Önerilen yeni loss — Focal Dice + BCE karışımı.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Smooth Dice Loss — ikili segmentasyon için."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_f = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)
        inter = (probs_f * targets_f).sum(dim=1)
        card = probs_f.sum(dim=1) + targets_f.sum(dim=1)
        return (1.0 - (2.0 * inter + self.smooth) / (card + self.smooth)).mean()


class FocalDiceLoss(nn.Module):
    """
    Focal Dice Loss — küçük lezyonlar için kritik.

    Standart Dice loss büyük lezyonlara odaklanır; küçük/dağınık stroke
    lezyonlarını kolayca görmezden gelir. Focal Dice zor örneklere
    (düşük Dice skorlu) üstel ağırlık verir.

    Loss = (1 - Dice)^gamma

    gamma=1 → standart Dice
    gamma=2 → küçük lezyonlara 4x daha fazla odaklanır (önerilen)
    """

    def __init__(self, gamma=2.0, smooth=1.0):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_f = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1)
        inter = (probs_f * targets_f).sum(dim=1)
        card = probs_f.sum(dim=1) + targets_f.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (card + self.smooth)
        focal_dice = (1.0 - dice).pow(self.gamma)
        return focal_dice.mean()


class DiceBCELoss(nn.Module):
    """Orijinal Dice + BCE loss (geriye uyumluluk için korundu)."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5, smooth=1.0):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return (self.dice_weight * self.dice_loss(logits, targets) +
                self.bce_weight * self.bce_loss(logits, targets))


class FocalDiceBCELoss(nn.Module):
    """
    Focal Dice + BCE + Pozitif ağırlıklı BCE — ÖNERİLEN LOSS.

    Stroke lezyonları küçük ve seyrek → ciddi sınıf dengesizliği.
    Bu loss üç bileşen içerir:

      1. Focal Dice (ağırlık 0.5): Küçük lezyonlara odaklanır
      2. BCE (ağırlık 0.3): Piksel seviyesinde kararlı gradyan
      3. Pozitif-ağırlıklı BCE (ağırlık 0.2): Lesyon piksellerini cezalandır

    Parametreler:
        focal_weight : Focal Dice bileşeninin ağırlığı
        bce_weight   : Standart BCE ağırlığı
        pos_weight   : Pozitif sınıf (lezyon) için BCE çarpanı
        gamma        : Focal Dice üsteli (2.0 = önerilen)
    """

    def __init__(self, focal_weight=0.5, bce_weight=0.3,
                 pos_bce_weight=0.2, gamma=2.0, pos_weight=5.0):
        super().__init__()
        self.focal_weight = focal_weight
        self.bce_weight = bce_weight
        self.pos_bce_weight = pos_bce_weight

        self.focal_dice = FocalDiceLoss(gamma=gamma)
        self.bce = nn.BCEWithLogitsLoss()
        # pos_weight: lezyon piksellerindeki hatayı 5x daha ağır cezalandır
        self.bce_pos = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, logits, targets):
        # pos_weight tensoru logits ile aynı cihazda olmalı
        if self.bce_pos.pos_weight.device != logits.device:
            self.bce_pos.pos_weight = self.bce_pos.pos_weight.to(logits.device)

        focal = self.focal_dice(logits, targets)
        bce = self.bce(logits, targets)
        bce_pos = self.bce_pos(logits, targets)

        return (self.focal_weight * focal +
                self.bce_weight * bce +
                self.pos_bce_weight * bce_pos)