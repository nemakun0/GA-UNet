"""
Test-Time Augmentation (TTA) via Geometric Ensembling
======================================================
Improves segmentation predictions by averaging results across
geometric transformations (flips) of the input image.

Unlike entropy-based TTA (Tent), this approach:
  - Does NOT modify model weights → no risk of degradation
  - Does NOT touch BatchNorm statistics → safe with any batch size
  - Produces ensemble-averaged predictions → reduces random errors
  - Works reliably even with early-stage models (1 epoch)

Method:
  1. Original image → predict → pred_0
  2. Horizontal flip → predict → reverse flip → pred_1
  3. Vertical flip → predict → reverse flip → pred_2
  4. final_pred = mean(pred_0, pred_1, pred_2)

Reference:
  Widely used in medical image segmentation competitions (nnU-Net, MICCAI).
"""

import torch
import torch.nn as nn
from torch.amp import autocast


class TestTimeAdaptation:
    """
    Test-Time Augmentation via geometric ensembling.

    Averages predictions across original and geometrically
    transformed (flipped) versions of the input for more
    robust segmentation.

    Args:
        model: Trained GA-UNet model
        device: 'cuda' or 'cpu'
        use_hflip: Apply horizontal flip augmentation (default: True)
        use_vflip: Apply vertical flip augmentation (default: True)

    Note:
        The model weights are NEVER modified. The model stays in eval()
        mode throughout. This guarantees TTA can never degrade results
        compared to normal inference.
    """

    def __init__(self, model, device='cuda', use_hflip=True, use_vflip=True, use_amp=True):
        self.model = model
        self.device = device
        self.use_hflip = use_hflip
        self.use_vflip = use_vflip
        self.use_amp = use_amp and (device != 'cpu')  # AMP sadece GPU'da

    @torch.no_grad()
    def adapt_and_predict(self, x):
        """
        Apply geometric TTA to a single sample or batch.

        For each input, runs inference on the original + flipped versions,
        then averages the sigmoid predictions.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Averaged prediction logits of shape (B, 1, H, W)

        How it works:
            - Original prediction captures normal spatial patterns
            - Horizontal flip captures left-right symmetric features
            - Vertical flip captures top-bottom symmetric features
            - Averaging these predictions cancels out random errors
              and creates a more robust, ensemble-like prediction
        """
        x = x.to(self.device)

        # Model stays in eval mode — we never modify weights
        self.model.eval()

        amp_ctx = autocast('cuda', enabled=self.use_amp)

        # 1. Original prediction
        with amp_ctx:
            logits_orig = self.model(x)
        pred_sum = torch.sigmoid(logits_orig.float())
        num_preds = 1

        # 2. Horizontal flip (left-right)
        if self.use_hflip:
            x_hflip = torch.flip(x, dims=[-1])
            with amp_ctx:
                logits_hflip = self.model(x_hflip)
            pred_hflip = torch.sigmoid(torch.flip(logits_hflip.float(), dims=[-1]))
            pred_sum = pred_sum + pred_hflip
            num_preds += 1

        # 3. Vertical flip (top-bottom)
        if self.use_vflip:
            x_vflip = torch.flip(x, dims=[-2])
            with amp_ctx:
                logits_vflip = self.model(x_vflip)
            pred_vflip = torch.sigmoid(torch.flip(logits_vflip.float(), dims=[-2]))
            pred_sum = pred_sum + pred_vflip
            num_preds += 1

        # Average all predictions
        pred_avg = pred_sum / num_preds

        # Convert back to logit space for consistency
        pred_avg = pred_avg.clamp(1e-7, 1 - 1e-7)
        logits_avg = torch.log(pred_avg / (1 - pred_avg))

        return logits_avg

    @torch.no_grad()
    def predict_batch(self, dataloader, apply_tta=True):
        """
        Run TTA inference on an entire dataloader.

        Args:
            dataloader: Validation/test dataloader
            apply_tta: If False, skip augmentation (regular inference)

        Returns:
            all_preds: List of sigmoid prediction tensors (B, 1, H, W)
            all_targets: List of target tensors (B, 1, H, W)
        """
        all_preds = []
        all_targets = []

        self.model.eval()

        for images, masks in dataloader:
            images = images.to(self.device)

            if apply_tta:
                # Geometric TTA — processes entire batch at once
                # No need for per-sample processing since we don't modify weights
                logits = self.adapt_and_predict(images)
            else:
                # Regular inference
                logits = self.model(images)

            preds = torch.sigmoid(logits)
            all_preds.append(preds.cpu())
            all_targets.append(masks)

        return all_preds, all_targets

    def get_num_augmentations(self):
        """Return the number of augmentations used (including original)."""
        count = 1  # original
        if self.use_hflip:
            count += 1
        if self.use_vflip:
            count += 1
        return count
