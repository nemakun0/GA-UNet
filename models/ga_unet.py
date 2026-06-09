"""
GA-UNet: GhostNetV2-Attention U-Net
=====================================
A lightweight hybrid architecture for stroke lesion segmentation,
combining a GhostNetV2 encoder with SimAM attention in the decoder.

Architecture:
  Encoder: GhostNetV2 backbone (4 resolution levels)
  Decoder: 4 upsampling stages with skip connections + SimAM attention
  Output:  Single-channel sigmoid map for binary lesion segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ghost_module import GhostNetV2Encoder, GhostModule
from .simam import SimAM


class DecoderBlock(nn.Module):
    """
    Decoder block: Upsample → Concatenate skip → SimAM → Ghost convolutions.
    
    Args:
        in_channels: Channels from the lower-resolution decoder output
        skip_channels: Channels from the encoder skip connection
        out_channels: Output channels after this block
    """
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        
        # Upsampling via transposed convolution
        self.upsample = nn.ConvTranspose2d(
            in_channels, in_channels, kernel_size=2, stride=2
        )
        
        # SimAM attention on skip connection
        self.simam = SimAM()
        
        # Ghost convolution blocks after concatenation
        concat_channels = in_channels + skip_channels
        self.conv_block = nn.Sequential(
            GhostModule(concat_channels, out_channels, kernel_size=3, relu=True),
            nn.BatchNorm2d(out_channels),
            GhostModule(out_channels, out_channels, kernel_size=3, relu=True),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x, skip):
        """
        Args:
            x: Feature map from lower decoder level (B, C_in, H, W)
            skip: Feature map from encoder skip connection (B, C_skip, 2H, 2W)
        """
        # Upsample to match skip spatial size
        x = self.upsample(x)
        
        # Handle size mismatch due to odd dimensions
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        
        # Apply SimAM attention to skip features
        skip = self.simam(skip)
        
        # Concatenate and convolve
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        return x


class GAUNet(nn.Module):
    """
    GA-UNet: GhostNetV2 + Attention U-Net for stroke lesion segmentation.
    
    Takes 2.5D input (multiple stacked axial slices) and produces
    a single-channel binary segmentation mask.
    
    Args:
        in_channels: Number of input channels (= number of 2.5D slices, default 3)
        num_classes: Number of output classes (1 for binary segmentation)
        width_mult: Width multiplier for GhostNetV2 backbone (default 1.0)
    """
    
    def __init__(self, in_channels=3, num_classes=1, width_mult=1.0):
        super().__init__()
        
        # ---- Encoder (GhostNetV2) ----
        self.encoder = GhostNetV2Encoder(
            in_channels=in_channels,
            width_mult=width_mult
        )
        enc_channels = self.encoder.out_channels_list
        # enc_channels: [level0, level1, level2, level3]
        
        # ---- Bottleneck ----
        self.bottleneck = nn.Sequential(
            GhostModule(enc_channels[-1], enc_channels[-1] * 2, kernel_size=3, relu=True),
            nn.BatchNorm2d(enc_channels[-1] * 2),
            GhostModule(enc_channels[-1] * 2, enc_channels[-1] * 2, kernel_size=3, relu=True),
            nn.BatchNorm2d(enc_channels[-1] * 2),
        )
        bottleneck_channels = enc_channels[-1] * 2
        
        # ---- Decoder ----
        # Decoder Level 3: bottleneck -> level3 skip
        self.decoder3 = DecoderBlock(
            bottleneck_channels, enc_channels[3], enc_channels[3]
        )
        # Decoder Level 2: level3 -> level2 skip
        self.decoder2 = DecoderBlock(
            enc_channels[3], enc_channels[2], enc_channels[2]
        )
        # Decoder Level 1: level2 -> level1 skip
        self.decoder1 = DecoderBlock(
            enc_channels[2], enc_channels[1], enc_channels[1]
        )
        # Decoder Level 0: level1 -> level0 skip
        self.decoder0 = DecoderBlock(
            enc_channels[1], enc_channels[0], enc_channels[0]
        )
        
        # ---- Final upsampling to original resolution ----
        self.final_upsample = nn.ConvTranspose2d(
            enc_channels[0], enc_channels[0], kernel_size=2, stride=2
        )
        
        # ---- Segmentation head ----
        self.seg_head = nn.Sequential(
            nn.Conv2d(enc_channels[0], enc_channels[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(enc_channels[0], num_classes, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, num_slices, H, W)
               where num_slices is the 2.5D slice count
        
        Returns:
            Segmentation logits of shape (B, 1, H, W)
        """
        # ---- Encoder ----
        enc_features = self.encoder(x)
        # enc_features[0]: H/2,  enc_features[1]: H/4
        # enc_features[2]: H/8,  enc_features[3]: H/16
        
        # ---- Bottleneck ----
        bottleneck = self.bottleneck(enc_features[-1])
        
        # ---- Decoder ----
        d3 = self.decoder3(bottleneck, enc_features[3])
        d2 = self.decoder2(d3, enc_features[2])
        d1 = self.decoder1(d2, enc_features[1])
        d0 = self.decoder0(d1, enc_features[0])
        
        # ---- Final upsample to original resolution ----
        out = self.final_upsample(d0)
        
        # Handle size mismatch
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # ---- Segmentation head ----
        out = self.seg_head(out)
        
        return out
    
    def count_parameters(self):
        """Returns total and trainable parameter counts."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable
