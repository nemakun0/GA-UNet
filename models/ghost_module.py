"""
GhostNetV2 Backbone for GA-UNet
================================
Implements Ghost Modules with DFC (Decoupled Fully Connected) attention
for lightweight and efficient feature extraction.

Reference:
  Tang et al., "GhostNetV2: Enhance Cheap Operation with Long-Range Attention", NeurIPS 2022
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DFCAttention(nn.Module):
    """
    Decoupled Fully Connected (DFC) Attention Module.
    
    Uses decomposed horizontal (1×K) and vertical (K×1) convolutions
    to capture long-range spatial dependencies efficiently.
    """
    
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Channel reduction
        mid_channels = max(channels // 4, 16)
        
        # FC layers for channel attention
        # Note: Using bias=True + ReLU instead of BatchNorm1d to support batch_size=1 (TTA)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=True),
        )
        
        # Spatial attention via decomposed convolutions
        pad = kernel_size // 2
        self.horizontal_conv = nn.Conv2d(
            channels, channels, kernel_size=(1, kernel_size),
            padding=(0, pad), groups=channels, bias=False
        )
        self.vertical_conv = nn.Conv2d(
            channels, channels, kernel_size=(kernel_size, 1),
            padding=(pad, 0), groups=channels, bias=False
        )
        self.bn = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Channel attention
        channel_att = self.avg_pool(x).view(B, C)
        channel_att = self.fc(channel_att).view(B, C, 1, 1)
        
        # Spatial attention via decomposed FC
        spatial_att = self.horizontal_conv(x)
        spatial_att = self.vertical_conv(spatial_att)
        spatial_att = self.bn(spatial_att)
        
        # Combine
        attention = torch.sigmoid(channel_att + spatial_att)
        return x * attention


class GhostModule(nn.Module):
    """
    Ghost Module: generates more features from cheap operations.
    
    Instead of applying a full convolution, it first generates a small set
    of intrinsic feature maps, then applies cheap linear operations (depthwise
    convolution) to produce "ghost" features.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size=1,
                 ratio=2, dw_size=3, stride=1, relu=True):
        super().__init__()
        self.out_channels = out_channels
        init_channels = math.ceil(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)
        
        # Primary convolution (intrinsic features)
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size,
                      stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
        
        # Cheap operation (ghost features via depthwise conv)
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size,
                      stride=1, padding=dw_size // 2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )
    
    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.out_channels, :, :]


class GhostBottleneckV2(nn.Module):
    """
    GhostNetV2 Bottleneck block with DFC attention.
    
    Structure:
      - Ghost Module 1 (expansion) + DFC Attention (parallel)
      - Depthwise Conv (if stride > 1)
      - Ghost Module 2 (projection, no ReLU)
      - Residual connection
    """
    
    def __init__(self, in_channels, mid_channels, out_channels,
                 dw_kernel_size=3, stride=1, use_dfc=True):
        super().__init__()
        self.stride = stride
        self.use_dfc = use_dfc
        
        # Ghost module 1: expansion
        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)
        
        # DFC attention (parallel with ghost1 output)
        if use_dfc:
            self.dfc_att = DFCAttention(mid_channels)
        
        # Depthwise conv for downsampling
        if stride > 1:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, dw_kernel_size,
                          stride=stride, padding=dw_kernel_size // 2,
                          groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
            )
        else:
            self.dw_conv = None
        
        # Ghost module 2: projection (no ReLU)
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)
        
        # Shortcut
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, dw_kernel_size,
                          stride=stride, padding=dw_kernel_size // 2,
                          groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = x
        
        # Ghost 1: expansion
        out = self.ghost1(x)
        
        # DFC attention
        if self.use_dfc:
            out = self.dfc_att(out)
        
        # Depthwise conv
        if self.dw_conv is not None:
            out = self.dw_conv(out)
        
        # Ghost 2: projection
        out = self.ghost2(out)
        
        # Residual
        out = out + self.shortcut(residual)
        return out


class GhostNetV2Encoder(nn.Module):
    """
    GhostNetV2-based encoder that produces multi-scale feature maps
    for use as a U-Net backbone.
    
    Produces features at 4 resolution levels:
      Level 0: H/2  × W/2   (stem)
      Level 1: H/4  × W/4
      Level 2: H/8  × W/8
      Level 3: H/16 × W/16
    
    Args:
        in_channels: Number of input channels (e.g., 3 for 2.5D with 3 slices)
        width_mult: Width multiplier to scale channel counts
    """
    
    def __init__(self, in_channels=3, width_mult=1.0):
        super().__init__()
        
        def _make_divisible(v, divisor=8):
            new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
            if new_v < 0.9 * v:
                new_v += divisor
            return new_v
        
        # Channel configuration per stage
        # (out_channels, mid_channels (expansion), num_blocks, stride, use_dfc)
        self.cfgs = [
            # Stage 1: H/2 -> H/4
            [(_make_divisible(24 * width_mult), _make_divisible(48 * width_mult), 2, 2, True)],
            # Stage 2: H/4 -> H/8
            [(_make_divisible(40 * width_mult), _make_divisible(120 * width_mult), 2, 2, True)],
            # Stage 3: H/8 -> H/16
            [(_make_divisible(80 * width_mult), _make_divisible(240 * width_mult), 3, 2, True)],
        ]
        
        # Stem: input -> H/2
        stem_channels = _make_divisible(16 * width_mult)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_channels),
            nn.ReLU(inplace=True),
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        self.out_channels_list = [stem_channels]  # Level 0
        
        prev_channels = stem_channels
        for stage_cfg in self.cfgs:
            layers = []
            for out_ch, mid_ch, num_blocks, stride, use_dfc in stage_cfg:
                for i in range(num_blocks):
                    s = stride if i == 0 else 1
                    layers.append(
                        GhostBottleneckV2(
                            prev_channels, mid_ch, out_ch,
                            stride=s, use_dfc=use_dfc
                        )
                    )
                    prev_channels = out_ch
            self.stages.append(nn.Sequential(*layers))
            self.out_channels_list.append(prev_channels)
    
    def forward(self, x):
        """Returns list of feature maps at 4 resolution levels."""
        features = []
        
        # Stem (Level 0)
        x = self.stem(x)
        features.append(x)
        
        # Stages (Level 1, 2, 3)
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        return features
