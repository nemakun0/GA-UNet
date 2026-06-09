"""
SimAM: Simple, Parameter-Free Attention Module
================================================
A parameter-free attention module that infers 3D attention weights
using an energy function based on neuroscience theories.

Reference:
  Yang et al., "SimAM: A Simple, Parameter-Free Attention Module for
  Convolutional Neural Networks", ICML 2021
"""

import torch
import torch.nn as nn


class SimAM(nn.Module):
    """
    SimAM (Simple, Parameter-Free Attention Module).
    
    Computes 3D attention weights (channel + spatial) without any
    learnable parameters. Uses an energy function to determine neuron
    importance based on the difference from the mean activation.
    
    Energy function:
        e_t = 1 / ((x - mu)^2 / (4 * (sigma^2 + eps)) + 1)
    
    Higher energy → less important neuron → lower attention weight.
    The inverse of energy is used as the attention weight.
    
    Args:
        eps: Small constant for numerical stability (default: 1e-4)
    """
    
    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Attention-weighted tensor of same shape
        """
        B, C, H, W = x.shape
        n = H * W - 1  # Degrees of freedom
        
        # Mean and variance across spatial dimensions
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        variance = x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n
        
        # Energy-based attention (inverse energy = importance)
        attention = x_minus_mu_sq / (4 * (variance + self.eps) + self.eps)
        attention = 1.0 / (attention + 1.0)
        
        return x * torch.sigmoid(attention)
