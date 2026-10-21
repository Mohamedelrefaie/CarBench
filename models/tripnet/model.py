"""TripNet pressure predictor (arXiv:2503.17400).

Frozen per-car triplane (3x32x128x128) -> shared 2D U-Net refine -> bilinear-sample at
surface query points -> 4-layer MLP -> pressure. Architectural choices that the paper
leaves open are documented in IMPLEMENTATION_NOTES.md (§4).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from triplane import sample_triplane
from unet import UNet2D


class QueryMLP(nn.Module):
    """4-layer MLP: 3C -> width -> width -> width -> 1."""

    def __init__(self, in_dim=96, width=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(self, feat):
        return self.net(feat).squeeze(-1)


class TripNet(nn.Module):
    """Shared U-Net + query MLP. The triplane is provided per sample (frozen, fitted)."""

    def __init__(self, channels=32, res=128, unet_base=64, mlp_width=256):
        super().__init__()
        self.C, self.res = channels, res
        self.unet = UNet2D(in_ch=3 * channels, out_ch=3 * channels, base=unet_base)
        self.mlp = QueryMLP(in_dim=3 * channels, width=mlp_width)

    def refine(self, planes: torch.Tensor) -> torch.Tensor:
        """planes (B,3,C,H,W) -> refined (B,3,C,H,W) via shared U-Net on concat channels."""
        B, _, C, H, W = planes.shape
        x = planes.reshape(B, 3 * C, H, W)
        y = self.unet(x)
        return y.reshape(B, 3, C, H, W)

    def forward(self, planes: torch.Tensor, query_ndc: torch.Tensor) -> torch.Tensor:
        """planes (B,3,C,H,W), query_ndc (B,N,3) in [-1,1] -> pressure (B,N)."""
        refined = self.refine(planes)
        feat = sample_triplane(refined, query_ndc)   # (B,N,3C)
        return self.mlp(feat)                         # (B,N)
