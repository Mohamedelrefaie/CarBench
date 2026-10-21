"""Triplane utilities for TripNet (rebuilt from arXiv:2503.17400).

A triplane is 3 axis-aligned feature grids (xy, yz, xz), each (C, H, W). A 3D point with
coords in [-1,1]^3 is projected onto each plane and bilinearly sampled; the 3 sampled
feature vectors are concatenated to a 3C-d feature. See IMPLEMENTATION_NOTES.md.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Plane k samples coords (axis_u, axis_v) of (x,y,z):
#   xy -> (0,1), yz -> (1,2), xz -> (0,2)
_PLANE_AXES = ((0, 1), (1, 2), (0, 2))


def normalize_coords(pts: torch.Tensor, bbox_min: torch.Tensor, bbox_max: torch.Tensor) -> torch.Tensor:
    """Map world coords to [-1,1]^3 using the (per-car) bbox. pts: (..., 3)."""
    center = (bbox_min + bbox_max) * 0.5
    scale = (bbox_max - bbox_min) * 0.5
    scale = torch.clamp(scale, min=1e-6)
    return (pts - center) / scale


def sample_triplane(planes: torch.Tensor, pts_ndc: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample a triplane at normalized points.

    Args:
        planes:  (B, 3, C, H, W) feature planes (xy, yz, xz).
        pts_ndc: (B, N, 3) query points in [-1, 1]^3.
    Returns:
        (B, N, 3C) concatenated sampled features.
    """
    B, _, C, H, W = planes.shape
    N = pts_ndc.shape[1]
    feats = []
    for k, (au, av) in enumerate(_PLANE_AXES):
        # grid_sample expects grid (B, N, 1, 2) with (x=col=u, y=row=v) in [-1,1]
        grid = torch.stack([pts_ndc[..., au], pts_ndc[..., av]], dim=-1).view(B, N, 1, 2)
        sampled = F.grid_sample(planes[:, k], grid, mode="bilinear",
                                padding_mode="border", align_corners=True)  # (B, C, N, 1)
        feats.append(sampled.squeeze(-1).permute(0, 2, 1))                  # (B, N, C)
    return torch.cat(feats, dim=-1)                                         # (B, N, 3C)


class TriplaneParam(torch.nn.Module):
    """A single learnable triplane (used during occupancy auto-decoding)."""

    def __init__(self, channels: int = 32, res: int = 128, init_std: float = 0.01):
        super().__init__()
        self.planes = torch.nn.Parameter(torch.randn(1, 3, channels, res, res) * init_std)

    def forward(self, pts_ndc: torch.Tensor) -> torch.Tensor:
        return sample_triplane(self.planes, pts_ndc)
