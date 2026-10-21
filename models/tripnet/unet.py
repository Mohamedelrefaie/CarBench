"""2D U-Net that refines the concatenated triplane (96ch @ 128x128) for TripNet.

Depth/width are not specified in the paper; we use a standard 4-level U-Net (base 64ch),
GroupNorm + SiLU, skip connections. See IMPLEMENTATION_NOTES.md (§4).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _gn(c):
    return nn.GroupNorm(num_groups=min(32, max(1, c // 4)), num_channels=c)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), _gn(cout), nn.SiLU(),
            nn.Conv2d(cout, cout, 3, padding=1), _gn(cout), nn.SiLU(),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    """96 -> 96 U-Net. base=64, 4 levels."""

    def __init__(self, in_ch=96, out_ch=96, base=64):
        super().__init__()
        c1, c2, c3, c4 = base, base * 2, base * 4, base * 8
        self.in_conv = ConvBlock(in_ch, c1)
        self.down1, self.enc1 = nn.MaxPool2d(2), ConvBlock(c1, c2)
        self.down2, self.enc2 = nn.MaxPool2d(2), ConvBlock(c2, c3)
        self.down3, self.bott = nn.MaxPool2d(2), ConvBlock(c3, c4)
        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = ConvBlock(c1 * 2, c1)
        self.out_conv = nn.Conv2d(c1, out_ch, 1)

    def forward(self, x):
        x0 = self.in_conv(x)              # c1 @ 128
        x1 = self.enc1(self.down1(x0))    # c2 @ 64
        x2 = self.enc2(self.down2(x1))    # c3 @ 32
        xb = self.bott(self.down3(x2))    # c4 @ 16
        y = self.dec3(torch.cat([self.up3(xb), x2], 1))
        y = self.dec2(torch.cat([self.up2(y), x1], 1))
        y = self.dec1(torch.cat([self.up1(y), x0], 1))
        return self.out_conv(y)


class OccupancyDecoder(nn.Module):
    """Shared decoder for stage-1/2 occupancy auto-decoding: 3C -> 1 logit."""

    def __init__(self, in_dim=96, width=128, depth=4):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth - 1):
            layers += [nn.Linear(d, width), nn.SiLU()]
            d = width
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, feat):  # feat: (..., 3C)
        return self.net(feat).squeeze(-1)
