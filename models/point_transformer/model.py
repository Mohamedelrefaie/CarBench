# models/point_transformer/model.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Point Transformer for pressure field prediction on 3D point clouds.
Adapted for the DrivAerNet++ dataset.

Based on "Point Transformer" (Zhao et al., 2021)
Uses self-attention mechanism for point cloud processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class PointTransformerLayer(nn.Module):
    """Point Transformer layer with local self-attention"""
    def __init__(self, in_channels, out_channels, num_neighbors=16):
        super(PointTransformerLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_neighbors = num_neighbors
        
        # Linear transformations for attention
        self.fc_qkv = nn.Linear(in_channels, out_channels * 3)
        self.fc_alpha = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Position encoding
        self.fc_pos = nn.Sequential(
            nn.Linear(3, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        
        # Output projection
        self.fc_out = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, features, positions):
        """
        Args:
            features: (B, N, C_in) point features
            positions: (B, N, 3) point positions
        Returns:
            (B, N, C_out) transformed features
        """
        B, N, C = features.shape
        
        # Compute Q, K, V
        qkv = self.fc_qkv(features).view(B, N, 3, self.out_channels)
        q, k, v = qkv.unbind(dim=2)  # Each is (B, N, C_out)
        
        # Find k-nearest neighbors for each point
        dist = torch.cdist(positions, positions)  # (B, N, N)
        _, idx = dist.topk(self.num_neighbors, dim=-1, largest=False)  # (B, N, K)
        
        # Gather neighbor features
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, self.out_channels)  # (B, N, K, C_out)
        k_neighbors = k.unsqueeze(2).expand(-1, -1, idx.size(2), -1).gather(1, idx_expanded)  # (B, N, K, C_out)
        v_neighbors = v.unsqueeze(2).expand(-1, -1, idx.size(2), -1).gather(1, idx_expanded)  # (B, N, K, C_out)
        
        # Position encoding
        pos_expanded = positions.unsqueeze(2).expand(-1, -1, idx.size(2), -1)  # (B, N, K, 3)
        pos_neighbors = positions.unsqueeze(1).expand(-1, N, -1, -1).gather(2, idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        pos_diff = pos_expanded - pos_neighbors  # (B, N, K, 3)
        pos_enc = self.fc_pos(pos_diff)  # (B, N, K, C_out)
        
        # Attention weights
        q_expanded = q.unsqueeze(2)  # (B, N, 1, C_out)
        attention = q_expanded - k_neighbors + pos_enc  # (B, N, K, C_out)
        attention = self.fc_alpha(attention)  # (B, N, K, C_out)
        attention = F.softmax(attention / np.sqrt(self.out_channels), dim=2)  # (B, N, K, C_out)
        
        # Weighted aggregation
        out = (attention * (v_neighbors + pos_enc)).sum(dim=2)  # (B, N, C_out)
        
        # Output projection and residual
        out = self.fc_out(out)
        if self.in_channels == self.out_channels:
            out = out + features
        
        return self.norm(out)


class PointTransformerBlock(nn.Module):
    """Point Transformer block with feed-forward network"""
    def __init__(self, channels, num_neighbors=16):
        super(PointTransformerBlock, self).__init__()
        self.attention = PointTransformerLayer(channels, channels, num_neighbors)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        self.norm = nn.LayerNorm(channels)
        
    def forward(self, features, positions):
        # Self-attention
        features = self.attention(features, positions)
        
        # Feed-forward network
        identity = features
        features = self.ffn(features)
        features = self.norm(features + identity)
        
        return features


class PointTransformer(nn.Module):
    """Point Transformer architecture"""
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=1, 
                 num_layers=12, num_neighbors=16, dropout=0.1):
        super(PointTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            PointTransformerBlock(hidden_dim, num_neighbors)
            for _ in range(num_layers)
        ])
        
        # Global feature extraction
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, positions):
        """
        Args:
            positions: (B, N, 3) point positions
        Returns:
            (B, N, 1) predictions
        """
        B, N, _ = positions.shape
        
        # Input projection
        features = self.input_proj(positions)
        
        # Apply transformer blocks
        for block in self.blocks:
            features = block(features, positions)
        
        # Global context
        global_feat = self.global_pool(features.mean(dim=1, keepdim=True))  # (B, 1, C)
        global_feat = global_feat.expand(-1, N, -1)  # (B, N, C)
        
        # Combine local and global features
        combined = torch.cat([features, global_feat], dim=-1)  # (B, N, 2C)
        
        # Output
        out = self.output_head(combined)  # (B, N, 1)
        
        return out


class PointTransformer_pressure(nn.Module):
    """Point Transformer wrapper for pressure prediction"""
    def __init__(self, args):
        super(PointTransformer_pressure, self).__init__()
        self.args = args
        
        # Model parameters
        self.hidden_dim = args.get('hidden_dim', 128)
        self.num_layers = args.get('num_layers', 12)
        self.num_neighbors = args.get('num_neighbors', 16)
        self.dropout = args.get('dropout', 0.1)
        
        # Point Transformer model
        self.transformer = PointTransformer(
            input_dim=3,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            num_layers=self.num_layers,
            num_neighbors=self.num_neighbors,
            dropout=self.dropout
        )
        
    def forward(self, points):
        """
        Args:
            points: (B, 3, N) point cloud coordinates
        Returns:
            pressure: (B, N) pressure values at each point
        """
        # Transpose to (B, N, 3) for transformer
        points = points.transpose(1, 2)
        
        # Apply transformer
        pressure = self.transformer(points)  # (B, N, 1)
        
        # Squeeze last dimension
        pressure = pressure.squeeze(-1)  # (B, N)
        
        return pressure


class LightweightPointTransformer_pressure(nn.Module):
    """Lightweight Point Transformer for faster training"""
    def __init__(self, args):
        super(LightweightPointTransformer_pressure, self).__init__()
        self.args = args
        
        # Reduced parameters
        self.hidden_dim = args.get('hidden_dim', 64)
        self.num_layers = args.get('num_layers', 6)
        self.num_neighbors = args.get('num_neighbors', 8)
        self.dropout = args.get('dropout', 0.1)
        
        # Lightweight transformer
        self.transformer = PointTransformer(
            input_dim=3,
            hidden_dim=self.hidden_dim,
            output_dim=1,
            num_layers=self.num_layers,
            num_neighbors=self.num_neighbors,
            dropout=self.dropout
        )
        
    def forward(self, points):
        """Forward pass"""
        points = points.transpose(1, 2)
        pressure = self.transformer(points).squeeze(-1)
        return pressure