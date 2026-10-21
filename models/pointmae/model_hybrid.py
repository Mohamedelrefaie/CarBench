# models/pointmae/model_hybrid.py
"""
Hybrid architecture combining PointNet++, Transformers, and dense prediction
Specifically designed for pressure field prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math


def get_graph_feature(x, k=20, idx=None):
    """Construct edge features for each point"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    
    if idx is None:
        # Compute pairwise distance
        inner = -2*torch.matmul(x.transpose(2, 1), x)
        xx = torch.sum(x**2, dim=1, keepdim=True)
        pairwise_distance = -xx - inner - xx.transpose(2, 1)
        
        idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points
    idx = idx + idx_base
    idx = idx.view(-1)
    
    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    
    return feature


class EdgeConv(nn.Module):
    """EdgeConv layer for local feature aggregation"""
    def __init__(self, in_channels, out_channels, k=20):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels*2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )
        
    def forward(self, x):
        x = get_graph_feature(x, k=self.k)  # (batch_size, 2*in_channels, num_points, k)
        x = self.conv(x)
        x = x.max(dim=-1, keepdim=False)[0]  # (batch_size, out_channels, num_points)
        return x


class LocalFeatureExtractor(nn.Module):
    """Extract rich local geometric features"""
    def __init__(self, k=20):
        super().__init__()
        self.k = k
        
        # Progressive edge convolutions
        self.edge1 = EdgeConv(3, 64, k)
        self.edge2 = EdgeConv(64, 128, k)
        self.edge3 = EdgeConv(128, 256, k)
        
        # Additional processing
        self.conv1 = nn.Conv1d(256, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        
    def forward(self, x):
        # x: (B, 3, N)
        # Extract multi-scale edge features
        x1 = self.edge1(x)  # (B, 64, N)
        x2 = self.edge2(x1)  # (B, 128, N)
        x3 = self.edge3(x2)  # (B, 256, N)
        
        # Further processing
        x = F.relu(self.bn1(self.conv1(x3)))  # (B, 512, N)
        
        return x, [x1, x2, x3]  # Return features at multiple scales


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 3D points"""
    def __init__(self, d_model, max_freq=10):
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
        
    def forward(self, xyz):
        # xyz: (B, N, 3)
        B, N, _ = xyz.shape
        
        # Create frequency bands
        freq_bands = 2.0 ** torch.linspace(0, self.max_freq - 1, self.d_model // 6).to(xyz.device)
        
        # Apply sinusoidal encoding
        encoded = []
        for freq in freq_bands:
            for func in [torch.sin, torch.cos]:
                encoded.append(func(xyz * freq))
        
        # Concatenate all encodings
        pos_encoding = torch.cat(encoded, dim=-1)  # (B, N, d_model)
        
        return pos_encoding[:, :, :self.d_model]  # Ensure correct dimension


class TransformerBlock(nn.Module):
    """Transformer block with local-global attention"""
    def __init__(self, d_model, nhead=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = F.gelu
        
    def forward(self, x, pos_encoding=None):
        # Add positional encoding
        if pos_encoding is not None:
            q = k = x + pos_encoding
        else:
            q = k = x
            
        # Self-attention
        x2 = self.self_attn(q, k, x)[0]
        x = x + self.dropout1(x2)
        x = self.norm1(x)
        
        # FFN
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout2(x2)
        x = self.norm2(x)
        
        return x


class HybridPressureNet(nn.Module):
    """Hybrid architecture for accurate pressure prediction"""
    def __init__(self, num_points=10000, k=20, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        # Local feature extraction (DGCNN-style)
        self.local_feat = LocalFeatureExtractor(k=k)
        
        # Project to transformer dimension
        self.proj = nn.Conv1d(512, d_model, 1)
        self.proj_bn = nn.BatchNorm1d(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model*4, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Multi-scale decoder
        self.decoder = nn.ModuleList([
            nn.Conv1d(d_model + 256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256 + 128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128 + 64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        ])
        
        # Final pressure prediction
        self.pressure_head = nn.Sequential(
            nn.Conv1d(64 + 3, 32, 1),  # Include original coordinates
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        # x: (B, 3, N) or (B, N, 3)
        if x.dim() == 3 and x.shape[-1] == 3:
            xyz = x
            x = x.transpose(1, 2)
        else:
            xyz = x.transpose(1, 2)
            
        B, _, N = x.shape
        
        # Extract local features
        local_feats, multi_scale_feats = self.local_feat(x)  # (B, 512, N)
        
        # Project to transformer dimension
        feat = F.relu(self.proj_bn(self.proj(local_feats)))  # (B, d_model, N)
        feat = feat.transpose(1, 2)  # (B, N, d_model)
        
        # Get positional encoding
        pos_enc = self.pos_encoder(xyz)  # (B, N, d_model)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            feat = layer(feat, pos_enc)
        
        # Transpose back for conv1d
        feat = feat.transpose(1, 2)  # (B, d_model, N)
        
        # Multi-scale decoding
        # Combine with features from encoder
        feat = torch.cat([feat, multi_scale_feats[2]], dim=1)  # Add 256-d features
        feat = self.decoder[2](self.decoder[1](self.decoder[0](feat)))
        
        feat = torch.cat([feat, multi_scale_feats[1]], dim=1)  # Add 128-d features
        feat = self.decoder[5](self.decoder[4](self.decoder[3](feat)))
        
        feat = torch.cat([feat, multi_scale_feats[0]], dim=1)  # Add 64-d features
        feat = self.decoder[8](self.decoder[7](self.decoder[6](feat)))
        
        # Final prediction with coordinate information
        feat = torch.cat([feat, x], dim=1)  # Add original coordinates
        pressure = self.pressure_head(feat).squeeze(1)  # (B, N)
        
        return pressure


class PointNetPlusPlusReg(nn.Module):
    """PointNet++ specifically for regression tasks"""
    def __init__(self, num_points=10000):
        super().__init__()
        
        # Set Abstraction layers
        self.sa1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.sa2 = nn.Sequential(
            nn.Conv1d(128, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.sa3 = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Global feature
        self.global_feat = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # Feature Propagation layers
        self.fp3 = nn.Sequential(
            nn.Conv1d(1024 + 512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        self.fp2 = nn.Sequential(
            nn.Conv1d(256 + 256, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.fp1 = nn.Sequential(
            nn.Conv1d(128 + 128 + 3, 128, 1),  # Include coordinates
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv1d(64, 1, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        if x.dim() == 3 and x.shape[-1] == 3:
            xyz = x.transpose(1, 2)
            x = xyz
        else:
            xyz = x
            
        B, _, N = x.shape
        
        # Set Abstraction
        l1 = self.sa1(x)  # (B, 128, N)
        l2 = self.sa2(l1)  # (B, 256, N)
        l3 = self.sa3(l2)  # (B, 512, N)
        
        # Global feature
        global_feat = self.global_feat(l3)
        global_feat = F.max_pool1d(global_feat, N)
        global_feat = global_feat.repeat(1, 1, N)
        
        # Feature Propagation
        fp3 = self.fp3(torch.cat([l3, global_feat], 1))  # (B, 256, N)
        fp2 = self.fp2(torch.cat([l2, fp3], 1))  # (B, 128, N)
        fp1 = self.fp1(torch.cat([l1, fp2, xyz], 1))  # (B, 1, N)
        
        return fp1.squeeze(1)


# Lightweight but effective model
class EfficientPressureNet(nn.Module):
    """Efficient architecture balancing speed and accuracy"""
    def __init__(self, num_points=10000):
        super().__init__()
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        
        # Local pattern learning
        self.local_conv = nn.Sequential(
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # Global context
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        
        # Pressure regression head
        self.regressor = nn.Sequential(
            nn.Conv1d(1024 + 256 + 3, 512, 1),  # Local + skip + coords
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(512, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1, 1)
        )
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        if x.dim() == 3 and x.shape[-1] == 3:
            x = x.transpose(1, 2)
            
        B, _, N = x.shape
        xyz = x
        
        # Encode
        feat_enc = self.encoder(x)  # (B, 256, N)
        
        # Local patterns
        feat_local = self.local_conv(feat_enc)  # (B, 1024, N)
        
        # Global context
        feat_global = self.global_pool(feat_local)  # (B, 1024, 1)
        feat_global = feat_global.repeat(1, 1, N)  # (B, 1024, N)
        
        # Mix local and global
        feat_mixed = feat_local * torch.sigmoid(feat_global)  # Gated fusion
        
        # Concatenate all information
        feat_all = torch.cat([feat_mixed, feat_enc, xyz], dim=1)
        
        # Predict pressure
        pressure = self.regressor(feat_all).squeeze(1)
        
        return pressure


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test models
    models = {
        "HybridPressureNet": HybridPressureNet(num_points=10000, k=20, d_model=256, num_layers=4),
        "PointNetPlusPlusReg": PointNetPlusPlusReg(num_points=10000),
        "EfficientPressureNet": EfficientPressureNet(num_points=10000)
    }
    
    x = torch.randn(2, 10000, 3).to(device)
    
    for name, model in models.items():
        model = model.to(device)
        print(f"\nTesting {name}...")
        
        with torch.no_grad():
            output = model(x)
            
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
        
        # Parameter count
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Parameters: {params:,}")
        
        # Measure inference time
        import time
        model.eval()
        times = []
        for _ in range(10):
            start = time.time()
            with torch.no_grad():
                _ = model(x)
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            times.append(time.time() - start)
        
        print(f"Inference time: {np.mean(times)*1000:.2f} ms")