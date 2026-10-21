# models/pointmae/model.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Efficient encoder-decoder model for pressure field prediction on DrivAerNet++
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EfficientPressureNet(nn.Module):
    """Efficient model balancing speed and accuracy"""
    def __init__(self, num_points=10000):
        super().__init__()
        
        # Efficient encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        
        # Global and local processing
        self.global_feat = nn.Sequential(
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(1024 + 512 + 3, 512, 1),
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
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
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
            x = x.transpose(1, 2)
            
        B, _, N = x.shape
        xyz = x
        
        # Encode
        local_feat = self.encoder(x)  # [B, 512, N]
        
        # Global feature
        global_feat = self.global_feat(local_feat)  # [B, 1024, N]
        global_max = F.max_pool1d(global_feat, N)   # [B, 1024, 1]
        global_max = global_max.repeat(1, 1, N)     # [B, 1024, N]
        
        # Combine features
        combined = torch.cat([global_max, local_feat, xyz], dim=1)
        
        # Decode
        pressure = self.decoder(combined).squeeze(1)
        
        return pressure


class SimplePressureNet(nn.Module):
    """Simple but effective baseline"""
    def __init__(self, num_points=10000):
        super().__init__()
        
        # Encoder
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.conv5 = nn.Conv1d(512, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1024)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv1d(1024 + 512 + 3, 512, 1),
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
            nn.Conv1d(128, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
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
            x = x.transpose(1, 2)
            
        B, _, N = x.shape
        xyz = x
        
        # Encode
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        x5 = F.relu(self.bn5(self.conv5(x4)))
        
        # Global max pooling
        global_feat = F.max_pool1d(x5, N)
        global_feat = global_feat.repeat(1, 1, N)
        
        # Concatenate
        concat_feat = torch.cat([global_feat, x4, xyz], dim=1)
        
        # Decode
        pressure = self.decoder(concat_feat).squeeze(1)
        
        return pressure


class PointMAEForPressure(nn.Module):
    """Main model that selects architecture"""
    def __init__(self, 
                 num_points: int = 10000,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 num_group: int = 128,
                 group_size: int = 32,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 use_pretrained: bool = True,
                 pretrained_path: Optional[str] = None,
                 architecture: str = "pointmae"):
        super().__init__()
        
        self.architecture = architecture
        self.num_points = num_points
        
        # Create the appropriate model
        if architecture == "hybrid":
            self.model = HybridPressureNet(
                num_points=num_points,
                k=20,
                d_model=min(embed_dim, 256),
                nhead=num_heads,
                num_layers=min(depth, 6)
            )
        elif architecture == "pointnet++":
            self.model = PointNetPlusPlusReg(num_points=num_points)
        elif architecture == "efficient":
            self.model = EfficientPressureNet(num_points=num_points)
        else:
            # Default to simple but effective model
            self.model = SimplePressureNet(num_points=num_points)
            if architecture != "pointmae":
                print(f"Architecture '{architecture}' not recognized. Using SimplePressureNet.")
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class PointMAELightweight(nn.Module):
    """Lightweight model"""
    def __init__(self, num_points: int = 10000):
        super().__init__()
        self.model = EfficientPressureNet(num_points)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# Test code
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 10000, 3).to(device)
    
    print("Testing all architectures...")
    
    architectures = {
        "hybrid": "HybridPressureNet",
        "pointnet++": "PointNetPlusPlusReg", 
        "efficient": "EfficientPressureNet",
        "pointmae": "SimplePressureNet"
    }
    
    for arch_key, arch_name in architectures.items():
        print(f"\n{'='*50}")
        print(f"Testing {arch_name} (key: {arch_key})")
        print('='*50)
        
        try:
            model = PointMAEForPressure(
                num_points=10000,
                architecture=arch_key,
                embed_dim=256,
                depth=4,
                use_pretrained=False
            ).to(device)
            
            # Test forward pass
            with torch.no_grad():
                output = model(x)
                
            print(f"✓ Input shape: {x.shape}")
            print(f"✓ Output shape: {output.shape}")
            print(f"✓ Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
            
            # Count parameters
            params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"✓ Parameters: {params:,}")
            
            # Test with batch of different size
            x_small = torch.randn(4, 5000, 3).to(device)
            with torch.no_grad():
                output_small = model(x_small)
            print(f"✓ Works with different batch/point sizes: {output_small.shape}")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Test lightweight
    print(f"\n{'='*50}")
    print("Testing PointMAELightweight")
    print('='*50)
    
    model_light = PointMAELightweight(num_points=10000).to(device)
    with torch.no_grad():
        output = model_light(x)
    print(f"✓ Output shape: {output.shape}")
    params = sum(p.numel() for p in model_light.parameters() if p.requires_grad)
    print(f"✓ Parameters: {params:,}")