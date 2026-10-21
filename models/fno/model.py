# models/fno/model.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Optimized Fourier Neural Operator (FNO) for pressure field prediction on 3D point clouds.
Adapted for the DrivAerNet++ dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralConv3d(nn.Module):
    """3D Fourier layer for FNO - Optimized version"""
    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        # Use a single weight tensor for all modes to reduce memory
        self.weights = nn.Parameter(self.scale * torch.randn(4, in_channels, out_channels, self.modes1, self.modes2, self.modes3, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        # Ensure input is float32
        x = x.float()
        
        # Compute Fourier coefficients
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Initialize output
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1)//2 + 1, 
                           dtype=x_ft.dtype, device=x.device)
        
        # Convert weights to complex
        weights_complex = torch.view_as_complex(self.weights)
        
        # Apply spectral convolution more efficiently
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], weights_complex[0])
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], weights_complex[1])
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], weights_complex[2])
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            torch.einsum("bixyz,ioxyz->boxyz", x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], weights_complex[3])

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class PointCloudToGrid(nn.Module):
    """Optimized point cloud to voxel grid conversion"""
    def __init__(self, grid_size=32):
        super(PointCloudToGrid, self).__init__()
        self.grid_size = grid_size
        
    def forward(self, points, features=None):
        """
        Args:
            points: (B, 3, N) point coordinates
            features: (B, C, N) point features (optional)
        Returns:
            grid: (B, C, H, W, D) voxel grid
        """
        B, _, N = points.shape
        device = points.device
        
        # Normalize points to [0, grid_size-1]
        points_min = points.min(dim=2, keepdim=True)[0]
        points_max = points.max(dim=2, keepdim=True)[0]
        points_norm = (points - points_min) / (points_max - points_min + 1e-8)
        points_norm = points_norm * (self.grid_size - 1)
        
        # Convert to indices
        indices = points_norm.long()
        indices = torch.clamp(indices, 0, self.grid_size - 1)
        
        # Initialize features if not provided
        if features is None:
            features = torch.ones(B, 1, N, device=device)
        
        C = features.shape[1]
        
        # Optimized voxelization using index_add_
        grid = torch.zeros(B, C, self.grid_size, self.grid_size, self.grid_size, device=device)
        
        # Process each batch
        for b in range(B):
            # Flatten the grid for efficient indexing
            grid_flat = grid[b].view(C, -1)  # (C, H*W*D)
            
            # Compute flat indices
            flat_indices = (indices[b, 0] * self.grid_size * self.grid_size + 
                          indices[b, 1] * self.grid_size + 
                          indices[b, 2])  # (N,)
            
            # Use index_add_ for efficient aggregation
            for c in range(C):
                grid_flat[c].index_add_(0, flat_indices, features[b, c])
            
            # Reshape back
            grid[b] = grid_flat.view(C, self.grid_size, self.grid_size, self.grid_size)
        
        return grid


class GridToPointCloud(nn.Module):
    """Optimized grid to point cloud sampling"""
    def __init__(self):
        super(GridToPointCloud, self).__init__()
        
    def forward(self, grid, points):
        """
        Args:
            grid: (B, C, H, W, D) voxel grid
            points: (B, 3, N) original point coordinates
        Returns:
            features: (B, C, N) sampled features
        """
        B, C, H, W, D = grid.shape
        _, _, N = points.shape
        device = grid.device
        
        # Ensure both tensors have the same dtype
        grid = grid.float()
        points = points.float()
        
        # Normalize points
        points_min = points.min(dim=2, keepdim=True)[0]
        points_max = points.max(dim=2, keepdim=True)[0]
        points_norm = (points - points_min) / (points_max - points_min + 1e-8)
        
        # Convert to grid coordinates [-1, 1]
        points_grid = points_norm * 2 - 1  # (B, 3, N)
        
        # Prepare for grid_sample: (B, N, 1, 1, 3)
        points_grid = points_grid.transpose(1, 2).unsqueeze(2).unsqueeze(2)
        
        # Sample from grid
        sampled = F.grid_sample(grid, points_grid, mode='bilinear', align_corners=True)
        
        # Reshape to (B, C, N)
        features = sampled.squeeze(-1).squeeze(-1)
        
        return features


class FNO3d(nn.Module):
    """Lightweight 3D Fourier Neural Operator"""
    def __init__(self, modes, width, in_channels=4, out_channels=1):
        super(FNO3d, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.modes3 = modes
        self.width = width
        
        self.fc0 = nn.Linear(in_channels, self.width)
        
        # Reduced number of Fourier layers for efficiency
        self.conv0 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        self.conv1 = SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3)
        
        self.w0 = nn.Conv3d(self.width, self.width, 1)
        self.w1 = nn.Conv3d(self.width, self.width, 1)
        
        self.fc1 = nn.Linear(self.width, out_channels)

    def forward(self, x):
        # x: (B, H, W, D, in_channels)
        B, H, W, D, C = x.shape
        
        # Lift to higher dimension
        x = self.fc0(x)  # (B, H, W, D, width)
        
        # Transform to (B, C, H, W, D) for convolutions
        x = x.permute(0, 4, 1, 2, 3)
        
        # Fourier layers with residual connections
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        
        # Project to output
        x = x.permute(0, 2, 3, 4, 1)  # (B, H, W, D, width)
        x = self.fc1(x)  # (B, H, W, D, out_channels)
        
        # Return to (B, out_channels, H, W, D)
        x = x.permute(0, 4, 1, 2, 3)
        
        return x


class FNO_pressure(nn.Module):
    """Optimized FNO for pressure prediction on point clouds"""
    def __init__(self, args):
        super(FNO_pressure, self).__init__()
        self.args = args
        
        # Reduced grid size for efficiency
        self.grid_size = args.get('grid_size', 32)  # Reduced from 64
        self.modes = args.get('modes', 8)  # Reduced from 12
        self.width = args.get('width', 16)  # Reduced from 32
        
        # Point cloud to grid
        self.point_to_grid = PointCloudToGrid(grid_size=self.grid_size)
        
        # FNO layers
        self.fno = FNO3d(modes=self.modes, width=self.width, in_channels=4, out_channels=1)
        
        # Grid to point cloud
        self.grid_to_point = GridToPointCloud()
        
        # Simplified refinement network
        self.refine = nn.Sequential(
            nn.Conv1d(1, 32, 1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 16, 1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 1, 1)
        )
        
    def get_grid_coordinates(self, B, H, W, D, device):
        """Get normalized grid coordinates"""
        if not hasattr(self, '_cached_coords') or self._cached_coords.shape[0] != B:
            x = torch.linspace(0, 1, H, device=device)
            y = torch.linspace(0, 1, W, device=device)
            z = torch.linspace(0, 1, D, device=device)
            
            xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
            coords = torch.stack([xx, yy, zz], dim=-1)
            self._cached_coords = coords.unsqueeze(0).expand(B, -1, -1, -1, -1)
        
        return self._cached_coords[:B]
        
    def forward(self, points):
        """
        Args:
            points: (B, 3, N) point cloud coordinates
        Returns:
            pressure: (B, N) pressure values at each point
        """
        B, _, N = points.shape
        device = points.device
        
        # Convert point cloud to grid (disable AMP for grid operations)
        with torch.cuda.amp.autocast(enabled=False):
            points = points.float()  # Ensure float32
            grid = self.point_to_grid(points)  # (B, 1, H, W, D)
        
        # Add coordinate channels
        H, W, D = grid.shape[2:]
        coords = self.get_grid_coordinates(B, H, W, D, device)  # (B, H, W, D, 3)
        
        # Combine occupancy and coordinates
        grid_input = torch.cat([
            grid.permute(0, 2, 3, 4, 1),  # (B, H, W, D, 1)
            coords  # (B, H, W, D, 3)
        ], dim=-1)  # (B, H, W, D, 4)
        
        # Apply FNO
        pressure_grid = self.fno(grid_input)  # (B, 1, H, W, D)
        
        # Sample back to point cloud (disable AMP for sampling)
        with torch.cuda.amp.autocast(enabled=False):
            pressure_grid = pressure_grid.float()
            points = points.float()
            pressure_points = self.grid_to_point(pressure_grid, points)  # (B, 1, N)
        
        # Refinement network
        pressure_refined = self.refine(pressure_points)
        
        # Output shape: (B, N)
        pressure = pressure_refined.squeeze(1)
        
        return pressure