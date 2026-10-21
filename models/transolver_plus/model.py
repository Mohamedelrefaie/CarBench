# models/transolver_plus/model.py
"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description:  Transolver++ for pressure field prediction on 3D point clouds with normals.
Adapted for the DrivAerNet++ dataset with 6D input features [x, y, z, nx, ny, nz] or 7D [x, y, z, sdf, nx, ny, nz].
FIXED: Corrected input dimension handling for both 6D and 7D modes
"""

import torch
import numpy as np
import torch.nn as nn
from timm.layers import trunc_normal_
from einops import rearrange
import torch.nn.functional as F
from typing import Optional, Dict, Any


ACTIVATION = {
    'gelu': nn.GELU, 
    'tanh': nn.Tanh, 
    'sigmoid': nn.Sigmoid, 
    'relu': nn.ReLU, 
    'leaky_relu': nn.LeakyReLU(0.1),
    'softplus': nn.Softplus, 
    'ELU': nn.ELU, 
    'silu': nn.SiLU
}


def gumbel_softmax(logits, tau=1, hard=False):
    """Gumbel softmax for differentiable discrete sampling"""
    u = torch.rand_like(logits)
    gumbel_noise = -torch.log(-torch.log(u + 1e-8) + 1e-8)
    
    y = logits + gumbel_noise
    y = y / tau
    
    y = F.softmax(y, dim=-1)
    
    if hard:
        _, y_hard = y.max(dim=-1)
        y_one_hot = torch.zeros_like(y).scatter_(-1, y_hard.unsqueeze(-1), 1.0)
        y = (y_one_hot - y).detach() + y
    return y


class Physics_Attention_1D(nn.Module):
    """Physics-informed attention mechanism (single GPU version)"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., slice_num=64):
        super().__init__()
        inner_dim = dim_head * heads
        self.dim_head = dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = nn.Parameter(torch.ones([1, heads, 1, 1]) * 0.5)
        
        self.in_project_x = nn.Linear(dim, inner_dim)
        self.in_project_fx = nn.Linear(dim, inner_dim)
        self.in_project_slice = nn.Linear(dim_head, slice_num)
        
        # Initialize with orthogonal weights
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, N, C) input features
        Returns:
            (B, N, C) output features
        """
        B, N, C = x.shape

        ### (1) Slice
        fx_mid = self.in_project_fx(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        x_mid = self.in_project_x(x).reshape(B, N, self.heads, self.dim_head) \
            .permute(0, 2, 1, 3).contiguous()  # B H N C
        slice_weights = self.softmax(self.in_project_slice(x_mid) / self.temperature)  # B H N G
        slice_norm = slice_weights.sum(2)  # B H G
        slice_token = torch.einsum("bhnc,bhng->bhgc", fx_mid, slice_weights)
        slice_token = slice_token / ((slice_norm + 1e-5)[:, :, :, None].repeat(1, 1, 1, self.dim_head))

        ### (2) Attention among slice tokens
        q_slice_token = self.to_q(slice_token)
        k_slice_token = self.to_k(slice_token)
        v_slice_token = self.to_v(slice_token)
        dots = torch.matmul(q_slice_token, k_slice_token.transpose(-1, -2)) * self.scale
        attn = self.softmax(dots)
        attn = self.dropout(attn)
        out_slice_token = torch.matmul(attn, v_slice_token)  # B H G D

        ### (3) Deslice
        out_x = torch.einsum("bhgc,bhng->bhnc", out_slice_token, slice_weights)
        out_x = rearrange(out_x, 'b h n d -> b n (h d)')
        return self.to_out(out_x)


class MLP(nn.Module):
    """Multi-layer perceptron with optional residual connections"""
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super(MLP, self).__init__()
        
        if act in ACTIVATION.keys():
            act = ACTIVATION[act]
        else:
            raise NotImplementedError(f"Activation {act} not implemented")
            
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act())
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act()) 
            for _ in range(n_layers)
        ])
        
    def forward(self, x):
        x = self.linear_pre(x)
        for i in range(self.n_layers):
            if self.res:
                x = self.linears[i](x) + x
            else:
                x = self.linears[i](x)
        x = self.linear_post(x)
        return x


class Transolver_plus_block(nn.Module):
    """Transolver++ transformer block"""
    def __init__(
            self,
            num_heads: int,
            hidden_dim: int,
            dropout: float,
            act='gelu',
            mlp_ratio=4,
            last_layer=False,
            out_dim=1,
            slice_num=32,
    ):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.attn = Physics_Attention_1D(
            hidden_dim, 
            heads=num_heads, 
            dim_head=hidden_dim // num_heads,
            dropout=dropout, 
            slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(
            hidden_dim, 
            hidden_dim * mlp_ratio, 
            hidden_dim, 
            n_layers=0, 
            res=False, 
            act=act
        )
        
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)
            
    def forward(self, fx):
        # Attention block
        fx = fx + self.attn(self.ln_1(fx))
        
        # MLP block
        fx = fx + self.mlp(self.ln_2(fx))
        
        # Output projection if last layer
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        else:
            return fx


class Transolver_plus(nn.Module):
    """Transolver++ model for 3D point cloud processing with configurable input features"""
    def __init__(self,
                 space_dim=3,
                 n_layers=5,
                 n_hidden=256,
                 dropout=0,
                 n_head=8,
                 act='gelu',
                 mlp_ratio=1,
                 fun_dim=7,  # Will be overridden based on use_sdf
                 out_dim=1,
                 slice_num=32,
                 ref=8,
                 unified_pos=False,
                 use_sdf=False  # Control SDF features
                 ):
        super(Transolver_plus, self).__init__()
        self.__name__ = 'Transolver_plus_3D'
        self.ref = ref
        self.unified_pos = unified_pos
        self.n_hidden = n_hidden
        self.space_dim = space_dim
        self.use_sdf = use_sdf
        
        # Set fun_dim based on use_sdf flag
        # use_sdf=False: 6D [x,y,z,nx,ny,nz] surface+normals only (default)
        # use_sdf=True: 7D [x,y,z,sdf,nx,ny,nz] with interior points
        self.fun_dim = 7 if use_sdf else 6
        
        # Input preprocessing
        if self.unified_pos:
            # Use unified position encoding + features
            input_dim = self.fun_dim + self.ref * self.ref * self.ref
        else:
            # Standard: concatenate features with positions for position awareness
            # When use_sdf=False: 6D features + 3D positions = 9D
            # When use_sdf=True: 7D features + 3D positions = 10D
            # Both modes concatenate positions to maintain position-awareness in the network
            input_dim = self.fun_dim + space_dim  # 6+3=9D or 7+3=10D
            
        self.preprocess = MLP(
            input_dim, 
            n_hidden * 2, 
            n_hidden, 
            n_layers=0, 
            res=False, 
            act=act
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Transolver_plus_block(
                num_heads=n_head, 
                hidden_dim=n_hidden,
                dropout=dropout,
                act=act,
                mlp_ratio=mlp_ratio,
                out_dim=out_dim,
                slice_num=slice_num,
                last_layer=(_ == n_layers - 1)
            )
            for _ in range(n_layers)
        ])
        
        # Learnable placeholder embedding
        self.placeholder = nn.Parameter(
            (1 / n_hidden) * torch.rand(n_hidden, dtype=torch.float)
        )
        
        self.initialize_weights()
        
    def initialize_weights(self):
        """Initialize model weights"""
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def get_grid(self, my_pos):
        """Generate reference grid for unified position encoding"""
        batchsize = my_pos.shape[0]
        
        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).to(my_pos.device)
        grid_ref = grid_ref.reshape(batchsize, self.ref ** 3, 3)
        
        # Compute distances to reference points
        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)
        ).reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        
        return pos
        
    def forward(self, features):
        """
        Args:
            features: (B, N, 6) or (B, N, 7) input features
                      6D: [x, y, z, nx, ny, nz] for surface+normals (default)
                      7D: [x, y, z, sdf, nx, ny, nz] for SDF mode
        Returns:
            (B, N, out_dim) predictions
        """
        # Extract positions for position encoding
        positions = features[:, :, :3]  # [B, N, 3]
        
        # Prepare input
        if self.unified_pos:
            # Use unified position encoding
            unified_pos = self.get_grid(positions)  # [B, N, ref^3]
            x = torch.cat((features, unified_pos), dim=-1)
        else:
            # Standard approach: always concatenate positions for position-awareness
            # This ensures the network knows the spatial location of each point
            x = torch.cat((features, positions), dim=-1)  # [B, N, 9] or [B, N, 10]
            
        # Preprocess input
        fx = self.preprocess(x)
        fx = fx + self.placeholder[None, None, :]
        
        # Apply transformer blocks
        for block in self.blocks:
            fx = block(fx)
            
        return fx


class TransolverPlusPressure(nn.Module):
    """Transolver++ wrapper for pressure prediction on DrivAerNet++ with configurable features"""
    def __init__(self, args):
        super(TransolverPlusPressure, self).__init__()
        self.args = args
        
        # Model parameters
        self.n_hidden = args.get('n_hidden', 256)
        self.n_layers = args.get('n_layers', 5)
        self.n_head = args.get('n_head', 8)
        self.mlp_ratio = args.get('mlp_ratio', 1)
        self.slice_num = args.get('slice_num', 32)
        self.dropout = args.get('dropout', 0.0)
        self.unified_pos = args.get('unified_pos', False)
        self.ref = args.get('ref', 8)
        self.use_sdf = args.get('use_sdf', False)  # Default: False for 6D mode
        
        # Initialize Transolver++ model with configurable features
        self.model = Transolver_plus(
            space_dim=3,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout=self.dropout,
            n_head=self.n_head,
            act='gelu',
            mlp_ratio=self.mlp_ratio,
            fun_dim=7 if self.use_sdf else 6,  # 6D [x,y,z,nx,ny,nz] or 7D [x,y,z,sdf,nx,ny,nz]
            out_dim=1,
            slice_num=self.slice_num,
            ref=self.ref,
            unified_pos=self.unified_pos,
            use_sdf=self.use_sdf  # Pass use_sdf flag
        )
        
    def forward(self, features):
        """
        Args:
            features: (B, N, 6) or (B, N, 7) input features
                      6D: [x, y, z, nx, ny, nz] for surface+normals (default)
                      7D: [x, y, z, sdf, nx, ny, nz] for SDF mode
        Returns:
            pressure: (B, N) pressure values at each point
        """
        # Apply model
        pressure = self.model(features)  # (B, N, 1)
        
        # Squeeze last dimension
        if pressure.dim() == 3 and pressure.shape[-1] == 1:
            pressure = pressure.squeeze(-1)  # (B, N)
        
        return pressure


class LightweightTransolverPlusPressure(nn.Module):
    """Lightweight Transolver++ for faster training with configurable features"""
    def __init__(self, args):
        super(LightweightTransolverPlusPressure, self).__init__()
        self.args = args
        
        # Reduced parameters for lightweight version
        self.n_hidden = args.get('n_hidden', 128)
        self.n_layers = args.get('n_layers', 3)
        self.n_head = args.get('n_head', 4)
        self.mlp_ratio = args.get('mlp_ratio', 1)
        self.slice_num = args.get('slice_num', 16)
        self.dropout = args.get('dropout', 0.0)
        self.unified_pos = args.get('unified_pos', False)
        self.ref = args.get('ref', 8)
        self.use_sdf = args.get('use_sdf', False)  # Default: False for 6D mode
        
        # Initialize lightweight model with configurable features
        self.model = Transolver_plus(
            space_dim=3,
            n_layers=self.n_layers,
            n_hidden=self.n_hidden,
            dropout=self.dropout,
            n_head=self.n_head,
            act='gelu',
            mlp_ratio=self.mlp_ratio,
            fun_dim=7 if self.use_sdf else 6,  # 6D [x,y,z,nx,ny,nz] or 7D [x,y,z,sdf,nx,ny,nz]
            out_dim=1,
            slice_num=self.slice_num,
            ref=self.ref,
            unified_pos=self.unified_pos,
            use_sdf=self.use_sdf  # Pass use_sdf flag
        )
        
    def forward(self, features):
        """Forward pass with configurable features"""
        pressure = self.model(features)
        if pressure.dim() == 3 and pressure.shape[-1] == 1:
            pressure = pressure.squeeze(-1)
        return pressure