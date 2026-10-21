"""
Author: Mohamed Elrefaie
Email: mohamed.elrefaie@mit.edu
Project: DrivAerNet-Benchmark
Description: Large Transolver model architecture for pressure field prediction on the DrivAerNet++ dataset.
This is a larger variant of the Transolver model with increased capacity.
"""

import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
from timm.layers import trunc_normal_


class Physics_Attention_Irregular_Mesh(nn.Module):
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
        torch.nn.init.orthogonal_(self.in_project_slice.weight)
        
        self.to_q = nn.Linear(dim_head, dim_head, bias=False)
        self.to_k = nn.Linear(dim_head, dim_head, bias=False)
        self.to_v = nn.Linear(dim_head, dim_head, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # B N C
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
    def __init__(self, n_input, n_hidden, n_output, n_layers=1, act='gelu', res=True):
        super().__init__()
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.n_layers = n_layers
        self.res = res
        
        if act == 'gelu':
            act_fn = nn.GELU()
        elif act == 'tanh':
            act_fn = nn.Tanh()
        elif act == 'relu':
            act_fn = nn.ReLU()
        elif act == 'silu':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.GELU()
            
        self.linear_pre = nn.Sequential(nn.Linear(n_input, n_hidden), act_fn)
        self.linear_post = nn.Linear(n_hidden, n_output)
        self.linears = nn.ModuleList([
            nn.Sequential(nn.Linear(n_hidden, n_hidden), act_fn) 
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


class Transolver_block(nn.Module):
    def __init__(self, num_heads, hidden_dim, dropout, act='gelu', mlp_ratio=4,
                 last_layer=False, out_dim=1, slice_num=32):
        super().__init__()
        self.last_layer = last_layer
        self.ln_1 = nn.LayerNorm(hidden_dim)
        self.Attn = Physics_Attention_Irregular_Mesh(
            hidden_dim, heads=num_heads, dim_head=hidden_dim // num_heads,
            dropout=dropout, slice_num=slice_num
        )
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.mlp = MLP(hidden_dim, hidden_dim * mlp_ratio, hidden_dim, 
                      n_layers=0, res=False, act=act)
        
        if self.last_layer:
            self.ln_3 = nn.LayerNorm(hidden_dim)
            self.mlp2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, fx):
        fx = self.Attn(self.ln_1(fx)) + fx
        fx = self.mlp(self.ln_2(fx)) + fx
        if self.last_layer:
            return self.mlp2(self.ln_3(fx))
        return fx


class LargeTransolverPressure(nn.Module):
    """Large Transolver for pressure prediction - supports both surface-only (6D) and SDF mode (7D)
    
    Larger variant with:
    - n_hidden: 384 (vs 256 in standard)
    - n_layers: 7 (vs 5 in standard)
    - n_head: 12 (vs 8 in standard)
    - slice_num: 48 (vs 32 in standard)
    """
    
    def __init__(self, args):
        super().__init__()
        
        # Extract parameters - LARGER defaults
        self.n_hidden = args.get('n_hidden', 384)  # Larger hidden dimension
        n_layers = args.get('n_layers', 7)  # More layers
        n_head = args.get('n_head', 12)  # More attention heads
        dropout = args.get('dropout', 0.0)
        mlp_ratio = args.get('mlp_ratio', 2)  # Larger MLP ratio
        slice_num = args.get('slice_num', 48)  # More slices
        self.unified_pos = args.get('unified_pos', False)
        self.ref = args.get('ref', 8)
        use_sdf = args.get('use_sdf', False)  # surface-only (6D) or SDF mode (7D)
        
        # Feature encoding: 6D [x,y,z,nx,ny,nz] for surface+normals or 7D [x,y,z,sdf,nx,ny,nz] for SDF mode
        self.fun_dim = 7 if use_sdf else 6
        space_dim = 3
        
        if self.unified_pos:
            # Use unified position encoding like original
            input_dim = self.fun_dim + self.ref * self.ref * self.ref
        else:
            # Standard approach
            # For surface+normals (fun_dim=6): input is [x,y,z,nx,ny,nz] → input_dim = 6
            # For SDF mode (fun_dim=7): input is [x,y,z,sdf,nx,ny,nz] + [x,y,z] → input_dim = 10
            if use_sdf:
                input_dim = self.fun_dim + space_dim  # 7 + 3 = 10
            else:
                input_dim = self.fun_dim  # 6 (positions + normals, no duplication)
        
        self.preprocess = MLP(
            input_dim, 
            self.n_hidden * 2, 
            self.n_hidden, 
            n_layers=0, 
            res=False, 
            act='gelu'
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Transolver_block(
                num_heads=n_head,
                hidden_dim=self.n_hidden,
                dropout=dropout,
                act='gelu',
                mlp_ratio=mlp_ratio,
                out_dim=1,
                slice_num=slice_num,
                last_layer=(i == n_layers - 1)
            ) for i in range(n_layers)
        ])
        
        # Learnable placeholder for feature augmentation
        self.placeholder = nn.Parameter((1/self.n_hidden) * torch.rand(self.n_hidden, dtype=torch.float))
        
        self.initialize_weights()
    
    def initialize_weights(self):
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def get_grid(self, my_pos):
        """Get unified position encoding grid (like original implementation)"""
        # my_pos: [B, N, 3]
        batchsize = my_pos.shape[0]

        gridx = torch.tensor(np.linspace(-1.5, 1.5, self.ref), dtype=torch.float)
        gridx = gridx.reshape(1, self.ref, 1, 1, 1).repeat([batchsize, 1, self.ref, self.ref, 1])
        gridy = torch.tensor(np.linspace(0, 2, self.ref), dtype=torch.float)
        gridy = gridy.reshape(1, 1, self.ref, 1, 1).repeat([batchsize, self.ref, 1, self.ref, 1])
        gridz = torch.tensor(np.linspace(-4, 4, self.ref), dtype=torch.float)
        gridz = gridz.reshape(1, 1, 1, self.ref, 1).repeat([batchsize, self.ref, self.ref, 1, 1])
        
        grid_ref = torch.cat((gridx, gridy, gridz), dim=-1).to(my_pos.device)
        grid_ref = grid_ref.reshape(batchsize, self.ref ** 3, 3)  # B ref^3 3

        # Compute distances from each point to each grid point
        pos = torch.sqrt(
            torch.sum((my_pos[:, :, None, :] - grid_ref[:, None, :, :]) ** 2, dim=-1)
        ).reshape(batchsize, my_pos.shape[1], self.ref * self.ref * self.ref).contiguous()
        
        return pos
    
    def forward(self, features):
        """
        Args:
            features: [B, N, 6] (surface+normals) or [B, N, 7] (with SDF) 
                     - [x, y, z, nx, ny, nz] or [x, y, z, sdf, nx, ny, nz]
        Returns:
            pressure: [B, N] - pressure predictions
        """
        # Extract positions for unified position encoding
        positions = features[:, :, :3]  # [B, N, 3]
        
        if self.unified_pos:
            # Use unified position encoding
            unified_pos = self.get_grid(positions)  # [B, N, ref^3]
            x = torch.cat((features, unified_pos), dim=-1)
        else:
            # For surface+normals mode (fun_dim=6), features are [x,y,z,nx,ny,nz] so don't duplicate positions
            # For SDF mode (fun_dim=7), features are [x,y,z,sdf,nx,ny,nz] so positions provide additional context
            if self.fun_dim == 7:
                # SDF mode: concatenate features with positions for additional positional context
                x = torch.cat((features, positions), dim=-1)  # [B, N, 10]
            else:
                # Surface+normals: features already contain positions, no need to concatenate
                x = features  # [B, N, 6]
        
        # Preprocess input features
        fx = self.preprocess(x)
        
        # Add learnable placeholder (like in original)
        fx = fx + self.placeholder[None, None, :]
        
        # Pass through transformer blocks
        for block in self.blocks:
            fx = block(fx)
        
        # fx is already pressure from last layer
        if fx.dim() == 3 and fx.shape[-1] == 1:
            fx = fx.squeeze(-1)
        
        return fx
