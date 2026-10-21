from typing import Any

import torch
from torch import nn

from modules.attention import PerceiverAttention
from modules.mlp import Mlp


class PerceiverBlock(nn.Module):
    """The PerceiverBlock takes different input tensors for the query and the key/value.

    Args:
        dim: Hidden dimension of the perceiver block.
        num_heads: Number of attention heads.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.norm1q = nn.LayerNorm(dim, eps=1e-6)
        self.norm1kv = nn.LayerNorm(dim, eps=1e-6)
        self.attn = PerceiverAttention(dim=dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, attn_kwargs: dict[str, Any] | None = None) -> torch.Tensor:
        """Forward pass of the PerceiverBlock.

        Args:
            q: Input tensor with shape (batch_size, num_q_tokens, dim) for the query representations.
            kv: Input tensor with shape (batch_size, num_kv_tokens, dim) for the key and value representations.
            attn_kwargs: Dict with arguments for the attention (such as rope frequencies). Defaults to None.

        Returns:
            (batch_size, num_q_tokens, dim)
        """
        q = q + self.attn(q=self.norm1q(q), kv=self.norm1kv(kv), **(attn_kwargs or {}))
        q = q + self.mlp(self.norm2(q))
        return q
