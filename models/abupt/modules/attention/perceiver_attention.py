import einops
import torch
import torch.nn.functional as F
from torch import nn

from modules.rope import rope


class PerceiverAttention(nn.Module):
    """Perceiver style attention module. This module is similar to a cross-attention modules.

    Args:
        dim: Hidden dimension of the layer/module.
        num_heads: Number of attention heads. Defaults to 8.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        q_freqs: torch.Tensor,
        k_freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward function of the PerceiverAttention module.

        Args:
            q: Query tensor, shape (batch size, number of points/tokens, dim).
            kv: Key/value tensor, shape (batch size, number of latent tokens, dim).
            q_freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries.
            k_freqs: Frequencies for Rotary Positional Embedding (RoPE) of keys.

        Returns:
            (batch size, query sequence length, dim)
        """
        # project to attention space
        q = self.q(q)
        kv = self.kv(kv)

        # split per head
        q = einops.rearrange(
            q,
            "bs seqlen_q (num_heads head_dim) -> bs num_heads seqlen_q head_dim",
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        )
        k, v = einops.rearrange(
            kv,
            "bs seqlen_kv (two num_heads head_dim) -> two bs num_heads seqlen_kv head_dim",
            two=2,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        # rope
        q = rope(q, freqs=q_freqs)
        k = rope(k, freqs=k_freqs)

        # attn
        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)
        return x
