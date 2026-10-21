import einops
import torch
import torch.nn.functional as F
from torch import nn

from modules.rope import rope


class DotProductAttention(nn.Module):
    """Scaled dot-product attention module.

    Args:
        dim: Input dimension of the attention module.
        num_heads: Number of attention heads. Defaults to 8.
    """

    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        """Forward function of the DotProductAttention module.

        Args:
            x: Tensor to apply self-attention over, shape (batch size, sequence length, dim).
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys.

        Returns:
            (batch_size, sequence_length, dim)
        """

        q, k, v = einops.rearrange(
            self.qkv(x),
            "bs seqlen (three num_heads head_dim) -> three bs num_heads seqlen head_dim",
            three=3,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
        ).unbind(0)

        q = rope(q, freqs=freqs)
        k = rope(k, freqs=freqs)

        x = F.scaled_dot_product_attention(q, k, v)
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)

        return x
