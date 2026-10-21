import einops
import torch


def rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """RoPE via polar coordinate rotations."""
    # adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py#L65
    assert torch.is_tensor(freqs) and torch.is_complex(freqs)
    assert x.ndim == 4, "x.shape should be (batch_size, num_heads, seqlen, head_dim)"
    assert freqs.ndim == 3, "freqs.shape should be (batch_size, seqlen, head_dim // 2)"
    x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    # add dim for num_heads
    freqs = einops.rearrange(freqs, "batch_size seqlen head_dim -> batch_size 1 seqlen head_dim")
    x_out = torch.view_as_real(x_ * freqs).flatten(start_dim=3)
    return x_out.type_as(x)
