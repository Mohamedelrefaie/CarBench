import einops
import torch
import torch.nn.functional as F

from modules.rope import rope
from modules.attention import DotProductAttention


class SharedweightsCrossattnAttention(DotProductAttention):
    def forward(
        self,
        x: torch.Tensor,
        split_size: list[int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Attention between:
        - q=surface_anchors -> kv=volume_anchors
        - q=volume_anchors -> kv=surface_anchors
        - q=surface_queries -> kv=volume_anchors
        - q=volume_queries -> kv=surface_anchors

        Args:
            x: Tensor containing all anchors/queries (batch size, sequence length, dim).
            split_size: How to split x into:
                len(split_size) == 2: (surface_anchors, volume_anchors)
                len(split_size) == 4: (surface_anchors, surface_queries, volume_anchors, volume_queries)
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys.

        Returns:
            (batch size, sequence length, dim)
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

        # split_size are (surface_anchors, surface_queries, volume_anchors, volume_queries)
        # len(split_size) == 2: (surface_anchors, volume_anchors) -> i.e., no queries
        # len(split_size) == 4: (surface_anchors, surface_queries, volume_anchors, volume_queries)
        # (could potentially be faster by skipping the kv parts of the qkv for the auxiliary splits)
        ks = k.split(split_size, dim=2)
        vs = v.split(split_size, dim=2)
        if isinstance(split_size, list) and len(split_size) == 4:
            # surface + volume queries
            qs = q.split([split_size[0] + split_size[1], split_size[2] + split_size[3]], dim=2)
            x1 = F.scaled_dot_product_attention(qs[0], ks[2], vs[2])
            x2 = F.scaled_dot_product_attention(qs[1], ks[0], vs[0])
            x = torch.concat([x1, x2], dim=2)
        else:
            if isinstance(split_size, list) and len(split_size) == 2 and split_size[0] == split_size[1]:
                # efficient implementation when both splits are equally sized
                q = einops.rearrange(
                    q,
                    "batch_size num_heads (two seqlen) head_dim -> (two batch_size) num_heads seqlen head_dim",
                    two=2,
                )
                k = torch.concat([ks[1], ks[0]])
                v = torch.concat([vs[1], vs[0]])
                x = F.scaled_dot_product_attention(q, k, v)
                x = einops.rearrange(
                    x,
                    "(two batch_size) num_heads seqlen head_dim -> batch_size num_heads (two seqlen) head_dim",
                    two=2,
                )
            elif isinstance(split_size, list) and len(split_size) == 2:
                # generic implementation for two sized splits
                qs = q.split(split_size, dim=2)
                x1 = F.scaled_dot_product_attention(qs[0], ks[1], vs[1])
                x2 = F.scaled_dot_product_attention(qs[1], ks[0], vs[0])
                x = torch.concat([x1, x2], dim=2)
            else:
                raise NotImplementedError
        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)

        return x
