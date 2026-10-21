import einops
import torch
import torch.nn.functional as F

from modules.attention import DotProductAttention
from modules.rope import rope

class SharedweightsSplitattnAttention(DotProductAttention):
    def forward(
        self,
        x: torch.Tensor,
        split_size: list[int],
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        """Attention between:
        - q=surface_anchors -> kv=surface_anchors
        - q=volume_anchors -> kv=volume_anchors
        - q=surface_queries -> kv=surface_anchors
        - q=volume_queries -> kv=volume_anchors

        Args:
            x: Tensor containing all anchors/queries (batch size, sequence length, dim).
            split_size: How to split x into:
                len(split_size) == 2: (surface_anchors, volume_anchors)
                len(split_size) == 4: (surface_anchors, surface_queries, volume_anchors, volume_queries)
            freqs: Frequencies for Rotary Positional Embedding (RoPE) of queries/keys. None if use_rope=False.

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
        qs = q.split(split_size, dim=2)
        ks = k.split(split_size, dim=2)
        vs = v.split(split_size, dim=2)
        if isinstance(split_size, list) and len(split_size) == 4:
            # surface + volume queries
            q1 = torch.concat([qs[0], qs[1]], dim=2)
            k1 = ks[0]
            v1 = vs[0]
            x1 = F.scaled_dot_product_attention(q1, k1, v1)
            q2 = torch.concat([qs[2], qs[3]], dim=2)
            k2 = ks[2]
            v2 = vs[2]
            x2 = F.scaled_dot_product_attention(q2, k2, v2)
            x = torch.concat([x1, x2], dim=2)
        else:
            # no queries -> self-attn within splits
            assert len(split_size) == 2
            if isinstance(split_size, list) and all(split_size[0] == ss for ss in split_size[1:]):
                # optimized for equal sized splits
                num_splits = len(qs)
                q = torch.concat(qs)
                k = torch.concat(ks)
                v = torch.concat(vs)
                x = F.scaled_dot_product_attention(q, k, v)
                x = torch.concat(x.chunk(chunks=num_splits), dim=2)
            else:
                # generic case
                x = torch.concat([F.scaled_dot_product_attention(qs[i], ks[i], vs[i]) for i in range(len(qs))], dim=2)

        x = einops.rearrange(x, "bs num_heads seqlen head_dim -> bs seqlen (num_heads head_dim)")
        x = self.proj(x)

        return x
