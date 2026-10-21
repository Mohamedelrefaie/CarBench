from functools import partial

import einops
import torch
from torch import nn


from modules.attention import AnchorAttention
from modules.blocks import TransformerBlock, PerceiverBlock
from modules.continuous_sincos_embed import ContinuousSincosEmbed
from modules.rope_frequency import RopeFrequency
from modules.supernode_pooling_posonly import SupernodePoolingPosonly


class SurfaceOnlyABUPT(nn.Module):
    def __init__(
        self,
        # problem dimensions
        ndim: int = 3,  # number of coordinates (typicaly 3 for 3D geometries)
        input_dim: int = 3,  # we only use coordinates as inputs
        output_dim_surface: int = 4,  # surface pressure (1D) and wallshearstress (3D)
        # model
        dim: int = 192,  # dimension of a ViT-tiny
        geometry_depth: int = 1,  # 1 transformer block after supernode pooling to encode the geometry
        num_heads: int = 3,  # number of attention heads in a ViT-tiny
        # "p": weight-shared cross-attention block to the geometry branch outputs
        # "s" weight-shared split attention block within surface; for surface-only there is no volume branch though
        blocks: str = "p",
        num_surface_blocks: int = 11,  # 11 modality-specific self-attention blocks
        radius: float = 1.0,  # radius for supernode pooling
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rope = RopeFrequency(dim=dim // num_heads, ndim=input_dim)
        # geometry
        self.encoder = SupernodePoolingPosonly(
            hidden_dim=dim,
            ndim=ndim,
            radius=radius,
            mode="relpos",
        )
        self.geometry_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=dim, num_heads=num_heads)
                for _ in range(geometry_depth)
            ],
        )
        # pos_embed with separate MLP for surface
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.surface_bias = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # weight-shared blocks
        self.blocks = nn.ModuleList()
        for block in blocks:
            if block == "s":
                # self-attention within surface tokens (would be weight shared in surface+volume AB-UPT)
                # block_ctor = partial(TransformerBlock, attn_ctor=SharedweightsSplitattnAttention)
                block_ctor = partial(TransformerBlock, attn_ctor=AnchorAttention)
            elif block == "p":
                # cross-attention from surface tokens to geometry (would be weight shared in surface+volume AB-UPT)
                block_ctor = PerceiverBlock
            else:
                raise NotImplementedError
            self.blocks.append(block_ctor(dim=dim, num_heads=num_heads))
        # surface-specific blocks
        self.surface_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    attn_ctor=AnchorAttention,
                )
                for _ in range(num_surface_blocks)
            ],
        )
        # final linear projections
        self.surface_decoder = nn.Linear(dim, output_dim_surface)

        # init weights
        # (there are only nn.Linear and nn.LayerNorm in AB-UPT, layernorms are initialized correctly by default)
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(init_weights)

    def forward(
        self,
        # geometry
        geometry_position: torch.Tensor,
        geometry_supernode_idx: torch.Tensor,
        geometry_batch_idx: torch.Tensor | None,
        # anchors
        surface_anchor_position: torch.Tensor,
        # queries
        surface_query_position: torch.Tensor | None = None,
        unbatch_mask_surface_anchor_position: None = None,
        unbatch_mask_surface_query_position: None = None,
    ) -> dict[str, torch.Tensor]:
        assert unbatch_mask_surface_anchor_position is None
        del unbatch_mask_surface_anchor_position
        assert unbatch_mask_surface_query_position is None
        del unbatch_mask_surface_query_position

        # outputs/kwargs
        outputs = {}
        geometry_attn_kwargs = {}
        surface_decoder_attn_kwargs = {}
        geometry_perceiver_attn_kwargs = {}
        shared_attn_kwargs = {}

        # create split sizes + optionally concat query positions
        num_surface_anchor_positions = surface_anchor_position.size(1)
        if surface_query_position is None:
            surface_position_all = surface_anchor_position
        elif surface_query_position is not None:
            # surface queries
            surface_decoder_attn_kwargs["num_anchor_tokens"] = surface_anchor_position.size(1)
            surface_position_all = torch.concat([surface_anchor_position, surface_query_position], dim=1)
        else:
            raise NotImplementedError

        # rope frequencies
        assert geometry_batch_idx is None or geometry_batch_idx.unique().numel() == 1, "batch_size > 1 not supported"
        geometry_rope = self.rope(geometry_position[geometry_supernode_idx].unsqueeze(0))
        geometry_attn_kwargs["freqs"] = geometry_rope
        rope_surface_all = self.rope(surface_position_all)
        rope_all = rope_surface_all
        surface_decoder_attn_kwargs["freqs"] = rope_surface_all
        geometry_perceiver_attn_kwargs["q_freqs"] = rope_all
        geometry_perceiver_attn_kwargs["k_freqs"] = geometry_rope
        shared_attn_kwargs["freqs"] = rope_all

        # encode geometry
        x = self.encoder(
            input_pos=geometry_position,
            supernode_idx=geometry_supernode_idx,
            batch_idx=geometry_batch_idx,
        )
        for block in self.geometry_blocks:
            x = block(x, attn_kwargs=geometry_attn_kwargs)
        geometry_encoding = x

        # shared-weights model (all tokens are concatenated into a single sequence for high GPU utilization)
        assert surface_position_all.ndim == 3
        surface_all_pos_embed = self.surface_bias(self.pos_embed(surface_position_all))
        x = surface_all_pos_embed
        for block in self.blocks:
            if isinstance(block, TransformerBlock):
                x = block(x, attn_kwargs=dict(**shared_attn_kwargs))
            elif isinstance(block, PerceiverBlock):
                # cross attention to geometry outputs
                x = block(q=x, kv=geometry_encoding, attn_kwargs=geometry_perceiver_attn_kwargs)
            else:
                raise NotImplementedError

        for block in self.surface_blocks:
            x = block(x, attn_kwargs=dict(**surface_decoder_attn_kwargs))
        x = self.surface_decoder(x)
        # convert to sparse tensor
        if surface_query_position is None:
            outputs["surface_anchor_pressure"] = einops.rearrange(x[:, :, :1], "bs seqlen dim -> (bs seqlen) dim")
            outputs["surface_anchor_wallshearstress"] = einops.rearrange(
                x[:, :, 1:],
                "bs seqlen dim -> (bs seqlen) dim",
            )
        else:
            x_surface_anchor = x[:, :num_surface_anchor_positions]
            x_surface_query = x[:, num_surface_anchor_positions:]
            outputs["surface_anchor_pressure"] = einops.rearrange(
                x_surface_anchor[:, :, :1],
                "bs seqlen dim -> (bs seqlen) dim",
            )
            outputs["surface_anchor_wallshearstress"] = einops.rearrange(
                x_surface_anchor[:, :, 1:],
                "bs seqlen dim -> (bs seqlen) dim",
            )
            outputs["surface_query_pressure"] = einops.rearrange(
                x_surface_query[:, :, :1],
                "bs seqlen dim -> (bs seqlen) dim",
            )
            outputs["surface_query_wallshearstress"] = einops.rearrange(
                x_surface_query[:, :, 1:],
                "bs seqlen dim -> (bs seqlen) dim",
            )

        return outputs


def main():
    torch.manual_seed(0)
    num_geometry_positions = 655
    num_geometry_supernodes = 123
    num_surface_anchors = 234
    num_surface_queries = 301
    data = dict(
        geometry_position=torch.rand(num_geometry_positions, 3) * 1000,
        geometry_supernode_idx=torch.randperm(num_geometry_positions)[:num_geometry_supernodes],
        geometry_batch_idx=None,
        # anchors
        surface_anchor_position=torch.rand(1, num_surface_anchors, 3) * 1000,
        # queries
        surface_query_position=torch.rand(1, num_surface_queries, 3) * 1000,
    )
    model = SurfaceOnlyABUPT()
    outputs = model(**data)
    for key, value in outputs.items():
        print(f"{key}: {value.shape}")


if __name__ == "__main__":
    main()
