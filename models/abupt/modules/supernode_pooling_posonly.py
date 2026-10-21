import einops
import torch
import torch_geometric
import torch_scatter
from torch import nn

from modules.continuous_sincos_embed import ContinuousSincosEmbed


class SupernodePoolingPosonly(nn.Module):
    """Supernode pooling layer.

    The permutation of the supernodes is preserved through the message passing (contrary to the (GP-)UPT code).
    Additionally, radius is used instead of radius_graph, which is more efficient.

    Args:
        radius: Radius around each supernode. From points within this radius, messages are passed to the supernode.
        k: Numer of neighbors for each supernode. From the k-NN points, messages are passed to the supernode.
        hidden_dim: Hidden dimension for positional embeddings, messages and the resulting output vector.
        ndim: Number of positional dimension (e.g., ndim=2 for a 2D position, ndim=3 for a 3D position)
        max_degree: Maximum degree of the radius graph. Defaults to 32.
        mode: Are positions embedded in absolute space ("abspos") or relative space ("relpos").
            "readd_supernode_pos" will always use the absolute position.
        readd_supernode_pos: If true, the absolute positional encoding of the supernode is concated to the
          supernode vector after message passing and linearly projected back to hidden_dim. Defaults to True.
    """

    def __init__(
        self,
        hidden_dim: int,
        ndim: int,
        radius: float | None = None,
        k: int | None = None,
        max_degree: int = 32,
        mode: str = "relpos",
    ):
        super().__init__()
        assert (radius is not None) ^ (k is not None)
        self.radius = radius
        self.k = k
        self.max_degree = max_degree
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.mode = mode

        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)
        if mode == "abspos":
            message_input_dim = hidden_dim * 2
            self.rel_pos_embed = None
        elif mode == "relpos":
            message_input_dim = hidden_dim
            self.rel_pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim + 1, assert_positive=False)
        else:
            raise NotImplementedError
        self.message = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.output_dim = hidden_dim

    def compute_src_and_dst_indices(
        self,
        input_pos: torch.Tensor,
        supernode_idx: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the source and destination indices for the message passing to the supernodes.

        Args:
            input_pos: Sparse tensor with shape (batch_size * numner of points, 3), representing the input geometries.
            supernode_idx: Indexes of the supernodes in the sparse tensor input_pos.
            batch_idx: 1D tensor, containing the batch index of each entry in input_pos. Default None.

        Returns:
            Tensor with src and destination indexes for the message passing into the supernodes.
        """
        input_pos = input_pos[..., : self.ndim]

        # radius graph
        if batch_idx is None:
            batch_y = None
        else:
            batch_y = batch_idx[supernode_idx]
        if self.radius is not None:
            assert self.k is None
            edges = torch_geometric.nn.pool.radius(
                x=input_pos,
                y=input_pos[supernode_idx],
                r=self.radius,
                max_num_neighbors=self.max_degree,
                batch_x=batch_idx,
                batch_y=batch_y,
            )
        elif self.k is not None:
            edges = torch_geometric.nn.pool.knn(
                x=input_pos,
                y=input_pos[supernode_idx],
                k=self.k,
                batch_x=batch_idx,
                batch_y=batch_y,
            )
        else:
            raise NotImplementedError
        # remap dst indices
        dst_idx, src_idx = edges.unbind()
        dst_idx = supernode_idx[dst_idx]

        return src_idx, dst_idx

    def create_messages(
        self,
        input_pos: torch.Tensor,
        src_idx: torch.Tensor,
        dst_idx: torch.Tensor,
        supernode_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create messages for the message passing to the supernodes, based on different positional encoding
        representations.

        Args:
            input_pos: Tensor of shape (batch_size * number_of_points_per_sample, {2,3}), representing the point cloud
                representation of the input geometry.
            src_idx: Index of the source nodes from input_pos.
            dst_idx: Source index of the destination nodes from input_pos tensor. These indexes should be the matching
                supernode indexes.
            supernode_idx: Indexes of the node in input_pos that are considered supernodes.

        Raises:
            NotImplementedError: Raised if the mode is not implemented. Either "abspos" or "relpos" are allowed.

        Returns:
            Tensor with messages for the message passing into the super nodes and the embedding coordinates of the
                supernodes.
        """

        # create message
        if self.mode == "abspos":
            x = self.pos_embed(input_pos)
            supernode_pos_embed = x[supernode_idx]
            x = torch.concat([x[src_idx], x[dst_idx]], dim=1)
        elif self.mode == "relpos":
            src_pos = input_pos[src_idx]
            dst_pos = input_pos[dst_idx]
            dist = dst_pos - src_pos
            mag = dist.norm(dim=1).unsqueeze(-1)
            x = self.rel_pos_embed(torch.concat([dist, mag], dim=1))
            supernode_pos_embed = self.pos_embed(input_pos[supernode_idx])
        else:
            raise NotImplementedError
        x = self.message(x)
        return x, supernode_pos_embed

    @staticmethod
    def accumulate_messages(
        x: torch.Tensor,
        dst_idx: torch.Tensor,
        supernode_idx: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Method the accumulate the messages of neighbouring points into the supernodes.

        Args:
            x: Tensor containing the message representation of each neighbour representation.
            dst_idx: Index of the destination (i.e., supernode) where each message should go to.
            supernode_idx: Indexes of the supernode in the input point cloud.
            batch_idx: Batch index of the points in the sparse tensor.

        Returns:
            Tensor with the aggregated messages for each supernode.
        """
        # accumulate messages
        # indptr is a tensor of indices betweeen which to aggregate
        # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
        dst_indices, counts = dst_idx.unique_consecutive(return_counts=True)
        assert torch.all(supernode_idx == dst_indices)
        # first index has to be 0
        # NOTE: padding for target indices that don't occour is not needed as self-loop is always present
        padded_counts = torch.zeros(len(counts) + 1, device=counts.device, dtype=counts.dtype)
        padded_counts[1:] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = torch_scatter.segment_csr(src=x, indptr=indptr, reduce="mean")

        # sanity check: dst_indices has len of batch_size * num_supernodes and has to be divisible by batch_size
        # if num_supernodes is not set in dataset this assertion fails
        if batch_idx is None:
            batch_size = 1
        else:
            batch_size = batch_idx.max() + 1
            assert dst_indices.numel() % batch_size == 0

        return x, batch_size

    def forward(
        self,
        input_pos: torch.Tensor,
        supernode_idx: torch.Tensor,
        batch_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of the supernode pooling layer.

        Args:
            input_pos: Sparse tensor with shape (batch_size * number_of_points_per_sample, 3), representing the point
                cloud representation of the input geometry.
            supernode_idx: indexes of the supernodes in the sparse tensor input_pos.
            batch_idx: 1D tensor, containing the batch index of each entry in input_pos. Default None.

        Returns:
            Tensor with the aggregated messages for each supernode.
        """
        assert input_pos.ndim == 2, f"input_pos has to be 2D, but has shape {input_pos.shape}"
        assert supernode_idx.ndim == 1, f"supernode_idx has to be 1D, but has shape {supernode_idx.shape}"

        src_idx, dst_idx = self.compute_src_and_dst_indices(
            batch_idx=batch_idx,
            input_pos=input_pos,
            supernode_idx=supernode_idx,
        )

        x, supernode_pos_embed = self.create_messages(
            input_pos=input_pos,
            src_idx=src_idx,
            dst_idx=dst_idx,
            supernode_idx=supernode_idx,
        )

        x, batch_size = self.accumulate_messages(x, dst_idx, supernode_idx, batch_idx)

        # convert to dense tensor (dim last)
        x = einops.rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
        )

        # concatenate supernode pos embedding
        supernode_pos_embed = einops.rearrange(
            supernode_pos_embed,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=len(x),
        )
        x = torch.concat([x, supernode_pos_embed], dim=-1)
        x = self.proj(x)

        return x
