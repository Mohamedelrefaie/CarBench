import einops
import torch
from torch import nn


class ContinuousSincosEmbed(nn.Module):
    """Embedding layer for continuous coordinates using sine and cosine functions as used in transformers.
    This implementation is able to deal with arbitrary coordinate dimensions (e.g., 2D and 3D coordinate systems).

    Args:
        dim: Dimensionality of the embedded input coordinates.
        ndim: Number of dimensions of the input domain.
        max_wavelength: Max length. Defaults to 10000.
        assert_positive: If true, assert if all input coordiantes are positive. Defaults to True.
    """

    def __init__(
        self,
        dim: int,
        ndim: int,
        max_wavelength: int = 10000,
        assert_positive: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        # if dim is not cleanly divisible -> cut away trailing dimensions
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        self.assert_positive = assert_positive
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        arange = torch.arange(0, effective_dim_per_wave, 2, dtype=torch.float32)
        self.register_buffer(
            "omega",
            1.0 / max_wavelength**(arange / effective_dim_per_wave),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward method of the ContinuousSincosEmbed layer.

        Args:
            coords: Tensor of coordinates. The shape of the tensor should be
                (batch size, number of points, coordinate dimension) or (number of points, coordinate dimension).

        Returns:
            Tensor with embedded coordinates.
        """
        if self.assert_positive:
            # check if coords are positive
            assert torch.all(coords >= 0)
        # fp32 to avoid numerical imprecision
        coords = coords.float()
        with torch.autocast(device_type=str(coords.device).split(":")[0], enabled=False):
            coordinate_ndim = coords.shape[-1]
            assert self.ndim == coordinate_ndim
            out = coords.unsqueeze(-1) @ self.omega.unsqueeze(0)
            emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
            if coords.ndim == 3:
                emb = einops.rearrange(emb, "bs num_points ndim dim -> bs num_points (ndim dim)")
            elif coords.ndim == 2:
                emb = einops.rearrange(emb, "num_points ndim dim -> num_points (ndim dim)")
            else:
                raise NotImplementedError
        if self.padding > 0:
            padding = torch.zeros(*emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype)
            emb = torch.concat([emb, padding], dim=-1)
        return emb
