import torch
from torch import nn


class Mlp(nn.Module):
    """MLP as used in transformers nn.Linear(dim, dim * 4) -> GELU -> nn.Linear(dim * 4, dim).

    Args:
        dim: Input dimension of the MLP.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x
