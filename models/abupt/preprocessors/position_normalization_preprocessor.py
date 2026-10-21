from collections.abc import Sequence
from typing import Any

import torch

from .preprocessor import Preprocessor


class PositionNormalizationPreprocessor(Preprocessor):
    """Pre-processes data on a sample-level to normalize positions."""

    def __init__(
        self,
        items: set[str],
        raw_pos_min: Sequence[float],
        raw_pos_max: Sequence[float],
        scale: int | float = 1000,
    ):
        """Initializes the PositionNormalizationPreprocessor.

        Args:
            items: The position items to normalize.
            raw_pos_min: The minimum position in the source domain.
            raw_pos_max: The maximum position in the source domain.
            scale: The maximum value of the position. Defaults to 1000.
        """
        assert len(raw_pos_min) == len(raw_pos_max), "Raw position min and max must have the same length."

        self.items = items
        self.scale = scale
        self.raw_pos_min_tensor = torch.tensor(raw_pos_min).unsqueeze(0)
        self.raw_pos_max_tensor = torch.tensor(raw_pos_max).unsqueeze(0)
        self.raw_size = self.raw_pos_max_tensor - self.raw_pos_min_tensor

    def __call__(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pre-processes data on a batch-level to normalize positions.

        Args:
            samples: Samples of a batch.

        Return:
            Pre-processed samples with normalized positions.
        """
        # copy to avoid changing method input
        samples = [dict(sample) for sample in samples]

        # process
        for sample in samples:
            for item in self.items:
                assert sample[item].ndim == self.raw_pos_min_tensor.ndim
                sample[item] = (sample[item] - self.raw_pos_min_tensor).div_(self.raw_size).mul_(self.scale)

        return samples

    def denormalize(self, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        if value.ndim == self.raw_pos_min_tensor.ndim:
            # sparse tensor -> no additional dimension needed
            # fmt: off
            denormalized_value = (
                (value / self.scale)
                .mul_(self.raw_size.to(value.device))
                .add_(self.raw_pos_min_tensor.to(value.device))
            )
            # fmt: on
        else:
            # dense tensor -> add batch dimension
            denormalized_value = (
                (value / self.scale)
                .mul_(self.raw_size.unsqueeze(0).to(value.device))
                .add_(self.raw_pos_min_tensor.unsqueeze(0).to(value.device))
            )
        return denormalized_value
