from typing import Any, Literal

import numpy as np
import torch

from .preprocessor import Preprocessor


class PointSamplingPreprocessor(Preprocessor):
    """Randomly subsamples points from a pointcloud."""

    def __init__(
        self,
        items: set[str],
        num_points: int,
        discarded_prefix: str | None = None,
        seed: int | None = None,
    ):
        """Initializes the point sampling preprocessor.

        Args:
            items: Which pointcloud items should be subsampled (e.g., input_position, output_position, ...). If multiple
                items are present, the subsampling will use identical indices for all items (e.g., to downsample
                output_position and output_pressure with the same subsampling).
            num_points: Number of points to sample.
            seed: Random seed for deterministic sampling for evaluation. Default None (i.e., no seed). If not None,
                requires sample index to be present in batch.
        """
        self.items = items
        self.num_points = num_points
        self.discarded_prefix = discarded_prefix
        self.seed = seed

    def __call__(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Subsamples the pointclouds identified by `self.items` with the same subsampling. The outer list and dicts
        are copied explicitly, the Any objects are not. However, the subsampled tensors are "copied" implicitly as
        sampling is implemented via random index access, which implicitly creates a copy of the underlying values.

        Args:
            samples: List of individual samples retrieved from the dataset.

        Returns:
            Preprocessed copy of `samples`.
        """

        # copy to avoid changing method input
        samples = [dict(sample) for sample in samples]

        # apply preprocessing
        any_item = next(iter(self.items))
        for sample_idx in range(len(samples)):
            # create perm
            if self.seed is not None:
                assert "index" in samples[sample_idx]
                seed = samples[sample_idx]["index"] + self.seed
                generator = torch.Generator().manual_seed(seed)
            else:
                generator = None
            first_item_tensor = samples[sample_idx][any_item]
            assert torch.is_tensor(first_item_tensor)
            if self.discarded_prefix is None:
                perm = torch.randperm(len(first_item_tensor), generator=generator)[: self.num_points]
                discarded_perm = None
            else:
                perm = torch.randperm(len(first_item_tensor), generator=generator)
                if len(first_item_tensor) <= self.num_points:
                    discarded_perm = None
                else:
                    discarded_perm = perm[self.num_points :]
                perm = perm[: self.num_points]
            # subsample
            for item in self.items:
                tensor = samples[sample_idx][item]
                assert torch.is_tensor(tensor)
                samples[sample_idx][item] = tensor[perm]
                if discarded_perm is not None:
                    samples[sample_idx][f"{self.discarded_prefix}_{item}"] = tensor[discarded_perm]

        return samples
