from typing import Any

import torch

from .preprocessor import Preprocessor


class SupernodeSamplingPreprocessor(Preprocessor):
    """Randomly samples supernodes from a pointcloud."""

    def __init__(
        self,
        item: str,
        num_supernodes: int,
        supernode_idx_key: str = "supernode_idx",
        items_at_supernodes: set[str] | None = None,
        seed: int | None = None,
    ):
        """Initializes the supernode sampling preprocessor.

        Args:
            item: Which pointcloud item is used to sample supernodes.
            num_supernodes: How many supernodes to sample.
            items_at_supernodes: Selects items at the supernodes (e.g., pressure at supernodes). Defaults to None.
            seed: Random seed for deterministic sampling for evaluation. Default None (i.e., no seed). If not None,
                requires sample index to be present in batch.
        """

        self.item = item
        self.num_supernodes = num_supernodes
        self.supernode_idx_key = supernode_idx_key
        self.items_at_supernodes = items_at_supernodes
        self.seed = seed

    def __call__(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Randomly samples supernodes from the pointcloud identified by `self.item`. The outer list and dicts are
        copied explicitly, the Any objects are not.

        Args:
            samples: List of individual samples retrieved from the dataset.

        Returns:
            Preprocessed `samples`.
        """

        # copy to avoid changing method input
        samples = [dict(sample) for sample in samples]

        # apply preprocessing
        offset = 0
        for sample_idx in range(len(samples)):
            # sample supernodes
            cur_num_points = len(samples[sample_idx][self.item])
            if self.seed is not None:
                assert "index" in samples[sample_idx]
                seed = samples[sample_idx]["index"] + self.seed
                generator = torch.Generator().manual_seed(seed)
            else:
                generator = None
            perm = torch.randperm(cur_num_points, generator=generator)[: self.num_supernodes]

            # select items at supernode positions
            for item_at_supernodes in self.items_at_supernodes or []:
                item = samples[sample_idx][item_at_supernodes]
                assert torch.is_tensor(item)
                data_at_supernode = item[perm]
                samples[sample_idx][f"supernode_{item_at_supernodes}"] = data_at_supernode

            # subsample
            assert self.supernode_idx_key not in samples[sample_idx]
            samples[sample_idx][self.supernode_idx_key] = perm + offset
            offset += cur_num_points

        return samples
