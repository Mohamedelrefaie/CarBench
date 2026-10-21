from collections import defaultdict
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence


class FieldDecoderCollator:
    """Collates a field to be used in a UPT-style decoder. It requires:
    - Positions as dense tensor (used as query for the Perceiver decoder)
    - Targets as sparse tensor (e.g., used for calculating a loss)
    - Unbatch mask to convert the dense output of the Perceiver decoder into a sparse tensor to compare it to the
        targets (i.e., calculate a loss)

    """

    def __init__(self, position_item: str, target_items: list[str], optional: bool = False):
        """Initializes the FieldDecoderCollator.

        Args:
            position_item: Identifier for the position.
            target_items: Identifiers for the position, can use multiple target_items if multiple values are predicted with
            the same decoder (e.g., predict surface pressure and surface wall shear stress).
        """

        self.position_item = position_item
        self.target_items = target_items
        self.optional = optional

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collates position into a dense tensor (potentially adding padding), collates the targets into a sparse
        tensor and creates an unbatch_mask to remove the padding (None if no padding was necessary).

        Args:
            samples: List of individual samples retrieved from the dataset.

        Returns:
             Batched position item, target items and unbatch mask.
        """

        assert len(samples) > 0, "No samples provided."
        positions = []
        lengths = []
        targets = defaultdict(list)
        for i in range(len(samples)):
            # extract position
            if self.position_item not in samples[i]:
                assert self.optional
                return {}
            position = samples[i][self.position_item]
            positions.append(position)
            lengths.append(len(position))
            # extract targets
            for target_item in self.target_items:
                target = samples[i][target_item]
                assert len(position) == len(target)
                targets[target_item].append(target)

        # positions are a dense tensor -> pad to the longest item
        collated_pos = pad_sequence(positions, batch_first=True)
        # targets are sparse tensors -> concat
        collated_targets = {key: torch.concat(value) for key, value in targets.items()}

        return {
            self.position_item: collated_pos,
            **collated_targets,
        }
