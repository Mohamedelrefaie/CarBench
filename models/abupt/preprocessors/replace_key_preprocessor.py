from typing import Any

import torch

from .preprocessor import Preprocessor


class ReplaceKeyPreprocessor(Preprocessor):
    """Utility processor that replaces the key with multiple other keys."""

    def __init__(self, source_key: str, target_keys: set[str]):
        """Initializes the ReplaceKeyPreprocessor

        Args:
            source_key: Key to be replaced.
            target_keys: List of keys where source_key should be replaced in.
        """
        self.source_key = source_key
        self.target_keys = target_keys

    def __call__(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Replaces a key in the batch with one or multiple other keys.
        Creates a new dictionary whose keys are duplicated but uses references to the values of the old dict.
        This avoids copying the data and at the same time does not modify this function's input.
        Args:
            samples: The samples to replace keys of.
        Returns:
            The samples with the keys replaced.
        """
        new_samples = []
        for sample in samples:
            new_sample = dict(sample)
            source_item = new_sample.pop(self.source_key)
            for target_key in self.target_keys:
                new_sample[target_key] = source_item
            new_samples.append(new_sample)
        return new_samples
