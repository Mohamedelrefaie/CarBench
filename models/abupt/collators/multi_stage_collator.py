from collections.abc import Callable
from typing import Any

import torch
from torch.utils.data import Dataset


class MultiStageCollator:
    """Collator that processes the list of samples into a batch in multiple stages:
    - preprocessors: Pre-processing on a per-sample level.
    - collators: Conversion from a list of samples into a batch (dict of tensors).
    - postprocessors: Post-procing on a batch-level.

    Args:
        dataset: The dataset from which the data originates from. Used to extract e.g., data normalization stats.
        preprocessors: A list of callables that will be applied sequentially to pre-process on a per-sample level
            (e.g., subsample a pointcloud).
        collators: A list of callables that will be applied sequentially to convert the list of individual samples into
            a batched format.
        postprocessors: A list of callables that will be applied sequentially to post-process on a per-batch level.
            dataset: Dataset which loaded the batch.
    """

    def __init__(
        self,
        dataset: Dataset,
        preprocessors: list[Callable[[list[dict[str, Any]]], list[dict[str, Any]]]] | None = None,
        collators: list[Callable[[list[dict[str, Any]]], dict[str, torch.Tensor]]] | None = None,
        postprocessors: list[Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]]] | None = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.preprocessors = preprocessors
        self.collators = collators
        self.postprocessors = postprocessors

    def get_preprocessor(self, predicate: Callable[[Any], bool]) -> Any:
        """
        Retrieves a preprocessor by a predicate function.
        Examples:
        - Search by type (assumes the preprocessor type only occurs once in the list of preprocessors)
          `collator.get_preprocessor(lambda p: isinstance(p, MyPreprocessorType))`
        - Search by type and member
          `collator.get_preprocessor(lambda p: isinstance(p, PointSamplingPreprocessor) and "input_pos" in p.items)`

        Args:
            predicate: A function that is called for each processor and selects if this is the right one.

        Returns:
            Any: The matching preprocessor.

        Raises:
            ValueError: If no matching preprocessor are found, multiple matching preprocessors are found or if there
                are no preprocessors.
        """
        if len(self.preprocessors) == 0:
            raise ValueError("No preprocessor matches predicate.")
        found_processors = []
        for preprocessor in self.preprocessors:
            if predicate(preprocessor):
                found_processors.append(preprocessor)
        if len(found_processors) == 0:
            raise ValueError("No preprocessor matches predicate.")
        if len(found_processors) > 1:
            raise ValueError(f"Multiple preprocessor matches predicate ({found_processors}).")
        return found_processors[0]

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Applies a multi-stage collation pipeline to the loaded samples.

        Args:
            samples (list[dict[str, Any]]): List of individual samples retrieved from the dataset.

        Returns:
            Collated batch.
        """
        # pre-process on a sample level
        for pre_collator in self.preprocessors:
            samples = pre_collator(samples)

        # create batch out of the samples
        batch = {}
        for batch_collator in self.collators:
            sub_batch = batch_collator(samples)
            # make sure that there is no overlap between collators
            for key, value in sub_batch.items():
                assert key not in batch
                batch[key] = value

        # post-process the batch
        for post_collator in self.postprocessors:
            batch = post_collator(batch)
        return batch
