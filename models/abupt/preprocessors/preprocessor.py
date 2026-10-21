from abc import abstractmethod
from typing import Any

import torch


class Preprocessor:
    @abstractmethod
    def __call__(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pre-processes data on a sample-level.

        Args:
            samples: Samples of a batch.

        Return:
            Pre-processed samples.
        """
