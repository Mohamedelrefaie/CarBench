from collections.abc import Sequence
from typing import Any

import torch


from .preprocessor import Preprocessor

import torch


def to_logscale(x: torch.Tensor) -> torch.Tensor:
    """Turns a tensor into log scale. Log is the natural logarithm of x + 1 .

    Args:
        x: Tensor to be transformed

    Returns:
        Tensor in log scale
    """
    return torch.sign(x) * torch.log1p(x.abs())


def from_logscale(x: torch.Tensor) -> torch.Tensor:
    """Turns a tensor from log scale into orginal scale.
        x = from_logscale(to_logscale(x))

    Args:
        x: Tensor to be de-transformed from log scale. Expected to be in natural logarithm + 1.

    Returns:
        Tensor in orginal scale
    """
    return torch.sign(x) * (x.abs().exp() - 1)


class MomentNormalizationPreprocessor(Preprocessor):
    """Normalizes a value with its mean and standard deviation (i.e., its moments)."""

    def __init__(
        self,
        item: str | None = None,
        items: set[str] | None = None,
        mean: Sequence[float] | None = None,
        std: Sequence[float] | None = None,
        logmean: Sequence[float] | None = None,
        logstd: Sequence[float] | None = None,
        logscale: bool = False,
    ):
        """Initializes the MomentNormalizationPreprocessor

        Args:
            item: The item to normalize.
            mean: The mean of the value. Mandatory if logscale=False.
            std: The standard deviation of the value. Mandatory if logscale=False.
            logmean: The mean of the value in logscale. Mandatory if logscale=True.
            logstd: The standard deviation of the value in logscale. Mandatory if logscale=True.
            logscale: Whether to convert the value to logscale before normalization.
        """
        if logscale:
            assert len(logmean) == len(logstd), "Mean and standard deviation must have the same length."
        else:
            assert len(mean) == len(std), "Mean and standard deviation must have the same length."
        if item is None:
            assert items is not None
            self.items = items
        else:
            self.items = {item}
        self.mean_tensor = None if mean is None else torch.tensor(mean).unsqueeze(0)
        self.std_tensor = None if std is None else torch.tensor(std).unsqueeze(0)
        self.logmean_tensor = None if logmean is None else torch.tensor(logmean).unsqueeze(0)
        self.logstd_tensor = None if logstd is None else torch.tensor(logstd).unsqueeze(0)
        self.logscale = logscale

    def __call__(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Pre-processes data on a sample-level to normalize a value to approximately mean=0 std=1.

        Args:
            samples: Samples of a batch.

        Return:
            Pre-processed samples with normalized values.
        """
        # copy to avoid changing method input
        samples = [dict(sample) for sample in samples]

        # process
        for item in self.items:
            for sample in samples:
                if self.logscale:
                    assert self.logmean_tensor is not None and self.logstd_tensor is not None
                    assert sample[item].ndim == self.logmean_tensor.ndim
                    sample[item] = to_logscale(sample[item]).sub_(self.logmean_tensor).div_(self.logstd_tensor)
                else:
                    assert self.mean_tensor is not None and self.std_tensor is not None
                    assert sample[item].ndim == self.mean_tensor.ndim, (
                        f"item={item} item.ndim={sample[item].ndim} mean.ndim={self.mean_tensor.ndim}"
                    )
                    sample[item] = (sample[item] - self.mean_tensor).div_(self.std_tensor)

        return samples

    def denormalize(self, value: torch.Tensor) -> tuple[str, torch.Tensor]:
        if self.logscale:
            assert self.logmean_tensor is not None and self.logstd_tensor is not None
            if value.ndim == self.logmean_tensor.ndim:
                # sparse tensor -> no additional dimension needed
                denormalized_value = value * self.logstd_tensor.to(value.device) + self.logmean_tensor.to(value.device)
            else:
                # dense tensor -> add batch dimension
                logstd_tensor = self.logstd_tensor.unsqueeze(0).to(value.device)
                logmean_tensor = self.logmean_tensor.unsqueeze(0).to(value.device)
                denormalized_value = value * logstd_tensor + logmean_tensor
            denormalized_value = from_logscale(denormalized_value)
        else:
            assert self.mean_tensor is not None and self.std_tensor is not None
            if value.ndim == self.mean_tensor.ndim:
                # sparse tensor -> no additional dimension needed
                denormalized_value = value * self.std_tensor.to(value.device) + self.mean_tensor.to(value.device)
            else:
                # dense tensor -> add batch dimension
                std_tensor = self.std_tensor.unsqueeze(0).to(value.device)
                mean_tensor = self.mean_tensor.unsqueeze(0).to(value.device)
                denormalized_value = value * std_tensor + mean_tensor

        return denormalized_value
