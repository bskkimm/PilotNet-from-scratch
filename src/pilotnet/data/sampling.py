"""Sampling helpers for imbalanced steering targets."""

from collections.abc import Sequence

import torch
from torch.utils.data import WeightedRandomSampler


def build_balanced_sampler(
    targets: Sequence[float], bins: int, generator: torch.Generator | None = None
) -> WeightedRandomSampler:
    """Sample steering bins inversely to their observed frequency."""
    if bins < 2:
        raise ValueError("bins must be at least 2")

    values = torch.tensor(targets, dtype=torch.float64)
    if values.min() == values.max():
        indices = torch.zeros(len(values), dtype=torch.long)
        counts = torch.tensor([len(values)])
    else:
        boundaries = torch.linspace(values.min(), values.max(), bins + 1, dtype=values.dtype)[1:-1]
        indices = torch.bucketize(values, boundaries)
        counts = torch.bincount(indices, minlength=bins)
    weights = counts[indices].to(torch.double).reciprocal()
    return WeightedRandomSampler(weights, len(targets), replacement=True, generator=generator)
