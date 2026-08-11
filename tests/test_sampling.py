"""Tests for steering-target sampling."""

import pytest

from pilotnet.data.sampling import build_balanced_sampler


def test_balanced_sampler_weights_rare_bins_more_heavily() -> None:
    sampler = build_balanced_sampler([0.0, 0.0, 0.0, 1.0], bins=2)

    assert sampler.weights[-1] > sampler.weights[0]


def test_balanced_sampler_uses_uniform_weights_for_identical_targets() -> None:
    sampler = build_balanced_sampler([0.5, 0.5, 0.5], bins=2)

    assert sampler.weights.tolist() == [1 / 3, 1 / 3, 1 / 3]


def test_balanced_sampler_requires_at_least_two_bins() -> None:
    with pytest.raises(ValueError, match="bins must be at least 2"):
        build_balanced_sampler([0.0, 1.0], bins=1)
