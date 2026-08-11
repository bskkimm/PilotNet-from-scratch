"""Tests for reproducible training metadata."""

import torch

from pilotnet.data import PreprocessConfig
from pilotnet.utils.reproducibility import build_run_manifest, seed_everything


def test_seed_everything_repeats_torch_values() -> None:
    seed_everything(7, deterministic=True)
    first = torch.rand(3)

    seed_everything(7, deterministic=True)

    assert torch.equal(first, torch.rand(3))


def test_build_run_manifest_records_preprocessing_sampling_and_seed() -> None:
    manifest = build_run_manifest(
        preprocessing=PreprocessConfig(crop_top_fraction=0.1),
        sampling={"balance_bins": 5},
        reproducibility={"seed": 7, "deterministic": True},
    )

    assert manifest["preprocessing"] == {
        "crop_top_fraction": 0.1,
        "crop_bottom_fraction": 0.0,
        "height": 66,
        "width": 200,
    }
    assert manifest["sampling"] == {"balance_bins": 5}
    assert manifest["reproducibility"] == {"seed": 7, "deterministic": True}
