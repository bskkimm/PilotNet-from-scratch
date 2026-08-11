"""Tests for the shared PilotNet image preprocessing transform."""

import pytest
import torch
from PIL import Image

from pilotnet.data.preprocessing import PreprocessConfig, preprocess_image


def test_preprocess_crops_converts_to_yuv_and_resizes() -> None:
    image = Image.new("RGB", (20, 10), color=(255, 0, 0))

    result = preprocess_image(
        image,
        PreprocessConfig(crop_top_fraction=0.2, crop_bottom_fraction=0.2),
    )

    assert result.shape == (3, 66, 200)
    assert result.dtype == torch.float32
    assert result.min() >= 0
    assert result.max() <= 1


def test_preprocess_outputs_ycbcr_channels_in_yuv_order() -> None:
    result = preprocess_image(Image.new("RGB", (1, 1), color=(255, 0, 0)), PreprocessConfig())

    torch.testing.assert_close(
        result[:, 0, 0],
        torch.tensor([76, 84, 255], dtype=torch.float32) / 255,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"crop_top_fraction": -0.1},
        {"crop_bottom_fraction": 1.0},
        {"crop_top_fraction": 0.5, "crop_bottom_fraction": 0.5},
    ],
)
def test_preprocess_rejects_invalid_crop_fractions(kwargs) -> None:
    with pytest.raises(ValueError):
        PreprocessConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"height": 0}, {"height": -1}, {"width": 0}, {"width": -1}]
)
def test_preprocess_rejects_nonpositive_output_dimensions(kwargs) -> None:
    with pytest.raises(ValueError):
        PreprocessConfig(**kwargs)
