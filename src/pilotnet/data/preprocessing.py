"""Shared image preprocessing for PilotNet inputs."""

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transforms


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for normalized vertical cropping and PilotNet resizing."""

    crop_top_fraction: float = 0.0
    crop_bottom_fraction: float = 0.0
    height: int = 66
    width: int = 200

    def __post_init__(self) -> None:
        if not 0 <= self.crop_top_fraction < 1 or not 0 <= self.crop_bottom_fraction < 1:
            raise ValueError("Crop fractions must be in [0, 1).")
        if self.crop_top_fraction + self.crop_bottom_fraction >= 1:
            raise ValueError("Crop fractions must leave at least one image row.")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Preprocess height and width must be positive.")


def preprocess_image(image: Image.Image, config: PreprocessConfig) -> torch.Tensor:
    """Crop, convert to YUV, resize, and scale an image for PilotNet."""
    top = round(image.height * config.crop_top_fraction)
    bottom = image.height - round(image.height * config.crop_bottom_fraction)
    cropped = image.convert("RGB").crop((0, top, image.width, bottom)).convert("YCbCr")
    resized = transforms.resize(
        cropped,
        (config.height, config.width),
        InterpolationMode.BILINEAR,
        antialias=True,
    )
    return transforms.to_tensor(resized)
