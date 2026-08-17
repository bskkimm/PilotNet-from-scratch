"""Causal RGB sequence loading for temporal steering models."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transforms


@dataclass(frozen=True)
class TemporalPreprocessConfig:
    """RGB preprocessing compatible with ImageNet-pretrained encoders."""

    height: int = 160
    width: int = 512
    crop_top_fraction: float = 0.0
    crop_bottom_fraction: float = 0.0
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError("Temporal preprocess height and width must be positive.")
        if not 0 <= self.crop_top_fraction < 1 or not 0 <= self.crop_bottom_fraction < 1:
            raise ValueError("Crop fractions must be in [0, 1).")
        if self.crop_top_fraction + self.crop_bottom_fraction >= 1:
            raise ValueError("Crop fractions must leave at least one image row.")


def preprocess_temporal_image(image: Image.Image, config: TemporalPreprocessConfig) -> torch.Tensor:
    """Crop, resize, and ImageNet-normalize an RGB image."""
    top = round(image.height * config.crop_top_fraction)
    bottom = image.height - round(image.height * config.crop_bottom_fraction)
    image = image.convert("RGB").crop((0, top, image.width, bottom))
    image = transforms.resize(
        image,
        (config.height, config.width),
        InterpolationMode.BILINEAR,
        antialias=True,
    )
    return transforms.normalize(transforms.to_tensor(image), config.mean, config.std)


@dataclass(frozen=True)
class _Row:
    image_path: Path
    steering: float
    speed: float
    timestamp: int
    scene_name: str


class TemporalDrivingDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Return fixed-length causal image windows that never cross a scene boundary."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        sequence_length: int = 5,
        max_frame_gap_us: int = 200_000,
        speed_scale: float = 30.0,
        augment: bool = False,
        preprocess: TemporalPreprocessConfig = TemporalPreprocessConfig(),
    ) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be positive.")
        if max_frame_gap_us < 1:
            raise ValueError("max_frame_gap_us must be positive.")
        if speed_scale <= 0:
            raise ValueError("speed_scale must be positive.")
        self.csv_path = Path(csv_path)
        self.sequence_length = sequence_length
        self.speed_scale = speed_scale
        self.augment = augment
        self.preprocess = preprocess
        with self.csv_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            required = {"image_path", "steering", "speed", "timestamp", "scene_name"}
            if required - set(reader.fieldnames or []):
                raise ValueError(f"CSV must contain {sorted(required)} columns.")
            scene_rows: dict[str, list[_Row]] = {}
            for number, row in enumerate(reader, start=2):
                try:
                    steering, speed = float(row["steering"]), float(row["speed"])
                    timestamp = int(row["timestamp"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"Invalid temporal sample in row {number}.") from error
                if not math.isfinite(steering) or not math.isfinite(speed):
                    raise ValueError(f"Invalid temporal sample in row {number}.")
                scene_rows.setdefault(row["scene_name"], []).append(
                    _Row(
                        self.csv_path.parent / row["image_path"],
                        steering,
                        speed,
                        timestamp,
                        row["scene_name"],
                    )
                )

        self.samples: list[tuple[_Row, ...]] = []
        for rows in scene_rows.values():
            rows.sort(key=lambda row: row.timestamp)
            for end in range(sequence_length - 1, len(rows)):
                window = tuple(rows[end - sequence_length + 1 : end + 1])
                if all(
                    later.timestamp - earlier.timestamp <= max_frame_gap_us
                    for earlier, later in zip(window, window[1:])
                ):
                    self.samples.append(window)
        if not self.samples:
            raise ValueError("CSV has no complete, contiguous temporal sequences.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        window = self.samples[index]
        flip = self.augment and torch.rand(()) < 0.5
        images = []
        for row in window:
            with Image.open(row.image_path) as image_file:
                image = image_file.convert("RGB")
            if flip:
                image = transforms.hflip(image)
            images.append(preprocess_temporal_image(image, self.preprocess))
        target = -window[-1].steering if flip else window[-1].steering
        speeds = torch.tensor([row.speed / self.speed_scale for row in window], dtype=torch.float32)
        return torch.stack(images), speeds, torch.tensor(target, dtype=torch.float32)
