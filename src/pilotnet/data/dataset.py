"""Dataset support for image-and-steering driving logs."""

from __future__ import annotations

import csv
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as transforms


class DrivingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load a CSV with ``image_path`` and ``steering`` columns.

    Image paths are resolved relative to the CSV file. Images are resized to
    PilotNet's ``200 x 66`` input. Training augmentation mirrors an image and
    negates its steering target, preserving the driving geometry.
    """

    def __init__(self, csv_path: str | Path, augment: bool = False) -> None:
        self.csv_path = Path(csv_path)
        self.augment = augment
        with self.csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if not rows or {"image_path", "steering"} - set(rows[0]):
            raise ValueError("CSV must contain at least one row with image_path and steering columns.")
        self.samples = [(self.csv_path.parent / row["image_path"], float(row["steering"])) for row in rows]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, steering = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        image = transforms.resize(
            image,
            size=(66, 200),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        if self.augment and torch.rand(()) < 0.5:
            image = transforms.hflip(image)
            steering = -steering
        return transforms.to_tensor(image), torch.tensor(steering, dtype=torch.float32)
