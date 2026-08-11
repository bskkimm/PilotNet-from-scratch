"""Dataset support for image-and-steering driving logs."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as transforms

from .preprocessing import PreprocessConfig, preprocess_image


class DrivingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Load a CSV with ``image_path`` and ``steering`` columns.

    Image paths are resolved relative to the CSV file. Available left and
    right recovery-camera paths expand each row when a correction is supplied.
    Training augmentation mirrors an image and negates its steering target.
    """

    def __init__(
        self,
        csv_path: str | Path,
        *,
        augment: bool = False,
        preprocess: PreprocessConfig = PreprocessConfig(),
        side_camera_correction: float | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.augment = augment
        self.preprocess = preprocess
        with self.csv_path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
        if not rows or {"image_path", "steering"} - set(reader.fieldnames or []):
            raise ValueError("CSV must contain at least one row with image_path and steering columns.")

        self.samples: list[tuple[Path, float]] = []
        for row_number, row in enumerate(rows, start=2):
            try:
                steering = float(row["steering"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid steering value in row {row_number}.") from error
            if not math.isfinite(steering):
                raise ValueError(f"Invalid steering value in row {row_number}.")
            self.samples.append((self.csv_path.parent / row["image_path"], steering))
            for column, direction in (("left_image_path", 1), ("right_image_path", -1)):
                image_path = row.get(column, "").strip()
                if not image_path:
                    continue
                if (
                    side_camera_correction is None
                    or not math.isfinite(side_camera_correction)
                    or side_camera_correction < 0
                ):
                    raise ValueError(
                        "side_camera_correction must be a finite nonnegative value for available side cameras."
                    )
                self.samples.append(
                    (
                        self.csv_path.parent / image_path,
                        round(steering + direction * side_camera_correction, 10),
                    )
                )

        for image_path, _ in self.samples:
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image_file:
                image_file.verify()
            with Image.open(image_path) as image_file:
                image_file.load()
        self.targets = [steering for _, steering in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, steering = self.samples[index]
        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")
        if self.augment and torch.rand(()) < 0.5:
            image = transforms.hflip(image)
            steering = -steering
        return preprocess_image(image, self.preprocess), torch.tensor(steering, dtype=torch.float32)
