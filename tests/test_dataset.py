"""Tests for driving-log loading and geometric augmentation."""

import csv

import pytest
import torch
from PIL import Image

from pilotnet.data import DrivingDataset


def test_dataset_loads_and_resizes_images(tmp_path) -> None:
    Image.new("RGB", (400, 100), color="red").save(tmp_path / "frame.jpg")
    with (tmp_path / "log.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["image_path", "steering"])
        writer.writeheader()
        writer.writerow({"image_path": "frame.jpg", "steering": "0.25"})

    image, steering = DrivingDataset(tmp_path / "log.csv")[0]

    assert image.shape == (3, 66, 200)
    assert steering.item() == pytest.approx(0.25)


def test_horizontal_flip_negates_the_steering_target(tmp_path, monkeypatch) -> None:
    Image.new("RGB", (200, 66), color="blue").save(tmp_path / "frame.jpg")
    (tmp_path / "log.csv").write_text("image_path,steering\nframe.jpg,0.25\n", encoding="utf-8")
    monkeypatch.setattr(torch, "rand", lambda _: torch.tensor(0.0))

    _, steering = DrivingDataset(tmp_path / "log.csv", augment=True)[0]

    assert steering.item() == pytest.approx(-0.25)
