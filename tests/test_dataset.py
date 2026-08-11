"""Tests for driving-log loading and geometric augmentation."""

import csv

import pytest
import torch
from PIL import Image, UnidentifiedImageError

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


def test_dataset_expands_available_side_cameras(tmp_path) -> None:
    for name in ("center.jpg", "left.jpg", "right.jpg"):
        Image.new("RGB", (200, 66)).save(tmp_path / name)
    (tmp_path / "log.csv").write_text(
        "image_path,steering,left_image_path,right_image_path\n"
        "center.jpg,0.1,left.jpg,right.jpg\n",
        encoding="utf-8",
    )

    dataset = DrivingDataset(tmp_path / "log.csv", side_camera_correction=0.2)

    assert dataset.targets == [0.1, 0.3, -0.1]


def test_dataset_requires_correction_for_available_side_camera(tmp_path) -> None:
    Image.new("RGB", (200, 66)).save(tmp_path / "center.jpg")
    Image.new("RGB", (200, 66)).save(tmp_path / "left.jpg")
    (tmp_path / "log.csv").write_text(
        "image_path,steering,left_image_path\ncenter.jpg,0.1,left.jpg\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="side_camera_correction"):
        DrivingDataset(tmp_path / "log.csv")


@pytest.mark.parametrize("steering", ["nan", "inf", "-inf"])
def test_dataset_rejects_nonfinite_steering_values(tmp_path, steering) -> None:
    Image.new("RGB", (200, 66)).save(tmp_path / "center.jpg")
    (tmp_path / "log.csv").write_text(
        f"image_path,steering\ncenter.jpg,{steering}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="steering"):
        DrivingDataset(tmp_path / "log.csv")


def test_dataset_rejects_nonimage_paths_during_construction(tmp_path) -> None:
    (tmp_path / "not-an-image.txt").write_text("not an image", encoding="utf-8")
    (tmp_path / "log.csv").write_text(
        "image_path,steering\nnot-an-image.txt,0.1\n", encoding="utf-8"
    )

    with pytest.raises(UnidentifiedImageError):
        DrivingDataset(tmp_path / "log.csv")


def test_dataset_rejects_corrupt_images_during_construction(tmp_path) -> None:
    image_path = tmp_path / "truncated.jpg"
    Image.new("RGB", (200, 66)).save(image_path)
    image_path.write_bytes(image_path.read_bytes()[:-10])
    (tmp_path / "log.csv").write_text(
        "image_path,steering\ntruncated.jpg,0.1\n", encoding="utf-8"
    )

    with pytest.raises(OSError):
        DrivingDataset(tmp_path / "log.csv")


def test_dataset_rejects_negative_correction_for_available_side_camera(tmp_path) -> None:
    for name in ("center.jpg", "left.jpg"):
        Image.new("RGB", (200, 66)).save(tmp_path / name)
    (tmp_path / "log.csv").write_text(
        "image_path,steering,left_image_path\ncenter.jpg,0.1,left.jpg\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="side_camera_correction"):
        DrivingDataset(tmp_path / "log.csv", side_camera_correction=-0.1)


@pytest.mark.parametrize("left_path,right_path", [("", ""), ("   ", "\t")])
def test_dataset_ignores_blank_side_camera_cells(tmp_path, left_path, right_path) -> None:
    Image.new("RGB", (200, 66)).save(tmp_path / "center.jpg")
    (tmp_path / "log.csv").write_text(
        "image_path,steering,left_image_path,right_image_path\n"
        f"center.jpg,0.1,{left_path},{right_path}\n",
        encoding="utf-8",
    )

    dataset = DrivingDataset(tmp_path / "log.csv")

    assert dataset.targets == [0.1]
