"""Tests for causal temporal driving inputs and the ResNet-GRU model."""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from pilotnet.data import TemporalDrivingDataset, TemporalPreprocessConfig
from pilotnet.models import TemporalResNetGRU

PROJECT_ROOT = Path(__file__).parents[1]


def test_temporal_dataset_builds_causal_windows_without_crossing_scenes(tmp_path) -> None:
    for index in range(5):
        Image.new("RGB", (40, 20), color=(index, 0, 0)).save(tmp_path / f"{index}.jpg")
    with (tmp_path / "log.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "steering", "speed", "timestamp", "scene_name"],
        )
        writer.writeheader()
        for index, scene in enumerate(("scene-a", "scene-a", "scene-a", "scene-b", "scene-b")):
            writer.writerow(
                {
                    "image_path": f"{index}.jpg",
                    "steering": str(index / 10),
                    "speed": str(index),
                    "timestamp": str(index * 100_000),
                    "scene_name": scene,
                }
            )

    dataset = TemporalDrivingDataset(
        tmp_path / "log.csv",
        sequence_length=3,
        preprocess=TemporalPreprocessConfig(height=16, width=32),
    )

    assert len(dataset) == 1
    images, speeds, steering = dataset[0]
    assert images.shape == (3, 3, 16, 32)
    assert speeds.tolist() == pytest.approx([0.0, 1 / 30, 2 / 30])
    assert steering.item() == pytest.approx(0.2)


def test_temporal_resnet_gru_returns_one_steering_value_per_sequence() -> None:
    model = TemporalResNetGRU(pretrained=False, hidden_size=32)

    prediction = model(torch.rand(2, 3, 3, 64, 128), torch.rand(2, 3))

    assert prediction.shape == (2,)


def test_temporal_resnet_gru_rejects_mismatched_speed_sequence() -> None:
    model = TemporalResNetGRU(pretrained=False, hidden_size=32)

    with pytest.raises(ValueError, match="speeds"):
        model(torch.rand(1, 3, 3, 64, 128), torch.rand(1, 2))


def test_temporal_training_cli_writes_a_json_manifest(tmp_path) -> None:
    for index in range(3):
        Image.new("RGB", (40, 20)).save(tmp_path / f"{index}.jpg")
    for name in ("train.csv", "val.csv"):
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["image_path", "steering", "speed", "timestamp", "scene_name"],
            )
            writer.writeheader()
            for index in range(3):
                writer.writerow(
                    {
                        "image_path": f"{index}.jpg",
                        "steering": "0.1",
                        "speed": "2.0",
                        "timestamp": str(index * 100_000),
                        "scene_name": "scene-a",
                    }
                )
    output_dir = tmp_path / "run"
    environment = {**os.environ, "PYTHONPATH": str(PROJECT_ROOT / "src")}

    subprocess.run(
        [
            sys.executable,
            "train_temporal.py",
            "--train-csv",
            str(tmp_path / "train.csv"),
            "--val-csv",
            str(tmp_path / "val.csv"),
            "--output-dir",
            str(output_dir),
            "--sequence-length",
            "3",
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--height",
            "64",
            "--width",
            "128",
            "--device",
            "cpu",
            "--no-pretrained",
        ],
        check=True,
        cwd=PROJECT_ROOT,
        env=environment,
    )

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["configuration"]["pretrained"] is False
    assert manifest["datasets"]["train_sequences"] == 1
    assert (output_dir / "last.pt").is_file()
