"""Integration tests for the training command-line interface."""

import json
import subprocess
import sys
from pathlib import Path

import torch
from PIL import Image

from pilotnet.models import PilotNet


PROJECT_ROOT = Path(__file__).parents[1]


def test_training_writes_preprocessing_to_manifest_and_checkpoint(tmp_path) -> None:
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (200, 66)).save(image)
    for name in ("train.csv", "val.csv"):
        (tmp_path / name).write_text("image_path,steering\nframe.jpg,0.0\n", encoding="utf-8")
    output_dir = tmp_path / "run"

    subprocess.run(
        [
            sys.executable,
            "train.py",
            "--train-csv",
            str(tmp_path / "train.csv"),
            "--val-csv",
            str(tmp_path / "val.csv"),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--workers",
            "0",
            "--crop-top-fraction",
            "0.1",
            "--balance-bins",
            "5",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    manifest = json.loads((output_dir / "run_manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(output_dir / "best.pt", map_location="cpu", weights_only=False)

    assert manifest["preprocessing"]["crop_top_fraction"] == 0.1
    assert manifest["sampling"]["balance_bins"] == 5
    assert checkpoint["run_manifest"] == manifest


def test_evaluation_rejects_checkpoint_without_preprocessing_metadata(tmp_path) -> None:
    image = tmp_path / "frame.jpg"
    Image.new("RGB", (200, 66)).save(image)
    csv_path = tmp_path / "eval.csv"
    csv_path.write_text("image_path,steering\nframe.jpg,0.0\n", encoding="utf-8")
    checkpoint_path = tmp_path / "missing-metadata.pt"
    torch.save({"model_state_dict": PilotNet().state_dict()}, checkpoint_path)

    result = subprocess.run(
        [
            sys.executable,
            "eval.py",
            "--checkpoint",
            str(checkpoint_path),
            "--csv",
            str(csv_path),
            "--workers",
            "0",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "preprocessing metadata" in result.stderr
