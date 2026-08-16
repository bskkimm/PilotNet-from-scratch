"""Integration tests for the training command-line interface."""

import json
from importlib.metadata import version
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
            "--device",
            "cpu",
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
    assert manifest["configuration"] == {
        "artifact_dir": str(output_dir.resolve()),
        "batch_size": 64,
        "crop_bottom_fraction": 0.0,
        "crop_top_fraction": 0.1,
        "deterministic": False,
        "device": "cpu",
        "epochs": 1,
            "lr": 1e-4,
            "mlflow_experiment": "PilotNet",
            "mlflow_run_name": None,
            "bev_every_epochs": 0,
            "bev_dataroot": None,
            "bev_version": "v1.0-trainval",
            "bev_scene": None,
            "bev_anchor": 0,
            "bev_horizon": 40,
            "bev_wheelbase": 2.5,
            "bev_steering_scale": 1.0,
            "mlflow_tracking_uri": None,
            "seed": 42,
        "side_camera_correction": None,
        "balance_bins": 5,
        "train_csv": str((tmp_path / "train.csv").resolve()),
        "val_csv": str((tmp_path / "val.csv").resolve()),
        "weight_decay": 0.0,
        "workers": 0,
    }
    assert manifest["datasets"] == {
        "train_csv": str((tmp_path / "train.csv").resolve()),
        "val_csv": str((tmp_path / "val.csv").resolve()),
        "train_size": 1,
        "val_size": 1,
    }
    assert manifest["environment"]["device"] == "cpu"
    assert manifest["environment"]["python"] == sys.version
    assert manifest["environment"]["packages"] == {
        "numpy": version("numpy"),
        "pillow": version("Pillow"),
        "torch": version("torch"),
        "torchvision": version("torchvision"),
    }
    assert checkpoint["run_manifest"] == manifest


def test_training_expands_validation_side_cameras_without_a_sampler(tmp_path, monkeypatch) -> None:
    import train

    for name in ("center.jpg", "left.jpg", "right.jpg"):
        Image.new("RGB", (200, 66)).save(tmp_path / name)
    for name in ("train.csv", "val.csv"):
        (tmp_path / name).write_text(
            "image_path,steering,left_image_path,right_image_path\n"
            "center.jpg,0.1,left.jpg,right.jpg\n",
            encoding="utf-8",
        )

    args = type(
        "Args",
        (),
        {
            "train_csv": str(tmp_path / "train.csv"),
            "val_csv": str(tmp_path / "val.csv"),
            "artifact_dir": str(tmp_path / "run"),
            "epochs": 0,
            "batch_size": 1,
            "workers": 0,
            "bev_anchor": 0,
            "bev_dataroot": None,
            "bev_every_epochs": 0,
            "bev_horizon": 40,
            "bev_scene": None,
            "bev_steering_scale": 1.0,
            "bev_version": "v1.0-trainval",
            "bev_wheelbase": 2.5,
            "lr": 1e-4,
            "weight_decay": 0.0,
            "seed": 42,
            "deterministic": False,
            "crop_top_fraction": 0.0,
            "crop_bottom_fraction": 0.0,
            "side_camera_correction": 0.2,
            "balance_bins": 5,
            "device": "cpu",
            "mlflow_tracking_uri": None,
            "mlflow_experiment": "PilotNet",
            "mlflow_run_name": None,
        },
    )()
    loaders = []

    class RecordingDataLoader:
        def __init__(self, dataset, **kwargs) -> None:
            loaders.append((dataset, kwargs))

    monkeypatch.setattr(train, "parse_args", lambda: args)
    monkeypatch.setattr(train, "DataLoader", RecordingDataLoader)

    train.main()

    validation_dataset, validation_options = loaders[1]
    assert len(validation_dataset) == 3
    assert validation_options == {
        "batch_size": 1,
        "num_workers": 0,
        "pin_memory": False,
        "shuffle": False,
    }


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
