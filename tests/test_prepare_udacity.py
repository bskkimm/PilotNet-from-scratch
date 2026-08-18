"""Integration tests for the Udacity Behavioral Cloning CSV converter."""

import csv
import subprocess
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).parents[1]


def test_converter_writes_contiguous_train_and_validation_csvs(tmp_path) -> None:
    image_dir = tmp_path / "udacity" / "IMG"
    image_dir.mkdir(parents=True)
    rows = []
    for index in range(5):
        paths = []
        for camera in ("center", "left", "right"):
            image_path = image_dir / f"{camera}_{index}.jpg"
            Image.new("RGB", (200, 66)).save(image_path)
            paths.append(f" /old/machine/IMG/{image_path.name} ")
        rows.append([*paths, str(index / 10), "0.0", "0.0", "10.0"])
    log_path = image_dir.parent / "driving_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as file:
        csv.writer(file).writerows(rows)
    output_dir = tmp_path / "prepared"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_udacity_pilotnet.py",
            "--driving-log",
            str(log_path),
            "--output-dir",
            str(output_dir),
            "--val-fraction",
            "0.4",
        ],
        check=True,
        cwd=PROJECT_ROOT,
    )

    with (output_dir / "train.csv").open(newline="", encoding="utf-8") as file:
        train_rows = list(csv.DictReader(file))
    with (output_dir / "val.csv").open(newline="", encoding="utf-8") as file:
        val_rows = list(csv.DictReader(file))

    assert [row["steering"] for row in train_rows] == ["0.0", "0.1", "0.2"]
    assert [row["steering"] for row in val_rows] == ["0.3", "0.4"]
    assert (output_dir / train_rows[0]["image_path"]).is_file()
    assert (output_dir / train_rows[0]["left_image_path"]).is_file()
    assert (output_dir / train_rows[0]["right_image_path"]).is_file()
