"""Convert Udacity Behavioral Cloning logs into PilotNet train/validation CSVs."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driving-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def resolve_image_path(raw_path: str, log_dir: Path) -> Path:
    """Resolve Udacity's often machine-specific camera paths by filename."""
    candidate = Path(raw_path.strip())
    if candidate.is_file():
        return candidate
    relative_candidate = log_dir / candidate
    if relative_candidate.is_file():
        return relative_candidate
    image_path = log_dir / "IMG" / candidate.name
    if image_path.is_file():
        return image_path
    raise FileNotFoundError(raw_path.strip())


def load_rows(driving_log: Path) -> list[tuple[Path, Path, Path, float]]:
    """Read Udacity's headerless center/left/right camera log."""
    rows = []
    with driving_log.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.reader(file)
        for number, row in enumerate(reader, start=1):
            if not row:
                continue
            if len(row) < 4:
                raise ValueError(f"Expected at least four columns in row {number}.")
            if number == 1 and row[0].strip().lower() == "center":
                continue
            try:
                steering = float(row[3].strip())
            except ValueError as error:
                raise ValueError(f"Invalid steering value in row {number}.") from error
            rows.append(
                (
                    resolve_image_path(row[0], driving_log.parent),
                    resolve_image_path(row[1], driving_log.parent),
                    resolve_image_path(row[2], driving_log.parent),
                    steering,
                )
            )
    if len(rows) < 2:
        raise ValueError("Udacity driving log needs at least two samples.")
    return rows


def write_split(rows: list[tuple[Path, Path, Path, float]], output_csv: Path) -> None:
    """Write the generic CSV format consumed by ``DrivingDataset``."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image_path", "steering", "left_image_path", "right_image_path"],
        )
        writer.writeheader()
        for center, left, right, steering in rows:
            writer.writerow(
                {
                    "image_path": os.path.relpath(center, output_csv.parent),
                    "steering": steering,
                    "left_image_path": os.path.relpath(left, output_csv.parent),
                    "right_image_path": os.path.relpath(right, output_csv.parent),
                }
            )


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("val_fraction must be between zero and one.")
    rows = load_rows(args.driving_log)
    validation_size = max(1, round(len(rows) * args.val_fraction))
    if validation_size >= len(rows):
        raise ValueError("val_fraction leaves no training samples.")
    split = len(rows) - validation_size
    write_split(rows[:split], args.output_dir / "train.csv")
    write_split(rows[split:], args.output_dir / "val.csv")
    print({"train_rows": split, "val_rows": validation_size, "output_dir": str(args.output_dir)})


if __name__ == "__main__":
    main()
