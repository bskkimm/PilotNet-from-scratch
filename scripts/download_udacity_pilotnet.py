"""Download the Kaggle Udacity mirror into a chosen directory and prepare PilotNet CSVs."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

DATASET_HANDLE = "aslanahmedov/self-driving-carbehavioural-cloning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def dataset_root(cache_path: Path) -> Path:
    """Locate the single Udacity driving-log directory in a KaggleHub download."""
    logs = list(cache_path.rglob("driving_log.csv"))
    if len(logs) != 1:
        raise ValueError("Kaggle mirror must contain exactly one driving_log.csv.")
    root = logs[0].parent
    if not (root / "IMG").is_dir():
        raise ValueError("Kaggle mirror driving_log.csv must have a sibling IMG directory.")
    return root


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {args.output_dir}")
    try:
        import kagglehub
    except ImportError as error:
        raise RuntimeError("Install with `python -m pip install -e '.[udacity]'`.") from error
    source = dataset_root(Path(kagglehub.dataset_download(DATASET_HANDLE)))
    shutil.copytree(source, args.output_dir)
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("prepare_udacity_pilotnet.py")),
            "--driving-log",
            str(args.output_dir / "driving_log.csv"),
            "--output-dir",
            str(args.output_dir / "pilotnet"),
            "--val-fraction",
            str(args.val_fraction),
        ],
        check=True,
    )
    print(f"dataset={args.output_dir}")
    print(f"prepared_csvs={args.output_dir / 'pilotnet'}")


if __name__ == "__main__":
    main()
