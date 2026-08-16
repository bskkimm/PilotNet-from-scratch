"""Benchmark PilotNet batch-size and DataLoader-worker candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pilotnet.data import PreprocessConfig
from pilotnet.tools import benchmark_configurations, select_best_configuration


def _integers(value: str) -> list[int]:
    values = [int(item) for item in value.split(",")]
    if not values or any(item < 0 for item in values):
        raise argparse.ArgumentTypeError("Candidates must be nonnegative integers.")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--batch-sizes", type=_integers, default=[32, 64, 128])
    parser.add_argument("--workers", type=_integers, default=[0, 2, 4])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--crop-top-fraction", type=float, default=0.0)
    parser.add_argument("--crop-bottom-fraction", type=float, default=0.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results, failures = benchmark_configurations(
        args.train_csv,
        preprocess=PreprocessConfig(args.crop_top_fraction, args.crop_bottom_fraction),
        device=torch.device(args.device),
        batch_sizes=args.batch_sizes,
        workers=args.workers,
        steps=args.steps,
        lr=args.lr,
    )
    selected = select_best_configuration(results)
    output = {
        "selected": selected.to_dict(),
        "results": [result.to_dict() for result in results],
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
