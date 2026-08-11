"""Train PilotNet from a CSV driving log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pilotnet.data import DrivingDataset, PreprocessConfig, build_balanced_sampler
from pilotnet.engine import evaluate, train_epoch
from pilotnet.models import PilotNet
from pilotnet.utils import build_run_manifest, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument(
        "--output-dir", "--artifact-dir", dest="artifact_dir", default="artifacts/train"
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--crop-top-fraction", type=float, default=0.0)
    parser.add_argument("--crop-bottom-fraction", type=float, default=0.0)
    parser.add_argument("--side-camera-correction", type=float)
    parser.add_argument("--balance-bins", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reproducibility = seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device)
    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocessing = PreprocessConfig(
        crop_top_fraction=args.crop_top_fraction,
        crop_bottom_fraction=args.crop_bottom_fraction,
    )
    train_dataset = DrivingDataset(
        args.train_csv,
        augment=True,
        preprocess=preprocessing,
        side_camera_correction=args.side_camera_correction,
    )
    val_dataset = DrivingDataset(args.val_csv, preprocess=preprocessing)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        sampler=build_balanced_sampler(train_dataset.targets, args.balance_bins),
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    run_manifest = build_run_manifest(
        preprocessing=preprocessing,
        sampling={"balance_bins": args.balance_bins},
        reproducibility=reproducibility,
        training={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "workers": args.workers,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "side_camera_correction": args.side_camera_correction,
        },
    )
    (artifact_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    model = PilotNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float | int]] = []
    best_mse = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch, **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        print(json.dumps(row))
        if val_metrics["mse"] < best_mse:
            best_mse = val_metrics["mse"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "args": vars(args),
                    "run_manifest": run_manifest,
                },
                artifact_dir / "best.pt",
            )
    (artifact_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
