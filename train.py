"""Train PilotNet from a CSV driving log."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pilotnet.data import DrivingDataset
from pilotnet.engine import evaluate, train_epoch
from pilotnet.models import PilotNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output-dir", default="artifacts/train")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loader_options = {"batch_size": args.batch_size, "num_workers": args.workers, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(DrivingDataset(args.train_csv, augment=True), shuffle=True, **loader_options)
    val_loader = DataLoader(DrivingDataset(args.val_csv), shuffle=False, **loader_options)
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
            torch.save({"model_state_dict": model.state_dict(), "epoch": epoch, "val_metrics": val_metrics, "args": vars(args)}, output_dir / "best.pt")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
