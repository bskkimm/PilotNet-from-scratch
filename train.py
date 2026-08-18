"""Train PilotNet from a CSV driving log."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pilotnet.data import DrivingDataset, PreprocessConfig, build_balanced_sampler
from pilotnet.engine import evaluate, train_epoch
from pilotnet.evaluation import write_bev_artifacts
from pilotnet.models import PilotNet
from pilotnet.tracking import MlflowTracker
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
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--mlflow-experiment", default="PilotNet")
    parser.add_argument("--mlflow-run-name")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--bev-every-epochs", type=int, default=0)
    parser.add_argument("--bev-dataroot")
    parser.add_argument("--bev-version", default="v1.0-trainval")
    parser.add_argument("--bev-scene")
    parser.add_argument("--bev-anchor", type=int, default=0)
    parser.add_argument("--bev-horizon", type=int, default=50)
    parser.add_argument("--bev-gif-frames", type=int, default=50)
    parser.add_argument("--bev-gif-frame-duration", type=float, default=0.1)
    parser.add_argument("--bev-wheelbase", type=float, default=2.5)
    parser.add_argument("--bev-steering-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bev_every_epochs < 0:
        raise ValueError("bev_every_epochs must be nonnegative.")
    if args.bev_every_epochs and not args.bev_dataroot:
        raise ValueError("bev_dataroot is required when BEV evaluation is enabled.")
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
    val_dataset = DrivingDataset(
        args.val_csv,
        preprocess=preprocessing,
        side_camera_correction=args.side_camera_correction,
    )
    resolved_train_csv = str(train_dataset.csv_path.resolve())
    resolved_val_csv = str(val_dataset.csv_path.resolve())
    resolved_configuration = {
        **vars(args),
        "artifact_dir": str(artifact_dir.resolve()),
        "device": str(device),
        "train_csv": resolved_train_csv,
        "val_csv": resolved_val_csv,
        "resume": str(args.resume.resolve()) if args.resume is not None else None,
    }
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
        configuration=resolved_configuration,
        datasets={
            "train_csv": resolved_train_csv,
            "val_csv": resolved_val_csv,
            "train_size": len(train_dataset),
            "val_size": len(val_dataset),
        },
        environment={
            "device": str(device),
            "python": sys.version,
            "packages": {
                "numpy": version("numpy"),
                "pillow": version("Pillow"),
                "torch": version("torch"),
                "torchvision": version("torchvision"),
            },
        },
    )
    (artifact_dir / "run_manifest.json").write_text(
        json.dumps(run_manifest, indent=2), encoding="utf-8"
    )
    tracker = None
    if args.mlflow_tracking_uri:
        tracker = MlflowTracker(
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
            run_name=args.mlflow_run_name,
            parameters={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "balance_bins": args.balance_bins,
                "seed": args.seed,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
            },
        )
        tracker.log_artifact(artifact_dir / "run_manifest.json", artifact_path="metadata")
        audit_path = train_dataset.csv_path.parent / "audit.json"
        if audit_path.is_file():
            tracker.log_artifact(audit_path, artifact_path="metadata")
    model = PilotNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    history: list[dict[str, float | int]] = []
    best_mse = float("inf")
    start_epoch = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", 0))
        checkpoint_best_mse = checkpoint.get(
            "best_val_mse",
            checkpoint.get("val_metrics", {}).get("mse"),
        )
        if checkpoint_best_mse is not None:
            best_mse = float(checkpoint_best_mse)
        history = list(checkpoint.get("history", []))
        print(f"resumed_checkpoint={args.resume} start_epoch={start_epoch}")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        if tracker is not None:
            eta = tracker.log_epoch(row, epoch, args.epochs)
            row["eta_hours"] = eta.eta_seconds / 3600.0
        history.append(row)
        print(json.dumps(row))
        if val_metrics["mse"] < best_mse:
            best_mse = val_metrics["mse"]
            checkpoint_path = artifact_dir / "best.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_metrics": val_metrics,
                    "best_val_mse": best_mse,
                    "history": history,
                    "args": vars(args),
                    "run_manifest": run_manifest,
                },
                checkpoint_path,
            )
            if tracker is not None:
                tracker.log_artifact(checkpoint_path, artifact_path="checkpoints")
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "best_val_mse": best_mse,
                "history": history,
                "args": vars(args),
                "run_manifest": run_manifest,
            },
            artifact_dir / "last.pt",
        )
        if args.bev_every_epochs and epoch % args.bev_every_epochs == 0:
            bev_artifacts = write_bev_artifacts(
                model,
                csv_path=args.val_csv,
                dataroot=args.bev_dataroot,
                version=args.bev_version,
                output_dir=artifact_dir / "bev",
                preprocess=preprocessing,
                device=device,
                epoch=epoch,
                scene_name=args.bev_scene,
                anchor=args.bev_anchor,
                horizon=args.bev_horizon,
                gif_frames=args.bev_gif_frames,
                gif_frame_duration=args.bev_gif_frame_duration,
                wheelbase=args.bev_wheelbase,
                steering_scale=args.bev_steering_scale,
            )
            if tracker is not None:
                tracker.log_artifact(bev_artifacts.png, artifact_path="bev")
                tracker.log_artifact(bev_artifacts.gif, artifact_path="bev")
    history_path = artifact_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if tracker is not None:
        tracker.log_artifact(history_path, artifact_path="metadata")
        tracker.close()


if __name__ == "__main__":
    main()
