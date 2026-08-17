"""Train a causal ResNet-34 + GRU steering model from nuScenes CSV logs."""

from __future__ import annotations

import argparse
import json
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pilotnet.data import TemporalDrivingDataset, TemporalPreprocessConfig
from pilotnet.engine import evaluate_temporal, train_temporal_epoch
from pilotnet.evaluation import write_bev_artifacts
from pilotnet.models import TemporalResNetGRU
from pilotnet.tracking import MlflowTracker
from pilotnet.utils import build_run_manifest, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--output-dir", default="artifacts/temporal")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--sequence-length", type=int, default=5)
    parser.add_argument("--max-frame-gap-us", type=int, default=200_000)
    parser.add_argument("--speed-scale", type=float, default=30.0)
    parser.add_argument("--height", type=int, default=160)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--crop-top-fraction", type=float, default=0.0)
    parser.add_argument("--crop-bottom-fraction", type=float, default=0.0)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--mlflow-experiment", default="PilotNet")
    parser.add_argument("--mlflow-run-name")
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


def _checkpoint(
    model: TemporalResNetGRU,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    best_mse: float,
    history: list[dict[str, float | int]],
    args: argparse.Namespace,
    run_manifest: dict[str, object],
) -> dict[str, object]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "val_metrics": metrics,
        "best_val_mse": best_mse,
        "history": history,
        "args": vars(args),
        "run_manifest": run_manifest,
        "architecture": "TemporalResNetGRU",
    }


def main() -> None:
    args = parse_args()
    if args.freeze_encoder_epochs < 0 or args.bev_every_epochs < 0:
        raise ValueError("freeze_encoder_epochs and bev_every_epochs must be nonnegative.")
    if args.workers < 0 or args.prefetch_factor < 1:
        raise ValueError("workers must be nonnegative and prefetch_factor must be positive.")
    if args.bev_every_epochs and not args.bev_dataroot:
        raise ValueError("bev_dataroot is required when BEV evaluation is enabled.")
    reproducibility = seed_everything(args.seed, args.deterministic)
    device = torch.device(args.device)
    artifact_dir = Path(args.output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    preprocess = TemporalPreprocessConfig(
        height=args.height,
        width=args.width,
        crop_top_fraction=args.crop_top_fraction,
        crop_bottom_fraction=args.crop_bottom_fraction,
    )
    dataset_options = {
        "sequence_length": args.sequence_length,
        "max_frame_gap_us": args.max_frame_gap_us,
        "speed_scale": args.speed_scale,
        "preprocess": preprocess,
    }
    train_dataset = TemporalDrivingDataset(args.train_csv, augment=True, **dataset_options)
    val_dataset = TemporalDrivingDataset(args.val_csv, **dataset_options)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        loader_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    configuration = {
        **vars(args),
        "output_dir": str(artifact_dir.resolve()),
        "train_csv": str(train_dataset.csv_path.resolve()),
        "val_csv": str(val_dataset.csv_path.resolve()),
        "resume": str(args.resume.resolve()) if args.resume else None,
        "device": str(device),
        "pretrained": not args.no_pretrained,
    }
    run_manifest = build_run_manifest(
        preprocessing={"temporal_rgb": vars(preprocess)},
        sampling={"strategy": "shuffle"},
        reproducibility=reproducibility,
        training={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr},
        configuration=configuration,
        datasets={
            "train_csv": configuration["train_csv"],
            "val_csv": configuration["val_csv"],
            "train_sequences": len(train_dataset),
            "val_sequences": len(val_dataset),
        },
        environment={
            "device": str(device),
            "python": sys.version,
            "packages": {
                name: version(name) for name in ("numpy", "Pillow", "torch", "torchvision")
            },
        },
    )
    manifest_path = artifact_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(run_manifest, indent=2), encoding="utf-8")
    tracker = None
    if args.mlflow_tracking_uri:
        tracker = MlflowTracker(
            tracking_uri=args.mlflow_tracking_uri,
            experiment_name=args.mlflow_experiment,
            run_name=args.mlflow_run_name,
            parameters={"architecture": "resnet34_gru", **configuration},
        )
        tracker.log_artifact(manifest_path, artifact_path="metadata")
    model = TemporalResNetGRU(
        pretrained=not args.no_pretrained, hidden_size=args.hidden_size
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_mse, start_epoch, history = float("inf"), 0, []
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_mse = float(checkpoint.get("best_val_mse", float("inf")))
        start_epoch = int(checkpoint["epoch"])
        history = list(checkpoint.get("history", []))
        print(f"resumed_checkpoint={args.resume} start_epoch={start_epoch}")
    for epoch in range(start_epoch + 1, args.epochs + 1):
        freeze_encoder = epoch <= args.freeze_encoder_epochs
        for parameter in model.encoder.parameters():
            parameter.requires_grad = not freeze_encoder
        train_metrics = train_temporal_epoch(
            model, train_loader, optimizer, device, freeze_encoder=freeze_encoder
        )
        val_metrics = evaluate_temporal(model, val_loader, device)
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"val_{key}": value for key, value in val_metrics.items()},
        }
        if tracker:
            row["eta_hours"] = tracker.log_epoch(row, epoch, args.epochs).eta_seconds / 3600.0
        history.append(row)
        print(json.dumps(row))
        payload = _checkpoint(
            model, optimizer, epoch, val_metrics, best_mse, history, args, run_manifest
        )
        if val_metrics["mse"] < best_mse:
            best_mse = val_metrics["mse"]
            payload["best_val_mse"] = best_mse
            best_path = artifact_dir / "best.pt"
            torch.save(payload, best_path)
            if tracker:
                tracker.log_artifact(best_path, artifact_path="checkpoints")
        torch.save(payload, artifact_dir / "last.pt")
        if args.bev_every_epochs and epoch % args.bev_every_epochs == 0:
            artifacts = write_bev_artifacts(
                model,
                csv_path=args.val_csv,
                dataroot=args.bev_dataroot,
                version=args.bev_version,
                output_dir=artifact_dir / "bev",
                preprocess=preprocess,
                temporal_preprocess=preprocess,
                sequence_length=args.sequence_length,
                speed_scale=args.speed_scale,
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
            if tracker:
                tracker.log_artifact(artifacts.png, artifact_path="bev")
                tracker.log_artifact(artifacts.gif, artifact_path="bev")
    history_path = artifact_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    if tracker:
        tracker.log_artifact(history_path, artifact_path="metadata")
        tracker.close()


if __name__ == "__main__":
    main()
