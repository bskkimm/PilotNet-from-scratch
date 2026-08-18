"""Update an active PilotNet MLflow run with progress and ETA status."""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from mlflow.tracking import MlflowClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--total-epochs", type=int, required=True)
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--initial-epochs", type=int, default=0)
    parser.add_argument("--initial-elapsed-hours", type=float, default=0.0)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--timezone", default="Asia/Seoul")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def format_duration(seconds: float) -> str:
    total_minutes = max(round(seconds / 60.0), 0)
    hours, minutes = divmod(total_minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def latest_epoch(client: MlflowClient, run_id: str) -> tuple[int, float] | None:
    history = client.get_metric_history(run_id, "train_mse")
    if not history:
        return None
    metric = max(history, key=lambda item: item.step)
    return metric.step, metric.timestamp / 1000.0


def overview(
    *,
    completed_epochs: int,
    total_epochs: int,
    start_time: float,
    initial_elapsed_seconds: float,
    latest_completion: float | None,
    now: float,
    timezone_name: str,
) -> tuple[dict[str, float], str, str]:
    elapsed_seconds = max(initial_elapsed_seconds + now - start_time, 0.0)
    metrics = {
        "training_completed_epochs": float(completed_epochs),
        "training_progress_percent": 100.0 * completed_epochs / total_epochs,
        "training_elapsed_hours": elapsed_seconds / 3600.0,
    }
    lines = ["## Training Progress", "", f"- **Elapsed:** {format_duration(elapsed_seconds)}"]
    finish_text = "Unknown until epoch 1 completes"
    if completed_epochs and latest_completion is not None:
        mean_epoch_seconds = (
            initial_elapsed_seconds + latest_completion - start_time
        ) / completed_epochs
        expected_finish = latest_completion + mean_epoch_seconds * (total_epochs - completed_epochs)
        eta_seconds = max(expected_finish - now, 0.0)
        metrics.update(
            {
                "training_mean_epoch_hours": mean_epoch_seconds / 3600.0,
                "training_estimated_total_hours": (expected_finish - start_time) / 3600.0,
                "training_eta_hours": eta_seconds / 3600.0,
                "training_expected_finish_unix_seconds": expected_finish,
            }
        )
        finish_text = datetime.fromtimestamp(expected_finish, timezone.utc).astimezone(
            ZoneInfo(timezone_name)
        ).strftime("%Y-%m-%d %H:%M:%S %Z")
        lines.extend(
            [
                f"- **Average epoch:** {format_duration(mean_epoch_seconds)}",
                f"- **Estimated total:** {format_duration(expected_finish - start_time)}",
                f"- **Remaining:** {format_duration(eta_seconds)}",
            ]
        )
    updated_at = datetime.fromtimestamp(now, timezone.utc).astimezone(ZoneInfo(timezone_name))
    lines.extend(
        [
            f"- **Progress:** {completed_epochs} / {total_epochs} epochs "
            f"({metrics['training_progress_percent']:.1f}%)",
            f"- **Expected finish:** {finish_text}",
            "",
            f"Last updated: {updated_at.strftime('%Y-%m-%d %H:%M:%S %Z')}",
        ]
    )
    return metrics, "\n".join(lines), finish_text


def update(client: MlflowClient, args: argparse.Namespace) -> int:
    run = client.get_run(args.run_id)
    latest = latest_epoch(client, args.run_id)
    completed_epochs, completion_time = latest or (args.initial_epochs, None)
    now = time.time()
    metrics, description, finish_text = overview(
        completed_epochs=completed_epochs,
        total_epochs=args.total_epochs,
        start_time=run.info.start_time / 1000.0,
        initial_elapsed_seconds=args.initial_elapsed_hours * 3600.0,
        latest_completion=completion_time,
        now=now,
        timezone_name=args.timezone,
    )
    timestamp = round(now * 1000)
    for name, value in metrics.items():
        client.log_metric(args.run_id, name, value, timestamp=timestamp, step=completed_epochs)
    client.set_tag(args.run_id, "mlflow.note.content", description)
    client.set_tag(args.run_id, "eta_monitor_status", "running")
    client.set_tag(args.run_id, "eta_monitor_timezone", args.timezone)
    updated_at_utc = datetime.fromtimestamp(now, timezone.utc).isoformat()
    client.set_tag(args.run_id, "eta_monitor_updated_at", updated_at_utc)
    client.set_tag(args.run_id, "training_expected_finish_local", finish_text)
    return completed_epochs


def main() -> None:
    args = parse_args()
    if (
        args.total_epochs < 1
        or args.poll_interval <= 0
        or not 0 <= args.initial_epochs <= args.total_epochs
        or args.initial_elapsed_hours < 0
    ):
        raise ValueError("Monitor arguments must be valid nonnegative values.")
    client = MlflowClient(tracking_uri=args.tracking_uri)
    while True:
        completed_epochs = update(client, args)
        status = client.get_run(args.run_id).info.status
        if args.once or completed_epochs >= args.total_epochs or status != "RUNNING":
            return
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
