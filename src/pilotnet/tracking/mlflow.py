"""Optional MLflow logging for PilotNet training runs."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingEta:
    """Elapsed and estimated remaining training time in seconds."""

    elapsed_seconds: float
    mean_epoch_seconds: float
    eta_seconds: float
    expected_finish_unix_seconds: float


def estimate_training_eta(
    start_time: float,
    now: float,
    completed_epochs: int,
    total_epochs: int,
) -> TrainingEta:
    """Estimate remaining time from completed whole epochs."""
    if total_epochs < 1:
        raise ValueError("total_epochs must be positive.")
    if completed_epochs < 1 or completed_epochs > total_epochs:
        raise ValueError("completed_epochs must be between 1 and total_epochs.")
    elapsed_seconds = max(now - start_time, 0.0)
    mean_epoch_seconds = elapsed_seconds / completed_epochs
    eta_seconds = mean_epoch_seconds * (total_epochs - completed_epochs)
    return TrainingEta(
        elapsed_seconds=elapsed_seconds,
        mean_epoch_seconds=mean_epoch_seconds,
        eta_seconds=eta_seconds,
        expected_finish_unix_seconds=now + eta_seconds,
    )


class MlflowTracker:
    """Log a training run without making MLflow a core package dependency."""

    def __init__(
        self,
        *,
        tracking_uri: str,
        experiment_name: str,
        run_name: str | None,
        parameters: dict[str, object],
    ) -> None:
        try:
            import mlflow
        except ImportError as error:
            message = "Install MLflow with `python -m pip install -e '.[mlflow]'`."
            raise RuntimeError(message) from error
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.mlflow = mlflow
        self.mlflow.start_run(run_name=run_name)
        self.mlflow.log_params({key: str(value) for key, value in parameters.items()})
        self.start_time = time.time()

    def log_artifact(self, path: str | Path, artifact_path: str | None = None) -> None:
        """Log an existing local file as an MLflow artifact."""
        self.mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def log_epoch(
        self,
        metrics: dict[str, float | int],
        epoch: int,
        total_epochs: int,
    ) -> TrainingEta:
        """Log epoch metrics and ETA using consistent names for CLI and notebooks."""
        eta = estimate_training_eta(self.start_time, time.time(), epoch, total_epochs)
        self.mlflow.log_metrics(
            {
                **{name: float(value) for name, value in metrics.items() if name != "epoch"},
                "training_progress_percent": 100.0 * epoch / total_epochs,
                "training_elapsed_hours": eta.elapsed_seconds / 3600.0,
                "training_mean_epoch_hours": eta.mean_epoch_seconds / 3600.0,
                "training_eta_hours": eta.eta_seconds / 3600.0,
                "training_expected_finish_unix_seconds": eta.expected_finish_unix_seconds,
            },
            step=epoch,
        )
        return eta

    def close(self) -> None:
        """Close the active MLflow run."""
        self.mlflow.end_run()
