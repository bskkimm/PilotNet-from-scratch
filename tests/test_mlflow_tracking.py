"""Tests for MLflow-independent tracking calculations."""

import pytest

from pilotnet.tracking.mlflow import estimate_training_eta


def test_eta_uses_mean_completed_epoch_duration() -> None:
    eta = estimate_training_eta(start_time=100.0, now=160.0, completed_epochs=2, total_epochs=5)

    assert eta.elapsed_seconds == 60.0
    assert eta.mean_epoch_seconds == 30.0
    assert eta.eta_seconds == 90.0
    assert eta.expected_finish_unix_seconds == 250.0


@pytest.mark.parametrize("completed_epochs", [0, 4])
def test_eta_rejects_invalid_completed_epoch_count(completed_epochs: int) -> None:
    with pytest.raises(ValueError, match="completed_epochs"):
        estimate_training_eta(0.0, 10.0, completed_epochs, total_epochs=3)
