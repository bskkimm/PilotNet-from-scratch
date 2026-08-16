"""Tests for causal image-to-CAN timestamp alignment."""

import pytest

from pilotnet.data import PreviousCanMessage, align_control


def test_align_control_uses_latest_prior_messages_within_each_gap() -> None:
    steering = PreviousCanMessage([{"utime": 100, "value": 1.5}, {"utime": 200, "value": 2.5}])
    speed = PreviousCanMessage(
        [{"utime": 50, "vehicle_speed": 10.0}, {"utime": 150, "vehicle_speed": 12.0}]
    )

    control = align_control(
        220,
        steering,
        speed,
        max_steering_gap_us=30,
        max_speed_gap_us=100,
    )

    assert control is not None
    assert control.steering == 2.5
    assert control.speed == 12.0
    assert control.steering_gap_us == 20
    assert control.speed_gap_us == 70


def test_align_control_rejects_missing_or_stale_messages() -> None:
    steering = PreviousCanMessage([{"utime": 100, "value": 1.5}])
    speed = PreviousCanMessage([{"utime": 100, "vehicle_speed": 10.0}])

    control = align_control(250, steering, speed, max_steering_gap_us=100, max_speed_gap_us=200)

    assert control is None


def test_message_aligner_rejects_unsorted_timestamps() -> None:
    with pytest.raises(ValueError, match="sorted"):
        PreviousCanMessage([{"utime": 2}, {"utime": 1}])
