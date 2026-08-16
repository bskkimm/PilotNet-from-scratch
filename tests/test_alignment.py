"""Tests for causal image-to-CAN timestamp alignment."""

import pytest

from pilotnet.data import TimestampInterpolator, align_control


def test_align_control_interpolates_messages_at_the_image_timestamp() -> None:
    steering = TimestampInterpolator([{"utime": 100, "value": 1.5}, {"utime": 200, "value": 2.5}])
    speed = TimestampInterpolator(
        [{"utime": 50, "vehicle_speed": 10.0}, {"utime": 150, "vehicle_speed": 12.0}]
    )

    control = align_control(
        125,
        steering,
        speed,
        max_steering_gap_us=100,
        max_speed_gap_us=100,
    )

    assert control is not None
    assert control.steering == 1.75
    assert control.speed == 11.5
    assert control.steering_before_gap_us == 25
    assert control.steering_after_gap_us == 75
    assert control.speed_before_gap_us == 75
    assert control.speed_after_gap_us == 25


def test_align_control_rejects_missing_or_stale_brackets() -> None:
    steering = TimestampInterpolator([{"utime": 100, "value": 1.5}])
    speed = TimestampInterpolator([{"utime": 100, "vehicle_speed": 10.0}])

    control = align_control(250, steering, speed, max_steering_gap_us=100, max_speed_gap_us=200)

    assert control is None


def test_interpolator_rejects_nonincreasing_timestamps() -> None:
    with pytest.raises(ValueError, match="strictly sorted"):
        TimestampInterpolator([{"utime": 2}, {"utime": 1}])
