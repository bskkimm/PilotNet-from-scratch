"""Timestamp interpolation for camera frames and CAN messages."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AlignedControl:
    """Interpolated CAN steering and speed at one camera timestamp."""

    steering: float
    speed: float
    steering_before_gap_us: int
    steering_after_gap_us: int
    speed_before_gap_us: int
    speed_after_gap_us: int


class TimestampInterpolator:
    """Linearly interpolate finite numeric CAN fields at a timestamp."""

    def __init__(self, messages: Sequence[Mapping[str, object]]) -> None:
        self.messages = list(messages)
        self.timestamps = [self._timestamp(message) for message in self.messages]
        timestamp_pairs = zip(self.timestamps, self.timestamps[1:])
        if any(current <= previous for previous, current in timestamp_pairs):
            raise ValueError("CAN messages must be strictly sorted by utime.")

    @staticmethod
    def _timestamp(message: Mapping[str, object]) -> int:
        timestamp = message.get("utime")
        if not isinstance(timestamp, int):
            raise ValueError("CAN messages must contain integer utime values.")
        return timestamp

    def interpolate(
        self,
        timestamp_us: int,
        field: str,
        max_gap_us: int,
    ) -> tuple[float, int, int] | None:
        """Return ``field`` at ``timestamp_us`` when both brackets are close enough."""
        if max_gap_us < 0:
            raise ValueError("max_gap_us must be nonnegative.")
        index = bisect_left(self.timestamps, timestamp_us)
        if index < len(self.timestamps) and self.timestamps[index] == timestamp_us:
            value = self.messages[index].get(field)
            if isinstance(value, (int, float)) and isfinite(value):
                return float(value), 0, 0
            return None
        if index == 0 or index == len(self.timestamps):
            return None
        before_timestamp = self.timestamps[index - 1]
        after_timestamp = self.timestamps[index]
        before_gap_us = timestamp_us - before_timestamp
        after_gap_us = after_timestamp - timestamp_us
        if before_gap_us > max_gap_us or after_gap_us > max_gap_us:
            return None
        before_value = self.messages[index - 1].get(field)
        after_value = self.messages[index].get(field)
        if not isinstance(before_value, (int, float)) or not isinstance(after_value, (int, float)):
            return None
        if not isfinite(before_value) or not isfinite(after_value):
            return None
        fraction = before_gap_us / (after_timestamp - before_timestamp)
        value = before_value + fraction * (after_value - before_value)
        return float(value), before_gap_us, after_gap_us


def align_control(
    timestamp_us: int,
    steering_messages: TimestampInterpolator,
    speed_messages: TimestampInterpolator,
    *,
    max_steering_gap_us: int,
    max_speed_gap_us: int,
) -> AlignedControl | None:
    """Interpolate steering and speed at one image timestamp."""
    steering_match = steering_messages.interpolate(timestamp_us, "value", max_steering_gap_us)
    speed_match = speed_messages.interpolate(timestamp_us, "vehicle_speed", max_speed_gap_us)
    if steering_match is None or speed_match is None:
        return None
    steering, steering_before_gap_us, steering_after_gap_us = steering_match
    speed, speed_before_gap_us, speed_after_gap_us = speed_match
    return AlignedControl(
        steering=steering,
        speed=speed,
        steering_before_gap_us=steering_before_gap_us,
        steering_after_gap_us=steering_after_gap_us,
        speed_before_gap_us=speed_before_gap_us,
        speed_after_gap_us=speed_after_gap_us,
    )
