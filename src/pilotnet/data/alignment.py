"""Timestamp alignment for camera frames and CAN messages."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AlignedControl:
    """CAN steering and speed associated with one camera timestamp."""

    steering: float
    speed: float
    steering_gap_us: int
    speed_gap_us: int


class PreviousCanMessage:
    """Return the most recent CAN message at or before a timestamp."""

    def __init__(self, messages: Sequence[Mapping[str, object]]) -> None:
        self.messages = list(messages)
        self.timestamps = [self._timestamp(message) for message in self.messages]
        if self.timestamps != sorted(self.timestamps):
            raise ValueError("CAN messages must be sorted by utime.")

    @staticmethod
    def _timestamp(message: Mapping[str, object]) -> int:
        timestamp = message.get("utime")
        if not isinstance(timestamp, int):
            raise ValueError("CAN messages must contain integer utime values.")
        return timestamp

    def within(self, timestamp_us: int, max_gap_us: int) -> tuple[Mapping[str, object], int] | None:
        """Return the latest prior message when it is within ``max_gap_us``."""
        if max_gap_us < 0:
            raise ValueError("max_gap_us must be nonnegative.")
        index = bisect_right(self.timestamps, timestamp_us) - 1
        if index < 0:
            return None
        gap_us = timestamp_us - self.timestamps[index]
        if gap_us > max_gap_us:
            return None
        return self.messages[index], gap_us


def align_control(
    timestamp_us: int,
    steering_messages: PreviousCanMessage,
    speed_messages: PreviousCanMessage,
    *,
    max_steering_gap_us: int,
    max_speed_gap_us: int,
) -> AlignedControl | None:
    """Align a camera frame to valid prior steering and speed CAN messages."""
    steering_match = steering_messages.within(timestamp_us, max_steering_gap_us)
    speed_match = speed_messages.within(timestamp_us, max_speed_gap_us)
    if steering_match is None or speed_match is None:
        return None
    steering_message, steering_gap_us = steering_match
    speed_message, speed_gap_us = speed_match
    steering = steering_message.get("value")
    speed = speed_message.get("vehicle_speed")
    if not isinstance(steering, (int, float)) or not isinstance(speed, (int, float)):
        return None
    if not isfinite(steering) or not isfinite(speed):
        return None
    return AlignedControl(
        steering=float(steering),
        speed=float(speed),
        steering_gap_us=steering_gap_us,
        speed_gap_us=speed_gap_us,
    )
