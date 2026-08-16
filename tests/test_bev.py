"""Tests for the dependency-free BEV trajectory math."""

import math

import pytest

from pilotnet.evaluation.bev import (
    camera_timestamp_index,
    local_to_global,
    rollout_bicycle,
)


def test_rollout_bicycle_integrates_straight_motion_from_timestamps() -> None:
    trajectory = rollout_bicycle(
        steering=[0.0, 0.0, 0.0],
        speed=[2.0, 2.0, 2.0],
        timestamps_us=[0, 1_000_000, 2_000_000],
        wheelbase=2.5,
        steering_scale=1.0,
    )

    assert trajectory == pytest.approx([(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0)])


def test_rollout_bicycle_applies_road_wheel_scale() -> None:
    trajectory = rollout_bicycle(
        steering=[0.1, 0.1],
        speed=[5.0, 5.0],
        timestamps_us=[0, 1_000_000],
        wheelbase=2.5,
        steering_scale=2.0,
    )

    assert trajectory[1][0] == pytest.approx(5.0)
    assert trajectory[1][1] == pytest.approx(0.0)
    assert trajectory[1][2] == pytest.approx(5.0 * math.tan(0.2) / 2.5)


def test_local_to_global_rotates_predicted_coordinates_at_anchor() -> None:
    position = local_to_global(2.0, 0.0, [10.0, 20.0], math.pi / 2)

    assert position == pytest.approx((10.0, 22.0))


def test_camera_timestamp_index_follows_selected_scene_camera_chain() -> None:
    class NuScenes:
        records = {
            ("sample", "first"): {"data": {"CAM_FRONT": "front-1"}},
            (
                "sample_data",
                "front-1",
            ): {"timestamp": 10, "next": "front-2", "ego_pose_token": "pose-1"},
            (
                "sample_data",
                "front-2",
            ): {"timestamp": 20, "next": "", "ego_pose_token": "pose-2"},
        }

        def get(self, table: str, token: str) -> dict[str, object]:
            return self.records[(table, token)]

    index = camera_timestamp_index(NuScenes(), {"first_sample_token": "first"})

    assert index[10]["ego_pose_token"] == "pose-1"
    assert index[20]["ego_pose_token"] == "pose-2"
