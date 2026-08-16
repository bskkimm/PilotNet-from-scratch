"""nuScenes bird's-eye-view trajectory evaluation."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image

from pilotnet.data import PreprocessConfig, preprocess_image


@dataclass(frozen=True)
class BevArtifacts:
    """Rendered trajectory artifact locations."""

    png: Path
    gif: Path


def rollout_bicycle(
    steering: list[float],
    speed: list[float],
    timestamps_us: list[int],
    *,
    wheelbase: float,
    steering_scale: float,
) -> list[tuple[float, float, float]]:
    """Integrate road-wheel angles and recorded speeds from a local origin."""
    if len(steering) != len(speed) or len(speed) != len(timestamps_us):
        raise ValueError("steering, speed, and timestamps must have equal lengths.")
    if wheelbase <= 0:
        raise ValueError("wheelbase must be positive.")
    trajectory = [(0.0, 0.0, 0.0)]
    x = y = yaw = 0.0
    for index in range(len(steering) - 1):
        dt = (timestamps_us[index + 1] - timestamps_us[index]) / 1_000_000.0
        if dt < 0:
            raise ValueError("timestamps must be nondecreasing.")
        x += speed[index] * math.cos(yaw) * dt
        y += speed[index] * math.sin(yaw) * dt
        yaw += speed[index] * math.tan(steering[index] * steering_scale) * dt / wheelbase
        trajectory.append((x, y, yaw))
    return trajectory


def _rows_for_anchor(
    csv_path: Path,
    scene_name: str | None,
    anchor: int,
    horizon: int,
) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    required = {"image_path", "steering", "speed", "timestamp", "scene_name"}
    if not rows or not required <= set(rows[0]):
        raise ValueError(
            "BEV evaluation CSV needs image_path, steering, speed, timestamp, and scene_name."
        )
    selected_scene = scene_name or rows[0]["scene_name"]
    scene_rows = sorted(
        (row for row in rows if row["scene_name"] == selected_scene),
        key=lambda row: int(row["timestamp"]),
    )
    selected = scene_rows[anchor : anchor + horizon]
    if len(selected) < 2:
        raise ValueError("Selected BEV scene/anchor has fewer than two consecutive frames.")
    return selected


def camera_timestamp_index(
    nusc: object,
    scene: dict[str, object],
) -> dict[int, dict[str, object]]:
    """Index the selected scene's CAM_FRONT records by timestamp."""
    sample = nusc.get("sample", scene["first_sample_token"])
    token = sample["data"]["CAM_FRONT"]
    records = {}
    while token:
        record = nusc.get("sample_data", token)
        records[record["timestamp"]] = record
        token = record["next"]
    return records


def _local_pose(
    position: list[float],
    yaw: float,
    origin: list[float],
    origin_yaw: float,
) -> tuple[float, float, float]:
    dx, dy = position[0] - origin[0], position[1] - origin[1]
    cos_yaw, sin_yaw = math.cos(origin_yaw), math.sin(origin_yaw)
    return (cos_yaw * dx + sin_yaw * dy, -sin_yaw * dx + cos_yaw * dy, yaw - origin_yaw)


def local_to_global(
    x: float,
    y: float,
    origin: list[float],
    origin_yaw: float,
) -> tuple[float, float]:
    """Transform a local ego-frame point into nuScenes global coordinates."""
    cos_yaw, sin_yaw = math.cos(origin_yaw), math.sin(origin_yaw)
    return (origin[0] + cos_yaw * x - sin_yaw * y, origin[1] + sin_yaw * x + cos_yaw * y)


def write_bev_artifacts(
    model: torch.nn.Module,
    *,
    csv_path: str | Path,
    dataroot: str | Path,
    version: str,
    output_dir: str | Path,
    preprocess: PreprocessConfig,
    device: torch.device,
    epoch: int,
    scene_name: str | None = None,
    anchor: int = 0,
    horizon: int = 40,
    wheelbase: float = 2.5,
    steering_scale: float = 1.0,
    gif_frames: int = 8,
) -> BevArtifacts:
    """Render fixed-scene local/global nuScenes maps with predicted and GT paths."""
    try:
        import imageio.v2 as imageio
        import matplotlib.pyplot as plt
        from nuscenes.map_expansion.bitmap import BitMap
        from nuscenes.map_expansion.map_api import NuScenesMap
        from nuscenes.nuscenes import NuScenes
        from pyquaternion import Quaternion
    except ImportError as error:
        message = "Install optional dependencies with `python -m pip install -e '.[nuscenes,bev]'`."
        raise RuntimeError(message) from error

    csv_path = Path(csv_path)
    if gif_frames < 1:
        raise ValueError("gif_frames must be positive.")
    rows = _rows_for_anchor(csv_path, scene_name, anchor, horizon)
    nusc = NuScenes(version=version, dataroot=str(dataroot), verbose=False)
    scene = next(scene for scene in nusc.scene if scene["name"] == rows[0]["scene_name"])
    location = nusc.get("log", scene["log_token"])["location"]
    sample_data = camera_timestamp_index(nusc, scene)
    records = [sample_data[int(row["timestamp"])] for row in rows]
    poses = [nusc.get("ego_pose", record["ego_pose_token"]) for record in records]
    headings = [Quaternion(pose["rotation"]).yaw_pitch_roll[0] for pose in poses]
    origin, origin_yaw = poses[0]["translation"], headings[0]

    model.eval()
    predictions: list[float] = []
    with torch.inference_mode():
        for row in rows:
            with Image.open(csv_path.parent / row["image_path"]) as image_file:
                image = preprocess_image(image_file.convert("RGB"), preprocess)
                image = image.unsqueeze(0).to(device)
            predictions.append(float(model(image).item()))
    predicted = rollout_bicycle(
        predictions,
        [float(row["speed"]) for row in rows],
        [int(row["timestamp"]) for row in rows],
        wheelbase=wheelbase,
        steering_scale=steering_scale,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"bev_epoch_{epoch:03d}.png"
    gif_path = output_dir / f"bev_epoch_{epoch:03d}.gif"
    map_api = NuScenesMap(dataroot=str(dataroot), map_name=location)
    global_points = [(pose["translation"][0], pose["translation"][1]) for pose in poses]
    predicted_global = [local_to_global(x, y, origin, origin_yaw) for x, y, _ in predicted]
    xs, ys = zip(*(global_points + predicted_global))
    margin = 30.0
    basemap = BitMap(str(dataroot), location, "basemap")

    def render_map(box: tuple[float, float, float, float], count: int, title: str) -> Image.Image:
        figure, axes = map_api.render_map_patch(
            box,
            layer_names=[],
            bitmap=basemap,
            render_legend=False,
        )
        axes.plot(*zip(*global_points[:count]), "g.-", label="ground truth")
        axes.plot(*zip(*predicted_global[:count]), "r.-", label="predicted")
        axes.set_title(title)
        axes.set_xlabel("global x (m)")
        axes.set_ylabel("global y (m)")
        axes.legend()
        buffer = BytesIO()
        figure.savefig(buffer, format="png", dpi=100, bbox_inches="tight")
        plt.close(figure)
        return Image.open(buffer).convert("RGB")

    def render(count: int, filename: Path | None = None):
        global_box = (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)
        local_box = (
            origin[0] - margin,
            origin[1] - margin,
            origin[0] + margin,
            origin[1] + margin,
        )
        global_image = render_map(global_box, count, "Global map")
        local_image = render_map(
            local_box,
            count,
            "Anchor-centered map patch (global coordinates)",
        )
        combined = Image.new(
            "RGB",
            (global_image.width + local_image.width, max(global_image.height, local_image.height)),
        )
        combined.paste(global_image, (0, 0))
        combined.paste(local_image, (global_image.width, 0))
        if filename is not None:
            combined.save(filename)
        buffer = BytesIO()
        combined.save(buffer, format="PNG")
        return imageio.imread(buffer.getvalue(), format="png")

    render(len(rows), png_path)
    frame_count = min(gif_frames, len(rows) - 1)
    frame_indices = {
        2 + round(index * (len(rows) - 2) / max(frame_count - 1, 1))
        for index in range(frame_count)
    }
    imageio.mimsave(gif_path, [render(index) for index in sorted(frame_indices)], duration=0.3)
    return BevArtifacts(png=png_path, gif=gif_path)
