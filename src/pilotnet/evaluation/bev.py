"""nuScenes bird's-eye-view trajectory evaluation."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from pilotnet.data import (
    PreprocessConfig,
    TemporalPreprocessConfig,
    preprocess_image,
    preprocess_temporal_image,
)


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


def vehicle_polygon(
    x: float,
    y: float,
    yaw: float,
    *,
    length: float = 4.1,
    width: float = 1.8,
) -> list[tuple[float, float]]:
    """Return a centered, rotated vehicle footprint."""
    half_length, half_width = length / 2.0, width / 2.0
    corners = [
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ]
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return [
        (x + cos_yaw * dx - sin_yaw * dy, y + sin_yaw * dx + cos_yaw * dy)
        for dx, dy in corners
    ]


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
    gif_frames: int = 50,
    gif_frame_duration: float = 0.1,
    sequence_length: int = 1,
    temporal_preprocess: TemporalPreprocessConfig | None = None,
    speed_scale: float = 30.0,
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
    if gif_frame_duration <= 0:
        raise ValueError("gif_frame_duration must be positive.")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive.")
    if speed_scale <= 0:
        raise ValueError("speed_scale must be positive.")
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
    all_scene_rows = _rows_for_anchor(csv_path, scene_name, 0, 1_000_000)
    positions = {int(row["timestamp"]): index for index, row in enumerate(all_scene_rows)}
    with torch.inference_mode():
        for row in rows:
            if temporal_preprocess is None:
                with Image.open(csv_path.parent / row["image_path"]) as image_file:
                    image = preprocess_image(image_file.convert("RGB"), preprocess)
                    image = image.unsqueeze(0).to(device)
                predictions.append(float(model(image).item()))
                continue
            end = positions[int(row["timestamp"])]
            window = all_scene_rows[max(0, end - sequence_length + 1) : end + 1]
            window = [window[0]] * (sequence_length - len(window)) + window
            images = []
            for history_row in window:
                with Image.open(csv_path.parent / history_row["image_path"]) as image_file:
                    images.append(
                        preprocess_temporal_image(image_file.convert("RGB"), temporal_preprocess)
                    )
            speeds = torch.tensor(
                [[float(history_row["speed"]) / speed_scale for history_row in window]],
                dtype=torch.float32,
                device=device,
            )
            sequence = torch.stack(images).unsqueeze(0).to(device)
            predictions.append(float(model(sequence, speeds).item()))
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
    predicted_global = [local_to_global(x, y, origin, origin_yaw) for x, y, _ in predicted]
    margin = 30.0
    basemap = BitMap(str(dataroot), location, "basemap")

    box = (
        origin[0] - margin,
        origin[1] - margin,
        origin[0] + margin,
        origin[1] + margin,
    )
    figure, axes = map_api.render_map_patch(
        box,
        layer_names=[],
        bitmap=basemap,
        render_legend=False,
        render_egoposes_range=False,
    )
    axes.set_title(f"Epoch {epoch}: anchor-centered BEV")
    axes.set_xlabel("global x (m)")
    axes.set_ylabel("global y (m)")
    figure.canvas.draw()
    width, height = figure.canvas.get_width_height()
    background = Image.frombytes(
        "RGBA",
        (width, height),
        figure.canvas.buffer_rgba(),
    ).convert("RGB")
    data_to_pixel = axes.transData.frozen()
    plt.close(figure)

    def render(frame_index: int, filename: Path | None = None) -> Image.Image:
        image = background.copy()
        draw = ImageDraw.Draw(image, "RGBA")
        ground_truth_pose = poses[frame_index]
        predicted_x, predicted_y = predicted_global[frame_index]
        predicted_yaw = origin_yaw + predicted[frame_index][2]
        ground_truth_polygon = vehicle_polygon(
            ground_truth_pose["translation"][0],
            ground_truth_pose["translation"][1],
            headings[frame_index],
        )
        predicted_polygon = vehicle_polygon(predicted_x, predicted_y, predicted_yaw)
        polygons = (
            (ground_truth_polygon, (0, 200, 0, 200)),
            (predicted_polygon, (255, 0, 0, 170)),
        )
        for polygon, color in polygons:
            points = data_to_pixel.transform(polygon)
            draw.polygon([(round(x), round(image.height - y)) for x, y in points], fill=color)
        draw.text((16, 16), "green: ground truth   red: predicted", fill=(0, 0, 0, 255))
        if filename is not None:
            image.save(filename)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return imageio.imread(buffer.getvalue(), format="png")

    render(len(rows) - 1, png_path)
    frame_count = min(gif_frames, len(rows) - 1)
    frame_indices = {
        2 + round(index * (len(rows) - 2) / max(frame_count - 1, 1))
        for index in range(frame_count)
    }
    imageio.mimsave(
        gif_path,
        [render(index - 1) for index in sorted(frame_indices)],
        duration=gif_frame_duration,
    )
    return BevArtifacts(png=png_path, gif=gif_path)
