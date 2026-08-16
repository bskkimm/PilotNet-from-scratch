"""Create PilotNet CSV logs from nuScenes front-camera and CAN-bus data."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

from pilotnet.data import PreviousCanMessage, align_control


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataroot", type=Path, default=Path.home() / "dataset/nuscenes")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--version", default="v1.0-trainval")
    parser.add_argument("--max-steering-gap-ms", type=float, default=100.0)
    parser.add_argument("--max-speed-gap-ms", type=float, default=600.0)
    return parser.parse_args()


def camera_records(nusc: object, scene: dict[str, object]) -> list[dict[str, object]]:
    """Follow the complete CAM_FRONT chain, including non-keyframe records."""
    sample = nusc.get("sample", scene["first_sample_token"])
    token = sample["data"]["CAM_FRONT"]
    records = []
    while token:
        record = nusc.get("sample_data", token)
        records.append(record)
        token = record["next"]
    return records


def extract_split(
    nusc: object,
    can_bus: object,
    scene_names: set[str],
    dataroot: Path,
    output_csv: Path,
    max_steering_gap_us: int,
    max_speed_gap_us: int,
) -> dict[str, int]:
    """Write one scene-disjoint CSV split and return its extraction audit."""
    audit: Counter[str] = Counter()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "steering",
                "speed",
                "timestamp",
                "scene_name",
                "location",
                "steering_gap_us",
                "speed_gap_us",
            ],
        )
        writer.writeheader()
        for scene in nusc.scene:
            if scene["name"] not in scene_names:
                continue
            audit["scenes"] += 1
            location = nusc.get("log", scene["log_token"])["location"]
            try:
                steering_messages = can_bus.get_messages(scene["name"], "steeranglefeedback")
                steering = PreviousCanMessage(steering_messages)
                speed = PreviousCanMessage(can_bus.get_messages(scene["name"], "vehicle_monitor"))
            except Exception:
                audit["missing_can_scene"] += 1
                continue
            for record in camera_records(nusc, scene):
                audit["camera_records"] += 1
                image_path = dataroot / record["filename"]
                if not image_path.is_file():
                    audit["missing_image"] += 1
                    continue
                control = align_control(
                    record["timestamp"],
                    steering,
                    speed,
                    max_steering_gap_us=max_steering_gap_us,
                    max_speed_gap_us=max_speed_gap_us,
                )
                if control is None:
                    audit["unaligned_can"] += 1
                    continue
                writer.writerow(
                    {
                        "image_path": os.path.relpath(image_path, output_csv.parent),
                        "steering": control.steering,
                        "speed": control.speed,
                        "timestamp": record["timestamp"],
                        "scene_name": scene["name"],
                        "location": location,
                        "steering_gap_us": control.steering_gap_us,
                        "speed_gap_us": control.speed_gap_us,
                    }
                )
                audit["rows"] += 1
    return dict(audit)


def main() -> None:
    args = parse_args()
    if args.max_steering_gap_ms < 0 or args.max_speed_gap_ms < 0:
        raise ValueError("CAN gap thresholds must be nonnegative.")
    from nuscenes.can_bus.can_bus_api import NuScenesCanBus
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    nusc = NuScenes(version=args.version, dataroot=str(args.dataroot), verbose=False)
    can_bus = NuScenesCanBus(dataroot=str(args.dataroot))
    splits = create_splits_scenes()
    audit = {
        split: extract_split(
            nusc,
            can_bus,
            set(splits[split]),
            args.dataroot,
            args.output_dir / f"{split}.csv",
            round(args.max_steering_gap_ms * 1_000),
            round(args.max_speed_gap_ms * 1_000),
        )
        for split in ("train", "val")
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
