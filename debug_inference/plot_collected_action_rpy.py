#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_DIR = REPO_ROOT / "robot_workspace/data/datasets/pencil_insertion/real_pencil_insertion"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "debug_inference"
AXES = ("roll", "pitch", "yaw")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot collected action command roll/pitch/yaw labels per episode."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument(
        "--absolute-time",
        action="store_true",
        help="Use wall-clock timestamps on the x axis instead of seconds since episode start.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.max_episodes <= 0:
        raise ValueError("--max-episodes must be positive")

    dataset_dir = args.dataset_dir.resolve()
    output_root = args.output_root.resolve()
    episodes = load_manifest_episodes(dataset_dir)[: args.max_episodes]
    if len(episodes) < args.max_episodes:
        raise ValueError(
            f"Dataset contains only {len(episodes)} episodes, requested {args.max_episodes}."
        )

    for axis in AXES:
        (output_root / axis).mkdir(parents=True, exist_ok=True)

    written: dict[str, list[str]] = {axis: [] for axis in AXES}
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        steps_path = dataset_dir / str(episode["steps_path"])
        timestamps, rpy_deg = load_action_rpy(steps_path, absolute_time=args.absolute_time)
        for axis_index, axis in enumerate(AXES):
            output_path = output_root / axis / f"{episode_id}_{axis}_action_command.png"
            plot_axis(
                output_path=output_path,
                episode_id=episode_id,
                axis=axis,
                timestamps=timestamps,
                values=rpy_deg[:, axis_index],
                absolute_time=args.absolute_time,
            )
            written[axis].append(str(output_path))

    print(
        json.dumps(
            {
                "dataset_dir": str(dataset_dir),
                "output_root": str(output_root),
                "episodes": len(episodes),
                "plots_per_axis": {axis: len(paths) for axis, paths in written.items()},
            },
            indent=2,
        )
    )


def load_manifest_episodes(dataset_dir: Path) -> list[dict[str, Any]]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list):
        raise ValueError(f"{manifest_path} is missing an episodes list")
    return episodes


def load_action_rpy(steps_path: Path, *, absolute_time: bool) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    rpy_values: list[np.ndarray] = []
    with steps_path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            step = json.loads(line)
            target_tcp = step.get("action", {}).get("target_tcp")
            if target_tcp is None:
                continue
            values = np.asarray(target_tcp, dtype=np.float64)
            if values.shape != (7,):
                raise ValueError(f"{steps_path}:{line_number} has bad target_tcp shape {values.shape}")
            timestamps.append(float(step["timestamp"]))
            quat_wxyz = values[3:7]
            quat_xyzw = np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            rpy_values.append(Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True))

    if not timestamps:
        raise ValueError(f"{steps_path} has no action.target_tcp labels")
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    if not absolute_time:
        timestamp_array = timestamp_array - timestamp_array[0]
    return timestamp_array, np.stack(rpy_values, axis=0)


def plot_axis(
    *,
    output_path: Path,
    episode_id: str,
    axis: str,
    timestamps: np.ndarray,
    values: np.ndarray,
    absolute_time: bool,
) -> None:
    width, height = 1120, 630
    left, right, top, bottom = 110, 35, 70, 95
    plot_w = width - left - right
    plot_h = height - top - bottom
    image = np.full((height, width, 3), 255, dtype=np.uint8)

    x_min = float(np.min(timestamps))
    x_max = float(np.max(timestamps))
    if abs(x_max - x_min) < 1e-9:
        x_max = x_min + 1.0

    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1.0
        y_max += 1.0
    else:
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad

    def to_pixel(x_value: float, y_value: float) -> tuple[int, int]:
        x_ratio = (x_value - x_min) / (x_max - x_min)
        y_ratio = (y_value - y_min) / (y_max - y_min)
        px = int(round(left + x_ratio * plot_w))
        py = int(round(top + (1.0 - y_ratio) * plot_h))
        return px, py

    grid_color = (225, 225, 225)
    axis_color = (40, 40, 40)
    line_color = (200, 70, 30)
    point_color = (80, 80, 80)
    text_color = (30, 30, 30)

    for tick in np.linspace(x_min, x_max, 6):
        px, _ = to_pixel(float(tick), y_min)
        cv2.line(image, (px, top), (px, top + plot_h), grid_color, 1)
        label = f"{tick:.1f}" if not absolute_time else f"{tick:.3f}"
        cv2.putText(image, label, (px - 35, top + plot_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1)

    for tick in np.linspace(y_min, y_max, 6):
        _, py = to_pixel(x_min, float(tick))
        cv2.line(image, (left, py), (left + plot_w, py), grid_color, 1)
        cv2.putText(image, f"{tick:.2f}", (18, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, text_color, 1)

    cv2.rectangle(image, (left, top), (left + plot_w, top + plot_h), axis_color, 2)

    points = [to_pixel(float(x), float(y)) for x, y in zip(timestamps, values)]
    if len(points) >= 2:
        cv2.polylines(image, [np.asarray(points, dtype=np.int32)], isClosed=False, color=line_color, thickness=3)
    for point in points:
        cv2.circle(image, point, 3, point_color, -1)

    title = f"{episode_id} collected {axis} action command"
    x_label = "timestamp (wall time sec)" if absolute_time else "timestamp (sec since episode start)"
    y_label = f"{axis} action command (deg)"
    cv2.putText(image, title, (left, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.86, text_color, 2)
    cv2.putText(image, x_label, (left + plot_w // 2 - 170, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, text_color, 2)
    cv2.putText(image, y_label, (left, top - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.62, text_color, 1)

    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to write {output_path}")


if __name__ == "__main__":
    main()
