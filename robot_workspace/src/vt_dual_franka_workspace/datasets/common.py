from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..recording.raw_recorder import _json_default


COMMON_DATASET_SCHEMA_VERSION = "vt_franka_common_dataset_v1"


@dataclass(frozen=True)
class MakeDatasetConfig:
    collect_task_dir: Path
    output_dir: Path | None = None
    dataset_name: str = "real_640x480_v1"
    target_hz: float = 10.0
    overwrite: bool = False
    max_action_lead_sec: float | None = None


@dataclass(frozen=True)
class MakeDatasetResult:
    output_dir: Path
    manifest_path: Path
    episode_count: int
    step_count: int


def make_common_dataset(config: MakeDatasetConfig) -> MakeDatasetResult:
    collect_task_dir = Path(config.collect_task_dir)
    if not collect_task_dir.exists():
        raise FileNotFoundError(f"Missing collect task directory: {collect_task_dir}")
    output_dir = Path(config.output_dir) if config.output_dir is not None else _default_output_dir(collect_task_dir, config.dataset_name)
    if output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"Dataset output already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_dirs = _collect_complete_episode_dirs(collect_task_dir)
    if not episode_dirs:
        raise FileNotFoundError(f"No saved complete episodes found under {collect_task_dir}")

    dataset_entries: list[dict[str, Any]] = []
    total_steps = 0
    for episode_dir in episode_dirs:
        episode_result = _build_episode_dataset(
            episode_dir,
            output_dir=output_dir,
            target_hz=float(config.target_hz),
            max_action_lead_sec=config.max_action_lead_sec,
        )
        dataset_entries.append(episode_result)
        total_steps += int(episode_result["num_steps"])

    manifest = {
        "schema_version": COMMON_DATASET_SCHEMA_VERSION,
        "created_at_wall_time": time.time(),
        "collect_task_dir": str(collect_task_dir),
        "output_dir": str(output_dir),
        "dataset_name": config.dataset_name,
        "target_hz": float(config.target_hz),
        "alignment_mode": "causal_observation_future_action",
        "camera_standardization": "RGB uint8 640x480",
        "standardized_image_jpeg_quality": 90,
        "streams": {
            "rgb_wrist": {"type": "rgb", "shape": [480, 640, 3], "color_space": "RGB"},
            "tactile_left": {"type": "tactile_rgb", "shape": [480, 640, 3], "color_space": "RGB"},
        },
        "episodes": dataset_entries,
        "episode_count": len(dataset_entries),
        "total_steps": total_steps,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return MakeDatasetResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        episode_count=len(dataset_entries),
        step_count=total_steps,
    )


def _build_episode_dataset(
    episode_dir: Path,
    *,
    output_dir: Path,
    target_hz: float,
    max_action_lead_sec: float | None,
) -> dict[str, Any]:
    streams_dir = episode_dir / "streams"
    rgb_records = _read_image_index(streams_dir, "rgb_wrist")
    tactile_records = _read_image_index(streams_dir, "tactile_left")
    controller_records = _read_jsonl(streams_dir / "controller_state.jsonl")
    teleop_records = [record for record in _read_jsonl(streams_dir / "teleop_commands.jsonl") if record.get("target_tcp") is not None]
    if not rgb_records:
        raise RuntimeError(f"Episode is missing rgb_wrist image stream: {episode_dir}")
    if not tactile_records:
        raise RuntimeError(f"Episode is missing tactile_left image stream: {episode_dir}")
    if not controller_records:
        raise RuntimeError(f"Episode is missing controller_state stream: {episode_dir}")
    if not teleop_records:
        raise RuntimeError(f"Episode is missing teleop command stream with target_tcp: {episode_dir}")

    rgb_times = _timestamp_array(rgb_records, "captured_wall_time")
    tactile_times = _timestamp_array(tactile_records, "captured_wall_time")
    controller_times = _timestamp_array(controller_records, "received_wall_time")
    teleop_times = _timestamp_array(teleop_records, "source_wall_time")
    start_time = max(float(rgb_times[0]), float(tactile_times[0]), float(controller_times[0]))
    end_time = min(float(rgb_times[-1]), float(tactile_times[-1]), float(controller_times[-1]), float(teleop_times[-1]))
    if end_time <= start_time:
        raise RuntimeError(f"Episode streams do not overlap: {episode_dir}")

    step = 1.0 / float(target_hz)
    action_horizon_sec = step if max_action_lead_sec is None else float(max_action_lead_sec)
    grid = np.arange(start_time, end_time + step * 0.5, step, dtype=np.float64)
    episode_out = output_dir / "episodes" / episode_dir.name
    rgb_out = episode_out / "images" / "rgb_wrist"
    tactile_out = episode_out / "images" / "tactile_left"
    rgb_out.mkdir(parents=True, exist_ok=True)
    tactile_out.mkdir(parents=True, exist_ok=True)

    step_records: list[dict[str, Any]] = []
    dropped_no_action = 0
    dropped_action_horizon = 0
    for timestamp in grid:
        rgb_item, rgb_time = _latest_record(rgb_records, rgb_times, timestamp)
        tactile_item, tactile_time = _latest_record(tactile_records, tactile_times, timestamp)
        controller_item, controller_time = _latest_record(controller_records, controller_times, timestamp)
        teleop_item, teleop_time = _next_record(teleop_records, teleop_times, timestamp)
        if rgb_item is None or tactile_item is None or controller_item is None:
            continue
        if teleop_item is None:
            dropped_no_action += 1
            continue
        action_lead_sec = float(teleop_time - timestamp)
        if action_lead_sec <= 0.0:
            dropped_no_action += 1
            continue
        if action_lead_sec > action_horizon_sec:
            dropped_action_horizon += 1
            continue

        step_index = len(step_records)
        rgb_rel = _copy_selected_image(episode_dir, rgb_item, rgb_out / f"{step_index:06d}.jpg", output_dir)
        tactile_rel = _copy_selected_image(episode_dir, tactile_item, tactile_out / f"{step_index:06d}.jpg", output_dir)
        state = controller_item.get("state", controller_item)
        step_records.append(
            {
                "episode_id": episode_dir.name,
                "step_index": step_index,
                "timestamp": float(timestamp),
                "images": {
                    "rgb_wrist": rgb_rel,
                    "tactile_left": tactile_rel,
                },
                "source": {
                    "rgb_wrist": _source_record(rgb_item, rgb_time),
                    "tactile_left": _source_record(tactile_item, tactile_time),
                    "controller_state": {"timestamp": controller_time, "age_sec": float(timestamp - controller_time)},
                    "teleop_command": {"timestamp": teleop_time, "lead_sec": action_lead_sec},
                },
                "controller_state": state,
                "action": {
                    "target_tcp": teleop_item.get("target_tcp"),
                    "gripper_closed": bool(teleop_item.get("gripper_closed", False)),
                    "gripper_width": teleop_item.get("gripper_width"),
                    "source": teleop_item,
                },
            }
        )

    if not step_records:
        raise RuntimeError(f"No common dataset samples survived alignment for {episode_dir}")

    steps_path = episode_out / "steps.jsonl"
    _write_jsonl(steps_path, step_records)
    episode_manifest = {
        "schema_version": "vt_franka_common_dataset_episode_v1",
        "source_episode_dir": str(episode_dir),
        "episode_id": episode_dir.name,
        "num_steps": len(step_records),
        "target_hz": target_hz,
        "grid_steps": int(len(grid)),
        "dropped_steps_without_future_action": dropped_no_action,
        "dropped_steps_action_outside_horizon": dropped_action_horizon,
        "steps_path": steps_path.relative_to(output_dir).as_posix(),
    }
    (episode_out / "episode_manifest.json").write_text(json.dumps(episode_manifest, indent=2, default=_json_default), encoding="utf-8")
    return {
        "episode_id": episode_dir.name,
        "source_episode_dir": str(episode_dir),
        "num_steps": len(step_records),
        "manifest_path": (episode_out / "episode_manifest.json").relative_to(output_dir).as_posix(),
        "steps_path": steps_path.relative_to(output_dir).as_posix(),
    }


def _default_output_dir(collect_task_dir: Path, dataset_name: str) -> Path:
    repo_data = collect_task_dir.parents[1] if collect_task_dir.parent.name == "collect" else collect_task_dir.parent
    return repo_data / "datasets" / collect_task_dir.name / dataset_name


def _collect_complete_episode_dirs(collect_task_dir: Path) -> list[Path]:
    episodes_dir = collect_task_dir / "episodes"
    candidates = sorted(path for path in episodes_dir.glob("episode_*") if path.is_dir()) if episodes_dir.exists() else []
    complete = []
    for episode_dir in candidates:
        manifest_path = episode_dir / "episode_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("outcome") == "saved":
            complete.append(episode_dir)
    return complete


def _read_image_index(streams_dir: Path, stream_name: str) -> list[dict[str, Any]]:
    index_path = streams_dir / stream_name / "index.jsonl"
    legacy_path = streams_dir / f"{stream_name}.jsonl"
    if index_path.exists():
        return _read_jsonl(index_path)
    return _read_jsonl(legacy_path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default))
            handle.write("\n")


def _timestamp_array(records: list[dict[str, Any]], preferred_key: str) -> np.ndarray:
    return np.asarray([_timestamp_value(record, preferred_key) for record in records], dtype=np.float64)


def _timestamp_value(record: dict[str, Any], preferred_key: str) -> float:
    for key in (preferred_key, "source_wall_time", "captured_wall_time", "received_wall_time", "recorded_at_wall_time"):
        value = record.get(key)
        if value is not None:
            return float(value)
    metadata = record.get("metadata")
    if isinstance(metadata, dict):
        value = metadata.get(preferred_key) or metadata.get("captured_wall_time")
        if value is not None:
            return float(value)
    raise KeyError(f"Record is missing timestamp key {preferred_key}: {record.keys()}")


def _latest_record(records: list[dict[str, Any]], timestamps: np.ndarray, timestamp: float) -> tuple[dict[str, Any] | None, float]:
    index = int(np.searchsorted(timestamps, timestamp, side="right") - 1)
    if index < 0:
        return None, float("nan")
    return records[index], float(timestamps[index])


def _next_record(records: list[dict[str, Any]], timestamps: np.ndarray, timestamp: float) -> tuple[dict[str, Any] | None, float]:
    index = int(np.searchsorted(timestamps, timestamp, side="right"))
    if index >= len(records):
        return None, float("nan")
    return records[index], float(timestamps[index])


def _copy_selected_image(source_episode_dir: Path, record: dict[str, Any], output_path: Path, dataset_root: Path) -> str:
    rel_path = record.get("frame_path")
    if not rel_path:
        raise RuntimeError(f"Image index record is missing frame_path: {record}")
    source_path = source_episode_dir / str(rel_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Missing source image: {source_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, output_path)
    return output_path.relative_to(dataset_root).as_posix()


def _source_record(record: dict[str, Any], timestamp: float) -> dict[str, Any]:
    return {
        "timestamp": timestamp,
        "frame_path": record.get("frame_path"),
        "frame_index": record.get("frame_index"),
        "sequence_id": record.get("sequence_id"),
        "metadata": record.get("metadata", {}),
    }
