from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vt_dual_franka_shared.models import ArmId, ControllerState

from ..policies.common.visuotactile.bimanual_runtime import ARM_ORDER, bimanual_command_to_20d, bimanual_states_to_20d
from ..recording.raw_recorder import _json_default

BIMANUAL_COMMON_DATASET_SCHEMA_VERSION = "vt_dual_franka_common_dataset_v1"


@dataclass(frozen=True)
class MakeBimanualDatasetConfig:
    collect_task_dir: Path
    dataset_name: str = "real_bimanual_v1"
    output_dir: Path | None = None
    target_hz: float = 10.0
    overwrite: bool = False
    max_state_skew_sec: float = 0.04
    max_action_lead_sec: float | None = None
    max_image_age_sec: float = 0.2
    gripper_open_width_m: float = 0.078


def make_bimanual_common_dataset(config: MakeBimanualDatasetConfig) -> Path:
    collect_task_dir = Path(config.collect_task_dir)
    output_dir = Path(config.output_dir) if config.output_dir is not None else _default_output_dir(collect_task_dir, config.dataset_name)
    if output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_entries: list[dict[str, Any]] = []
    for episode_dir in _episode_dirs(collect_task_dir):
        episode_entries.append(_build_episode(episode_dir, output_dir=output_dir, config=config))
    manifest = {
        "schema_version": BIMANUAL_COMMON_DATASET_SCHEMA_VERSION,
        "created_at_wall_time": time.time(),
        "collect_task_dir": str(collect_task_dir),
        "dataset_name": config.dataset_name,
        "target_hz": float(config.target_hz),
        "arm_order": list(ARM_ORDER),
        "action_provenance": "future_commanded_action",
        "action_dim": 20,
        "qpos_dim": 20,
        "streams": {
            "rgb_wrist_left": {"type": "rgb", "shape": [480, 640, 3]},
            "rgb_wrist_right": {"type": "rgb", "shape": [480, 640, 3]},
            "tactile_left": {"type": "tactile_rgb", "shape": [480, 640, 3]},
            "tactile_right": {"type": "tactile_rgb", "shape": [480, 640, 3]},
        },
        "episodes": episode_entries,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return manifest_path


def _build_episode(episode_dir: Path, *, output_dir: Path, config: MakeBimanualDatasetConfig) -> dict[str, Any]:
    streams_dir = episode_dir / "streams"
    command_records = _read_jsonl(streams_dir / "teleop_commands.jsonl")
    state_records = _read_jsonl(streams_dir / "controller_state_by_arm.jsonl")
    if not command_records:
        raise RuntimeError(f"Episode is missing commanded teleop stream: {episode_dir}")
    if not state_records:
        raise RuntimeError(f"Episode is missing dual controller state stream: {episode_dir}")
    image_streams = {
        stream_name: _read_jsonl(streams_dir / f"{stream_name}.jsonl")
        for stream_name in ("rgb_wrist_left", "rgb_wrist_right", "tactile_left", "tactile_right")
    }
    missing_streams = [name for name, records in image_streams.items() if not records]
    if missing_streams:
        raise RuntimeError(f"Episode is missing bimanual image stream(s) {missing_streams}: {episode_dir}")
    state_times = _times(state_records, "received_wall_time")
    command_times = _times(command_records, "source_wall_time")
    image_times = {
        name: _times(records, "captured_wall_time") for name, records in image_streams.items()
    }
    start = float(state_times[0])
    end = min(float(state_times[-1]), float(command_times[-1]))
    step = 1.0 / float(config.target_hz)
    action_horizon_sec = step if config.max_action_lead_sec is None else float(config.max_action_lead_sec)
    episode_out = output_dir / "episodes" / episode_dir.name
    episode_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for timestamp in np.arange(start, end + step * 0.5, step):
        state_item, state_time = _latest(state_records, state_times, float(timestamp))
        command_item, command_time = _next(command_records, command_times, float(timestamp))
        if state_item is None or command_item is None:
            continue
        if float(timestamp - state_time) > float(config.max_state_skew_sec):
            continue
        action_lead_sec = float(command_time - timestamp)
        if action_lead_sec <= 0.0 or action_lead_sec > action_horizon_sec:
            continue
        image_items: dict[str, tuple[dict[str, Any], float]] = {}
        for stream_name, records in image_streams.items():
            image_item, image_time = _latest(records, image_times[stream_name], float(timestamp))
            if image_item is None or float(timestamp - image_time) > float(config.max_image_age_sec):
                image_items = {}
                break
            image_items[stream_name] = (image_item, image_time)
        if len(image_items) != 4:
            continue
        states = _state_by_arm(state_item)
        commanded = _target_tcp_by_arm(command_item)
        if commanded is None:
            continue
        gripper_closedness = _gripper_closedness_by_arm(command_item)
        step_index = len(rows)
        image_paths = {
            stream_name: _copy_step_image(
                source_episode_dir=episode_dir,
                output_episode_dir=episode_out,
                stream_name=stream_name,
                step_index=step_index,
                record=item,
            )
            for stream_name, (item, _) in image_items.items()
        }
        rows.append(
            {
                "episode_id": episode_dir.name,
                "step_index": step_index,
                "timestamp": float(timestamp),
                "controller_state_by_arm": {arm: states[arm].model_dump(mode="json") for arm in ARM_ORDER},
                "qpos20": bimanual_states_to_20d(states, gripper_open_width_m=config.gripper_open_width_m).astype(float).tolist(),
                "images": image_paths,
                "commanded_actions": {
                    "target_tcp": commanded,
                    "closedness": gripper_closedness,
                    "action20": bimanual_command_to_20d(commanded, gripper_closedness).astype(float).tolist(),
                    "source": command_item,
                },
                "source": {
                    "controller_state": {"timestamp": state_time, "age_sec": float(timestamp - state_time)},
                    "commanded_action": {"timestamp": command_time, "lead_sec": action_lead_sec},
                    "images": {
                        stream_name: {
                            "timestamp": image_time,
                            "age_sec": float(timestamp - image_time),
                        }
                        for stream_name, (_, image_time) in image_items.items()
                    },
                },
            }
        )
    if not rows:
        raise RuntimeError(f"No bimanual samples survived alignment for {episode_dir}")
    steps_path = episode_out / "steps.jsonl"
    _write_jsonl(steps_path, rows)
    return {
        "episode_id": episode_dir.name,
        "num_steps": len(rows),
        "steps_path": steps_path.relative_to(output_dir).as_posix(),
    }


def _state_by_arm(record: dict[str, Any]) -> dict[ArmId, ControllerState]:
    payload = record.get("state_by_arm") or record.get("controller_state_by_arm") or record.get("state")
    if isinstance(payload, dict) and "left" in payload and "right" in payload:
        return {arm: ControllerState.model_validate(payload[arm]) for arm in ARM_ORDER}
    raise KeyError("bimanual controller state must contain left and right")


def _target_tcp_by_arm(record: dict[str, Any]) -> dict[ArmId, list[float]] | None:
    target = record.get("target_tcp")
    if isinstance(target, dict) and all(arm in target for arm in ARM_ORDER):
        return {arm: list(target[arm]) for arm in ARM_ORDER}
    actions = record.get("commanded_actions")
    if isinstance(actions, dict):
        target = actions.get("target_tcp")
        if isinstance(target, dict) and all(arm in target for arm in ARM_ORDER):
            return {arm: list(target[arm]) for arm in ARM_ORDER}
    return None


def _gripper_closedness_by_arm(record: dict[str, Any]) -> dict[ArmId, float]:
    raw = record.get("gripper_closedness") or record.get("closedness")
    if isinstance(raw, dict):
        return {arm: float(raw.get(arm, 0.0)) for arm in ARM_ORDER}
    raw_closed = record.get("gripper_closed")
    if isinstance(raw_closed, dict):
        return {arm: 1.0 if raw_closed.get(arm) else 0.0 for arm in ARM_ORDER}
    return {"left": 0.0, "right": 0.0}


def _copy_step_image(
    *,
    source_episode_dir: Path,
    output_episode_dir: Path,
    stream_name: str,
    step_index: int,
    record: dict[str, Any],
) -> str:
    rel_source = record.get("frame_path")
    if not rel_source:
        raise KeyError(f"{stream_name} record is missing frame_path")
    source = source_episode_dir / str(rel_source)
    if not source.is_file():
        raise FileNotFoundError(source)
    suffix = source.suffix.lower() or ".jpg"
    destination = output_episode_dir / "images" / stream_name / f"{step_index:06d}{suffix}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    dataset_root = output_episode_dir.parent.parent
    return destination.relative_to(dataset_root).as_posix()


def _default_output_dir(collect_task_dir: Path, dataset_name: str) -> Path:
    repo_data = collect_task_dir.parents[1] if collect_task_dir.parent.name == "collect" else collect_task_dir.parent
    return repo_data / "datasets" / collect_task_dir.name / dataset_name


def _episode_dirs(collect_task_dir: Path) -> list[Path]:
    episodes = collect_task_dir / "episodes"
    return sorted(path for path in episodes.glob("episode_*") if path.is_dir()) if episodes.exists() else []


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default))
            handle.write("\n")


def _times(records: list[dict[str, Any]], preferred_key: str) -> np.ndarray:
    return np.asarray([_timestamp(record, preferred_key) for record in records], dtype=np.float64)


def _timestamp(record: dict[str, Any], preferred_key: str) -> float:
    for key in (preferred_key, "source_wall_time", "received_wall_time", "recorded_at_wall_time"):
        if record.get(key) is not None:
            return float(record[key])
    return float(record.get("timestamp", 0.0))


def _latest(records: list[dict[str, Any]], timestamps: np.ndarray, timestamp: float) -> tuple[dict[str, Any] | None, float]:
    index = int(np.searchsorted(timestamps, timestamp, side="right") - 1)
    if index < 0:
        return None, float("nan")
    return records[index], float(timestamps[index])


def _next(records: list[dict[str, Any]], timestamps: np.ndarray, timestamp: float) -> tuple[dict[str, Any] | None, float]:
    index = int(np.searchsorted(timestamps, timestamp, side="right"))
    if index >= len(records):
        return None, float("nan")
    return records[index], float(timestamps[index])
