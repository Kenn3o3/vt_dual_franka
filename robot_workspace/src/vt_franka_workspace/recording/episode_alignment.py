from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def align_episode(
    episode_dir: str | Path,
    *,
    target_hz: float = 10.0,
    max_action_lead_sec: float | None = None,
    overwrite: bool = False,
) -> Path:
    episode_dir = Path(episode_dir)
    output_path = episode_dir / "aligned_episode.npz"
    manifest_path = episode_dir / "aligned_episode_manifest.json"
    if output_path.exists() and not overwrite:
        return output_path

    streams_dir = episode_dir / "streams"
    controller_records = _read_jsonl(streams_dir / "controller_state.jsonl")
    teleop_records = [record for record in _read_jsonl(streams_dir / "teleop_commands.jsonl") if record.get("target_tcp") is not None]
    gelsight_frame_records = _read_jsonl(streams_dir / "gelsight_frames.jsonl")
    rgb_streams = _discover_rgb_streams(streams_dir)

    if not controller_records:
        raise RuntimeError(f"No controller_state stream found in {episode_dir}")
    if not teleop_records:
        raise RuntimeError(f"No teleop_commands stream with target_tcp found in {episode_dir}")

    target_hz = float(target_hz)
    if target_hz <= 0.0:
        raise ValueError("target_hz must be positive")
    step = 1.0 / target_hz
    action_horizon_sec = step if max_action_lead_sec is None else float(max_action_lead_sec)
    if action_horizon_sec <= 0.0:
        raise ValueError("max_action_lead_sec must be positive")

    controller_times = _timestamp_array(controller_records, "received_wall_time")
    teleop_times = _timestamp_array(teleop_records, "source_wall_time")
    gelsight_frame_times = _timestamp_array(gelsight_frame_records, "captured_wall_time")
    rgb_times = {name: _timestamp_array(records, "captured_wall_time") for name, records in rgb_streams.items()}

    start_time = max(float(controller_times[0]), float(teleop_times[0]))
    end_time = min(float(controller_times[-1]), float(teleop_times[-1]))
    if end_time <= start_time:
        raise RuntimeError(f"Controller and teleop streams do not overlap in {episode_dir}")
    grid = np.arange(start_time, end_time + step * 0.5, step, dtype=np.float64)

    aligned_timestamps: list[float] = []
    tcp_pose: list[list[float]] = []
    tcp_velocity: list[list[float]] = []
    tcp_wrench: list[list[float]] = []
    joint_positions: list[list[float]] = []
    joint_velocities: list[list[float]] = []
    gripper_width: list[float] = []
    gripper_force: list[float] = []
    controller_age_sec: list[float] = []
    controller_source_timestamps: list[float] = []
    teleop_target_tcp: list[list[float]] = []
    teleop_gripper_closed: list[bool] = []
    teleop_source_timestamps: list[float] = []
    teleop_action_lead_sec: list[float] = []
    gelsight_frame_paths: list[str] = []
    gelsight_frame_indices: list[int] = []
    gelsight_capture_timestamps: list[float] = []
    rgb_frame_paths: dict[str, list[str]] = {name: [] for name in rgb_streams}
    rgb_frame_indices: dict[str, list[int]] = {name: [] for name in rgb_streams}
    rgb_capture_timestamps: dict[str, list[float]] = {name: [] for name in rgb_streams}
    dropped_without_future_action = 0
    dropped_action_outside_horizon = 0

    for timestamp in grid:
        controller_item, controller_time = _latest_record(controller_records, controller_times, timestamp)
        teleop_item, teleop_time = _next_record(teleop_records, teleop_times, timestamp)
        if controller_item is None:
            continue
        if teleop_item is None:
            dropped_without_future_action += 1
            continue
        action_lead_sec = teleop_time - timestamp
        if action_lead_sec <= 0.0:
            dropped_without_future_action += 1
            continue
        if action_lead_sec > action_horizon_sec:
            dropped_action_outside_horizon += 1
            continue

        state = controller_item.get("state", controller_item)
        aligned_timestamps.append(float(timestamp))
        tcp_pose.append(_as_float_list(state.get("tcp_pose"), 7, "tcp_pose"))
        tcp_velocity.append(_as_float_list(state.get("tcp_velocity", [0.0] * 6), 6, "tcp_velocity"))
        tcp_wrench.append(_as_float_list(state.get("tcp_wrench", [0.0] * 6), 6, "tcp_wrench"))
        joint_positions.append(_as_float_list(state.get("joint_positions", [0.0] * 7), 7, "joint_positions"))
        joint_velocities.append(_as_float_list(state.get("joint_velocities", [0.0] * 7), 7, "joint_velocities"))
        gripper_width.append(float(state.get("gripper_width", 0.0)))
        gripper_force.append(float(state.get("gripper_force", 0.0)))
        controller_age_sec.append(float(timestamp - controller_time))
        controller_source_timestamps.append(controller_time)
        teleop_target_tcp.append(_as_float_list(teleop_item.get("target_tcp"), 7, "target_tcp"))
        teleop_gripper_closed.append(bool(teleop_item.get("gripper_closed", False)))
        teleop_source_timestamps.append(teleop_time)
        teleop_action_lead_sec.append(float(action_lead_sec))

        gelsight_item, gelsight_time = _latest_record(gelsight_frame_records, gelsight_frame_times, timestamp)
        if gelsight_item is None:
            gelsight_frame_paths.append("")
            gelsight_frame_indices.append(-1)
            gelsight_capture_timestamps.append(np.nan)
        else:
            gelsight_frame_paths.append(str(gelsight_item.get("frame_path", "")))
            gelsight_frame_indices.append(int(gelsight_item.get("index_in_chunk", -1)))
            gelsight_capture_timestamps.append(gelsight_time)

        for stream_name, records in rgb_streams.items():
            rgb_item, rgb_time = _latest_record(records, rgb_times[stream_name], timestamp)
            if rgb_item is None:
                rgb_frame_paths[stream_name].append("")
                rgb_frame_indices[stream_name].append(-1)
                rgb_capture_timestamps[stream_name].append(np.nan)
            else:
                rgb_frame_paths[stream_name].append(str(rgb_item.get("frame_path", "")))
                rgb_frame_indices[stream_name].append(int(rgb_item.get("index_in_chunk", -1)))
                rgb_capture_timestamps[stream_name].append(rgb_time)

    if not aligned_timestamps:
        raise RuntimeError(f"No aligned samples survived causal alignment in {episode_dir}")

    rgb_arrays: dict[str, np.ndarray] = {}
    for stream_name in rgb_streams:
        rgb_arrays[f"{stream_name}_frame_paths"] = np.asarray(rgb_frame_paths[stream_name], dtype=object)
        rgb_arrays[f"{stream_name}_frame_indices"] = np.asarray(rgb_frame_indices[stream_name], dtype=np.int64)
        rgb_arrays[f"{stream_name}_capture_timestamps"] = np.asarray(rgb_capture_timestamps[stream_name], dtype=np.float64)

    np.savez_compressed(
        output_path,
        timestamps=np.asarray(aligned_timestamps, dtype=np.float64),
        robot_tcp_pose=_float_matrix(tcp_pose, 7),
        robot_tcp_velocity=_float_matrix(tcp_velocity, 6),
        robot_tcp_wrench=_float_matrix(tcp_wrench, 6),
        robot_joint_positions=_float_matrix(joint_positions, 7),
        robot_joint_velocities=_float_matrix(joint_velocities, 7),
        gripper_width=np.asarray(gripper_width, dtype=np.float64),
        gripper_force=np.asarray(gripper_force, dtype=np.float64),
        gripper_state=np.asarray(list(zip(gripper_width, gripper_force)), dtype=np.float64),
        controller_state_valid=np.ones((len(aligned_timestamps),), dtype=bool),
        controller_state_age_sec=np.asarray(controller_age_sec, dtype=np.float64),
        controller_state_source_timestamps=np.asarray(controller_source_timestamps, dtype=np.float64),
        teleop_target_tcp=_float_matrix(teleop_target_tcp, 7),
        teleop_gripper_closed=np.asarray(teleop_gripper_closed, dtype=bool),
        teleop_command_source_timestamps=np.asarray(teleop_source_timestamps, dtype=np.float64),
        teleop_action_lead_sec=np.asarray(teleop_action_lead_sec, dtype=np.float64),
        gelsight_frame_paths=np.asarray(gelsight_frame_paths, dtype=object),
        gelsight_frame_indices=np.asarray(gelsight_frame_indices, dtype=np.int64),
        gelsight_capture_timestamps=np.asarray(gelsight_capture_timestamps, dtype=np.float64),
        **rgb_arrays,
    )

    manifest = {
        "schema_version": "vt_franka_aligned_episode_v2",
        "target_hz": target_hz,
        "num_steps": int(len(aligned_timestamps)),
        "grid_steps": int(len(grid)),
        "alignment_mode": "causal_observation_future_action",
        "controller_alignment_timestamp": "received_wall_time",
        "teleop_alignment_timestamp": "source_wall_time",
        "action_horizon_sec": action_horizon_sec,
        "dropped_steps_without_future_action": int(dropped_without_future_action),
        "dropped_steps_action_outside_horizon": int(dropped_action_outside_horizon),
        "streams_used": _streams_used(controller_records, teleop_records, gelsight_frame_records, rgb_streams),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return output_path


def collect_episode_dirs(paths: list[str | Path]) -> list[Path]:
    episode_dirs: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.name.startswith("episode_") and path.is_dir():
            episode_dirs.append(path)
            continue
        episodes_dir = path / "episodes"
        if episodes_dir.exists():
            episode_dirs.extend(sorted(item for item in episodes_dir.glob("episode_*") if item.is_dir()))
            continue
        if path.is_dir():
            episode_dirs.extend(sorted(item for item in path.glob("episode_*") if item.is_dir()))
            continue
        raise FileNotFoundError(f"Cannot resolve episode path: {path}")
    return sorted(dict.fromkeys(episode_dirs))


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


def _timestamp_array(records: list[dict[str, Any]], preferred_key: str) -> np.ndarray:
    return np.asarray([_timestamp_value(record, preferred_key) for record in records], dtype=np.float64)


def _timestamp_value(record: dict[str, Any], preferred_key: str) -> float:
    for key in (preferred_key, "source_wall_time", "captured_wall_time", "recorded_at_wall_time"):
        value = record.get(key)
        if value is not None:
            return float(value)
    raise KeyError(f"Record is missing timestamp key {preferred_key}: {record.keys()}")


def _latest_record(records: list[dict[str, Any]], times: np.ndarray, timestamp: float) -> tuple[dict[str, Any] | None, float]:
    if len(records) == 0:
        return None, np.nan
    index = int(np.searchsorted(times, timestamp, side="right") - 1)
    if index < 0:
        return None, np.nan
    return records[index], float(times[index])


def _next_record(records: list[dict[str, Any]], times: np.ndarray, timestamp: float) -> tuple[dict[str, Any] | None, float]:
    if len(records) == 0:
        return None, np.nan
    index = int(np.searchsorted(times, timestamp, side="right"))
    if index >= len(records):
        return None, np.nan
    return records[index], float(times[index])


def _discover_rgb_streams(streams_dir: Path) -> dict[str, list[dict[str, Any]]]:
    streams: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(streams_dir.glob("*.jsonl")):
        stream_name = path.stem
        if stream_name in {"controller_state", "teleop_commands", "quest_messages", "gelsight_frames"}:
            continue
        records = _read_jsonl(path)
        if records and "frame_path" in records[0] and "captured_wall_time" in records[0]:
            streams[stream_name] = records
    return streams


def _streams_used(
    controller_records: list[dict[str, Any]],
    teleop_records: list[dict[str, Any]],
    gelsight_frame_records: list[dict[str, Any]],
    rgb_streams: dict[str, list[dict[str, Any]]],
) -> list[str]:
    streams = []
    if controller_records:
        streams.append("controller_state")
    if teleop_records:
        streams.append("teleop_commands")
    if gelsight_frame_records:
        streams.append("gelsight_frames")
    streams.extend(sorted(rgb_streams))
    return streams


def _as_float_list(value: Any, length: int, name: str) -> list[float]:
    if value is None:
        raise ValueError(f"Missing required value {name}")
    items = [float(item) for item in value]
    if len(items) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(items)}")
    return items


def _float_matrix(rows: list[list[float]], width: int) -> np.ndarray:
    if not rows:
        return np.empty((0, width), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)
