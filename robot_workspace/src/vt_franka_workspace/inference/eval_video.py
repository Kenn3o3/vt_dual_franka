from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def write_rollout_video(
    episode_dir: str | Path,
    *,
    stream_name: str,
    output_name: str,
    fps: float,
) -> Path | None:
    episode_path = Path(episode_dir)
    event_path = episode_path / "streams" / f"{stream_name}.jsonl"
    if not event_path.exists():
        return None

    frame_paths = _frame_paths_from_events(episode_path, event_path)
    if not frame_paths:
        return None
    frame_events = _frame_events_from_events(episode_path, event_path)
    action_overlays = _action_overlays_from_policy_steps(episode_path)

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenCV is required to write eval rollout videos") from exc

    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        return None
    height, width = first_frame.shape[:2]
    output_path = episode_path / output_name
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    try:
        _draw_action_overlay(cv2, first_frame, _overlay_for_frame(frame_events[0], action_overlays))
        writer.write(first_frame)
        for index, frame_path in enumerate(frame_paths[1:], start=1):
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            if index < len(frame_events):
                _draw_action_overlay(cv2, frame, _overlay_for_frame(frame_events[index], action_overlays))
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def _frame_paths_from_events(episode_dir: Path, event_path: Path) -> list[Path]:
    frame_paths: list[Path] = []
    seen: set[Path] = set()
    for line in event_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        frame_path = _frame_path_from_event(event)
        if frame_path is None:
            continue
        absolute_path = episode_dir / frame_path
        if absolute_path in seen or not absolute_path.exists():
            continue
        seen.add(absolute_path)
        frame_paths.append(absolute_path)
    return frame_paths


def _frame_events_from_events(episode_dir: Path, event_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for line in event_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        frame_path = _frame_path_from_event(event)
        if frame_path is None:
            continue
        absolute_path = episode_dir / frame_path
        if absolute_path in seen or not absolute_path.exists():
            continue
        seen.add(absolute_path)
        events.append(event)
    return events


def _frame_path_from_event(event: dict[str, Any]) -> Path | None:
    frame_path = event.get("frame_path")
    if isinstance(frame_path, str) and frame_path:
        return Path(frame_path)
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        nested_frame_path = metadata.get("frame_path")
        if isinstance(nested_frame_path, str) and nested_frame_path:
            return Path(nested_frame_path)
    return None


def _action_overlays_from_policy_steps(episode_dir: Path) -> list[dict[str, Any]]:
    path = episode_dir / "streams" / "policy_steps.jsonl"
    if not path.exists():
        return []
    overlays: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("phase") != "policy_chunk":
            continue
        actions = record.get("actions_executed") or []
        observations = record.get("observations_after_actions") or []
        for action_index, action in enumerate(actions):
            timestamp = _action_wall_time(record, observations, action_index)
            if timestamp is None:
                continue
            overlays.append(_overlay_from_action(timestamp, int(record.get("step_index", 0)) + action_index, action))
    return sorted(overlays, key=lambda item: float(item["wall_time"]))


def _action_wall_time(record: dict[str, Any], observations: list[Any], action_index: int) -> float | None:
    if action_index < len(observations):
        observation_item = observations[action_index]
        if isinstance(observation_item, dict):
            observation = observation_item.get("observation") or observation_item
            wall_time = observation.get("assembled_wall_time") if isinstance(observation, dict) else None
            if wall_time is not None:
                return float(wall_time)
            state = ((observation.get("proprioception") or {}).get("controller_state") or {}) if isinstance(observation, dict) else {}
            if state.get("wall_time") is not None:
                return float(state["wall_time"])
    policy_wall_time = record.get("policy_wall_time")
    if policy_wall_time is None:
        return None
    return float(policy_wall_time) + 0.1 * float(action_index)


def _overlay_from_action(wall_time: float, step_index: int, action: dict[str, Any]) -> dict[str, Any]:
    metadata = action.get("metadata") if isinstance(action.get("metadata"), dict) else {}
    raw_state = metadata.get("mpd_tcp_state")
    raw_policy_output = _raw_policy_output_from_state(raw_state)
    raw_gripper = raw_policy_output.get("gripper") if raw_policy_output is not None else None
    if isinstance(raw_state, list) and len(raw_state) >= 10:
        raw_gripper = float(raw_state[9])
    target_tcp = action.get("target_tcp") if isinstance(action.get("target_tcp"), list) else None
    xyz = target_tcp[:3] if target_tcp is not None and len(target_tcp) >= 3 else None
    target_rpy_deg = _rpy_deg_from_pose7d(target_tcp)
    if action.get("gripper_closed") is True:
        gripper_command = "close"
    elif action.get("gripper_width") is not None:
        gripper_command = "open"
    else:
        gripper_command = "none"
    return {
        "wall_time": float(wall_time),
        "step_index": int(step_index),
        "target_xyz": xyz,
        "target_rpy_deg": target_rpy_deg,
        "gripper_command": gripper_command,
        "raw_gripper": raw_gripper,
        "raw_policy_output": raw_policy_output,
        "algorithm": metadata.get("mpd_algorithm"),
        "action_convention": metadata.get("mpd_action_convention"),
    }


def _overlay_for_frame(frame_event: dict[str, Any], overlays: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not overlays:
        return None
    frame_time = _frame_event_time(frame_event)
    if frame_time is None:
        return None
    times = np.asarray([float(item["wall_time"]) for item in overlays], dtype=np.float64)
    index = int(np.argmin(np.abs(times - float(frame_time))))
    return overlays[index]


def _frame_event_time(frame_event: dict[str, Any]) -> float | None:
    for key in ("captured_wall_time", "recorded_at_wall_time"):
        value = frame_event.get(key)
        if value is not None:
            return float(value)
    metadata = frame_event.get("metadata")
    if isinstance(metadata, dict):
        for key in ("captured_wall_time", "recorded_at_wall_time"):
            value = metadata.get(key)
            if value is not None:
                return float(value)
    return None


def _draw_action_overlay(cv2: Any, frame: Any, overlay: dict[str, Any] | None) -> None:
    if overlay is None:
        return
    height, width = frame.shape[:2]
    panel_height = min(height, 160)
    cv2.rectangle(frame, (0, 0), (width, panel_height), (0, 0, 0), thickness=-1)
    cv2.rectangle(frame, (0, 0), (width, panel_height), (255, 255, 255), thickness=1)
    command = str(overlay.get("gripper_command", "none")).upper()
    command_color = (60, 80, 255) if command == "CLOSE" else (80, 220, 80) if command == "OPEN" else (220, 220, 220)
    raw_gripper = overlay.get("raw_gripper")
    raw_text = "raw_gripper=n/a" if raw_gripper is None else f"raw_dim9={float(raw_gripper): .3f}"
    xyz = overlay.get("target_xyz")
    xyz_text = "target_xyz=n/a"
    if isinstance(xyz, list) and len(xyz) >= 3:
        xyz_text = f"target_xyz=[{float(xyz[0]): .3f}, {float(xyz[1]): .3f}, {float(xyz[2]): .3f}]"
    target_rpy = overlay.get("target_rpy_deg")
    target_rpy_text = ""
    if isinstance(target_rpy, list) and len(target_rpy) >= 3:
        target_rpy_text = f" target_rpy_deg=[{float(target_rpy[0]): .1f}, {float(target_rpy[1]): .1f}, {float(target_rpy[2]): .1f}]"
    raw_output = overlay.get("raw_policy_output")
    raw_pose_text = "raw_out=n/a"
    raw_rot_text = ""
    if isinstance(raw_output, dict):
        raw_xyz = raw_output.get("xyz")
        raw_rpy = raw_output.get("rpy_deg")
        raw_rot6d = raw_output.get("rot6d")
        if isinstance(raw_xyz, list) and isinstance(raw_rpy, list) and len(raw_xyz) >= 3 and len(raw_rpy) >= 3:
            raw_pose_text = (
                f"raw_out xyz=[{float(raw_xyz[0]): .3f}, {float(raw_xyz[1]): .3f}, {float(raw_xyz[2]): .3f}] "
                f"rpy_deg=[{float(raw_rpy[0]): .1f}, {float(raw_rpy[1]): .1f}, {float(raw_rpy[2]): .1f}]"
            )
        if isinstance(raw_rot6d, list) and len(raw_rot6d) >= 6:
            raw_rot_text = (
                f"raw_rot6d=[{float(raw_rot6d[0]): .2f}, {float(raw_rot6d[1]): .2f}, {float(raw_rot6d[2]): .2f} | "
                f"{float(raw_rot6d[3]): .2f}, {float(raw_rot6d[4]): .2f}, {float(raw_rot6d[5]): .2f}]"
            )
    convention = overlay.get("action_convention") or ""
    line1 = f"step {int(overlay.get('step_index', 0)):04d}   gripper: {command}   {raw_text}"
    line2 = raw_pose_text
    line3 = f"{xyz_text}{target_rpy_text}"
    line4 = raw_rot_text or str(convention)
    cv2.putText(frame, line1, (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.78, command_color, 2, cv2.LINE_AA)
    cv2.putText(frame, line2[:110], (18, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(frame, line3[:115], (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(frame, line4[:120], (18, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (205, 205, 205), 1, cv2.LINE_AA)
    if convention and raw_rot_text:
        cv2.putText(frame, str(convention)[:120], (18, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1, cv2.LINE_AA)


def _raw_policy_output_from_state(raw_state: Any) -> dict[str, Any] | None:
    if not isinstance(raw_state, list) or len(raw_state) < 10:
        return None
    values = np.asarray(raw_state[:10], dtype=np.float64)
    raw_output: dict[str, Any] = {
        "vector_10d": values.astype(float).tolist(),
        "xyz": values[:3].astype(float).tolist(),
        "rot6d": values[3:9].astype(float).tolist(),
        "gripper": float(values[9]),
    }
    rpy_deg = _rpy_deg_from_rot6d(values[3:9])
    if rpy_deg is not None:
        raw_output["rpy_deg"] = rpy_deg
    return raw_output


def _rpy_deg_from_rot6d(rot6d: Any) -> list[float] | None:
    try:
        values = np.asarray(rot6d, dtype=np.float64)
        if values.shape != (6,):
            return None
        x_raw = values[:3]
        y_raw = values[3:]
        x = _normalize(x_raw)
        z = _normalize(np.cross(x, y_raw))
        y = np.cross(z, x)
        matrix = np.stack([x, y, z], axis=1)
        return _matrix_to_rpy_deg(matrix)
    except (FloatingPointError, ValueError):
        return None


def _rpy_deg_from_pose7d(pose7d: Any) -> list[float] | None:
    if not isinstance(pose7d, list) or len(pose7d) < 7:
        return None
    try:
        w, x, y, z = [float(value) for value in pose7d[3:7]]
        norm = float(np.linalg.norm(np.asarray([w, x, y, z], dtype=np.float64)))
        if norm < 1e-8:
            return None
        w, x, y, z = w / norm, x / norm, y / norm, z / norm
        matrix = np.asarray(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        return _matrix_to_rpy_deg(matrix)
    except (TypeError, ValueError):
        return None


def _matrix_to_rpy_deg(matrix: np.ndarray) -> list[float]:
    # XYZ fixed-axis roll/pitch/yaw convention, matching the usual Franka/operator display.
    sy = float(np.sqrt(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0]))
    singular = sy < 1e-8
    if not singular:
        roll = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
        pitch = float(np.arctan2(-matrix[2, 0], sy))
        yaw = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
    else:
        roll = float(np.arctan2(-matrix[1, 2], matrix[1, 1]))
        pitch = float(np.arctan2(-matrix[2, 0], sy))
        yaw = 0.0
    return np.rad2deg(np.asarray([roll, pitch, yaw], dtype=np.float64)).astype(float).tolist()


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm
