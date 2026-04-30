from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...config import WorkspaceSettings
from .config import DEFAULT_DATASET_NAME, default_prepared_dataset_dir
from .math import finite_difference, gripper_width_to_closedness, pose7d_and_gripper_to_tcp_state


@dataclass(frozen=True)
class PrepareMPDDatasetConfig:
    task_name: str
    raw_run_dir: Path
    output_dir: Path
    dataset_name: str = DEFAULT_DATASET_NAME
    target_hz: float = 10.0
    val_ratio: float = 0.2
    val_episodes: int | None = None
    max_state_age_sec: float = 0.25
    max_command_future_sec: float = 0.25
    min_steps: int = 15
    gripper_open_width_m: float = 0.078
    overwrite: bool = False

    @property
    def dt(self) -> float:
        return 1.0 / max(float(self.target_hz), 1e-8)


@dataclass(frozen=True)
class PreparedDatasetResult:
    output_dir: Path
    train_episodes: int
    val_episodes: int
    total_steps: int
    manifest_path: Path


@dataclass(frozen=True)
class EpisodeStreams:
    episode_dir: Path
    controller_states: list[dict[str, Any]]
    teleop_commands: list[dict[str, Any]]


def build_prepare_config_from_workspace(
    workspace: WorkspaceSettings,
    *,
    task_name: str,
    raw_run_dir: Path | None = None,
    output_dir: Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    target_hz: float = 10.0,
    val_ratio: float = 0.2,
    val_episodes: int | None = None,
    overwrite: bool = False,
) -> PrepareMPDDatasetConfig:
    return PrepareMPDDatasetConfig(
        task_name=task_name,
        raw_run_dir=raw_run_dir or (Path(workspace.recording.collect_root) / task_name),
        output_dir=output_dir or default_prepared_dataset_dir(workspace, task_name, dataset_name),
        dataset_name=dataset_name,
        target_hz=target_hz,
        val_ratio=val_ratio,
        val_episodes=val_episodes,
        gripper_open_width_m=workspace.teleop.max_gripper_width,
        overwrite=overwrite,
    )


def prepare_mpd_dataset(config: PrepareMPDDatasetConfig) -> PreparedDatasetResult:
    episodes = _list_episode_streams(config.raw_run_dir)
    if len(episodes) < 2:
        raise ValueError("MPD fixed-split training requires at least two collected episodes")
    val_count = _resolve_val_count(len(episodes), config.val_ratio, config.val_episodes)
    train_episodes = episodes[: len(episodes) - val_count]
    val_episodes = episodes[len(episodes) - val_count :]

    output_dir = Path(config.output_dir)
    if output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"Prepared dataset already exists: {output_dir}. Pass overwrite=True to replace it.")
        shutil.rmtree(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=False)
    (output_dir / "val").mkdir(parents=True, exist_ok=False)

    manifest_entries: list[dict[str, Any]] = []
    total_steps = 0
    for split_name, split_episodes in (("train", train_episodes), ("val", val_episodes)):
        for index, episode in enumerate(split_episodes):
            demo_dir = output_dir / split_name / f"demo_{index:03d}"
            entry = _convert_episode(episode, demo_dir, config)
            entry["split"] = split_name
            entry["demo_name"] = demo_dir.name
            manifest_entries.append(entry)
            total_steps += int(entry["num_steps"])

    scaler_values = _compute_scaler_values(output_dir)
    np.savez_compressed(output_dir / "scaler_values.npz", **_flatten_scaler_values(scaler_values))

    manifest = {
        "schema_version": "vt_franka_mpd_v1",
        "task_name": config.task_name,
        "dataset_name": config.dataset_name,
        "raw_format": "vt_franka_raw_streams",
        "raw_run_dir": str(config.raw_run_dir),
        "output_dir": str(output_dir),
        "dt": config.dt,
        "target_hz": config.target_hz,
        "action_convention": "tcp_xyz_rot6d_gripper_closedness",
        "action_alignment": "causal_future_command",
        "velocity_convention": "finite_difference_after_mpd_normalization",
        "keys": ["agent_pos", "agent_vel", "action", "action_vel"],
        "vector_dim": 10,
        "episodes": manifest_entries,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedDatasetResult(
        output_dir=output_dir,
        train_episodes=len(train_episodes),
        val_episodes=len(val_episodes),
        total_steps=total_steps,
        manifest_path=manifest_path,
    )


def _list_episode_streams(raw_run_dir: Path) -> list[EpisodeStreams]:
    episodes_dir = Path(raw_run_dir) / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Missing collected episodes directory: {episodes_dir}")
    episodes: list[EpisodeStreams] = []
    for episode_dir in sorted(path for path in episodes_dir.glob("episode_*") if path.is_dir()):
        controller_path = episode_dir / "streams" / "controller_state.jsonl"
        command_path = episode_dir / "streams" / "teleop_commands.jsonl"
        if not controller_path.exists() or not command_path.exists():
            continue
        controller_states = _read_jsonl(controller_path)
        teleop_commands = [record for record in _read_jsonl(command_path) if record.get("target_tcp") is not None]
        if controller_states and teleop_commands:
            episodes.append(
                EpisodeStreams(
                    episode_dir=episode_dir,
                    controller_states=controller_states,
                    teleop_commands=teleop_commands,
                )
            )
    if not episodes:
        raise FileNotFoundError(f"No raw episodes with controller_state and teleop_commands streams found in {episodes_dir}")
    return episodes


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def _resolve_val_count(num_episodes: int, val_ratio: float, val_episodes: int | None) -> int:
    if val_episodes is not None:
        val_count = int(val_episodes)
    else:
        val_count = max(1, int(round(float(val_ratio) * num_episodes)))
    if val_count <= 0 or val_count >= num_episodes:
        raise ValueError("Validation episode count must be at least 1 and smaller than total episode count")
    return val_count


def _convert_episode(episode: EpisodeStreams, demo_dir: Path, config: PrepareMPDDatasetConfig) -> dict[str, Any]:
    state_times, state_values = _extract_controller_states(episode.controller_states, config.gripper_open_width_m)
    command_times, command_values = _extract_teleop_commands(episode.teleop_commands)
    timestamps, agent_pos, actions = _align_streams(
        state_times=state_times,
        state_values=state_values,
        command_times=command_times,
        command_values=command_values,
        dt=config.dt,
        max_state_age_sec=config.max_state_age_sec,
        max_command_future_sec=config.max_command_future_sec,
    )
    if len(timestamps) < config.min_steps:
        raise RuntimeError(
            f"Episode {episode.episode_dir} produced only {len(timestamps)} aligned MPD steps; "
            f"minimum is {config.min_steps}"
        )

    agent_vel = finite_difference(agent_pos, config.dt)
    action_vel = finite_difference(actions, config.dt)
    demo_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(demo_dir / "agent_pos.npz", agent_pos.astype(np.float32))
    np.savez_compressed(demo_dir / "agent_vel.npz", agent_vel.astype(np.float32))
    np.savez_compressed(demo_dir / "action.npz", actions.astype(np.float32))
    np.savez_compressed(demo_dir / "action_vel.npz", action_vel.astype(np.float32))
    np.savez_compressed(demo_dir / "timestamps.npz", timestamps.astype(np.float64))

    entry = {
        "episode_dir": str(episode.episode_dir),
        "num_steps": int(len(timestamps)),
        "start_wall_time": float(timestamps[0]),
        "end_wall_time": float(timestamps[-1]),
        "keys": ["agent_pos", "agent_vel", "action", "action_vel"],
    }
    (demo_dir / "dataset_manifest.json").write_text(json.dumps(entry, indent=2), encoding="utf-8")
    return entry


def _extract_controller_states(records: list[dict[str, Any]], open_width_m: float) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[np.ndarray] = []
    for record in records:
        state = record.get("state", record)
        tcp_pose = state.get("tcp_pose")
        if tcp_pose is None:
            continue
        timestamp = float(state.get("wall_time", record.get("source_wall_time", record.get("recorded_at_wall_time", 0.0))))
        closedness = gripper_width_to_closedness(float(state.get("gripper_width", open_width_m)), open_width_m=open_width_m)
        times.append(timestamp)
        values.append(pose7d_and_gripper_to_tcp_state(tcp_pose, closedness))
    return _sort_by_time(times, values)


def _extract_teleop_commands(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    values: list[np.ndarray] = []
    for record in records:
        target_tcp = record.get("target_tcp")
        if target_tcp is None:
            continue
        timestamp = float(record.get("source_wall_time", record.get("recorded_at_wall_time", 0.0)))
        closedness = 1.0 if bool(record.get("gripper_closed", False)) else 0.0
        times.append(timestamp)
        values.append(pose7d_and_gripper_to_tcp_state(target_tcp, closedness))
    return _sort_by_time(times, values)


def _sort_by_time(times: list[float], values: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not times:
        return np.zeros((0,), dtype=np.float64), np.zeros((0, 10), dtype=np.float64)
    order = np.argsort(np.asarray(times, dtype=np.float64))
    return np.asarray(times, dtype=np.float64)[order], np.stack(values, axis=0).astype(np.float64)[order]


def _align_streams(
    *,
    state_times: np.ndarray,
    state_values: np.ndarray,
    command_times: np.ndarray,
    command_values: np.ndarray,
    dt: float,
    max_state_age_sec: float,
    max_command_future_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    start_time = max(float(state_times[0]), float(command_times[0]))
    end_time = min(float(state_times[-1]), float(command_times[-1]))
    if end_time <= start_time:
        raise RuntimeError("Controller state and teleop command streams do not overlap")

    grid = np.arange(start_time, end_time + 0.5 * dt, dt, dtype=np.float64)
    kept_times: list[float] = []
    kept_states: list[np.ndarray] = []
    kept_actions: list[np.ndarray] = []
    state_index = 0
    command_index = 0
    for timestamp in grid:
        while state_index + 1 < len(state_times) and state_times[state_index + 1] <= timestamp:
            state_index += 1
        while command_index < len(command_times) and command_times[command_index] <= timestamp:
            command_index += 1
        if command_index >= len(command_times):
            break
        state_age = timestamp - state_times[state_index]
        command_future = command_times[command_index] - timestamp
        if state_age < -1e-6 or state_age > max_state_age_sec:
            continue
        if command_future < -1e-6 or command_future > max_command_future_sec:
            continue
        kept_times.append(float(timestamp))
        kept_states.append(state_values[state_index])
        kept_actions.append(command_values[command_index])
    if not kept_times:
        raise RuntimeError("No aligned controller/teleop samples survived causal MPD alignment")
    return (
        np.asarray(kept_times, dtype=np.float64),
        np.stack(kept_states, axis=0).astype(np.float64),
        np.stack(kept_actions, axis=0).astype(np.float64),
    )


def _compute_scaler_values(output_dir: Path) -> dict[str, dict[str, np.ndarray]]:
    arrays_by_key: dict[str, list[np.ndarray]] = {key: [] for key in ("agent_pos", "agent_vel", "action", "action_vel")}
    for split_name in ("train", "val"):
        split_dir = output_dir / split_name
        for demo_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for key in arrays_by_key:
                arrays_by_key[key].append(np.load(demo_dir / f"{key}.npz")["arr_0"].astype(np.float32))
    scalers: dict[str, dict[str, np.ndarray]] = {}
    for key, arrays in arrays_by_key.items():
        combined = np.concatenate(arrays, axis=0).astype(np.float32)
        data_min = combined.min(axis=0)
        data_max = combined.max(axis=0)
        same = data_min == data_max
        data_min[same] = 0.0
        std = combined.std(axis=0)
        std[std == 0.0] = 1.0
        scalers[key] = {
            "min": data_min,
            "max": data_max,
            "mean": combined.mean(axis=0),
            "std": std,
        }
    return scalers


def _flatten_scaler_values(scaler_values: dict[str, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        f"{key}_{stat_name}": np.asarray(value, dtype=np.float32)
        for key, stats in scaler_values.items()
        for stat_name, value in stats.items()
    }
