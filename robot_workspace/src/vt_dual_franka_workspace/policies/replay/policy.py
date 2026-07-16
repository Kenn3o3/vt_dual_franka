from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...config import InferenceRuntimeSettings, PolicyConfig, WorkspaceSettings
from ..base import Policy


@dataclass(frozen=True)
class ReplayEpisode:
    target_tcp: np.ndarray
    gripper_closed: np.ndarray
    timestamps: np.ndarray


class ReplayPolicy(Policy):
    def __init__(
        self,
        episode: ReplayEpisode,
        *,
        chunk_size: int,
        skip_gripper: bool = False,
        gripper_open_width: float = 0.078,
        gripper_velocity: float = 0.1,
        gripper_force_limit: float = 7.0,
    ) -> None:
        self.episode = episode
        self.chunk_size = max(1, int(chunk_size))
        self.skip_gripper = bool(skip_gripper)
        self.gripper_open_width = float(gripper_open_width)
        self.gripper_velocity = float(gripper_velocity)
        self.gripper_force_limit = float(gripper_force_limit)
        self._next_index = 0
        self._last_gripper_closed: bool | None = None

    @classmethod
    def from_config(
        cls,
        policy_config: PolicyConfig,
        inference_config: InferenceRuntimeSettings,
        workspace: WorkspaceSettings,
    ) -> "ReplayPolicy":
        episode_path = policy_config.config.get("episode_dir") or policy_config.config.get("episode_path")
        if episode_path is None:
            episode_path = policy_config.checkpoint_path
        if episode_path is None:
            raise ValueError("Replay policy requires config.episode_dir or checkpoint_path")
        episode = load_replay_episode(Path(episode_path))
        chunk_size = int(policy_config.config.get("chunk_size", max(inference_config.exe_horizon, 1)))
        return cls(
            episode,
            chunk_size=chunk_size,
            skip_gripper=bool(policy_config.config.get("skip_gripper", False)),
            gripper_open_width=workspace.teleop.max_gripper_width,
            gripper_velocity=workspace.teleop.gripper_velocity,
            gripper_force_limit=workspace.teleop.grasp_force,
        )

    def reset(self) -> None:
        self._next_index = 0
        self._last_gripper_closed = None

    def predict(self, observation_window: list[dict[str, Any]]) -> list[dict[str, Any]]:
        del observation_window
        if len(self.episode.target_tcp) == 0:
            return [{"terminate": True}]
        actions: list[dict[str, Any]] = []
        final_index = len(self.episode.target_tcp) - 1
        for _ in range(self.chunk_size):
            index = min(self._next_index, final_index)
            action: dict[str, Any] = {"target_tcp": self.episode.target_tcp[index].astype(float).tolist()}
            if not self.skip_gripper:
                gripper_closed = bool(self.episode.gripper_closed[index])
                if self._last_gripper_closed is None or gripper_closed != self._last_gripper_closed:
                    action["gripper_velocity"] = self.gripper_velocity
                    action["gripper_force_limit"] = self.gripper_force_limit
                    if gripper_closed:
                        action["gripper_closed"] = True
                    else:
                        action["gripper_width"] = self.gripper_open_width
                    self._last_gripper_closed = gripper_closed
            if index >= final_index:
                action["terminate"] = True
                actions.append(action)
                break
            actions.append(action)
            self._next_index += 1
        return actions


def load_replay_episode(path: Path) -> ReplayEpisode:
    if path.is_dir():
        replay_path = path / "aligned_episode.npz"
    else:
        replay_path = path
    if replay_path.suffix != ".npz":
        raise ValueError(f"Replay policy expects an aligned_episode.npz file or episode directory, got: {replay_path}")
    if not replay_path.exists():
        raise FileNotFoundError(f"Replay aligned episode does not exist: {replay_path}")

    data = np.load(replay_path, allow_pickle=True)
    required_keys = ("teleop_target_tcp", "teleop_gripper_closed", "timestamps")
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(f"Replay aligned episode is missing keys {missing_keys}: {replay_path}")

    target_tcp = np.asarray(data["teleop_target_tcp"], dtype=np.float64)
    if target_tcp.ndim != 2 or target_tcp.shape[1] != 7:
        raise ValueError(f"Expected teleop_target_tcp to have shape (N, 7), got {target_tcp.shape}: {replay_path}")
    if target_tcp.shape[0] == 0:
        raise ValueError(f"Replay aligned episode is empty: {replay_path}")

    gripper_closed = np.asarray(data["teleop_gripper_closed"], dtype=bool)
    if gripper_closed.shape != (target_tcp.shape[0],):
        raise ValueError(
            f"Expected teleop_gripper_closed to have shape ({target_tcp.shape[0]},), "
            f"got {gripper_closed.shape}: {replay_path}"
        )

    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    if timestamps.shape != (target_tcp.shape[0],):
        raise ValueError(f"Expected timestamps to have shape ({target_tcp.shape[0]},), got {timestamps.shape}: {replay_path}")

    return ReplayEpisode(
        target_tcp=target_tcp,
        gripper_closed=gripper_closed,
        timestamps=timestamps,
    )
