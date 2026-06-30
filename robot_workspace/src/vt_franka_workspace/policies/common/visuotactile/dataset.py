from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class VisuotactileNpzDataset:
    """Small dependency-light dataset reader for prepared visuotactile NPZ files.

    Torch is intentionally not imported here. Training backends can wrap this class
    in a torch Dataset when torch is available on the training machine.
    """

    def __init__(
        self,
        dataset_dir: str | Path,
        *,
        split: str = "train",
        qpos_key: str | None = None,
        action_key: str | None = None,
        obs_horizon: int | None = None,
        action_horizon: int | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.manifest = json.loads((self.dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        self.normalizer_stats = json.loads((self.dataset_dir / "normalizer_stats.json").read_text(encoding="utf-8"))
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.split = split
        self.qpos_key = qpos_key or self.normalizer_stats["preferred_qpos_key"]
        self.action_key = action_key or self.normalizer_stats["preferred_action_key"]
        self.obs_horizon = int(obs_horizon or self.manifest.get("obs_horizon", 2))
        self.action_horizon = int(action_horizon or self.manifest.get("action_horizon", 8))
        self.episodes = [
            item
            for item in self.manifest["episodes"]
            if item.get("split") == split
        ]
        self._episode_arrays: list[dict[str, np.ndarray] | None] = [None] * len(self.episodes)
        self._index: list[tuple[int, int]] = []
        for episode_idx, entry in enumerate(self.episodes):
            num_steps = int(entry["num_steps"])
            for step in range(num_steps):
                self._index.append((episode_idx, step))
        if not self._index:
            raise RuntimeError(f"No samples found for split={split} in {dataset_dir}")

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_idx, step = self._index[index]
        arrays = self._load_episode(episode_idx)
        obs_indices = _padded_indices(step - self.obs_horizon + 1, step + 1, len(arrays["timestamps"]))
        action_indices = _padded_indices(step, step + self.action_horizon, len(arrays["timestamps"]))
        return {
            "rgb_wrist": arrays["rgb_wrist"][obs_indices],
            "gelsight": arrays["gelsight"][obs_indices],
            "qpos": arrays[self.qpos_key][obs_indices],
            "action": arrays[self.action_key][action_indices],
            "timestamps": arrays["timestamps"][obs_indices],
            "episode_index": episode_idx,
            "step_index": step,
        }

    def _load_episode(self, episode_idx: int) -> dict[str, np.ndarray]:
        cached = self._episode_arrays[episode_idx]
        if cached is not None:
            return cached
        entry = self.episodes[episode_idx]
        path = self.dataset_dir / entry["file"]
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        self._episode_arrays[episode_idx] = arrays
        return arrays


def normalize_array(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return (np.asarray(values, dtype=np.float32) - mean) / std


def unnormalize_array(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    return np.asarray(values, dtype=np.float32) * std + mean


def _padded_indices(start: int, stop: int, length: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("length must be positive")
    return np.asarray([min(max(idx, 0), length - 1) for idx in range(start, stop)], dtype=np.int64)
