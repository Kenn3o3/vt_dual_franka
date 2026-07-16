from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ....config import WorkspaceSettings
from ....recording.image_io import read_rgb_image
from .config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_PREPROCESS1_PROFILE,
    VisuotactileModelSpec,
    default_prepared_dataset_dir,
    default_preprocess1_dir,
    get_model_spec,
)
from .image_preprocess import CropSpec, ImagePreprocessSpec, preprocess_image_rgb


@dataclass(frozen=True)
class PrepareVisuotactileDatasetConfig:
    task_name: str
    model: str
    raw_run_dir: Path
    output_dir: Path
    preprocess1_root: Path
    source: Literal["common"] = "common"
    source_root: Path | None = None
    dataset_name: str = DEFAULT_DATASET_NAME
    preprocess1_profile: str = DEFAULT_PREPROCESS1_PROFILE
    target_hz: float = 10.0
    image_size: int | None = None
    val_ratio: float = 0.2
    val_episodes: int | None = None
    gripper_open_width_m: float = 0.078
    overwrite: bool = False
    build_preprocess1_if_missing: bool = False
    canonical_size: int = 480
    gelsight_crop_box: tuple[int, int, int, int] | None = None
    gelsight_margin_fraction: float = 0.0


@dataclass(frozen=True)
class PreparedVisuotactileDatasetResult:
    output_dir: Path
    train_episodes: int
    val_episodes: int
    total_steps: int
    manifest_path: Path


class _ArrayStats:
    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.count = 0
        self.sum = np.zeros(self.width, dtype=np.float64)
        self.sum_sq = np.zeros(self.width, dtype=np.float64)
        self.minimum = np.full(self.width, np.inf, dtype=np.float64)
        self.maximum = np.full(self.width, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        if not array.size:
            return
        self.count += int(array.shape[0])
        self.sum += array.sum(axis=0)
        self.sum_sq += np.square(array).sum(axis=0)
        self.minimum = np.minimum(self.minimum, array.min(axis=0))
        self.maximum = np.maximum(self.maximum, array.max(axis=0))

    def to_json(self) -> dict[str, list[float]]:
        if self.count <= 0:
            raise RuntimeError("Cannot finalize empty bimanual normalizer statistics")
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.astype(float).tolist(),
            "std": np.clip(np.sqrt(variance), 1e-6, np.inf).astype(float).tolist(),
            "min": self.minimum.astype(float).tolist(),
            "max": self.maximum.astype(float).tolist(),
        }


class _ImageStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = np.zeros(3, dtype=np.float64)
        self.sum_sq = np.zeros(3, dtype=np.float64)

    def update(self, images: np.ndarray) -> None:
        array = np.asarray(images, dtype=np.float32) / 255.0
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(f"Expected TxHxWx3 image batch, got {array.shape}")
        self.count += int(np.prod(array.shape[:-1]))
        self.sum += array.sum(axis=(0, 1, 2), dtype=np.float64)
        self.sum_sq += np.square(array).sum(axis=(0, 1, 2), dtype=np.float64)

    def to_json(self) -> dict[str, list[float]]:
        if self.count <= 0:
            raise RuntimeError("Cannot finalize empty bimanual image statistics")
        mean = self.sum / self.count
        variance = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        return {
            "mean": mean.astype(float).tolist(),
            "std": np.clip(np.sqrt(variance), 1e-6, np.inf).astype(float).tolist(),
        }


def build_prepare_config_from_workspace(
    workspace: WorkspaceSettings,
    *,
    task_name: str,
    model: str,
    raw_run_dir: Path | None = None,
    output_dir: Path | None = None,
    dataset_name: str = DEFAULT_DATASET_NAME,
    preprocess1_profile: str = DEFAULT_PREPROCESS1_PROFILE,
    target_hz: float = 10.0,
    image_size: int | None = None,
    val_ratio: float = 0.2,
    val_episodes: int | None = None,
    overwrite: bool = False,
    build_preprocess1_if_missing: bool = False,
    canonical_size: int = 480,
    gelsight_crop_box: tuple[int, int, int, int] | None = None,
    gelsight_margin_fraction: float = 0.0,
    source: str = "common",
    source_root: Path | None = None,
) -> PrepareVisuotactileDatasetConfig:
    spec = get_model_spec(model)
    if spec.name != "dp_bimanual":
        raise ValueError("VT Dual Franka preparation supports only dp_bimanual")
    if source != "common":
        raise ValueError(
            "VT Dual Franka training accepts only the commanded-action common dataset; "
            "run `vt-dual-franka-workspace make-dataset` first"
        )
    common_root = Path(source_root or raw_run_dir or (Path(workspace.recording.collect_root) / task_name))
    return PrepareVisuotactileDatasetConfig(
        task_name=task_name,
        model=spec.name,
        raw_run_dir=common_root,
        output_dir=output_dir
        or default_prepared_dataset_dir(
            workspace,
            task_name,
            dataset_name,
            model=spec.name,
        ),
        preprocess1_root=default_preprocess1_dir(workspace, task_name, preprocess1_profile),
        source="common",
        source_root=common_root,
        dataset_name=dataset_name,
        preprocess1_profile=preprocess1_profile,
        target_hz=target_hz,
        image_size=image_size,
        val_ratio=val_ratio,
        val_episodes=val_episodes,
        gripper_open_width_m=workspace.teleop.max_gripper_width,
        overwrite=overwrite,
        build_preprocess1_if_missing=build_preprocess1_if_missing,
        canonical_size=canonical_size,
        gelsight_crop_box=gelsight_crop_box,
        gelsight_margin_fraction=gelsight_margin_fraction,
    )


def prepare_visuotactile_dataset(
    config: PrepareVisuotactileDatasetConfig,
) -> PreparedVisuotactileDatasetResult:
    spec = get_model_spec(config.model)
    if spec.name != "dp_bimanual" or config.source != "common":
        raise ValueError("Only dp_bimanual commanded-action common datasets are supported")
    source_root = Path(config.source_root or config.raw_run_dir)
    source_manifest = _read_json(source_root / "dataset_manifest.json")
    if source_manifest.get("schema_version") != "vt_dual_franka_common_dataset_v1":
        raise ValueError(f"Not a VT Dual Franka common dataset: {source_root}")
    episode_entries = list(source_manifest.get("episodes") or [])
    if len(episode_entries) < 2:
        raise ValueError("Bimanual fixed split requires at least two episodes")

    val_count = _resolve_val_count(
        len(episode_entries),
        config.val_ratio,
        config.val_episodes,
    )
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(output_dir)
        shutil.rmtree(output_dir)
    (output_dir / "train").mkdir(parents=True)
    (output_dir / "val").mkdir(parents=True)

    qpos_stats = _ArrayStats(20)
    action_stats = _ArrayStats(20)
    image_stats = {
        stream_name: _ImageStats()
        for stream_name in (
            "rgb_wrist_left",
            "rgb_wrist_right",
            "tactile_left",
            "tactile_right",
        )
    }
    manifest_entries: list[dict[str, Any]] = []
    total_steps = 0
    train_cutoff = len(episode_entries) - val_count
    for index, source_entry in enumerate(episode_entries):
        split = "train" if index < train_cutoff else "val"
        steps_path = source_root / str(source_entry["steps_path"])
        rows = _read_jsonl(steps_path)
        arrays = _convert_common_rows(
            rows,
            dataset_root=source_root,
            spec=spec,
            image_size=config.image_size,
        )
        output_path = output_dir / split / f"episode_{index:04d}.npz"
        np.savez_compressed(output_path, **arrays)
        entry = {
            "episode_id": str(source_entry.get("episode_id") or steps_path.parent.name),
            "split": split,
            "file": output_path.relative_to(output_dir).as_posix(),
            "num_steps": int(arrays["timestamps"].shape[0]),
            "source_steps": str(steps_path),
        }
        manifest_entries.append(entry)
        total_steps += entry["num_steps"]
        if split == "train":
            qpos_stats.update(arrays["qpos20"])
            action_stats.update(arrays["action20"])
            for stream_name, stats in image_stats.items():
                stats.update(arrays[stream_name])

    normalizer = {
        "schema_version": "vt_dual_franka_bimanual_normalizer_v1",
        "preferred_qpos_key": "qpos20",
        "preferred_action_key": "action20",
        "qpos20": qpos_stats.to_json(),
        "action20": action_stats.to_json(),
        **{stream_name: stats.to_json() for stream_name, stats in image_stats.items()},
    }
    (output_dir / "normalizer_stats.json").write_text(
        json.dumps(normalizer, indent=2),
        encoding="utf-8",
    )
    model_size = int(config.image_size or spec.default_image_size)
    manifest = {
        "schema_version": "vt_dual_franka_bimanual_training_dataset_v1",
        "task_name": config.task_name,
        "model": spec.name,
        "dataset_name": config.dataset_name,
        "source": "commanded_action_common",
        "source_root": str(source_root),
        "output_dir": str(output_dir),
        "action_provenance": "future_commanded_action",
        "action_representation": spec.action_representation,
        "action_dim": 20,
        "qpos_dim": 20,
        "obs_horizon": spec.obs_horizon,
        "action_horizon": spec.action_horizon,
        "model_input": {
            "camera_names": list(spec.camera_names),
            "tactile_names": list(spec.tactile_names),
            "shape_meta": spec.backend_shape_meta(),
        },
        "preprocess2": {
            "model_image_size": model_size,
            **spec.preprocess2_specs(),
        },
        "splits": {"train": train_cutoff, "val": val_count},
        "total_steps": total_steps,
        "keys": [
            "rgb_wrist_left",
            "rgb_wrist_right",
            "tactile_left",
            "tactile_right",
            "qpos20",
            "action20",
            "timestamps",
        ],
        "episodes": manifest_entries,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedVisuotactileDatasetResult(
        output_dir=output_dir,
        train_episodes=train_cutoff,
        val_episodes=val_count,
        total_steps=total_steps,
        manifest_path=manifest_path,
    )


def _convert_common_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_root: Path,
    spec: VisuotactileModelSpec,
    image_size: int | None,
) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("Common bimanual episode contains no rows")
    size = int(image_size or spec.default_image_size)
    image_spec = ImagePreprocessSpec(
        output_size=(size, size),
        crop=CropSpec(mode="none"),
        interpolation="area",
    )
    streams = {
        name: []
        for name in (
            "rgb_wrist_left",
            "rgb_wrist_right",
            "tactile_left",
            "tactile_right",
        )
    }
    qpos: list[list[float]] = []
    actions: list[list[float]] = []
    timestamps: list[float] = []
    for row in rows:
        image_paths = row.get("images")
        if not isinstance(image_paths, dict):
            raise ValueError("Bimanual common row is missing images")
        for stream_name in streams:
            path = dataset_root / str(image_paths[stream_name])
            streams[stream_name].append(
                preprocess_image_rgb(read_rgb_image(path), image_spec)
            )
        qpos_row = list(row["qpos20"])
        action_row = list(row["commanded_actions"]["action20"])
        if len(qpos_row) != 20 or len(action_row) != 20:
            raise ValueError("Bimanual qpos/action rows must be 20D")
        qpos.append(qpos_row)
        actions.append(action_row)
        timestamps.append(float(row["timestamp"]))
    return {
        **{
            stream_name: np.stack(images, axis=0).astype(np.uint8)
            for stream_name, images in streams.items()
        },
        "qpos20": np.asarray(qpos, dtype=np.float32),
        "action20": np.asarray(actions, dtype=np.float32),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
    }


def _resolve_val_count(
    episode_count: int,
    val_ratio: float,
    val_episodes: int | None,
) -> int:
    if episode_count < 2:
        raise ValueError("At least two episodes are required")
    if val_episodes is not None:
        count = int(val_episodes)
    else:
        count = max(1, int(round(episode_count * float(val_ratio))))
    return min(max(count, 1), episode_count - 1)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    raise SystemExit(
        "Use `vt-dual-franka-workspace train`; standalone legacy preparation is removed"
    )


__all__ = [
    "PrepareVisuotactileDatasetConfig",
    "PreparedVisuotactileDatasetResult",
    "build_prepare_config_from_workspace",
    "prepare_visuotactile_dataset",
]
