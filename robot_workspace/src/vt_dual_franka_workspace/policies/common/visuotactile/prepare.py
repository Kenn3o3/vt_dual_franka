from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ....config import WorkspaceSettings, load_workspace_config
from ...mpd.math import gripper_width_to_closedness, pose7d_and_gripper_to_tcp_state
from .canonical import (
    CanonicalPreprocessConfig,
    build_preprocess1_from_collection_streams,
    ensure_preprocess1_episode_metadata,
    load_canonical_arrays,
    preprocess_aligned_episode_images,
    write_preprocess1_dataset_manifest,
)
from .config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_PREPROCESS1_PROFILE,
    VisuotactileModelSpec,
    default_prepared_dataset_dir,
    default_preprocess1_dir,
    get_model_spec,
)
from .image_preprocess import CropSpec, ImagePreprocessSpec, make_contact_sheet_rgb, preprocess_image_rgb, rgb_to_bgr
from ....recording.image_io import read_rgb_image


@dataclass(frozen=True)
class PrepareVisuotactileDatasetConfig:
    task_name: str
    model: str
    raw_run_dir: Path
    output_dir: Path
    preprocess1_root: Path
    source: Literal["raw", "preprocess1", "common"] = "raw"
    source_root: Path | None = None
    dataset_name: str = DEFAULT_DATASET_NAME
    preprocess1_profile: str = DEFAULT_PREPROCESS1_PROFILE
    target_hz: float = 10.0
    image_size: int | None = None
    val_ratio: float = 0.2
    val_episodes: int | None = None
    gripper_open_width_m: float = 0.078
    overwrite: bool = False
    build_preprocess1_if_missing: bool = True
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
        self.sum = np.zeros((self.width,), dtype=np.float64)
        self.sum_sq = np.zeros((self.width,), dtype=np.float64)
        self.min = np.full((self.width,), np.inf, dtype=np.float64)
        self.max = np.full((self.width,), -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1, self.width)
        if array.size == 0:
            return
        self.count += int(array.shape[0])
        self.sum += array.sum(axis=0)
        self.sum_sq += np.square(array).sum(axis=0)
        self.min = np.minimum(self.min, array.min(axis=0))
        self.max = np.maximum(self.max, array.max(axis=0))

    def to_json(self) -> dict[str, list[float]]:
        if self.count == 0:
            raise RuntimeError("Cannot finalize empty stats")
        mean = self.sum / self.count
        var = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        std = np.sqrt(var)
        std = np.clip(std, 1e-6, np.inf)
        return {
            "mean": mean.astype(float).tolist(),
            "std": std.astype(float).tolist(),
            "min": self.min.astype(float).tolist(),
            "max": self.max.astype(float).tolist(),
        }


class _ImageChannelStats:
    def __init__(self) -> None:
        self.count = 0
        self.sum = np.zeros((3,), dtype=np.float64)
        self.sum_sq = np.zeros((3,), dtype=np.float64)

    def update(self, images_uint8: np.ndarray) -> None:
        array = np.asarray(images_uint8, dtype=np.float32) / 255.0
        if array.ndim != 4 or array.shape[-1] != 3:
            raise ValueError(f"Expected image batch TxHxWx3, got {array.shape}")
        pixels = int(np.prod(array.shape[:-1]))
        self.count += pixels
        self.sum += array.sum(axis=(0, 1, 2), dtype=np.float64)
        self.sum_sq += np.square(array, dtype=np.float32).sum(axis=(0, 1, 2), dtype=np.float64)

    def to_json(self) -> dict[str, list[float]]:
        if self.count == 0:
            raise RuntimeError("Cannot finalize empty image stats")
        mean = self.sum / self.count
        var = np.maximum(self.sum_sq / self.count - np.square(mean), 0.0)
        std = np.clip(np.sqrt(var), 1e-6, np.inf)
        return {"mean": mean.astype(float).tolist(), "std": std.astype(float).tolist()}


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
    build_preprocess1_if_missing: bool = True,
    canonical_size: int = 480,
    gelsight_crop_box: tuple[int, int, int, int] | None = None,
    gelsight_margin_fraction: float = 0.0,
    source: Literal["raw", "preprocess1", "common"] = "raw",
    source_root: Path | None = None,
) -> PrepareVisuotactileDatasetConfig:
    spec = get_model_spec(model)
    resolved_source_root = source_root
    if resolved_source_root is None and source == "preprocess1":
        resolved_source_root = default_preprocess1_dir(workspace, task_name, preprocess1_profile)
    return PrepareVisuotactileDatasetConfig(
        task_name=task_name,
        model=spec.name,
        raw_run_dir=raw_run_dir or (Path(workspace.recording.collect_root) / task_name),
        output_dir=output_dir or default_prepared_dataset_dir(workspace, task_name, dataset_name, model=spec.name),
        preprocess1_root=default_preprocess1_dir(workspace, task_name, preprocess1_profile),
        source=source,
        source_root=resolved_source_root,
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


def prepare_visuotactile_dataset(config: PrepareVisuotactileDatasetConfig) -> PreparedVisuotactileDatasetResult:
    spec = get_model_spec(config.model)
    episodes = _list_prepare_episodes(config)
    if len(episodes) < 2:
        raise ValueError("Visuotactile fixed split requires at least two aligned episodes")

    val_count = _resolve_val_count(len(episodes), config.val_ratio, config.val_episodes)
    train_episodes = episodes[: len(episodes) - val_count]
    val_episodes = episodes[len(episodes) - val_count :]
    output_dir = Path(config.output_dir)
    if output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"Prepared visuotactile dataset already exists: {output_dir}")
        shutil.rmtree(output_dir)
    (output_dir / "train").mkdir(parents=True, exist_ok=False)
    (output_dir / "val").mkdir(parents=True, exist_ok=False)

    normalizer = {
        "qpos_pose7_gripper": _ArrayStats(8),
        "action_pose7_gripper": _ArrayStats(8),
        "qpos_pose10_rot6d_gripper": _ArrayStats(10),
        "action_pose10_rot6d_gripper": _ArrayStats(10),
    }
    image_stats = {"rgb_wrist": _ImageChannelStats(), "gelsight": _ImageChannelStats()}
    manifest_entries: list[dict[str, Any]] = []
    total_steps = 0
    preview_panels: list[tuple[str, np.ndarray]] = []

    for split_name, split_episodes in (("train", train_episodes), ("val", val_episodes)):
        for split_index, episode_dir in enumerate(split_episodes):
            entry, arrays = _convert_episode(
                episode_dir=episode_dir,
                split_name=split_name,
                split_index=split_index,
                output_dir=output_dir,
                config=config,
                spec=spec,
            )
            manifest_entries.append(entry)
            total_steps += int(entry["num_steps"])
            if split_name == "train":
                for key in normalizer:
                    normalizer[key].update(arrays[key])
                image_stats["rgb_wrist"].update(arrays["rgb_wrist"])
                image_stats["gelsight"].update(arrays["gelsight"])
            if len(preview_panels) < 12:
                preview_panels.append((f"{split_name} rgb {episode_dir.name}", arrays["rgb_wrist"][0]))
                preview_panels.append((f"{split_name} gel {episode_dir.name}", arrays["gelsight"][0]))

    normalizer_payload = {
        key: stats.to_json() for key, stats in normalizer.items()
    }
    normalizer_payload["rgb_wrist"] = image_stats["rgb_wrist"].to_json()
    normalizer_payload["gelsight"] = image_stats["gelsight"].to_json()
    normalizer_payload["preferred_action_key"] = (
        "action_pose10_rot6d_gripper" if spec.action_representation == "pose10_rot6d_gripper" else "action_pose7_gripper"
    )
    normalizer_payload["preferred_qpos_key"] = (
        "qpos_pose10_rot6d_gripper" if spec.action_representation == "pose10_rot6d_gripper" else "qpos_pose7_gripper"
    )
    (output_dir / "normalizer_stats.json").write_text(json.dumps(normalizer_payload, indent=2), encoding="utf-8")

    preview = make_contact_sheet_rgb(preview_panels, panel_size=180, columns=4)
    cv2 = _require_cv2()
    cv2.imwrite(str(output_dir / "preview_contact_sheet.jpg"), rgb_to_bgr(preview))

    preprocess2_specs = _preprocess2_specs(config, spec)
    manifest = {
        "schema_version": "vt_franka_visuotactile_dataset_v1",
        "task_name": config.task_name,
        "model": spec.name,
        "model_family": spec.family,
        "dataset_name": config.dataset_name,
        "source": config.source,
        "raw_run_dir": str(config.raw_run_dir),
        "source_root": None if config.source_root is None else str(config.source_root),
        "output_dir": str(output_dir),
        "preprocess1_root": str(config.preprocess1_root),
        "target_hz": float(config.target_hz),
        "preprocess1_profile": config.preprocess1_profile,
        "preprocess1_storage": "centralized",
        "preprocess2": {
            **{key: value.to_json() for key, value in preprocess2_specs.items()},
            "model_image_size": int(_model_image_size(config, spec)),
            "wrist_image_size": int(_stream_image_size(config, spec, "rgb_wrist")),
            "tactile_image_size": int(_stream_image_size(config, spec, "gelsight")),
        },
        "model_input": {
            "camera_names": list(spec.camera_names),
            "tactile_names": list(spec.tactile_names),
            "shape_meta": spec.backend_shape_meta(),
        },
        "action_representation": spec.action_representation,
        "action_dim": spec.action_dim,
        "qpos_dim": spec.qpos_dim,
        "obs_horizon": spec.obs_horizon,
        "action_horizon": spec.action_horizon,
        "splits": {"train": len(train_episodes), "val": len(val_episodes)},
        "total_steps": int(total_steps),
        "keys": [
            "rgb_wrist",
            "gelsight",
            "timestamps",
            "qpos_pose7_gripper",
            "action_pose7_gripper",
            "qpos_pose10_rot6d_gripper",
            "action_pose10_rot6d_gripper",
        ],
        "episodes": manifest_entries,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return PreparedVisuotactileDatasetResult(
        output_dir=output_dir,
        train_episodes=len(train_episodes),
        val_episodes=len(val_episodes),
        total_steps=total_steps,
        manifest_path=manifest_path,
    )


def main() -> None:
    import argparse

    from .image_preprocess import parse_crop_box

    parser = argparse.ArgumentParser(description="Prepare aligned VT Dual Franka episodes for visuotactile training")
    parser.add_argument("--workspace-config", default="robot_workspace/config/workspace.yaml")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--model", required=True, choices=sorted(get_model_spec(name).name for name in _model_names()))
    parser.add_argument("--raw-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--preprocess1-profile", default=DEFAULT_PREPROCESS1_PROFILE)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-episodes", type=int, default=None)
    parser.add_argument("--canonical-size", type=int, default=480)
    parser.add_argument("--gelsight-crop-box", default=None, help="Optional x0,y0,x1,y1 crop box before canonical resize")
    parser.add_argument("--gelsight-margin-fraction", type=float, default=0.0)
    parser.add_argument("--source", choices=["raw", "preprocess1", "common"], default="raw")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--no-build-preprocess1", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    workspace = load_workspace_config(args.workspace_config)
    config = build_prepare_config_from_workspace(
        workspace,
        task_name=args.task_name,
        model=args.model,
        raw_run_dir=args.raw_run_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        preprocess1_profile=args.preprocess1_profile,
        target_hz=args.target_hz,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        val_episodes=args.val_episodes,
        overwrite=args.overwrite,
        build_preprocess1_if_missing=not args.no_build_preprocess1,
        canonical_size=args.canonical_size,
        gelsight_crop_box=parse_crop_box(args.gelsight_crop_box),
        gelsight_margin_fraction=args.gelsight_margin_fraction,
        source=args.source,
        source_root=args.source_root,
    )
    result = prepare_visuotactile_dataset(config)
    print(f"Prepared visuotactile dataset: {result.output_dir}")
    print(f"Train episodes: {result.train_episodes}")
    print(f"Val episodes: {result.val_episodes}")
    print(f"Total steps: {result.total_steps}")
    print(f"Manifest: {result.manifest_path}")


def _convert_episode(
    *,
    episode_dir: Path,
    split_name: str,
    split_index: int,
    output_dir: Path,
    config: PrepareVisuotactileDatasetConfig,
    spec: VisuotactileModelSpec,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if config.source == "common":
        return _convert_common_episode(
            episode_dir=episode_dir,
            split_name=split_name,
            split_index=split_index,
            output_dir=output_dir,
            config=config,
            spec=spec,
        )
    preprocess_dir = _ensure_preprocess1(episode_dir, config)
    episode_bundle = _load_episode_bundle(preprocess_dir, episode_dir, config)
    canonical = episode_bundle["canonical"]
    aligned_indices = canonical["aligned_indices"].astype(np.int64)
    qpos_pose7, action_pose7, qpos_pose10, action_pose10 = _aligned_actions_from_canonical(
        episode_bundle["episode_npz"],
        aligned_indices,
        gripper_open_width_m=config.gripper_open_width_m,
    )
    preprocess2 = _preprocess2_specs(config, spec)
    rgb = _preprocess_image_batch(canonical["rgb_wrist"], preprocess2["rgb_wrist"])
    gelsight = _preprocess_image_batch(canonical["gelsight"], preprocess2["gelsight"])
    arrays = {
        "rgb_wrist": rgb,
        "gelsight": gelsight,
        "timestamps": canonical["timestamps"].astype(np.float64),
        "aligned_indices": aligned_indices.astype(np.int64),
        "qpos_pose7_gripper": qpos_pose7.astype(np.float32),
        "action_pose7_gripper": action_pose7.astype(np.float32),
        "qpos_pose10_rot6d_gripper": qpos_pose10.astype(np.float32),
        "action_pose10_rot6d_gripper": action_pose10.astype(np.float32),
    }
    episode_name = f"{episode_dir.name}.npz"
    rel_path = Path(split_name) / episode_name
    output_path = output_dir / rel_path
    with output_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    entry = {
        "episode_dir": str(episode_dir),
        "split": split_name,
        "split_index": int(split_index),
        "episode_name": episode_dir.name,
        "file": rel_path.as_posix(),
        "num_steps": int(len(arrays["timestamps"])),
        "start_wall_time": float(arrays["timestamps"][0]),
        "end_wall_time": float(arrays["timestamps"][-1]),
        "preprocess1_dir": str(preprocess_dir),
        "preprocess1_manifest": str(preprocess_dir / "preprocess1_manifest.json"),
        "image_shape": [int(rgb.shape[1]), int(rgb.shape[2]), 3],
        "gelsight_shape": [int(gelsight.shape[1]), int(gelsight.shape[2]), 3],
    }
    return entry, arrays


def _ensure_preprocess1(episode_dir: Path, config: PrepareVisuotactileDatasetConfig) -> Path:
    if config.source == "preprocess1":
        return Path(episode_dir)
    preprocess_dir = Path(config.preprocess1_root) / "episodes" / episode_dir.name
    if (preprocess_dir / "preprocess1_manifest.json").exists():
        ensure_preprocess1_episode_metadata(
            preprocess_dir,
            episode_dir=episode_dir,
            task_name=config.task_name,
            profile_name=config.preprocess1_profile,
        )
        write_preprocess1_dataset_manifest(
            config.preprocess1_root,
            task_name=config.task_name,
            profile_name=config.preprocess1_profile,
            raw_run_dir=config.raw_run_dir,
        )
        return preprocess_dir
    legacy_dir = episode_dir / "preprocessed" / config.preprocess1_profile
    if (legacy_dir / "preprocess1_manifest.json").exists() and not config.build_preprocess1_if_missing:
        ensure_preprocess1_episode_metadata(
            legacy_dir,
            episode_dir=episode_dir,
            task_name=config.task_name,
            profile_name=config.preprocess1_profile,
        )
        return legacy_dir
    if not config.build_preprocess1_if_missing:
        raise FileNotFoundError(f"Missing preprocess1 output: {preprocess_dir}")
    preprocess_config = CanonicalPreprocessConfig(
        profile_name=config.preprocess1_profile,
        canonical_size=config.canonical_size,
        overwrite=False,
        output_root=config.preprocess1_root,
        task_name=config.task_name,
        gelsight_crop_box=config.gelsight_crop_box,
        gelsight_margin_fraction=config.gelsight_margin_fraction,
    )
    if _has_collection_preprocess1_streams(episode_dir):
        result = build_preprocess1_from_collection_streams(episode_dir, preprocess_config)
    else:
        result = preprocess_aligned_episode_images(episode_dir, preprocess_config)
    write_preprocess1_dataset_manifest(
        config.preprocess1_root,
        task_name=config.task_name,
        profile_name=config.preprocess1_profile,
        raw_run_dir=config.raw_run_dir,
    )
    return result.output_dir


def _has_collection_preprocess1_streams(episode_dir: Path) -> bool:
    streams = Path(episode_dir) / "streams"
    return (
        (streams / "preprocess1_rgb_wrist.jsonl").exists()
        and (streams / "preprocess1_gelsight.jsonl").exists()
        and (streams / "preprocess1_rgb_wrist" / "manifest.json").exists()
        and (streams / "preprocess1_gelsight" / "manifest.json").exists()
    )


def _load_episode_bundle(
    preprocess_dir: Path,
    episode_dir: Path,
    config: PrepareVisuotactileDatasetConfig,
) -> dict[str, Any]:
    image_arrays = load_canonical_arrays(preprocess_dir)
    canonical_path = Path(preprocess_dir) / "canonical_episode.npz"
    if not canonical_path.exists():
        if config.source == "preprocess1":
            raise FileNotFoundError(f"Missing portable preprocess1 episode metadata: {canonical_path}")
        canonical_path = ensure_preprocess1_episode_metadata(
            preprocess_dir,
            episode_dir=episode_dir,
            task_name=config.task_name,
            profile_name=config.preprocess1_profile,
        )
    if config.source == "preprocess1":
        return {"canonical": image_arrays, "episode_npz": _load_preprocess1_episode_npz(preprocess_dir)}
    with np.load(episode_dir / "aligned_episode.npz", allow_pickle=True) as aligned:
        episode_npz = {key: np.asarray(aligned[key]) for key in aligned.files}
    return {"canonical": image_arrays, "episode_npz": episode_npz}


def _load_preprocess1_episode_npz(preprocess_dir: Path) -> dict[str, np.ndarray]:
    path = Path(preprocess_dir) / "canonical_episode.npz"
    with np.load(path, allow_pickle=False) as canonical:
        return {key: np.asarray(canonical[key]) for key in canonical.files}


def _aligned_actions_from_canonical(
    aligned: dict[str, np.ndarray],
    aligned_indices: np.ndarray,
    *,
    gripper_open_width_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    robot_pose = _select_or_use_canonical(aligned["robot_tcp_pose"], aligned_indices, dtype=np.float64)
    target_pose = _select_or_use_canonical(aligned["teleop_target_tcp"], aligned_indices, dtype=np.float64)
    gripper_width = _select_or_use_canonical(aligned["gripper_width"], aligned_indices, dtype=np.float64)
    teleop_closed = _select_or_use_canonical(aligned["teleop_gripper_closed"], aligned_indices, dtype=bool)
    qpos_gripper = np.asarray(
        [gripper_width_to_closedness(width, open_width_m=gripper_open_width_m) for width in gripper_width],
        dtype=np.float64,
    )
    action_gripper = teleop_closed.astype(np.float64)
    qpos_pose7 = np.concatenate([robot_pose, qpos_gripper[:, None]], axis=-1)
    action_pose7 = np.concatenate([target_pose, action_gripper[:, None]], axis=-1)
    qpos_pose10 = np.stack(
        [pose7d_and_gripper_to_tcp_state(pose, grip) for pose, grip in zip(robot_pose, qpos_gripper)],
        axis=0,
    )
    action_pose10 = np.stack(
        [pose7d_and_gripper_to_tcp_state(pose, grip) for pose, grip in zip(target_pose, action_gripper)],
        axis=0,
    )
    return qpos_pose7, action_pose7, qpos_pose10, action_pose10


def _select_or_use_canonical(array: np.ndarray, aligned_indices: np.ndarray, *, dtype: Any) -> np.ndarray:
    value = np.asarray(array, dtype=dtype)
    if len(value) == len(aligned_indices):
        return value
    return value[aligned_indices]


def _preprocess_image_batch(images: np.ndarray, spec: ImagePreprocessSpec) -> np.ndarray:
    return np.stack([preprocess_image_rgb(image, spec) for image in images], axis=0).astype(np.uint8)


def _preprocess2_spec(image_size: int) -> ImagePreprocessSpec:
    return ImagePreprocessSpec(
        output_size=(int(image_size), int(image_size)),
        crop=CropSpec(mode="center_square"),
        interpolation="area",
    )


def _preprocess2_specs(config: PrepareVisuotactileDatasetConfig, spec: VisuotactileModelSpec) -> dict[str, ImagePreprocessSpec]:
    return {
        "rgb_wrist": _preprocess2_spec(_stream_image_size(config, spec, "rgb_wrist")),
        "gelsight": _preprocess2_spec(_stream_image_size(config, spec, "gelsight")),
    }


def _stream_image_size(config: PrepareVisuotactileDatasetConfig, spec: VisuotactileModelSpec, stream: str) -> int:
    if config.image_size is not None:
        image_size = int(config.image_size)
    elif stream == "rgb_wrist":
        image_size = int(spec.wrist_image_size)
    elif stream == "gelsight":
        image_size = int(spec.tactile_image_size)
    else:
        raise ValueError(f"Unsupported stream: {stream}")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return image_size


def _model_image_size(config: PrepareVisuotactileDatasetConfig, spec: VisuotactileModelSpec) -> int:
    image_size = int(config.image_size or spec.default_image_size)
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    return image_size


def _list_aligned_episodes(raw_run_dir: Path) -> list[Path]:
    episodes_dir = Path(raw_run_dir) / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Missing episodes directory: {episodes_dir}")
    episodes = [
        path
        for path in sorted(episodes_dir.glob("episode_*"))
        if path.is_dir() and (path / "aligned_episode.npz").exists()
    ]
    if not episodes:
        raise FileNotFoundError(f"No aligned episodes found under {episodes_dir}")
    return episodes


def _list_prepare_episodes(config: PrepareVisuotactileDatasetConfig) -> list[Path]:
    if config.source == "raw":
        return _list_aligned_episodes(config.raw_run_dir)
    if config.source == "common":
        root = Path(config.source_root or config.raw_run_dir)
        episodes_dir = root / "episodes"
        if not episodes_dir.exists():
            raise FileNotFoundError(f"Missing common dataset episodes directory: {episodes_dir}")
        episodes = [
            path
            for path in sorted(episodes_dir.glob("episode_*"))
            if path.is_dir() and (path / "steps.jsonl").exists()
        ]
        if not episodes:
            raise FileNotFoundError(f"No common dataset episodes found under {episodes_dir}")
        return episodes
    if config.source != "preprocess1":
        raise ValueError(f"Unsupported visuotactile prepare source: {config.source}")
    root = Path(config.source_root or config.preprocess1_root)
    episodes_dir = root / "episodes"
    if episodes_dir.exists():
        episodes = [
            path
            for path in sorted(episodes_dir.glob("episode_*"))
            if path.is_dir() and (path / "preprocess1_manifest.json").exists() and (path / "canonical_episode.npz").exists()
        ]
        if episodes:
            return episodes
    manifest_path = root / "dataset_manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        episodes: list[Path] = []
        for entry in payload.get("episodes", []):
            preprocess_dir = entry.get("preprocess1_dir")
            if not preprocess_dir:
                continue
            path = Path(preprocess_dir)
            if not path.is_absolute():
                path = root / path
            if (path / "preprocess1_manifest.json").exists() and (path / "canonical_episode.npz").exists():
                episodes.append(path)
        if episodes:
            return sorted(episodes)
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Missing preprocess1 episodes directory: {episodes_dir}")
    episodes = [
        path
        for path in sorted(episodes_dir.glob("episode_*"))
        if path.is_dir() and (path / "preprocess1_manifest.json").exists() and (path / "canonical_episode.npz").exists()
    ]
    if not episodes:
        raise FileNotFoundError(f"No preprocess1 episode bundles found under {episodes_dir}")
    return episodes


def _convert_common_episode(
    *,
    episode_dir: Path,
    split_name: str,
    split_index: int,
    output_dir: Path,
    config: PrepareVisuotactileDatasetConfig,
    spec: VisuotactileModelSpec,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dataset_root = Path(config.source_root or config.raw_run_dir)
    steps = _read_common_steps(episode_dir / "steps.jsonl")
    if not steps:
        raise RuntimeError(f"Common dataset episode has no steps: {episode_dir}")
    preprocess2 = _preprocess2_specs(config, spec)
    rgb_frames: list[np.ndarray] = []
    gelsight_frames: list[np.ndarray] = []
    timestamps: list[float] = []
    qpos_pose7_rows: list[np.ndarray] = []
    action_pose7_rows: list[np.ndarray] = []
    qpos_pose10_rows: list[np.ndarray] = []
    action_pose10_rows: list[np.ndarray] = []
    for step in steps:
        images = step.get("images", {})
        rgb_path = _resolve_common_image_path(dataset_root, images.get("rgb_wrist"))
        tactile_path = _resolve_common_image_path(dataset_root, images.get("tactile_left"))
        rgb_frames.append(preprocess_image_rgb(read_rgb_image(rgb_path), preprocess2["rgb_wrist"]))
        gelsight_frames.append(preprocess_image_rgb(read_rgb_image(tactile_path), preprocess2["gelsight"]))
        timestamps.append(float(step["timestamp"]))
        controller_state = step.get("controller_state", {})
        action = step.get("action", {})
        robot_pose = controller_state.get("tcp_pose")
        target_pose = action.get("target_tcp")
        if robot_pose is None or target_pose is None:
            raise KeyError(f"Common dataset step is missing tcp pose/action: {episode_dir} step={step.get('step_index')}")
        gripper_width = float(controller_state.get("gripper_width", config.gripper_open_width_m))
        qpos_gripper = gripper_width_to_closedness(gripper_width, open_width_m=config.gripper_open_width_m)
        action_gripper = 1.0 if bool(action.get("gripper_closed", False)) else 0.0
        qpos_pose7 = np.concatenate([np.asarray(robot_pose, dtype=np.float64), np.asarray([qpos_gripper], dtype=np.float64)])
        action_pose7 = np.concatenate([np.asarray(target_pose, dtype=np.float64), np.asarray([action_gripper], dtype=np.float64)])
        qpos_pose7_rows.append(qpos_pose7)
        action_pose7_rows.append(action_pose7)
        qpos_pose10_rows.append(pose7d_and_gripper_to_tcp_state(np.asarray(robot_pose, dtype=np.float64), qpos_gripper))
        action_pose10_rows.append(pose7d_and_gripper_to_tcp_state(np.asarray(target_pose, dtype=np.float64), action_gripper))

    arrays = {
        "rgb_wrist": np.stack(rgb_frames, axis=0).astype(np.uint8),
        "gelsight": np.stack(gelsight_frames, axis=0).astype(np.uint8),
        "timestamps": np.asarray(timestamps, dtype=np.float64),
        "aligned_indices": np.arange(len(steps), dtype=np.int64),
        "qpos_pose7_gripper": np.stack(qpos_pose7_rows, axis=0).astype(np.float32),
        "action_pose7_gripper": np.stack(action_pose7_rows, axis=0).astype(np.float32),
        "qpos_pose10_rot6d_gripper": np.stack(qpos_pose10_rows, axis=0).astype(np.float32),
        "action_pose10_rot6d_gripper": np.stack(action_pose10_rows, axis=0).astype(np.float32),
    }
    rel_path = Path(split_name) / f"{episode_dir.name}.npz"
    output_path = output_dir / rel_path
    with output_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    entry = {
        "episode_dir": str(episode_dir),
        "split": split_name,
        "split_index": int(split_index),
        "episode_name": episode_dir.name,
        "file": rel_path.as_posix(),
        "num_steps": int(len(arrays["timestamps"])),
        "start_wall_time": float(arrays["timestamps"][0]),
        "end_wall_time": float(arrays["timestamps"][-1]),
        "common_episode_dir": str(episode_dir),
        "common_steps": str(episode_dir / "steps.jsonl"),
        "image_shape": [int(arrays["rgb_wrist"].shape[1]), int(arrays["rgb_wrist"].shape[2]), 3],
        "gelsight_shape": [int(arrays["gelsight"].shape[1]), int(arrays["gelsight"].shape[2]), 3],
    }
    return entry, arrays


def _read_common_steps(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _resolve_common_image_path(dataset_root: Path, rel_path: Any) -> Path:
    if not rel_path:
        raise RuntimeError("Common dataset step is missing image path")
    path = Path(str(rel_path))
    if not path.is_absolute():
        path = dataset_root / path
    if not path.exists():
        raise FileNotFoundError(f"Missing common dataset image: {path}")
    return path


def _resolve_val_count(num_episodes: int, val_ratio: float, val_episodes: int | None) -> int:
    if val_episodes is not None:
        val_count = int(val_episodes)
    else:
        val_count = max(1, int(round(float(val_ratio) * num_episodes)))
    if val_count <= 0 or val_count >= num_episodes:
        raise ValueError("Validation episode count must be at least 1 and smaller than total episode count")
    return val_count


def _model_names() -> list[str]:
    from .config import MODEL_SPECS

    return sorted(MODEL_SPECS)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for visuotactile dataset preparation") from exc
    return cv2


if __name__ == "__main__":
    main()
