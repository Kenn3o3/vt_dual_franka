from __future__ import annotations

import json
import os
import shutil
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import DEFAULT_PREPROCESS1_PROFILE
from .image_preprocess import (
    ImagePreprocessSpec,
    bgr_to_rgb,
    default_preprocess1_specs,
    make_contact_sheet_rgb,
    preprocess_image_rgb,
    read_image_file_as_rgb,
    rgb_to_bgr,
)


@dataclass(frozen=True)
class CanonicalPreprocessConfig:
    profile_name: str = DEFAULT_PREPROCESS1_PROFILE
    canonical_size: int = 480
    chunk_frames: int = 64
    overwrite: bool = False
    output_root: Path | None = None
    task_name: str | None = None
    gelsight_crop_box: tuple[int, int, int, int] | None = None
    gelsight_margin_fraction: float = 0.0


@dataclass(frozen=True)
class CanonicalPreprocessResult:
    episode_dir: Path
    output_dir: Path
    manifest_path: Path
    kept_steps: int
    dropped_steps: int


PREPROCESS1_EPISODE_SCHEMA_VERSION = "vt_franka_preprocess1_episode_v2"
PREPROCESS1_DATASET_SCHEMA_VERSION = "vt_franka_preprocess1_dataset_v2"

CANONICAL_EPISODE_REQUIRED_KEYS = (
    "robot_tcp_pose",
    "teleop_target_tcp",
    "gripper_width",
    "teleop_gripper_closed",
)

CANONICAL_EPISODE_OPTIONAL_KEYS = (
    "robot_tcp_velocity",
    "robot_tcp_wrench",
    "robot_joint_positions",
    "robot_joint_velocities",
    "gripper_force",
    "gripper_state",
    "controller_state_valid",
    "controller_state_age_sec",
    "controller_state_source_timestamps",
    "teleop_command_source_timestamps",
    "teleop_action_lead_sec",
)


def preprocess_aligned_episode_images(
    episode_dir: str | Path,
    config: CanonicalPreprocessConfig | None = None,
) -> CanonicalPreprocessResult:
    config = config or CanonicalPreprocessConfig()
    episode_dir = Path(episode_dir)
    aligned_path = episode_dir / "aligned_episode.npz"
    if not aligned_path.exists():
        raise FileNotFoundError(f"Missing aligned episode file: {aligned_path}")

    output_dir = _resolve_output_dir(episode_dir, config)
    manifest_path = output_dir / "preprocess1_manifest.json"
    if output_dir.exists():
        if not config.overwrite and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return CanonicalPreprocessResult(
                episode_dir=episode_dir,
                output_dir=output_dir,
                manifest_path=manifest_path,
                kept_steps=int(manifest.get("kept_steps", 0)),
                dropped_steps=int(manifest.get("dropped_steps", 0)),
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir()

    specs = default_preprocess1_specs(
        canonical_size=config.canonical_size,
        gelsight_crop_box=config.gelsight_crop_box,
        gelsight_margin_fraction=config.gelsight_margin_fraction,
    )
    started = time.time()
    with np.load(aligned_path, allow_pickle=True) as aligned:
        timestamps = np.asarray(aligned["timestamps"], dtype=np.float64)
        rgb_paths = np.asarray(aligned.get("rgb_wrist_frame_paths", np.asarray([""] * len(timestamps))), dtype=object)
        gelsight_paths = np.asarray(aligned.get("gelsight_frame_paths", np.asarray([""] * len(timestamps))), dtype=object)
        gelsight_indices = np.asarray(aligned.get("gelsight_frame_indices", np.full(len(timestamps), -1)), dtype=np.int64)

        chunk_rgb: list[np.ndarray] = []
        chunk_gelsight: list[np.ndarray] = []
        chunk_timestamps: list[float] = []
        chunk_aligned_indices: list[int] = []
        records: list[dict[str, Any]] = []
        preview_panels: list[tuple[str, np.ndarray]] = []
        frame_reader = _EpisodeFrameReader(episode_dir)
        chunk_index = 0
        source_shapes: dict[str, list[int]] = {}
        dropped = 0

        for aligned_index, timestamp in enumerate(timestamps):
            rgb_path = str(rgb_paths[aligned_index])
            gelsight_path = str(gelsight_paths[aligned_index])
            if not rgb_path or not gelsight_path:
                dropped += 1
                continue
            try:
                rgb_raw = frame_reader.read_rgb(rgb_path, -1, source_kind="rgb_wrist")
                gelsight_raw = frame_reader.read_rgb(
                    gelsight_path,
                    int(gelsight_indices[aligned_index]),
                    source_kind="gelsight",
                )
            except Exception:
                dropped += 1
                continue
            source_shapes.setdefault("rgb_wrist", list(rgb_raw.shape))
            source_shapes.setdefault("gelsight", list(gelsight_raw.shape))
            rgb = preprocess_image_rgb(rgb_raw, specs["rgb_wrist"])
            gelsight = preprocess_image_rgb(gelsight_raw, specs["gelsight"])
            canonical_index = len(records)
            chunk_rgb.append(rgb)
            chunk_gelsight.append(gelsight)
            chunk_timestamps.append(float(timestamp))
            chunk_aligned_indices.append(int(aligned_index))
            if canonical_index in {0, len(timestamps) // 2, len(timestamps) - 1} or len(preview_panels) < 6:
                preview_panels.append((f"rgb {aligned_index}", rgb))
                preview_panels.append((f"gel {aligned_index}", gelsight))
            if len(chunk_rgb) >= config.chunk_frames:
                _flush_chunk(
                    chunks_dir=chunks_dir,
                    records=records,
                    chunk_index=chunk_index,
                    rgb=chunk_rgb,
                    gelsight=chunk_gelsight,
                    timestamps=chunk_timestamps,
                    aligned_indices=chunk_aligned_indices,
                )
                chunk_index += 1
                chunk_rgb, chunk_gelsight, chunk_timestamps, chunk_aligned_indices = [], [], [], []

        if chunk_rgb:
            _flush_chunk(
                chunks_dir=chunks_dir,
                records=records,
                chunk_index=chunk_index,
                rgb=chunk_rgb,
                gelsight=chunk_gelsight,
                timestamps=chunk_timestamps,
                aligned_indices=chunk_aligned_indices,
            )

    if not records:
        raise RuntimeError(f"No canonical image pairs could be generated for {episode_dir}")

    kept_aligned_indices = np.asarray([record["aligned_index"] for record in records], dtype=np.int64)
    canonical_episode_path = output_dir / "canonical_episode.npz"
    _write_canonical_episode_npz(
        aligned_path=aligned_path,
        output_path=canonical_episode_path,
        aligned_indices=kept_aligned_indices,
    )

    index_path = output_dir / "canonical_index.jsonl"
    index_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    preview = make_contact_sheet_rgb(preview_panels, panel_size=180, columns=4)
    cv2 = _require_cv2()
    cv2.imwrite(str(output_dir / "preprocess1_preview.jpg"), rgb_to_bgr(preview))

    manifest = {
        "schema_version": PREPROCESS1_EPISODE_SCHEMA_VERSION,
        "compatible_schema_versions": ["vt_franka_preprocess1_v1"],
        "profile_name": config.profile_name,
        "task_name": config.task_name,
        "episode_dir": str(episode_dir),
        "aligned_episode": "aligned_episode.npz",
        "output_dir": str(output_dir),
        "storage_layout": "centralized" if config.output_root is not None else "episode_local",
        "source_episode_name": episode_dir.name,
        "canonical_size": int(config.canonical_size),
        "chunk_frames": int(config.chunk_frames),
        "kept_steps": int(len(records)),
        "dropped_steps": int(dropped),
        "streams": {
            "rgb_wrist": {
                "source_color_order": "bgr_opencv",
                "canonical_color_order": "rgb",
                "source_shape": source_shapes.get("rgb_wrist"),
                "preprocess": specs["rgb_wrist"].to_json(),
            },
            "gelsight": {
                "source_color_order": "bgr_opencv",
                "canonical_color_order": "rgb",
                "source_shape": source_shapes.get("gelsight"),
                "preprocess": specs["gelsight"].to_json(),
            },
        },
        "index_path": "canonical_index.jsonl",
        "canonical_episode_path": "canonical_episode.npz",
        "chunks_dir": "chunks",
        "generated_at_wall_time": time.time(),
        "duration_sec": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return CanonicalPreprocessResult(
        episode_dir=episode_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        kept_steps=len(records),
        dropped_steps=dropped,
    )


def build_preprocess1_from_collection_streams(
    episode_dir: str | Path,
    config: CanonicalPreprocessConfig | None = None,
) -> CanonicalPreprocessResult:
    config = config or CanonicalPreprocessConfig()
    episode_dir = Path(episode_dir)
    aligned_path = episode_dir / "aligned_episode.npz"
    if not aligned_path.exists():
        raise FileNotFoundError(f"Missing aligned episode file: {aligned_path}")

    output_dir = _resolve_output_dir(episode_dir, config)
    manifest_path = output_dir / "preprocess1_manifest.json"
    if output_dir.exists():
        if not config.overwrite and manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            return CanonicalPreprocessResult(
                episode_dir=episode_dir,
                output_dir=output_dir,
                manifest_path=manifest_path,
                kept_steps=int(manifest.get("kept_steps", 0)),
                dropped_steps=int(manifest.get("dropped_steps", 0)),
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    chunks_dir = output_dir / "chunks"
    chunks_dir.mkdir()

    started = time.time()
    specs = default_preprocess1_specs(
        canonical_size=config.canonical_size,
        gelsight_crop_box=config.gelsight_crop_box,
        gelsight_margin_fraction=config.gelsight_margin_fraction,
    )
    stream_manifests = _collection_preprocess1_stream_manifests(episode_dir)
    rgb_stream = stream_manifests.get("rgb_wrist")
    gelsight_stream = stream_manifests.get("gelsight")
    if rgb_stream is None or gelsight_stream is None:
        raise FileNotFoundError(f"Missing collection-time preprocess1 streams for {episode_dir}")

    with np.load(aligned_path, allow_pickle=True) as aligned:
        timestamps = np.asarray(aligned["timestamps"], dtype=np.float64)
        rgb_paths = np.asarray(aligned.get("preprocess1_rgb_wrist_frame_paths", np.asarray([""] * len(timestamps))), dtype=object)
        rgb_indices = np.asarray(aligned.get("preprocess1_rgb_wrist_frame_indices", np.full(len(timestamps), -1)), dtype=np.int64)
        gelsight_paths = np.asarray(aligned.get("preprocess1_gelsight_frame_paths", np.asarray([""] * len(timestamps))), dtype=object)
        gelsight_indices = np.asarray(aligned.get("preprocess1_gelsight_frame_indices", np.full(len(timestamps), -1)), dtype=np.int64)

        chunk_rgb: list[np.ndarray] = []
        chunk_gelsight: list[np.ndarray] = []
        chunk_timestamps: list[float] = []
        chunk_aligned_indices: list[int] = []
        records: list[dict[str, Any]] = []
        preview_panels: list[tuple[str, np.ndarray]] = []
        chunk_index = 0
        dropped = 0

        for aligned_index, timestamp in enumerate(timestamps):
            rgb_path = str(rgb_paths[aligned_index])
            gelsight_path = str(gelsight_paths[aligned_index])
            if not rgb_path or not gelsight_path:
                dropped += 1
                continue
            try:
                rgb = _read_collection_canonical_frame(episode_dir, rgb_path, int(rgb_indices[aligned_index]))
                gelsight = _read_collection_canonical_frame(
                    episode_dir,
                    gelsight_path,
                    int(gelsight_indices[aligned_index]),
                )
            except Exception:
                dropped += 1
                continue
            canonical_index = len(records)
            chunk_rgb.append(rgb)
            chunk_gelsight.append(gelsight)
            chunk_timestamps.append(float(timestamp))
            chunk_aligned_indices.append(int(aligned_index))
            if canonical_index in {0, len(timestamps) // 2, len(timestamps) - 1} or len(preview_panels) < 6:
                preview_panels.append((f"rgb {aligned_index}", rgb))
                preview_panels.append((f"gel {aligned_index}", gelsight))
            if len(chunk_rgb) >= config.chunk_frames:
                _flush_chunk(
                    chunks_dir=chunks_dir,
                    records=records,
                    chunk_index=chunk_index,
                    rgb=chunk_rgb,
                    gelsight=chunk_gelsight,
                    timestamps=chunk_timestamps,
                    aligned_indices=chunk_aligned_indices,
                )
                chunk_index += 1
                chunk_rgb, chunk_gelsight, chunk_timestamps, chunk_aligned_indices = [], [], [], []

        if chunk_rgb:
            _flush_chunk(
                chunks_dir=chunks_dir,
                records=records,
                chunk_index=chunk_index,
                rgb=chunk_rgb,
                gelsight=chunk_gelsight,
                timestamps=chunk_timestamps,
                aligned_indices=chunk_aligned_indices,
            )

    if not records:
        raise RuntimeError(f"No collection-time preprocess1 image pairs could be generated for {episode_dir}")

    kept_aligned_indices = np.asarray([record["aligned_index"] for record in records], dtype=np.int64)
    _write_canonical_episode_npz(
        aligned_path=aligned_path,
        output_path=output_dir / "canonical_episode.npz",
        aligned_indices=kept_aligned_indices,
    )
    (output_dir / "canonical_index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    cv2 = _require_cv2()
    preview = make_contact_sheet_rgb(preview_panels, panel_size=180, columns=4)
    cv2.imwrite(str(output_dir / "preprocess1_preview.jpg"), rgb_to_bgr(preview))

    manifest = {
        "schema_version": PREPROCESS1_EPISODE_SCHEMA_VERSION,
        "compatible_schema_versions": ["vt_franka_preprocess1_v1"],
        "profile_name": config.profile_name,
        "task_name": config.task_name,
        "episode_dir": str(episode_dir),
        "aligned_episode": "aligned_episode.npz",
        "output_dir": str(output_dir),
        "storage_layout": "centralized" if config.output_root is not None else "episode_local",
        "source_episode_name": episode_dir.name,
        "source_stage": "collection_time_preprocess1",
        "canonical_size": int(config.canonical_size),
        "chunk_frames": int(config.chunk_frames),
        "kept_steps": int(len(records)),
        "dropped_steps": int(dropped),
        "streams": {
            "rgb_wrist": {
                "source_color_order": "rgb",
                "canonical_color_order": "rgb",
                "source_shape": rgb_stream.get("source_shape"),
                "collection_stream": rgb_stream,
                "compatibility_transforms": rgb_stream.get("compatibility_transforms", []),
                "preprocess": specs["rgb_wrist"].to_json(),
            },
            "gelsight": {
                "source_color_order": "rgb",
                "canonical_color_order": "rgb",
                "source_shape": gelsight_stream.get("source_shape"),
                "collection_stream": gelsight_stream,
                "compatibility_transforms": gelsight_stream.get("compatibility_transforms", []),
                "preprocess": specs["gelsight"].to_json(),
            },
        },
        "index_path": "canonical_index.jsonl",
        "canonical_episode_path": "canonical_episode.npz",
        "chunks_dir": "chunks",
        "generated_at_wall_time": time.time(),
        "duration_sec": time.time() - started,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return CanonicalPreprocessResult(
        episode_dir=episode_dir,
        output_dir=output_dir,
        manifest_path=manifest_path,
        kept_steps=len(records),
        dropped_steps=dropped,
    )


def write_preprocess1_dataset_manifest(
    preprocess1_root: Path,
    *,
    task_name: str | None = None,
    profile_name: str | None = None,
    raw_run_dir: Path | None = None,
) -> Path:
    preprocess1_root = Path(preprocess1_root)
    episodes_dir = preprocess1_root / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Missing preprocess1 episodes directory: {episodes_dir}")

    entries: list[dict[str, Any]] = []
    total_steps = 0
    dropped_steps = 0
    stream_shapes: dict[str, Any] = {}
    for episode_dir in sorted(path for path in episodes_dir.glob("episode_*") if path.is_dir()):
        manifest_path = episode_dir / "preprocess1_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        kept = int(manifest.get("kept_steps", 0))
        dropped = int(manifest.get("dropped_steps", 0))
        total_steps += kept
        dropped_steps += dropped
        if not stream_shapes:
            stream_shapes = {
                name: {
                    "source_shape": payload.get("source_shape"),
                    "preprocess": payload.get("preprocess"),
                }
                for name, payload in manifest.get("streams", {}).items()
            }
        entries.append(
            {
                "episode_name": episode_dir.name,
                "preprocess1_dir": f"episodes/{episode_dir.name}",
                "preprocess1_manifest": f"episodes/{episode_dir.name}/preprocess1_manifest.json",
                "canonical_episode": f"episodes/{episode_dir.name}/{manifest.get('canonical_episode_path', 'canonical_episode.npz')}",
                "kept_steps": kept,
                "dropped_steps": dropped,
                "start_wall_time": _manifest_first_timestamp(episode_dir),
                "end_wall_time": _manifest_last_timestamp(episode_dir),
            }
        )
    if not entries:
        raise FileNotFoundError(f"No preprocess1 episode manifests found under {episodes_dir}")

    payload = {
        "schema_version": PREPROCESS1_DATASET_SCHEMA_VERSION,
        "task_name": task_name,
        "profile_name": profile_name or _infer_profile_name(preprocess1_root, entries),
        "preprocess1_root": str(preprocess1_root),
        "raw_run_dir": None if raw_run_dir is None else str(raw_run_dir),
        "storage_layout": "centralized",
        "total_episodes": len(entries),
        "total_steps": int(total_steps),
        "total_dropped_steps": int(dropped_steps),
        "streams": stream_shapes,
        "episodes": entries,
        "generated_at_wall_time": time.time(),
    }
    manifest_path = preprocess1_root / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def ensure_preprocess1_episode_metadata(
    preprocess_dir: Path,
    *,
    episode_dir: Path,
    task_name: str | None = None,
    profile_name: str | None = None,
) -> Path:
    preprocess_dir = Path(preprocess_dir)
    episode_dir = Path(episode_dir)
    manifest_path = preprocess_dir / "preprocess1_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing preprocess1 episode manifest: {manifest_path}")
    aligned_path = episode_dir / "aligned_episode.npz"
    if not aligned_path.exists():
        raise FileNotFoundError(f"Missing aligned episode file for preprocess1 metadata upgrade: {aligned_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    canonical_rel = str(manifest.get("canonical_episode_path", "canonical_episode.npz"))
    canonical_path = preprocess_dir / canonical_rel
    if not canonical_path.exists():
        records = load_canonical_records(preprocess_dir)
        aligned_indices = np.asarray([record["aligned_index"] for record in records], dtype=np.int64)
        _write_canonical_episode_npz(
            aligned_path=aligned_path,
            output_path=canonical_path,
            aligned_indices=aligned_indices,
        )
    changed = False
    if manifest.get("schema_version") != PREPROCESS1_EPISODE_SCHEMA_VERSION:
        manifest["schema_version"] = PREPROCESS1_EPISODE_SCHEMA_VERSION
        manifest.setdefault("compatible_schema_versions", ["vt_franka_preprocess1_v1"])
        changed = True
    if manifest.get("canonical_episode_path") != canonical_rel:
        manifest["canonical_episode_path"] = canonical_rel
        changed = True
    if task_name is not None and manifest.get("task_name") != task_name:
        manifest["task_name"] = task_name
        changed = True
    if profile_name is not None and manifest.get("profile_name") != profile_name:
        manifest["profile_name"] = profile_name
        changed = True
    if changed:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return canonical_path


def load_canonical_records(preprocess_dir: Path) -> list[dict[str, Any]]:
    index_path = Path(preprocess_dir) / "canonical_index.jsonl"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing canonical index: {index_path}")
    return [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_canonical_arrays(preprocess_dir: Path, records: list[dict[str, Any]] | None = None) -> dict[str, np.ndarray]:
    preprocess_dir = Path(preprocess_dir)
    records = records or load_canonical_records(preprocess_dir)
    rgb_chunks: list[np.ndarray] = []
    gelsight_chunks: list[np.ndarray] = []
    timestamp_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    for chunk_path in _unique_chunk_paths(records):
        with np.load(preprocess_dir / chunk_path) as chunk:
            rgb_chunks.append(chunk["rgb_wrist"])
            gelsight_chunks.append(chunk["gelsight"])
            timestamp_chunks.append(chunk["timestamps"])
            index_chunks.append(chunk["aligned_indices"])
    return {
        "rgb_wrist": np.concatenate(rgb_chunks, axis=0),
        "gelsight": np.concatenate(gelsight_chunks, axis=0),
        "timestamps": np.concatenate(timestamp_chunks, axis=0),
        "aligned_indices": np.concatenate(index_chunks, axis=0),
    }


def _flush_chunk(
    *,
    chunks_dir: Path,
    records: list[dict[str, Any]],
    chunk_index: int,
    rgb: list[np.ndarray],
    gelsight: list[np.ndarray],
    timestamps: list[float],
    aligned_indices: list[int],
) -> None:
    chunk_name = f"chunk_{chunk_index:06d}.npz"
    chunk_path = chunks_dir / chunk_name
    tmp_path = chunks_dir / f".{chunk_name}.tmp"
    start_index = len(records)
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            rgb_wrist=np.stack(rgb, axis=0).astype(np.uint8),
            gelsight=np.stack(gelsight, axis=0).astype(np.uint8),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            aligned_indices=np.asarray(aligned_indices, dtype=np.int64),
        )
    os.replace(tmp_path, chunk_path)
    for index_in_chunk, (timestamp, aligned_index) in enumerate(zip(timestamps, aligned_indices)):
        records.append(
            {
                "canonical_index": int(start_index + index_in_chunk),
                "aligned_index": int(aligned_index),
                "timestamp": float(timestamp),
                "chunk_path": f"chunks/{chunk_name}",
                "index_in_chunk": int(index_in_chunk),
            }
        )


def _write_canonical_episode_npz(
    *,
    aligned_path: Path,
    output_path: Path,
    aligned_indices: np.ndarray,
) -> None:
    arrays: dict[str, np.ndarray] = {
        "aligned_indices": np.asarray(aligned_indices, dtype=np.int64),
    }
    with np.load(aligned_path, allow_pickle=True) as aligned:
        source_timestamps = np.asarray(aligned["timestamps"], dtype=np.float64)
        arrays["timestamps"] = source_timestamps[aligned_indices]
        for key in CANONICAL_EPISODE_REQUIRED_KEYS:
            if key not in aligned:
                raise KeyError(f"Aligned episode is missing required key for preprocess1 v2: {key}")
            arrays[key] = _select_aligned_array(aligned[key], aligned_indices, key)
        for key in CANONICAL_EPISODE_OPTIONAL_KEYS:
            if key in aligned:
                arrays[key] = _select_aligned_array(aligned[key], aligned_indices, key)
    tmp_path = output_path.with_name(f".{output_path.name}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp_path, output_path)


def _select_aligned_array(array: np.ndarray, aligned_indices: np.ndarray, key: str) -> np.ndarray:
    value = np.asarray(array)
    if len(value) <= int(aligned_indices.max()):
        raise ValueError(f"Aligned array {key!r} has length {len(value)} but needs index {int(aligned_indices.max())}")
    return value[aligned_indices]


def _manifest_first_timestamp(preprocess_dir: Path) -> float | None:
    records = load_canonical_records(preprocess_dir)
    return None if not records else float(records[0]["timestamp"])


def _manifest_last_timestamp(preprocess_dir: Path) -> float | None:
    records = load_canonical_records(preprocess_dir)
    return None if not records else float(records[-1]["timestamp"])


def _infer_profile_name(preprocess1_root: Path, entries: list[dict[str, Any]]) -> str | None:
    first_manifest = entries[0].get("preprocess1_manifest")
    if first_manifest:
        try:
            payload = json.loads(Path(first_manifest).read_text(encoding="utf-8"))
            profile = payload.get("profile_name")
            if profile:
                return str(profile)
        except OSError:
            pass
    return preprocess1_root.name


def _unique_chunk_paths(records: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    paths: list[str] = []
    for record in records:
        path = str(record["chunk_path"])
        if path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def _read_frame_ref_as_rgb(episode_dir: Path, rel_path: str, index_in_chunk: int, *, source_kind: str) -> np.ndarray:
    path = episode_dir / rel_path
    if path.suffix.lower() == ".npz":
        if index_in_chunk < 0:
            raise ValueError(f"{source_kind} chunk frame requires index_in_chunk: {rel_path}")
        with np.load(path) as chunk:
            frames = chunk["frames"]
            if index_in_chunk >= len(frames):
                raise IndexError(f"{source_kind} index {index_in_chunk} out of range for {rel_path}")
            return bgr_to_rgb(np.asarray(frames[index_in_chunk]))
    return read_image_file_as_rgb(path)


def _read_collection_canonical_frame(episode_dir: Path, rel_path: str, index_in_chunk: int) -> np.ndarray:
    if index_in_chunk < 0:
        raise ValueError(f"Collection preprocess1 frame requires index_in_chunk: {rel_path}")
    with np.load(episode_dir / rel_path) as chunk:
        frames = chunk["frames"]
        if index_in_chunk >= len(frames):
            raise IndexError(f"Collection preprocess1 index {index_in_chunk} out of range for {rel_path}")
        return np.asarray(frames[index_in_chunk], dtype=np.uint8)


def _collection_preprocess1_stream_manifests(episode_dir: Path) -> dict[str, dict[str, Any]]:
    manifests: dict[str, dict[str, Any]] = {}
    for stream_name in ("preprocess1_rgb_wrist", "preprocess1_gelsight"):
        path = episode_dir / "streams" / stream_name / "manifest.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        canonical_key = str(payload.get("canonical_key", ""))
        if canonical_key:
            manifests[canonical_key] = payload
    return manifests


def _resolve_output_dir(episode_dir: Path, config: CanonicalPreprocessConfig) -> Path:
    if config.output_root is None:
        return episode_dir / "preprocessed" / config.profile_name
    return Path(config.output_root) / "episodes" / episode_dir.name


class _EpisodeFrameReader:
    def __init__(self, episode_dir: Path, *, max_npz_cache: int = 1) -> None:
        self.episode_dir = Path(episode_dir)
        self.max_npz_cache = max(1, int(max_npz_cache))
        self._npz_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def read_rgb(self, rel_path: str, index_in_chunk: int, *, source_kind: str) -> np.ndarray:
        path = self.episode_dir / rel_path
        if path.suffix.lower() != ".npz":
            return read_image_file_as_rgb(path)
        if index_in_chunk < 0:
            raise ValueError(f"{source_kind} chunk frame requires index_in_chunk: {rel_path}")
        frames = self._load_npz_frames(rel_path)
        if index_in_chunk >= len(frames):
            raise IndexError(f"{source_kind} index {index_in_chunk} out of range for {rel_path}")
        return bgr_to_rgb(np.asarray(frames[index_in_chunk]))

    def _load_npz_frames(self, rel_path: str) -> np.ndarray:
        cached = self._npz_cache.get(rel_path)
        if cached is not None:
            self._npz_cache.move_to_end(rel_path)
            return cached
        with np.load(self.episode_dir / rel_path) as chunk:
            frames = np.asarray(chunk["frames"])
        self._npz_cache[rel_path] = frames
        while len(self._npz_cache) > self.max_npz_cache:
            self._npz_cache.popitem(last=False)
        return frames


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for visuotactile preprocessing") from exc
    return cv2
