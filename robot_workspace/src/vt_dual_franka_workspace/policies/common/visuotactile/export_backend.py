from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import get_model_spec
from .image_preprocess import rgb_to_bgr


@dataclass(frozen=True)
class BackendExportResult:
    backend_dataset_root: Path
    task_dir: Path
    hdf5_dir: Path
    act_hdf5_dir: Path
    num_episodes: int
    manifest_path: Path


def export_prepared_dataset_for_backend(
    prepared_dataset_dir: str | Path,
    output_root: str | Path,
    *,
    model: str,
    task_name: str | None = None,
    overwrite: bool = False,
) -> BackendExportResult:
    spec = get_model_spec(model)
    if spec.name != "dp_bimanual":
        raise ValueError("VT Dual Franka backend export supports only dp_bimanual")
    prepared_dataset_dir = Path(prepared_dataset_dir)
    output_root = Path(output_root)
    dataset_manifest = _read_json(prepared_dataset_dir / "dataset_manifest.json")
    if dataset_manifest.get("schema_version") != "vt_dual_franka_bimanual_training_dataset_v1":
        raise ValueError(f"Not a bimanual prepared dataset: {prepared_dataset_dir}")
    if dataset_manifest.get("model") != "dp_bimanual":
        raise ValueError("Prepared dataset model must be dp_bimanual")

    resolved_task_name = task_name or str(dataset_manifest.get("task_name") or "bimanual_demo")
    task_dir = output_root / resolved_task_name
    hdf5_dir = task_dir / "hdf5"
    act_hdf5_dir = task_dir / "act_hdf5"
    manifest_path = task_dir / "backend_dataset_manifest.json"
    if task_dir.exists():
        if not overwrite:
            if manifest_path.is_file():
                payload = _read_json(manifest_path)
                return BackendExportResult(
                    backend_dataset_root=output_root,
                    task_dir=task_dir,
                    hdf5_dir=hdf5_dir,
                    act_hdf5_dir=act_hdf5_dir,
                    num_episodes=int(payload.get("num_episodes", 0)),
                    manifest_path=manifest_path,
                )
            raise FileExistsError(task_dir)
        shutil.rmtree(task_dir)
    hdf5_dir.mkdir(parents=True)
    act_hdf5_dir.mkdir(parents=True)

    entries = list(dataset_manifest.get("episodes") or [])
    if not entries:
        raise RuntimeError(f"No bimanual episodes in {prepared_dataset_dir}")
    h5py = _require_h5py()
    for export_index, entry in enumerate(entries):
        with np.load(prepared_dataset_dir / str(entry["file"]), allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        output_path = hdf5_dir / f"{export_index}.hdf5"
        _write_bimanual_hdf5(h5py, output_path, arrays)
        _link_or_copy_file(output_path, act_hdf5_dir / f"episode_{export_index}.hdf5")

    normalizer_source = prepared_dataset_dir / "normalizer_stats.json"
    if normalizer_source.is_file():
        shutil.copy2(normalizer_source, task_dir / "normalizer_stats.json")
    manifest = {
        "schema_version": "vt_dual_franka_bimanual_backend_hdf5_v1",
        "model": "dp_bimanual",
        "task_name": resolved_task_name,
        "prepared_dataset_dir": str(prepared_dataset_dir),
        "backend_dataset_root": str(output_root),
        "hdf5_dir": str(hdf5_dir),
        "num_episodes": len(entries),
        "action_dim": 20,
        "qpos_dim": 20,
        "action_provenance": "future_commanded_action",
        "arm_order": ["left", "right"],
        "source_episodes": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return BackendExportResult(
        backend_dataset_root=output_root,
        task_dir=task_dir,
        hdf5_dir=hdf5_dir,
        act_hdf5_dir=act_hdf5_dir,
        num_episodes=len(entries),
        manifest_path=manifest_path,
    )


def _write_bimanual_hdf5(
    h5py,
    path: Path,
    arrays: dict[str, np.ndarray],
) -> None:
    required = {
        "rgb_wrist_left",
        "rgb_wrist_right",
        "tactile_left",
        "tactile_right",
        "qpos20",
        "action20",
    }
    missing = sorted(required - set(arrays))
    if missing:
        raise KeyError(f"Bimanual prepared episode is missing arrays: {missing}")
    lengths = {name: int(np.asarray(arrays[name]).shape[0]) for name in required}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Bimanual temporal length mismatch: {lengths}")
    steps = next(iter(lengths.values()))
    if np.asarray(arrays["qpos20"]).shape[1:] != (20,):
        raise ValueError("qpos20 must have shape [T,20]")
    if np.asarray(arrays["action20"]).shape[1:] != (20,):
        raise ValueError("action20 must have shape [T,20]")

    encoded = {
        name: _encode_jpeg_sequence(np.asarray(arrays[name], dtype=np.uint8))
        for name in (
            "rgb_wrist_left",
            "rgb_wrist_right",
            "tactile_left",
            "tactile_right",
        )
    }
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = False
        root.attrs["num_timesteps"] = steps
        root.attrs["schema_version"] = "vt_dual_franka_bimanual_hdf5_v1"
        root.create_dataset(
            "action",
            data=np.asarray(arrays["action20"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
        observations = root.create_group("observations")
        observations.create_dataset(
            "qpos",
            data=np.asarray(arrays["qpos20"], dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
        observation = root.create_group("observation")
        left_wrist = observation.create_group("wrist")
        right_wrist = observation.create_group("right_wrist")
        _write_vlen_uint8_dataset(h5py, left_wrist, "rgb", encoded["rgb_wrist_left"])
        _write_vlen_uint8_dataset(h5py, right_wrist, "rgb", encoded["rgb_wrist_right"])
        tactile = root.create_group("tactile")
        left_tactile = tactile.create_group("left_tactile")
        right_tactile = tactile.create_group("right_tactile")
        _write_vlen_uint8_dataset(h5py, left_tactile, "rgb_marker", encoded["tactile_left"])
        _write_vlen_uint8_dataset(h5py, right_tactile, "rgb_marker", encoded["tactile_right"])


def _encode_jpeg_sequence(images: np.ndarray) -> list[np.ndarray]:
    cv2 = _require_cv2()
    encoded: list[np.ndarray] = []
    for image in images:
        ok, payload = cv2.imencode(
            ".jpg",
            rgb_to_bgr(image),
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        )
        if not ok:
            raise RuntimeError("OpenCV failed to encode a bimanual image")
        encoded.append(np.asarray(payload, dtype=np.uint8).reshape(-1))
    return encoded


def _write_vlen_uint8_dataset(
    h5py,
    group,
    name: str,
    values: list[np.ndarray],
) -> None:
    dataset = group.create_dataset(
        name,
        shape=(len(values),),
        dtype=h5py.vlen_dtype(np.dtype("uint8")),
    )
    for index, value in enumerate(values):
        dataset[index] = np.asarray(value, dtype=np.uint8)


def _link_or_copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for bimanual backend export") from exc
    return h5py


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for bimanual backend export") from exc
    return cv2
