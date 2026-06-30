from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...mpd.math import tcp_state_to_pose7d_and_gripper
from .config import VisuotactileModelSpec, get_model_spec
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
    prepared_dataset_dir = Path(prepared_dataset_dir)
    output_root = Path(output_root)
    dataset_manifest = _read_json(prepared_dataset_dir / "dataset_manifest.json")
    if dataset_manifest.get("model") != spec.name:
        raise ValueError(
            f"Prepared dataset model mismatch: manifest has {dataset_manifest.get('model')!r}, requested {spec.name!r}"
        )
    resolved_task_name = task_name or str(dataset_manifest.get("task_name", "task"))
    task_dir = output_root / resolved_task_name
    hdf5_dir = task_dir / "hdf5"
    act_hdf5_dir = task_dir / "act_hdf5"
    if task_dir.exists():
        if not overwrite:
            existing_manifest = task_dir / "backend_dataset_manifest.json"
            if existing_manifest.exists():
                payload = _read_json(existing_manifest)
                return BackendExportResult(
                    backend_dataset_root=output_root,
                    task_dir=task_dir,
                    hdf5_dir=hdf5_dir,
                    act_hdf5_dir=Path(payload.get("act_hdf5_dir", act_hdf5_dir)),
                    num_episodes=int(payload.get("num_episodes", 0)),
                    manifest_path=existing_manifest,
                )
            raise FileExistsError(f"Backend export already exists: {task_dir}")
        shutil.rmtree(task_dir)
    hdf5_dir.mkdir(parents=True, exist_ok=True)
    act_hdf5_dir.mkdir(parents=True, exist_ok=True)

    h5py = _require_h5py()
    entries = [entry for entry in dataset_manifest.get("episodes", []) if entry.get("split") in {"train", "val"}]
    if not entries:
        raise RuntimeError(f"No episode entries in dataset manifest: {prepared_dataset_dir}")
    vital_stats = _BackendStats()
    for export_index, entry in enumerate(entries):
        with np.load(prepared_dataset_dir / entry["file"], allow_pickle=False) as data:
            arrays = {key: np.asarray(data[key]) for key in data.files}
        hdf5_path = hdf5_dir / f"{export_index}.hdf5"
        _write_backend_hdf5(
            h5py,
            hdf5_path,
            arrays,
            spec=spec,
        )
        _link_or_copy_file(hdf5_path, act_hdf5_dir / f"episode_{export_index}.hdf5")
        vital_stats.update(arrays)

    vital_stats_path = act_hdf5_dir / "norm_stats.json"
    vital_stats_path.write_text(
        json.dumps(
            vital_stats.to_vital_metadata(task_name=resolved_task_name, num_episodes=len(entries)),
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": "vt_franka_visuotactile_backend_hdf5_v1",
        "model": spec.name,
        "task_name": resolved_task_name,
        "prepared_dataset_dir": str(prepared_dataset_dir),
        "backend_dataset_root": str(output_root),
        "task_dir": str(task_dir),
        "hdf5_dir": str(hdf5_dir),
        "act_hdf5_dir": str(act_hdf5_dir),
        "num_episodes": len(entries),
        "source_episodes": entries,
        "format": {
            "action_representation": dataset_manifest.get("action_representation", spec.action_representation),
            "model_input": dataset_manifest.get("model_input", {"shape_meta": spec.backend_shape_meta()}),
            "embodiment/ee": "T x 7 pose [x,y,z,qw,qx,qy,qz]",
            "embodiment/joint": "T x 9 synthetic joint vector; last two entries carry gripper scalar",
            "observation/wrist/rgb": "T variable-length JPEG payloads for DP/VISTA loaders",
            "tactile/left_tactile/rgb_marker": "T variable-length JPEG payloads from the single GelSight stream",
            "observations/images/*": "T x H x W x 3 uint8 raw RGB arrays for ACT/ViTAL ACT loaders",
        },
    }
    manifest_path = task_dir / "backend_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return BackendExportResult(
        backend_dataset_root=output_root,
        task_dir=task_dir,
        hdf5_dir=hdf5_dir,
        act_hdf5_dir=act_hdf5_dir,
        num_episodes=len(entries),
        manifest_path=manifest_path,
    )


def _write_backend_hdf5(h5py, path: Path, arrays: dict[str, np.ndarray], *, spec: VisuotactileModelSpec) -> None:
    rgb = np.asarray(arrays["rgb_wrist"], dtype=np.uint8)
    gelsight = np.asarray(arrays["gelsight"], dtype=np.uint8)
    pose7 = _pose7_actions(arrays, spec=spec)
    qpos7 = _pose7_qpos(arrays, spec=spec)
    q_gripper = np.asarray(qpos7[:, 7], dtype=np.float32)
    a_gripper = np.asarray(pose7[:, 7], dtype=np.float32)
    ee = np.asarray(qpos7[:, :7], dtype=np.float32)
    ee_action = np.asarray(pose7[:, :7], dtype=np.float32)
    joint = _synthetic_joint(q_gripper)
    joint_action = _synthetic_joint(a_gripper)
    encoded_rgb = _encode_jpeg_sequence(rgb)
    encoded_gelsight = _encode_jpeg_sequence(gelsight)
    with h5py.File(path, "w") as root:
        root.attrs["sim"] = True
        root.attrs["image_height"] = int(rgb.shape[1])
        root.attrs["image_width"] = int(rgb.shape[2])
        root.attrs["gelsight_height"] = int(gelsight.shape[1])
        root.attrs["gelsight_width"] = int(gelsight.shape[2])
        root.attrs["num_timesteps"] = int(rgb.shape[0])
        root.create_dataset("action", data=pose7.astype(np.float32), compression="gzip", compression_opts=4)
        embodiment = root.create_group("embodiment")
        embodiment.create_dataset("ee", data=ee.astype(np.float32), compression="gzip", compression_opts=4)
        embodiment.create_dataset("joint", data=joint.astype(np.float32), compression="gzip", compression_opts=4)
        embodiment.create_dataset("ee_action", data=ee_action.astype(np.float32), compression="gzip", compression_opts=4)
        embodiment.create_dataset("joint_action", data=joint_action.astype(np.float32), compression="gzip", compression_opts=4)
        obs = root.create_group("observation")
        wrist = obs.create_group("wrist")
        _write_vlen_uint8_dataset(h5py, wrist, "rgb", encoded_rgb)
        head = obs.create_group("head")
        _write_vlen_uint8_dataset(h5py, head, "rgb", encoded_rgb)
        tactile = root.create_group("tactile")
        left = tactile.create_group("left_tactile")
        right = tactile.create_group("right_tactile")
        _write_vlen_uint8_dataset(h5py, left, "rgb_marker", encoded_gelsight)
        _write_vlen_uint8_dataset(h5py, right, "rgb_marker", encoded_gelsight)
        tactile["left_gsmini"] = left
        tactile["right_gsmini"] = right
        obs2 = root.create_group("observations")
        obs2.create_dataset("qpos", data=qpos7.astype(np.float32), compression="gzip", compression_opts=4)
        obs2.create_dataset("ee", data=ee.astype(np.float32), compression="gzip", compression_opts=4)
        images = obs2.create_group("images")
        images.create_dataset("cam_wrist", data=rgb, compression="gzip", compression_opts=4)
        images.create_dataset("cam_high", data=rgb, compression="gzip", compression_opts=4)
        images.create_dataset("tac_left", data=gelsight, compression="gzip", compression_opts=4)
        images.create_dataset("tac_right", data=gelsight, compression="gzip", compression_opts=4)
        images.create_dataset("cam_left_tactile", data=gelsight, compression="gzip", compression_opts=4)
        images.create_dataset("cam_right_tactile", data=gelsight, compression="gzip", compression_opts=4)


class _BackendStats:
    def __init__(self) -> None:
        self.qpos: list[np.ndarray] = []
        self.action: list[np.ndarray] = []
        self.gelsight_sum = np.zeros((3,), dtype=np.float64)
        self.gelsight_sum_sq = np.zeros((3,), dtype=np.float64)
        self.gelsight_count = 0

    def update(self, arrays: dict[str, np.ndarray]) -> None:
        self.qpos.append(_pose7_qpos(arrays, spec=get_model_spec("act_univtac")))
        self.action.append(_pose7_actions(arrays, spec=get_model_spec("act_univtac")))
        gelsight = np.asarray(arrays["gelsight"], dtype=np.float32) / 255.0
        self.gelsight_sum += gelsight.sum(axis=(0, 1, 2), dtype=np.float64)
        self.gelsight_sum_sq += np.square(gelsight, dtype=np.float32).sum(axis=(0, 1, 2), dtype=np.float64)
        self.gelsight_count += int(np.prod(gelsight.shape[:-1]))

    def to_vital_metadata(self, *, task_name: str, num_episodes: int) -> dict[str, Any]:
        qpos = np.concatenate(self.qpos, axis=0).astype(np.float64)
        action = np.concatenate(self.action, axis=0).astype(np.float64)
        qpos_std = np.clip(qpos.std(axis=0), 1e-2, np.inf)
        action_std = np.clip(action.std(axis=0), 1e-2, np.inf)
        if self.gelsight_count <= 0:
            raise RuntimeError("Cannot write ViTAL metadata without GelSight frames")
        gel_mean = self.gelsight_sum / self.gelsight_count
        gel_var = np.maximum(self.gelsight_sum_sq / self.gelsight_count - np.square(gel_mean), 0.0)
        gel_std = np.clip(np.sqrt(gel_var), 1e-2, np.inf)
        return {
            "schema_version": "vt_franka_visuotactile_vital_metadata_v1",
            "task_name": task_name,
            "num_episodes": int(num_episodes),
            "camera": ["cam_wrist"],
            "tactile": ["cam_left_tactile", "cam_right_tactile"],
            "qpos_mean": qpos.mean(axis=0).astype(float).tolist(),
            "qpos_std": qpos_std.astype(float).tolist(),
            "action_mean": action.mean(axis=0).astype(float).tolist(),
            "action_std": action_std.astype(float).tolist(),
            "left_tac_mean": gel_mean.astype(float).tolist(),
            "left_tac_std": gel_std.astype(float).tolist(),
            "right_tac_mean": gel_mean.astype(float).tolist(),
            "right_tac_std": gel_std.astype(float).tolist(),
        }


def _encode_jpeg_sequence(images: np.ndarray) -> list[np.ndarray]:
    cv2 = _require_cv2()
    encoded: list[np.ndarray] = []
    for image in np.asarray(images, dtype=np.uint8):
        ok, payload = cv2.imencode(".jpg", rgb_to_bgr(image), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if not ok:
            raise RuntimeError("OpenCV failed to JPEG-encode a visuotactile frame")
        encoded.append(np.asarray(payload, dtype=np.uint8).reshape(-1))
    return encoded


def _write_vlen_uint8_dataset(h5py, group, name: str, values: list[np.ndarray]) -> None:
    dtype = h5py.vlen_dtype(np.dtype("uint8"))
    dataset = group.create_dataset(name, shape=(len(values),), dtype=dtype)
    for index, value in enumerate(values):
        dataset[index] = np.asarray(value, dtype=np.uint8)


def _pose7_qpos(arrays: dict[str, np.ndarray], *, spec: VisuotactileModelSpec) -> np.ndarray:
    if "qpos_pose7_gripper" in arrays:
        return np.asarray(arrays["qpos_pose7_gripper"], dtype=np.float32)
    values = np.asarray(arrays["qpos_pose10_rot6d_gripper"], dtype=np.float64)
    return np.stack([_pose10_row_to_pose8(row) for row in values], axis=0).astype(np.float32)


def _pose7_actions(arrays: dict[str, np.ndarray], *, spec: VisuotactileModelSpec) -> np.ndarray:
    if "action_pose7_gripper" in arrays:
        return np.asarray(arrays["action_pose7_gripper"], dtype=np.float32)
    values = np.asarray(arrays["action_pose10_rot6d_gripper"], dtype=np.float64)
    return np.stack([_pose10_row_to_pose8(row) for row in values], axis=0).astype(np.float32)


def _pose10_row_to_pose8(row: np.ndarray) -> np.ndarray:
    pose7, gripper = tcp_state_to_pose7d_and_gripper(row)
    return np.concatenate([pose7, np.asarray([gripper], dtype=np.float64)])


def _synthetic_joint(gripper_scalar: np.ndarray) -> np.ndarray:
    gripper = np.asarray(gripper_scalar, dtype=np.float32).reshape(-1)
    joint = np.zeros((len(gripper), 9), dtype=np.float32)
    joint[:, 7] = gripper
    joint[:, 8] = gripper
    return joint


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _link_or_copy_file(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required to export VT Franka prepared datasets to vendor HDF5 backend format. "
            "Install it in the training environment."
        ) from exc
    return h5py


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to export VT Franka backend HDF5 image payloads.") from exc
    return cv2
