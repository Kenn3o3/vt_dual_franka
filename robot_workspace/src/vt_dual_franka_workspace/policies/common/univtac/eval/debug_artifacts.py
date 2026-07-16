from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover - debug plotting is optional at import time
    plt = None

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _jsonable(value: Any) -> Any:
    if torch is not None and isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.generic,)):
        return value.item()
    return value


def _to_display_image(image: Any) -> np.ndarray:
    if torch is not None and isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    else:
        image = np.asarray(image)

    if image.ndim == 3 and image.shape[0] in {1, 3, 4} and image.shape[-1] not in {1, 3, 4}:
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)

    if image.ndim != 3 or image.shape[2] not in {3, 4}:
        raise ValueError(f"Unsupported debug image shape: {image.shape}")

    if image.shape[2] == 4:
        image = image[:, :, :3]

    if image.dtype != np.uint8:
        image = image.astype(np.float32)
        finite_mask = np.isfinite(image)
        if not finite_mask.any():
            image = np.zeros_like(image, dtype=np.uint8)
        elif image.min() >= 0.0 and image.max() <= 1.0:
            image = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            lo = float(np.nanmin(image))
            hi = float(np.nanmax(image))
            if hi > lo:
                image = ((image - lo) / (hi - lo) * 255.0).clip(0.0, 255.0).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

    return np.ascontiguousarray(image)


def _fit_to_tile(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    src_h, src_w = image.shape[:2]
    scale = min(target_w / max(src_w, 1), target_h / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((target_h, target_w, 3), 24, dtype=np.uint8)
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _make_labeled_tile(label: str, image: Any, tile_size: tuple[int, int]) -> np.ndarray:
    tile = _fit_to_tile(_to_display_image(image), tile_size)
    cv2.putText(
        tile,
        label,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        lineType=cv2.LINE_AA,
    )
    return tile


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2)
        f.write("\n")


def _quat_wxyz_to_rpy_deg(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    was_1d = quat.ndim == 1
    quat = np.atleast_2d(quat)
    if quat.shape[1] != 4:
        raise ValueError(f"Expected quaternion shape (*, 4), got {quat.shape}")

    quat_norm = np.linalg.norm(quat, axis=1, keepdims=True)
    quat_norm = np.clip(quat_norm, 1e-12, None)
    quat = quat / quat_norm

    w = quat[:, 0]
    x = quat[:, 1]
    y = quat[:, 2]
    z = quat[:, 3]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    rpy_deg = np.rad2deg(np.stack([roll, pitch, yaw], axis=1)).astype(np.float32)
    if was_1d:
        return rpy_deg[0]
    return rpy_deg


class RolloutDebugWriter:
    def __init__(self, *, enabled: bool, root_dir: str | Path | None, policy_name: str, variant: str) -> None:
        self.enabled = bool(enabled)
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.policy_name = policy_name
        self.variant = variant
        self.max_inference_montages = 3
        self.rollout_dir: Path | None = None
        self.inference_input_dir: Path | None = None
        self.plot_dir: Path | None = None
        self._current_seed: int | None = None
        self._rows: list[dict[str, Any]] = []
        self._metadata: dict[str, Any] = {}

    def start_rollout(self, *, seed: int, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled or self.root_dir is None:
            return
        self._current_seed = int(seed)
        self.rollout_dir = self.root_dir / f"seed_{seed}"
        self.inference_input_dir = self.rollout_dir / "inference_inputs"
        self.plot_dir = self.rollout_dir / "plots"
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.inference_input_dir.mkdir(parents=True, exist_ok=True)
        self._rows = []
        self._metadata = {
            "policy_name": self.policy_name,
            "variant": self.variant,
            "seed": int(seed),
        }
        if metadata:
            self._metadata.update(_jsonable(metadata))
        _write_json(self.rollout_dir / "rollout_metadata.json", self._metadata)

    def should_save_inference_montage(self, *, inference_index: int) -> bool:
        return (
            self.enabled
            and self.inference_input_dir is not None
            and inference_index < self.max_inference_montages
        )

    def save_inference_montage(self, *, inference_index: int, tiles: Mapping[str, Any]) -> None:
        if (
            not self.should_save_inference_montage(inference_index=inference_index)
            or not tiles
        ):
            return

        tile_size = (256, 256)
        rendered = [_make_labeled_tile(label, image, tile_size) for label, image in tiles.items()]
        cols = min(4, max(1, len(rendered)))
        rows = int(np.ceil(len(rendered) / cols))
        canvas = np.full((rows * tile_size[1], cols * tile_size[0], 3), 16, dtype=np.uint8)
        for idx, tile in enumerate(rendered):
            row = idx // cols
            col = idx % cols
            y0 = row * tile_size[1]
            x0 = col * tile_size[0]
            canvas[y0 : y0 + tile_size[1], x0 : x0 + tile_size[0]] = tile
        cv2.imwrite(str(self.inference_input_dir / f"{inference_index:03d}.png"), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    def record_step(self, payload: dict[str, Any]) -> None:
        if not self.enabled or self.rollout_dir is None:
            return
        self._rows.append(_jsonable(payload))

    def finalize_rollout(self, *, result: str | None = None) -> None:
        if not self.enabled or self.rollout_dir is None:
            return

        metadata = dict(self._metadata)
        if result is not None:
            metadata["result"] = result
        _write_json(self.rollout_dir / "rollout_metadata.json", metadata)
        _write_json(self.rollout_dir / "rollout.json", self._rows)
        self._write_csv()
        self._write_physx_joint_command_files()
        self._write_plots()

        self.rollout_dir = None
        self.inference_input_dir = None
        self.plot_dir = None
        self._current_seed = None
        self._rows = []
        self._metadata = {}

    def close(self) -> None:
        if self.rollout_dir is not None:
            self.finalize_rollout()

    def _write_csv(self) -> None:
        if self.rollout_dir is None:
            return

        columns = [
            "sim_step",
            "did_inference",
            "inference_index",
            "action_buffer_index",
            "execution_horizon",
            "policy_latency_ms",
            "proprio_x",
            "proprio_y",
            "proprio_z",
            "proprio_qw",
            "proprio_qx",
            "proprio_qy",
            "proprio_qz",
            "proprio_gripper",
            "action_x",
            "action_y",
            "action_z",
            "action_qw",
            "action_qx",
            "action_qy",
            "action_qz",
            "action_gripper",
            "n_inference_actions",
            "inference_actions_json",
        ]

        with open(self.rollout_dir / "rollout.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for row in self._rows:
                proprio = row.get("proprio", {})
                final_action = row.get("final_action", {})
                inference_actions = row.get("inference_actions")
                flat = {
                    "sim_step": row.get("sim_step"),
                    "did_inference": row.get("did_inference"),
                    "inference_index": row.get("inference_index"),
                    "action_buffer_index": row.get("action_buffer_index"),
                    "execution_horizon": row.get("execution_horizon"),
                    "policy_latency_ms": row.get("policy_latency_ms"),
                    "proprio_x": _safe_index(proprio.get("ee_pos_xyz"), 0),
                    "proprio_y": _safe_index(proprio.get("ee_pos_xyz"), 1),
                    "proprio_z": _safe_index(proprio.get("ee_pos_xyz"), 2),
                    "proprio_qw": _safe_index(proprio.get("ee_quat_wxyz"), 0),
                    "proprio_qx": _safe_index(proprio.get("ee_quat_wxyz"), 1),
                    "proprio_qy": _safe_index(proprio.get("ee_quat_wxyz"), 2),
                    "proprio_qz": _safe_index(proprio.get("ee_quat_wxyz"), 3),
                    "proprio_gripper": proprio.get("gripper"),
                    "action_x": _safe_index(final_action.get("pos_xyz"), 0),
                    "action_y": _safe_index(final_action.get("pos_xyz"), 1),
                    "action_z": _safe_index(final_action.get("pos_xyz"), 2),
                    "action_qw": _safe_index(final_action.get("quat_wxyz"), 0),
                    "action_qx": _safe_index(final_action.get("quat_wxyz"), 1),
                    "action_qy": _safe_index(final_action.get("quat_wxyz"), 2),
                    "action_qz": _safe_index(final_action.get("quat_wxyz"), 3),
                    "action_gripper": final_action.get("gripper"),
                    "n_inference_actions": 0 if inference_actions is None else len(inference_actions),
                    "inference_actions_json": "" if inference_actions is None else json.dumps(inference_actions),
                }
                writer.writerow(flat)

    def _collect_physx_joint_writes(self) -> list[dict[str, Any]]:
        writes: list[dict[str, Any]] = []
        for row_index, row in enumerate(self._rows):
            row_writes = row.get("physx_joint_writes") or []
            for write_index_in_action, write in enumerate(row_writes):
                flattened = dict(write)
                flattened.setdefault("sim_step", row.get("sim_step"))
                flattened["rollout_step_index"] = row_index
                flattened["write_index_in_action"] = write_index_in_action
                flattened["write_index"] = len(writes)
                writes.append(flattened)
        return writes

    @staticmethod
    def _joint_names_from_writes(writes: list[dict[str, Any]]) -> list[str]:
        for write in writes:
            joint_names = write.get("joint_names") or []
            if joint_names:
                return [str(name) for name in joint_names]
        return []

    def _write_physx_joint_command_files(self) -> None:
        if self.rollout_dir is None:
            return

        writes = self._collect_physx_joint_writes()
        if not writes:
            return

        _write_json(self.rollout_dir / "joint_commands.json", writes)

        joint_names = self._joint_names_from_writes(writes)
        columns = [
            "write_index",
            "write_index_in_action",
            "rollout_step_index",
            "sim_step",
            "take_action_cnt",
            "source",
            "joint_pos_target_json",
        ]
        columns.extend(f"joint_target__{name}" for name in joint_names)

        with open(self.rollout_dir / "joint_commands.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            for write in writes:
                joint_pos_target = write.get("joint_pos_target") or []
                row = {
                    "write_index": write.get("write_index"),
                    "write_index_in_action": write.get("write_index_in_action"),
                    "rollout_step_index": write.get("rollout_step_index"),
                    "sim_step": write.get("sim_step"),
                    "take_action_cnt": write.get("take_action_cnt"),
                    "source": write.get("source"),
                    "joint_pos_target_json": json.dumps(joint_pos_target),
                }
                for joint_idx, joint_name in enumerate(joint_names):
                    row[f"joint_target__{joint_name}"] = _safe_index(joint_pos_target, joint_idx)
                writer.writerow(row)

    def _write_plots(self) -> None:
        if self.plot_dir is None or not self._rows:
            return
        if plt is None:
            _write_json(
                self.plot_dir / "plot_error.json",
                {"error": "matplotlib is not installed; rollout plots were skipped."},
            )
            return

        sim_steps = np.array([int(row["sim_step"]) for row in self._rows], dtype=np.int32)
        action_xyz = np.array([row["final_action"]["pos_xyz"] for row in self._rows], dtype=np.float32)
        proprio_xyz = np.array([row["proprio"]["ee_pos_xyz"] for row in self._rows], dtype=np.float32)
        action_quat = np.array([row["final_action"]["quat_wxyz"] for row in self._rows], dtype=np.float32)
        proprio_quat = np.array([row["proprio"]["ee_quat_wxyz"] for row in self._rows], dtype=np.float32)
        action_rpy_deg = _quat_wxyz_to_rpy_deg(action_quat)
        proprio_rpy_deg = _quat_wxyz_to_rpy_deg(proprio_quat)
        did_inference_steps = [int(row["sim_step"]) for row in self._rows if row.get("did_inference")]

        self._plot_series(
            sim_steps,
            action_xyz[:, 0],
            primary_label="action x",
            overlay_y=proprio_xyz[:, 0],
            overlay_label="proprio ee x",
            ylabel="action x (m)",
            output_path=self.plot_dir / "action_x.png",
            inference_steps=did_inference_steps,
        )
        self._plot_series(
            sim_steps,
            action_xyz[:, 1],
            primary_label="action y",
            overlay_y=proprio_xyz[:, 1],
            overlay_label="proprio ee y",
            ylabel="action y (m)",
            output_path=self.plot_dir / "action_y.png",
            inference_steps=did_inference_steps,
        )
        self._plot_series(
            sim_steps,
            action_xyz[:, 2],
            primary_label="action z",
            overlay_y=proprio_xyz[:, 2],
            overlay_label="proprio ee z",
            ylabel="action z (m)",
            output_path=self.plot_dir / "action_z.png",
            inference_steps=did_inference_steps,
        )

        delta_xyz_mm = np.zeros(len(self._rows), dtype=np.float32)
        if len(action_xyz) > 1:
            delta_xyz_mm[1:] = np.linalg.norm(np.diff(action_xyz, axis=0), axis=1) * 1000.0
        self._plot_series(
            sim_steps,
            delta_xyz_mm,
            ylabel="delta xyz L2 (mm)",
            output_path=self.plot_dir / "delta_action.png",
            inference_steps=did_inference_steps,
        )
        self._plot_series(
            sim_steps,
            action_rpy_deg[:, 0],
            primary_label="action roll",
            overlay_y=proprio_rpy_deg[:, 0],
            overlay_label="proprio ee roll",
            ylabel="roll (deg)",
            output_path=self.plot_dir / "action_roll.png",
            inference_steps=did_inference_steps,
        )
        self._plot_series(
            sim_steps,
            action_rpy_deg[:, 1],
            primary_label="action pitch",
            overlay_y=proprio_rpy_deg[:, 1],
            overlay_label="proprio ee pitch",
            ylabel="pitch (deg)",
            output_path=self.plot_dir / "action_pitch.png",
            inference_steps=did_inference_steps,
        )
        self._plot_series(
            sim_steps,
            action_rpy_deg[:, 2],
            primary_label="action yaw",
            overlay_y=proprio_rpy_deg[:, 2],
            overlay_label="proprio ee yaw",
            ylabel="yaw (deg)",
            output_path=self.plot_dir / "action_yaw.png",
            inference_steps=did_inference_steps,
        )

        self._write_physx_joint_plots()

    def _write_physx_joint_plots(self) -> None:
        if self.plot_dir is None:
            return

        writes = self._collect_physx_joint_writes()
        joint_names = self._joint_names_from_writes(writes)
        if not writes or not joint_names:
            return

        valid_writes = [
            write for write in writes if len(write.get("joint_pos_target") or []) == len(joint_names)
        ]
        if not valid_writes:
            return

        write_indices = np.arange(len(valid_writes), dtype=np.int32)
        joint_targets = np.array(
            [write["joint_pos_target"] for write in valid_writes],
            dtype=np.float32,
        )
        joint_delta = np.zeros_like(joint_targets)
        if len(joint_targets) > 1:
            joint_delta[1:] = np.diff(joint_targets, axis=0)

        action_boundaries: list[int] = []
        last_action_cnt = None
        for idx, write in enumerate(valid_writes):
            action_cnt = write.get("take_action_cnt")
            if action_cnt != last_action_cnt:
                action_boundaries.append(idx)
                last_action_cnt = action_cnt

        self._plot_joint_grid(
            write_indices,
            joint_targets,
            joint_names=joint_names,
            ylabel="joint target",
            output_path=self.plot_dir / "physx_joint_targets_grid.png",
            boundary_indices=action_boundaries,
        )
        self._plot_joint_grid(
            write_indices,
            joint_delta,
            joint_names=joint_names,
            ylabel="delta from prev write",
            output_path=self.plot_dir / "physx_joint_delta_grid.png",
            boundary_indices=action_boundaries,
        )

    @staticmethod
    def _plot_series(
        x: np.ndarray,
        y: np.ndarray,
        *,
        primary_label: str | None = None,
        overlay_y: np.ndarray | None = None,
        overlay_label: str | None = None,
        ylabel: str,
        output_path: Path,
        inference_steps: list[int],
    ) -> None:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for step in inference_steps:
            ax.axvline(step, color="#d0d0d0", linewidth=1.0, zorder=0)
        ax.plot(
            x,
            y,
            color="#1f77b4",
            linewidth=1.8,
            marker="o",
            markersize=3,
            label=primary_label,
        )
        if overlay_y is not None:
            ax.plot(
                x,
                overlay_y,
                color="#ff7f0e",
                linewidth=1.6,
                alpha=0.9,
                label=overlay_label,
            )
        ax.set_xlabel("simulation step")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if primary_label is not None or overlay_label is not None:
            ax.legend()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

    @staticmethod
    def _plot_joint_grid(
        x: np.ndarray,
        values: np.ndarray,
        *,
        joint_names: list[str],
        ylabel: str,
        output_path: Path,
        boundary_indices: list[int],
    ) -> None:
        if values.ndim != 2 or values.shape[1] != len(joint_names):
            return

        n_joints = len(joint_names)
        n_cols = min(3, max(1, n_joints))
        n_rows = int(np.ceil(n_joints / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 2.8 * n_rows), sharex=True)
        axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

        for ax in axes.flat:
            ax.set_visible(False)

        for joint_idx, joint_name in enumerate(joint_names):
            ax = axes.flat[joint_idx]
            ax.set_visible(True)
            for boundary in boundary_indices[1:]:
                ax.axvline(boundary, color="#d0d0d0", linewidth=1.0, zorder=0)
            ax.plot(
                x,
                values[:, joint_idx],
                color="#1f77b4",
                linewidth=1.6,
                marker="o",
                markersize=2.8,
            )
            ax.set_title(joint_name)
            ax.grid(True, alpha=0.3)
            if joint_idx % n_cols == 0:
                ax.set_ylabel(ylabel)
            if joint_idx >= (n_rows - 1) * n_cols:
                ax.set_xlabel("physx write index")

        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def _safe_index(values: list[Any] | tuple[Any, ...] | None, index: int) -> Any:
    if values is None or len(values) <= index:
        return None
    return values[index]
