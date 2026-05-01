from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ...config import load_workspace_config
from .config import (
    ACTION_CONVENTION_CLOSEDNESS,
    ACTION_CONVENTION_OPEN_FRACTION,
    DEFAULT_DATASET_NAME,
    default_prepared_dataset_dir,
)
from .data import write_prepared_dataset_scaler_values
from .math import finite_difference

SOURCE_ACTION_CONVENTION = ACTION_CONVENTION_CLOSEDNESS
OUTPUT_ACTION_CONVENTION = ACTION_CONVENTION_OPEN_FRACTION


@dataclass(frozen=True)
class SmoothGripperDatasetConfig:
    source_dataset_dir: Path
    output_dataset_dir: Path
    switch_threshold: float = 0.5
    pre_switch_ramp_steps: int = 4
    post_switch_ramp_steps: int = 4
    overwrite: bool = False
    plot_first_demo: bool = True


@dataclass(frozen=True)
class SmoothGripperDatasetResult:
    output_dataset_dir: Path
    manifest_path: Path
    transformed_demos: int
    transformed_transitions: int
    first_demo_plot_path: Path | None
    first_demo_plot_written: bool


def smooth_gripper_dataset(config: SmoothGripperDatasetConfig) -> SmoothGripperDatasetResult:
    source_dir = Path(config.source_dataset_dir)
    output_dir = Path(config.output_dataset_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source prepared dataset does not exist: {source_dir}")
    _validate_config(config)
    if output_dir.resolve() == source_dir.resolve():
        raise ValueError("Output dataset must be different from source dataset")
    source_manifest = _read_manifest(source_dir)
    source_convention = source_manifest.get("action_convention", SOURCE_ACTION_CONVENTION)
    if source_convention != SOURCE_ACTION_CONVENTION:
        raise ValueError(
            f"Expected source action_convention={SOURCE_ACTION_CONVENTION!r}, got {source_convention!r}: {source_dir}"
        )
    if output_dir.exists():
        if not config.overwrite:
            raise FileExistsError(f"Output dataset already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)

    shutil.copytree(source_dir, output_dir)
    _remove_copied_gripper_plots(output_dir)
    dt = float(source_manifest.get("dt", 0.1))
    transformed_demos = 0
    transformed_transitions = 0
    first_demo_plot_path: Path | None = None
    first_demo_plot_payload: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for demo_dir in _iter_demo_dirs(output_dir):
        source_action = _load_array(demo_dir / "action.npz")
        source_agent_pos = _load_array(demo_dir / "agent_pos.npz")
        if source_action.ndim != 2 or source_action.shape[1] != 10:
            raise ValueError(f"Expected action.npz to have shape [T, 10], got {source_action.shape}: {demo_dir}")
        if source_agent_pos.shape != source_action.shape:
            raise ValueError(
                f"agent_pos.npz shape {source_agent_pos.shape} does not match action.npz shape {source_action.shape}: {demo_dir}"
            )

        action = source_action.astype(np.float64).copy()
        agent_pos = source_agent_pos.astype(np.float64).copy()
        source_action_closedness = np.clip(action[:, 9], 0.0, 1.0)
        source_agent_closedness = np.clip(agent_pos[:, 9], 0.0, 1.0)
        binary_open_target = np.where(source_action_closedness >= config.switch_threshold, 0.0, 1.0)
        smooth_open_target = build_transition_ramp(
            binary_open_target,
            switch_threshold=config.switch_threshold,
            pre_switch_ramp_steps=config.pre_switch_ramp_steps,
            post_switch_ramp_steps=config.post_switch_ramp_steps,
        )

        action[:, 9] = smooth_open_target
        agent_pos[:, 9] = 1.0 - source_agent_closedness
        action_vel = finite_difference(action, dt)
        agent_vel = finite_difference(agent_pos, dt)

        _save_array(demo_dir / "action.npz", action)
        _save_array(demo_dir / "agent_pos.npz", agent_pos)
        _save_array(demo_dir / "action_vel.npz", action_vel)
        _save_array(demo_dir / "agent_vel.npz", agent_vel)

        transitions = int(np.count_nonzero(np.diff(binary_open_target) != 0.0))
        transformed_demos += 1
        transformed_transitions += transitions
        _update_demo_manifest(demo_dir, transitions, config)
        if first_demo_plot_payload is None:
            first_demo_plot_payload = (source_action_closedness, binary_open_target, smooth_open_target)
            if config.plot_first_demo:
                first_demo_plot_path = demo_dir / "gripper_open_fraction_transition_ramp.png"

    write_prepared_dataset_scaler_values(output_dir)
    manifest = dict(source_manifest)
    manifest["schema_version"] = "vt_franka_mpd_v1_gripper_open_smooth"
    manifest["source_dataset_dir"] = str(source_dir)
    manifest["output_dir"] = str(output_dir)
    manifest["action_convention"] = OUTPUT_ACTION_CONVENTION
    manifest["agent_pos_convention"] = OUTPUT_ACTION_CONVENTION
    manifest["gripper_label_transform"] = {
        "type": "transition_ramp",
        "source_action_convention": SOURCE_ACTION_CONVENTION,
        "output_action_convention": OUTPUT_ACTION_CONVENTION,
        "open_value": 1.0,
        "close_value": 0.0,
        "switch_threshold": float(config.switch_threshold),
        "pre_switch_ramp_steps": int(config.pre_switch_ramp_steps),
        "post_switch_ramp_steps": int(config.post_switch_ramp_steps),
        "inference_semantics": "open_fraction <= threshold closes; open_fraction >= threshold opens when rising from closed",
    }
    manifest["num_gripper_transitions"] = transformed_transitions
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    first_demo_plot_written = False
    if first_demo_plot_path is not None and first_demo_plot_payload is not None:
        first_demo_plot_written = _write_first_demo_plot(
            first_demo_plot_path,
            *first_demo_plot_payload,
            threshold=config.switch_threshold,
        )

    return SmoothGripperDatasetResult(
        output_dataset_dir=output_dir,
        manifest_path=manifest_path,
        transformed_demos=transformed_demos,
        transformed_transitions=transformed_transitions,
        first_demo_plot_path=first_demo_plot_path,
        first_demo_plot_written=first_demo_plot_written,
    )


def build_transition_ramp(
    binary_open_target: np.ndarray,
    *,
    switch_threshold: float,
    pre_switch_ramp_steps: int,
    post_switch_ramp_steps: int,
) -> np.ndarray:
    target = np.asarray(binary_open_target, dtype=np.float64)
    if target.ndim != 1:
        raise ValueError(f"binary_open_target must be 1D, got {target.shape}")
    if target.size == 0:
        return target.copy()
    output = np.where(target >= switch_threshold, 1.0, 0.0).astype(np.float64)
    transition_indices = np.flatnonzero(output[1:] != output[:-1]) + 1
    for index in transition_indices:
        start_value = float(output[index - 1])
        end_value = float(output[index])
        for offset in range(pre_switch_ramp_steps, 0, -1):
            step = index - offset
            if step < 0:
                continue
            fraction = float(pre_switch_ramp_steps - offset + 1) / float(pre_switch_ramp_steps + 1)
            output[step] = start_value + (switch_threshold - start_value) * fraction
        output[index] = float(switch_threshold)
        for offset in range(1, post_switch_ramp_steps + 1):
            step = index + offset
            if step >= output.size:
                continue
            fraction = float(offset) / float(post_switch_ramp_steps + 1)
            output[step] = switch_threshold + (end_value - switch_threshold) * fraction
    return np.clip(output, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an MPD prepared dataset with smooth gripper open-fraction labels")
    parser.add_argument("--workspace-config", default="robot_workspace/config/workspace.yaml")
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--source-dataset-dir", type=Path, default=None)
    parser.add_argument("--output-dataset-dir", type=Path, default=None)
    parser.add_argument("--source-dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--output-dataset-name", default=None)
    parser.add_argument("--switch-threshold", type=float, default=0.5)
    parser.add_argument("--pre-switch-ramp-steps", type=int, default=4)
    parser.add_argument("--post-switch-ramp-steps", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dataset_dir
    output_dir = args.output_dataset_dir
    if source_dir is None:
        if args.task_name is None:
            raise SystemExit("--task-name is required when --source-dataset-dir is not provided")
        workspace = load_workspace_config(args.workspace_config)
        source_dir = default_prepared_dataset_dir(workspace, args.task_name, args.source_dataset_name)
        if output_dir is None and args.output_dataset_name is not None:
            output_dir = default_prepared_dataset_dir(workspace, args.task_name, args.output_dataset_name)
    if output_dir is None:
        threshold_suffix = int(round(float(args.switch_threshold) * 100.0))
        output_dir = source_dir.parent / f"{source_dir.name}_gripper_open_smooth_t{threshold_suffix:03d}"

    result = smooth_gripper_dataset(
        SmoothGripperDatasetConfig(
            source_dataset_dir=source_dir,
            output_dataset_dir=output_dir,
            switch_threshold=args.switch_threshold,
            pre_switch_ramp_steps=args.pre_switch_ramp_steps,
            post_switch_ramp_steps=args.post_switch_ramp_steps,
            overwrite=args.overwrite,
            plot_first_demo=not args.no_plot,
        )
    )
    print(f"Smoothed MPD dataset: {result.output_dataset_dir}")
    print(f"Manifest: {result.manifest_path}")
    print(f"Transformed demos: {result.transformed_demos}")
    print(f"Gripper transitions: {result.transformed_transitions}")
    if result.first_demo_plot_written and result.first_demo_plot_path is not None:
        print(f"First-demo plot: {result.first_demo_plot_path}")
    elif not args.no_plot:
        print("First-demo plot: not written because matplotlib is not available")


def _validate_config(config: SmoothGripperDatasetConfig) -> None:
    if not 0.0 < float(config.switch_threshold) < 1.0:
        raise ValueError("switch_threshold must be inside (0, 1)")
    if int(config.pre_switch_ramp_steps) < 0:
        raise ValueError("pre_switch_ramp_steps must be >= 0")
    if int(config.post_switch_ramp_steps) < 0:
        raise ValueError("post_switch_ramp_steps must be >= 0")


def _read_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = Path(dataset_dir) / "dataset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared dataset manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_demo_dirs(dataset_dir: Path) -> list[Path]:
    demo_dirs: list[Path] = []
    for split_name in ("train", "val"):
        split_dir = Path(dataset_dir) / split_name
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing prepared dataset split: {split_dir}")
        demo_dirs.extend(sorted(path for path in split_dir.iterdir() if path.is_dir()))
    if not demo_dirs:
        raise FileNotFoundError(f"No demo directories found in prepared dataset: {dataset_dir}")
    return demo_dirs


def _remove_copied_gripper_plots(dataset_dir: Path) -> None:
    for path in Path(dataset_dir).glob("**/*gripper*.png"):
        path.unlink()


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared dataset array: {path}")
    return np.load(path)["arr_0"]


def _save_array(path: Path, values: np.ndarray) -> None:
    np.savez_compressed(path, np.asarray(values, dtype=np.float32))


def _update_demo_manifest(demo_dir: Path, transitions: int, config: SmoothGripperDatasetConfig) -> None:
    path = demo_dir / "dataset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    manifest["action_convention"] = OUTPUT_ACTION_CONVENTION
    manifest["agent_pos_convention"] = OUTPUT_ACTION_CONVENTION
    manifest["num_gripper_transitions"] = int(transitions)
    manifest["gripper_label_transform"] = {
        "type": "transition_ramp",
        "switch_threshold": float(config.switch_threshold),
        "pre_switch_ramp_steps": int(config.pre_switch_ramp_steps),
        "post_switch_ramp_steps": int(config.post_switch_ramp_steps),
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _write_first_demo_plot(
    path: Path,
    source_closedness: np.ndarray,
    binary_open_target: np.ndarray,
    smooth_open_target: np.ndarray,
    *,
    threshold: float,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    steps = np.arange(len(smooth_open_target))
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(steps, source_closedness, label="source closedness", color="tab:red")
    axes[0].set_ylabel("closedness")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.25)
    axes[1].plot(steps, binary_open_target, label="binary open target", color="tab:gray", linestyle="--")
    axes[1].plot(steps, smooth_open_target, label="smooth open target", color="tab:blue")
    axes[1].axhline(threshold, color="tab:orange", linewidth=1.0, linestyle=":", label="switch threshold")
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("open fraction")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


if __name__ == "__main__":
    main()
