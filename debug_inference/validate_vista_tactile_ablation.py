#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


REPO_ROOT = Path(__file__).resolve().parents[1]
for source_root in (REPO_ROOT / "shared" / "src", REPO_ROOT / "robot_workspace" / "src"):
    source_root_str = str(source_root)
    if source_root_str not in sys.path:
        sys.path.insert(0, source_root_str)

from vt_franka_workspace.config import load_workspace_config  # noqa: E402
from vt_franka_workspace.policies.common.visuotactile.mockrun_episode import (  # noqa: E402
    DEFAULT_WORKSPACE_CONFIG,
    _resolve_checkpoint_reference,
    _resolve_inference_config,
    _resolve_policy_config,
    build_observation_window,
    load_episode_steps,
)
from vt_franka_workspace.policies.common.visuotactile.policy import VisuotactilePolicy  # noqa: E402
from vt_franka_workspace.policies.common.visuotactile.runtime import action_row_to_vt_action  # noqa: E402
from vt_franka_workspace.policies.common.visuotactile.vendor_vista_runtime import (  # noqa: E402
    _configure_vista_sampling,
)


DEFAULT_DATASET_DIR = REPO_ROOT / "robot_workspace/data/datasets/pencil_insertion/real_pencil_insertion"
DEFAULT_CHECKPOINT = REPO_ROOT / "robot_workspace/data/checkpoints/pencil_insertion/vista_so3/checkpoints/epoch=209.ckpt"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "debug_inference/tactile_ablation"


@dataclass(frozen=True)
class EpisodeCase:
    episode_id: str
    role: str
    step_index: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate whether VISTA first-step pitch depends on vision/tactile inputs."
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--workspace-config", type=Path, default=DEFAULT_WORKSPACE_CONFIG)
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument("--inference-config", type=Path, default=None)
    parser.add_argument("--model", default="vista_so3")
    parser.add_argument("--task-name", default="pencil_insertion")
    parser.add_argument("--device", default="cuda", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--scheduler", default="ddim", choices=("checkpoint", "ddpm", "ddim"))
    parser.add_argument("--num-inference-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strong-episode", default="episode_0050")
    parser.add_argument("--mild-episode", default="episode_0075")
    parser.add_argument("--step-index", type=int, default=0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset_dir = args.dataset_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace = load_workspace_config(args.workspace_config)
    checkpoint_dir, checkpoint_file = _resolve_checkpoint_reference(args.checkpoint)
    policy_config, model_name, task_name = _resolve_policy_config(
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_file,
        policy_config_path=args.policy_config,
        workspace=workspace,
        model=args.model,
        task_name=args.task_name,
        device=args.device,
    )
    inference_config = _resolve_inference_config(
        inference_config_path=args.inference_config,
        task_name=task_name,
        model_name=model_name,
        obs_horizon=None,
    )
    policy = VisuotactilePolicy.from_config(policy_config, inference_config, workspace)
    policy.ensure_loaded()
    model = getattr(policy.backend, "_model", None)
    torch = getattr(policy.backend, "_torch", None)
    if model is None or torch is None:
        raise RuntimeError("This validation requires the vendor VISTA checkpoint backend.")
    if args.scheduler != "checkpoint":
        _configure_vista_sampling(
            model,
            scheduler_name=args.scheduler,
            num_inference_steps=args.num_inference_steps,
        )

    cases = [
        EpisodeCase(args.strong_episode, "strong_label_pitch_down", args.step_index),
        EpisodeCase(args.mild_episode, "mild_label_pitch_down", args.step_index),
    ]
    case_data = {
        case.episode_id: load_case_inputs(dataset_dir, case, policy, inference_config.obs_horizon)
        for case in cases
    }

    rows: list[dict[str, Any]] = []
    for case in cases:
        peer = cases[1] if case is cases[0] else cases[0]
        base = case_data[case.episode_id]
        peer_inputs = case_data[peer.episode_id]["model_inputs"]
        variants = build_variants(base["model_inputs"], peer_inputs)
        for variant_name, model_inputs in variants.items():
            seed_torch(torch, args.seed)
            with torch.inference_mode():
                raw_chunk = np.asarray(policy.backend.predict_action_chunk(model_inputs), dtype=np.float64)
            decoded_pose7 = decode_pose7_chunk(policy, raw_chunk)
            pred_rpy = pose7_chunk_to_rpy(decoded_pose7)
            gt_rpy = pose7_chunk_to_rpy(base["gt_pose7"])
            pitch_col = 4
            roll_col = 3
            yaw_col = 5
            rows.append(
                {
                    "episode_id": case.episode_id,
                    "role": case.role,
                    "peer_episode_id": peer.episode_id,
                    "step_index": int(case.step_index),
                    "variant": variant_name,
                    "scheduler": args.scheduler,
                    "num_inference_steps": int(getattr(model, "num_inference_steps", -1)),
                    "label_pitch_deg_t0": float(gt_rpy[0, pitch_col]),
                    "label_pitch_deg_t1": float(gt_rpy[min(1, len(gt_rpy) - 1), pitch_col]),
                    "pred_pitch_deg_h0": float(pred_rpy[0, pitch_col]),
                    "pred_pitch_deg_h1": float(pred_rpy[min(1, len(pred_rpy) - 1), pitch_col]),
                    "pred_pitch_deg_h7": float(pred_rpy[min(7, len(pred_rpy) - 1), pitch_col]),
                    "pred_roll_deg_h0": float(pred_rpy[0, roll_col]),
                    "pred_yaw_deg_h0": float(pred_rpy[0, yaw_col]),
                    "pred_pitch_delta_h0_vs_label_t0": float(pred_rpy[0, pitch_col] - gt_rpy[0, pitch_col]),
                    "raw_action_first": raw_chunk[0].astype(float).tolist(),
                    "predicted_xyz_rpy_deg": pred_rpy.astype(float).tolist(),
                    "ground_truth_xyz_rpy_deg": gt_rpy.astype(float).tolist(),
                }
            )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(args.checkpoint),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": None if checkpoint_file is None else str(checkpoint_file),
        "model": model_name,
        "task_name": task_name,
        "obs_horizon": int(inference_config.obs_horizon),
        "cases": [case.__dict__ for case in cases],
        "seed": int(args.seed),
        "scheduler": args.scheduler,
        "num_inference_steps": int(getattr(model, "num_inference_steps", -1)),
    }
    write_json(output_dir / "metadata.json", metadata)
    write_jsonl(output_dir / "results.jsonl", rows)
    write_summary_csv(output_dir / "summary.csv", rows)
    print(json.dumps({"output_dir": str(output_dir), "rows": len(rows)}, indent=2))
    for row in rows:
        print(
            f"{row['episode_id']} {row['variant']}: "
            f"label_t0={row['label_pitch_deg_t0']:.2f} "
            f"pred_h0={row['pred_pitch_deg_h0']:.2f} "
            f"pred_h1={row['pred_pitch_deg_h1']:.2f} "
            f"pred_h7={row['pred_pitch_deg_h7']:.2f}"
        )


def load_case_inputs(
    dataset_dir: Path,
    case: EpisodeCase,
    policy: VisuotactilePolicy,
    obs_horizon: int,
) -> dict[str, Any]:
    episode_dir = dataset_dir / "episodes" / case.episode_id
    steps = load_episode_steps(episode_dir)
    observation_window = build_observation_window(steps, case.step_index, obs_horizon)
    model_inputs = policy.build_model_inputs(observation_window)
    gt_pose7 = []
    for offset in range(policy.model_spec.action_horizon):
        index = min(case.step_index + offset, len(steps) - 1)
        gt_pose7.append(np.asarray(steps[index].action["target_tcp"], dtype=np.float64))
    return {
        "episode_dir": episode_dir,
        "steps": steps,
        "model_inputs": model_inputs,
        "gt_pose7": np.stack(gt_pose7, axis=0),
    }


def build_variants(
    base_inputs: dict[str, np.ndarray],
    peer_inputs: dict[str, np.ndarray],
) -> dict[str, dict[str, np.ndarray]]:
    variants = {"original": copy_inputs(base_inputs)}
    blank = copy_inputs(base_inputs)
    blank["gelsight"] = np.zeros_like(blank["gelsight"])
    variants["blank_tactile"] = blank

    mean = copy_inputs(base_inputs)
    mean_value = np.mean(mean["gelsight"], axis=(0, 1, 2), keepdims=True)
    mean["gelsight"] = np.broadcast_to(mean_value, mean["gelsight"].shape).copy()
    variants["mean_tactile"] = mean

    swapped = copy_inputs(base_inputs)
    swapped["gelsight"] = np.asarray(peer_inputs["gelsight"], dtype=np.float32).copy()
    variants["swap_tactile"] = swapped

    blank_vision = copy_inputs(base_inputs)
    blank_vision["rgb_wrist"] = np.zeros_like(blank_vision["rgb_wrist"])
    variants["blank_vision"] = blank_vision

    mean_vision = copy_inputs(base_inputs)
    mean_vision_value = np.mean(mean_vision["rgb_wrist"], axis=(0, 1, 2), keepdims=True)
    mean_vision["rgb_wrist"] = np.broadcast_to(mean_vision_value, mean_vision["rgb_wrist"].shape).copy()
    variants["mean_vision"] = mean_vision

    swapped_vision = copy_inputs(base_inputs)
    swapped_vision["rgb_wrist"] = np.asarray(peer_inputs["rgb_wrist"], dtype=np.float32).copy()
    variants["swap_vision"] = swapped_vision

    blank_both = copy_inputs(base_inputs)
    blank_both["rgb_wrist"] = np.zeros_like(blank_both["rgb_wrist"])
    blank_both["gelsight"] = np.zeros_like(blank_both["gelsight"])
    variants["blank_vision_and_tactile"] = blank_both
    return variants


def copy_inputs(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value).copy() for key, value in inputs.items()}


def decode_pose7_chunk(policy: VisuotactilePolicy, raw_chunk: np.ndarray) -> np.ndarray:
    actions = [
        action_row_to_vt_action(
            row,
            model_spec=policy.model_spec,
            target_duration_sec=policy.target_duration_sec,
            gripper_open_width_m=policy.gripper_open_width_m,
            gripper_close_threshold=policy.settings.gripper_close_threshold,
        )
        for row in raw_chunk
    ]
    return np.stack([np.asarray(action["target_tcp"], dtype=np.float64) for action in actions], axis=0)


def pose7_chunk_to_rpy(pose7: np.ndarray) -> np.ndarray:
    values = np.asarray(pose7, dtype=np.float64)
    quat_wxyz = values[:, 3:7]
    quat_xyzw = np.stack(
        [quat_wxyz[:, 1], quat_wxyz[:, 2], quat_wxyz[:, 3], quat_wxyz[:, 0]],
        axis=1,
    )
    rpy = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
    return np.concatenate([values[:, :3], rpy], axis=1)


def seed_torch(torch: Any, seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "episode_id",
        "role",
        "variant",
        "label_pitch_deg_t0",
        "label_pitch_deg_t1",
        "pred_pitch_deg_h0",
        "pred_pitch_deg_h1",
        "pred_pitch_deg_h7",
        "pred_pitch_delta_h0_vs_label_t0",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


if __name__ == "__main__":
    main()
