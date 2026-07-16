#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vt_dual_franka_workspace.config import load_inference_config, load_policy_config, load_workspace_config
from vt_dual_franka_workspace.policies.registry import resolve_policy
from vt_dual_franka_workspace.policies.visuotactile.config import VisuotactilePolicySettings, get_model_spec
from vt_dual_franka_workspace.policies.visuotactile.runtime import action_row_to_vt_action


def _load_episode(dataset_dir: Path, episode_index: int | None, split: str) -> tuple[Path, dict[str, np.ndarray]]:
    manifest_path = dataset_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing prepared dataset manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    episodes = [entry for entry in manifest.get("episodes", []) if entry.get("split") == split]
    if not episodes:
        episodes = list(manifest.get("episodes", []))
    if not episodes:
        raise RuntimeError(f"no episodes listed in {manifest_path}")
    index = 0 if episode_index is None else int(episode_index)
    if index < 0 or index >= len(episodes):
        raise IndexError(f"episode index {index} out of range for {len(episodes)} {split} episodes")
    episode_path = dataset_dir / episodes[index]["file"]
    with np.load(episode_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    return episode_path, arrays


def _observation_from_arrays(arrays: dict[str, np.ndarray], index: int, *, open_width_m: float) -> dict[str, Any]:
    pose7_gripper = np.asarray(arrays["qpos_pose7_gripper"][index], dtype=np.float64)
    pose7 = pose7_gripper[:7]
    closedness = float(pose7_gripper[7])
    gripper_width = float(open_width_m * (1.0 - closedness))
    return {
        "proprioception": {
            "controller_state": {
                "tcp_pose": pose7.astype(float).tolist(),
                "gripper_width": gripper_width,
            }
        },
        "images": {
            "wrist": {
                "image": np.asarray(arrays["rgb_wrist"][index]),
                "metadata": {},
                "captured_wall_time": float(arrays["timestamps"][index]) if "timestamps" in arrays else 0.0,
            }
        },
        "tactile": {
            "gelsight_frame": {
                "image": np.asarray(arrays["gelsight"][index]),
                "metadata": {},
                "captured_wall_time": float(arrays["timestamps"][index]) if "timestamps" in arrays else 0.0,
            }
        },
    }


def _model_inputs_from_prepared_arrays(
    arrays: dict[str, np.ndarray],
    *,
    start_index: int,
    obs_horizon: int,
    model: str,
) -> dict[str, np.ndarray]:
    spec = get_model_spec(model)
    rgb = []
    gelsight = []
    qpos = []
    qpos_key = "qpos_pose10_rot6d_gripper" if spec.action_representation == "pose10_rot6d_gripper" else "qpos_pose7_gripper"
    for offset in range(int(obs_horizon)):
        source_index = max(0, start_index - int(obs_horizon) + 1 + offset)
        rgb.append(np.asarray(arrays["rgb_wrist"][source_index], dtype=np.float32) / 255.0)
        gelsight.append(np.asarray(arrays["gelsight"][source_index], dtype=np.float32) / 255.0)
        qpos.append(np.asarray(arrays[qpos_key][source_index], dtype=np.float32))
    return {
        "rgb_wrist": np.stack(rgb, axis=0),
        "gelsight": np.stack(gelsight, axis=0),
        "qpos": np.stack(qpos, axis=0),
    }


def _summarize_actions(actions: list[dict[str, Any]], arrays: dict[str, np.ndarray], start_index: int, model: str) -> dict[str, Any]:
    spec = get_model_spec(model)
    pred_rows = np.asarray([action["metadata"]["visuotactile_action_row"] for action in actions], dtype=np.float64)
    target_key = "action_pose10_rot6d_gripper" if spec.action_representation == "pose10_rot6d_gripper" else "action_pose7_gripper"
    end_index = min(start_index + len(pred_rows), len(arrays[target_key]))
    gt_rows = np.asarray(arrays[target_key][start_index:end_index], dtype=np.float64)
    comparable = pred_rows[: len(gt_rows)]
    abs_delta = np.abs(comparable - gt_rows) if len(gt_rows) else np.empty((0, pred_rows.shape[1]))
    return {
        "pred_shape": list(pred_rows.shape),
        "pred_min": pred_rows.min(axis=0).round(6).tolist(),
        "pred_max": pred_rows.max(axis=0).round(6).tolist(),
        "pred_mean": pred_rows.mean(axis=0).round(6).tolist(),
        "gt_shape": list(gt_rows.shape),
        "gt_min": gt_rows.min(axis=0).round(6).tolist() if len(gt_rows) else [],
        "gt_max": gt_rows.max(axis=0).round(6).tolist() if len(gt_rows) else [],
        "mean_abs_delta_vs_gt": abs_delta.mean(axis=0).round(6).tolist() if len(gt_rows) else [],
        "first_action": actions[0] if actions else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline sanity-check a VT_Franka visuotactile checkpoint.")
    parser.add_argument("--workspace-config", default="robot_workspace/config/workspace.yaml")
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--inference-config", default="robot_workspace/config/inference/usb_insertion_visuotactile.yaml")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    workspace = load_workspace_config(args.workspace_config)
    policy_config = load_policy_config(args.policy_config)
    inference = load_inference_config(args.inference_config)
    if args.checkpoint_dir is not None:
        policy_config.checkpoint_path = args.checkpoint_dir
    if args.device is not None:
        payload = dict(policy_config.config)
        payload["device"] = args.device
        policy_config.config = payload

    settings = VisuotactilePolicySettings.model_validate(policy_config.config)
    spec = get_model_spec(settings.model)
    policy = resolve_policy(policy_config, inference, workspace)
    episode_path, arrays = _load_episode(args.dataset_dir, args.episode_index, args.split)
    if args.start_index < 0 or args.start_index >= len(arrays["rgb_wrist"]):
        raise IndexError(f"start index {args.start_index} out of range for {episode_path}")

    open_width = float(settings.gripper_open_width_m or workspace.teleop.max_gripper_width)
    horizon = int(getattr(policy, "backend").obs_horizon)
    observations = []
    first = args.start_index
    for offset in range(horizon):
        source_index = max(0, first - horizon + 1 + offset)
        observations.append(_observation_from_arrays(arrays, source_index, open_width_m=open_width))

    print(f"loading checkpoint: {policy_config.checkpoint_path}")
    policy.ensure_loaded()
    print("loaded")
    if hasattr(policy, "backend"):
        inputs = _model_inputs_from_prepared_arrays(
            arrays,
            start_index=args.start_index,
            obs_horizon=int(getattr(policy.backend, "obs_horizon")),
            model=spec.name,
        )
        action_chunk = policy.backend.predict_action_chunk(inputs)
        actions = [
            action_row_to_vt_action(
                row,
                model_spec=policy.model_spec,
                target_duration_sec=policy.target_duration_sec,
                gripper_open_width_m=policy.gripper_open_width_m,
                gripper_close_threshold=policy.settings.gripper_close_threshold,
            )
            for row in action_chunk
        ]
    else:
        actions = policy.predict(observations)
    summary = _summarize_actions(actions, arrays, args.start_index, spec.name)
    print(json.dumps({"episode": str(episode_path), "model": spec.name, **summary}, indent=2))
    policy.close()


if __name__ == "__main__":
    main()
