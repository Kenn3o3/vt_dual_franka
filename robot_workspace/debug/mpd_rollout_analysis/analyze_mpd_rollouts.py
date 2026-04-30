#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
ROBOT_WORKSPACE = REPO_ROOT / "robot_workspace"
SRC_DIR = ROBOT_WORKSPACE / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vt_franka_workspace.mpd.config import (  # noqa: E402
    build_mpd_inference_config,
    build_mpd_policy_config,
)
from vt_franka_workspace.mpd.policies import build_mpd_policy  # noqa: E402
from vt_franka_workspace.rollout.action_math import pose7d_and_gripper_to_tcp_state  # noqa: E402
from vt_franka_workspace.settings import WorkspaceSettings  # noqa: E402
from vt_franka_shared.config import load_yaml_model  # noqa: E402


TASK = "put_cup_on_plate"
DATASET_NAME = "vt_franka_mpd_v1"
POLICIES = ("dp_state", "sfp_state", "mpd_state", "motif_state")
POLICY_ALG = {"dp_state": "dp", "sfp_state": "sfp", "mpd_state": "mpd", "motif_state": "motif"}


@dataclass
class Paths:
    collect_dir: Path
    prepared_dir: Path
    train_dir: Path
    eval_dir: Path
    out_dir: Path
    plots_dir: Path
    tables_dir: Path
    artifacts_dir: Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze VT Franka MPD rollout failures.")
    parser.add_argument("--workspace-config", default=str(ROBOT_WORKSPACE / "config" / "workspace.yaml"))
    parser.add_argument("--output-dir", default=str(ROBOT_WORKSPACE / "debug" / "mpd_rollout_analysis"))
    parser.add_argument("--max-offline-windows-per-demo", type=int, default=8)
    parser.add_argument("--max-offline-demos", type=int, default=10)
    parser.add_argument("--skip-offline-inference", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir).resolve()
    paths = Paths(
        collect_dir=ROBOT_WORKSPACE / "data" / "collect" / TASK,
        prepared_dir=ROBOT_WORKSPACE / "data" / "prepared" / "mpd" / TASK / DATASET_NAME,
        train_dir=ROBOT_WORKSPACE / "data" / "train" / "mpd",
        eval_dir=ROBOT_WORKSPACE / "data" / "eval",
        out_dir=out_dir,
        plots_dir=out_dir / "plots",
        tables_dir=out_dir / "tables",
        artifacts_dir=out_dir / "artifacts",
    )
    for directory in (paths.out_dir, paths.plots_dir, paths.tables_dir, paths.artifacts_dir):
        directory.mkdir(parents=True, exist_ok=True)

    dataset_rows, dataset_findings = analyze_dataset(paths)
    collect_rows, collect_findings = analyze_collect(paths)
    rollout_rows, rollout_findings = analyze_rollouts(paths)
    checkpoint_rows, checkpoint_findings = analyze_checkpoints(paths)
    offline_rows: list[dict[str, Any]] = []
    offline_findings: list[str] = []
    if args.skip_offline_inference:
        offline_findings.append("Offline inference skipped by --skip-offline-inference.")
    else:
        offline_rows, offline_findings = analyze_offline_inference(
            paths,
            workspace_config=Path(args.workspace_config),
            max_demos=args.max_offline_demos,
            max_windows_per_demo=args.max_offline_windows_per_demo,
        )

    write_csv(paths.tables_dir / "dataset_summary.csv", dataset_rows)
    write_csv(paths.tables_dir / "collect_alignment_summary.csv", collect_rows)
    write_csv(paths.tables_dir / "rollout_summary.csv", rollout_rows)
    write_csv(paths.tables_dir / "checkpoint_config_summary.csv", checkpoint_rows)
    write_csv(paths.tables_dir / "offline_inference_summary.csv", offline_rows)

    report = build_report(
        paths=paths,
        dataset_rows=dataset_rows,
        collect_rows=collect_rows,
        rollout_rows=rollout_rows,
        checkpoint_rows=checkpoint_rows,
        offline_rows=offline_rows,
        findings={
            "Dataset": dataset_findings,
            "Collect Alignment": collect_findings,
            "Rollout": rollout_findings,
            "Checkpoints": checkpoint_findings,
            "Offline Inference": offline_findings,
        },
    )
    (paths.out_dir / "report.md").write_text(report, encoding="utf-8")
    print(paths.out_dir / "report.md")
    return 0


def analyze_dataset(paths: Paths) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    findings: list[str] = []
    manifest = read_json(paths.prepared_dir / "dataset_manifest.json")
    findings.append(
        "Prepared manifest: dt=%s, action_alignment=%s, velocity_convention=%s."
        % (manifest.get("dt"), manifest.get("action_alignment"), manifest.get("velocity_convention"))
    )
    demo_dirs = sorted(paths.prepared_dir.glob("*/demo_*"))
    all_agent_xyz: list[np.ndarray] = []
    all_action_xyz: list[np.ndarray] = []
    all_action_gripper: list[np.ndarray] = []
    all_lengths: list[int] = []
    for demo_dir in demo_dirs:
        split = demo_dir.parent.name
        arrays = load_demo_arrays(demo_dir)
        if not arrays:
            continue
        agent = arrays["agent_pos"]
        action = arrays["action"]
        agent_vel = arrays.get("agent_vel")
        action_vel = arrays.get("action_vel")
        all_agent_xyz.append(agent[:, :3])
        all_action_xyz.append(action[:, :3])
        all_action_gripper.append(action[:, 9])
        all_lengths.append(len(action))
        close_idx = first_threshold_crossing(action[:, 9], 0.5, above=True)
        open_after = first_threshold_crossing(action[:, 9], 0.5, above=False, start=(close_idx + 1 if close_idx is not None else 0))
        row = {
            "split": split,
            "demo": demo_dir.name,
            "steps": len(action),
            "agent_x_min": safe_min(agent[:, 0]),
            "agent_x_max": safe_max(agent[:, 0]),
            "agent_y_min": safe_min(agent[:, 1]),
            "agent_y_max": safe_max(agent[:, 1]),
            "agent_z_min": safe_min(agent[:, 2]),
            "agent_z_max": safe_max(agent[:, 2]),
            "action_x_min": safe_min(action[:, 0]),
            "action_x_max": safe_max(action[:, 0]),
            "action_y_min": safe_min(action[:, 1]),
            "action_y_max": safe_max(action[:, 1]),
            "action_z_min": safe_min(action[:, 2]),
            "action_z_max": safe_max(action[:, 2]),
            "action_gripper_max": safe_max(action[:, 9]),
            "first_close_step": none_to_empty(close_idx),
            "first_open_after_close_step": none_to_empty(open_after),
            "agent_vel_abs_max": safe_max(np.abs(agent_vel)) if agent_vel is not None else "",
            "action_vel_abs_max": safe_max(np.abs(action_vel)) if action_vel is not None else "",
        }
        rows.append(row)
    if all_lengths:
        agent_xyz = np.concatenate(all_agent_xyz, axis=0)
        action_xyz = np.concatenate(all_action_xyz, axis=0)
        action_gripper = np.concatenate(all_action_gripper, axis=0)
        findings.append(
            "Prepared demos: count=%d, length mean=%.1f, min=%d, max=%d."
            % (len(all_lengths), float(np.mean(all_lengths)), min(all_lengths), max(all_lengths))
        )
        findings.append(
            "Action xyz range: x[%.4f, %.4f], y[%.4f, %.4f], z[%.4f, %.4f], gripper max=%.3f."
            % (
                safe_min(action_xyz[:, 0]),
                safe_max(action_xyz[:, 0]),
                safe_min(action_xyz[:, 1]),
                safe_max(action_xyz[:, 1]),
                safe_min(action_xyz[:, 2]),
                safe_max(action_xyz[:, 2]),
                safe_max(action_gripper),
            )
        )
        findings.append(
            "Agent xyz range: x[%.4f, %.4f], y[%.4f, %.4f], z[%.4f, %.4f]."
            % (
                safe_min(agent_xyz[:, 0]),
                safe_max(agent_xyz[:, 0]),
                safe_min(agent_xyz[:, 1]),
                safe_max(agent_xyz[:, 1]),
                safe_min(agent_xyz[:, 2]),
                safe_max(agent_xyz[:, 2]),
            )
        )
    plot_dataset_overview(paths, demo_dirs[: min(16, len(demo_dirs))])
    return rows, findings


def analyze_collect(paths: Paths) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    findings: list[str] = []
    episode_dirs = sorted((paths.collect_dir / "episodes").glob("episode_*"))
    same_step_errors = []
    next_step_errors = []
    lead_values = []
    command_measured_z = []
    for episode_dir in episode_dirs:
        aligned_path = episode_dir / "aligned_episode.npz"
        manifest_path = episode_dir / "aligned_episode_manifest.json"
        if not aligned_path.exists():
            continue
        data = np.load(aligned_path, allow_pickle=True)
        robot = np.asarray(data["robot_tcp_pose"], dtype=float)
        target = np.asarray(data["teleop_target_tcp"], dtype=float)
        gripper_width = np.asarray(data["gripper_width"], dtype=float)
        gripper_closed = np.asarray(data["teleop_gripper_closed"], dtype=bool)
        lead = np.asarray(data.get("teleop_action_lead_sec", np.full(len(robot), np.nan)), dtype=float)
        same = np.linalg.norm(target[:, :3] - robot[:, :3], axis=1)
        nxt = np.linalg.norm(target[:-1, :3] - robot[1:, :3], axis=1) if len(robot) > 1 else np.array([])
        same_step_errors.extend(same.tolist())
        next_step_errors.extend(nxt.tolist())
        lead_values.extend(lead[np.isfinite(lead)].tolist())
        command_measured_z.extend((target[:, 2] - robot[:, 2]).tolist())
        manifest = read_json(manifest_path)
        close_idx = first_threshold_crossing(gripper_closed.astype(float), 0.5, above=True)
        width_close_idx = first_threshold_crossing(1.0 - gripper_width / 0.078, 0.5, above=True)
        rows.append(
            {
                "episode": episode_dir.name,
                "steps": len(robot),
                "manifest_target_hz": manifest.get("target_hz", ""),
                "alignment_mode": manifest.get("alignment_mode", ""),
                "dropped_without_future_action": manifest.get("dropped_steps_without_future_action", ""),
                "same_step_xyz_error_mm_mean": float(np.mean(same) * 1000.0),
                "same_step_xyz_error_mm_p95": percentile(same * 1000.0, 95),
                "next_step_xyz_error_mm_mean": float(np.mean(nxt) * 1000.0) if len(nxt) else "",
                "teleop_lead_ms_mean": float(np.nanmean(lead) * 1000.0),
                "teleop_lead_ms_p95": percentile(lead[np.isfinite(lead)] * 1000.0, 95),
                "target_minus_robot_z_mm_mean": float(np.mean(target[:, 2] - robot[:, 2]) * 1000.0),
                "target_z_min": safe_min(target[:, 2]),
                "robot_z_min": safe_min(robot[:, 2]),
                "first_command_close_step": none_to_empty(close_idx),
                "first_measured_closed_step": none_to_empty(width_close_idx),
                "measured_close_delay_steps": (
                    width_close_idx - close_idx if close_idx is not None and width_close_idx is not None else ""
                ),
            }
        )
    if same_step_errors:
        findings.append(
            "Collect command-vs-measured same-step xyz error mean=%.1fmm, p95=%.1fmm."
            % (float(np.mean(same_step_errors) * 1000.0), percentile(np.asarray(same_step_errors) * 1000.0, 95))
        )
    if next_step_errors:
        findings.append(
            "Collect command-vs-next-measured xyz error mean=%.1fmm, p95=%.1fmm."
            % (float(np.mean(next_step_errors) * 1000.0), percentile(np.asarray(next_step_errors) * 1000.0, 95))
        )
    if lead_values:
        findings.append(
            "Teleop action lead mean=%.1fms, p95=%.1fms."
            % (float(np.mean(lead_values) * 1000.0), percentile(np.asarray(lead_values) * 1000.0, 95))
        )
    if command_measured_z:
        findings.append(
            "Collect teleop target z - measured z mean=%.1fmm, min=%.1fmm, max=%.1fmm."
            % (
                float(np.mean(command_measured_z) * 1000.0),
                float(np.min(command_measured_z) * 1000.0),
                float(np.max(command_measured_z) * 1000.0),
            )
        )
    return rows, findings


def analyze_rollouts(paths: Paths) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    findings: list[str] = []
    policy_step_paths = sorted(paths.eval_dir.glob("**/streams/policy_steps.jsonl"))
    policy_step_paths += sorted(paths.train_dir.glob("**/streams/policy_steps.jsonl"))
    if not policy_step_paths:
        findings.append("No rollout policy_steps.jsonl found under data/eval or data/train/mpd.")
        return rows, findings
    for path in policy_step_paths:
        try:
            records = read_jsonl(path)
            if not records:
                continue
            summary = summarize_rollout(path, records)
            rows.append(summary)
            plot_rollout(path, records, paths.plots_dir, paths.artifacts_dir)
        except Exception as exc:
            rows.append({"policy_steps": str(path), "error": repr(exc)})
    if rows:
        for row in rows:
            name = row.get("run", Path(str(row.get("policy_steps", ""))).parts[-5] if row.get("policy_steps") else "")
            findings.append(
                "%s: policy=%s, device=%s, steps=%s, z raw/action/obs min=%s/%s/%s, close score max=%s, close commands=%s, clamps=%s."
                % (
                    name,
                    row.get("policy", ""),
                    row.get("policy_device_values", ""),
                    row.get("steps", ""),
                    fmt(row.get("raw_z_min")),
                    fmt(row.get("target_z_min")),
                    fmt(row.get("observed_z_min")),
                    fmt(row.get("gripper_score_max")),
                    row.get("gripper_closed_command_count", ""),
                    row.get("workspace_clamped_count", ""),
                )
            )
    return rows, findings


def summarize_rollout(path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    obs_pose, target_pose, raw_pose, limited_pose = [], [], [], []
    gripper_scores, candidate_max_scores = [], []
    close_commands = []
    widths = []
    devices, sources = [], []
    clamp_count = 0
    translation_limited_count = 0
    waited_count = 0
    inference_ms = []
    loop_ms = []
    for record in records:
        action = record.get("action", {}) or {}
        debug = action.get("debug", {}) or {}
        obs = ((record.get("observation", {}) or {}).get("controller_state", {}) or {})
        append_pose(obs_pose, obs.get("tcp_pose"))
        widths.append(float_or_nan(obs.get("gripper_width")))
        append_pose(target_pose, action.get("target_tcp"))
        append_pose(raw_pose, debug.get("raw_target_tcp"))
        append_pose(limited_pose, debug.get("step_limited_target_tcp"))
        if "gripper_score" in debug:
            gripper_scores.append(float_or_nan(debug.get("gripper_score")))
        cands = debug.get("candidate_gripper_scores")
        if isinstance(cands, list) and cands:
            candidate_max_scores.append(float(np.nanmax(np.asarray(cands, dtype=float))))
        close_commands.append(bool(action.get("gripper_closed") is True))
        if debug.get("workspace_clamped") is True:
            clamp_count += 1
        if debug.get("translation_step_limited") is True:
            translation_limited_count += 1
        if debug.get("waited_for_inference") is True:
            waited_count += 1
        if debug.get("policy_device") is not None:
            devices.append(str(debug.get("policy_device")))
        if debug.get("source") is not None:
            sources.append(str(debug.get("source")))
        if debug.get("completed_inference_ms") is not None:
            inference_ms.append(float_or_nan(debug.get("completed_inference_ms")))
        timing = record.get("timing", {}) or {}
        if timing.get("loop_duration_sec") is not None:
            loop_ms.append(float_or_nan(timing.get("loop_duration_sec")) * 1000.0)
    obs_pose_arr = to_pose_array(obs_pose)
    target_pose_arr = to_pose_array(target_pose)
    raw_pose_arr = to_pose_array(raw_pose)
    limited_pose_arr = to_pose_array(limited_pose)
    run = infer_run_name(path)
    policy = infer_policy_name(run)
    first_close = first_true(close_commands)
    close_transitions = count_rising_edges(close_commands)
    first_close_target = target_pose_arr[first_close] if first_close is not None and len(target_pose_arr) > first_close else np.full(7, np.nan)
    first_close_obs = obs_pose_arr[first_close] if first_close is not None and len(obs_pose_arr) > first_close else np.full(7, np.nan)
    row = {
        "policy_steps": str(path),
        "run": run,
        "policy": policy,
        "steps": len(records),
        "policy_device_values": "|".join(sorted(set(devices))),
        "policy_device_counts": "|".join(f"{key}:{devices.count(key)}" for key in sorted(set(devices))),
        "source_values": "|".join(sorted(set(sources))),
        "source_counts": "|".join(f"{key}:{sources.count(key)}" for key in sorted(set(sources))),
        "observed_x_min": pose_min(obs_pose_arr, 0),
        "observed_x_max": pose_max(obs_pose_arr, 0),
        "observed_y_min": pose_min(obs_pose_arr, 1),
        "observed_y_max": pose_max(obs_pose_arr, 1),
        "observed_z_min": pose_min(obs_pose_arr, 2),
        "observed_z_max": pose_max(obs_pose_arr, 2),
        "target_x_min": pose_min(target_pose_arr, 0),
        "target_x_max": pose_max(target_pose_arr, 0),
        "target_y_min": pose_min(target_pose_arr, 1),
        "target_y_max": pose_max(target_pose_arr, 1),
        "target_z_min": pose_min(target_pose_arr, 2),
        "target_z_max": pose_max(target_pose_arr, 2),
        "raw_x_min": pose_min(raw_pose_arr, 0),
        "raw_x_max": pose_max(raw_pose_arr, 0),
        "raw_y_min": pose_min(raw_pose_arr, 1),
        "raw_y_max": pose_max(raw_pose_arr, 1),
        "raw_z_min": pose_min(raw_pose_arr, 2),
        "raw_z_max": pose_max(raw_pose_arr, 2),
        "limited_z_min": pose_min(limited_pose_arr, 2),
        "limited_z_max": pose_max(limited_pose_arr, 2),
        "workspace_clamped_count": clamp_count,
        "translation_step_limited_count": translation_limited_count,
        "waited_for_inference_count": waited_count,
        "gripper_score_max": nanmax_or_empty(gripper_scores),
        "candidate_gripper_score_max": nanmax_or_empty(candidate_max_scores),
        "gripper_closed_command_count": int(sum(close_commands)),
        "gripper_close_transition_count": close_transitions,
        "first_gripper_closed_step": none_to_empty(first_true(close_commands)),
        "first_close_target_x": float(first_close_target[0]) if np.isfinite(first_close_target[0]) else "",
        "first_close_target_y": float(first_close_target[1]) if np.isfinite(first_close_target[1]) else "",
        "first_close_target_z": float(first_close_target[2]) if np.isfinite(first_close_target[2]) else "",
        "first_close_observed_x": float(first_close_obs[0]) if np.isfinite(first_close_obs[0]) else "",
        "first_close_observed_y": float(first_close_obs[1]) if np.isfinite(first_close_obs[1]) else "",
        "first_close_observed_z": float(first_close_obs[2]) if np.isfinite(first_close_obs[2]) else "",
        "observed_gripper_width_min": nanmin_or_empty(widths),
        "inference_ms_nonzero_mean": mean_positive(inference_ms),
        "inference_ms_max": nanmax_or_empty(inference_ms),
        "loop_ms_mean": nanmean_or_empty(loop_ms),
        "loop_ms_max": nanmax_or_empty(loop_ms),
    }
    return row


def analyze_checkpoints(paths: Paths) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    findings: list[str] = []
    for policy in POLICIES:
        run_dir = paths.train_dir / POLICY_ALG[policy] / policy
        cfg_path = run_dir / "resolved_config.yaml"
        row: dict[str, Any] = {"policy": policy, "run_dir": str(run_dir)}
        if not cfg_path.exists():
            row["error"] = "missing resolved_config.yaml"
            rows.append(row)
            continue
        cfg = read_yaml(cfg_path)
        process = (((cfg.get("agent_config") or {}).get("process_batch_config")) or {})
        row.update(
            {
                "algorithm_dir": POLICY_ALG[policy],
                "method_name": nested_get(cfg, ["agent_config", "method_name"], ""),
                "root_method_name": cfg.get("method_name", ""),
                "epochs": cfg.get("epochs", ""),
                "batch_size": nested_get(cfg, ["data_loader_config", "batch_size"], ""),
                "t_obs": cfg.get("t_obs", ""),
                "t_pred": cfg.get("t_pred", ""),
                "t_act": cfg.get("t_act", ""),
                "predict_past": cfg.get("predict_past", ""),
                "device": cfg.get("device", ""),
                "initial_values_come_from_action_data": process.get("initial_values_come_from_action_data", ""),
                "initial_position_keys": "|".join(process.get("initial_position_keys", []) or []),
                "initial_velocity_keys": "|".join(process.get("initial_velocity_keys", []) or []),
                "has_best_model": (run_dir / "best_model.pth").exists(),
                "has_scaler_values": (run_dir / "scaler_values.npz").exists(),
                "has_dataset_manifest": (run_dir / "dataset_manifest.json").exists(),
                "has_train_log": (run_dir / "train.log").exists(),
            }
        )
        log_stats = parse_train_log(run_dir / "train.log")
        row.update(log_stats)
        rows.append(row)
    for row in rows:
        findings.append(
            "%s: method=%s, epochs=%s, batch=%s, t_obs/t_pred/t_act=%s/%s/%s, predict_past=%s, initial_keys=%s/%s."
            % (
                row.get("policy"),
                row.get("method_name") or row.get("root_method_name"),
                row.get("epochs"),
                row.get("batch_size"),
                row.get("t_obs"),
                row.get("t_pred"),
                row.get("t_act"),
                row.get("predict_past"),
                row.get("initial_position_keys"),
                row.get("initial_velocity_keys"),
            )
        )
    return rows, findings


def analyze_offline_inference(paths: Paths, *, workspace_config: Path, max_demos: int, max_windows_per_demo: int) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    findings: list[str] = []
    try:
        import torch

        if not torch.cuda.is_available():
            findings.append("Offline inference skipped: torch.cuda.is_available() is false.")
            return rows, findings
    except Exception as exc:
        findings.append(f"Offline inference skipped: failed to import torch: {exc!r}.")
        return rows, findings

    try:
        settings = load_yaml_model(workspace_config, WorkspaceSettings)
    except Exception:
        settings = WorkspaceSettings()
    demo_dirs = sorted((paths.prepared_dir / "val").glob("demo_*"))[:max_demos]
    if not demo_dirs:
        demo_dirs = sorted((paths.prepared_dir / "train").glob("demo_*"))[:max_demos]
    for policy in POLICIES:
        checkpoint = paths.train_dir / POLICY_ALG[policy] / policy / "best_model.pth"
        if not checkpoint.exists():
            rows.append({"policy": policy, "error": "missing checkpoint", "checkpoint": str(checkpoint)})
            continue
        try:
            policy_config = build_mpd_policy_config(
                settings,
                workspace_root=ROBOT_WORKSPACE,
                task_name=TASK,
                policy_name=policy,
                checkpoint_path=checkpoint.resolve(),
                device="cuda",
            )
            inference_config = build_mpd_inference_config(settings, control_hz=10.0, replan_interval_steps=3)
            realtime_policy = build_mpd_policy(policy_config, inference_config)
            raw_policy = realtime_policy.policy
        except Exception as exc:
            rows.append({"policy": policy, "error": f"load_failed: {exc!r}", "traceback": traceback.format_exc(limit=5)})
            findings.append(f"{policy}: offline inference load failed: {exc!r}.")
            continue
        per_policy_errors = []
        per_policy_z_errors = []
        per_policy_gripper_errors = []
        try:
            for demo_dir in demo_dirs:
                arrays = load_demo_arrays(demo_dir)
                if not arrays:
                    continue
                agent = arrays["agent_pos"]
                action = arrays["action"]
                starts = sample_start_indices(len(action), raw_policy.t_obs, raw_policy.t_pred, max_windows_per_demo)
                for start in starts:
                    obs_window = []
                    for idx in range(start, start + raw_policy.t_obs):
                        obs_window.append(observation_from_tcp_state(agent[idx], executed_action=action[idx]))
                    pred = np.asarray(raw_policy.infer_action_chunk(obs_window), dtype=float)
                    gt_start = start + raw_policy.t_obs - 1
                    gt = action[gt_start : gt_start + len(pred)]
                    if len(gt) != len(pred):
                        continue
                    err = pred - gt
                    xyz_err = np.linalg.norm(err[:, :3], axis=1)
                    z_err = err[:, 2]
                    gripper_err = np.abs(err[:, 9])
                    per_policy_errors.extend(xyz_err.tolist())
                    per_policy_z_errors.extend(z_err.tolist())
                    per_policy_gripper_errors.extend(gripper_err.tolist())
                    rows.append(
                        {
                            "policy": policy,
                            "demo": demo_dir.name,
                            "start_step": start,
                            "horizon": len(pred),
                            "xyz_error_mm_mean": float(np.mean(xyz_err) * 1000.0),
                            "xyz_error_mm_final": float(xyz_err[-1] * 1000.0),
                            "z_error_mm_mean": float(np.mean(z_err) * 1000.0),
                            "z_error_mm_min": float(np.min(z_err) * 1000.0),
                            "gripper_abs_error_mean": float(np.mean(gripper_err)),
                            "pred_gripper_max": float(np.max(pred[:, 9])),
                            "gt_gripper_max": float(np.max(gt[:, 9])),
                        }
                    )
                    save_offline_chunk_plot(paths.plots_dir, policy, demo_dir.name, start, pred, gt)
        except Exception as exc:
            rows.append({"policy": policy, "error": f"infer_failed: {exc!r}", "traceback": traceback.format_exc(limit=5)})
            findings.append(f"{policy}: offline inference failed during prediction: {exc!r}.")
        finally:
            close = getattr(realtime_policy, "close", None)
            if callable(close):
                close()
        if per_policy_errors:
            findings.append(
                "%s offline teacher-forced: xyz mean=%.1fmm p95=%.1fmm, z bias mean=%.1fmm, gripper abs mean=%.3f."
                % (
                    policy,
                    float(np.mean(per_policy_errors) * 1000.0),
                    percentile(np.asarray(per_policy_errors) * 1000.0, 95),
                    float(np.mean(per_policy_z_errors) * 1000.0),
                    float(np.mean(per_policy_gripper_errors)),
                )
            )
    return rows, findings


def plot_dataset_overview(paths: Paths, demo_dirs: list[Path]) -> None:
    if not demo_dirs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_xy, ax_z, ax_g = axes
    for demo_dir in demo_dirs:
        arrays = load_demo_arrays(demo_dir)
        if not arrays:
            continue
        agent = arrays["agent_pos"]
        action = arrays["action"]
        t = np.arange(len(action)) * 0.1
        ax_xy.plot(agent[:, 0], agent[:, 1], color="tab:blue", alpha=0.28, linewidth=1)
        ax_xy.plot(action[:, 0], action[:, 1], color="tab:orange", alpha=0.28, linewidth=1)
        ax_z.plot(t, agent[:, 2], color="tab:blue", alpha=0.25, linewidth=1)
        ax_z.plot(t, action[:, 2], color="tab:orange", alpha=0.25, linewidth=1)
        ax_g.plot(t, action[:, 9], color="tab:red", alpha=0.25, linewidth=1)
    ax_xy.set_title("Prepared XY: measured agent_pos (blue) vs action label (orange)")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.axis("equal")
    ax_z.set_title("Prepared z over time")
    ax_z.set_xlabel("time [s]")
    ax_z.set_ylabel("z [m]")
    ax_g.set_title("Prepared action gripper closedness")
    ax_g.set_xlabel("time [s]")
    ax_g.set_ylabel("closedness")
    fig.tight_layout()
    fig.savefig(paths.plots_dir / "dataset_prepared_overview.png", dpi=160)
    plt.close(fig)


def plot_rollout(path: Path, records: list[dict[str, Any]], plots_dir: Path, artifacts_dir: Path) -> None:
    obs, target, raw, limited = [], [], [], []
    gripper_scores, gripper_closed, widths = [], [], []
    clamp, step_limited = [], []
    for record in records:
        action = record.get("action", {}) or {}
        debug = action.get("debug", {}) or {}
        obs_state = ((record.get("observation", {}) or {}).get("controller_state", {}) or {})
        append_pose(obs, obs_state.get("tcp_pose"))
        append_pose(target, action.get("target_tcp"))
        append_pose(raw, debug.get("raw_target_tcp"))
        append_pose(limited, debug.get("step_limited_target_tcp"))
        gripper_scores.append(float_or_nan(debug.get("gripper_score")))
        gripper_closed.append(1.0 if action.get("gripper_closed") is True else 0.0)
        widths.append(float_or_nan(obs_state.get("gripper_width")))
        clamp.append(1.0 if debug.get("workspace_clamped") is True else 0.0)
        step_limited.append(1.0 if debug.get("translation_step_limited") is True else 0.0)
    obs_arr, target_arr, raw_arr, limited_arr = map(to_pose_array, (obs, target, raw, limited))
    run = sanitize_filename(infer_run_name(path))
    np.savez_compressed(
        artifacts_dir / f"{run}_trajectory_arrays.npz",
        observed_tcp=obs_arr,
        target_tcp=target_arr,
        raw_target_tcp=raw_arr,
        step_limited_target_tcp=limited_arr,
        gripper_score=np.asarray(gripper_scores, dtype=float),
        gripper_closed_command=np.asarray(gripper_closed, dtype=float),
        gripper_width=np.asarray(widths, dtype=float),
        workspace_clamped=np.asarray(clamp, dtype=float),
        translation_step_limited=np.asarray(step_limited, dtype=float),
    )
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes[0, 0]
    plot_xy(ax, obs_arr, "observed", "black")
    plot_xy(ax, target_arr, "target", "tab:green")
    plot_xy(ax, raw_arr, "raw", "tab:red")
    plot_xy(ax, limited_arr, "limited", "tab:orange")
    ax.set_title(f"{run}: XY")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    t = np.arange(len(records)) * 0.1
    plot_dim(ax, t, obs_arr, 2, "observed", "black")
    plot_dim(ax, t, target_arr, 2, "target", "tab:green")
    plot_dim(ax, t, raw_arr, 2, "raw", "tab:red")
    plot_dim(ax, t, limited_arr, 2, "limited", "tab:orange")
    ax.set_title("Z over time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("z [m]")
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    ax.plot(t, gripper_scores, label="gripper score", color="tab:red")
    ax.plot(t, gripper_closed, label="closed command", color="tab:purple")
    ax.plot(t, np.asarray(widths) / 0.078, label="width/open_width", color="tab:blue")
    ax.set_title("Gripper")
    ax.set_xlabel("time [s]")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, clamp, label="workspace clamp", color="tab:orange")
    ax.plot(t, step_limited, label="translation step limited", color="tab:red")
    ax.set_title("Postprocess events")
    ax.set_xlabel("time [s]")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / f"{run}_rollout_decomposition.png", dpi=160)
    plt.close(fig)


def save_offline_chunk_plot(plots_dir: Path, policy: str, demo: str, start: int, pred: np.ndarray, gt: np.ndarray) -> None:
    if start != 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    h = np.arange(len(pred))
    axes[0].plot(gt[:, 0], gt[:, 1], label="gt", color="black")
    axes[0].plot(pred[:, 0], pred[:, 1], label="pred", color="tab:red")
    axes[0].set_title("XY chunk")
    axes[0].axis("equal")
    axes[0].legend()
    axes[1].plot(h, gt[:, 2], label="gt", color="black")
    axes[1].plot(h, pred[:, 2], label="pred", color="tab:red")
    axes[1].set_title("Z chunk")
    axes[1].legend()
    axes[2].plot(h, gt[:, 9], label="gt", color="black")
    axes[2].plot(h, pred[:, 9], label="pred", color="tab:red")
    axes[2].set_title("Gripper chunk")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(plots_dir / f"offline_{policy}_{demo}_start{start:03d}.png", dpi=160)
    plt.close(fig)


def build_report(
    *,
    paths: Paths,
    dataset_rows: list[dict[str, Any]],
    collect_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
    offline_rows: list[dict[str, Any]],
    findings: dict[str, list[str]],
) -> str:
    lines = [
        "# MPD Rollout Analysis",
        "",
        "Generated by `robot_workspace/debug/mpd_rollout_analysis/analyze_mpd_rollouts.py`.",
        "",
        "## High-Signal Findings",
        "",
    ]
    lines.extend(build_diagnosis(dataset_rows, collect_rows, rollout_rows, offline_rows))
    lines.append("")
    for section, items in findings.items():
        lines.append(f"### {section}")
        if not items:
            lines.append("- No findings.")
        else:
            for item in items:
                lines.append(f"- {item}")
        lines.append("")
    lines.extend(
        [
            "## Evidence Tables",
            "",
            f"- Dataset summary: `{rel(paths.tables_dir / 'dataset_summary.csv')}` ({len(dataset_rows)} rows)",
            f"- Collect alignment summary: `{rel(paths.tables_dir / 'collect_alignment_summary.csv')}` ({len(collect_rows)} rows)",
            f"- Rollout summary: `{rel(paths.tables_dir / 'rollout_summary.csv')}` ({len(rollout_rows)} rows)",
            f"- Checkpoint config summary: `{rel(paths.tables_dir / 'checkpoint_config_summary.csv')}` ({len(checkpoint_rows)} rows)",
            f"- Offline inference summary: `{rel(paths.tables_dir / 'offline_inference_summary.csv')}` ({len(offline_rows)} rows)",
            "",
            "## Plots",
            "",
            f"- Prepared dataset overview: `{rel(paths.plots_dir / 'dataset_prepared_overview.png')}`",
        ]
    )
    for png in sorted(paths.plots_dir.glob("*rollout_decomposition.png")):
        lines.append(f"- Rollout decomposition: `{rel(png)}`")
    offline_pngs = sorted(paths.plots_dir.glob("offline_*.png"))
    if offline_pngs:
        lines.append(f"- Offline chunk examples: {len(offline_pngs)} PNGs under `{rel(paths.plots_dir)}`")
    lines.extend(
        [
            "",
            "## Interpretation Checklist",
            "",
            "- If `raw_z_min` is already below dataset action z range, the low-z issue is model output or inference input, not postprocess lowering it.",
            "- If `target_z_min` is lower than `raw_z_min`, postprocess is responsible; otherwise clamp/step limit is not the source of low z.",
            "- If `gripper_score_max < 0.8`, current postprocess will never send a close command.",
            "- If offline teacher-forced error is large, prioritize model training/label/input semantics before robot controller tuning.",
            "- If offline teacher-forced error is small but rollout decompositions diverge, prioritize realtime aggregation/action history/postprocess distribution shift.",
            "- `policy_device_counts=cpu:1|cuda:N` means step 0 was logged before lazy CUDA loading; sustained CPU counts would indicate an old rollout path.",
            "",
        ]
    )
    return "\n".join(lines)


def build_diagnosis(
    dataset_rows: list[dict[str, Any]],
    collect_rows: list[dict[str, Any]],
    rollout_rows: list[dict[str, Any]],
    offline_rows: list[dict[str, Any]],
) -> list[str]:
    lines = ["### Diagnosis Against Current Hypotheses"]
    if dataset_rows:
        dataset_z_min = min(float(row["action_z_min"]) for row in dataset_rows if row.get("action_z_min") not in ("", None))
        dataset_z_max = max(float(row["action_z_max"]) for row in dataset_rows if row.get("action_z_max") not in ("", None))
        first_close_steps = [int(row["first_close_step"]) for row in dataset_rows if str(row.get("first_close_step", "")).isdigit()]
        first_open_steps = [int(row["first_open_after_close_step"]) for row in dataset_rows if str(row.get("first_open_after_close_step", "")).isdigit()]
        lines.append(
            "- Training data is internally consistent at 10Hz: action z range is %.3f-%.3fm; close normally starts around step %.1f and open around %.1f."
            % (
                dataset_z_min,
                dataset_z_max,
                float(np.mean(first_close_steps)) if first_close_steps else math.nan,
                float(np.mean(first_open_steps)) if first_open_steps else math.nan,
            )
        )
    if collect_rows:
        same = [float(row["same_step_xyz_error_mm_mean"]) for row in collect_rows if row.get("same_step_xyz_error_mm_mean")]
        nxt = [float(row["next_step_xyz_error_mm_mean"]) for row in collect_rows if row.get("next_step_xyz_error_mm_mean")]
        z_bias = [float(row["target_minus_robot_z_mm_mean"]) for row in collect_rows if row.get("target_minus_robot_z_mm_mean")]
        lines.append(
            "- Command label vs measured EEF: same-step error mean %.1fmm, next-step error mean %.1fmm, z command-measured bias %.1fmm. This supports checking a next-measured-pose label ablation, but does not by itself explain 3-5cm low raw model z."
            % (float(np.mean(same)), float(np.mean(nxt)), float(np.mean(z_bias)))
        )
    if rollout_rows:
        for row in rollout_rows:
            policy = row.get("policy")
            raw_z = parse_float(row.get("raw_z_min"))
            target_z = parse_float(row.get("target_z_min"))
            obs_z = parse_float(row.get("observed_z_min"))
            close_step = row.get("first_gripper_closed_step", "")
            transitions = row.get("gripper_close_transition_count", "")
            clamps = row.get("workspace_clamped_count", "")
            device_counts = row.get("policy_device_counts", "")
            if math.isfinite(raw_z):
                if raw_z < 0.3567:
                    z_reason = "raw model z is below the training action minimum, so low-z starts before postprocess"
                elif math.isfinite(target_z) and target_z < raw_z - 1e-4:
                    z_reason = "postprocess lowered z"
                else:
                    z_reason = "postprocess did not lower z"
            else:
                z_reason = "raw target not logged"
            lines.append(
                "- %s rollout: raw/target/observed z min %.3f/%.3f/%.3f, %s; close first step=%s transitions=%s; workspace clamps=%s; device counts=%s."
                % (policy, raw_z, target_z, obs_z, z_reason, close_step, transitions, clamps, device_counts)
            )
    if offline_rows:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in offline_rows:
            if row.get("error"):
                continue
            grouped.setdefault(str(row.get("policy")), []).append(row)
        for policy, rows in grouped.items():
            xyz = np.asarray([float(row["xyz_error_mm_mean"]) for row in rows], dtype=float)
            z = np.asarray([float(row["z_error_mm_mean"]) for row in rows], dtype=float)
            grip = np.asarray([float(row["gripper_abs_error_mean"]) for row in rows], dtype=float)
            lines.append(
                "- %s offline teacher-forced error: xyz mean %.1fmm, z bias %.1fmm, gripper abs %.3f. This is a policy/input/training signal independent of robot execution."
                % (policy, float(np.mean(xyz)), float(np.mean(z)), float(np.mean(grip)))
            )
    lines.append(
        "- Frequency hypothesis: prepared and rollout are both 10Hz (`dt=0.1` and rollout control_hz=10), so raw 60Hz collection is not the direct training/rollout frequency mismatch after preparation."
    )
    lines.append(
        "- Postprocess hypothesis: SFP and MOTIF raw z go below training range and are clamped upward to 0.345m; postprocess is not causing their low z, though workspace clamps/step limits can distort timing."
    )
    lines.append(
        "- Inference hypothesis remains plausible: offline teacher-forced errors are nontrivial, and SFP/MOTIF show negative z bias even without robot dynamics."
    )
    lines.append(
        "- Reset/control mismatch still needs a hardware-side audit, but collect tracking error is around 9mm next-step, much smaller than the worst low-z raw predictions."
    )
    return lines


def load_demo_arrays(demo_dir: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for key in ("agent_pos", "agent_vel", "action", "action_vel"):
        path = demo_dir / f"{key}.npz"
        if path.exists():
            arrays[key] = np.load(path)["arr_0"]
    return arrays


def observation_from_tcp_state(tcp_state: np.ndarray, *, executed_action: np.ndarray | None = None) -> dict[str, Any]:
    tcp = np.asarray(tcp_state, dtype=float)
    gripper_width = float((1.0 - np.clip(tcp[9], 0.0, 1.0)) * 0.078)
    obs = {
        "controller_state": {
            "tcp_pose": tcp_state_to_pose7d(tcp).tolist(),
            "tcp_velocity": [0.0] * 6,
            "tcp_wrench": [0.0] * 6,
            "joint_positions": [0.0] * 7,
            "joint_velocities": [0.0] * 7,
            "gripper_width": gripper_width,
            "gripper_force": 0.0,
            "wall_time": 0.0,
            "monotonic_time": 0.0,
            "control_frequency_hz": 10.0,
            "backend": "offline_debug",
        }
    }
    if executed_action is not None:
        obs["executed_action"] = np.asarray(executed_action, dtype=float).tolist()
    return obs


def tcp_state_to_pose7d(tcp_state: np.ndarray) -> np.ndarray:
    from vt_franka_workspace.rollout.action_math import tcp_state_to_pose7d_and_gripper

    pose, _ = tcp_state_to_pose7d_and_gripper(tcp_state)
    return pose


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_train_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(errors="ignore")
    numbers = [float(item) for item in re.findall(r"(?:val_loss|validation_loss|loss)[^0-9+\-.]*([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)", text, flags=re.I)]
    return {
        "train_log_bytes": path.stat().st_size,
        "parsed_loss_count": len(numbers),
        "parsed_loss_min": min(numbers) if numbers else "",
        "parsed_loss_last": numbers[-1] if numbers else "",
    }


def sample_start_indices(length: int, t_obs: int, t_pred: int, max_count: int) -> list[int]:
    max_start = length - t_obs - t_pred
    if max_start < 0:
        return []
    if max_count <= 1:
        return [0]
    return sorted(set(int(round(x)) for x in np.linspace(0, max_start, min(max_count, max_start + 1))))


def first_threshold_crossing(values: np.ndarray, threshold: float, *, above: bool, start: int = 0) -> int | None:
    arr = np.asarray(values)
    for idx in range(max(0, start), len(arr)):
        if above and arr[idx] >= threshold:
            return idx
        if not above and arr[idx] < threshold:
            return idx
    return None


def first_true(values: list[bool]) -> int | None:
    for idx, value in enumerate(values):
        if value:
            return idx
    return None


def count_rising_edges(values: list[bool]) -> int:
    count = 0
    prev = False
    for value in values:
        if value and not prev:
            count += 1
        prev = value
    return count


def append_pose(items: list[np.ndarray], value: Any) -> None:
    if value is None:
        items.append(np.full(7, np.nan, dtype=float))
        return
    arr = np.asarray(value, dtype=float)
    if arr.shape[0] < 3:
        items.append(np.full(7, np.nan, dtype=float))
        return
    if arr.shape[0] >= 7:
        items.append(arr[:7].astype(float))
    else:
        out = np.full(7, np.nan, dtype=float)
        out[: len(arr)] = arr
        items.append(out)


def to_pose_array(items: list[np.ndarray]) -> np.ndarray:
    if not items:
        return np.empty((0, 7), dtype=float)
    return np.stack(items, axis=0).astype(float)


def plot_xy(ax, arr: np.ndarray, label: str, color: str) -> None:
    if arr.size and np.isfinite(arr[:, :2]).any():
        ax.plot(arr[:, 0], arr[:, 1], label=label, color=color, alpha=0.85, linewidth=1.4)
        ax.scatter(arr[0, 0], arr[0, 1], color=color, marker="o", s=20)
        ax.scatter(arr[-1, 0], arr[-1, 1], color=color, marker="x", s=25)


def plot_dim(ax, t: np.ndarray, arr: np.ndarray, dim: int, label: str, color: str) -> None:
    if arr.size and np.isfinite(arr[:, dim]).any():
        ax.plot(t[: len(arr)], arr[:, dim], label=label, color=color, alpha=0.85, linewidth=1.3)


def pose_min(arr: np.ndarray, dim: int) -> Any:
    if arr.size == 0 or not np.isfinite(arr[:, dim]).any():
        return ""
    return float(np.nanmin(arr[:, dim]))


def pose_max(arr: np.ndarray, dim: int) -> Any:
    if arr.size == 0 or not np.isfinite(arr[:, dim]).any():
        return ""
    return float(np.nanmax(arr[:, dim]))


def float_or_nan(value: Any) -> float:
    try:
        if value is None:
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def safe_min(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmin(arr)) if arr.size else math.nan


def safe_max(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.nanmax(arr)) if arr.size else math.nan


def percentile(values: np.ndarray, pct: float) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return float(np.percentile(arr, pct))


def nanmean_or_empty(values: list[float]) -> Any:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else ""


def nanmax_or_empty(values: list[float]) -> Any:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.max(arr)) if arr.size else ""


def nanmin_or_empty(values: list[float]) -> Any:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.min(arr)) if arr.size else ""


def mean_positive(values: list[float]) -> Any:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    return float(np.mean(arr)) if arr.size else ""


def none_to_empty(value: Any) -> Any:
    return "" if value is None else value


def nested_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def infer_run_name(path: Path) -> str:
    parts = path.parts
    if "episodes" in parts:
        idx = parts.index("episodes")
        if idx > 0:
            return parts[idx - 1]
    return path.parent.parent.parent.name


def infer_policy_name(run: str) -> str:
    for policy in POLICIES:
        if policy in run:
            return policy
    return ""


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def fmt(value: Any) -> str:
    if value == "" or value is None:
        return ""
    try:
        return f"{float(value):.4f}"
    except Exception:
        return str(value)


def parse_float(value: Any) -> float:
    try:
        if value in ("", None):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


if __name__ == "__main__":
    raise SystemExit(main())
