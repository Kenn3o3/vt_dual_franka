from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from ....config import (
    InferenceRuntimeSettings,
    PolicyConfig,
    WorkspaceSettings,
    load_inference_config,
    load_policy_config,
    load_workspace_config,
)
from ....recording.image_io import read_rgb_image
from ...mpd.math import gripper_width_to_closedness, pose7d_and_gripper_to_tcp_state
from .config import get_model_spec
from .policy import VisuotactilePolicy
from .runtime import action_row_to_vt_action, load_runtime_manifests


REPO_ROOT = Path(__file__).resolve().parents[6]
DEFAULT_WORKSPACE_CONFIG = REPO_ROOT / "robot_workspace" / "config" / "workspace.yaml"


@dataclass(frozen=True)
class EpisodeStep:
    raw: dict[str, Any]
    step_index: int
    timestamp: float
    rgb_path: Path
    tactile_path: Path
    rgb_metadata: dict[str, Any]
    tactile_metadata: dict[str, Any]
    controller_state: dict[str, Any]
    action: dict[str, Any]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a visuotactile checkpoint offline on every timestamp of a recorded common dataset episode "
            "and render predicted-vs-ground-truth action horizon plots."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint file or checkpoint directory.")
    parser.add_argument("--episode-dir", type=Path, required=True, help="Common dataset episode directory with steps.jsonl.")
    parser.add_argument("--workspace-config", type=Path, default=DEFAULT_WORKSPACE_CONFIG)
    parser.add_argument(
        "--policy-config",
        type=Path,
        default=None,
        help="Optional policy YAML. When omitted, model/task are inferred from the checkpoint manifests.",
    )
    parser.add_argument(
        "--inference-config",
        type=Path,
        default=None,
        help="Optional inference YAML. Used mainly for obs_horizon/control_hz defaults.",
    )
    parser.add_argument("--model", default=None, help="Override model name when checkpoint manifest is unavailable.")
    parser.add_argument("--task-name", default=None, help="Override task name when checkpoint manifest is unavailable.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--horizon", type=int, default=8, help="Action prediction/plot horizon.")
    parser.add_argument("--obs-horizon", type=int, default=None, help="Override observation horizon.")
    parser.add_argument("--max-steps", type=int, default=None, help="Limit number of episode timestamps for smoke tests.")
    parser.add_argument("--stride", type=int, default=1, help="Run every Nth timestamp.")
    parser.add_argument("--fps", type=float, default=5.0, help="Output video FPS.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-video", type=Path, default=None)
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Write JSONL/NPZ metrics only; skip MP4 rendering.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = mockrun_episode(
        checkpoint=args.checkpoint,
        episode_dir=args.episode_dir,
        workspace_config=args.workspace_config,
        policy_config_path=args.policy_config,
        inference_config_path=args.inference_config,
        model=args.model,
        task_name=args.task_name,
        device=args.device,
        horizon=args.horizon,
        obs_horizon=args.obs_horizon,
        max_steps=args.max_steps,
        stride=args.stride,
        fps=args.fps,
        output_dir=args.output_dir,
        output_video=args.output_video,
        write_video=not args.no_video,
    )
    print(f"output_dir={result['output_dir']}")
    print(f"predictions_jsonl={result['predictions_jsonl']}")
    print(f"arrays_npz={result['arrays_npz']}")
    if result.get("video"):
        print(f"video={result['video']}")


def mockrun_episode(
    *,
    checkpoint: Path,
    episode_dir: Path,
    workspace_config: Path = DEFAULT_WORKSPACE_CONFIG,
    policy_config_path: Path | None = None,
    inference_config_path: Path | None = None,
    model: str | None = None,
    task_name: str | None = None,
    device: str = "auto",
    horizon: int = 8,
    obs_horizon: int | None = None,
    max_steps: int | None = None,
    stride: int = 1,
    fps: float = 5.0,
    output_dir: Path | None = None,
    output_video: Path | None = None,
    write_video: bool = True,
) -> dict[str, str | None]:
    checkpoint = Path(checkpoint).expanduser().resolve()
    episode_dir = Path(episode_dir).expanduser().resolve()
    workspace = load_workspace_config(workspace_config)
    checkpoint_dir, checkpoint_file = _resolve_checkpoint_reference(checkpoint)
    policy_config, model_name, task_name = _resolve_policy_config(
        checkpoint_dir=checkpoint_dir,
        checkpoint_file=checkpoint_file,
        policy_config_path=policy_config_path,
        workspace=workspace,
        model=model,
        task_name=task_name,
        device=device,
    )
    inference_config = _resolve_inference_config(
        inference_config_path=inference_config_path,
        task_name=task_name,
        model_name=model_name,
        obs_horizon=obs_horizon,
    )
    if horizon <= 0:
        raise ValueError("--horizon must be positive")
    if stride <= 0:
        raise ValueError("--stride must be positive")

    steps = load_episode_steps(episode_dir)
    selected_indices = list(range(0, len(steps), int(stride)))
    if max_steps is not None:
        selected_indices = selected_indices[: max(0, int(max_steps))]
    if not selected_indices:
        raise ValueError("No episode steps selected")

    policy = VisuotactilePolicy.from_config(policy_config, inference_config, workspace)
    policy.ensure_loaded()
    policy.reset()
    output_dir = _resolve_output_dir(output_dir, episode_dir=episode_dir, checkpoint=checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "checkpoint": str(checkpoint),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file": None if checkpoint_file is None else str(checkpoint_file),
        "episode_dir": str(episode_dir),
        "model": model_name,
        "task_name": task_name,
        "obs_horizon": int(inference_config.obs_horizon),
        "horizon": int(horizon),
        "stride": int(stride),
        "num_episode_steps": int(len(steps)),
        "num_selected_steps": int(len(selected_indices)),
        "policy_config": policy_config.model_dump(mode="json"),
        "inference_config": inference_config.model_dump(mode="json"),
    }
    _write_json(output_dir / "metadata.json", metadata)

    rows: list[dict[str, Any]] = []
    step_indices: list[int] = []
    prediction_times: list[float] = []
    pred_pose6: list[np.ndarray] = []
    gt_pose6: list[np.ndarray] = []
    pred_pose7: list[np.ndarray] = []
    gt_pose7: list[np.ndarray] = []
    pred_raw: list[np.ndarray] = []
    gt_gripper: list[np.ndarray] = []

    predictions_path = output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for count, step_index in enumerate(selected_indices, start=1):
            observation_window = build_observation_window(steps, step_index, inference_config.obs_horizon)
            model_inputs = policy.build_model_inputs(observation_window)
            start = time.perf_counter()
            raw_chunk = np.asarray(policy.backend.predict_action_chunk(model_inputs), dtype=np.float64)
            elapsed = time.perf_counter() - start
            raw_chunk = _pad_rows(raw_chunk, horizon)
            decoded_actions = [
                action_row_to_vt_action(
                    raw_chunk[offset],
                    model_spec=policy.model_spec,
                    target_duration_sec=policy.target_duration_sec,
                    gripper_open_width_m=policy.gripper_open_width_m,
                    gripper_close_threshold=policy.settings.gripper_close_threshold,
                )
                for offset in range(horizon)
            ]
            pred_pose7_chunk = np.stack(
                [np.asarray(action["target_tcp"], dtype=np.float64) for action in decoded_actions],
                axis=0,
            )
            gt_pose7_chunk, gt_gripper_chunk = ground_truth_action_horizon(steps, step_index, horizon)
            pred_pose6_chunk = pose7_chunk_to_xyz_rpy_deg(pred_pose7_chunk)
            gt_pose6_chunk = pose7_chunk_to_xyz_rpy_deg(gt_pose7_chunk)
            row = {
                "step_index": int(step_index),
                "episode_step_index": int(steps[step_index].step_index),
                "timestamp": float(steps[step_index].timestamp),
                "inference_duration_sec": float(elapsed),
                "prediction_horizon": int(horizon),
                "raw_prediction": raw_chunk.astype(float).tolist(),
                "predicted_target_tcp": pred_pose7_chunk.astype(float).tolist(),
                "ground_truth_target_tcp": gt_pose7_chunk.astype(float).tolist(),
                "predicted_xyz_rpy_deg": pred_pose6_chunk.astype(float).tolist(),
                "ground_truth_xyz_rpy_deg": gt_pose6_chunk.astype(float).tolist(),
                "ground_truth_gripper_closed": gt_gripper_chunk.astype(bool).tolist(),
                "model_input_shapes": {key: list(np.asarray(value).shape) for key, value in model_inputs.items()},
                "source_images": {
                    "rgb_wrist": str(steps[step_index].rgb_path),
                    "tactile_left": str(steps[step_index].tactile_path),
                },
            }
            handle.write(json.dumps(_json_safe(row), separators=(",", ":")) + "\n")
            rows.append(row)
            step_indices.append(int(step_index))
            prediction_times.append(float(elapsed))
            pred_pose6.append(pred_pose6_chunk)
            gt_pose6.append(gt_pose6_chunk)
            pred_pose7.append(pred_pose7_chunk)
            gt_pose7.append(gt_pose7_chunk)
            pred_raw.append(raw_chunk)
            gt_gripper.append(gt_gripper_chunk.astype(np.float32))
            print(
                f"[{count}/{len(selected_indices)}] step={step_index} inference={elapsed * 1000.0:.1f} ms",
                flush=True,
            )

    arrays_path = output_dir / "mockrun_arrays.npz"
    np.savez_compressed(
        arrays_path,
        step_indices=np.asarray(step_indices, dtype=np.int64),
        inference_duration_sec=np.asarray(prediction_times, dtype=np.float64),
        predicted_xyz_rpy_deg=np.stack(pred_pose6, axis=0).astype(np.float32),
        ground_truth_xyz_rpy_deg=np.stack(gt_pose6, axis=0).astype(np.float32),
        predicted_target_tcp=np.stack(pred_pose7, axis=0).astype(np.float32),
        ground_truth_target_tcp=np.stack(gt_pose7, axis=0).astype(np.float32),
        raw_prediction=np.stack(pred_raw, axis=0).astype(np.float32),
        ground_truth_gripper_closed=np.stack(gt_gripper, axis=0).astype(np.float32),
    )

    video_path: Path | None = None
    if write_video:
        video_path = output_video or (output_dir / "mockrun_episode.mp4")
        write_mockrun_video(
            rows,
            steps=steps,
            output_path=video_path,
            fps=fps,
            horizon=horizon,
        )

    policy.close()
    return {
        "output_dir": str(output_dir),
        "predictions_jsonl": str(predictions_path),
        "arrays_npz": str(arrays_path),
        "video": None if video_path is None else str(video_path),
    }


def load_episode_steps(episode_dir: Path) -> list[EpisodeStep]:
    episode_dir = Path(episode_dir)
    steps_path = episode_dir / "steps.jsonl"
    if not steps_path.is_file():
        raise FileNotFoundError(f"Missing episode steps.jsonl: {steps_path}")
    dataset_root = _resolve_dataset_root(episode_dir)
    steps: list[EpisodeStep] = []
    with steps_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            images = payload.get("images") or {}
            source = payload.get("source") or {}
            rgb_path = _resolve_image_path(dataset_root, episode_dir, images.get("rgb_wrist"))
            tactile_path = _resolve_image_path(dataset_root, episode_dir, images.get("tactile_left"))
            rgb_source = source.get("rgb_wrist") if isinstance(source, dict) else {}
            tactile_source = source.get("tactile_left") if isinstance(source, dict) else {}
            steps.append(
                EpisodeStep(
                    raw=payload,
                    step_index=int(payload.get("step_index", len(steps))),
                    timestamp=float(payload.get("timestamp", len(steps))),
                    rgb_path=rgb_path,
                    tactile_path=tactile_path,
                    rgb_metadata=_metadata_from_source(rgb_source, stream_name="rgb_wrist"),
                    tactile_metadata=_metadata_from_source(tactile_source, stream_name="tactile_left"),
                    controller_state=dict(payload.get("controller_state") or {}),
                    action=dict(payload.get("action") or {}),
                )
            )
    if not steps:
        raise RuntimeError(f"Episode has no steps: {steps_path}")
    return steps


def build_observation_window(steps: list[EpisodeStep], step_index: int, obs_horizon: int) -> list[dict[str, Any]]:
    if obs_horizon <= 0:
        raise ValueError("obs_horizon must be positive")
    indices = [min(max(idx, 0), len(steps) - 1) for idx in range(step_index - obs_horizon + 1, step_index + 1)]
    return [step_to_observation(steps[index]) for index in indices]


def step_to_observation(step: EpisodeStep) -> dict[str, Any]:
    return {
        "images": {
            "wrist": {
                "image": read_rgb_image(step.rgb_path),
                "metadata": dict(step.rgb_metadata),
                "captured_wall_time": step.rgb_metadata.get("captured_wall_time", step.timestamp),
            },
        },
        "tactile": {
            "tactile_left": {
                "image": read_rgb_image(step.tactile_path),
                "metadata": dict(step.tactile_metadata),
                "captured_wall_time": step.tactile_metadata.get("captured_wall_time", step.timestamp),
            },
        },
        "proprioception": {"controller_state": dict(step.controller_state)},
        "timestamps": {"assembled_wall_time": float(step.timestamp)},
    }


def ground_truth_action_horizon(steps: list[EpisodeStep], step_index: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    poses: list[np.ndarray] = []
    gripper_closed: list[bool] = []
    for index in range(step_index, step_index + horizon):
        clamped = min(max(index, 0), len(steps) - 1)
        action = steps[clamped].action
        target_tcp = action.get("target_tcp")
        if target_tcp is None:
            raise KeyError(f"Episode step {clamped} is missing action.target_tcp")
        poses.append(np.asarray(target_tcp, dtype=np.float64))
        gripper_closed.append(bool(action.get("gripper_closed", False)))
    pose_array = np.stack(poses, axis=0)
    if pose_array.shape != (horizon, 7):
        raise ValueError(f"Expected GT target_tcp horizon ({horizon}, 7), got {pose_array.shape}")
    return pose_array, np.asarray(gripper_closed, dtype=bool)


def pose7_chunk_to_xyz_rpy_deg(poses: np.ndarray) -> np.ndarray:
    pose_array = np.asarray(poses, dtype=np.float64)
    if pose_array.ndim != 2 or pose_array.shape[1] != 7:
        raise ValueError(f"Expected pose7 chunk [T,7], got {pose_array.shape}")
    quat_wxyz = pose_array[:, 3:7]
    norms = np.linalg.norm(quat_wxyz, axis=1, keepdims=True)
    quat_wxyz = quat_wxyz / np.clip(norms, 1e-12, None)
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    rpy_deg = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
    return np.concatenate([pose_array[:, :3], rpy_deg], axis=1)


def write_mockrun_video(
    rows: list[dict[str, Any]],
    *,
    steps: list[EpisodeStep],
    output_path: Path,
    fps: float,
    horizon: int,
) -> Path:
    cv2 = _require_cv2()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_size = (1920, 1080)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    try:
        for row in rows:
            step = steps[int(row["step_index"])]
            wrist = read_rgb_image(step.rgb_path)
            tactile = read_rgb_image(step.tactile_path)
            plot = render_horizon_plot(
                np.asarray(row["ground_truth_xyz_rpy_deg"], dtype=np.float64),
                np.asarray(row["predicted_xyz_rpy_deg"], dtype=np.float64),
                horizon=horizon,
            )
            frame_rgb = compose_video_frame(
                wrist,
                tactile,
                plot,
                step_index=int(row["step_index"]),
                timestamp=float(row["timestamp"]),
                inference_duration_sec=float(row["inference_duration_sec"]),
            )
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
    finally:
        writer.release()
    return output_path


def render_horizon_plot(gt: np.ndarray, pred: np.ndarray, *, horizon: int) -> np.ndarray:
    try:
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("matplotlib is required to render mockrun videos") from exc

    gt = np.asarray(gt, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    if gt.shape != pred.shape or gt.shape != (horizon, 6):
        raise ValueError(f"Expected gt/pred shape ({horizon}, 6), got gt={gt.shape}, pred={pred.shape}")
    gt = _unwrap_xyz_rpy_horizon(gt)
    pred = _unwrap_xyz_rpy_horizon(pred)
    x = np.arange(horizon)
    labels = ["x m", "y m", "z m", "roll deg", "pitch deg", "yaw deg"]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), dpi=100)
    axes_flat = axes.reshape(-1)
    for dim, ax in enumerate(axes_flat):
        ax.plot(x, gt[:, dim], color="#1f77b4", marker="o", linewidth=2.0, label="gt")
        ax.plot(x, pred[:, dim], color="#d62728", marker="x", linewidth=2.0, label="pred")
        ax.set_title(labels[dim], fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.set_xlim(0, max(horizon - 1, 1))
        if dim >= 4:
            ax.set_xlabel("horizon")
        if dim == 0:
            ax.legend(loc="best", fontsize=8)
    fig.tight_layout(pad=1.0)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(height, width, 3)
    rendered = np.ascontiguousarray(image)
    plt.close(fig)
    return rendered


def _unwrap_xyz_rpy_horizon(values: np.ndarray) -> np.ndarray:
    unwrapped = np.asarray(values, dtype=np.float64).copy()
    if unwrapped.ndim != 2 or unwrapped.shape[1] != 6:
        raise ValueError(f"Expected xyz+rpy horizon [T,6], got {unwrapped.shape}")
    unwrapped[:, 3:6] = np.rad2deg(np.unwrap(np.deg2rad(unwrapped[:, 3:6]), axis=0))
    return unwrapped


def compose_video_frame(
    wrist_rgb: np.ndarray,
    tactile_rgb: np.ndarray,
    plot_rgb: np.ndarray,
    *,
    step_index: int,
    timestamp: float,
    inference_duration_sec: float,
) -> np.ndarray:
    cv2 = _require_cv2()
    frame = np.full((1080, 1920, 3), 18, dtype=np.uint8)
    left_w = 760
    right_w = 1120
    image_tile_h = 500
    wrist_tile = _fit_rgb_to_tile(wrist_rgb, left_w, image_tile_h)
    tactile_tile = _fit_rgb_to_tile(tactile_rgb, left_w, image_tile_h)
    plot_tile = cv2.resize(plot_rgb, (right_w, 1000), interpolation=cv2.INTER_AREA)
    frame[40 : 40 + image_tile_h, 40 : 40 + left_w] = wrist_tile
    frame[560 : 560 + image_tile_h, 40 : 40 + left_w] = tactile_tile
    frame[40:1040, 800:1920] = plot_tile
    _draw_label(frame, "wrist", 58, 78)
    _draw_label(frame, "gelsight", 58, 598)
    title = f"step {step_index:06d}   t={timestamp:.3f}   inference={inference_duration_sec * 1000.0:.1f} ms"
    cv2.rectangle(frame, (800, 0), (1920, 38), (18, 18, 18), thickness=-1)
    cv2.putText(frame, title, (820, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (235, 235, 235), 2, cv2.LINE_AA)
    return frame


def _resolve_policy_config(
    *,
    checkpoint_dir: Path,
    checkpoint_file: Path | None,
    policy_config_path: Path | None,
    workspace: WorkspaceSettings,
    model: str | None,
    task_name: str | None,
    device: str,
) -> tuple[PolicyConfig, str, str]:
    if policy_config_path is not None:
        policy_config = load_policy_config(policy_config_path)
        config = dict(policy_config.config)
        if model is not None:
            config["model"] = get_model_spec(model).name
        if task_name is not None:
            config["task_name"] = task_name
        if device != "auto":
            config["device"] = device
        if checkpoint_file is not None:
            config["checkpoint_file"] = checkpoint_file.name
        policy_config = policy_config.model_copy(update={"checkpoint_path": checkpoint_dir, "config": config})
        model_name = get_model_spec(config["model"]).name
        resolved_task_name = str(config.get("task_name") or task_name or "policy_run")
        return policy_config, model_name, resolved_task_name

    manifest_model: str | None = None
    manifest_task_name: str | None = None
    try:
        manifests = load_runtime_manifests(checkpoint_dir)
        manifest_model = str(manifests.policy.get("model") or "")
        manifest_task_name = str(manifests.policy.get("task_name") or "")
    except FileNotFoundError:
        pass
    model_name = get_model_spec(model or manifest_model).name if (model or manifest_model) else None
    if model_name is None:
        raise ValueError("Cannot infer model from checkpoint manifest; pass --model")
    resolved_task_name = task_name or manifest_task_name or _infer_task_name_from_checkpoint(workspace, checkpoint_dir) or "policy_run"
    config: dict[str, Any] = {
        "model": model_name,
        "task_name": resolved_task_name,
        "device": device,
    }
    if checkpoint_file is not None:
        config["checkpoint_file"] = checkpoint_file.name
    return PolicyConfig(type="visuotactile", checkpoint_path=checkpoint_dir, config=config), model_name, resolved_task_name


def _resolve_inference_config(
    *,
    inference_config_path: Path | None,
    task_name: str,
    model_name: str,
    obs_horizon: int | None,
) -> InferenceRuntimeSettings:
    if inference_config_path is not None:
        inference = load_inference_config(inference_config_path)
    else:
        spec = get_model_spec(model_name)
        inference = InferenceRuntimeSettings(
            task_name=task_name,
            obs_horizon=spec.obs_horizon,
            exe_horizon=1,
            control_hz=10.0,
        )
    if obs_horizon is not None:
        inference = inference.model_copy(update={"obs_horizon": int(obs_horizon)})
    return inference


def _resolve_checkpoint_reference(checkpoint: Path) -> tuple[Path, Path | None]:
    if checkpoint.is_file():
        if checkpoint.parent.name == "checkpoints":
            return checkpoint.parent.parent, checkpoint
        return checkpoint.parent, checkpoint
    if checkpoint.is_dir():
        return checkpoint, None
    raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")


def _resolve_dataset_root(episode_dir: Path) -> Path:
    episode_dir = Path(episode_dir)
    if episode_dir.parent.name == "episodes":
        return episode_dir.parent.parent
    manifest = episode_dir / "episode_manifest.json"
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        steps_path = payload.get("steps_path")
        if isinstance(steps_path, str) and steps_path.startswith("episodes/"):
            return episode_dir.parent.parent
    return episode_dir


def _resolve_image_path(dataset_root: Path, episode_dir: Path, value: Any) -> Path:
    if not value:
        raise RuntimeError(f"Episode step is missing image path under {episode_dir}")
    path = Path(str(value))
    candidates = [path] if path.is_absolute() else [dataset_root / path, episode_dir / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing episode image {value!r}; tried {', '.join(str(path) for path in candidates)}")


def _metadata_from_source(source: Any, *, stream_name: str) -> dict[str, Any]:
    payload = source if isinstance(source, dict) else {}
    metadata = payload.get("metadata")
    result = dict(metadata) if isinstance(metadata, dict) else {}
    result.setdefault("stream_name", stream_name)
    result.setdefault("color_space", "RGB")
    result.setdefault("standardization", "vt_franka_camera_standard_rgb_640x480_v1")
    if "captured_wall_time" not in result and payload.get("timestamp") is not None:
        result["captured_wall_time"] = float(payload["timestamp"])
    return result


def _pad_rows(values: np.ndarray, horizon: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"Expected action chunk [T,D], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError("Policy returned an empty action chunk")
    if array.shape[0] >= horizon:
        return array[:horizon]
    pad = np.repeat(array[-1:, :], horizon - array.shape[0], axis=0)
    return np.concatenate([array, pad], axis=0)


def _resolve_output_dir(output_dir: Path | None, *, episode_dir: Path, checkpoint: Path) -> Path:
    if output_dir is not None:
        return Path(output_dir).expanduser().resolve()
    ckpt_name = checkpoint.stem.replace("=", "_")
    return episode_dir / "mockrun_episode" / ckpt_name


def _fit_rgb_to_tile(image_rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    cv2 = _require_cv2()
    image = np.asarray(image_rgb, dtype=np.uint8)
    src_h, src_w = image.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    new_w = max(1, int(round(src_w * scale)))
    new_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 28, dtype=np.uint8)
    x0 = (width - new_w) // 2
    y0 = (height - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _draw_label(frame: np.ndarray, label: str, x: int, y: int) -> None:
    cv2 = _require_cv2()
    cv2.rectangle(frame, (x - 8, y - 28), (x + 190, y + 8), (0, 0, 0), thickness=-1)
    cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2) + "\n", encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _infer_task_name_from_checkpoint(workspace: WorkspaceSettings, checkpoint_dir: Path) -> str | None:
    root = Path(workspace.recording.checkpoints_root)
    try:
        rel = Path(checkpoint_dir).resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return rel.parts[0] if rel.parts else None


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenCV is required to render mockrun episode videos") from exc
    return cv2


if __name__ == "__main__":
    main()
