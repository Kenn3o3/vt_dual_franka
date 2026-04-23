"""Diffusion policy inference wrapper for vt_franka RolloutSupervisor.

Loads a trained movement-primitive-diffusion checkpoint and wraps it as a
vt_franka-compatible policy callable with observation horizon buffering,
asynchronous inference, latency-aware future action scheduling, temporal
ensemble, and execution-time safety clamping.

Usage via RolloutSupervisor:
    policy_spec = "kenny.policies.mpd_diffusion_policy:create_policy"
    policy_kwargs = {
        "checkpoint_path": "/path/to/best_model.pth",
        "config_path": "/path/to/hydra/config.yaml",
        "scaler_path": "/path/to/scaler_values.npz",
    }
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

LOGGER = logging.getLogger(__name__)

IMAGE_SIZE = (320, 240)  # (W, H) — resize target, must match training
CROP_SIZE = (288, 216)   # (W, H) — center crop for inference


@dataclass
class _PredictedAction:
    pose: np.ndarray
    gripper_score: float
    source_step: int


@dataclass
class _InferenceResult:
    episode_token: int
    obs_step: int
    actions_normalized: np.ndarray
    latency_steps: int
    duration_sec: float
    error: Exception | None = None


def _preprocess_image(image_bgr: np.ndarray) -> torch.Tensor:
    """Resize, center-crop, normalize, and convert HWC uint8 → CHW float32."""
    img = cv2.resize(image_bgr, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    h, w = img.shape[:2]
    ch, cw = CROP_SIZE[1], CROP_SIZE[0]
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    img = img[y0 : y0 + ch, x0 : x0 + cw]
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)


def _normalize(value: torch.Tensor, scaler: dict, symmetric: bool = True) -> torch.Tensor:
    normed = (value - scaler["min"]) / (scaler["max"] - scaler["min"])
    if symmetric:
        normed = 2.0 * normed - 1.0
    return normed


def _denormalize(value: torch.Tensor, scaler: dict, symmetric: bool = True) -> torch.Tensor:
    if symmetric:
        value = (value + 1.0) / 2.0
    return value * (scaler["max"] - scaler["min"]) + scaler["min"]


def _normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quat = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def _quat_hemisphere_align_wxyz(quaternion: np.ndarray, reference: np.ndarray) -> np.ndarray:
    quat = _normalize_quaternion_wxyz(quaternion)
    ref = _normalize_quaternion_wxyz(reference)
    if float(np.dot(quat, ref)) < 0.0:
        quat = -quat
    return quat


def _quat_slerp_wxyz(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(np.clip(alpha, 0.0, 1.0))
    q0 = _normalize_quaternion_wxyz(start)
    q1 = _quat_hemisphere_align_wxyz(end, q0)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        quat = q0 + alpha * (q1 - q0)
        return _normalize_quaternion_wxyz(quat)
    theta_0 = float(np.arccos(dot))
    sin_theta_0 = float(np.sin(theta_0))
    theta = theta_0 * alpha
    sin_theta = float(np.sin(theta))
    s0 = float(np.sin(theta_0 - theta) / sin_theta_0)
    s1 = float(sin_theta / sin_theta_0)
    quat = s0 * q0 + s1 * q1
    return _normalize_quaternion_wxyz(quat)


def _quat_angle_deg_wxyz(start: np.ndarray, end: np.ndarray) -> float:
    q0 = _normalize_quaternion_wxyz(start)
    q1 = _quat_hemisphere_align_wxyz(end, q0)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(abs(dot))))


def _weighted_pose_average_wxyz(poses: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    if not poses:
        raise ValueError("poses must not be empty")
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or weights.shape[0] != len(poses):
        raise ValueError("weights shape does not match poses")
    weights = weights / max(float(weights.sum()), 1e-8)

    pos_stack = np.stack([pose[:3] for pose in poses], axis=0)
    pos_avg = (pos_stack * weights[:, None]).sum(axis=0)

    ref_quat = _normalize_quaternion_wxyz(poses[-1][3:7])
    quat_stack = np.stack([_quat_hemisphere_align_wxyz(pose[3:7], ref_quat) for pose in poses], axis=0)
    quat_avg = (quat_stack * weights[:, None]).sum(axis=0)
    quat_avg = _normalize_quaternion_wxyz(quat_avg)

    return np.concatenate([pos_avg, quat_avg], axis=0)


class MpdDiffusionPolicy:
    """vt_franka-compatible policy wrapping a trained MPD diffusion agent.

    Maintains:
    - Observation horizon buffer (last t_obs frames)
    - Background inference worker
    - Future-timestep action candidates for temporal ensemble
    - Execution-time target smoothing and clamp
    """

    __vt_franka_control_hz__ = 10
    __vt_franka_policy_factory__ = False  # instances are created by create_policy()

    def __init__(
        self,
        agent,
        scaler_values: dict[str, dict[str, torch.Tensor]],
        t_obs: int,
        t_pred: int,
        device: str = "cuda",
        confirm_before_execute: bool = False,
        *,
        control_hz: float = 10.0,
        replan_interval_steps: int = 3,
        latency_step: int = 3,
        ensemble_mode: str = "hato",
        ensemble_tau: float = 0.6,
        max_translation_step_m: float = 0.008,
        max_rotation_step_deg: float = 0.6,
        gripper_close_threshold: float = 0.65,
        gripper_open_threshold: float = 0.35,
        open_gripper_width: float = 0.078,
        gripper_velocity: float = 0.1,
        gripper_force_limit: float = 7.0,
        open_width_tolerance_m: float = 0.015,
        open_resend_interval_steps: int = 5,
        max_future_steps: int | None = None,
        startup_hold_steps: int | None = None,
        debug_timing_log_interval_sec: float = 2.0,
    ) -> None:
        self.agent = agent
        self.scaler_values = scaler_values
        self.t_obs = t_obs
        self.t_pred = t_pred
        self.device = device
        self.confirm_before_execute = confirm_before_execute
        self.control_hz = float(control_hz)
        self.replan_interval_steps = max(1, int(replan_interval_steps))
        self.latency_step = max(0, int(latency_step))
        self.ensemble_mode = ensemble_mode
        self.ensemble_tau = float(ensemble_tau)
        self.max_translation_step_m = float(max_translation_step_m)
        self.max_rotation_step_deg = float(max_rotation_step_deg)
        self.gripper_close_threshold = float(gripper_close_threshold)
        self.gripper_open_threshold = float(gripper_open_threshold)
        self.open_gripper_width = float(open_gripper_width)
        self.gripper_velocity = float(gripper_velocity)
        self.gripper_force_limit = float(gripper_force_limit)
        self.open_width_tolerance_m = float(open_width_tolerance_m)
        self.open_resend_interval_steps = max(1, int(open_resend_interval_steps))
        self.max_future_steps = max_future_steps if max_future_steps is not None else max(self.t_pred * 3, 32)
        self.startup_hold_steps = startup_hold_steps if startup_hold_steps is not None else self.latency_step + 1
        self.debug_timing_log_interval_sec = float(debug_timing_log_interval_sec)

        self._obs_buffer: deque[dict[str, torch.Tensor]] = deque(maxlen=t_obs)
        self._future_actions: dict[int, list[_PredictedAction]] = {}
        self._step_counter = 0
        self._last_target_tcp: np.ndarray | None = None
        self._last_gripper_closed = False
        self._last_gripper_score = 0.0
        self._last_open_command_step = -10_000
        self._latest_observation: dict[str, Any] | None = None
        self._last_state_tcp: np.ndarray | None = None

        self._worker_lock = threading.Lock()
        self._worker_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._inference_requested_step = -1
        self._inference_running = False
        self._pending_result: _InferenceResult | None = None
        self._last_inference_duration_sec = 0.0
        self._last_inference_wall_time = 0.0
        self._last_log_wall_time = 0.0
        self._last_scheduled_step = -1
        self._episode_token = 0

        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="mpd-inference-worker",
            daemon=True,
        )
        self._worker_thread.start()

    def reset(self) -> None:
        self._obs_buffer.clear()
        self._future_actions.clear()
        self._step_counter = 0
        self._last_target_tcp = None
        self._last_gripper_closed = False
        self._last_gripper_score = 0.0
        self._last_open_command_step = -10_000
        self._latest_observation = None
        self._last_state_tcp = None
        with self._worker_lock:
            self._episode_token += 1
            self._pending_result = None
            self._inference_requested_step = -1
            self._last_scheduled_step = -1

    def __call__(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._latest_observation = observation
        obs_frame = self._extract_observation(observation)
        self._obs_buffer.append(obs_frame)
        state = observation.get("controller_state", {})
        current_tcp = np.asarray(state.get("tcp_pose", [0.0] * 7), dtype=np.float64)
        current_tcp[3:7] = _normalize_quaternion_wxyz(current_tcp[3:7])
        self._last_state_tcp = current_tcp.copy()

        self._consume_pending_result()
        self._maybe_request_inference()

        step_index = self._step_counter
        target_tcp_arr, gripper_score, debug_info = self._select_action_for_step(step_index, current_tcp)
        previous_gripper_closed = self._last_gripper_closed
        gripper_closed = self._update_gripper_state(gripper_score)

        target_tcp = target_tcp_arr.astype(float).tolist()

        if self.confirm_before_execute:
            self._print_step_and_confirm(observation, target_tcp, gripper_closed, gripper_score)
            self._last_target_tcp = np.asarray(target_tcp, dtype=np.float64)
            self._last_target_tcp[3:7] = _normalize_quaternion_wxyz(self._last_target_tcp[3:7])

        self._maybe_log_debug(debug_info)
        self._step_counter += 1

        action = {
            "target_tcp": target_tcp,
            "gripper_closed": gripper_closed,
            "gripper_score": float(gripper_score),
            "terminate": False,
        }
        if self._should_command_open_gripper(
            previous_gripper_closed=previous_gripper_closed,
            gripper_closed=gripper_closed,
            current_width=float(state.get("gripper_width", 0.0)),
            step_index=step_index,
        ):
            action["gripper_width"] = self.open_gripper_width
            action["gripper_velocity"] = self.gripper_velocity
            action["gripper_force_limit"] = self.gripper_force_limit
            self._last_open_command_step = step_index
        return action

    def _print_step_and_confirm(
        self,
        observation: dict[str, Any],
        target_tcp,
        gripper_closed: bool,
        gripper_raw: float,
    ) -> None:
        """Print current proprioception + proposed action, wait for user confirmation."""
        state = observation.get("controller_state", {})
        cur_tcp = np.asarray(state.get("tcp_pose", [0.0] * 7), dtype=np.float64)
        cur_gw = float(state.get("gripper_width", 0.0))
        cur_gf = float(state.get("gripper_force", 0.0))
        target_tcp_arr = np.asarray(target_tcp, dtype=np.float64)
        delta_tcp = target_tcp_arr - cur_tcp

        np.set_printoptions(precision=4, suppress=True)
        print("\n" + "=" * 70)
        print(f"STEP {self._step_counter}  (future_actions={len(self._future_actions)})")
        print("-" * 70)
        print("Current proprioception:")
        print(f"  tcp_pose        [xyz,quat]: {cur_tcp}")
        print(f"  gripper_width             : {cur_gw:.4f} m")
        print(f"  gripper_force             : {cur_gf:.3f} N")
        print("Proposed action:")
        print(f"  target_tcp      [xyz,quat]: {target_tcp_arr}")
        print(f"  delta_tcp (target - cur)  : {delta_tcp}")
        print(f"  gripper_closed            : {gripper_closed}  (raw={gripper_raw:.3f})")
        print("-" * 70)
        try:
            user = input("Press ENTER to execute, 's' to skip, 'q' to terminate episode: ").strip().lower()
        except EOFError:
            user = ""
        if user == "q":
            raise KeyboardInterrupt("User requested episode termination")
        if user == "s":
            print("  -> skipped (executing zero-delta no-op)")
            target_tcp[:] = cur_tcp.tolist()
        print("=" * 70)

    def _extract_observation(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract and preprocess one observation frame from RolloutSupervisor."""
        frame: dict[str, torch.Tensor] = {}

        state = obs.get("controller_state", {})
        tcp_pose = np.array(state.get("tcp_pose", [0.0] * 7), dtype=np.float32)
        gripper_w = np.array([state.get("gripper_width", 0.0)], dtype=np.float32)

        tcp_tensor = torch.from_numpy(tcp_pose)
        grip_tensor = torch.from_numpy(gripper_w)
        frame["tcp_pose"] = _normalize(tcp_tensor, self.scaler_values["tcp_pose"])
        frame["gripper_width"] = _normalize(grip_tensor, self.scaler_values["gripper_width"])

        for cam_role in ("wrist", "third_person"):
            cam_data = obs.get(cam_role, {})
            image = cam_data.get("image")
            if image is not None:
                frame[f"rgb_{cam_role}"] = _preprocess_image(image)
            else:
                frame[f"rgb_{cam_role}"] = torch.zeros(3, CROP_SIZE[1], CROP_SIZE[0])

        return frame

    @torch.no_grad()
    def _predict_from_obs_list(self, obs_list: list[dict[str, torch.Tensor]]) -> np.ndarray:
        obs_list = list(obs_list)
        while len(obs_list) < self.t_obs:
            obs_list.insert(0, obs_list[0])

        obs_window: dict[str, torch.Tensor] = {}
        for key in ("rgb_wrist", "rgb_third_person"):
            stacked = torch.stack([f[key] for f in obs_list], dim=0)  # (t_obs, C, H, W)
            obs_window[key] = stacked.unsqueeze(0).to(self.device)   # (1, t_obs, C, H, W)

        action_chunk = self.agent.predict(obs_window, extra_inputs={})  # (1, t_pred, action_dim)
        return action_chunk[0].detach().cpu().numpy()

    def _worker_loop(self) -> None:
        while not self._shutdown_event.is_set():
            self._worker_event.wait(timeout=0.1)
            self._worker_event.clear()
            if self._shutdown_event.is_set():
                break

            with self._worker_lock:
                request_step = self._inference_requested_step
                episode_token = self._episode_token
                if request_step < 0 or self._latest_observation is None:
                    self._inference_running = False
                    continue
                obs_list = list(self._obs_buffer)
                self._inference_requested_step = -1
                self._inference_running = True

            if not obs_list:
                with self._worker_lock:
                    self._inference_running = False
                continue

            try:
                started = time.monotonic()
                actions_np = self._predict_from_obs_list(obs_list)
                duration_sec = time.monotonic() - started
                latency_steps = max(self.latency_step, int(np.ceil(duration_sec * self.control_hz)))
                result = _InferenceResult(
                    episode_token=episode_token,
                    obs_step=request_step,
                    actions_normalized=actions_np,
                    latency_steps=latency_steps,
                    duration_sec=duration_sec,
                )
            except Exception as exc:  # pragma: no cover - hardware dependent failure path
                LOGGER.exception("MPD inference worker failed")
                result = _InferenceResult(
                    episode_token=episode_token,
                    obs_step=request_step,
                    actions_normalized=np.zeros((0, 8), dtype=np.float32),
                    latency_steps=self.latency_step,
                    duration_sec=0.0,
                    error=exc,
                )

            with self._worker_lock:
                self._pending_result = result
                self._inference_running = False

    def _consume_pending_result(self) -> None:
        with self._worker_lock:
            result = self._pending_result
            self._pending_result = None
        if result is None:
            return
        if result.episode_token != self._episode_token:
            return
        if result.error is not None:
            raise result.error

        self._last_inference_duration_sec = result.duration_sec
        self._last_inference_wall_time = time.time()

        start_step = max(self._step_counter, result.obs_step + result.latency_steps)
        start_index = max(0, start_step - result.obs_step)
        if start_index >= len(result.actions_normalized):
            LOGGER.warning(
                "Discarded MPD chunk: duration=%.3fs latency_steps=%d obs_step=%d chunk_len=%d",
                result.duration_sec,
                result.latency_steps,
                result.obs_step,
                len(result.actions_normalized),
            )
            return

        kept = 0
        for action_offset in range(start_index, len(result.actions_normalized)):
            step = result.obs_step + action_offset
            if step < self._step_counter:
                continue
            action_raw = self._denormalize_action(result.actions_normalized[action_offset])
            candidate = _PredictedAction(
                pose=action_raw[:7],
                gripper_score=float(action_raw[7]),
                source_step=result.obs_step,
            )
            self._future_actions.setdefault(step, []).append(candidate)
            kept += 1

        self._last_scheduled_step = max(self._last_scheduled_step, result.obs_step + len(result.actions_normalized) - 1)
        self._prune_future_actions()
        LOGGER.debug(
            "MPD inference scheduled %d/%d future actions (duration=%.3fs, latency=%d steps)",
            kept,
            len(result.actions_normalized),
            result.duration_sec,
            result.latency_steps,
        )

    def _maybe_request_inference(self) -> None:
        if not self._obs_buffer:
            return
        step = self._step_counter
        need_startup = self._last_scheduled_step < step + self.startup_hold_steps
        interval_hit = (step % self.replan_interval_steps) == 0
        if not need_startup and not interval_hit:
            return
        with self._worker_lock:
            if self._inference_running or self._inference_requested_step >= 0:
                return
            self._inference_requested_step = step
            self._worker_event.set()

    def _prune_future_actions(self) -> None:
        min_step = self._step_counter
        max_step = self._step_counter + self.max_future_steps
        stale = [step for step in self._future_actions if step < min_step or step > max_step]
        for step in stale:
            self._future_actions.pop(step, None)

    def _denormalize_action(self, action_normalized: np.ndarray) -> np.ndarray:
        action_tensor = torch.from_numpy(np.asarray(action_normalized, dtype=np.float32))
        action_raw = _denormalize(action_tensor, self.scaler_values["actions"]).numpy().astype(np.float64)
        action_raw[3:7] = _normalize_quaternion_wxyz(action_raw[3:7])
        return action_raw

    def _select_action_for_step(
        self,
        step_index: int,
        current_tcp: np.ndarray,
    ) -> tuple[np.ndarray, float, dict[str, float | int | str]]:
        candidates = self._future_actions.pop(step_index, [])
        if candidates:
            pose, gripper_score = self._ensemble_candidates(candidates, step_index, current_tcp)
            source = "ensemble"
            num_candidates = len(candidates)
        else:
            pose = current_tcp.copy() if self._last_target_tcp is None else self._last_target_tcp.copy()
            gripper_score = self._last_gripper_score
            source = "hold"
            num_candidates = 0

        clamped_pose = self._clamp_pose_step(
            target_pose=pose,
            current_tcp=current_tcp,
            previous_target=self._last_target_tcp,
        )
        self._last_target_tcp = clamped_pose.copy()
        self._last_gripper_score = float(gripper_score)

        debug_info: dict[str, float | int | str] = {
            "step": step_index,
            "source": source,
            "num_candidates": num_candidates,
            "inference_ms": self._last_inference_duration_sec * 1000.0,
            "buffer_size": len(self._future_actions),
            "translation_error_mm": float(np.linalg.norm(clamped_pose[:3] - current_tcp[:3]) * 1000.0),
            "rotation_error_deg": _quat_angle_deg_wxyz(current_tcp[3:7], clamped_pose[3:7]),
        }
        return clamped_pose, float(gripper_score), debug_info

    def _ensemble_candidates(
        self,
        candidates: list[_PredictedAction],
        step_index: int,
        current_tcp: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        if len(candidates) == 1 or self.ensemble_mode == "new":
            latest = max(candidates, key=lambda item: item.source_step)
            return latest.pose.copy(), latest.gripper_score

        if self.ensemble_mode != "hato":
            raise ValueError(f"Unsupported ensemble_mode: {self.ensemble_mode}")

        ages = np.array([max(0, step_index - item.source_step) for item in candidates], dtype=np.float64)
        weights = np.power(self.ensemble_tau, ages)
        weights = weights / max(float(weights.sum()), 1e-8)

        poses = [item.pose for item in sorted(candidates, key=lambda item: item.source_step)]
        sorted_weights = np.array(
            [weight for _, weight in sorted(zip(candidates, weights), key=lambda pair: pair[0].source_step)],
            dtype=np.float64,
        )
        pose = _weighted_pose_average_wxyz(poses, sorted_weights)
        if float(np.dot(pose[3:7], current_tcp[3:7])) < 0.0:
            pose[3:7] *= -1.0

        gripper_scores = np.array(
            [item.gripper_score for item in sorted(candidates, key=lambda item: item.source_step)],
            dtype=np.float64,
        )
        gripper_score = float((gripper_scores * sorted_weights).sum())
        return pose, gripper_score

    def _clamp_pose_step(
        self,
        *,
        target_pose: np.ndarray,
        current_tcp: np.ndarray,
        previous_target: np.ndarray | None,
    ) -> np.ndarray:
        base_pose = previous_target if previous_target is not None else current_tcp
        base_pose = np.asarray(base_pose, dtype=np.float64)
        target_pose = np.asarray(target_pose, dtype=np.float64).copy()

        translation_delta = target_pose[:3] - base_pose[:3]
        translation_norm = float(np.linalg.norm(translation_delta))
        if translation_norm > self.max_translation_step_m > 0.0:
            translation_delta *= self.max_translation_step_m / translation_norm
        pose = base_pose.copy()
        pose[:3] = base_pose[:3] + translation_delta

        rotation_delta_deg = _quat_angle_deg_wxyz(base_pose[3:7], target_pose[3:7])
        if rotation_delta_deg > self.max_rotation_step_deg > 0.0:
            alpha = self.max_rotation_step_deg / max(rotation_delta_deg, 1e-8)
            pose[3:7] = _quat_slerp_wxyz(base_pose[3:7], target_pose[3:7], alpha)
        else:
            pose[3:7] = _quat_hemisphere_align_wxyz(target_pose[3:7], base_pose[3:7])
        pose[3:7] = _normalize_quaternion_wxyz(pose[3:7])
        return pose

    def _update_gripper_state(self, gripper_score: float) -> bool:
        if self._last_gripper_closed:
            if gripper_score <= self.gripper_open_threshold:
                self._last_gripper_closed = False
        else:
            if gripper_score >= self.gripper_close_threshold:
                self._last_gripper_closed = True
        return self._last_gripper_closed

    def _should_command_open_gripper(
        self,
        *,
        previous_gripper_closed: bool,
        gripper_closed: bool,
        current_width: float,
        step_index: int,
    ) -> bool:
        if gripper_closed:
            return False
        transitioned_open = previous_gripper_closed
        gripper_still_closed = current_width < self.open_gripper_width - self.open_width_tolerance_m
        resend_due = step_index - self._last_open_command_step >= self.open_resend_interval_steps
        return transitioned_open or (gripper_still_closed and resend_due)

    def _maybe_log_debug(self, debug_info: dict[str, float | int | str]) -> None:
        now = time.time()
        if now - self._last_log_wall_time < self.debug_timing_log_interval_sec:
            return
        self._last_log_wall_time = now
        LOGGER.info(
            "MPD rollout step=%s source=%s candidates=%s inf=%.1fms future=%s err=%.1fmm/%.2fdeg",
            debug_info["step"],
            debug_info["source"],
            debug_info["num_candidates"],
            debug_info["inference_ms"],
            debug_info["buffer_size"],
            debug_info["translation_error_mm"],
            debug_info["rotation_error_deg"],
        )

    def close(self) -> None:
        self._shutdown_event.set()
        self._worker_event.set()
        if self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)


def create_policy(
    checkpoint_path: str,
    config_path: str,
    scaler_path: str,
    device: str = "cuda",
    confirm_before_execute: bool = False,
    control_hz: float = 10.0,
    replan_interval_steps: int = 3,
    latency_step: int = 3,
    ensemble_mode: str = "hato",
    ensemble_tau: float = 0.6,
    max_translation_step_m: float = 0.008,
    max_rotation_step_deg: float = 0.6,
    gripper_close_threshold: float = 0.65,
    gripper_open_threshold: float = 0.35,
    open_gripper_width: float = 0.078,
    gripper_velocity: float = 0.1,
    gripper_force_limit: float = 7.0,
    open_width_tolerance_m: float = 0.015,
    open_resend_interval_steps: int = 5,
    max_future_steps: int | None = None,
    startup_hold_steps: int | None = None,
    debug_timing_log_interval_sec: float = 2.0,
    **kwargs,
) -> MpdDiffusionPolicy:
    """Factory function for RolloutSupervisor's load_policy().

    Args:
        checkpoint_path: Path to best_model.pth from MPD training.
        config_path: Path to the Hydra config YAML used for training.
        scaler_path: Path to scaler_values.npz saved during conversion.
        device: "cuda" or "cpu".
        confirm_before_execute: If True, print current state + proposed action
            and wait for ENTER before executing each step. Use for safe debugging.
        control_hz: Rollout control frequency used for latency matching.
        replan_interval_steps: How often to request a fresh inference.
        latency_step: Minimum action steps to skip at the front of each chunk.
        ensemble_mode: "new" or "hato" for temporal action fusion.
        ensemble_tau: Exponential decay used when ensemble_mode="hato".
        max_translation_step_m: Max commanded translation change per control step.
        max_rotation_step_deg: Max commanded rotation change per control step.
        open_gripper_width: Width command used when policy decides to release.
    """
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    cfg = OmegaConf.load(config_path)
    cfg.device = device
    cfg.agent_config.device = device

    # Fill in feature sizes that setup_train normally sets from data.
    # action_keys: actions is 8D (7D TCP + 1D gripper)
    for info in cfg.agent_config.process_batch_config.action_keys:
        if info.feature_size is None:
            info.feature_size = [8]
    # observation_keys: set from encoder config
    encoder_obs_keys = [nc.observation_key for nc in cfg.agent_config.encoder_config.network_configs]
    cfg.agent_config.process_batch_config.observation_keys = encoder_obs_keys
    # image feature sizes: (3, crop_H, crop_W)
    for nc in cfg.agent_config.encoder_config.network_configs:
        if nc.feature_size is None:
            nc.feature_size = [3, CROP_SIZE[1], CROP_SIZE[0]]  # [3, 216, 288]
        if hasattr(nc.network_config, "feature_size") and nc.network_config.feature_size is None:
            nc.network_config.feature_size = nc.feature_size

    raw = np.load(scaler_path)
    scaler_values: dict[str, dict[str, torch.Tensor]] = {}
    for full_key in raw.files:
        parts = full_key.rsplit("_", 1)
        if len(parts) != 2:
            continue
        key_name, stat = parts
        if key_name not in scaler_values:
            scaler_values[key_name] = {}
        scaler_values[key_name][stat] = torch.from_numpy(raw[full_key])  # keep on CPU

    # LR scheduler needs num_training_steps (only relevant for training, not inference)
    if hasattr(cfg.agent_config, "lr_scheduler_config"):
        if hasattr(cfg.agent_config.lr_scheduler_config, "num_training_steps"):
            if cfg.agent_config.lr_scheduler_config.num_training_steps is None:
                cfg.agent_config.lr_scheduler_config.num_training_steps = 1

    agent = hydra.utils.instantiate(cfg.agent_config)
    agent.load_pretrained(Path(checkpoint_path))
    agent.model.eval()
    agent.encoder.eval()
    t_obs = int(cfg.t_obs)
    t_pred = int(cfg.t_pred)

    policy = MpdDiffusionPolicy(
        agent=agent,
        scaler_values=scaler_values,
        t_obs=t_obs,
        t_pred=t_pred,
        device=device,
        confirm_before_execute=confirm_before_execute,
        control_hz=control_hz,
        replan_interval_steps=replan_interval_steps,
        latency_step=latency_step,
        ensemble_mode=ensemble_mode,
        ensemble_tau=ensemble_tau,
        max_translation_step_m=max_translation_step_m,
        max_rotation_step_deg=max_rotation_step_deg,
        gripper_close_threshold=gripper_close_threshold,
        gripper_open_threshold=gripper_open_threshold,
        open_gripper_width=open_gripper_width,
        gripper_velocity=gripper_velocity,
        gripper_force_limit=gripper_force_limit,
        open_width_tolerance_m=open_width_tolerance_m,
        open_resend_interval_steps=open_resend_interval_steps,
        max_future_steps=max_future_steps,
        startup_hold_steps=startup_hold_steps,
        debug_timing_log_interval_sec=debug_timing_log_interval_sec,
    )
    LOGGER.info(
        (
            "Loaded MPD diffusion policy: t_obs=%d, t_pred=%d, device=%s, confirm=%s, "
            "replan=%d, latency=%d, ensemble=%s, checkpoint=%s"
        ),
        t_obs,
        t_pred,
        device,
        confirm_before_execute,
        replan_interval_steps,
        latency_step,
        ensemble_mode,
        checkpoint_path,
    )
    return policy


create_policy.__vt_franka_policy_factory__ = True
