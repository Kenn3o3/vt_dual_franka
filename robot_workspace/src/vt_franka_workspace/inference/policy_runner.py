from __future__ import annotations

import csv
import json
import logging
import re
import socket
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from vt_franka_shared.timing import precise_sleep
from vt_franka_shared.transforms import SingleArmCalibration

from ..collection.controller_state import ControllerStateMonitor
from ..config import InferenceRuntimeSettings, WorkspaceSettings
from ..controller.client import ControllerClient
from ..operator import ManagedUvicornServer, OperatorActionError, OperatorLogBuffer, OperatorSnapshot, create_operator_app
from ..policies.base import Policy
from ..publishers.quest_udp import QuestUdpPublisher
from ..recording import JsonlStreamRecorder, RunSessionManager
from ..runtime.keys import KeyReader
from ..runtime.live_buffer import LiveSampleBuffer
from ..runtime.motion import move_to_eef_pose
from ..runtime.workers import ThreadWorker, start_thread_worker, stop_thread_workers
from ..sensors.rgb_camera import build_rgb_camera_recorder, resolve_rgb_camera_specs
from .actions import ActionExecutor, action_to_json, normalize_action_chunk
from .eval_video import write_rollout_video
from .observations import ObservationAssembler, ObservationHistory, _json_safe

LOGGER = logging.getLogger(__name__)


class GripperStatusEstimator:
    def __init__(self, settings) -> None:
        self.settings = settings
        self._force = 0.0
        self._width_history: list[float] = []

    def update(self, state) -> None:
        self._force = float(state.gripper_force)
        self._width_history.append(float(state.gripper_width))
        if len(self._width_history) > self.settings.gripper_stability_window:
            self._width_history = self._width_history[-self.settings.gripper_stability_window :]

    def get_status(self) -> dict[str, bool]:
        stable_open = False
        stable_closed = False
        if len(self._width_history) >= self.settings.gripper_stability_window:
            width_variation = max(self._width_history) - min(self._width_history)
            stable_open = self._force < self.settings.gripper_force_open_threshold and width_variation < self.settings.gripper_width_vis_precision
            stable_closed = self._force >= self.settings.gripper_force_close_threshold and width_variation < self.settings.gripper_width_vis_precision
        return {
            "left_gripper_stable_closed": stable_closed,
            "right_gripper_stable_closed": False,
            "left_gripper_stable_open": stable_open,
            "right_gripper_stable_open": True,
        }


class PolicyRunner:
    def __init__(
        self,
        workspace: WorkspaceSettings,
        inference: InferenceRuntimeSettings,
        controller: ControllerClient,
        calibration: SingleArmCalibration,
        policy: Policy,
        *,
        run_name: str | None = None,
        log_buffer: OperatorLogBuffer | None = None,
        resume_run: bool = True,
    ) -> None:
        self.workspace = workspace
        self.inference = inference
        self.controller = controller
        self.calibration = calibration
        self.policy = policy
        self.policy_family, self.policy_name = self._resolve_policy_group()
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_buffer = log_buffer or OperatorLogBuffer(workspace.operator_ui.log_buffer_size)
        self.resume_run = resume_run

        self.sessions = RunSessionManager(Path(workspace.recording.eval_root) / self.policy_family / self.policy_name)
        self.quest_publisher = QuestUdpPublisher(
            quest_ip=workspace.quest_feedback.quest_ip,
            robot_state_udp_port=workspace.quest_feedback.robot_state_udp_port,
            tactile_udp_port=workspace.quest_feedback.tactile_udp_port,
            image_udp_port=workspace.quest_feedback.image_udp_port,
            force_udp_port=workspace.quest_feedback.force_udp_port,
            calibration=calibration,
            force_scale_factor=workspace.quest_feedback.force_scale_factor,
        )
        self.state_monitor = ControllerStateMonitor(
            controller,
            poll_hz=max(inference.controller_state_poll_hz, inference.control_hz),
        )
        self.gripper_status = GripperStatusEstimator(workspace.teleop)
        self.action_executor = ActionExecutor(controller)
        self.rgb_camera_buffers: dict[str, LiveSampleBuffer] = {}
        self.eval_rgb_camera_buffers: dict[str, LiveSampleBuffer] = {}
        self.eval_camera_stream_names: dict[str, str] = {}
        self.gelsight_marker_buffer: LiveSampleBuffer | None = None
        self.gelsight_frame_buffer: LiveSampleBuffer | None = None
        self.workers: dict[str, ThreadWorker] = {}
        self.operator_server: ManagedUvicornServer | None = None
        self.assembler = ObservationAssembler(
            modality=inference.modality,
            state_provider=self._get_state_for_observation,
            image_format=workspace.recording.image_format,
        )

        self._operator_lock = threading.RLock()
        self._quit_requested = threading.Event()
        self._current_episode_dir: Path | None = None
        self._latest_saved_episode_dir: Path | None = None
        self._initial_pose_completed = inference.initial_eef_pose_xyz_rpy_deg is None
        self._episode_thread: threading.Thread | None = None
        self._episode_stop_event = threading.Event()
        self._episode_error: Exception | None = None
        self._policy_terminated = False
        self._timeout_reached = False
        self._pending_outcome_episode_dir: Path | None = None
        self._last_status_print_wall_time = 0.0
        self._frozen_rgb_snapshots: dict[str, OperatorSnapshot] = {}

    def run(self) -> None:
        run_dir = self.sessions.start_run(
            self.run_name,
            metadata={
                "workspace_hostname": socket.gethostname(),
                "controller_host": self.workspace.controller.host,
                "mode": "run_policy",
                "policy_spec": f"{self.policy.__class__.__module__}.{self.policy.__class__.__name__}",
                "policy": self._policy_metadata(),
                "inference": self.inference.model_dump(mode="json"),
            },
            resume=self.resume_run,
        )
        self._latest_saved_episode_dir = self.sessions.get_latest_saved_episode_dir()
        self.sessions.record_operator_event("run_started", {"run_dir": str(run_dir)})
        try:
            self._start_workers()
            self._print_banner(run_dir)
            with KeyReader() as key_reader:
                try:
                    self._run_event_loop(key_reader)
                except KeyboardInterrupt:
                    LOGGER.info("Policy run interrupted")
        finally:
            self._shutdown()

    def get_operator_status(self) -> dict:
        with self._operator_lock:
            self._poll_episode_status_locked()
            status = self._build_status_locked()
        self.sessions.write_latest_status(status)
        return status

    def get_operator_snapshot(self, name: str) -> OperatorSnapshot | None:
        with self._operator_lock:
            return self._frozen_rgb_snapshots.get(name)

    def operator_reset_ready_pose(self) -> None:
        with self._operator_lock:
            self._move_to_initial_pose_locked()

    def operator_start_episode(self) -> None:
        with self._operator_lock:
            self._start_episode_locked()

    def operator_stop_episode(self) -> None:
        with self._operator_lock:
            self._stop_episode_locked()

    def operator_mark_episode_success(self) -> None:
        with self._operator_lock:
            self._mark_latest_episode_outcome_locked("success")

    def operator_mark_episode_fail(self) -> None:
        with self._operator_lock:
            self._mark_latest_episode_outcome_locked("fail")

    def operator_discard_latest_episode(self) -> None:
        with self._operator_lock:
            self._discard_latest_episode_locked()

    def operator_quit(self) -> None:
        with self._operator_lock:
            if self._current_episode_dir is not None:
                raise OperatorActionError("Cannot quit while a policy episode is active. Stop/save it first.")
            self.sessions.record_operator_event("run_quit_requested")
            self._quit_requested.set()

    def _start_workers(self) -> None:
        self.state_monitor.start()
        if self.workspace.operator_ui.enabled:
            self.operator_server = ManagedUvicornServer(
                create_operator_app(self, self.log_buffer, title="VT Franka Policy Runner"),
                self.workspace.operator_ui.host,
                self.workspace.operator_ui.port,
            )
            self.operator_server.start()

        policy_camera_roles = list(self.inference.modality.rgb_cameras)
        eval_camera_roles = list(self.inference.eval.cameras) if self.inference.eval.enabled else []
        requested_camera_roles = list(dict.fromkeys(policy_camera_roles + eval_camera_roles))
        rgb_specs = {spec.role: spec for spec in resolve_rgb_camera_specs(self.inference.rgb_cameras)}
        for role in requested_camera_roles:
            if role not in rgb_specs:
                source = "policy modality" if role in policy_camera_roles else "eval recording"
                raise RuntimeError(f"Inference {source} requested RGB camera role not configured: {role}")
            spec = rgb_specs[role]
            live_buffer = LiveSampleBuffer(spec.stream_name)
            recorder = None
            if role in policy_camera_roles:
                self.rgb_camera_buffers[role] = live_buffer
            if role in eval_camera_roles:
                self.eval_rgb_camera_buffers[role] = live_buffer
                self.eval_camera_stream_names[role] = spec.stream_name
                recorder = JsonlStreamRecorder(self.sessions, spec.stream_name, record_hz=self.inference.eval.video_hz)
            start_thread_worker(
                self.workers,
                f"rgb_camera:{role}",
                lambda stop_event, spec=spec, live_buffer=live_buffer, recorder=recorder: build_rgb_camera_recorder(
                    spec,
                    recorder=recorder,
                    live_buffer=live_buffer,
                    quest_publisher=self.quest_publisher,
                    image_format=self.workspace.recording.image_format,
                ).run(stop_event=stop_event),
                required=True,
            )

        if self.inference.modality.needs_gelsight():
            from ..sensors.gelsight.publisher import GelsightPublisher

            if not self.inference.gelsight.enabled:
                raise RuntimeError("Inference modality requested GelSight, but inference.gelsight.enabled is false")
            self.gelsight_marker_buffer = LiveSampleBuffer("gelsight_markers")
            self.gelsight_frame_buffer = LiveSampleBuffer("gelsight_frame")
            start_thread_worker(
                self.workers,
                "gelsight",
                lambda stop_event: GelsightPublisher(
                    self.inference.gelsight,
                    self.quest_publisher,
                    marker_recorder=None,
                    frame_recorder=None,
                    marker_buffer=self.gelsight_marker_buffer,
                    frame_buffer=self.gelsight_frame_buffer,
                    image_format=self.workspace.recording.image_format,
                    gripper_status_provider=self.gripper_status.get_status,
                ).run(stop_event=stop_event),
                required=True,
            )

        self.assembler = ObservationAssembler(
            modality=self.inference.modality,
            state_provider=self._get_state_for_observation,
            rgb_camera_buffers=self.rgb_camera_buffers,
            gelsight_marker_buffer=self.gelsight_marker_buffer,
            gelsight_frame_buffer=self.gelsight_frame_buffer,
            image_format=self.workspace.recording.image_format,
        )

    def _run_event_loop(self, key_reader: KeyReader) -> None:
        while not self._quit_requested.is_set():
            with self._operator_lock:
                self._poll_episode_status_locked()
            self._print_status_if_needed()
            key = key_reader.read_key(0.1)
            if key is None:
                continue
            command = key.lower()
            if command == "h":
                self._run_terminal_action(self.operator_reset_ready_pose)
            elif command == "r":
                self._run_terminal_action(self.operator_start_episode)
            elif command == "e":
                self._run_terminal_action(self.operator_stop_episode)
            elif command == "s":
                self._run_terminal_action(self.operator_mark_episode_success)
            elif command == "f":
                self._run_terminal_action(self.operator_mark_episode_fail)
            elif command == "d":
                self._handle_terminal_discard(key_reader)
            elif command == "q":
                self._run_terminal_action(self.operator_quit)

    def _run_terminal_action(self, action) -> None:
        try:
            action()
        except OperatorActionError as exc:
            LOGGER.warning("%s", exc)

    def _handle_terminal_discard(self, key_reader: KeyReader) -> None:
        with self._operator_lock:
            self._poll_episode_status_locked()
            episode_dir = self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()
            if self._current_episode_dir is not None:
                LOGGER.warning("Cannot discard while a policy episode is active. Press E first.")
                return
            if episode_dir is None:
                LOGGER.warning("No saved policy episode to discard")
                return
        print(f"Press Enter to confirm discarding {episode_dir.name}, or any other key to cancel.", flush=True)
        key = key_reader.read_key(30.0)
        if key not in ("\n", "\r"):
            LOGGER.info("Discard cancelled")
            return
        self._run_terminal_action(self.operator_discard_latest_episode)

    def _move_to_initial_pose_locked(self) -> None:
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot move to initial pose while a policy episode is active. Stop/save it first.")
        if self._pending_outcome_episode_dir is not None:
            raise OperatorActionError(
                f"Mark {self._pending_outcome_episode_dir.name} as success (S) or fail (F) before resetting."
            )
        if self.inference.initial_eef_pose_xyz_rpy_deg is None:
            self._initial_pose_completed = True
            self.sessions.record_operator_event("initial_pose_skipped", {"reason": "not_configured"})
            return
        LOGGER.info("Moving robot to policy initial EEF pose")
        try:
            target_tcp = move_to_eef_pose(
                controller=self.controller,
                state_provider=self.state_monitor,
                pose_xyz_rpy_deg=self.inference.initial_eef_pose_xyz_rpy_deg,
                duration_sec=self.inference.initial_move_duration_sec,
                source="policy_runner_initial_pose",
                position_tolerance_m=self.inference.initial_pose_tolerance_m,
                rotation_tolerance_deg=self.inference.initial_pose_tolerance_deg,
                settle_timeout_sec=self.inference.initial_pose_settle_timeout_sec,
                settle_dwell_sec=self.inference.initial_pose_settle_dwell_sec,
                state_max_age_sec=self.inference.modality.controller_state_max_age_sec,
            )
            self._open_gripper_for_initial_pose_locked()
        except Exception as exc:
            raise OperatorActionError(f"Failed to move robot to initial pose: {exc}") from exc
        self._initial_pose_completed = True
        self._clear_snapshot_locked()
        self.sessions.record_operator_event("initial_pose_requested", {"target_tcp": target_tcp})
        LOGGER.info("Initial pose reached. Ready.")

    def _open_gripper_for_initial_pose_locked(self) -> None:
        open_width = float(self.workspace.teleop.max_gripper_width)
        LOGGER.info("Opening gripper for policy initial pose")
        self.controller.move_gripper(
            open_width,
            velocity=self.workspace.teleop.gripper_velocity,
            force_limit=self.workspace.teleop.grasp_force,
            source="policy_runner_initial_pose",
            blocking=True,
        )
        self._wait_for_gripper_width_locked(
            target_width=open_width,
            tolerance_m=max(float(self.workspace.teleop.gripper_width_vis_precision), 0.006),
            timeout_sec=5.0,
        )
        self.sessions.record_operator_event("initial_gripper_open_requested", {"target_width": open_width})

    def _wait_for_gripper_width_locked(self, *, target_width: float, tolerance_m: float, timeout_sec: float) -> None:
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        last_width: float | None = None
        min_open_width = max(float(self.workspace.teleop.min_gripper_width), target_width * 0.9)
        while time.monotonic() <= deadline:
            state = self.state_monitor.get_state(max_age_sec=self.inference.modality.controller_state_max_age_sec)
            last_width = float(state.gripper_width)
            if last_width >= min_open_width or abs(last_width - target_width) <= tolerance_m:
                return
            time.sleep(0.05)
        raise RuntimeError(
            f"Initial gripper did not open near {target_width:.4f}m "
            f"(required >= {min_open_width:.4f}m); last width={last_width}"
        )

    def _start_episode_locked(self) -> None:
        self._poll_episode_status_locked()
        ready, reasons = self._is_ready_for_episode_locked()
        if not ready:
            raise OperatorActionError(f"Cannot start policy episode: {'; '.join(reasons)}")
        if self._current_episode_dir is not None:
            raise OperatorActionError("A policy episode is already active.")
        countdown = self.inference.start_countdown_sec
        self.sessions.record_operator_event("episode_start_requested", {"countdown_sec": countdown})
        if countdown > 0.0:
            LOGGER.info("Starting policy episode in %.1f seconds", countdown)
            time.sleep(countdown)
        episode_index = self.sessions.get_next_episode_index()
        episode_dir = self.sessions.start_episode(
            name=f"episode_{episode_index:04d}",
            metadata={
                "inference": self.inference.model_dump(mode="json"),
                "policy_spec": f"{self.policy.__class__.__module__}.{self.policy.__class__.__name__}",
                "policy": self._policy_metadata(),
            },
        )
        self._current_episode_dir = episode_dir
        self._initial_pose_completed = self.inference.initial_eef_pose_xyz_rpy_deg is None
        self._episode_error = None
        self._policy_terminated = False
        self._timeout_reached = False
        self._episode_stop_event = threading.Event()
        self._clear_snapshot_locked()
        self.action_executor.reset()
        self.policy.reset()
        self._episode_thread = threading.Thread(target=self._episode_loop, name="policy-runner-episode", daemon=True)
        self._episode_thread.start()
        self.sessions.record_operator_event("episode_started", {"episode_dir": str(episode_dir)})
        LOGGER.info("Policy episode started: %s", episode_dir)

    def _stop_episode_locked(self) -> None:
        self._poll_episode_status_locked()
        if self._current_episode_dir is None:
            raise OperatorActionError("No active policy episode to stop.")
        self._episode_stop_event.set()
        self._wait_for_episode_finish_locked(manual_stop=True)

    def _wait_for_episode_finish_locked(self, manual_stop: bool = False) -> None:
        episode_thread = self._episode_thread
        if episode_thread is not None:
            episode_thread.join(timeout=max(self.inference.max_duration_sec, 5.0))
        self._finalize_current_episode_locked(manual_stop=manual_stop)

    def _poll_episode_status_locked(self) -> None:
        if self._current_episode_dir is None or self._episode_thread is None:
            return
        if self._episode_thread.is_alive():
            return
        self._finalize_current_episode_locked(manual_stop=False)

    def _finalize_current_episode_locked(self, manual_stop: bool) -> None:
        if self._current_episode_dir is None:
            return
        episode_dir = self._current_episode_dir
        self._current_episode_dir = None
        self._episode_thread = None
        self._clear_snapshot_locked()
        if self._episode_error is not None:
            outcome = "failed"
            termination_reason = "error"
        elif self._timeout_reached:
            outcome = "saved"
            termination_reason = "timeout"
        elif self._policy_terminated:
            outcome = "saved"
            termination_reason = "policy_terminate"
        elif manual_stop or self._episode_stop_event.is_set():
            outcome = "saved"
            termination_reason = "manual_stop"
        else:
            outcome = "saved"
            termination_reason = "completed"
        self.sessions.stop_episode(
            outcome=outcome,
            metadata_updates={"termination_reason": termination_reason},
        )
        if outcome == "saved":
            self._write_eval_videos(episode_dir)
            self._pending_outcome_episode_dir = episode_dir
        self._latest_saved_episode_dir = episode_dir if outcome == "saved" else self._latest_saved_episode_dir
        self.sessions.record_operator_event(
            "episode_stopped",
            {"episode_dir": str(episode_dir), "outcome": outcome, "termination_reason": termination_reason},
        )
        if self._episode_error is not None:
            LOGGER.warning("Policy episode failed: %s", self._episode_error)
        else:
            LOGGER.info("Policy episode saved: %s (%s)", episode_dir, termination_reason)
            if outcome == "saved":
                LOGGER.info("Mark outcome before reset: S=success, F=fail")

    def _write_eval_videos(self, episode_dir: Path) -> None:
        if not self.inference.eval.enabled:
            return
        for role in self.inference.eval.cameras:
            stream_name = self.eval_camera_stream_names.get(role)
            if stream_name is None:
                continue
            output_name = "rollout.mp4" if len(self.inference.eval.cameras) == 1 else f"rollout_{role}.mp4"
            try:
                video_path = write_rollout_video(
                    episode_dir,
                    stream_name=stream_name,
                    output_name=output_name,
                    fps=self.inference.eval.video_hz,
                )
            except Exception as exc:  # pragma: no cover - OpenCV/codec dependent
                LOGGER.warning("Failed to write eval video for %s: %s", role, exc)
                continue
            if video_path is None:
                LOGGER.warning("No eval frames available for %s in %s", role, episode_dir)
                continue
            LOGGER.info("Eval video written: %s", video_path)

    def _discard_latest_episode_locked(self) -> None:
        self._poll_episode_status_locked()
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot discard while a policy episode is active. Stop/save it first.")
        episode_dir = self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()
        if episode_dir is None:
            raise OperatorActionError("No saved policy episode to discard.")
        self.sessions.discard_episode(episode_dir)
        self.sessions.record_operator_event("episode_discarded", {"episode_dir": str(episode_dir)})
        LOGGER.info("Discarded policy episode: %s", episode_dir)
        self._latest_saved_episode_dir = self.sessions.get_latest_saved_episode_dir()
        if self._pending_outcome_episode_dir == episode_dir:
            self._pending_outcome_episode_dir = None

    def _mark_latest_episode_outcome_locked(self, outcome: str) -> None:
        self._poll_episode_status_locked()
        if outcome not in {"success", "fail"}:
            raise ValueError(f"Unsupported policy episode outcome: {outcome}")
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot mark outcome while a policy episode is active. Press E first.")
        episode_dir = self._pending_outcome_episode_dir or self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()
        if episode_dir is None:
            raise OperatorActionError("No saved policy episode to mark.")
        self._write_episode_outcome_locked(episode_dir, outcome)
        self._pending_outcome_episode_dir = None
        self.sessions.record_operator_event("episode_outcome_marked", {"episode": episode_dir.name, "outcome": outcome})
        LOGGER.info("Marked %s as %s", episode_dir.name, outcome)

    def _write_episode_outcome_locked(self, episode_dir: Path, outcome: str) -> None:
        run_dir = self.sessions.get_active_run_dir()
        if run_dir is None:
            raise RuntimeError("No active policy run for outcome logging")
        path = run_dir / "episode_outcomes.csv"
        rows: list[tuple[str, str]] = []
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    episode = row.get("episode")
                    existing_outcome = row.get("outcome")
                    if episode and existing_outcome and episode != episode_dir.name:
                        rows.append((existing_outcome, episode))
        rows.append((outcome, episode_dir.name))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["outcome", "episode"])
            writer.writerows(rows)
        manifest_path = episode_dir / "episode_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("metadata", {})["operator_outcome"] = outcome
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _is_ready_for_episode_locked(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not self.state_monitor.is_healthy(max_age_sec=self.inference.modality.controller_state_max_age_sec):
            reasons.append("controller state is not healthy")
        ready, modality_reasons = self.assembler.assert_ready()
        del ready
        reasons.extend(modality_reasons)
        if not self._initial_pose_completed:
            reasons.append("robot has not been moved to the policy initial pose with H")
        if self._pending_outcome_episode_dir is not None:
            reasons.append(f"mark {self._pending_outcome_episode_dir.name} as success (S) or fail (F)")
        for name, worker in self.workers.items():
            if worker.required and worker.error is not None:
                reasons.append(f"{name} failed: {worker.error}")
        return not reasons, reasons

    def _episode_loop(self) -> None:
        assert self._current_episode_dir is not None
        episode_dir = self._current_episode_dir
        recorder = JsonlStreamRecorder(self.sessions, "policy_steps")
        inference_recorder = JsonlStreamRecorder(self.sessions, "policy_inference")
        history = ObservationHistory(self.inference.obs_horizon)
        recorded_history = ObservationHistory(self.inference.obs_horizon)
        period = 1.0 / max(self.inference.control_hz, 1e-6)
        start_monotonic = time.monotonic()
        step_index = 0
        try:
            initial_observation, initial_recorded = self.assembler.assemble(episode_dir, step_index)
            history.initialize_with_padding(initial_observation)
            recorded_history.initialize_with_padding(initial_recorded)
            recorder.record_event(
                {
                    "step_index": step_index,
                    "phase": "initial_padding",
                    "observation": initial_recorded,
                    "obs_horizon": self.inference.obs_horizon,
                },
                event_time=time.time(),
            )
            self.policy.start_episode(history.window())
            step_index += 1
        except Exception as exc:
            self._episode_error = exc
            LOGGER.exception("Policy initial observation failed")
            return

        while not self._episode_stop_event.is_set():
            loop_start = time.monotonic()
            elapsed = loop_start - start_monotonic
            if elapsed >= self.inference.max_duration_sec:
                self._timeout_reached = True
                break
            try:
                inference_start = time.monotonic()
                observation_window = history.window()
                recorded_observation_window = recorded_history.window()
                raw_actions = self.policy.predict(observation_window)
                inference_duration_sec = time.monotonic() - inference_start
                action_chunk = normalize_action_chunk(raw_actions)
                actions_returned_json = [action_to_json(action) for action in action_chunk]
                inference_recorder.record_event(
                    {
                        "step_index": step_index,
                        "chunk_index": step_index // max(self.inference.exe_horizon, 1),
                        "policy_wall_time": time.time(),
                        "policy_monotonic_time": loop_start,
                        "episode_elapsed_sec": elapsed,
                        "obs_horizon": self.inference.obs_horizon,
                        "exe_horizon": self.inference.exe_horizon,
                        "prediction_horizon": len(action_chunk),
                        "raw_observation_window": _json_safe(recorded_observation_window),
                        "raw_policy_output": _json_safe(raw_actions),
                        "actions_returned": actions_returned_json,
                        "raw_action_vectors_10d": _extract_raw_action_vectors(actions_returned_json),
                        "timing": {
                            "inference_duration_sec": inference_duration_sec,
                        },
                    },
                    event_time=time.time(),
                )
                actions_to_execute = action_chunk[: self.inference.exe_horizon]
                executed_actions = []
                observations_after_actions = []
                first_observation_step_index = step_index
                for action_index, action in enumerate(actions_to_execute):
                    if self._episode_stop_event.is_set():
                        break
                    self.action_executor.execute(action)
                    executed_actions.append(action)
                    precise_sleep(period)
                    observation, recorded_observation = self.assembler.assemble(episode_dir, step_index)
                    history.append(observation)
                    recorded_history.append(recorded_observation)
                    observations_after_actions.append(
                        {
                            "step_index": step_index,
                            "chunk_action_index": action_index,
                            "observation": recorded_observation,
                        }
                    )
                    step_index += 1
                    if action.terminate:
                        self._policy_terminated = True
                        break
                executed_actions_json = [action_to_json(action) for action in executed_actions]
                self.policy.observe_executed_actions(executed_actions_json)
                recorder.record_event(
                    {
                        "step_index": first_observation_step_index,
                        "phase": "policy_chunk",
                        "policy_wall_time": time.time(),
                        "policy_monotonic_time": loop_start,
                        "episode_elapsed_sec": elapsed,
                        "observations_after_actions": observations_after_actions,
                        "actions_returned": actions_returned_json,
                        "actions_executed": executed_actions_json,
                        "timing": {
                            "inference_duration_sec": inference_duration_sec,
                            "loop_duration_sec": time.monotonic() - loop_start,
                        },
                        "termination_flag": self._policy_terminated,
                    },
                    event_time=time.time(),
                )
                if self._policy_terminated:
                    break
            except Exception as exc:  # pragma: no cover - hardware dependent failure path
                self._episode_error = exc
                LOGGER.exception("Policy episode iteration failed")
                break

    def _get_state_for_observation(self, max_age_sec: float | None = None):
        state = self.state_monitor.get_state(max_age_sec=max_age_sec)
        self.gripper_status.update(state)
        return state

    def _policy_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "policy_spec": f"{self.policy.__class__.__module__}.{self.policy.__class__.__name__}",
        }
        settings = getattr(self.policy, "settings", None)
        for key in ("algorithm", "policy_name", "task_name"):
            if settings is not None and hasattr(settings, key):
                value = getattr(settings, key)
                if value is not None:
                    metadata[key] = value
        checkpoint_path = getattr(self.policy, "checkpoint_path", None)
        if checkpoint_path is not None:
            metadata["checkpoint_path"] = str(checkpoint_path)
        return metadata

    def _resolve_policy_group(self) -> tuple[str, str]:
        settings = getattr(self.policy, "settings", None)
        policy_type = self.policy.__class__.__name__.replace("Policy", "") or "policy"
        family = getattr(settings, "family", None) if settings is not None else None
        policy_name = getattr(settings, "policy_name", None) if settings is not None else None
        algorithm = getattr(settings, "algorithm", None) if settings is not None else None
        if self.policy.__class__.__module__.endswith(".mpd.policy"):
            family = "mpd"
            policy_name = policy_name or algorithm
        return _slugify_path_part(family or policy_type), _slugify_path_part(policy_name or policy_type)

    def _build_status_locked(self) -> dict:
        ready, reasons = self._is_ready_for_episode_locked()
        self._update_frozen_snapshot_locked(ready=ready)
        next_episode_index = self.sessions.get_next_episode_index()
        run_dir = self.sessions.get_active_run_dir()
        return {
            "mode": "run_policy",
            "run_name": self.run_name,
            "policy_family": self.policy_family,
            "policy_name": self.policy_name,
            "run_dir": None if run_dir is None else str(run_dir),
            "ready": ready,
            "reasons": reasons,
            "active_episode": None if self._current_episode_dir is None else str(self._current_episode_dir),
            "active_episode_name": None if self._current_episode_dir is None else self._current_episode_dir.name,
            "latest_saved_episode": None if self._latest_saved_episode_dir is None else str(self._latest_saved_episode_dir),
            "latest_saved_episode_name": None if self._latest_saved_episode_dir is None else self._latest_saved_episode_dir.name,
            "pending_outcome_episode": None if self._pending_outcome_episode_dir is None else str(self._pending_outcome_episode_dir),
            "pending_outcome_episode_name": None
            if self._pending_outcome_episode_dir is None
            else self._pending_outcome_episode_dir.name,
            "next_episode_index": next_episode_index,
            "next_episode_name": f"episode_{next_episode_index:04d}",
            "quest_connected": None,
            "teleop_enabled": None,
            "controller_state": self.state_monitor.snapshot(),
            "rgb_cameras": {role: buffer.snapshot() for role, buffer in self.rgb_camera_buffers.items()},
            "eval_rgb_cameras": {role: buffer.snapshot() for role, buffer in self.eval_rgb_camera_buffers.items()},
            "gelsight_markers": None if self.gelsight_marker_buffer is None else self.gelsight_marker_buffer.snapshot(),
            "gelsight_frame": None if self.gelsight_frame_buffer is None else self.gelsight_frame_buffer.snapshot(),
            "workers": {
                name: {"alive": worker.is_alive(), "error": None if worker.error is None else str(worker.error)}
                for name, worker in self.workers.items()
            },
            "allowed_actions": {
                "reset": self._current_episode_dir is None and self._pending_outcome_episode_dir is None,
                "start": ready and self._current_episode_dir is None,
                "stop": self._current_episode_dir is not None,
                "mark_success": self._current_episode_dir is None and self._pending_outcome_episode_dir is not None,
                "mark_fail": self._current_episode_dir is None and self._pending_outcome_episode_dir is not None,
                "discard": self._current_episode_dir is None
                and (self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()) is not None,
                "quit": self._current_episode_dir is None,
            },
            "snapshots": self._snapshot_metadata_locked(),
            "preview_note": None if not self._frozen_rgb_snapshots else "frozen idle RGB snapshots",
        }

    def _snapshot_metadata_locked(self) -> dict[str, dict[str, object]]:
        metadata: dict[str, dict[str, object]] = {}
        for role in dict.fromkeys([*self.rgb_camera_buffers, *self.eval_rgb_camera_buffers]):
            snapshot = self._frozen_rgb_snapshots.get(role)
            if snapshot is None:
                metadata[role] = {"available": False}
                continue
            metadata[role] = {
                "available": True,
                "token": snapshot.token,
                "captured_wall_time": snapshot.captured_wall_time,
                "label": snapshot.label,
            }
        return metadata

    def _clear_snapshot_locked(self) -> None:
        self._frozen_rgb_snapshots.clear()

    def _update_frozen_snapshot_locked(self, *, ready: bool) -> None:
        snapshot_buffers = self.rgb_camera_buffers or self.eval_rgb_camera_buffers
        if self._current_episode_dir is not None or not ready or not snapshot_buffers:
            self._clear_snapshot_locked()
            return
        for role, buffer in snapshot_buffers.items():
            if role in self._frozen_rgb_snapshots:
                continue
            sample = buffer.get_latest_optional(max_age_sec=self.workspace.operator_ui.snapshot_max_age_sec)
            if sample is None:
                continue
            self._frozen_rgb_snapshots[role] = OperatorSnapshot(
                name=role,
                image=sample.data.copy(),
                captured_wall_time=sample.captured_wall_time,
                label=f"Frozen pre-policy RGB view ({role}): {sample.name}",
                image_format=self.workspace.recording.image_format,
            )

    def _print_status_if_needed(self) -> None:
        now = time.time()
        if now - self._last_status_print_wall_time < 1.0 / max(self.inference.status_print_hz, 1e-6):
            return
        self._last_status_print_wall_time = now
        with self._operator_lock:
            self._poll_episode_status_locked()
            status = self._build_status_locked()
        self.sessions.write_latest_status(status)
        summary = (
            f"[{'READY' if status['ready'] else 'WAIT'}] "
            f"next={status['next_episode_name']} "
            f"recording={status['active_episode_name'] or 'off'} "
            f"controller_age={status['controller_state']['age_sec'] if status['controller_state']['age_sec'] is not None else 'n/a'} "
            f"rgb_cameras={len(self.rgb_camera_buffers)} "
            f"eval_cameras={len(self.eval_rgb_camera_buffers)} "
            f"gelsight={'on' if self.gelsight_marker_buffer is not None or self.gelsight_frame_buffer is not None else 'off'}"
        )
        if status["reasons"]:
            summary = f"{summary} reasons={'; '.join(status['reasons'])}"
        print(summary, flush=True)

    def _print_banner(self, run_dir: Path) -> None:
        print(f"Policy Runner run started: {run_dir}", flush=True)
        print(f"Task: {self.inference.task_name}", flush=True)
        print(f"Policy: {self.policy.__class__.__module__}.{self.policy.__class__.__name__}", flush=True)
        print(f"Eval group: {self.policy_family}/{self.policy_name}/{self.run_name}", flush=True)
        print("Checklist:", flush=True)
        print("- Controller PC: vt-franka-controller is already running", flush=True)
        print("- Required policy inputs are producing fresh samples", flush=True)
        print("- Press H to move to the policy initial EEF pose when one is configured", flush=True)
        print("Hotkeys: H=initial pose  R=start policy  E=end/save  S=success  F=fail  D=discard last saved  Q=quit", flush=True)
        if self.workspace.operator_ui.enabled:
            print(
                f"Operator UI: http://{self.workspace.operator_ui.host}:{self.workspace.operator_ui.port}/operator",
                flush=True,
            )

    def _shutdown(self) -> None:
        with self._operator_lock:
            if self._current_episode_dir is not None:
                self._episode_stop_event.set()
                self._wait_for_episode_finish_locked(manual_stop=True)
        self.sessions.record_operator_event("run_stopped")
        stop_thread_workers(self.workers)
        if self.operator_server is not None:
            self.operator_server.stop()
        self.state_monitor.stop()
        self.policy.close()
        self.sessions.stop_run()


def _jsonable(value: Any) -> Any:
    return _json_safe(value)


def _extract_raw_action_vectors(actions: list[dict[str, Any]]) -> list[list[float] | None]:
    vectors: list[list[float] | None] = []
    for action in actions:
        metadata = action.get("metadata") or {}
        state = metadata.get("mpd_tcp_state")
        if isinstance(state, list) and len(state) == 10:
            vectors.append([float(value) for value in state])
        else:
            vectors.append(None)
    return vectors


def _slugify_path_part(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "policy"
