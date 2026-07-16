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

import numpy as np

from vt_dual_franka_shared.timing import precise_sleep
from vt_dual_franka_shared.transforms import SingleArmCalibration

from ..collection.controller_state import ControllerStateMonitor
from ..config import InferenceRuntimeSettings, WorkspaceSettings
from ..controller.client import ControllerClient
from ..operator import ManagedUvicornServer, OperatorActionError, OperatorLogBuffer, OperatorSnapshot, create_operator_app
from ..policies.base import Policy
from ..publishers.quest_udp import QuestUdpPublisher
from ..recording import AsyncRolloutVideoRecorder, AsyncStreamVideoRecorder, JsonlStreamRecorder, RunSessionManager
from ..recording.image_io import write_rgb_image
from ..runtime.keys import KeyReader
from ..runtime.live_buffer import LiveSampleBuffer
from ..runtime.motion import RandomizedInitialPose, move_to_eef_pose, move_to_home_joints, sample_randomized_initial_pose
from ..runtime.workers import ThreadWorker, start_thread_worker, stop_thread_workers
from ..sensors.rgb_camera import build_rgb_camera_recorder, resolve_rgb_camera_specs
from .actions import Action, ActionExecutor, action_to_json, normalize_action_chunk
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
        self.eval_task_name, self.eval_model_name = self._resolve_eval_group()
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_buffer = log_buffer or OperatorLogBuffer(workspace.operator_ui.log_buffer_size)
        self.resume_run = resume_run

        self.sessions = RunSessionManager(Path(workspace.recording.eval_root) / self.eval_task_name / self.eval_model_name)
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
        self.action_executor = ActionExecutor(
            controller,
            force_gripper_closed=inference.gripper_forever_closed,
        )
        self.rgb_camera_buffers: dict[str, LiveSampleBuffer] = {}
        self.eval_rgb_camera_buffers: dict[str, LiveSampleBuffer] = {}
        self.eval_camera_stream_names: dict[str, str] = {}
        self.eval_video_recorders: dict[str, AsyncRolloutVideoRecorder] = {}
        self.eval_stream_video_recorders: dict[str, AsyncStreamVideoRecorder] = {}
        self.gelsight_frame_buffer: LiveSampleBuffer | None = None
        self.workers: dict[str, ThreadWorker] = {}
        self.operator_server: ManagedUvicornServer | None = None
        self.assembler = ObservationAssembler(
            modality=inference.modality,
            state_provider=self._get_state_for_observation,
            image_format=workspace.recording.image_format,
            record_rgb_frames=False,
            record_gelsight_frames=False,
        )

        self._operator_lock = threading.RLock()
        self._quit_requested = threading.Event()
        self._current_episode_dir: Path | None = None
        self._latest_saved_episode_dir: Path | None = None
        self._home_joint_completed = False
        self._initial_pose_completed = False
        self._pending_initial_gripper_close = False
        self._current_initial_pose: RandomizedInitialPose | None = None
        self._current_initial_target_tcp: list[float] | None = None
        self._episode_thread: threading.Thread | None = None
        self._episode_stop_event = threading.Event()
        self._episode_error: Exception | None = None
        self._policy_terminated = False
        self._timeout_reached = False
        self._pending_outcome_episode_dir: Path | None = None
        self._last_status_print_wall_time = 0.0

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
            self._poll_episode_status_locked()
            ready, _ = self._is_ready_for_episode_locked()
            return self._build_live_preview_snapshot_locked(name, ready=ready)

    def operator_reset_home_joints(self) -> None:
        with self._operator_lock:
            self._reset_home_joints_locked()

    def operator_reset_ready_pose(self) -> None:
        with self._operator_lock:
            self._move_to_initial_pose_locked()

    def operator_confirm_gripper_closed(self) -> None:
        with self._operator_lock:
            self._confirm_initial_gripper_closed_locked()

    def operator_open_gripper(self) -> None:
        with self._operator_lock:
            self._operator_open_gripper_locked()

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
                create_operator_app(self, self.log_buffer, title="VT Dual Franka Policy Runner"),
                self.workspace.operator_ui.host,
                self.workspace.operator_ui.port,
            )
            self.operator_server.start()

        policy_camera_roles = list(self.inference.modality.rgb_cameras)
        eval_camera_roles = list(self.inference.eval.cameras) if self.inference.eval.enabled else []
        eval_stream_camera_roles = list(self.inference.eval.stream_cameras) if self.inference.eval.enabled else []
        requested_camera_roles = list(
            dict.fromkeys(
                policy_camera_roles
                + [role for role in eval_camera_roles if role != "gelsight"]
                + [role for role in eval_stream_camera_roles if role != "gelsight"]
            )
        )
        rgb_specs = {spec.role: spec for spec in resolve_rgb_camera_specs(self.inference.rgb_cameras)}
        for role in requested_camera_roles:
            if role not in rgb_specs:
                if role in policy_camera_roles:
                    source = "policy modality"
                elif role in eval_camera_roles:
                    source = "eval action-step recording"
                else:
                    source = "eval stream recording"
                raise RuntimeError(f"Inference {source} requested RGB camera role not configured: {role}")
            spec = rgb_specs[role]
            live_buffer = LiveSampleBuffer(spec.stream_name)
            if role in policy_camera_roles:
                self.rgb_camera_buffers[role] = live_buffer
            if role in eval_camera_roles or role in eval_stream_camera_roles:
                self.eval_rgb_camera_buffers[role] = live_buffer
                self.eval_camera_stream_names[role] = spec.stream_name
            if role in eval_camera_roles:
                self.eval_video_recorders[role] = AsyncRolloutVideoRecorder(
                    stream_name=spec.stream_name,
                    output_name=f"rollout_{role}.mp4",
                    fps=self.inference.eval.video_hz,
                )
            if role in eval_stream_camera_roles:
                self.eval_stream_video_recorders[role] = AsyncStreamVideoRecorder(
                    self.sessions,
                    stream_name=spec.stream_name,
                    output_name=f"rollout_{role}.mp4",
                    fps=self.inference.eval.video_hz,
                )
            camera_spec = spec
            stream_video_recorder = self.eval_stream_video_recorders.get(role)
            start_thread_worker(
                self.workers,
                f"rgb_camera:{role}",
                lambda stop_event, spec=camera_spec, live_buffer=live_buffer, stream_video_recorder=stream_video_recorder: build_rgb_camera_recorder(
                    spec,
                    recorder=None,
                    stream_video_recorder=stream_video_recorder,
                    live_buffer=live_buffer,
                    quest_publisher=self.quest_publisher,
                    image_format=self.workspace.recording.image_format,
                ).run(stop_event=stop_event),
                required=True,
            )

        needs_gelsight = self.inference.modality.needs_gelsight() or "gelsight" in eval_camera_roles
        if needs_gelsight:
            from ..sensors.gelsight.publisher import GelsightPublisher

            if not self.inference.gelsight.enabled:
                raise RuntimeError("Inference requested GelSight recording, but inference.gelsight.enabled is false")
            self.gelsight_frame_buffer = LiveSampleBuffer("gelsight_frame")
            if "gelsight" in eval_camera_roles:
                self.eval_camera_stream_names["gelsight"] = "gelsight"
                self.eval_video_recorders["gelsight"] = AsyncRolloutVideoRecorder(
                    stream_name="gelsight",
                    output_name="rollout_gelsight.mp4",
                    fps=self.inference.eval.video_hz,
                )
            gelsight_settings = (
                self.inference.gelsight.model_copy(update={"save_frames": False})
                if "gelsight" in eval_camera_roles
                else self.inference.gelsight
            )
            start_thread_worker(
                self.workers,
                "gelsight",
                lambda stop_event, gelsight_settings=gelsight_settings: GelsightPublisher(
                    gelsight_settings,
                    self.quest_publisher,
                    frame_recorder=None,
                    frame_buffer=self.gelsight_frame_buffer,
                    image_format=self.workspace.recording.image_format,
                ).run(stop_event=stop_event),
                required=True,
            )

        self.assembler = ObservationAssembler(
            modality=self.inference.modality,
            state_provider=self._get_state_for_observation,
            rgb_camera_buffers=self.rgb_camera_buffers,
            gelsight_frame_buffer=self.gelsight_frame_buffer,
            image_format=self.workspace.recording.image_format,
            record_rgb_frames=False,
            record_gelsight_frames=False,
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
            elif command == "j":
                self._run_terminal_action(self.operator_reset_home_joints)
            elif command == "c":
                self._run_terminal_action(self.operator_confirm_gripper_closed)
            elif command == "o":
                self._run_terminal_action(self.operator_open_gripper)
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

    def _reset_home_joints_locked(self) -> None:
        self._poll_episode_status_locked()
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot reset home joints while a policy episode is active. Stop/save it first.")
        if self._pending_outcome_episode_dir is not None:
            raise OperatorActionError(
                f"Mark {self._pending_outcome_episode_dir.name} as success (S) or fail (F) before resetting."
            )
        if self.inference.home_joint_positions_rad is None:
            raise OperatorActionError("Inference home_joint_positions_rad is not configured.")
        LOGGER.info("Resetting robot to policy home joint positions")
        try:
            result = move_to_home_joints(
                controller=self.controller,
                state_provider=self.state_monitor,
                joint_positions=self.inference.home_joint_positions_rad,
                duration_sec=self.inference.home_joint_duration_sec,
                source="policy_runner_home_joints",
                tolerance_rad=self.inference.home_joint_tolerance_rad,
                settle_timeout_sec=self.inference.home_joint_settle_timeout_sec,
                state_max_age_sec=self.inference.modality.controller_state_max_age_sec,
            )
        except Exception as exc:
            raise OperatorActionError(f"Failed to reset home joints: {exc}") from exc
        self._home_joint_completed = True
        self._initial_pose_completed = False
        self._pending_initial_gripper_close = False
        self._current_initial_pose = None
        self._current_initial_target_tcp = None
        self.sessions.record_operator_event(
            "home_joint_reset_completed",
            {
                "joint_positions": list(self.inference.home_joint_positions_rad),
                "duration_sec": self.inference.home_joint_duration_sec,
                "result": result,
            },
        )
        LOGGER.info("Home joint reset complete. Press H to move to the policy initial EEF pose.")

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
            self._pending_initial_gripper_close = False
            self._current_initial_pose = None
            self._current_initial_target_tcp = None
            self.sessions.record_operator_event("initial_pose_skipped", {"reason": "not_configured"})
            return
        LOGGER.info("Moving robot to policy initial EEF pose")
        initial_pose = sample_randomized_initial_pose(
            self.inference.initial_eef_pose_xyz_rpy_deg,
            self.inference.rand_init_pose,
        )
        try:
            target_tcp = move_to_eef_pose(
                controller=self.controller,
                state_provider=self.state_monitor,
                pose_xyz_rpy_deg=initial_pose.pose_xyz_rpy_deg,
                duration_sec=self.inference.initial_move_duration_sec,
                source="policy_runner_initial_pose",
                position_tolerance_m=self.inference.initial_pose_tolerance_m,
                rotation_tolerance_deg=self.inference.initial_pose_tolerance_deg,
                settle_timeout_sec=self.inference.initial_pose_settle_timeout_sec,
                settle_dwell_sec=self.inference.initial_pose_settle_dwell_sec,
                state_max_age_sec=self.inference.modality.controller_state_max_age_sec,
            )
            if self.inference.gripper_forever_closed:
                self._prepare_forever_closed_initial_gripper_locked(target_tcp, initial_pose)
            else:
                self._open_gripper_for_initial_pose_locked()
        except Exception as exc:
            raise OperatorActionError(f"Failed to move robot to initial pose: {exc}") from exc
        self._current_initial_pose = initial_pose
        self._current_initial_target_tcp = target_tcp
        if not self._pending_initial_gripper_close:
            self._initial_pose_completed = True
        self.sessions.record_operator_event(
            "initial_pose_requested",
            {
                "target_tcp": target_tcp,
                "gripper_forever_closed": self.inference.gripper_forever_closed,
                **initial_pose.metadata(),
            },
        )
        if self._pending_initial_gripper_close:
            LOGGER.info("Initial pose reached. Press C to close the gripper before starting the episode.")
            return
        LOGGER.info("Initial pose reached. Ready.")

    def _prepare_forever_closed_initial_gripper_locked(
        self,
        target_tcp: list[float],
        initial_pose: RandomizedInitialPose,
    ) -> None:
        self._initial_pose_completed = False
        self._pending_initial_gripper_close = True
        self.sessions.record_operator_event(
            "initial_gripper_close_pending",
            {
                "target_tcp": target_tcp,
                **initial_pose.metadata(),
            },
        )

    def _confirm_initial_gripper_closed_locked(self) -> None:
        if not self.inference.gripper_forever_closed:
            raise OperatorActionError("gripper_forever_closed is disabled for this inference run.")
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot close initial gripper while a policy episode is active. Stop/save it first.")
        if self._current_initial_pose is None:
            raise OperatorActionError("Move to the policy initial pose with H before closing the gripper.")
        LOGGER.info("Closing gripper for forever-closed policy episode")
        try:
            self.controller.grasp_gripper(
                velocity=self.workspace.teleop.gripper_velocity,
                force_limit=self.workspace.teleop.grasp_force,
                source="policy_runner_initial_gripper_close",
                blocking=True,
            )
        except Exception as exc:
            raise OperatorActionError(f"Failed to close initial gripper: {exc}") from exc
        self._pending_initial_gripper_close = False
        self._initial_pose_completed = True
        self.sessions.record_operator_event(
            "initial_gripper_closed_confirmed",
            {
                "target_tcp": self._current_initial_target_tcp,
                **self._current_initial_pose.metadata(),
            },
        )
        LOGGER.info("Initial gripper closed. Ready.")

    def _open_gripper_for_initial_pose_locked(self) -> None:
        LOGGER.info("Opening gripper for policy initial pose")
        open_width = self._move_gripper_open_locked(source="policy_runner_initial_pose")
        self.sessions.record_operator_event("initial_gripper_open_requested", {"target_width": open_width})

    def _move_gripper_open_locked(self, *, source: str) -> float:
        open_width = float(self.workspace.teleop.max_gripper_width)
        self.controller.move_gripper(
            open_width,
            velocity=self.workspace.teleop.gripper_velocity,
            force_limit=self.workspace.teleop.grasp_force,
            source=source,
            blocking=True,
        )
        self._wait_for_gripper_width_locked(
            target_width=open_width,
            tolerance_m=max(float(self.workspace.teleop.gripper_width_vis_precision), 0.006),
            timeout_sec=5.0,
        )
        return open_width

    def _operator_open_gripper_locked(self) -> None:
        if not self.inference.gripper_forever_closed:
            raise OperatorActionError("gripper_forever_closed is disabled for this run.")
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot open gripper while a policy episode is active. Stop/save it first.")
        if self._current_initial_pose is None:
            raise OperatorActionError("Move to the task initial pose with H before opening the gripper.")
        LOGGER.info("Opening gripper for object adjustment. Press C to close/confirm before starting the episode.")
        try:
            open_width = self._move_gripper_open_locked(source="policy_runner_gripper_adjustment_open")
        except Exception as exc:
            raise OperatorActionError(f"Failed to open gripper for adjustment: {exc}") from exc
        self._pending_initial_gripper_close = True
        self._initial_pose_completed = False
        self.sessions.record_operator_event(
            "gripper_opened_for_adjustment",
            {
                "target_tcp": self._current_initial_target_tcp,
                "open_width": open_width,
                **self._current_initial_pose.metadata(),
            },
        )

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
        LOGGER.info("Loading policy before rollout countdown")
        self.policy.ensure_loaded()
        LOGGER.info("Policy loaded")
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
                "initial_pose": None if self._current_initial_pose is None else self._current_initial_pose.metadata(),
                "initial_target_tcp": self._current_initial_target_tcp,
                "gripper_forever_closed": self.inference.gripper_forever_closed,
            },
        )
        self._current_episode_dir = episode_dir
        self._initial_pose_completed = False
        self._pending_initial_gripper_close = False
        self._episode_error = None
        self._policy_terminated = False
        self._timeout_reached = False
        self._episode_stop_event = threading.Event()
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
        self._home_joint_completed = False
        self._episode_thread = None
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
        for role, recorder in self.eval_video_recorders.items():
            try:
                summary = recorder.flush_episode(episode_dir)
            except Exception as exc:  # pragma: no cover - defensive against disk/codec issues
                LOGGER.warning("Failed to flush eval rollout video for %s: %s", role, exc)
                continue
            if summary is None:
                LOGGER.warning("No eval frames available for %s in %s", role, episode_dir)
                continue
            if summary["dropped_due_to_backpressure"] or summary["write_errors"]:
                LOGGER.warning("Eval rollout recorder summary for %s: %s", role, summary)
            output_path = summary.get("output_path")
            if output_path:
                LOGGER.info("Eval video written: %s", output_path)
        for role, recorder in self.eval_stream_video_recorders.items():
            try:
                summary = recorder.flush_episode(episode_dir)
            except Exception as exc:  # pragma: no cover - defensive against disk/codec issues
                LOGGER.warning("Failed to flush eval stream video for %s: %s", role, exc)
                continue
            if summary is None:
                LOGGER.warning("No eval stream frames available for %s in %s", role, episode_dir)
                continue
            if summary["dropped_due_to_backpressure"] or summary["write_errors"]:
                LOGGER.warning("Eval stream recorder summary for %s: %s", role, summary)
            output_path = summary.get("output_path")
            if output_path:
                LOGGER.info("Eval stream video written: %s", output_path)

    def _discard_latest_episode_locked(self) -> None:
        self._poll_episode_status_locked()
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot discard while a policy episode is active. Stop/save it first.")
        episode_dir = self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()
        if episode_dir is None:
            raise OperatorActionError("No saved policy episode to discard.")
        self._remove_episode_outcome_locked(episode_dir)
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

    def _remove_episode_outcome_locked(self, episode_dir: Path) -> None:
        run_dir = self.sessions.get_active_run_dir()
        if run_dir is None:
            return
        path = run_dir / "episode_outcomes.csv"
        if not path.exists():
            return
        rows: list[tuple[str, str]] = []
        with path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                episode = row.get("episode")
                outcome = row.get("outcome")
                if episode and outcome and episode != episode_dir.name:
                    rows.append((outcome, episode))
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["outcome", "episode"])
            writer.writerows(rows)

    def _is_ready_for_episode_locked(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not self.state_monitor.is_healthy(max_age_sec=self.inference.modality.controller_state_max_age_sec):
            reasons.append("controller state is not healthy")
        ready, modality_reasons = self.assembler.assert_ready()
        del ready
        reasons.extend(modality_reasons)
        if not self._initial_pose_completed:
            reasons.append("robot has not been moved to the policy initial pose with H")
        if self._pending_initial_gripper_close:
            reasons.append("initial gripper close is pending; press C to confirm")
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
        last_valid_action: Action | None = None
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
                hold_stop_event, hold_thread, hold_stats = self._start_inference_hold_stream(
                    last_valid_action,
                    period_sec=period,
                )
                try:
                    model_inputs, model_input_record = self._build_and_record_model_inputs(
                        observation_window,
                        recorded_observation_window,
                        episode_dir=episode_dir,
                        step_index=step_index,
                    )
                    raw_actions = self._predict_policy_actions(observation_window, model_inputs)
                    inference_duration_sec = time.monotonic() - inference_start
                finally:
                    if hold_stop_event is not None:
                        hold_stop_event.set()
                    if hold_thread is not None:
                        hold_thread.join(timeout=max(period * 2.0, 0.1))
                action_chunk = normalize_action_chunk(raw_actions)
                actions_returned_json = [action_to_json(action) for action in action_chunk]
                inference_event = {
                    "step_index": step_index,
                    "chunk_index": step_index // max(self.inference.exe_horizon, 1),
                    "policy_wall_time": time.time(),
                    "policy_monotonic_time": loop_start,
                    "episode_elapsed_sec": elapsed,
                    "obs_horizon": self.inference.obs_horizon,
                    "exe_horizon": self.inference.exe_horizon,
                    "prediction_horizon": len(action_chunk),
                    "observation_window": _summarize_observation_window(recorded_observation_window),
                    "raw_policy_output": _json_safe(raw_actions),
                    "actions_returned": actions_returned_json,
                    "raw_action_vectors_10d": _extract_raw_action_vectors(actions_returned_json),
                    "timing": {
                        "inference_duration_sec": inference_duration_sec,
                        "inference_hold_command_count": hold_stats["commands"],
                    },
                }
                if model_input_record is not None:
                    inference_event["model_input_record"] = model_input_record
                inference_recorder.record_event(inference_event, event_time=time.time())
                actions_to_execute = action_chunk[: self.inference.exe_horizon]
                executed_actions = []
                observations_after_actions = []
                first_observation_step_index = step_index
                for action_index, action in enumerate(actions_to_execute):
                    if self._episode_stop_event.is_set():
                        break
                    action_start = time.monotonic()
                    executed_action = self.action_executor.normalize_for_execution(action)
                    self.action_executor.execute_normalized(executed_action)
                    executed_actions.append(executed_action)
                    if executed_action.target_tcp is not None:
                        last_valid_action = executed_action.model_copy(deep=True)
                    precise_sleep(max(0.0, period - (time.monotonic() - action_start)))
                    observation, recorded_observation = self.assembler.assemble(episode_dir, step_index)
                    history.append(observation)
                    recorded_history.append(recorded_observation)
                    self._record_eval_video_frames(
                        episode_dir=episode_dir,
                        event_time=time.time(),
                        observation=observation,
                    )
                    observations_after_actions.append(
                        {
                            "step_index": step_index,
                            "chunk_action_index": action_index,
                            "observation": recorded_observation,
                        }
                    )
                    step_index += 1
                    if executed_action.terminate:
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

    def _start_inference_hold_stream(
        self,
        last_valid_action: Action | None,
        *,
        period_sec: float,
    ) -> tuple[threading.Event | None, threading.Thread | None, dict[str, int]]:
        hold_action = last_valid_action or self._current_pose_hold_action(period_sec=period_sec)
        stats = {"commands": 0}
        if hold_action is None or hold_action.target_tcp is None:
            return None, None, stats
        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._run_inference_hold_stream,
            name="policy-runner-inference-hold",
            args=(hold_action, stop_event, period_sec, stats),
            daemon=True,
        )
        thread.start()
        return stop_event, thread, stats

    def _current_pose_hold_action(self, *, period_sec: float) -> Action | None:
        try:
            state = self.state_monitor.get_state(max_age_sec=self.inference.modality.controller_state_max_age_sec)
        except Exception:
            LOGGER.exception("Failed to read current pose for policy inference hold stream")
            return None
        return Action(
            target_tcp=list(state.tcp_pose),
            target_duration_sec=max(float(period_sec), 1e-3),
            metadata={"inference_hold": "current_pose"},
        )

    def _run_inference_hold_stream(
        self,
        hold_action: Action,
        stop_event: threading.Event,
        period_sec: float,
        stats: dict[str, int],
    ) -> None:
        target_tcp = list(hold_action.target_tcp or [])
        if len(target_tcp) != 7:
            return
        target_duration_sec = hold_action.target_duration_sec or max(float(period_sec), 1e-3)
        while not stop_event.is_set():
            try:
                self.controller.queue_tcp(
                    target_tcp,
                    source="policy_runner_inference_hold",
                    target_duration_sec=target_duration_sec,
                )
                stats["commands"] += 1
            except Exception:
                LOGGER.exception("Failed to stream policy inference hold command")
                return
            stop_event.wait(max(float(period_sec), 1e-3))

    def _build_and_record_model_inputs(
        self,
        observation_window: list[dict[str, Any]],
        recorded_observation_window: list[dict[str, Any]],
        *,
        episode_dir: Path,
        step_index: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        if not self.inference.model_input_recording.enabled:
            return None, None
        builder = getattr(self.policy, "build_model_inputs", None)
        if builder is None:
            raise RuntimeError(
                "model_input_recording is enabled, but the active policy does not expose build_model_inputs(). "
                "This recording mode is currently intended for visuotactile policies."
            )
        model_inputs = builder(observation_window)
        record = self._record_model_inputs(
            model_inputs,
            recorded_observation_window,
            episode_dir=episode_dir,
            step_index=step_index,
        )
        return model_inputs, record

    def _predict_policy_actions(
        self,
        observation_window: list[dict[str, Any]],
        model_inputs: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        if model_inputs is None:
            return self.policy.predict(observation_window)
        predictor = getattr(self.policy, "predict_from_model_inputs", None)
        if predictor is None:
            raise RuntimeError(
                "model_input_recording built model inputs, but the active policy does not expose "
                "predict_from_model_inputs()."
            )
        return predictor(model_inputs)

    def _record_eval_video_frames(
        self,
        *,
        episode_dir: Path,
        event_time: float,
        observation: dict[str, Any],
    ) -> None:
        if not self.inference.eval.enabled:
            return
        for role, recorder in self.eval_video_recorders.items():
            sample = self._sample_eval_frame(role, observation)
            if sample is None:
                continue
            recorder.record_frame(episode_dir, sample, event_time=event_time)

    def _sample_eval_frame(self, role: str, observation: dict[str, Any]) -> np.ndarray | None:
        if role == "gelsight":
            tactile = observation.get("tactile", {})
            item = tactile.get("gelsight_frame") or tactile.get("tactile_left")
            if isinstance(item, dict):
                return np.asarray(item.get("image")) if item.get("image") is not None else None
            return None
        if role in self.rgb_camera_buffers:
            sample = self.rgb_camera_buffers[role].get_latest_optional(max_age_sec=self.inference.modality.rgb_camera_max_age_sec)
            if sample is not None:
                return np.asarray(sample.data)
        if role in self.eval_rgb_camera_buffers:
            sample = self.eval_rgb_camera_buffers[role].get_latest_optional(max_age_sec=self.inference.modality.rgb_camera_max_age_sec)
            if sample is not None:
                return np.asarray(sample.data)
        images = observation.get("images", {})
        item = images.get(role if role != "third_person" else "third_person")
        if isinstance(item, dict) and item.get("image") is not None:
            return np.asarray(item["image"])
        return None

    def _record_model_inputs(
        self,
        model_inputs: dict[str, Any],
        recorded_observation_window: list[dict[str, Any]],
        *,
        episode_dir: Path,
        step_index: int,
    ) -> dict[str, Any]:
        settings = self.inference.model_input_recording
        stream_dir = episode_dir / "streams" / "model_inputs"
        step_dir = stream_dir / f"step_{step_index:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        image_format = settings.format.lower().lstrip(".")
        record: dict[str, Any] = {
            "step_index": int(step_index),
            "obs_horizon": len(recorded_observation_window),
            "recorded_at_wall_time": time.time(),
            "streams": {},
        }

        for stream in settings.streams:
            if stream not in model_inputs:
                continue
            array = np.asarray(model_inputs[stream])
            stream_record = _model_input_array_summary(array)
            if array.ndim != 4 or array.shape[-1] != 3:
                raise ValueError(f"Model input stream {stream!r} must be [T,H,W,3], got {array.shape}")
            frame_paths: list[str] = []
            for obs_index, image in enumerate(array):
                rel_path = Path("streams") / "model_inputs" / f"step_{step_index:06d}" / f"{stream}_t{obs_index}.{image_format}"
                abs_path = episode_dir / rel_path
                write_rgb_image(abs_path, _unit_image_to_uint8(image), quality=95)
                frame_paths.append(rel_path.as_posix())
            stream_record["frame_paths"] = frame_paths
            stream_record["source_observation_records"] = [
                _source_record_for_model_input(recorded, stream=stream) for recorded in recorded_observation_window
            ]
            record["streams"][stream] = stream_record

        if settings.save_npz:
            npz_rel_path = Path("streams") / "model_inputs" / f"step_{step_index:06d}" / "inputs.npz"
            np.savez_compressed(
                episode_dir / npz_rel_path,
                **{key: np.asarray(value) for key, value in model_inputs.items()},
            )
            record["npz_path"] = npz_rel_path.as_posix()
            record["model_input_keys"] = {
                str(key): _model_input_array_summary(np.asarray(value)) for key, value in model_inputs.items()
            }

        _append_jsonl(episode_dir / "streams" / "model_inputs.jsonl", record)
        return record

    def _get_state_for_observation(self, max_age_sec: float | None = None):
        state = self.state_monitor.get_state(max_age_sec=max_age_sec)
        self.gripper_status.update(state)
        return state

    def _policy_metadata(self) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "policy_spec": f"{self.policy.__class__.__module__}.{self.policy.__class__.__name__}",
        }
        settings = getattr(self.policy, "settings", None)
        for key in (
            "algorithm",
            "policy_name",
            "task_name",
            "model",
            "checkpoint_file",
            "sampling_scheduler",
            "num_inference_steps",
        ):
            if settings is not None and hasattr(settings, key):
                value = getattr(settings, key)
                if value is not None:
                    metadata[key] = str(value) if isinstance(value, Path) else value
        checkpoint_path = getattr(self.policy, "checkpoint_path", None)
        if checkpoint_path is not None:
            metadata["checkpoint_path"] = str(checkpoint_path)
        return metadata

    def _resolve_eval_group(self) -> tuple[str, str]:
        settings = getattr(self.policy, "settings", None)
        policy_type = self.policy.__class__.__name__.replace("Policy", "") or "policy"
        task_name = getattr(settings, "task_name", None) if settings is not None else None
        task_name = task_name or self.inference.task_name
        model_name = getattr(settings, "model", None) if settings is not None else None
        policy_name = getattr(settings, "policy_name", None) if settings is not None else None
        algorithm = getattr(settings, "algorithm", None) if settings is not None else None
        if model_name is None:
            model_spec = getattr(self.policy, "model_spec", None)
            model_name = getattr(model_spec, "name", None)
        if self.policy.__class__.__module__.endswith(".mpd.policy"):
            model_name = policy_name or algorithm or "mpd"
        return _slugify_path_part(task_name or "policy_run"), _slugify_path_part(model_name or policy_name or policy_type)

    def _build_status_locked(self) -> dict:
        ready, reasons = self._is_ready_for_episode_locked()
        preview = self._preview_metadata_locked(ready=ready)
        next_episode_index = self.sessions.get_next_episode_index()
        run_dir = self.sessions.get_active_run_dir()
        return {
            "mode": "run_policy",
            "run_name": self.run_name,
            "task_name": self.eval_task_name,
            "model_name": self.eval_model_name,
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
            "gripper_forever_closed": self.inference.gripper_forever_closed,
            "pending_initial_gripper_close": self._pending_initial_gripper_close,
            "home_joint_completed": self._home_joint_completed,
            "controller_state": self.state_monitor.snapshot(),
            "rgb_cameras": {role: buffer.snapshot() for role, buffer in self.rgb_camera_buffers.items()},
            "eval_rgb_cameras": {role: buffer.snapshot() for role, buffer in self.eval_rgb_camera_buffers.items()},
            "gelsight_frame": None if self.gelsight_frame_buffer is None else self.gelsight_frame_buffer.snapshot(),
            "eval_stream_videos": {role: recorder.snapshot() for role, recorder in self.eval_stream_video_recorders.items()},
            "workers": {
                name: {"alive": worker.is_alive(), "error": None if worker.error is None else str(worker.error)}
                for name, worker in self.workers.items()
            },
            "allowed_actions": {
                "reset_home_joints": self._current_episode_dir is None
                and self._pending_outcome_episode_dir is None
                and self.inference.home_joint_positions_rad is not None,
                "reset": self._current_episode_dir is None
                and self._pending_outcome_episode_dir is None,
                "confirm_gripper_closed": self._current_episode_dir is None
                and self._pending_initial_gripper_close,
                "open_gripper": self._current_episode_dir is None
                and self.inference.gripper_forever_closed
                and self._current_initial_pose is not None,
                "start": ready and self._current_episode_dir is None,
                "stop": self._current_episode_dir is not None,
                "mark_success": self._current_episode_dir is None and self._pending_outcome_episode_dir is not None,
                "mark_fail": self._current_episode_dir is None and self._pending_outcome_episode_dir is not None,
                "discard": self._current_episode_dir is None
                and (self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()) is not None,
                "quit": self._current_episode_dir is None,
            },
            "preview": preview,
            "snapshots": self._snapshot_metadata_from_preview(preview),
            "preview_note": preview["label"] if preview["streaming"] else None,
        }

    def _preview_metadata_locked(self, *, ready: bool) -> dict[str, object]:
        role = self.workspace.operator_ui.preview_camera_role
        sample = self._get_live_preview_sample_locked(role, ready=ready)
        label = f"Live pre-policy {role} view"
        if sample is None:
            return {
                "role": role,
                "available": False,
                "streaming": False,
                "refresh_hz": self.workspace.operator_ui.preview_refresh_hz,
                "label": label,
            }
        return {
            "role": role,
            "available": True,
            "streaming": True,
            "token": f"{role}:{sample.captured_wall_time:.6f}",
            "captured_wall_time": sample.captured_wall_time,
            "refresh_hz": self.workspace.operator_ui.preview_refresh_hz,
            "label": f"{label}: {sample.name}",
        }

    def _snapshot_metadata_from_preview(self, preview: dict[str, object]) -> dict[str, dict[str, object]]:
        role = str(preview["role"])
        metadata = {
            "available": bool(preview["available"]),
        }
        for key in ("token", "captured_wall_time", "label"):
            if key in preview:
                metadata[key] = preview[key]
        return {role: metadata}

    def _build_live_preview_snapshot_locked(self, name: str, *, ready: bool) -> OperatorSnapshot | None:
        sample = self._get_live_preview_sample_locked(name, ready=ready)
        if sample is None:
            return None
        return OperatorSnapshot(
            name=name,
            image=sample.data.copy(),
            captured_wall_time=sample.captured_wall_time,
            label=f"Live pre-policy {name} view: {sample.name}",
            image_format=self.workspace.recording.image_format,
        )

    def _get_live_preview_sample_locked(self, name: str, *, ready: bool):
        del ready
        role = self.workspace.operator_ui.preview_camera_role
        if name != role or self._current_episode_dir is not None:
            return None
        buffer = self.rgb_camera_buffers.get(role) or self.eval_rgb_camera_buffers.get(role)
        if buffer is None:
            return None
        return buffer.get_latest_optional(max_age_sec=self.workspace.operator_ui.snapshot_max_age_sec)

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
            f"gelsight={'on' if self.gelsight_frame_buffer is not None else 'off'}"
        )
        if status["reasons"]:
            summary = f"{summary} reasons={'; '.join(status['reasons'])}"
        print(summary, flush=True)

    def _print_banner(self, run_dir: Path) -> None:
        print(f"Policy Runner run started: {run_dir}", flush=True)
        print(f"Task: {self.inference.task_name}", flush=True)
        print(f"Policy: {self.policy.__class__.__module__}.{self.policy.__class__.__name__}", flush=True)
        print(f"Eval group: {self.eval_task_name}/{self.eval_model_name}/{self.run_name}", flush=True)
        print("Checklist:", flush=True)
        print("- Controller PC: vt-dual-franka-controller is already running", flush=True)
        print("- Required policy inputs are producing fresh samples", flush=True)
        print("- Press H for the policy initial EEF pose. Optional: press J before R when a home-joint reset is needed.", flush=True)
        print("- If you press J, press H again before R.", flush=True)
        if self.inference.gripper_forever_closed:
            print(
                "- Forever-closed gripper is enabled. Press C after every H to close the gripper; "
                "press O between episodes to open for adjustment, then C before R.",
                flush=True,
            )
        print(
            "Hotkeys: J=home joints  H=initial pose  O=open gripper  C=confirm/close gripper  R=start policy  "
            "E=end/save  S=success  F=fail  D=discard last saved  Q=quit",
            flush=True,
        )
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
        for recorder in self.eval_video_recorders.values():
            recorder.close()
        for recorder in self.eval_stream_video_recorders.values():
            recorder.close()
        self.policy.close()
        self.sessions.stop_run()


def _jsonable(value: Any) -> Any:
    return _json_safe(value)


def _extract_raw_action_vectors(actions: list[dict[str, Any]]) -> list[list[float] | None]:
    vectors: list[list[float] | None] = []
    for action in actions:
        metadata = action.get("metadata") or {}
        state = metadata.get("mpd_tcp_state")
        if not isinstance(state, list):
            state = metadata.get("visuotactile_action_row")
        if isinstance(state, list) and len(state) == 10:
            vectors.append([float(value) for value in state])
        else:
            vectors.append(None)
    return vectors


def _summarize_observation_window(observation_window: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for observation in observation_window:
        item: dict[str, Any] = {
            "assembled_wall_time": observation.get("assembled_wall_time"),
        }
        controller_state = observation.get("proprioception", {}).get("controller_state")
        if isinstance(controller_state, dict):
            item["controller_state"] = {
                key: _json_safe(value)
                for key, value in controller_state.items()
                if key in {"tcp_pose", "tcp_velocity", "gripper_width", "gripper_force", "joint_positions", "joint_velocities"}
            }
        image_summary = {
            role: _summarize_image_item(image_item)
            for role, image_item in observation.get("images", {}).items()
            if isinstance(image_item, dict)
        }
        if image_summary:
            item["images"] = image_summary
        tactile_summary = {
            role: _summarize_image_item(tactile_item)
            for role, tactile_item in observation.get("tactile", {}).items()
            if isinstance(tactile_item, dict)
        }
        if tactile_summary:
            item["tactile"] = tactile_summary
        summary.append(item)
    return summary


def _summarize_image_item(item: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in ("captured_wall_time", "frame_width", "frame_height", "metadata"):
        if key in item:
            payload[key] = _json_safe(item[key])
    return payload


def _model_input_array_summary(array: np.ndarray) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "shape": [int(value) for value in array.shape],
        "dtype": str(array.dtype),
    }
    if array.size:
        finite = np.asarray(array, dtype=np.float64)
        payload.update(
            {
                "min": float(np.nanmin(finite)),
                "max": float(np.nanmax(finite)),
                "mean": float(np.nanmean(finite)),
            }
        )
    return payload


def _unit_image_to_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype == np.uint8:
        return np.ascontiguousarray(array)
    return np.clip(np.asarray(array, dtype=np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)


def _source_record_for_model_input(recorded_observation: dict[str, Any], *, stream: str) -> dict[str, Any] | None:
    if stream == "rgb_wrist":
        source = recorded_observation.get("images", {}).get("wrist")
    elif stream == "gelsight":
        tactile = recorded_observation.get("tactile", {})
        source = tactile.get("gelsight_frame") or tactile.get("tactile_left")
    else:
        source = None
    if not isinstance(source, dict):
        return None
    keep = {
        "captured_wall_time",
        "frame_path",
        "frame_width",
        "frame_height",
        "metadata",
    }
    return {key: _json_safe(value) for key, value in source.items() if key in keep}


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=_json_safe))
        handle.write("\n")


def _slugify_path_part(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "policy"
