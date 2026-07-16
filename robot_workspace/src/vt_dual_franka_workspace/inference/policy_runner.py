from __future__ import annotations

import logging
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from vt_dual_franka_shared.models import ArmId, ResetCommand
from vt_dual_franka_shared.timing import precise_sleep
from vt_dual_franka_shared.transforms import BimanualCalibration

from ..config import InferenceRuntimeSettings, WorkspaceSettings
from ..operator import ManagedUvicornServer, OperatorActionError, OperatorLogBuffer, OperatorSnapshot, create_operator_app
from ..policies.base import Policy
from ..publishers.quest_udp import QuestUdpPublisher
from ..publishers.state_bridge import DualStateBridge
from ..recording import JsonlStreamRecorder, RunSessionManager
from ..runtime.dual_arm import ARM_ORDER, DualArmCoordinator
from ..runtime.keys import KeyReader
from ..runtime.live_buffer import LiveSampleBuffer
from ..runtime.motion import RandomizedInitialPose, eef_xyz_rpy_deg_to_tcp_pose, sample_randomized_initial_pose
from ..runtime.workers import ThreadWorker, start_thread_worker, stop_thread_workers
from ..sensors.rgb_camera import build_rgb_camera_recorder, resolve_rgb_camera_specs
from .actions import Action, DualActionExecutor, action_to_json, normalize_action_chunk
from .bimanual_observations import BimanualObservationAssembler
from .history import ObservationHistory

LOGGER = logging.getLogger(__name__)


class BimanualPolicyRunner:
    """Runs a 20D bimanual policy against both arm controllers."""

    def __init__(
        self,
        workspace: WorkspaceSettings,
        inference: InferenceRuntimeSettings,
        coordinator: DualArmCoordinator,
        calibration: BimanualCalibration,
        policy: Policy,
        *,
        run_name: str | None = None,
        log_buffer: OperatorLogBuffer | None = None,
        resume_run: bool = True,
    ) -> None:
        self.workspace = workspace
        self.inference = inference
        self.coordinator = coordinator
        self.calibration = calibration
        self.policy = policy
        self.run_name = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_buffer = log_buffer or OperatorLogBuffer(workspace.operator_ui.log_buffer_size)
        self.resume_run = resume_run

        self.sessions = RunSessionManager(Path(workspace.recording.eval_root) / inference.task_name / "dp_bimanual")
        self.quest_publisher = QuestUdpPublisher(
            quest_ip=workspace.quest_feedback.quest_ip,
            robot_state_udp_port=workspace.quest_feedback.robot_state_udp_port,
            tactile_udp_port=workspace.quest_feedback.tactile_udp_port,
            image_udp_port=workspace.quest_feedback.image_udp_port,
            force_udp_port=workspace.quest_feedback.force_udp_port,
            calibration=calibration,
            force_scale_factor=workspace.quest_feedback.force_scale_factor,
        )
        self.state_bridge: DualStateBridge | None = None
        self.action_executor = DualActionExecutor(coordinator)
        self.rgb_camera_buffers: dict[str, LiveSampleBuffer] = {}
        self.tactile_buffers: dict[ArmId, LiveSampleBuffer] = {}
        self.workers: dict[str, ThreadWorker] = {}
        self.operator_server: ManagedUvicornServer | None = None
        self.assembler = BimanualObservationAssembler(
            coordinator=coordinator,
            rgb_camera_buffers=self.rgb_camera_buffers,
            tactile_buffers=self.tactile_buffers,
            image_format=workspace.recording.image_format,
            state_max_age_sec=inference.modality.controller_state_max_age_sec,
            image_max_age_sec=inference.modality.rgb_camera_max_age_sec,
            tactile_max_age_sec=inference.modality.gelsight_max_age_sec,
        )

        self._operator_lock = threading.RLock()
        self._quit_requested = threading.Event()
        self._current_episode_dir: Path | None = None
        self._latest_saved_episode_dir: Path | None = None
        self._pending_outcome_episode_dir: Path | None = None
        self._episode_thread: threading.Thread | None = None
        self._episode_stop_event = threading.Event()
        self._episode_error: Exception | None = None
        self._policy_terminated = False
        self._timeout_reached = False
        self._initial_pose_completed = False
        self._pending_initial_gripper_close = False
        self._current_initial_poses: dict[ArmId, RandomizedInitialPose] | None = None
        self._current_initial_targets: dict[ArmId, list[float]] | None = None
        self._last_status_print_wall_time = 0.0

    def run(self) -> None:
        run_dir = self.sessions.start_run(
            self.run_name,
            metadata={
                "schema_version": "vt_dual_franka_policy_run_v1",
                "workspace_hostname": socket.gethostname(),
                "mode": "bimanual_policy",
                "controller_endpoints": {
                    arm: self.workspace.arms[arm].model_dump(mode="json") for arm in ARM_ORDER
                },
                "inference": self.inference.model_dump(mode="json"),
                "policy": f"{self.policy.__class__.__module__}.{self.policy.__class__.__name__}",
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
                    LOGGER.info("Bimanual policy run interrupted")
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
            sample = self._preview_sample(name)
            if sample is None:
                return None
            return OperatorSnapshot(
                name=name,
                image=sample.data.copy(),
                captured_wall_time=sample.captured_wall_time,
                label=f"Live bimanual policy preview: {name}",
                image_format=self.workspace.recording.image_format,
            )

    def operator_reset_home_joints(self) -> None:
        with self._operator_lock:
            self._assert_can_reset()
            poses = self._required_initial_poses()
            missing = [arm for arm in ARM_ORDER if poses[arm].joint_positions_rad is None]
            if missing:
                raise OperatorActionError(f"Inference joint poses are not configured for: {missing}")
            commands = {
                arm: ResetCommand(
                    profile="bimanual_policy_home_joints",
                    joint_positions=list(poses[arm].joint_positions_rad or []),
                    joint_duration_sec=self.inference.home_joint_duration_sec,
                    source="bimanual_policy_home_joints",
                )
                for arm in ARM_ORDER
            }
            try:
                self.coordinator.reset_pair(commands)
                self._wait_for_joint_targets()
            except Exception as exc:
                raise OperatorActionError(f"Failed to reset both policy arms to joint poses: {exc}") from exc
            self._clear_initial_pose_state()

    def operator_reset_ready_pose(self) -> None:
        with self._operator_lock:
            self._assert_can_reset()
            poses = self._required_initial_poses()
            sampled = {
                arm: sample_randomized_initial_pose(
                    poses[arm].eef_pose_xyz_rpy_deg,
                    poses[arm].random_xyz_range_m,
                )
                for arm in ARM_ORDER
            }
            commands = {
                arm: ResetCommand(
                    profile="bimanual_policy_initial_pose",
                    eef_pose_xyz_rpy_deg=list(sampled[arm].pose_xyz_rpy_deg),
                    eef_duration_sec=self.inference.initial_move_duration_sec,
                    source="bimanual_policy_initial_pose",
                )
                for arm in ARM_ORDER
            }
            try:
                self.coordinator.reset_pair(commands)
                if self.inference.gripper_forever_closed:
                    self._pending_initial_gripper_close = True
                    self._initial_pose_completed = False
                else:
                    self.coordinator.move_grippers(
                        {arm: self.workspace.teleop.max_gripper_width for arm in ARM_ORDER},
                        velocity=self.workspace.teleop.gripper_velocity,
                        force_limit=self.workspace.teleop.grasp_force,
                        source="bimanual_policy_initial_open",
                        blocking=True,
                    )
                    self._pending_initial_gripper_close = False
                    self._initial_pose_completed = True
            except Exception as exc:
                raise OperatorActionError(f"Failed to move both policy arms to initial poses: {exc}") from exc
            self._current_initial_poses = sampled
            self._current_initial_targets = {
                arm: eef_xyz_rpy_deg_to_tcp_pose(sampled[arm].pose_xyz_rpy_deg) for arm in ARM_ORDER
            }

    def operator_confirm_gripper_closed(self) -> None:
        with self._operator_lock:
            if not self.inference.gripper_forever_closed:
                raise OperatorActionError("gripper_forever_closed is disabled.")
            self._assert_can_reset()
            if self._current_initial_poses is None:
                raise OperatorActionError("Press H before closing both grippers.")
            try:
                self.coordinator.grasp_grippers(
                    velocity=self.workspace.teleop.gripper_velocity,
                    force_limit=self.workspace.teleop.grasp_force,
                    source="bimanual_policy_initial_close",
                    blocking=True,
                )
            except Exception as exc:
                raise OperatorActionError(f"Failed to close both grippers: {exc}") from exc
            self._pending_initial_gripper_close = False
            self._initial_pose_completed = True

    def operator_open_gripper(self) -> None:
        with self._operator_lock:
            self._assert_can_reset()
            try:
                self.coordinator.move_grippers(
                    {arm: self.workspace.teleop.max_gripper_width for arm in ARM_ORDER},
                    velocity=self.workspace.teleop.gripper_velocity,
                    force_limit=self.workspace.teleop.grasp_force,
                    source="bimanual_policy_adjustment_open",
                    blocking=True,
                )
            except Exception as exc:
                raise OperatorActionError(f"Failed to open both grippers: {exc}") from exc
            if self.inference.gripper_forever_closed:
                self._pending_initial_gripper_close = True
                self._initial_pose_completed = False

    def operator_start_episode(self) -> None:
        with self._operator_lock:
            self._start_episode_locked()

    def operator_stop_episode(self) -> None:
        with self._operator_lock:
            self._poll_episode_status_locked()
            if self._current_episode_dir is None:
                raise OperatorActionError("No active policy episode.")
            self._episode_stop_event.set()
            self._finish_episode_locked(manual_stop=True)

    def operator_mark_episode_success(self) -> None:
        with self._operator_lock:
            self._mark_outcome_locked("success")

    def operator_mark_episode_fail(self) -> None:
        with self._operator_lock:
            self._mark_outcome_locked("fail")

    def operator_discard_latest_episode(self) -> None:
        with self._operator_lock:
            self._assert_no_active_episode()
            episode_dir = self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()
            if episode_dir is None:
                raise OperatorActionError("No saved policy episode to discard.")
            self.sessions.discard_episode(episode_dir)
            self._latest_saved_episode_dir = self.sessions.get_latest_saved_episode_dir()
            if self._pending_outcome_episode_dir == episode_dir:
                self._pending_outcome_episode_dir = None

    def operator_quit(self) -> None:
        with self._operator_lock:
            self._assert_no_active_episode()
            self._quit_requested.set()

    def _start_workers(self) -> None:
        self.coordinator.start()
        if self.workspace.operator_ui.enabled:
            self.operator_server = ManagedUvicornServer(
                create_operator_app(self, self.log_buffer, title="VT Dual Franka Bimanual Policy"),
                self.workspace.operator_ui.host,
                self.workspace.operator_ui.port,
            )
            self.operator_server.start()
        self.state_bridge = DualStateBridge(
            self.coordinator,
            self.quest_publisher,
            self.workspace.quest_feedback,
        )
        self.state_bridge.start()
        self._start_rgb_workers()
        self._start_tactile_workers()
        self.assembler = BimanualObservationAssembler(
            coordinator=self.coordinator,
            rgb_camera_buffers=self.rgb_camera_buffers,
            tactile_buffers=self.tactile_buffers,
            image_format=self.workspace.recording.image_format,
            state_max_age_sec=self.inference.modality.controller_state_max_age_sec,
            image_max_age_sec=self.inference.modality.rgb_camera_max_age_sec,
            tactile_max_age_sec=self.inference.modality.gelsight_max_age_sec,
        )

    def _start_rgb_workers(self) -> None:
        specs = {spec.role: spec for spec in resolve_rgb_camera_specs(self.inference.rgb_cameras)}
        required_roles = ("left_wrist", "right_wrist")
        for role in required_roles:
            if role not in self.inference.modality.rgb_cameras or role not in specs:
                raise RuntimeError(f"Bimanual policy requires configured RGB camera role: {role}")
            spec = specs[role]
            buffer = LiveSampleBuffer(spec.stream_name)
            self.rgb_camera_buffers[spec.stream_name] = buffer
            service = build_rgb_camera_recorder(
                spec,
                live_buffer=buffer,
                quest_publisher=self.quest_publisher,
                image_format=self.workspace.recording.image_format,
            )
            start_thread_worker(
                self.workers,
                f"rgb_camera:{role}",
                lambda stop_event, service=service: service.run(stop_event=stop_event),
                required=True,
            )

    def _start_tactile_workers(self) -> None:
        if not self.inference.modality.needs_gelsight():
            raise RuntimeError("Bimanual policy requires both tactile streams")
        from ..sensors.gelsight.publisher import GelsightPublisher

        for arm in ARM_ORDER:
            settings = self.inference.gelsights[arm]
            if not settings.enabled:
                raise RuntimeError(f"Bimanual policy requires enabled {arm} GelSight")
            buffer = LiveSampleBuffer(f"tactile_{arm}")
            self.tactile_buffers[arm] = buffer
            service = GelsightPublisher(
                settings,
                self.quest_publisher,
                frame_buffer=buffer,
                image_format=self.workspace.recording.image_format,
            )
            start_thread_worker(
                self.workers,
                f"gelsight:{arm}",
                lambda stop_event, service=service: service.run(stop_event=stop_event),
                required=True,
            )

    def _run_event_loop(self, key_reader: KeyReader) -> None:
        while not self._quit_requested.is_set():
            with self._operator_lock:
                self._poll_episode_status_locked()
            self._print_status_if_needed()
            key = key_reader.read_key(0.1)
            if key is None:
                continue
            action = {
                "h": self.operator_reset_ready_pose,
                "j": self.operator_reset_home_joints,
                "c": self.operator_confirm_gripper_closed,
                "o": self.operator_open_gripper,
                "r": self.operator_start_episode,
                "e": self.operator_stop_episode,
                "s": self.operator_mark_episode_success,
                "f": self.operator_mark_episode_fail,
                "d": self.operator_discard_latest_episode,
                "q": self.operator_quit,
            }.get(key.lower())
            if action is not None:
                try:
                    action()
                except OperatorActionError as exc:
                    LOGGER.warning("%s", exc)

    def _start_episode_locked(self) -> None:
        self._poll_episode_status_locked()
        ready, reasons = self._is_ready_for_episode_locked()
        if not ready:
            raise OperatorActionError(f"Cannot start policy episode: {'; '.join(reasons)}")
        LOGGER.info("Loading bimanual policy")
        self.policy.ensure_loaded()
        if self.inference.start_countdown_sec > 0.0:
            time.sleep(self.inference.start_countdown_sec)
        episode_dir = self.sessions.start_episode(
            metadata={
                "schema_version": "vt_dual_franka_bimanual_policy_episode_v1",
                "arm_order": list(ARM_ORDER),
                "initial_pose_by_arm": None
                if self._current_initial_poses is None
                else {arm: self._current_initial_poses[arm].metadata() for arm in ARM_ORDER},
                "initial_target_tcp_by_arm": self._current_initial_targets,
                "action_dim": 20,
                "action_provenance": "policy_command",
            }
        )
        self._current_episode_dir = episode_dir
        self._pending_outcome_episode_dir = None
        self._episode_error = None
        self._policy_terminated = False
        self._timeout_reached = False
        self._episode_stop_event = threading.Event()
        self._initial_pose_completed = False
        self.action_executor.reset()
        self.policy.reset()
        self._episode_thread = threading.Thread(
            target=self._episode_loop,
            name="bimanual-policy-episode",
            daemon=True,
        )
        self._episode_thread.start()

    def _episode_loop(self) -> None:
        assert self._current_episode_dir is not None
        recorder = JsonlStreamRecorder(self.sessions, "policy_steps")
        history = ObservationHistory(self.inference.obs_horizon)
        period = 1.0 / max(self.inference.control_hz, 1e-6)
        start = time.monotonic()
        step_index = 0
        try:
            observation, recorded = self.assembler.assemble()
            history.initialize_with_padding(observation)
            self.policy.start_episode(history.window())
            recorder.record_event(
                {"step_index": step_index, "phase": "initial_padding", "observation": recorded},
                event_time=time.time(),
            )
            step_index += 1
            while not self._episode_stop_event.is_set():
                if time.monotonic() - start >= self.inference.max_duration_sec:
                    self._timeout_reached = True
                    break
                raw_actions = self.policy.predict(history.window())
                action_chunk = normalize_action_chunk(raw_actions)
                actions_to_execute = action_chunk[: self.inference.exe_horizon]
                executed: list[Action] = []
                for action in actions_to_execute:
                    if self._episode_stop_event.is_set():
                        break
                    if action.target_tcp is not None or action.target_tcp_by_arm is None:
                        raise ValueError("Bimanual policy actions must provide target_tcp_by_arm only")
                    loop_start = time.monotonic()
                    normalized = self.action_executor.normalize_for_execution(action)
                    self.action_executor.execute_normalized(normalized)
                    precise_sleep(max(0.0, period - (time.monotonic() - loop_start)))
                    observation, recorded = self.assembler.assemble()
                    history.append(observation)
                    executed.append(normalized)
                    recorder.record_event(
                        {
                            "step_index": step_index,
                            "observation": recorded,
                            "commanded_action": action_to_json(normalized),
                        },
                        event_time=time.time(),
                    )
                    step_index += 1
                    if normalized.terminate:
                        self._policy_terminated = True
                        break
                self.policy.observe_executed_actions([action_to_json(item) for item in executed])
                if self._policy_terminated:
                    break
        except Exception as exc:
            self._episode_error = exc
            LOGGER.exception("Bimanual policy episode failed")

    def _poll_episode_status_locked(self) -> None:
        if self._current_episode_dir is None or self._episode_thread is None:
            return
        if not self._episode_thread.is_alive():
            self._finish_episode_locked(manual_stop=False)

    def _finish_episode_locked(self, *, manual_stop: bool) -> None:
        if self._current_episode_dir is None:
            return
        thread = self._episode_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(self.inference.max_duration_sec, 5.0))
        episode_dir = self._current_episode_dir
        if self._episode_error is not None:
            outcome, reason = "failed", "error"
        elif self._timeout_reached:
            outcome, reason = "saved", "timeout"
        elif self._policy_terminated:
            outcome, reason = "saved", "policy_terminate"
        elif manual_stop:
            outcome, reason = "saved", "manual_stop"
        else:
            outcome, reason = "saved", "completed"
        self.sessions.stop_episode(outcome=outcome, metadata_updates={"termination_reason": reason})
        self._current_episode_dir = None
        self._episode_thread = None
        if outcome == "saved":
            self._latest_saved_episode_dir = episode_dir
            self._pending_outcome_episode_dir = episode_dir

    def _mark_outcome_locked(self, outcome: str) -> None:
        self._poll_episode_status_locked()
        self._assert_no_active_episode()
        episode_dir = self._pending_outcome_episode_dir
        if episode_dir is None:
            raise OperatorActionError("No policy episode is waiting for an outcome.")
        self.sessions.update_episode_metadata(episode_dir, {"operator_outcome": outcome})
        self._pending_outcome_episode_dir = None

    def _required_initial_poses(self):
        if self.inference.initial_poses is None:
            raise OperatorActionError("Bimanual inference initial_poses are not configured.")
        return self.inference.initial_poses

    def _wait_for_joint_targets(self) -> None:
        poses = self._required_initial_poses()
        deadline = time.monotonic() + self.inference.home_joint_settle_timeout_sec
        errors: dict[ArmId, float] = {"left": float("inf"), "right": float("inf")}
        while time.monotonic() <= deadline:
            states = self.coordinator.get_state(max_age_sec=None).states
            for arm in ARM_ORDER:
                target = np.asarray(poses[arm].joint_positions_rad, dtype=np.float64)
                actual = np.asarray(states[arm].joint_positions, dtype=np.float64)
                errors[arm] = float(np.max(np.abs(target - actual)))
            if all(value <= self.inference.home_joint_tolerance_rad for value in errors.values()):
                return
            time.sleep(0.05)
        raise RuntimeError(f"Joint targets did not settle; max_error_rad={errors}")

    def _is_ready_for_episode_locked(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not self.coordinator.is_healthy(self.inference.modality.controller_state_max_age_sec):
            reasons.append("one or both controllers are unhealthy")
        if not self._initial_pose_completed:
            reasons.append("both arms have not reached policy initial poses with H")
        if self._pending_initial_gripper_close:
            reasons.append("both grippers must be closed with C")
        if self._pending_outcome_episode_dir is not None:
            reasons.append(f"mark {self._pending_outcome_episode_dir.name} success or fail")
        for stream_name in ("rgb_wrist_left", "rgb_wrist_right"):
            buffer = self.rgb_camera_buffers.get(stream_name)
            if buffer is None or buffer.get_latest_optional(self.inference.modality.rgb_camera_max_age_sec) is None:
                reasons.append(f"{stream_name} is not fresh")
        for arm in ARM_ORDER:
            buffer = self.tactile_buffers.get(arm)
            if buffer is None or buffer.get_latest_optional(self.inference.modality.gelsight_max_age_sec) is None:
                reasons.append(f"tactile_{arm} is not fresh")
        for name, worker in self.workers.items():
            if worker.required and worker.error is not None:
                reasons.append(f"{name} failed: {worker.error}")
        return not reasons, reasons

    def _build_status_locked(self) -> dict:
        ready, reasons = self._is_ready_for_episode_locked()
        snapshots = self.coordinator.snapshot()
        ages = [item.get("age_sec") for item in snapshots.values() if item.get("age_sec") is not None]
        preview_role = self.workspace.operator_ui.preview_camera_role
        preview = self._preview_sample(preview_role)
        next_index = self.sessions.get_next_episode_index()
        return {
            "mode": "bimanual_policy",
            "run_name": self.run_name,
            "task_name": self.inference.task_name,
            "model_name": "dp_bimanual",
            "ready": ready,
            "reasons": reasons,
            "active_episode": None if self._current_episode_dir is None else str(self._current_episode_dir),
            "active_episode_name": None if self._current_episode_dir is None else self._current_episode_dir.name,
            "latest_saved_episode": None if self._latest_saved_episode_dir is None else str(self._latest_saved_episode_dir),
            "latest_saved_episode_name": None
            if self._latest_saved_episode_dir is None
            else self._latest_saved_episode_dir.name,
            "pending_outcome_episode": None
            if self._pending_outcome_episode_dir is None
            else str(self._pending_outcome_episode_dir),
            "pending_outcome_episode_name": None
            if self._pending_outcome_episode_dir is None
            else self._pending_outcome_episode_dir.name,
            "next_episode_index": next_index,
            "next_episode_name": f"episode_{next_index:04d}",
            "controller_state": {
                "healthy": self.coordinator.is_healthy(self.inference.modality.controller_state_max_age_sec),
                "age_sec": max(ages) if ages else None,
                "arms": snapshots,
            },
            "controller_state_by_arm": snapshots,
            "rgb_cameras": {name: buffer.snapshot() for name, buffer in self.rgb_camera_buffers.items()},
            "gelsights": {arm: buffer.snapshot() for arm, buffer in self.tactile_buffers.items()},
            "pending_initial_gripper_close": self._pending_initial_gripper_close,
            "allowed_actions": {
                "reset_home_joints": self._current_episode_dir is None
                and self._pending_outcome_episode_dir is None
                and self.inference.initial_poses is not None
                and all(self.inference.initial_poses[arm].joint_positions_rad is not None for arm in ARM_ORDER),
                "reset": self._current_episode_dir is None and self._pending_outcome_episode_dir is None,
                "confirm_gripper_closed": self._pending_initial_gripper_close,
                "open_gripper": self._current_episode_dir is None,
                "start": ready and self._current_episode_dir is None,
                "stop": self._current_episode_dir is not None,
                "mark_success": self._pending_outcome_episode_dir is not None,
                "mark_fail": self._pending_outcome_episode_dir is not None,
                "discard": self._current_episode_dir is None and self._latest_saved_episode_dir is not None,
                "quit": self._current_episode_dir is None,
            },
            "preview": {
                "role": preview_role,
                "available": preview is not None,
                "streaming": preview is not None,
                "refresh_hz": self.workspace.operator_ui.preview_refresh_hz,
                "label": f"Live bimanual policy preview: {preview_role}",
            },
            "snapshots": {preview_role: {"available": preview is not None}},
            "workers": {
                name: {"alive": worker.is_alive(), "error": None if worker.error is None else str(worker.error)}
                for name, worker in self.workers.items()
            },
        }

    def _preview_sample(self, name: str):
        if self._current_episode_dir is not None or name != self.workspace.operator_ui.preview_camera_role:
            return None
        settings = self.inference.rgb_cameras.get(name)
        if settings is None:
            return None
        buffer = self.rgb_camera_buffers.get(settings.stream_name)
        if buffer is None:
            return None
        return buffer.get_latest_optional(self.workspace.operator_ui.snapshot_max_age_sec)

    def _assert_can_reset(self) -> None:
        self._poll_episode_status_locked()
        self._assert_no_active_episode()
        if self._pending_outcome_episode_dir is not None:
            raise OperatorActionError("Mark the previous episode success/fail before resetting.")

    def _assert_no_active_episode(self) -> None:
        if self._current_episode_dir is not None:
            raise OperatorActionError("A policy episode is active.")

    def _clear_initial_pose_state(self) -> None:
        self._initial_pose_completed = False
        self._pending_initial_gripper_close = False
        self._current_initial_poses = None
        self._current_initial_targets = None

    def _print_status_if_needed(self) -> None:
        now = time.time()
        if now - self._last_status_print_wall_time < 1.0 / max(self.inference.status_print_hz, 1e-6):
            return
        self._last_status_print_wall_time = now
        with self._operator_lock:
            status = self._build_status_locked()
        self.sessions.write_latest_status(status)
        print(
            f"[{'READY' if status['ready'] else 'WAIT'}] "
            f"recording={status['active_episode_name'] or 'off'} "
            f"left_age={status['controller_state_by_arm']['left']['age_sec']} "
            f"right_age={status['controller_state_by_arm']['right']['age_sec']}"
            + (f" reasons={'; '.join(status['reasons'])}" if status["reasons"] else ""),
            flush=True,
        )

    def _print_banner(self, run_dir: Path) -> None:
        print(f"Bimanual Policy Runner: {run_dir}", flush=True)
        print("20D action order: left pose9, right pose9, left/right gripper closedness.", flush=True)
        print("Hotkeys: J=paired joints H=paired pose R=start E=stop S=success F=fail Q=quit", flush=True)

    def _shutdown(self) -> None:
        with self._operator_lock:
            if self._current_episode_dir is not None:
                self._episode_stop_event.set()
                self._finish_episode_locked(manual_stop=True)
        stop_thread_workers(self.workers)
        if self.state_bridge is not None:
            self.state_bridge.stop()
        if self.operator_server is not None:
            self.operator_server.stop()
        self.coordinator.stop()
        self.policy.close()
        self.sessions.stop_run()
