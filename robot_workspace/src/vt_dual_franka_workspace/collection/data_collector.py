from __future__ import annotations

import logging
import socket
import threading
import time
from pathlib import Path

import numpy as np

from vt_dual_franka_shared.models import ArmId, ResetCommand
from vt_dual_franka_shared.transforms import BimanualCalibration

from ..config import TaskConfig, WorkspaceSettings
from ..operator import ManagedUvicornServer, OperatorActionError, OperatorLogBuffer, OperatorSnapshot, create_operator_app
from ..publishers.quest_udp import QuestUdpPublisher
from ..publishers.state_bridge import DualStateBridge
from ..recording import (
    EpisodeImageStreamRecorder,
    JsonlStreamRecorder,
    RunSessionManager,
    analyze_episode_quality,
    build_expected_episode_hz,
    episode_qc_manifest_summary,
    format_episode_qc_summary,
)
from ..runtime.dual_arm import ARM_ORDER, DualArmCoordinator
from ..runtime.keys import KeyReader
from ..runtime.live_buffer import LiveSampleBuffer
from ..runtime.motion import RandomizedInitialPose, eef_xyz_rpy_deg_to_tcp_pose, sample_randomized_initial_pose
from ..runtime.workers import ThreadWorker, start_thread_worker, stop_thread_workers
from ..sensors.rgb_camera import build_rgb_camera_recorder, resolve_rgb_camera_specs
from ..teleop.dual_quest_server import DualQuestTeleopService, create_dual_teleop_app

LOGGER = logging.getLogger(__name__)


class BimanualDataCollector:
    """Dual-only collection runtime.

    Both controller identities are checked before any worker starts. Cartesian,
    joint, gripper, and teleoperation commands are always issued as paired
    left/right operations through ``DualArmCoordinator``.
    """

    def __init__(
        self,
        workspace: WorkspaceSettings,
        task: TaskConfig,
        coordinator: DualArmCoordinator,
        calibration: BimanualCalibration,
        *,
        log_buffer: OperatorLogBuffer | None = None,
    ) -> None:
        self.workspace = workspace
        self.task = task
        self.coordinator = coordinator
        self.calibration = calibration
        self.log_buffer = log_buffer or OperatorLogBuffer(workspace.operator_ui.log_buffer_size)

        self.sessions = RunSessionManager(workspace.recording.collect_root)
        self.quest_publisher = QuestUdpPublisher(
            quest_ip=workspace.quest_feedback.quest_ip,
            robot_state_udp_port=workspace.quest_feedback.robot_state_udp_port,
            tactile_udp_port=workspace.quest_feedback.tactile_udp_port,
            image_udp_port=workspace.quest_feedback.image_udp_port,
            force_udp_port=workspace.quest_feedback.force_udp_port,
            calibration=calibration,
            force_scale_factor=workspace.quest_feedback.force_scale_factor,
        )
        self.teleop_service: DualQuestTeleopService | None = None
        self.teleop_server: ManagedUvicornServer | None = None
        self.operator_server: ManagedUvicornServer | None = None
        self.state_bridge: DualStateBridge | None = None
        self.workers: dict[str, ThreadWorker] = {}
        self.rgb_camera_buffers: dict[str, LiveSampleBuffer] = {}
        self.tactile_buffers: dict[ArmId, LiveSampleBuffer] = {}
        self.image_stream_recorders: dict[str, EpisodeImageStreamRecorder] = {}

        self._operator_lock = threading.RLock()
        self._quit_requested = threading.Event()
        self._current_episode_dir: Path | None = None
        self._latest_saved_episode_dir: Path | None = None
        self._home_joint_completed = False
        self._initial_pose_completed = False
        self._pending_initial_gripper_close = False
        self._current_initial_poses: dict[ArmId, RandomizedInitialPose] | None = None
        self._current_initial_targets: dict[ArmId, list[float]] | None = None
        self._last_status_print_wall_time = 0.0

    def run(self) -> None:
        run_dir = self.sessions.start_run(
            self.task.task_name,
            metadata={
                "schema_version": "vt_dual_franka_collection_run_v1",
                "workspace_hostname": socket.gethostname(),
                "controller_endpoints": {
                    arm_id: self.workspace.arms[arm_id].model_dump(mode="json") for arm_id in ARM_ORDER
                },
                "mode": "bimanual_collect",
                "task": self.task.model_dump(mode="json"),
                "workspace": self.workspace.model_dump(mode="json"),
            },
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
                    LOGGER.info("Bimanual data collection interrupted")
        finally:
            self._shutdown()

    def get_operator_status(self) -> dict:
        with self._operator_lock:
            self._poll_worker_failures_locked()
            status = self._build_status_locked()
        self.sessions.write_latest_status(status)
        return status

    def get_operator_snapshot(self, name: str) -> OperatorSnapshot | None:
        with self._operator_lock:
            sample = self._get_live_preview_sample_locked(name)
            if sample is None:
                return None
            return OperatorSnapshot(
                name=name,
                image=sample.data.copy(),
                captured_wall_time=sample.captured_wall_time,
                label=f"Live pre-episode {name} view: {sample.name}",
                image_format=self.workspace.recording.image_format,
            )

    def operator_reset_home_joints(self) -> None:
        with self._operator_lock:
            self._reset_home_joints_locked()

    def operator_reset_ready_pose(self) -> None:
        with self._operator_lock:
            self._move_to_initial_poses_locked()

    def operator_confirm_gripper_closed(self) -> None:
        with self._operator_lock:
            self._confirm_initial_grippers_closed_locked()

    def operator_open_gripper(self) -> None:
        with self._operator_lock:
            self._open_grippers_for_adjustment_locked()

    def operator_start_episode(self) -> None:
        with self._operator_lock:
            self._start_episode_locked()

    def operator_stop_episode(self) -> None:
        with self._operator_lock:
            self._stop_episode_locked()

    def operator_mark_episode_success(self) -> None:
        raise OperatorActionError("Collection episodes do not support success/fail outcome marking.")

    def operator_mark_episode_fail(self) -> None:
        raise OperatorActionError("Collection episodes do not support success/fail outcome marking.")

    def operator_discard_latest_episode(self) -> None:
        with self._operator_lock:
            self._discard_latest_episode_locked()

    def operator_quit(self) -> None:
        with self._operator_lock:
            if self._current_episode_dir is not None:
                raise OperatorActionError("Cannot quit while recording. Stop/save the active episode first.")
            self.sessions.record_operator_event("run_quit_requested")
            self._quit_requested.set()

    def _start_workers(self) -> None:
        self.coordinator.start()
        quest_recorder = (
            JsonlStreamRecorder(
                self.sessions,
                "quest_messages",
                record_hz=self.workspace.teleop.quest_message_record_hz,
            )
            if self.task.collection.record_raw_quest_messages
            else None
        )
        command_recorder = JsonlStreamRecorder(
            self.sessions,
            "teleop_commands",
            record_hz=self.workspace.teleop.command_record_hz,
        )
        self.teleop_service = DualQuestTeleopService(
            self.workspace.teleop,
            self.coordinator,
            self.calibration,
            quest_message_recorder=quest_recorder,
            command_recorder=command_recorder,
        )
        self.teleop_service.set_teleop_enabled(False)
        self.teleop_server = ManagedUvicornServer(
            create_dual_teleop_app(self.teleop_service),
            self.workspace.teleop.host,
            self.workspace.teleop.port,
        )
        self.teleop_server.start()

        if self.workspace.operator_ui.enabled:
            self.operator_server = ManagedUvicornServer(
                create_operator_app(self, self.log_buffer, title="VT Dual Franka Bimanual Collector"),
                self.workspace.operator_ui.host,
                self.workspace.operator_ui.port,
            )
            self.operator_server.start()

        state_recorder = JsonlStreamRecorder(
            self.sessions,
            "controller_state_by_arm",
            record_hz=self.workspace.quest_feedback.record_hz,
        )
        self.state_bridge = DualStateBridge(
            self.coordinator,
            self.quest_publisher,
            self.workspace.quest_feedback,
            recorder=state_recorder,
        )
        self.state_bridge.start()
        self._start_rgb_workers()
        self._start_tactile_workers()

    def _start_rgb_workers(self) -> None:
        rgb_specs = {spec.role: spec for spec in resolve_rgb_camera_specs(self.task.rgb_cameras)}
        for role in self.task.modality.rgb_cameras:
            if role not in rgb_specs:
                raise RuntimeError(f"Task modality requested RGB camera role not configured: {role}")
            spec = rgb_specs[role]
            live_buffer = LiveSampleBuffer(spec.stream_name)
            self.rgb_camera_buffers[role] = live_buffer
            image_recorder = EpisodeImageStreamRecorder(
                self.sessions,
                spec.stream_name,
                record_hz=spec.settings.record_hz,
                image_format=self.workspace.recording.image_format,
                jpeg_quality=90,
            )
            self.image_stream_recorders[spec.stream_name] = image_recorder
            service = build_rgb_camera_recorder(
                spec.__class__(
                    role=spec.role,
                    stream_name=spec.stream_name,
                    settings=spec.settings.model_copy(update={"save_frames": False}),
                ),
                episode_image_recorder=image_recorder,
                live_buffer=live_buffer,
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
        if not self.task.modality.needs_gelsight():
            return
        from ..sensors.gelsight.publisher import GelsightPublisher

        for arm_id in ARM_ORDER:
            settings = self.task.gelsights[arm_id]
            if not settings.enabled:
                raise RuntimeError(f"Task requires {arm_id} GelSight, but task.gelsights.{arm_id}.enabled is false")
            stream_name = f"tactile_{arm_id}"
            live_buffer = LiveSampleBuffer(stream_name)
            self.tactile_buffers[arm_id] = live_buffer
            image_recorder = EpisodeImageStreamRecorder(
                self.sessions,
                stream_name,
                record_hz=settings.record_hz,
                image_format=self.workspace.recording.image_format,
                jpeg_quality=90,
                max_frames=settings.buffer_max_frames if settings.buffered_recording else None,
            )
            self.image_stream_recorders[stream_name] = image_recorder
            service = GelsightPublisher(
                settings.model_copy(update={"save_frames": False}),
                self.quest_publisher,
                episode_image_recorder=image_recorder,
                frame_buffer=live_buffer,
                image_format=self.workspace.recording.image_format,
            )
            start_thread_worker(
                self.workers,
                f"gelsight:{arm_id}",
                lambda stop_event, service=service: service.run(stop_event=stop_event),
                required=True,
            )

    def _run_event_loop(self, key_reader: KeyReader) -> None:
        while not self._quit_requested.is_set():
            with self._operator_lock:
                self._poll_worker_failures_locked()
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
                "q": self.operator_quit,
            }.get(key.lower())
            if action is not None:
                self._run_terminal_action(action)
            elif key.lower() == "d":
                self._handle_terminal_discard(key_reader)

    @staticmethod
    def _run_terminal_action(action) -> None:
        try:
            action()
        except OperatorActionError as exc:
            LOGGER.warning("%s", exc)

    def _reset_home_joints_locked(self) -> None:
        self._assert_idle("reset home joints")
        missing = [arm for arm in ARM_ORDER if self.task.initial_poses[arm].joint_positions_rad is None]
        if missing:
            raise OperatorActionError(f"Task joint initial pose is not configured for: {missing}")
        self._disable_teleop()
        commands = {
            arm_id: ResetCommand(
                profile="bimanual_home_joints",
                joint_positions=list(self.task.initial_poses[arm_id].joint_positions_rad or []),
                joint_duration_sec=self.task.home_joint_duration_sec,
                source="bimanual_collector_home_joints",
            )
            for arm_id in ARM_ORDER
        }
        try:
            results = self.coordinator.reset_pair(commands)
            self._wait_for_joint_targets()
        except Exception as exc:
            raise OperatorActionError(f"Failed to reset both arms to home joints: {exc}") from exc
        self._home_joint_completed = True
        self._clear_initial_pose_state()
        self.sessions.record_operator_event(
            "bimanual_home_joint_reset_completed",
            {
                "joint_positions_by_arm": {
                    arm: self.task.initial_poses[arm].joint_positions_rad for arm in ARM_ORDER
                },
                "results": results,
            },
        )
        LOGGER.info("Both arms reached task joint poses. Press H for paired Cartesian initial poses.")

    def _move_to_initial_poses_locked(self) -> None:
        self._assert_idle("move to initial poses")
        self._disable_teleop()
        sampled = {
            arm_id: sample_randomized_initial_pose(
                self.task.initial_poses[arm_id].eef_pose_xyz_rpy_deg,
                self.task.initial_poses[arm_id].random_xyz_range_m,
            )
            for arm_id in ARM_ORDER
        }
        commands = {
            arm_id: ResetCommand(
                profile="bimanual_initial_pose",
                eef_pose_xyz_rpy_deg=list(sampled[arm_id].pose_xyz_rpy_deg),
                eef_duration_sec=self.task.initial_move_duration_sec,
                source="bimanual_collector_initial_pose",
            )
            for arm_id in ARM_ORDER
        }
        try:
            results = self.coordinator.reset_pair(commands)
        except Exception as exc:
            raise OperatorActionError(f"Failed to move both arms to initial poses: {exc}") from exc
        targets = {
            arm_id: eef_xyz_rpy_deg_to_tcp_pose(sampled[arm_id].pose_xyz_rpy_deg) for arm_id in ARM_ORDER
        }
        self._current_initial_poses = sampled
        self._current_initial_targets = targets
        self._pending_initial_gripper_close = self.task.gripper_forever_closed
        self._initial_pose_completed = not self._pending_initial_gripper_close
        self.sessions.record_operator_event(
            "bimanual_initial_pose_completed",
            {
                "target_tcp_by_arm": targets,
                "sampled_pose_by_arm": {arm: sampled[arm].metadata() for arm in ARM_ORDER},
                "results": results,
            },
        )
        if self._pending_initial_gripper_close:
            LOGGER.info("Both initial poses reached. Press C to close both grippers.")
        else:
            LOGGER.info("Both initial poses reached. Ready.")

    def _confirm_initial_grippers_closed_locked(self) -> None:
        if not self.task.gripper_forever_closed:
            raise OperatorActionError("gripper_forever_closed is disabled for this task.")
        self._assert_idle("close grippers")
        if self._current_initial_poses is None:
            raise OperatorActionError("Move both arms to initial poses with H before closing grippers.")
        try:
            self.coordinator.grasp_grippers(
                velocity=self.workspace.teleop.gripper_velocity,
                force_limit=self.workspace.teleop.grasp_force,
                source="bimanual_collector_initial_gripper_close",
                blocking=True,
            )
        except Exception as exc:
            raise OperatorActionError(f"Failed to close both grippers: {exc}") from exc
        self._pending_initial_gripper_close = False
        self._initial_pose_completed = True
        self.sessions.record_operator_event("bimanual_initial_grippers_closed")

    def _open_grippers_for_adjustment_locked(self) -> None:
        if not self.task.gripper_forever_closed:
            raise OperatorActionError("gripper_forever_closed is disabled for this task.")
        self._assert_idle("open grippers")
        if self._current_initial_poses is None:
            raise OperatorActionError("Move both arms to initial poses with H before opening grippers.")
        try:
            self.coordinator.move_grippers(
                {arm: self.workspace.teleop.max_gripper_width for arm in ARM_ORDER},
                velocity=self.workspace.teleop.gripper_velocity,
                force_limit=self.workspace.teleop.grasp_force,
                source="bimanual_collector_gripper_adjustment_open",
                blocking=True,
            )
        except Exception as exc:
            raise OperatorActionError(f"Failed to open both grippers: {exc}") from exc
        self._pending_initial_gripper_close = True
        self._initial_pose_completed = False

    def _wait_for_joint_targets(self) -> None:
        deadline = time.monotonic() + self.task.home_joint_settle_timeout_sec
        last_error: dict[ArmId, float] = {"left": float("inf"), "right": float("inf")}
        while time.monotonic() <= deadline:
            states = self.coordinator.get_state(max_age_sec=None).states
            for arm_id in ARM_ORDER:
                target = np.asarray(self.task.initial_poses[arm_id].joint_positions_rad, dtype=np.float64)
                actual = np.asarray(states[arm_id].joint_positions, dtype=np.float64)
                last_error[arm_id] = float(np.max(np.abs(actual - target)))
            if all(error <= self.task.home_joint_tolerance_rad for error in last_error.values()):
                return
            time.sleep(0.05)
        raise RuntimeError(f"Joint targets did not settle; max_error_rad={last_error}")

    def _start_episode_locked(self) -> None:
        ready, reasons = self._is_ready_for_episode_locked()
        if not ready:
            raise OperatorActionError(f"Cannot start episode: {'; '.join(reasons)}")
        if self._current_episode_dir is not None:
            raise OperatorActionError("An episode is already active.")
        if self.task.collection.start_countdown_sec > 0.0:
            time.sleep(self.task.collection.start_countdown_sec)
        episode_dir = self.sessions.start_episode(
            metadata={
                "schema_version": "vt_dual_franka_bimanual_episode_v1",
                "task_name": self.task.task_name,
                "arm_order": list(ARM_ORDER),
                "modality": self.task.modality.model_dump(mode="json"),
                "controller_status_by_arm": self.coordinator.snapshot(),
                "initial_pose_by_arm": None
                if self._current_initial_poses is None
                else {arm: self._current_initial_poses[arm].metadata() for arm in ARM_ORDER},
                "initial_target_tcp_by_arm": self._current_initial_targets,
                "action_provenance": "commanded_action",
            },
        )
        self._current_episode_dir = episode_dir
        self._clear_initial_pose_state()
        if self.teleop_service is not None:
            self.teleop_service.set_teleop_enabled(True)
        self.sessions.record_operator_event("episode_started", {"episode_dir": str(episode_dir)})
        LOGGER.info("Bimanual episode started: %s", episode_dir)

    def _stop_episode_locked(self) -> None:
        if self._current_episode_dir is None:
            raise OperatorActionError("No active episode to stop.")
        episode_dir = self._current_episode_dir
        worker_error = self._required_worker_error()
        outcome = "failed" if worker_error is not None else "saved"
        self.sessions.close_active_episode()
        self._current_episode_dir = None
        self._home_joint_completed = False
        metadata_updates: dict = {}
        try:
            summaries = {
                name: summary
                for name, recorder in self.image_stream_recorders.items()
                if (summary := recorder.flush_episode(episode_dir)) is not None
            }
            if summaries:
                metadata_updates["image_streams"] = summaries
        except Exception as exc:
            outcome = "failed"
            metadata_updates["image_stream_error"] = str(exc)
        if worker_error is not None:
            metadata_updates["failure_reason"] = str(worker_error)
        self.sessions.finalize_episode(episode_dir, outcome=outcome, metadata_updates=metadata_updates)
        qc_report = None
        if outcome == "saved":
            try:
                qc_report = analyze_episode_quality(
                    episode_dir,
                    expected_hz=build_expected_episode_hz(self.workspace, self.task),
                    write=True,
                )
                self.sessions.update_episode_metadata(
                    episode_dir,
                    {"episode_qc": episode_qc_manifest_summary(qc_report)},
                )
            except Exception as exc:
                LOGGER.exception("Episode QC failed for %s", episode_dir)
                self.sessions.update_episode_metadata(episode_dir, {"episode_qc_error": str(exc)})
        self._latest_saved_episode_dir = episode_dir if outcome == "saved" else self._latest_saved_episode_dir
        if self.teleop_service is not None:
            self.teleop_service.set_teleop_enabled(outcome == "saved")
        LOGGER.info("Bimanual episode %s: %s", outcome, episode_dir)
        if qc_report is not None:
            print(f"[QC] {episode_dir.name}: {format_episode_qc_summary(qc_report)}", flush=True)

    def _is_ready_for_episode_locked(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        max_age = self.task.modality.controller_state_max_age_sec
        if self.teleop_server is None or not self.teleop_server.is_alive():
            reasons.append("dual teleop server is not running")
        if not self.coordinator.is_healthy(max_age_sec=max_age):
            reasons.append("one or both controller states are not healthy")
        if not self._initial_pose_completed:
            reasons.append("both arms have not reached task initial poses with H")
        if self._pending_initial_gripper_close:
            reasons.append("paired initial gripper close is pending; press C")
        if self.task.collection.require_quest_connection and (
            self.teleop_service is None
            or not self.teleop_service.has_recent_message(self.task.collection.quest_message_timeout_sec)
        ):
            reasons.append("Quest connection is not active")
        for role in self.task.modality.rgb_cameras:
            buffer = self.rgb_camera_buffers.get(role)
            if buffer is None or buffer.get_latest_optional(self.task.modality.rgb_camera_max_age_sec) is None:
                reasons.append(f"RGB camera {role} is not fresh")
        if self.task.modality.needs_gelsight():
            for arm_id in ARM_ORDER:
                buffer = self.tactile_buffers.get(arm_id)
                if buffer is None or buffer.get_latest_optional(self.task.modality.gelsight_max_age_sec) is None:
                    reasons.append(f"{arm_id} GelSight is not fresh")
        for name, worker in self.workers.items():
            if worker.required and worker.error is not None:
                reasons.append(f"{name} failed: {worker.error}")
        return not reasons, reasons

    def _build_status_locked(self) -> dict:
        ready, reasons = self._is_ready_for_episode_locked()
        preview_role = self.workspace.operator_ui.preview_camera_role
        preview_sample = self._get_live_preview_sample_locked(preview_role)
        snapshots = self.coordinator.snapshot()
        ages = [item.get("age_sec") for item in snapshots.values() if item.get("age_sec") is not None]
        next_index = self.sessions.get_next_episode_index()
        return {
            "mode": "bimanual_collect",
            "run_name": self.task.task_name,
            "ready": ready,
            "reasons": reasons,
            "active_episode": None if self._current_episode_dir is None else str(self._current_episode_dir),
            "active_episode_name": None if self._current_episode_dir is None else self._current_episode_dir.name,
            "latest_saved_episode": None if self._latest_saved_episode_dir is None else str(self._latest_saved_episode_dir),
            "latest_saved_episode_name": None
            if self._latest_saved_episode_dir is None
            else self._latest_saved_episode_dir.name,
            "next_episode_index": next_index,
            "next_episode_name": f"episode_{next_index:04d}",
            "quest_connected": self.teleop_service is not None
            and self.teleop_service.has_recent_message(self.task.collection.quest_message_timeout_sec),
            "teleop_enabled": self.teleop_service is not None and self.teleop_service.is_teleop_enabled(),
            "pending_initial_gripper_close": self._pending_initial_gripper_close,
            "home_joint_completed": self._home_joint_completed,
            "controller_state": {
                "healthy": self.coordinator.is_healthy(self.task.modality.controller_state_max_age_sec),
                "age_sec": max(ages) if ages else None,
                "arms": snapshots,
            },
            "controller_state_by_arm": snapshots,
            "rgb_cameras": {role: buffer.snapshot() for role, buffer in self.rgb_camera_buffers.items()},
            "gelsights": {arm: buffer.snapshot() for arm, buffer in self.tactile_buffers.items()},
            "image_stream_recorders": {
                name: recorder.snapshot() for name, recorder in self.image_stream_recorders.items()
            },
            "workers": {
                name: {"alive": worker.is_alive(), "error": None if worker.error is None else str(worker.error)}
                for name, worker in self.workers.items()
            },
            "allowed_actions": {
                "reset_home_joints": self._current_episode_dir is None
                and all(self.task.initial_poses[arm].joint_positions_rad is not None for arm in ARM_ORDER),
                "reset": self._current_episode_dir is None,
                "confirm_gripper_closed": self._current_episode_dir is None
                and self._pending_initial_gripper_close,
                "open_gripper": self._current_episode_dir is None
                and self.task.gripper_forever_closed
                and self._current_initial_poses is not None,
                "start": ready and self._current_episode_dir is None,
                "stop": self._current_episode_dir is not None,
                "mark_success": False,
                "mark_fail": False,
                "discard": self._current_episode_dir is None
                and (self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()) is not None,
                "quit": self._current_episode_dir is None,
            },
            "preview": {
                "role": preview_role,
                "available": preview_sample is not None,
                "streaming": preview_sample is not None,
                "refresh_hz": self.workspace.operator_ui.preview_refresh_hz,
                "label": f"Live pre-episode {preview_role} view",
            },
            "snapshots": {preview_role: {"available": preview_sample is not None}},
        }

    def _get_live_preview_sample_locked(self, name: str):
        if name != self.workspace.operator_ui.preview_camera_role or self._current_episode_dir is not None:
            return None
        buffer = self.rgb_camera_buffers.get(name)
        if buffer is None:
            return None
        return buffer.get_latest_optional(max_age_sec=self.workspace.operator_ui.snapshot_max_age_sec)

    def _poll_worker_failures_locked(self) -> None:
        if self._current_episode_dir is not None and self._required_worker_error() is not None:
            self._stop_episode_locked()

    def _required_worker_error(self) -> Exception | None:
        return next(
            (worker.error for worker in self.workers.values() if worker.required and worker.error is not None),
            None,
        )

    def _discard_latest_episode_locked(self) -> None:
        self._assert_idle("discard an episode")
        episode_dir = self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()
        if episode_dir is None:
            raise OperatorActionError("No saved episode to discard.")
        self.sessions.discard_episode(episode_dir)
        self._latest_saved_episode_dir = self.sessions.get_latest_saved_episode_dir()

    def _handle_terminal_discard(self, key_reader: KeyReader) -> None:
        episode_dir = self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()
        if self._current_episode_dir is not None or episode_dir is None:
            LOGGER.warning("No inactive saved episode is available to discard")
            return
        print(f"Press Enter to confirm discarding {episode_dir.name}, or any other key to cancel.", flush=True)
        if key_reader.read_key(30.0) in ("\n", "\r"):
            self._run_terminal_action(self.operator_discard_latest_episode)

    def _assert_idle(self, operation: str) -> None:
        if self._current_episode_dir is not None:
            raise OperatorActionError(f"Cannot {operation} while recording.")

    def _disable_teleop(self) -> None:
        if self.teleop_service is not None:
            self.teleop_service.set_teleop_enabled(False)

    def _clear_initial_pose_state(self) -> None:
        self._initial_pose_completed = False
        self._pending_initial_gripper_close = False
        self._current_initial_poses = None
        self._current_initial_targets = None

    def _print_status_if_needed(self) -> None:
        now = time.time()
        if now - self._last_status_print_wall_time < 1.0 / max(self.task.collection.status_print_hz, 1e-6):
            return
        self._last_status_print_wall_time = now
        with self._operator_lock:
            status = self._build_status_locked()
        self.sessions.write_latest_status(status)
        print(
            f"[{'READY' if status['ready'] else 'WAIT'}] "
            f"next={status['next_episode_name']} "
            f"recording={status['active_episode_name'] or 'off'} "
            f"left_age={status['controller_state_by_arm']['left']['age_sec']} "
            f"right_age={status['controller_state_by_arm']['right']['age_sec']} "
            f"quest={'ok' if status['quest_connected'] else 'missing'}"
            + (f" reasons={'; '.join(status['reasons'])}" if status["reasons"] else ""),
            flush=True,
        )

    def _print_banner(self, run_dir: Path) -> None:
        print(f"Bimanual Data Collector run started: {run_dir}", flush=True)
        print("Left Quest hand -> left Franka; right Quest hand -> right Franka.", flush=True)
        print("Hotkeys: J=paired joints  H=paired initial poses  R=start  E=save  D=discard  Q=quit", flush=True)
        if self.workspace.operator_ui.enabled:
            print(
                f"Operator UI: http://{self.workspace.operator_ui.host}:{self.workspace.operator_ui.port}/operator",
                flush=True,
            )

    def _shutdown(self) -> None:
        self._disable_teleop()
        self.sessions.record_operator_event("run_stopped")
        stop_thread_workers(self.workers)
        if self.state_bridge is not None:
            self.state_bridge.stop()
        if self.operator_server is not None:
            self.operator_server.stop()
        if self.teleop_server is not None:
            self.teleop_server.stop()
        self.coordinator.stop()
        self.sessions.stop_run()
