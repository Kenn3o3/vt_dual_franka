from __future__ import annotations

import logging
import socket
import threading
import time
from pathlib import Path

import uvicorn

from vt_dual_franka_shared.transforms import SingleArmCalibration

from ..config import TaskConfig, WorkspaceSettings
from ..controller.client import ControllerClient
from ..operator import ManagedUvicornServer, OperatorActionError, OperatorLogBuffer, OperatorSnapshot, create_operator_app
from ..publishers.quest_udp import QuestUdpPublisher
from ..publishers.state_bridge import StateBridge
from ..recording import (
    EpisodeImageStreamRecorder,
    JsonlStreamRecorder,
    RunSessionManager,
    analyze_episode_quality,
    build_expected_episode_hz,
    episode_qc_manifest_summary,
    format_episode_qc_summary,
)
from ..runtime.keys import KeyReader
from ..runtime.live_buffer import LiveSampleBuffer
from ..runtime.motion import RandomizedInitialPose, move_to_eef_pose, move_to_home_joints, sample_randomized_initial_pose
from ..runtime.workers import ThreadWorker, start_thread_worker, stop_thread_workers
from ..sensors.rgb_camera import build_rgb_camera_recorder, resolve_rgb_camera_specs
from ..teleop.quest_server import QuestTeleopService, create_teleop_app
from .controller_state import ControllerStateMonitor

LOGGER = logging.getLogger(__name__)


class DataCollector:
    def __init__(
        self,
        workspace: WorkspaceSettings,
        task: TaskConfig,
        controller: ControllerClient,
        calibration: SingleArmCalibration,
        *,
        log_buffer: OperatorLogBuffer | None = None,
    ) -> None:
        self.workspace = workspace
        self.task = task
        self.controller = controller
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
        self.state_monitor = ControllerStateMonitor(
            controller,
            poll_hz=task.collection.controller_state_poll_hz,
        )
        self.teleop_service: QuestTeleopService | None = None
        self.teleop_server: ManagedUvicornServer | None = None
        self.operator_server: ManagedUvicornServer | None = None
        self.state_bridge: StateBridge | None = None
        self.workers: dict[str, ThreadWorker] = {}
        self.rgb_camera_buffers: dict[str, LiveSampleBuffer] = {}
        self.gelsight_frame_buffer: LiveSampleBuffer | None = None
        self.image_stream_recorders: dict[str, EpisodeImageStreamRecorder] = {}

        self._operator_lock = threading.RLock()
        self._quit_requested = threading.Event()
        self._current_episode_dir: Path | None = None
        self._latest_saved_episode_dir: Path | None = None
        self._home_joint_completed = False
        self._initial_pose_completed = False
        self._pending_initial_gripper_close = False
        self._current_initial_pose: RandomizedInitialPose | None = None
        self._current_initial_target_tcp: list[float] | None = None
        self._last_status_print_wall_time = 0.0

    def run(self) -> None:
        run_dir = self.sessions.start_run(
            self.task.task_name,
            metadata={
                "workspace_hostname": socket.gethostname(),
                "controller_host": self.workspace.controller.host,
                "mode": "collect",
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
                    LOGGER.info("Data collection interrupted")
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
            self._open_gripper_for_adjustment_locked()

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
        self.state_monitor.start()
        state_provider = lambda: self.state_monitor.get_state(
            max_age_sec=self.task.modality.controller_state_max_age_sec
        )

        quest_recorder = None
        if self.task.collection.record_raw_quest_messages:
            quest_recorder = JsonlStreamRecorder(
                self.sessions,
                "quest_messages",
                record_hz=self.workspace.teleop.quest_message_record_hz,
            )
        command_recorder = JsonlStreamRecorder(
            self.sessions,
            "teleop_commands",
            record_hz=self.workspace.teleop.command_record_hz,
        )
        self.teleop_service = QuestTeleopService(
            self.workspace.teleop,
            self.controller,
            self.calibration,
            quest_message_recorder=quest_recorder,
            command_recorder=command_recorder,
            state_provider=state_provider,
            gripper_forever_closed=self.task.gripper_forever_closed,
        )
        self.teleop_service.set_teleop_enabled(False)
        self.teleop_server = ManagedUvicornServer(
            create_teleop_app(self.teleop_service),
            self.workspace.teleop.host,
            self.workspace.teleop.port,
        )
        self.teleop_server.start()

        if self.workspace.operator_ui.enabled:
            self.operator_server = ManagedUvicornServer(
                create_operator_app(self, self.log_buffer, title="VT Dual Franka Data Collector"),
                self.workspace.operator_ui.host,
                self.workspace.operator_ui.port,
            )
            self.operator_server.start()

        state_recorder = JsonlStreamRecorder(
            self.sessions,
            "controller_state",
            record_hz=self.workspace.quest_feedback.record_hz,
        )
        self.state_bridge = StateBridge(
            self.controller,
            self.quest_publisher,
            self.workspace.quest_feedback,
            recorder=state_recorder,
            state_provider=state_provider,
        )
        self.state_bridge.start()

        rgb_specs = {spec.role: spec for spec in resolve_rgb_camera_specs(self.task.rgb_cameras)}
        for role in self.task.modality.rgb_cameras:
            if role not in rgb_specs:
                raise RuntimeError(f"Task modality requested RGB camera role not configured: {role}")
            spec = rgb_specs[role]
            live_buffer = LiveSampleBuffer(spec.stream_name)
            self.rgb_camera_buffers[spec.role] = live_buffer
            image_recorder = EpisodeImageStreamRecorder(
                self.sessions,
                spec.stream_name,
                record_hz=spec.settings.record_hz,
                image_format=self.workspace.recording.image_format,
                jpeg_quality=90,
            )
            self.image_stream_recorders[spec.stream_name] = image_recorder
            rgb_settings = spec.settings.model_copy(update={"save_frames": False})
            spec = spec.__class__(role=spec.role, stream_name=spec.stream_name, settings=rgb_settings)
            service = build_rgb_camera_recorder(
                spec,
                recorder=None,
                canonical_recorder=None,
                episode_image_recorder=image_recorder,
                live_buffer=live_buffer,
                quest_publisher=self.quest_publisher,
                image_format=self.workspace.recording.image_format,
            )
            start_thread_worker(
                self.workers,
                f"rgb_camera:{spec.role}",
                lambda stop_event, service=service: service.run(stop_event=stop_event),
                required=True,
            )

        if self.task.modality.needs_gelsight():
            from ..sensors.gelsight.publisher import GelsightPublisher

            if not self.task.gelsight.enabled:
                raise RuntimeError("Task modality requested GelSight, but task.gelsight.enabled is false")
            image_recorder = EpisodeImageStreamRecorder(
                self.sessions,
                "tactile_left",
                record_hz=self.task.gelsight.record_hz,
                image_format=self.workspace.recording.image_format,
                jpeg_quality=90,
                max_frames=self.task.gelsight.buffer_max_frames if self.task.gelsight.buffered_recording else None,
            )
            self.image_stream_recorders["tactile_left"] = image_recorder
            self.gelsight_frame_buffer = LiveSampleBuffer("gelsight_frame")
            gelsight_settings = self.task.gelsight.model_copy(update={"save_frames": False})
            service = GelsightPublisher(
                gelsight_settings,
                self.quest_publisher,
                frame_recorder=None,
                canonical_recorder=None,
                episode_image_recorder=image_recorder,
                frame_buffer=self.gelsight_frame_buffer,
                image_format=self.workspace.recording.image_format,
            )
            start_thread_worker(
                self.workers,
                "gelsight",
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
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot reset home joints while recording. Stop/save the active episode first.")
        if self.task.home_joint_positions_rad is None:
            raise OperatorActionError("Task home_joint_positions_rad is not configured.")
        if self.teleop_service is not None:
            self.teleop_service.set_teleop_enabled(False)
        LOGGER.info("Resetting robot to task home joint positions")
        try:
            result = move_to_home_joints(
                controller=self.controller,
                state_provider=self.state_monitor,
                joint_positions=self.task.home_joint_positions_rad,
                duration_sec=self.task.home_joint_duration_sec,
                source="data_collector_home_joints",
                tolerance_rad=self.task.home_joint_tolerance_rad,
                settle_timeout_sec=self.task.home_joint_settle_timeout_sec,
                state_max_age_sec=self.task.modality.controller_state_max_age_sec,
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
                "joint_positions": list(self.task.home_joint_positions_rad),
                "duration_sec": self.task.home_joint_duration_sec,
                "result": result,
            },
        )
        LOGGER.info("Home joint reset complete. Press H to move to the task initial EEF pose.")

    def _handle_terminal_discard(self, key_reader: KeyReader) -> None:
        with self._operator_lock:
            episode_dir = self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()
            if self._current_episode_dir is not None:
                LOGGER.warning("Cannot discard while recording. Press E first.")
                return
            if episode_dir is None:
                LOGGER.warning("No saved episode to discard")
                return
        print(f"Press Enter to confirm discarding {episode_dir.name}, or any other key to cancel.", flush=True)
        key = key_reader.read_key(30.0)
        if key not in ("\n", "\r"):
            LOGGER.info("Discard cancelled")
            return
        self._run_terminal_action(self.operator_discard_latest_episode)

    def _move_to_initial_pose_locked(self) -> None:
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot move to initial pose while recording. Stop/save the active episode first.")
        if self.teleop_service is not None:
            self.teleop_service.set_teleop_enabled(False)
        LOGGER.info("Moving robot to task initial EEF pose")
        initial_pose = sample_randomized_initial_pose(self.task.initial_eef_pose_xyz_rpy_deg, self.task.rand_init_pose)
        try:
            target_tcp = move_to_eef_pose(
                controller=self.controller,
                state_provider=self.state_monitor,
                pose_xyz_rpy_deg=initial_pose.pose_xyz_rpy_deg,
                duration_sec=self.task.initial_move_duration_sec,
                source="data_collector_initial_pose",
                position_tolerance_m=self.task.collection.initial_pose_tolerance_m,
                rotation_tolerance_deg=self.task.collection.initial_pose_tolerance_deg,
                settle_timeout_sec=self.task.collection.initial_pose_settle_timeout_sec,
                settle_dwell_sec=self.task.collection.initial_pose_settle_dwell_sec,
                state_max_age_sec=self.task.modality.controller_state_max_age_sec,
            )
        except Exception as exc:
            raise OperatorActionError(f"Failed to move robot to initial pose: {exc}") from exc
        self._current_initial_pose = initial_pose
        self._current_initial_target_tcp = target_tcp
        self.sessions.record_operator_event(
            "initial_pose_requested",
            {
                "target_tcp": target_tcp,
                "gripper_forever_closed": self.task.gripper_forever_closed,
                **initial_pose.metadata(),
            },
        )
        if self.task.gripper_forever_closed:
            self._initial_pose_completed = False
            self._pending_initial_gripper_close = True
            self.sessions.record_operator_event("initial_gripper_close_pending", {"target_tcp": target_tcp})
            LOGGER.info("Initial pose reached. Press C to close the gripper before starting the episode.")
            return
        self._pending_initial_gripper_close = False
        self._initial_pose_completed = True
        LOGGER.info("Initial pose reached. Ready.")

    def _confirm_initial_gripper_closed_locked(self) -> None:
        if not self.task.gripper_forever_closed:
            raise OperatorActionError("gripper_forever_closed is disabled for this task.")
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot close initial gripper while recording. Stop/save the active episode first.")
        if self._current_initial_pose is None:
            raise OperatorActionError("Move to the task initial pose with H before closing the gripper.")
        LOGGER.info("Closing gripper for forever-closed episode")
        try:
            self.controller.grasp_gripper(
                velocity=self.workspace.teleop.gripper_velocity,
                force_limit=self.workspace.teleop.grasp_force,
                source="data_collector_initial_gripper_close",
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

    def _open_gripper_for_adjustment_locked(self) -> None:
        if not self.task.gripper_forever_closed:
            raise OperatorActionError("gripper_forever_closed is disabled for this task.")
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot open gripper while recording. Stop/save the active episode first.")
        if self._current_initial_pose is None:
            raise OperatorActionError("Move to the task initial pose with H before opening the gripper.")
        LOGGER.info("Opening gripper for object adjustment. Press C to close/confirm before starting the episode.")
        try:
            self.controller.move_gripper(
                width=self.workspace.teleop.max_gripper_width,
                velocity=self.workspace.teleop.gripper_velocity,
                force_limit=self.workspace.teleop.grasp_force,
                source="data_collector_gripper_adjustment_open",
                blocking=True,
            )
        except Exception as exc:
            raise OperatorActionError(f"Failed to open gripper for adjustment: {exc}") from exc
        self._pending_initial_gripper_close = True
        self._initial_pose_completed = False
        self.sessions.record_operator_event(
            "gripper_opened_for_adjustment",
            {
                "target_tcp": self._current_initial_target_tcp,
                "open_width": self.workspace.teleop.max_gripper_width,
                **self._current_initial_pose.metadata(),
            },
        )

    def _start_episode_locked(self) -> None:
        ready, reasons = self._is_ready_for_episode_locked()
        if not ready:
            raise OperatorActionError(f"Cannot start episode: {'; '.join(reasons)}")
        if self._current_episode_dir is not None:
            raise OperatorActionError("An episode is already active.")
        countdown = self.task.collection.start_countdown_sec
        self.sessions.record_operator_event("episode_start_requested", {"countdown_sec": countdown})
        if countdown > 0.0:
            LOGGER.info("Starting episode in %.1f seconds", countdown)
            time.sleep(countdown)
        episode_index = self.sessions.get_next_episode_index()
        episode_dir = self.sessions.start_episode(
            name=f"episode_{episode_index:04d}",
            metadata={
                "task_name": self.task.task_name,
                "modality": self.task.modality.model_dump(mode="json"),
                "controller_status": self.state_monitor.snapshot(),
                "initial_pose": None if self._current_initial_pose is None else self._current_initial_pose.metadata(),
                "initial_target_tcp": self._current_initial_target_tcp,
                "gripper_forever_closed": self.task.gripper_forever_closed,
            },
        )
        self._current_episode_dir = episode_dir
        self._initial_pose_completed = False
        self._pending_initial_gripper_close = False
        self._current_initial_pose = None
        self._current_initial_target_tcp = None
        if self.teleop_service is not None:
            self.teleop_service.set_teleop_enabled(True)
        self.sessions.record_operator_event("episode_started", {"episode_dir": str(episode_dir)})
        LOGGER.info("Episode started: %s", episode_dir)

    def _stop_episode_locked(self) -> None:
        if self._current_episode_dir is None:
            raise OperatorActionError("No active episode to stop.")
        episode_dir = self._current_episode_dir
        worker_error = self._required_worker_error()
        if worker_error is not None:
            outcome = "failed"
        else:
            outcome = "saved"
        self.sessions.close_active_episode()
        self._current_episode_dir = None
        self._home_joint_completed = False
        metadata_updates = {}
        if self.image_stream_recorders:
            metadata_updates["image_stream_recorders"] = {
                name: recorder.snapshot() for name, recorder in self.image_stream_recorders.items()
            }
        if episode_dir is not None:
            try:
                image_stream_summary = self._flush_image_streams_locked(episode_dir)
            except Exception as exc:
                outcome = "failed"
                metadata_updates["image_stream_error"] = str(exc)
                image_stream_summary = None
            if image_stream_summary is not None:
                metadata_updates["image_streams"] = image_stream_summary
        if worker_error is not None:
            metadata_updates["failure_reason"] = str(worker_error)
        self.sessions.finalize_episode(episode_dir, outcome=outcome, metadata_updates=metadata_updates)
        qc_report = None
        if outcome == "saved":
            try:
                qc_report = self._run_episode_qc_locked(episode_dir)
                self.sessions.update_episode_metadata(episode_dir, {"episode_qc": episode_qc_manifest_summary(qc_report)})
            except Exception as exc:
                LOGGER.exception("Episode QC failed for %s", episode_dir)
                self.sessions.update_episode_metadata(episode_dir, {"episode_qc_error": str(exc)})
        self._latest_saved_episode_dir = episode_dir if outcome == "saved" else self._latest_saved_episode_dir
        if self.teleop_service is not None:
            self.teleop_service.set_teleop_enabled(outcome == "saved")
        self.sessions.record_operator_event("episode_stopped", {"episode_dir": str(episode_dir), "outcome": outcome})
        LOGGER.info("Episode %s: %s", outcome, episode_dir)
        if qc_report is not None:
            print(f"[QC] {episode_dir.name}: {format_episode_qc_summary(qc_report)}", flush=True)

    def _poll_worker_failures_locked(self) -> None:
        if self._current_episode_dir is None:
            return
        if self._required_worker_error() is None:
            return
        self._stop_episode_locked()

    def _flush_image_streams_locked(self, episode_dir: Path) -> dict | None:
        if not self.image_stream_recorders:
            return None
        summaries = {}
        for name, recorder in self.image_stream_recorders.items():
            summary = recorder.flush_episode(episode_dir)
            if summary is not None:
                summaries[name] = summary
        if not summaries:
            return None
        return summaries

    def _run_episode_qc_locked(self, episode_dir: Path) -> dict:
        return analyze_episode_quality(
            episode_dir,
            expected_hz=build_expected_episode_hz(self.workspace, self.task),
            write=True,
        )

    def _required_worker_error(self) -> Exception | None:
        for worker in self.workers.values():
            if worker.required and worker.error is not None:
                return worker.error
        return None

    def _discard_latest_episode_locked(self) -> None:
        if self._current_episode_dir is not None:
            raise OperatorActionError("Cannot discard while recording. Stop/save the active episode first.")
        episode_dir = self._latest_saved_episode_dir or self.sessions.get_latest_saved_episode_dir()
        if episode_dir is None:
            raise OperatorActionError("No saved episode to discard.")
        self.sessions.discard_episode(episode_dir)
        self.sessions.record_operator_event("episode_discarded", {"episode_dir": str(episode_dir)})
        LOGGER.info("Discarded episode: %s", episode_dir)
        self._latest_saved_episode_dir = self.sessions.get_latest_saved_episode_dir()

    def _is_ready_for_episode_locked(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if self.teleop_server is None or not self.teleop_server.is_alive():
            reasons.append("teleop server is not running")
        if not self.state_monitor.is_healthy(max_age_sec=self.task.modality.controller_state_max_age_sec):
            reasons.append("controller state is not healthy")
        if not self._initial_pose_completed:
            reasons.append("robot has not been moved to the task initial pose with H")
        if self._pending_initial_gripper_close:
            reasons.append("initial gripper close is pending; press C to confirm")
        if self.task.collection.require_quest_connection and (
            self.teleop_service is None
            or not self.teleop_service.has_recent_message(self.task.collection.quest_message_timeout_sec)
        ):
            reasons.append("Quest connection is not active")
        for name, worker in self.workers.items():
            if worker.required and worker.error is not None:
                reasons.append(f"{name} failed: {worker.error}")
        return not reasons, reasons

    def _build_status_locked(self) -> dict:
        ready, reasons = self._is_ready_for_episode_locked()
        preview = self._preview_metadata_locked(ready=ready)
        next_episode_index = self.sessions.get_next_episode_index()
        quest_connected = self.teleop_service is not None and self.teleop_service.has_recent_message(
            self.task.collection.quest_message_timeout_sec
        )
        teleop_enabled = self.teleop_service is not None and self.teleop_service.is_teleop_enabled()
        run_dir = self.sessions.get_active_run_dir()
        return {
            "mode": "collect",
            "run_name": self.task.task_name,
            "run_dir": None if run_dir is None else str(run_dir),
            "ready": ready,
            "reasons": reasons,
            "active_episode": None if self._current_episode_dir is None else str(self._current_episode_dir),
            "active_episode_name": None if self._current_episode_dir is None else self._current_episode_dir.name,
            "latest_saved_episode": None if self._latest_saved_episode_dir is None else str(self._latest_saved_episode_dir),
            "latest_saved_episode_name": None if self._latest_saved_episode_dir is None else self._latest_saved_episode_dir.name,
            "next_episode_index": next_episode_index,
            "next_episode_name": f"episode_{next_episode_index:04d}",
            "quest_connected": quest_connected,
            "teleop_enabled": teleop_enabled,
            "gripper_forever_closed": self.task.gripper_forever_closed,
            "pending_initial_gripper_close": self._pending_initial_gripper_close,
            "home_joint_completed": self._home_joint_completed,
            "controller_state": self.state_monitor.snapshot(),
            "rgb_cameras": {role: buffer.snapshot() for role, buffer in self.rgb_camera_buffers.items()},
            "gelsight_enabled": self.task.gelsight.enabled,
            "gelsight_frame": None if self.gelsight_frame_buffer is None else self.gelsight_frame_buffer.snapshot(),
            "image_stream_recorders": {
                name: recorder.snapshot() for name, recorder in self.image_stream_recorders.items()
            },
            "workers": {
                name: {"alive": worker.is_alive(), "error": None if worker.error is None else str(worker.error)}
                for name, worker in self.workers.items()
            },
            "allowed_actions": {
                "reset_home_joints": self._current_episode_dir is None and self.task.home_joint_positions_rad is not None,
                "reset": self._current_episode_dir is None,
                "confirm_gripper_closed": self._current_episode_dir is None and self._pending_initial_gripper_close,
                "open_gripper": self._current_episode_dir is None
                and self.task.gripper_forever_closed
                and self._current_initial_pose is not None,
                "start": ready and self._current_episode_dir is None,
                "stop": self._current_episode_dir is not None,
                "mark_success": False,
                "mark_fail": False,
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
        label = f"Live pre-episode {role} view"
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
            label=f"Live pre-episode {name} view: {sample.name}",
            image_format=self.workspace.recording.image_format,
        )

    def _get_live_preview_sample_locked(self, name: str, *, ready: bool):
        del ready
        role = self.workspace.operator_ui.preview_camera_role
        if name != role or self._current_episode_dir is not None:
            return None
        buffer = self.rgb_camera_buffers.get(role)
        if buffer is None:
            return None
        return buffer.get_latest_optional(max_age_sec=self.workspace.operator_ui.snapshot_max_age_sec)

    def _print_status_if_needed(self) -> None:
        now = time.time()
        if now - self._last_status_print_wall_time < 1.0 / max(self.task.collection.status_print_hz, 1e-6):
            return
        self._last_status_print_wall_time = now
        with self._operator_lock:
            status = self._build_status_locked()
        self.sessions.write_latest_status(status)
        summary = (
            f"[{'READY' if status['ready'] else 'WAIT'}] "
            f"next={status['next_episode_name']} "
            f"recording={status['active_episode_name'] or 'off'} "
            f"quest={'ok' if status['quest_connected'] else 'missing'} "
            f"teleop={'on' if status['teleop_enabled'] else 'blocked'} "
            f"controller_age={status['controller_state']['age_sec'] if status['controller_state']['age_sec'] is not None else 'n/a'} "
            f"rgb_cameras={len(self.rgb_camera_buffers)} "
            f"gelsight={'on' if self.task.gelsight.enabled else 'off'}"
        )
        if status["reasons"]:
            summary = f"{summary} reasons={'; '.join(status['reasons'])}"
        print(summary, flush=True)

    def _print_banner(self, run_dir: Path) -> None:
        print(f"Data Collector run started: {run_dir}", flush=True)
        print(f"Task: {self.task.task_name}", flush=True)
        print("Checklist:", flush=True)
        print("- Controller PC: vt-dual-franka-controller is already running", flush=True)
        print("- Workspace PC: Quest connected and streaming", flush=True)
        print("- Press H for the task initial EEF pose. Optional: press J before R when a home-joint reset is needed.", flush=True)
        print("- If you press J, press H again before R.", flush=True)
        print("- After E, Quest teleop stays enabled until the next H blocks teleop and moves to the initial pose.", flush=True)
        if self.task.gripper_forever_closed:
            print("- Forever-closed gripper is enabled. Press O to open between episodes, then C to close before R.", flush=True)
        print(
            "Hotkeys: J=home joints  H=initial pose  O=open gripper  C=confirm/close gripper  R=start recording  E=end/save  D=discard last saved  Q=quit",
            flush=True,
        )
        if self.workspace.operator_ui.enabled:
            print(
                f"Operator UI: http://{self.workspace.operator_ui.host}:{self.workspace.operator_ui.port}/operator",
                flush=True,
            )

    def _shutdown(self) -> None:
        if self.teleop_service is not None:
            self.teleop_service.set_teleop_enabled(False)
        self.sessions.record_operator_event("run_stopped")
        stop_thread_workers(self.workers)
        if self.state_bridge is not None:
            self.state_bridge.stop()
        if self.operator_server is not None:
            self.operator_server.stop()
        if self.teleop_server is not None:
            self.teleop_server.stop()
        self.state_monitor.stop()
        self.sessions.stop_run()


def run_standalone_teleop_app(service: QuestTeleopService, workspace: WorkspaceSettings) -> None:
    uvicorn.run(create_teleop_app(service), host=workspace.teleop.host, port=workspace.teleop.port)
