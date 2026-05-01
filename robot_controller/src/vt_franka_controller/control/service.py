from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Deque, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from vt_franka_shared.interpolation import PoseTrajectoryInterpolator, pose_distance
from vt_franka_shared.models import (
    ControllerState,
    GripperGraspCommand,
    GripperWidthCommand,
    HealthStatus,
    ResetCommand,
    TcpTargetCommand,
)
from vt_franka_shared.pose_math import pose7d_to_pose6d

from ..settings import ControllerSettings
from ..backends.base import FrankaBackend

LOGGER = logging.getLogger(__name__)


class ControllerBusyError(RuntimeError):
    """Raised when controller cannot accept a command in its current state."""


class ControllerService:
    def __init__(self, settings: ControllerSettings, backend: FrankaBackend) -> None:
        self.settings = settings
        self.backend = backend
        self.command_queue: Deque[dict] = deque(maxlen=settings.control.max_queue_size)
        self._queue_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._cached_state: Optional[ControllerState] = None
        self._control_thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._reset_in_progress = threading.Event()
        self._gripper_lock = threading.Lock()
        self._mode_lock = threading.RLock()
        self._pose_interp: Optional[PoseTrajectoryInterpolator] = None
        self._last_waypoint_time: Optional[float] = None
        self._state_cache_period = 1.0 / max(self.settings.control.state_cache_hz, 1e-6)
        self._next_state_refresh_time: Optional[float] = None

    def start(self) -> None:
        with self._mode_lock:
            self._start_control_loop_locked()

    def shutdown(self) -> None:
        with self._mode_lock:
            self._stop_control_loop_locked()
            self.backend.shutdown()

    def queue_tcp_command(self, command: TcpTargetCommand) -> None:
        self._assert_accepting_commands()
        target_pose = pose7d_to_pose6d(command.target_tcp)
        target_duration_sec = (
            float(command.target_duration_sec)
            if command.target_duration_sec is not None
            else 1.0 / self.settings.control.teleop_command_hz
        )
        target_time = time.monotonic() + target_duration_sec
        with self._queue_lock:
            self.command_queue.append({"target_pose": target_pose, "target_time": target_time})

    def queue_gripper_width_command(self, command: GripperWidthCommand) -> None:
        self._assert_accepting_commands()

        if command.blocking:
            with self._gripper_lock:
                self.backend.move_gripper(command.width, command.velocity, command.force_limit)
            return

        def move() -> None:
            with self._gripper_lock:
                self.backend.move_gripper(command.width, command.velocity, command.force_limit)

        threading.Thread(target=move, name="gripper-width", daemon=True).start()

    def queue_gripper_grasp_command(self, command: GripperGraspCommand) -> None:
        self._assert_accepting_commands()

        if command.blocking:
            with self._gripper_lock:
                self.backend.grasp(command.velocity, command.force_limit)
            return

        def grasp() -> None:
            with self._gripper_lock:
                self.backend.grasp(command.velocity, command.force_limit)

        threading.Thread(target=grasp, name="gripper-grasp", daemon=True).start()

    def stop_gripper(self) -> None:
        self.backend.stop_gripper()

    def go_home(self) -> None:
        command = ResetCommand(
            profile="home",
            eef_pose_xyz_rpy_deg=list(self.settings.control.home_ee_pose),
            eef_duration_sec=self.settings.control.home_duration_sec,
            gripper_target="unchanged",
            source="controller_home",
        )
        self.run_reset(command)

    def go_ready(self) -> None:
        if self.settings.control.ready_ee_pose is None:
            raise RuntimeError("Ready EE pose is not configured")
        command = ResetCommand(
            profile="ready",
            joint_positions=None if self.settings.control.ready_joint_positions is None else list(self.settings.control.ready_joint_positions),
            joint_duration_sec=self.settings.control.ready_joint_duration_sec,
            eef_pose_xyz_rpy_deg=list(self.settings.control.ready_ee_pose),
            eef_duration_sec=self.settings.control.ready_duration_sec,
            gripper_target="unchanged",
            source="controller_ready",
        )
        self.run_reset(command)

    def run_reset(self, command: ResetCommand) -> dict[str, str]:
        if self._reset_in_progress.is_set():
            raise ControllerBusyError("Controller reset is already in progress")
        self._reset_in_progress.set()
        try:
            with self._mode_lock:
                path = self._run_reset_locked(command)
            return {"status": "ok", "profile": command.profile, "path": path, "gripper_target": command.gripper_target}
        finally:
            self._reset_in_progress.clear()

    def get_state(self) -> ControllerState:
        with self._state_lock:
            if self._cached_state is not None:
                return self._cached_state
        state = self.backend.get_controller_state(self.settings.control.control_frequency_hz)
        with self._state_lock:
            self._cached_state = state
        return state

    def get_health(self) -> HealthStatus:
        with self._state_lock:
            last_state = self._cached_state
        return HealthStatus(
            ok=self._running.is_set(),
            backend=self.backend.name,
            message="resetting" if self._reset_in_progress.is_set() else "running",
            queue_depth=self._queue_depth(),
            control_loop_running=self._running.is_set(),
            last_state_monotonic_time=last_state.monotonic_time if last_state else None,
        )

    def _refresh_state(self) -> None:
        state = self.backend.get_controller_state(self.settings.control.control_frequency_hz)
        with self._state_lock:
            self._cached_state = state

    def _run_reset_locked(self, command: ResetCommand) -> str:
        target_pose6d = self._reset_target_pose6d(command)
        use_fast_path = self._can_use_fast_path(command, target_pose6d)
        with self._queue_lock:
            self.command_queue.clear()
        if use_fast_path:
            self._queue_fast_reset_target(target_pose6d, command.eef_duration_sec)
            self._execute_reset_gripper(command)
            self._wait_for_pose_settle(target_pose6d, motion_duration_sec=command.eef_duration_sec)
            return "fast"

        self._run_blocking_reset_command(command)
        self._refresh_state()
        if target_pose6d is not None:
            self._wait_for_pose_settle(target_pose6d)
        return "slow"

    def _run_blocking_reset_command(self, command: ResetCommand) -> None:
        self._stop_control_loop_locked()
        with self._queue_lock:
            self.command_queue.clear()
        self.backend.shutdown()
        if command.joint_positions is not None:
            self.backend.move_to_joints(
                command.joint_positions,
                duration_sec=command.joint_duration_sec,
            )
            self.backend.shutdown()
        if command.eef_pose_xyz_rpy_deg is not None:
            self.backend.go_home(command.eef_pose_xyz_rpy_deg, command.eef_duration_sec or self.settings.control.ready_duration_sec)
        self._execute_reset_gripper(command)
        self._start_control_loop_locked()

    def _reset_target_pose6d(self, command: ResetCommand) -> np.ndarray | None:
        if command.eef_pose_xyz_rpy_deg is None:
            return None
        eef_pose = np.asarray(command.eef_pose_xyz_rpy_deg, dtype=np.float64)
        return np.concatenate(
            [
                eef_pose[:3],
                Rotation.from_euler("xyz", eef_pose[3:], degrees=True).as_rotvec(),
            ]
        )

    def _can_use_fast_path(self, command: ResetCommand, target_pose6d: np.ndarray | None) -> bool:
        if command.joint_positions is not None or target_pose6d is None:
            return False
        if not self._running.is_set():
            return False
        with self._state_lock:
            current_state = self._cached_state
        if current_state is None:
            current_pose6d = pose7d_to_pose6d(self.backend.get_tcp_pose())
        else:
            current_pose6d = pose7d_to_pose6d(current_state.tcp_pose)
        position_error, rotation_error = pose_distance(current_pose6d, target_pose6d)
        return (
            position_error <= self.settings.control.reset_fast_path_position_threshold_m
            and np.degrees(rotation_error) <= self.settings.control.reset_fast_path_rotation_threshold_deg
        )

    def _queue_fast_reset_target(self, target_pose6d: np.ndarray, duration_sec: float | None) -> None:
        now = time.monotonic()
        target_duration_sec = duration_sec or self.settings.control.ready_duration_sec
        target_time = now + target_duration_sec
        with self._queue_lock:
            self.command_queue.clear()
            self.command_queue.append({"target_pose": target_pose6d, "target_time": target_time})

    def _execute_reset_gripper(self, command: ResetCommand) -> None:
        if command.gripper_target == "unchanged":
            return
        with self._gripper_lock:
            if command.gripper_target == "open":
                if command.gripper_width is None:
                    raise RuntimeError("Reset gripper target 'open' requires gripper_width")
                self.backend.move_gripper(
                    command.gripper_width,
                    command.gripper_velocity or 0.1,
                    command.gripper_force_limit or 5.0,
                )
                return
            self.backend.grasp(
                command.gripper_velocity or 0.1,
                command.gripper_force_limit or 5.0,
            )

    def _wait_for_pose_settle(self, target_pose6d: np.ndarray, motion_duration_sec: float | None = None) -> None:
        deadline = time.monotonic() + self.settings.control.reset_settle_timeout_sec + max(motion_duration_sec or 0.0, 0.0)
        dwell_start: float | None = None
        while time.monotonic() <= deadline:
            state = self.backend.get_controller_state(self.settings.control.control_frequency_hz)
            with self._state_lock:
                self._cached_state = state
            current_pose6d = pose7d_to_pose6d(state.tcp_pose)
            position_error, rotation_error = pose_distance(current_pose6d, target_pose6d)
            if (
                position_error <= self.settings.control.reset_settle_position_threshold_m
                and np.degrees(rotation_error) <= self.settings.control.reset_settle_rotation_threshold_deg
            ):
                if dwell_start is None:
                    dwell_start = time.monotonic()
                elif time.monotonic() - dwell_start >= self.settings.control.reset_settle_dwell_sec:
                    return
            else:
                dwell_start = None
            time.sleep(0.05)
        raise RuntimeError("Reset target did not settle within timeout")

    def _assert_accepting_commands(self) -> None:
        if self._reset_in_progress.is_set():
            raise ControllerBusyError("Controller reset is in progress")

    def _queue_depth(self) -> int:
        with self._queue_lock:
            return len(self.command_queue)

    def _start_control_loop_locked(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._control_thread = threading.Thread(target=self._control_loop, name="controller-loop", daemon=True)
        self._control_thread.start()

    def _stop_control_loop_locked(self) -> None:
        self._running.clear()
        if self._control_thread is not None:
            self._control_thread.join(timeout=2.0)
            self._control_thread = None

    def _control_loop(self) -> None:
        current_pose = pose7d_to_pose6d(self.backend.get_tcp_pose())
        start_time = time.monotonic()
        self._pose_interp = PoseTrajectoryInterpolator(times=np.array([start_time]), poses=np.array([current_pose]))
        self._last_waypoint_time = start_time
        self._next_state_refresh_time = start_time
        self.backend.start_cartesian_impedance(
            self.settings.control.cartesian_stiffness,
            self.settings.control.cartesian_damping,
        )

        iteration = 0
        period = 1.0 / self.settings.control.control_frequency_hz
        while self._running.is_set():
            now = time.monotonic()
            target_pose = self._pose_interp(now)
            self.backend.update_desired_tcp(target_pose)

            if self._next_state_refresh_time is None or now >= self._next_state_refresh_time:
                try:
                    self._refresh_state()
                except Exception:  # pragma: no cover - hardware dependent failure path
                    LOGGER.exception("Failed to refresh controller state cache")
                self._next_state_refresh_time = now + self._state_cache_period

            with self._queue_lock:
                try:
                    command = self.command_queue.popleft()
                except IndexError:
                    command = None

            if command is not None:
                current_time = now + period
                self._pose_interp = self._pose_interp.schedule_waypoint(
                    pose=command["target_pose"],
                    time=command["target_time"],
                    curr_time=current_time,
                    last_waypoint_time=self._last_waypoint_time,
                    max_pos_speed=self.settings.control.max_policy_pos_speed_m_s,
                    max_rot_speed=self.settings.control.max_policy_rot_speed_rad_s,
                )
                self._last_waypoint_time = command["target_time"]

            deadline = start_time + (iteration + 1) * period
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            iteration += 1
