#!/usr/bin/env python3
from __future__ import annotations

import csv
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from threading import Event, RLock, Thread

import numpy as np
import torch
from polymetis import GripperInterface, RobotInterface
from scipy.spatial.transform import Rotation, Slerp
import uvicorn

from vt_dual_franka_controller.api.demo_state_app import create_demo_state_app
from vt_dual_franka_controller.settings import BackendSettings, ControlSettings, ControllerSettings, ServerSettings
from vt_dual_franka_shared.models import ControllerState, HealthStatus
from vt_dual_franka_shared.pose_math import xyzw_to_wxyz


# Edit these values directly for your test.
ROBOT_IP = "127.0.0.1"
ROBOT_PORT = 50051
GRIPPER_IP = "127.0.0.1"
GRIPPER_PORT = 50052

# Serve read-only robot state to the workspace PC while this demo is running.
STATE_SERVER_HOST = "0.0.0.0"
STATE_SERVER_PORT = 8092
STATE_CACHE_HZ = 60.0

# Initial pose before starting the waypoint sequence.
INITIAL_POSITION_M = [-0.12, 0.48, 0.5]
END_POSITION_M = [-0.12, 0.45, 0.6]
INITIAL_RPY_DEG = [180, 0, 45]
RESET_POSITION_M = INITIAL_POSITION_M
RESET_RPY_DEG = INITIAL_RPY_DEG
DRAW_Z_M = 0.45
LIFT_Z_M = 0.48

# Motion timing
POSE_MOVE_TIME_S = 4.0
CONTROL_HZ = 100.0
DENSE_WAYPOINT_DT_S = 1.0 / CONTROL_HZ
PEN_DOWN_KEY_WAYPOINT_HOLD_S = 0.0
PEN_UP_KEY_WAYPOINT_HOLD_S = 0.05
# Classify pure transit segments at the lifted writing height as pen-up.
PEN_UP_Z_THRESHOLD_M = 0.5 * (DRAW_Z_M + LIFT_Z_M)
MIN_SEGMENT_DURATION_S = 0.25
PEN_DOWN_TRANSLATION_SPEED_MPS = 0.05
PEN_UP_TRANSLATION_SPEED_MPS = 0.070
PEN_DOWN_ROTATION_SPEED_DEG_S = 45.0
PEN_UP_ROTATION_SPEED_DEG_S = 90.0
PEN_DOWN_PROGRESS_THRESHOLD = 0.99
PEN_UP_PROGRESS_THRESHOLD = 0.995
PEN_DOWN_ENDPOINT_TOLERANCE_M = 0.003
PEN_UP_ENDPOINT_TOLERANCE_M = 0.004
PEN_DOWN_LATERAL_TOLERANCE_M = 0.003
PEN_UP_LATERAL_TOLERANCE_M = 0.004
PEN_DOWN_CATCH_UP_TIMEOUT_S = 0.2
PEN_UP_CATCH_UP_TIMEOUT_S = 0.3

# Per-command tracking log. This records the action command and measured EE pose
# at each dense waypoint so contact-induced tracking error can be inspected.
TRAJECTORY_LOG_ENABLED = True
TRAJECTORY_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"

# Keep one Cartesian impedance policy active for the whole trajectory.
CARTESIAN_STIFFNESS = [750.0, 750.0, 750.0, 15.0, 15.0, 15.0]
CARTESIAN_DAMPING = [37.0, 37.0, 37.0, 2.0, 2.0, 2.0]
HOLD_FINAL_POSE_S = 0.5

# Gripper commands use meters/second and Newtons.
CLOSE_SPEED_MPS = 0.03
CLOSE_FORCE_N = 20.0

# Safety: keep this on so the robot does not move until you explicitly confirm.
ASK_FOR_CONFIRMATION = True

# Optional: ask before *each* waypoint (True) or only once before the whole sequence (False).
CONFIRM_EACH_WAYPOINT = False

# Waypoints after the initial pose: (position [m], rpy [deg])
# Each stroke is written independently and then returns to a reset pose.
# This reduces error accumulation across letters and follows a more human-like
# stroke order for A and M.
KEY_WAYPOINTS = [
    # A stroke 1: top tip to bottom-left.
    ([-0.13, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([-0.13, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([-0.16, 0.45, DRAW_Z_M], [-180, 0, 45]),
    ([-0.16, 0.45, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # A stroke 2: top tip to bottom-right.
    ([-0.13, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([-0.13, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([-0.10, 0.45, DRAW_Z_M], [-180, 0, 45]),
    ([-0.10, 0.45, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # A stroke 3: crossbar from left midpoint to right midpoint.
    ([-0.145, 0.51, LIFT_Z_M], [-180, 0, 45]),
    ([-0.145, 0.51, DRAW_Z_M], [-180, 0, 45]),
    ([-0.115, 0.51, DRAW_Z_M], [-180, 0, 45]),
    ([-0.115, 0.51, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # M stroke 1: left vertical.
    ([-0.08, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([-0.08, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([-0.08, 0.45, DRAW_Z_M], [-180, 0, 45]),
    ([-0.08, 0.45, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # M stroke 2: left top to center valley.
    ([-0.08, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([-0.08, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([-0.055, 0.49, DRAW_Z_M], [-180, 0, 45]),
    ([-0.055, 0.49, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # M stroke 3: right top to center valley.
    ([-0.03, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([-0.03, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([-0.055, 0.49, DRAW_Z_M], [-180, 0, 45]),
    ([-0.055, 0.49, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # M stroke 4: right vertical.
    ([-0.03, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([-0.03, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([-0.03, 0.45, DRAW_Z_M], [-180, 0, 45]),
    ([-0.03, 0.45, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # O stroke 1: top horizontal.
    ([-0.005, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([-0.005, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([0.045, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([0.045, 0.57, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # O stroke 2: right vertical.
    ([0.045, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([0.045, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([0.045, 0.45, DRAW_Z_M], [-180, 0, 45]),
    ([0.045, 0.45, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # O stroke 3: bottom horizontal.
    ([-0.005, 0.46, LIFT_Z_M], [-180, 0, 45]),
    ([-0.005, 0.46, DRAW_Z_M], [-180, 0, 45]),
    ([0.045, 0.46, DRAW_Z_M], [-180, 0, 45]),
    ([0.045, 0.46, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # O stroke 4: left vertical.
    ([-0.005, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([-0.005, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([-0.005, 0.45, DRAW_Z_M], [-180, 0, 45]),
    ([-0.005, 0.45, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # S stroke 1: top horizontal.
    ([0.065, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([0.065, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([0.10, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([0.10, 0.57, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # S stroke 2: upper vertical.
    ([0.065, 0.57, LIFT_Z_M], [-180, 0, 45]),
    ([0.065, 0.57, DRAW_Z_M], [-180, 0, 45]),
    ([0.065, 0.51, DRAW_Z_M], [-180, 0, 45]),
    ([0.065, 0.51, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # S stroke 3: middle horizontal.
    ([0.065, 0.51, LIFT_Z_M], [-180, 0, 45]),
    ([0.065, 0.51, DRAW_Z_M], [-180, 0, 45]),
    ([0.10, 0.51, DRAW_Z_M], [-180, 0, 45]),
    ([0.10, 0.51, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # S stroke 4: lower vertical.
    ([0.10, 0.51, LIFT_Z_M], [-180, 0, 45]),
    ([0.10, 0.51, DRAW_Z_M], [-180, 0, 45]),
    ([0.10, 0.45, DRAW_Z_M], [-180, 0, 45]),
    ([0.10, 0.45, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),

    # S stroke 5: bottom horizontal.
    ([0.065, 0.46, LIFT_Z_M], [-180, 0, 45]),
    ([0.065, 0.46, DRAW_Z_M], [-180, 0, 45]),
    ([0.11, 0.46, DRAW_Z_M], [-180, 0, 45]),
    ([0.11, 0.46, LIFT_Z_M], [-180, 0, 45]),
    (RESET_POSITION_M, RESET_RPY_DEG),
    (END_POSITION_M, RESET_RPY_DEG),
]


@dataclass(frozen=True)
class DenseWaypoint:
    position_m: np.ndarray
    quaternion_xyzw: np.ndarray
    segment_index: int
    segment_step_index: int
    segment_step_count: int
    is_hold: bool = False
    phase: str = "move"


def tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def format_array(values) -> str:
    return np.array2string(np.asarray(values, dtype=np.float64), precision=4, suppress_small=True)


def quaternion_xyzw_to_rpy_deg(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quaternion_xyzw).as_euler("xyz", degrees=True)


def maybe_lock(lock):
    return lock if lock is not None else nullcontext()


def confirm_or_abort(message: str) -> None:
    if not ASK_FOR_CONFIRMATION:
        return
    response = input(f"{message} Type 'yes' to continue: ").strip().lower()
    if response != "yes":
        raise SystemExit("Aborted.")


def rpy_deg_to_quat_xyzw(rpy_deg: np.ndarray) -> np.ndarray:
    return Rotation.from_euler("xyz", rpy_deg, degrees=True).as_quat()


def quaternion_angle_error_deg(target_quaternion_xyzw: np.ndarray, actual_quaternion_xyzw: np.ndarray) -> float:
    target_rotation = Rotation.from_quat(target_quaternion_xyzw)
    actual_rotation = Rotation.from_quat(actual_quaternion_xyzw)
    return float((target_rotation.inv() * actual_rotation).magnitude() * 180.0 / np.pi)


def rotation_distance_deg(start_quaternion_xyzw: np.ndarray, target_quaternion_xyzw: np.ndarray) -> float:
    start_rotation = Rotation.from_quat(start_quaternion_xyzw)
    target_rotation = Rotation.from_quat(target_quaternion_xyzw)
    return float((start_rotation.inv() * target_rotation).magnitude() * 180.0 / np.pi)


def segment_duration_s(
    start_position_m: np.ndarray,
    start_quaternion_xyzw: np.ndarray,
    target_position_m: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
) -> float:
    translation_distance_m = float(np.linalg.norm(target_position_m - start_position_m))
    rotation_distance = rotation_distance_deg(start_quaternion_xyzw, target_quaternion_xyzw)
    pen_is_up = start_position_m[2] >= PEN_UP_Z_THRESHOLD_M and target_position_m[2] >= PEN_UP_Z_THRESHOLD_M

    translation_speed_mps = PEN_UP_TRANSLATION_SPEED_MPS if pen_is_up else PEN_DOWN_TRANSLATION_SPEED_MPS
    rotation_speed_deg_s = PEN_UP_ROTATION_SPEED_DEG_S if pen_is_up else PEN_DOWN_ROTATION_SPEED_DEG_S

    translation_duration_s = translation_distance_m / max(translation_speed_mps, 1e-6)
    rotation_duration_s = rotation_distance / max(rotation_speed_deg_s, 1e-6)
    return max(MIN_SEGMENT_DURATION_S, translation_duration_s, rotation_duration_s)


def segment_control_params(start_position_m: np.ndarray, target_position_m: np.ndarray) -> tuple[float, float, float, float]:
    pen_is_up = start_position_m[2] >= PEN_UP_Z_THRESHOLD_M and target_position_m[2] >= PEN_UP_Z_THRESHOLD_M
    if pen_is_up:
        return (
            PEN_UP_PROGRESS_THRESHOLD,
            PEN_UP_ENDPOINT_TOLERANCE_M,
            PEN_UP_LATERAL_TOLERANCE_M,
            PEN_UP_CATCH_UP_TIMEOUT_S,
        )
    return (
        PEN_DOWN_PROGRESS_THRESHOLD,
        PEN_DOWN_ENDPOINT_TOLERANCE_M,
        PEN_DOWN_LATERAL_TOLERANCE_M,
        PEN_DOWN_CATCH_UP_TIMEOUT_S,
    )


def project_progress(
    start_position_m: np.ndarray,
    target_position_m: np.ndarray,
    actual_position_m: np.ndarray,
) -> tuple[float, float, float]:
    segment_vector = target_position_m - start_position_m
    segment_length_sq = float(segment_vector @ segment_vector)
    if segment_length_sq <= 1e-12:
        endpoint_distance = float(np.linalg.norm(actual_position_m - target_position_m))
        return 1.0, 0.0, endpoint_distance

    actual_vector = actual_position_m - start_position_m
    progress = float((actual_vector @ segment_vector) / segment_length_sq)
    projected_position = start_position_m + np.clip(progress, 0.0, 1.0) * segment_vector
    lateral_error = float(np.linalg.norm(actual_position_m - projected_position))
    endpoint_distance = float(np.linalg.norm(actual_position_m - target_position_m))
    return progress, lateral_error, endpoint_distance


def write_tracking_summary(log_path: Path, position_errors_m: list[float], orientation_errors_deg: list[float]) -> None:
    if not position_errors_m:
        print("Trajectory tracking log is empty.")
        return

    position_errors = np.asarray(position_errors_m, dtype=np.float64)
    orientation_errors = np.asarray(orientation_errors_deg, dtype=np.float64)
    print("Trajectory tracking log:")
    print(f"  path: {log_path}")
    print(
        "  position error (m): "
        f"mean={np.mean(position_errors):.5f}, "
        f"p95={np.percentile(position_errors, 95):.5f}, "
        f"max={np.max(position_errors):.5f}"
    )
    print(
        "  orientation error (deg): "
        f"mean={np.mean(orientation_errors):.3f}, "
        f"p95={np.percentile(orientation_errors, 95):.3f}, "
        f"max={np.max(orientation_errors):.3f}"
    )


class DemoStateService:
    def __init__(
        self,
        robot: RobotInterface,
        gripper: GripperInterface,
        settings: ControllerSettings,
        *,
        robot_lock=None,
        gripper_lock=None,
    ) -> None:
        self.robot = robot
        self.gripper = gripper
        self.settings = settings
        self.robot_lock = robot_lock
        self.gripper_lock = gripper_lock
        self._running = Event()
        self._thread: Thread | None = None
        self._state_lock = RLock()
        self._cached_state: ControllerState | None = None
        self._last_error: str | None = None

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._thread = Thread(target=self._loop, name="demo-state-cache", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def get_state(self) -> ControllerState:
        with self._state_lock:
            if self._cached_state is not None:
                return self._cached_state
            last_error = self._last_error
        state = self._compute_state()
        with self._state_lock:
            self._cached_state = state
            self._last_error = None
        if last_error is not None:
            return state
        return state

    def get_health(self) -> HealthStatus:
        with self._state_lock:
            last_state = self._cached_state
            last_error = self._last_error
        return HealthStatus(
            ok=self._running.is_set() and last_error is None,
            backend="polymetis-demo",
            message="running" if last_error is None else last_error,
            queue_depth=0,
            control_loop_running=self._running.is_set(),
            last_state_monotonic_time=last_state.monotonic_time if last_state is not None else None,
        )

    def _loop(self) -> None:
        period = 1.0 / max(self.settings.control.state_cache_hz, 1e-6)
        while self._running.is_set():
            try:
                state = self._compute_state()
                with self._state_lock:
                    self._cached_state = state
                    self._last_error = None
            except Exception as exc:
                with self._state_lock:
                    self._last_error = str(exc)
            time.sleep(period)

    def _compute_state(self) -> ControllerState:
        with maybe_lock(self.robot_lock):
            robot_state = self.robot.get_robot_state()
            joint_positions = torch.as_tensor(robot_state.joint_positions)
            joint_velocities = torch.as_tensor(robot_state.joint_velocities)
            tau_external = torch.as_tensor(robot_state.motor_torques_external)

            position, quaternion_xyzw = self.robot.robot_model.forward_kinematics(joint_positions)
            pose = np.concatenate([tensor_to_numpy(position), xyzw_to_wxyz(tensor_to_numpy(quaternion_xyzw))])

            rotation_base_to_flange = torch.as_tensor(
                Rotation.from_quat(tensor_to_numpy(quaternion_xyzw)).as_matrix(),
                dtype=joint_positions.dtype,
            )
            rotation_flange_to_base = rotation_base_to_flange.T

            jacobian = self.robot.robot_model.compute_jacobian(joint_positions)
            tcp_velocity_base = jacobian @ joint_velocities
            linear_velocity = tensor_to_numpy(rotation_flange_to_base @ tcp_velocity_base[0:3])
            angular_velocity = tensor_to_numpy(rotation_flange_to_base @ tcp_velocity_base[3:6])
            tcp_velocity = np.concatenate([linear_velocity, angular_velocity])

            wrench_base, _, _, _ = torch.linalg.lstsq(jacobian.T, tau_external)
            force = tensor_to_numpy(rotation_flange_to_base @ wrench_base[0:3])
            torque = tensor_to_numpy(rotation_flange_to_base @ wrench_base[3:6])
            tcp_wrench = np.concatenate([force, torque])

        with maybe_lock(self.gripper_lock):
            gripper_state = self.gripper.get_state()

        try:
            gripper_force = float(gripper_state.force)
        except (AttributeError, TypeError):
            gripper_force = 0.0

        return ControllerState(
            tcp_pose=pose.tolist(),
            tcp_velocity=tcp_velocity.tolist(),
            tcp_wrench=tcp_wrench.tolist(),
            joint_positions=list(robot_state.joint_positions),
            joint_velocities=list(robot_state.joint_velocities),
            gripper_width=float(gripper_state.width),
            gripper_force=gripper_force,
            control_frequency_hz=self.settings.control.control_frequency_hz,
            backend="polymetis-demo",
        )


class ManagedUvicornServer:
    def __init__(self, app, host: str, port: int) -> None:
        self.app = app
        self.host = host
        self.port = port
        self._server: uvicorn.Server | None = None
        self._thread: Thread | None = None

    def start(self) -> None:
        config = uvicorn.Config(self.app, host=self.host, port=self.port, log_level="info")
        self._server = uvicorn.Server(config)
        self._thread = Thread(target=self._server.run, name="demo-state-api", daemon=True)
        self._thread.start()
        time.sleep(0.2)

    def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=2.0)

def print_pose_and_joints(robot: RobotInterface, label: str, *, robot_lock=None) -> None:
    with maybe_lock(robot_lock):
        joint_positions = tensor_to_numpy(robot.get_joint_positions())
        ee_position, ee_quaternion_xyzw = robot.get_ee_pose()
        ee_position = tensor_to_numpy(ee_position)
        ee_quaternion_xyzw = tensor_to_numpy(ee_quaternion_xyzw)
    ee_rpy_deg = quaternion_xyzw_to_rpy_deg(ee_quaternion_xyzw)

    print(f"{label} position xyz (m): {format_array(ee_position)}")
    print(f"{label} orientation quaternion xyzw: {format_array(ee_quaternion_xyzw)}")
    print(f"{label} orientation roll/pitch/yaw (deg): {format_array(ee_rpy_deg)}")
    print(f"{label} joints (rad): {format_array(joint_positions)}")
    print()


def get_ee_pose_numpy(robot: RobotInterface, *, robot_lock=None) -> tuple[np.ndarray, np.ndarray]:
    with maybe_lock(robot_lock):
        ee_position, ee_quaternion_xyzw = robot.get_ee_pose()
    return tensor_to_numpy(ee_position), tensor_to_numpy(ee_quaternion_xyzw)


def print_gripper_state(gripper: GripperInterface, label: str, *, gripper_lock=None) -> None:
    with maybe_lock(gripper_lock):
        gripper_state = gripper.get_state()
    width = float(gripper_state.width)
    force = float(getattr(gripper_state, "force", 0.0))
    print(f"{label} gripper width (m): {width:.4f}")
    print(f"{label} gripper force (N): {force:.2f}")
    print()


def start_cartesian_hold(robot: RobotInterface, *, robot_lock=None) -> None:
    with maybe_lock(robot_lock):
        robot.start_cartesian_impedance(
            Kx=torch.tensor(np.asarray(CARTESIAN_STIFFNESS, dtype=np.float32)),
            Kxd=torch.tensor(np.asarray(CARTESIAN_DAMPING, dtype=np.float32)),
        )
    current_position, current_quaternion_xyzw = get_ee_pose_numpy(robot, robot_lock=robot_lock)
    with maybe_lock(robot_lock):
        robot.update_desired_ee_pose(
            position=torch.tensor(current_position.astype(np.float32)),
            orientation=torch.tensor(current_quaternion_xyzw.astype(np.float32)),
        )


def stream_pose_segment(
    robot: RobotInterface,
    start_position_m: np.ndarray,
    start_quaternion_xyzw: np.ndarray,
    target_position_m: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
    duration_s: float,
    *,
    robot_lock=None,
) -> None:
    if duration_s <= 0.0:
        with maybe_lock(robot_lock):
            robot.update_desired_ee_pose(
                position=torch.tensor(target_position_m.astype(np.float32)),
                orientation=torch.tensor(target_quaternion_xyzw.astype(np.float32)),
            )
        return

    steps = max(1, int(np.ceil(duration_s * CONTROL_HZ)))
    slerp = Slerp(
        [0.0, duration_s],
        Rotation.from_quat(np.vstack([start_quaternion_xyzw, target_quaternion_xyzw])),
    )
    start_time = time.monotonic()

    for step in range(1, steps + 1):
        alpha = step / steps
        segment_time = alpha * duration_s
        position = (1.0 - alpha) * start_position_m + alpha * target_position_m
        quaternion_xyzw = slerp([segment_time]).as_quat()[0]
        with maybe_lock(robot_lock):
            robot.update_desired_ee_pose(
                position=torch.tensor(position.astype(np.float32)),
                orientation=torch.tensor(quaternion_xyzw.astype(np.float32)),
            )

        if step < steps:
            deadline = start_time + segment_time
            remaining = deadline - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)


def interpolate_pose_segment(
    start_position_m: np.ndarray,
    start_quaternion_xyzw: np.ndarray,
    target_position_m: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
    duration_s: float,
    *,
    segment_index: int = 0,
    include_start: bool = False,
) -> list[DenseWaypoint]:
    if duration_s <= 0.0:
        return [
            DenseWaypoint(
                position_m=target_position_m.copy(),
                quaternion_xyzw=target_quaternion_xyzw.copy(),
                segment_index=segment_index,
                segment_step_index=1,
                segment_step_count=1,
                is_hold=False,
                phase="move",
            )
        ]

    steps = max(1, int(np.ceil(duration_s / DENSE_WAYPOINT_DT_S)))
    slerp = Slerp(
        [0.0, 1.0],
        Rotation.from_quat(np.vstack([start_quaternion_xyzw, target_quaternion_xyzw])),
    )
    first_step = 0 if include_start else 1
    dense_waypoints: list[DenseWaypoint] = []

    for step in range(first_step, steps + 1):
        alpha = step / steps
        position = (1.0 - alpha) * start_position_m + alpha * target_position_m
        quaternion_xyzw = slerp([alpha]).as_quat()[0]
        dense_waypoints.append(
            DenseWaypoint(
                position_m=position.astype(np.float64),
                quaternion_xyzw=quaternion_xyzw.astype(np.float64),
                segment_index=segment_index,
                segment_step_index=step,
                segment_step_count=steps,
                is_hold=False,
                phase="move",
            )
        )

    return dense_waypoints


def build_hold_waypoints(
    position_m: np.ndarray,
    quaternion_xyzw: np.ndarray,
    *,
    segment_index: int,
    duration_s: float,
) -> list[DenseWaypoint]:
    hold_steps = max(0, int(np.ceil(duration_s / DENSE_WAYPOINT_DT_S)))
    return [
        DenseWaypoint(
            position_m=position_m.copy(),
            quaternion_xyzw=quaternion_xyzw.copy(),
            segment_index=segment_index,
            segment_step_index=step,
            segment_step_count=hold_steps,
            is_hold=True,
            phase="hold",
        )
        for step in range(1, hold_steps + 1)
    ]


def build_dense_waypoints(
    start_position_m: np.ndarray,
    start_quaternion_xyzw: np.ndarray,
    key_waypoints: list[tuple[list[float], list[float]]],
) -> list[DenseWaypoint]:
    dense_waypoints: list[DenseWaypoint] = []
    previous_position = start_position_m.astype(np.float64)
    previous_quaternion_xyzw = start_quaternion_xyzw.astype(np.float64)

    for segment_index, (position_list, rpy_list) in enumerate(key_waypoints, start=1):
        target_position = np.asarray(position_list, dtype=np.float64)
        target_quaternion_xyzw = rpy_deg_to_quat_xyzw(np.asarray(rpy_list, dtype=np.float64)).astype(np.float64)
        duration_s = segment_duration_s(
            previous_position,
            previous_quaternion_xyzw,
            target_position,
            target_quaternion_xyzw,
        )
        dense_waypoints.extend(
            interpolate_pose_segment(
                previous_position,
                previous_quaternion_xyzw,
                target_position,
                target_quaternion_xyzw,
                duration_s,
                segment_index=segment_index,
            )
        )
        hold_s = PEN_UP_KEY_WAYPOINT_HOLD_S if target_position[2] >= PEN_UP_Z_THRESHOLD_M else PEN_DOWN_KEY_WAYPOINT_HOLD_S
        if hold_s > 0.0:
            dense_waypoints.extend(
                build_hold_waypoints(
                    target_position,
                    target_quaternion_xyzw,
                    segment_index=segment_index,
                    duration_s=hold_s,
                )
            )
        previous_position = target_position
        previous_quaternion_xyzw = target_quaternion_xyzw

    return dense_waypoints


def build_dense_segment(
    start_position_m: np.ndarray,
    start_quaternion_xyzw: np.ndarray,
    target_position_m: np.ndarray,
    target_quaternion_xyzw: np.ndarray,
    *,
    segment_index: int,
) -> list[DenseWaypoint]:
    duration_s = segment_duration_s(
        start_position_m,
        start_quaternion_xyzw,
        target_position_m,
        target_quaternion_xyzw,
    )
    dense_segment = interpolate_pose_segment(
        start_position_m,
        start_quaternion_xyzw,
        target_position_m,
        target_quaternion_xyzw,
        duration_s,
        segment_index=segment_index,
    )
    hold_s = PEN_UP_KEY_WAYPOINT_HOLD_S if target_position_m[2] >= PEN_UP_Z_THRESHOLD_M else PEN_DOWN_KEY_WAYPOINT_HOLD_S
    if hold_s > 0.0:
        dense_segment.extend(
            build_hold_waypoints(
                target_position_m,
                target_quaternion_xyzw,
                segment_index=segment_index,
                duration_s=hold_s,
            )
        )
    return dense_segment


def stream_dense_waypoints(
    robot: RobotInterface,
    dense_waypoints: list[DenseWaypoint],
    *,
    robot_lock=None,
    log_label: str = "amos",
) -> None:
    log_path: Path | None = None
    csv_file = None
    writer = None
    position_errors_m: list[float] = []
    orientation_errors_deg: list[float] = []

    if TRAJECTORY_LOG_ENABLED:
        TRAJECTORY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = TRAJECTORY_LOG_DIR / f"{log_label}_tracking_{timestamp}.csv"
        csv_file = log_path.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "step_index",
            "segment_index",
            "segment_step_index",
            "segment_step_count",
            "is_hold",
            "command_monotonic_time",
            "scheduled_sample_monotonic_time",
            "sample_monotonic_time",
            "sample_delay_sec",
            "target_x",
            "target_y",
            "target_z",
            "target_qx",
            "target_qy",
            "target_qz",
            "target_qw",
            "actual_x",
            "actual_y",
            "actual_z",
            "actual_qx",
            "actual_qy",
            "actual_qz",
            "actual_qw",
            "error_x",
            "error_y",
            "error_z",
            "position_error_m",
            "orientation_error_deg",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    start_time = time.monotonic()
    try:
        for index, waypoint in enumerate(dense_waypoints):
            position = waypoint.position_m
            quaternion_xyzw = waypoint.quaternion_xyzw
            command_time = time.monotonic()
            with maybe_lock(robot_lock):
                robot.update_desired_ee_pose(
                    position=torch.tensor(position.astype(np.float32)),
                    orientation=torch.tensor(quaternion_xyzw.astype(np.float32)),
                )

            scheduled_sample_time = start_time + (index + 1) * DENSE_WAYPOINT_DT_S
            remaining = scheduled_sample_time - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)

            with maybe_lock(robot_lock):
                actual_position, actual_quaternion_xyzw = robot.get_ee_pose()

            sample_time = time.monotonic()
            actual_position = tensor_to_numpy(actual_position)
            actual_quaternion_xyzw = tensor_to_numpy(actual_quaternion_xyzw)
            position_error = actual_position - position
            position_error_norm = float(np.linalg.norm(position_error))
            orientation_error = quaternion_angle_error_deg(quaternion_xyzw, actual_quaternion_xyzw)
            position_errors_m.append(position_error_norm)
            orientation_errors_deg.append(orientation_error)

            if writer is not None:
                writer.writerow(
                    {
                        "step_index": index + 1,
                        "segment_index": waypoint.segment_index,
                        "segment_step_index": waypoint.segment_step_index,
                        "segment_step_count": waypoint.segment_step_count,
                        "is_hold": int(waypoint.is_hold),
                        "command_monotonic_time": f"{command_time:.9f}",
                        "scheduled_sample_monotonic_time": f"{scheduled_sample_time:.9f}",
                        "sample_monotonic_time": f"{sample_time:.9f}",
                        "sample_delay_sec": f"{sample_time - scheduled_sample_time:.9f}",
                        "target_x": f"{position[0]:.9f}",
                        "target_y": f"{position[1]:.9f}",
                        "target_z": f"{position[2]:.9f}",
                        "target_qx": f"{quaternion_xyzw[0]:.9f}",
                        "target_qy": f"{quaternion_xyzw[1]:.9f}",
                        "target_qz": f"{quaternion_xyzw[2]:.9f}",
                        "target_qw": f"{quaternion_xyzw[3]:.9f}",
                        "actual_x": f"{actual_position[0]:.9f}",
                        "actual_y": f"{actual_position[1]:.9f}",
                        "actual_z": f"{actual_position[2]:.9f}",
                        "actual_qx": f"{actual_quaternion_xyzw[0]:.9f}",
                        "actual_qy": f"{actual_quaternion_xyzw[1]:.9f}",
                        "actual_qz": f"{actual_quaternion_xyzw[2]:.9f}",
                        "actual_qw": f"{actual_quaternion_xyzw[3]:.9f}",
                        "error_x": f"{position_error[0]:.9f}",
                        "error_y": f"{position_error[1]:.9f}",
                        "error_z": f"{position_error[2]:.9f}",
                        "position_error_m": f"{position_error_norm:.9f}",
                        "orientation_error_deg": f"{orientation_error:.9f}",
                    }
                )
                csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()
        if log_path is not None:
            write_tracking_summary(log_path, position_errors_m, orientation_errors_deg)


def stream_closed_loop_segment(
    robot: RobotInterface,
    dense_segment: list[DenseWaypoint],
    start_position_m: np.ndarray,
    target_position_m: np.ndarray,
    *,
    robot_lock=None,
    csv_file=None,
    writer=None,
    starting_step_index: int = 0,
    position_errors_m: list[float] | None = None,
    orientation_errors_deg: list[float] | None = None,
) -> int:
    if not dense_segment:
        return starting_step_index

    progress_threshold, endpoint_tolerance_m, lateral_tolerance_m, catch_up_timeout_s = segment_control_params(
        start_position_m,
        target_position_m,
    )
    segment_move_waypoints = [waypoint for waypoint in dense_segment if not waypoint.is_hold]
    segment_hold_waypoints = [waypoint for waypoint in dense_segment if waypoint.is_hold]
    start_time = time.monotonic()
    step_index = starting_step_index
    position_errors_m = position_errors_m if position_errors_m is not None else []
    orientation_errors_deg = orientation_errors_deg if orientation_errors_deg is not None else []

    for move_index, waypoint in enumerate(segment_move_waypoints):
        position = waypoint.position_m
        quaternion_xyzw = waypoint.quaternion_xyzw
        command_time = time.monotonic()
        with maybe_lock(robot_lock):
            robot.update_desired_ee_pose(
                position=torch.tensor(position.astype(np.float32)),
                orientation=torch.tensor(quaternion_xyzw.astype(np.float32)),
            )

        scheduled_sample_time = start_time + (move_index + 1) * DENSE_WAYPOINT_DT_S
        remaining = scheduled_sample_time - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

        with maybe_lock(robot_lock):
            actual_position, actual_quaternion_xyzw = robot.get_ee_pose()

        sample_time = time.monotonic()
        actual_position = tensor_to_numpy(actual_position)
        actual_quaternion_xyzw = tensor_to_numpy(actual_quaternion_xyzw)
        progress, lateral_error, endpoint_distance = project_progress(start_position_m, target_position_m, actual_position)
        position_error = actual_position - position
        position_error_norm = float(np.linalg.norm(position_error))
        orientation_error = quaternion_angle_error_deg(quaternion_xyzw, actual_quaternion_xyzw)
        position_errors_m.append(position_error_norm)
        orientation_errors_deg.append(orientation_error)
        step_index += 1

        if writer is not None:
            writer.writerow(
                {
                    "step_index": step_index,
                    "segment_index": waypoint.segment_index,
                    "segment_step_index": waypoint.segment_step_index,
                    "segment_step_count": waypoint.segment_step_count,
                    "is_hold": int(waypoint.is_hold),
                    "phase": waypoint.phase,
                    "command_monotonic_time": f"{command_time:.9f}",
                    "scheduled_sample_monotonic_time": f"{scheduled_sample_time:.9f}",
                    "sample_monotonic_time": f"{sample_time:.9f}",
                    "sample_delay_sec": f"{sample_time - scheduled_sample_time:.9f}",
                    "target_x": f"{position[0]:.9f}",
                    "target_y": f"{position[1]:.9f}",
                    "target_z": f"{position[2]:.9f}",
                    "target_qx": f"{quaternion_xyzw[0]:.9f}",
                    "target_qy": f"{quaternion_xyzw[1]:.9f}",
                    "target_qz": f"{quaternion_xyzw[2]:.9f}",
                    "target_qw": f"{quaternion_xyzw[3]:.9f}",
                    "actual_x": f"{actual_position[0]:.9f}",
                    "actual_y": f"{actual_position[1]:.9f}",
                    "actual_z": f"{actual_position[2]:.9f}",
                    "actual_qx": f"{actual_quaternion_xyzw[0]:.9f}",
                    "actual_qy": f"{actual_quaternion_xyzw[1]:.9f}",
                    "actual_qz": f"{actual_quaternion_xyzw[2]:.9f}",
                    "actual_qw": f"{actual_quaternion_xyzw[3]:.9f}",
                    "error_x": f"{position_error[0]:.9f}",
                    "error_y": f"{position_error[1]:.9f}",
                    "error_z": f"{position_error[2]:.9f}",
                    "position_error_m": f"{position_error_norm:.9f}",
                    "orientation_error_deg": f"{orientation_error:.9f}",
                    "segment_progress": f"{progress:.9f}",
                    "segment_lateral_error_m": f"{lateral_error:.9f}",
                    "segment_endpoint_distance_m": f"{endpoint_distance:.9f}",
                }
            )
            csv_file.flush()

    catch_up_start_time = time.monotonic()
    catch_up_step_index = 0
    while True:
        with maybe_lock(robot_lock):
            robot.update_desired_ee_pose(
                position=torch.tensor(target_position_m.astype(np.float32)),
                orientation=torch.tensor(segment_move_waypoints[-1].quaternion_xyzw.astype(np.float32)),
            )
            actual_position, actual_quaternion_xyzw = robot.get_ee_pose()

        sample_time = time.monotonic()
        actual_position = tensor_to_numpy(actual_position)
        actual_quaternion_xyzw = tensor_to_numpy(actual_quaternion_xyzw)
        progress, lateral_error, endpoint_distance = project_progress(start_position_m, target_position_m, actual_position)
        if (
            progress >= progress_threshold
            and endpoint_distance <= endpoint_tolerance_m
            and lateral_error <= lateral_tolerance_m
        ):
            break
        if sample_time - catch_up_start_time >= catch_up_timeout_s:
            break

        position_error = actual_position - target_position_m
        position_error_norm = float(np.linalg.norm(position_error))
        orientation_error = quaternion_angle_error_deg(segment_move_waypoints[-1].quaternion_xyzw, actual_quaternion_xyzw)
        position_errors_m.append(position_error_norm)
        orientation_errors_deg.append(orientation_error)
        step_index += 1
        catch_up_step_index += 1

        if writer is not None:
            writer.writerow(
                {
                    "step_index": step_index,
                    "segment_index": segment_move_waypoints[-1].segment_index,
                    "segment_step_index": segment_move_waypoints[-1].segment_step_count + catch_up_step_index,
                    "segment_step_count": segment_move_waypoints[-1].segment_step_count,
                    "is_hold": 0,
                    "phase": "catch_up",
                    "command_monotonic_time": f"{sample_time:.9f}",
                    "scheduled_sample_monotonic_time": f"{sample_time:.9f}",
                    "sample_monotonic_time": f"{sample_time:.9f}",
                    "sample_delay_sec": "0.000000000",
                    "target_x": f"{target_position_m[0]:.9f}",
                    "target_y": f"{target_position_m[1]:.9f}",
                    "target_z": f"{target_position_m[2]:.9f}",
                    "target_qx": f"{segment_move_waypoints[-1].quaternion_xyzw[0]:.9f}",
                    "target_qy": f"{segment_move_waypoints[-1].quaternion_xyzw[1]:.9f}",
                    "target_qz": f"{segment_move_waypoints[-1].quaternion_xyzw[2]:.9f}",
                    "target_qw": f"{segment_move_waypoints[-1].quaternion_xyzw[3]:.9f}",
                    "actual_x": f"{actual_position[0]:.9f}",
                    "actual_y": f"{actual_position[1]:.9f}",
                    "actual_z": f"{actual_position[2]:.9f}",
                    "actual_qx": f"{actual_quaternion_xyzw[0]:.9f}",
                    "actual_qy": f"{actual_quaternion_xyzw[1]:.9f}",
                    "actual_qz": f"{actual_quaternion_xyzw[2]:.9f}",
                    "actual_qw": f"{actual_quaternion_xyzw[3]:.9f}",
                    "error_x": f"{position_error[0]:.9f}",
                    "error_y": f"{position_error[1]:.9f}",
                    "error_z": f"{position_error[2]:.9f}",
                    "position_error_m": f"{position_error_norm:.9f}",
                    "orientation_error_deg": f"{orientation_error:.9f}",
                    "segment_progress": f"{progress:.9f}",
                    "segment_lateral_error_m": f"{lateral_error:.9f}",
                    "segment_endpoint_distance_m": f"{endpoint_distance:.9f}",
                }
            )
            csv_file.flush()
        time.sleep(DENSE_WAYPOINT_DT_S)

    for hold_waypoint in segment_hold_waypoints:
        command_time = time.monotonic()
        with maybe_lock(robot_lock):
            robot.update_desired_ee_pose(
                position=torch.tensor(hold_waypoint.position_m.astype(np.float32)),
                orientation=torch.tensor(hold_waypoint.quaternion_xyzw.astype(np.float32)),
            )
        time.sleep(DENSE_WAYPOINT_DT_S)
        with maybe_lock(robot_lock):
            actual_position, actual_quaternion_xyzw = robot.get_ee_pose()
        sample_time = time.monotonic()
        actual_position = tensor_to_numpy(actual_position)
        actual_quaternion_xyzw = tensor_to_numpy(actual_quaternion_xyzw)
        position_error = actual_position - hold_waypoint.position_m
        position_error_norm = float(np.linalg.norm(position_error))
        orientation_error = quaternion_angle_error_deg(hold_waypoint.quaternion_xyzw, actual_quaternion_xyzw)
        progress, lateral_error, endpoint_distance = project_progress(start_position_m, target_position_m, actual_position)
        position_errors_m.append(position_error_norm)
        orientation_errors_deg.append(orientation_error)
        step_index += 1
        if writer is not None:
            writer.writerow(
                {
                    "step_index": step_index,
                    "segment_index": hold_waypoint.segment_index,
                    "segment_step_index": hold_waypoint.segment_step_index,
                    "segment_step_count": hold_waypoint.segment_step_count,
                    "is_hold": 1,
                    "phase": hold_waypoint.phase,
                    "command_monotonic_time": f"{command_time:.9f}",
                    "scheduled_sample_monotonic_time": f"{sample_time:.9f}",
                    "sample_monotonic_time": f"{sample_time:.9f}",
                    "sample_delay_sec": "0.000000000",
                    "target_x": f"{hold_waypoint.position_m[0]:.9f}",
                    "target_y": f"{hold_waypoint.position_m[1]:.9f}",
                    "target_z": f"{hold_waypoint.position_m[2]:.9f}",
                    "target_qx": f"{hold_waypoint.quaternion_xyzw[0]:.9f}",
                    "target_qy": f"{hold_waypoint.quaternion_xyzw[1]:.9f}",
                    "target_qz": f"{hold_waypoint.quaternion_xyzw[2]:.9f}",
                    "target_qw": f"{hold_waypoint.quaternion_xyzw[3]:.9f}",
                    "actual_x": f"{actual_position[0]:.9f}",
                    "actual_y": f"{actual_position[1]:.9f}",
                    "actual_z": f"{actual_position[2]:.9f}",
                    "actual_qx": f"{actual_quaternion_xyzw[0]:.9f}",
                    "actual_qy": f"{actual_quaternion_xyzw[1]:.9f}",
                    "actual_qz": f"{actual_quaternion_xyzw[2]:.9f}",
                    "actual_qw": f"{actual_quaternion_xyzw[3]:.9f}",
                    "error_x": f"{position_error[0]:.9f}",
                    "error_y": f"{position_error[1]:.9f}",
                    "error_z": f"{position_error[2]:.9f}",
                    "position_error_m": f"{position_error_norm:.9f}",
                    "orientation_error_deg": f"{orientation_error:.9f}",
                    "segment_progress": f"{progress:.9f}",
                    "segment_lateral_error_m": f"{lateral_error:.9f}",
                    "segment_endpoint_distance_m": f"{endpoint_distance:.9f}",
                }
            )
            csv_file.flush()

    return step_index


def stream_closed_loop_key_waypoints(
    robot: RobotInterface,
    start_position_m: np.ndarray,
    start_quaternion_xyzw: np.ndarray,
    key_waypoints: list[tuple[list[float], list[float]]],
    *,
    robot_lock=None,
    log_label: str = "amos",
) -> None:
    log_path: Path | None = None
    csv_file = None
    writer = None
    position_errors_m: list[float] = []
    orientation_errors_deg: list[float] = []

    if TRAJECTORY_LOG_ENABLED:
        TRAJECTORY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        log_path = TRAJECTORY_LOG_DIR / f"{log_label}_tracking_{timestamp}.csv"
        csv_file = log_path.open("w", newline="", encoding="utf-8")
        fieldnames = [
            "step_index",
            "segment_index",
            "segment_step_index",
            "segment_step_count",
            "is_hold",
            "phase",
            "command_monotonic_time",
            "scheduled_sample_monotonic_time",
            "sample_monotonic_time",
            "sample_delay_sec",
            "target_x",
            "target_y",
            "target_z",
            "target_qx",
            "target_qy",
            "target_qz",
            "target_qw",
            "actual_x",
            "actual_y",
            "actual_z",
            "actual_qx",
            "actual_qy",
            "actual_qz",
            "actual_qw",
            "error_x",
            "error_y",
            "error_z",
            "position_error_m",
            "orientation_error_deg",
            "segment_progress",
            "segment_lateral_error_m",
            "segment_endpoint_distance_m",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

    desired_position = start_position_m.astype(np.float64)
    desired_quaternion_xyzw = start_quaternion_xyzw.astype(np.float64)
    step_index = 0

    try:
        for segment_index, (position_list, rpy_list) in enumerate(key_waypoints, start=1):
            target_position = np.asarray(position_list, dtype=np.float64)
            target_quaternion_xyzw = rpy_deg_to_quat_xyzw(np.asarray(rpy_list, dtype=np.float64)).astype(np.float64)
            dense_segment = build_dense_segment(
                desired_position,
                desired_quaternion_xyzw,
                target_position,
                target_quaternion_xyzw,
                segment_index=segment_index,
            )
            step_index = stream_closed_loop_segment(
                robot,
                dense_segment,
                desired_position,
                target_position,
                robot_lock=robot_lock,
                csv_file=csv_file,
                writer=writer,
                starting_step_index=step_index,
                position_errors_m=position_errors_m,
                orientation_errors_deg=orientation_errors_deg,
            )
            desired_position = target_position
            desired_quaternion_xyzw = target_quaternion_xyzw
    finally:
        if csv_file is not None:
            csv_file.close()
        if log_path is not None:
            write_tracking_summary(log_path, position_errors_m, orientation_errors_deg)


def move_to_pose(
    robot: RobotInterface,
    label: str,
    position_m: np.ndarray,
    rpy_deg: np.ndarray,
    *,
    robot_lock=None,
) -> None:
    quat_xyzw = rpy_deg_to_quat_xyzw(rpy_deg).astype(np.float32)
    current_position, current_quaternion_xyzw = get_ee_pose_numpy(robot, robot_lock=robot_lock)
    stream_pose_segment(
        robot=robot,
        start_position_m=current_position.astype(np.float64),
        start_quaternion_xyzw=current_quaternion_xyzw.astype(np.float64),
        target_position_m=position_m.astype(np.float64),
        target_quaternion_xyzw=quat_xyzw.astype(np.float64),
        duration_s=POSE_MOVE_TIME_S,
        robot_lock=robot_lock,
    )


def main() -> None:
    robot = RobotInterface(ip_address=ROBOT_IP, port=ROBOT_PORT)
    gripper = GripperInterface(ip_address=GRIPPER_IP, port=GRIPPER_PORT)
    robot_lock = RLock()
    gripper_lock = RLock()
    state_service = DemoStateService(
        robot,
        gripper,
        ControllerSettings(
            server=ServerSettings(host=STATE_SERVER_HOST, port=STATE_SERVER_PORT),
            backend=BackendSettings(
                kind="polymetis",
                robot_ip=ROBOT_IP,
                robot_port=ROBOT_PORT,
                gripper_ip=GRIPPER_IP,
                gripper_port=GRIPPER_PORT,
            ),
            control=ControlSettings(
                control_frequency_hz=CONTROL_HZ,
                state_cache_hz=STATE_CACHE_HZ,
                teleop_command_hz=CONTROL_HZ,
            ),
        ),
        robot_lock=robot_lock,
        gripper_lock=gripper_lock,
    )
    server = ManagedUvicornServer(create_demo_state_app(state_service), STATE_SERVER_HOST, STATE_SERVER_PORT)
    try:
        state_service.start()
        server.start()
        print("Demo state server:")
        print(f"  host: {STATE_SERVER_HOST}")
        print(f"  port: {STATE_SERVER_PORT}")
        print()

        print_pose_and_joints(robot, "Current", robot_lock=robot_lock)
        print_gripper_state(gripper, "Current", gripper_lock=gripper_lock)
        start_cartesian_hold(robot, robot_lock=robot_lock)

        initial_position = np.asarray(INITIAL_POSITION_M, dtype=np.float32)
        initial_rpy = np.asarray(INITIAL_RPY_DEG, dtype=np.float64)

        confirm_or_abort("Move robot to the initial pose?")
        move_to_pose(robot, "Initial pose", initial_position, initial_rpy, robot_lock=robot_lock)

        print("Initial gripper command:")
        print("  action: grasp until contact")
        print(f"  speed (m/s): {CLOSE_SPEED_MPS:.4f}")
        print(f"  force (N): {CLOSE_FORCE_N:.2f}")
        print()
        confirm_or_abort("Close gripper at the initial pose?")
        with maybe_lock(gripper_lock):
            gripper.grasp(speed=CLOSE_SPEED_MPS, force=CLOSE_FORCE_N)
        print_gripper_state(gripper, "After initial close", gripper_lock=gripper_lock)

        if not CONFIRM_EACH_WAYPOINT:
            confirm_or_abort("Run inference automatically?")
            initial_quaternion_xyzw = rpy_deg_to_quat_xyzw(initial_rpy).astype(np.float64)
            print(
                f"Streaming closed-loop AMOS segments at {CONTROL_HZ:.1f} Hz "
                f"with pen-up hold={PEN_UP_KEY_WAYPOINT_HOLD_S:.2f} s and "
                f"pen-down hold={PEN_DOWN_KEY_WAYPOINT_HOLD_S:.2f} s. "
                f"Segment timing uses pen-down speed={PEN_DOWN_TRANSLATION_SPEED_MPS:.3f} m/s "
                f"and pen-up speed={PEN_UP_TRANSLATION_SPEED_MPS:.3f} m/s."
            )
            stream_closed_loop_key_waypoints(
                robot,
                initial_position.astype(np.float64),
                initial_quaternion_xyzw,
                KEY_WAYPOINTS,
                robot_lock=robot_lock,
            )
        else:
            desired_position = initial_position.astype(np.float64)
            desired_quaternion_xyzw = rpy_deg_to_quat_xyzw(initial_rpy).astype(np.float64)
            for i, (pos_list, rpy_list) in enumerate(KEY_WAYPOINTS, start=1):
                confirm_or_abort(f"Move robot to key waypoint {i}?")
                target_position = np.asarray(pos_list, dtype=np.float64)
                target_quaternion_xyzw = rpy_deg_to_quat_xyzw(np.asarray(rpy_list, dtype=np.float64)).astype(np.float64)
                dense_segment = build_dense_segment(
                    desired_position,
                    desired_quaternion_xyzw,
                    target_position,
                    target_quaternion_xyzw,
                    segment_index=i,
                )
                stream_closed_loop_segment(
                    robot,
                    dense_segment,
                    desired_position,
                    target_position,
                    robot_lock=robot_lock,
                )
                desired_position = target_position
                desired_quaternion_xyzw = target_quaternion_xyzw

        if HOLD_FINAL_POSE_S > 0.0:
            print(f"Holding final pose for {HOLD_FINAL_POSE_S:.2f} s before releasing policy.")
            time.sleep(HOLD_FINAL_POSE_S)

        print("Done.")
    finally:
        server.stop()
        state_service.stop()
        try:
            with maybe_lock(robot_lock):
                robot.terminate_current_policy()
        except Exception:
            pass


if __name__ == "__main__":
    main()