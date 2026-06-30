from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from vt_franka_shared.models import ResetCommand
from vt_franka_shared.pose_math import pose7d_to_pose6d


class TcpMotionController(Protocol):
    def queue_tcp(self, target_tcp: list[float], source: str = "workspace", target_duration_sec: float | None = None) -> None:
        ...


class JointResetController(Protocol):
    def reset(self, command: ResetCommand) -> dict:
        ...


class StateProvider(Protocol):
    def get_state(self, max_age_sec: float | None = None):
        ...


@dataclass(frozen=True)
class RandomizedInitialPose:
    base_pose_xyz_rpy_deg: list[float]
    pose_xyz_rpy_deg: list[float]
    delta_xyz_m: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def metadata(self) -> dict[str, list[float]]:
        return {
            "base_pose_xyz_rpy_deg": list(self.base_pose_xyz_rpy_deg),
            "pose_xyz_rpy_deg": list(self.pose_xyz_rpy_deg),
            "delta_xyz_m": list(self.delta_xyz_m),
        }


def sample_randomized_initial_pose(
    base_pose_xyz_rpy_deg: list[float],
    rand_xyz_range_m: list[float] | None,
) -> RandomizedInitialPose:
    if len(base_pose_xyz_rpy_deg) != 6:
        raise ValueError("base initial EEF pose must be [x, y, z, roll_deg, pitch_deg, yaw_deg]")
    ranges = [0.0, 0.0, 0.0] if rand_xyz_range_m is None else list(rand_xyz_range_m)
    if len(ranges) != 3:
        raise ValueError("rand_init_pose must contain exactly 3 xyz range values")
    if any(value < 0.0 for value in ranges):
        raise ValueError("rand_init_pose values must be non-negative")
    delta = np.random.uniform(-np.asarray(ranges, dtype=np.float64), np.asarray(ranges, dtype=np.float64))
    pose = [float(value) for value in base_pose_xyz_rpy_deg]
    for index in range(3):
        pose[index] += float(delta[index])
    return RandomizedInitialPose(
        base_pose_xyz_rpy_deg=[float(value) for value in base_pose_xyz_rpy_deg],
        pose_xyz_rpy_deg=pose,
        delta_xyz_m=[float(value) for value in delta],
    )


def eef_xyz_rpy_deg_to_tcp_pose(pose_xyz_rpy_deg: list[float]) -> list[float]:
    if len(pose_xyz_rpy_deg) != 6:
        raise ValueError("EEF pose must be [x, y, z, roll_deg, pitch_deg, yaw_deg]")
    quat_xyzw = Rotation.from_euler("xyz", pose_xyz_rpy_deg[3:], degrees=True).as_quat()
    return [
        float(pose_xyz_rpy_deg[0]),
        float(pose_xyz_rpy_deg[1]),
        float(pose_xyz_rpy_deg[2]),
        float(quat_xyzw[3]),
        float(quat_xyzw[0]),
        float(quat_xyzw[1]),
        float(quat_xyzw[2]),
    ]


def move_to_eef_pose(
    *,
    controller: TcpMotionController,
    state_provider: StateProvider,
    pose_xyz_rpy_deg: list[float],
    duration_sec: float,
    source: str,
    position_tolerance_m: float,
    rotation_tolerance_deg: float,
    settle_timeout_sec: float,
    settle_dwell_sec: float,
    state_max_age_sec: float = 2.0,
) -> list[float]:
    target_tcp = eef_xyz_rpy_deg_to_tcp_pose(pose_xyz_rpy_deg)
    controller.queue_tcp(target_tcp, source=source, target_duration_sec=max(float(duration_sec), 1e-3))
    _wait_for_tcp_pose(
        state_provider=state_provider,
        target_tcp=target_tcp,
        position_tolerance_m=position_tolerance_m,
        rotation_tolerance_deg=rotation_tolerance_deg,
        timeout_sec=settle_timeout_sec + max(float(duration_sec), 0.0),
        dwell_sec=settle_dwell_sec,
        state_max_age_sec=state_max_age_sec,
    )
    return target_tcp


def move_to_home_joints(
    *,
    controller: JointResetController,
    state_provider: StateProvider,
    joint_positions: list[float],
    duration_sec: float,
    source: str,
    tolerance_rad: float,
    settle_timeout_sec: float,
    state_max_age_sec: float = 2.0,
) -> dict:
    target_joints = [float(value) for value in joint_positions]
    if len(target_joints) != 7:
        raise ValueError("home joint reset target must contain exactly 7 values")
    command = ResetCommand(
        profile="task_home_joint",
        joint_positions=target_joints,
        joint_duration_sec=max(float(duration_sec), 1e-3),
        gripper_target="unchanged",
        source=source,
    )
    result = controller.reset(command)
    _wait_for_joint_positions(
        state_provider=state_provider,
        target_joints=target_joints,
        tolerance_rad=tolerance_rad,
        timeout_sec=settle_timeout_sec + max(float(duration_sec), 0.0),
        state_max_age_sec=state_max_age_sec,
    )
    return result


def _wait_for_joint_positions(
    *,
    state_provider: StateProvider,
    target_joints: list[float],
    tolerance_rad: float,
    timeout_sec: float,
    state_max_age_sec: float,
) -> None:
    target = np.asarray(target_joints, dtype=np.float64)
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    last_error: float | None = None
    while time.monotonic() <= deadline:
        state = state_provider.get_state(max_age_sec=state_max_age_sec)
        current = np.asarray(state.joint_positions, dtype=np.float64)
        if current.shape == target.shape:
            last_error = float(np.max(np.abs(current - target)))
            if last_error <= tolerance_rad:
                return
        time.sleep(0.05)
    raise RuntimeError(f"Home joint reset did not settle within timeout; max_error_rad={last_error}")


def _wait_for_tcp_pose(
    *,
    state_provider: StateProvider,
    target_tcp: list[float],
    position_tolerance_m: float,
    rotation_tolerance_deg: float,
    timeout_sec: float,
    dwell_sec: float,
    state_max_age_sec: float,
) -> None:
    target_pose6d = pose7d_to_pose6d(target_tcp)
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    dwell_start: float | None = None
    while time.monotonic() <= deadline:
        state = state_provider.get_state(max_age_sec=state_max_age_sec)
        current_pose6d = pose7d_to_pose6d(state.tcp_pose)
        position_error, rotation_error = _pose_distance(current_pose6d, target_pose6d)
        if position_error <= position_tolerance_m and np.degrees(rotation_error) <= rotation_tolerance_deg:
            if dwell_start is None:
                dwell_start = time.monotonic()
            elif time.monotonic() - dwell_start >= dwell_sec:
                return
        else:
            dwell_start = None
        time.sleep(0.05)
    raise RuntimeError("Initial EEF pose did not settle within timeout")


def _pose_distance(left_pose6d: np.ndarray, right_pose6d: np.ndarray) -> tuple[float, float]:
    position_error = float(np.linalg.norm(np.asarray(left_pose6d[:3]) - np.asarray(right_pose6d[:3])))
    left_rot = Rotation.from_rotvec(left_pose6d[3:])
    right_rot = Rotation.from_rotvec(right_pose6d[3:])
    rotation_error = float((right_rot.inv() * left_rot).magnitude())
    return position_error, rotation_error
