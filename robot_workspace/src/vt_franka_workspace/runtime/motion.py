from __future__ import annotations

import time
from typing import Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from vt_franka_shared.pose_math import pose7d_to_pose6d


class TcpMotionController(Protocol):
    def queue_tcp(self, target_tcp: list[float], source: str = "workspace", target_duration_sec: float | None = None) -> None:
        ...


class StateProvider(Protocol):
    def get_state(self, max_age_sec: float | None = None):
        ...


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
