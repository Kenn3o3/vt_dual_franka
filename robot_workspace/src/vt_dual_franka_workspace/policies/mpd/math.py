from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

from vt_dual_franka_shared.pose_math import wxyz_to_xyzw, xyzw_to_wxyz


def pose7d_to_rot6d(pose7d: Iterable[float]) -> np.ndarray:
    pose = np.asarray(list(pose7d), dtype=np.float64)
    if pose.shape != (7,):
        raise ValueError("pose7d must have shape (7,)")
    rotation = Rotation.from_quat(wxyz_to_xyzw(pose[3:])).as_matrix()
    rot6d = rotation[:, :2].T.reshape(6)
    return np.concatenate([pose[:3], rot6d]).astype(np.float64)


def rot6d_to_pose7d(values: Iterable[float]) -> np.ndarray:
    state = np.asarray(list(values), dtype=np.float64)
    if state.shape != (9,):
        raise ValueError("rot6d pose state must have shape (9,)")
    matrix = rot6d_to_matrix(state[3:])
    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.concatenate([state[:3], xyzw_to_wxyz(quat_xyzw)]).astype(np.float64)


def rot6d_to_matrix(rot6d: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(rot6d), dtype=np.float64)
    if values.shape != (6,):
        raise ValueError("rot6d must have shape (6,)")
    x_raw = values[:3]
    y_raw = values[3:]
    x = _normalize(x_raw)
    z = _normalize(np.cross(x, y_raw))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def pose7d_and_gripper_to_tcp_state(pose7d: Iterable[float], gripper_closedness: float) -> np.ndarray:
    pose_state = pose7d_to_rot6d(pose7d)
    closedness = float(np.clip(gripper_closedness, 0.0, 1.0))
    return np.concatenate([pose_state, np.array([closedness], dtype=np.float64)]).astype(np.float64)


def tcp_state_to_pose7d_and_gripper(state: Iterable[float]) -> tuple[np.ndarray, float]:
    values = np.asarray(list(state), dtype=np.float64)
    if values.shape != (10,):
        raise ValueError("tcp state must have shape (10,)")
    pose7d = rot6d_to_pose7d(values[:9])
    closedness = float(np.clip(values[9], 0.0, 1.0))
    return pose7d, closedness


def finite_difference(values: np.ndarray, dt: float, *, first_velocity_zero: bool = False) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    velocity = np.zeros_like(array, dtype=np.float64)
    if len(array) > 1:
        velocity[1:] = (array[1:] - array[:-1]) / float(dt)
        if not first_velocity_zero:
            velocity[0] = velocity[1]
    return velocity


def gripper_width_to_closedness(width_m: float, *, open_width_m: float = 0.078) -> float:
    return float(np.clip(1.0 - float(width_m) / max(float(open_width_m), 1e-8), 0.0, 1.0))


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero rotation vector")
    return vector / norm
