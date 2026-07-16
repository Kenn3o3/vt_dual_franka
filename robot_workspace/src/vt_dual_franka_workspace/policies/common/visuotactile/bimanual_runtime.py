from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from vt_dual_franka_shared.models import ArmId, ControllerState

from ...mpd.math import gripper_width_to_closedness, pose7d_and_gripper_to_tcp_state, tcp_state_to_pose7d_and_gripper

ARM_ORDER: tuple[ArmId, ArmId] = ("left", "right")
BIMANUAL_ACTION_DIM = 20
BIMANUAL_LAYOUT = (
    "left_xyz",
    "left_rot6d",
    "right_xyz",
    "right_rot6d",
    "left_closedness",
    "right_closedness",
)


@dataclass(frozen=True)
class BimanualDecodedAction:
    target_tcp: dict[ArmId, list[float]]
    gripper_closedness: dict[ArmId, float]
    metadata: dict[str, Any]


def bimanual_states_to_20d(
    states: dict[ArmId, ControllerState],
    *,
    gripper_open_width_m: float = 0.078,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    closedness: list[float] = []
    for arm_id in ARM_ORDER:
        state = states[arm_id]
        value = pose7d_and_gripper_to_tcp_state(
            state.tcp_pose,
            gripper_width_to_closedness(float(state.gripper_width), open_width_m=gripper_open_width_m),
        )
        rows.append(value[:9])
        closedness.append(float(value[9]))
    return np.concatenate([rows[0], rows[1], np.asarray(closedness, dtype=np.float64)]).astype(np.float64)


def bimanual_command_to_20d(
    target_tcp: dict[ArmId, list[float]],
    gripper_closedness: dict[ArmId, float],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    closedness: list[float] = []
    for arm_id in ARM_ORDER:
        value = pose7d_and_gripper_to_tcp_state(target_tcp[arm_id], float(gripper_closedness[arm_id]))
        rows.append(value[:9])
        closedness.append(float(value[9]))
    return np.concatenate([rows[0], rows[1], np.asarray(closedness, dtype=np.float64)]).astype(np.float64)


def decode_bimanual_20d_action(row: np.ndarray) -> BimanualDecodedAction:
    values = np.asarray(row, dtype=np.float64)
    if values.shape != (BIMANUAL_ACTION_DIM,):
        raise ValueError(f"Expected 20D bimanual action, got {values.shape}")
    left_pose, _ = tcp_state_to_pose7d_and_gripper(np.concatenate([values[:9], values[18:19]]))
    right_pose, _ = tcp_state_to_pose7d_and_gripper(np.concatenate([values[9:18], values[19:20]]))
    gripper = {"left": float(np.clip(values[18], 0.0, 1.0)), "right": float(np.clip(values[19], 0.0, 1.0))}
    return BimanualDecodedAction(
        target_tcp={"left": left_pose.astype(float).tolist(), "right": right_pose.astype(float).tolist()},
        gripper_closedness=gripper,
        metadata={
            "action_dim": BIMANUAL_ACTION_DIM,
            "arm_order": list(ARM_ORDER),
            "layout": list(BIMANUAL_LAYOUT),
            "rotation": "pytorch3d_first_two_rows",
            "frame": "per_arm_franka_base",
        },
    )


def bimanual_model_manifest() -> dict[str, Any]:
    return {
        "schema_version": "vt_dual_franka_bimanual_policy_v1",
        "action_dim": BIMANUAL_ACTION_DIM,
        "qpos_dim": BIMANUAL_ACTION_DIM,
        "arm_order": list(ARM_ORDER),
        "layout": list(BIMANUAL_LAYOUT),
        "obs": {
            "rgb_wrist_left": {"type": "rgb"},
            "rgb_wrist_right": {"type": "rgb"},
            "tactile_left": {"type": "tactile_rgb"},
            "tactile_right": {"type": "tactile_rgb"},
            "qpos": {"shape": [BIMANUAL_ACTION_DIM], "type": "low_dim"},
        },
        "action": {"shape": [BIMANUAL_ACTION_DIM], "type": "low_dim"},
    }
