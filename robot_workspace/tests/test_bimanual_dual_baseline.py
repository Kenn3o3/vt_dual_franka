from __future__ import annotations

import numpy as np

from vt_dual_franka_shared.models import ControllerState, TcpTargetCommand
from vt_dual_franka_workspace.config import ArmEndpointSettings, WorkspaceSettings
from vt_dual_franka_workspace.policies.common.visuotactile.bimanual_runtime import (
    BIMANUAL_ACTION_DIM,
    bimanual_command_to_20d,
    bimanual_states_to_20d,
    decode_bimanual_20d_action,
)


def _state(x: float, arm_id: str) -> ControllerState:
    return ControllerState(
        arm_id=arm_id,
        tcp_pose=[x, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0],
        gripper_width=0.039,
    )


def test_bimanual_20d_round_trip_preserves_arm_order():
    action = bimanual_command_to_20d(
        {
            "left": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            "right": [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
        },
        {"left": 1.0, "right": 0.0},
    )

    assert action.shape == (BIMANUAL_ACTION_DIM,)
    decoded = decode_bimanual_20d_action(action)
    assert decoded.target_tcp["left"][:3] == [0.1, 0.2, 0.3]
    assert decoded.target_tcp["right"][:3] == [0.4, 0.5, 0.6]
    assert decoded.gripper_closedness == {"left": 1.0, "right": 0.0}


def test_bimanual_state_vector_uses_left_then_right():
    vector = bimanual_states_to_20d({"left": _state(0.1, "left"), "right": _state(0.4, "right")})

    assert vector.shape == (20,)
    np.testing.assert_allclose(vector[:3], [0.1, 0.1, 0.2])
    np.testing.assert_allclose(vector[9:12], [0.4, 0.1, 0.2])


def test_tcp_command_carries_identity_and_shared_timing():
    command = TcpTargetCommand(
        arm_id="right",
        command_id="pair-1",
        target_tcp=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        target_monotonic_time=10.0,
    )

    assert command.arm_id == "right"
    assert command.command_id == "pair-1"
    assert command.target_monotonic_time == 10.0


def test_workspace_has_default_dual_arm_endpoints():
    workspace = WorkspaceSettings()

    assert workspace.arms["left"] == ArmEndpointSettings(arm_id="left", host="127.0.0.1", port=8092, request_timeout_sec=0.1)
    assert workspace.arms["right"] == ArmEndpointSettings(arm_id="right", host="127.0.0.1", port=8093, request_timeout_sec=0.1)
