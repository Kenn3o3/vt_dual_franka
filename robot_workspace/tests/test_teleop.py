from __future__ import annotations

import time

import numpy as np

from vt_dual_franka_shared.models import (
    ControllerState,
    DualArmControllerState,
    QuestHandState,
    UnityTeleopMessage,
)
from vt_dual_franka_workspace.config import TeleopSettings
from vt_dual_franka_workspace.teleop import DualQuestTeleopService


class IdentityArmCalibration:
    def unity_to_robot_pose(self, pose):
        return np.asarray(pose, dtype=np.float64)


class FakeBimanualCalibration:
    def arm(self, arm_id):
        del arm_id
        return IdentityArmCalibration()


class FakeController:
    def __init__(self):
        self.grasps = []
        self.moves = []
        self.stops = 0

    def grasp_gripper(self, velocity, force_limit, source="workspace", blocking=False):
        self.grasps.append((velocity, force_limit, source, blocking))

    def move_gripper(
        self,
        width,
        velocity,
        force_limit,
        source="workspace",
        blocking=False,
    ):
        self.moves.append((width, velocity, force_limit, source, blocking))

    def stop_gripper(self):
        self.stops += 1


class Context:
    def __init__(self):
        self.controller = FakeController()


class FakeCoordinator:
    def __init__(self):
        self.arms = {"left": Context(), "right": Context()}
        self.commands = []
        self.state = DualArmControllerState(
            left=ControllerState(
                arm_id="left",
                tcp_pose=[0.0, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0],
            ),
            right=ControllerState(
                arm_id="right",
                tcp_pose=[0.0, -0.3, 0.4, 1.0, 0.0, 0.0, 0.0],
            ),
        )

    def get_state(self, max_age_sec=None):
        del max_age_sec
        return self.state

    def queue_tcp_pair(self, targets, **kwargs):
        self.commands.append((targets, kwargs))
        return f"pair-{len(self.commands)}"


def _hand(x: float, tracking: bool, trigger: float = 0.0) -> QuestHandState:
    return QuestHandState(
        wristPos=[x, 0.0, 0.0],
        wristQuat=[1.0, 0.0, 0.0, 0.0],
        triggerState=trigger,
        buttonState=[False, False, False, False, tracking],
    )


def _message(
    left_x: float,
    right_x: float,
    *,
    left_tracking: bool,
    right_tracking: bool,
) -> UnityTeleopMessage:
    return UnityTeleopMessage(
        timestamp=time.time(),
        leftHand=_hand(left_x, left_tracking),
        rightHand=_hand(right_x, right_tracking),
    )


def test_left_quest_hand_controls_left_arm_and_holds_right_arm():
    coordinator = FakeCoordinator()
    service = DualQuestTeleopService(
        TeleopSettings(
            loop_hz=200.0,
            operator_yaw_offset_deg=0.0,
            relative_translation_scale=1.0,
        ),
        coordinator,
        FakeBimanualCalibration(),
    )
    service.set_teleop_enabled(True)
    service.start()
    try:
        service.submit_message(
            _message(0.0, 0.0, left_tracking=True, right_tracking=False)
        )
        time.sleep(0.03)
        service.submit_message(
            _message(0.1, 0.0, left_tracking=True, right_tracking=False)
        )
        time.sleep(0.05)
    finally:
        service.stop()

    targets = coordinator.commands[-1][0]
    assert set(targets) == {"left", "right"}
    assert targets["left"] != coordinator.state.left.tcp_pose
    assert targets["right"] == coordinator.state.right.tcp_pose


def test_right_quest_hand_controls_right_arm_and_holds_left_arm():
    coordinator = FakeCoordinator()
    service = DualQuestTeleopService(
        TeleopSettings(
            loop_hz=200.0,
            operator_yaw_offset_deg=0.0,
            relative_translation_scale=1.0,
        ),
        coordinator,
        FakeBimanualCalibration(),
    )
    service.set_teleop_enabled(True)
    service.start()
    try:
        service.submit_message(
            _message(0.0, 0.0, left_tracking=False, right_tracking=True)
        )
        time.sleep(0.03)
        service.submit_message(
            _message(0.0, 0.1, left_tracking=False, right_tracking=True)
        )
        time.sleep(0.05)
    finally:
        service.stop()

    targets = coordinator.commands[-1][0]
    assert targets["left"] == coordinator.state.left.tcp_pose
    assert targets["right"] != coordinator.state.right.tcp_pose


def test_both_tracking_hands_produce_one_paired_command():
    coordinator = FakeCoordinator()
    service = DualQuestTeleopService(
        TeleopSettings(loop_hz=200.0, operator_yaw_offset_deg=0.0),
        coordinator,
        FakeBimanualCalibration(),
    )
    service.set_teleop_enabled(True)
    service.start()
    try:
        service.submit_message(
            _message(0.0, 0.0, left_tracking=True, right_tracking=True)
        )
        time.sleep(0.03)
        service.submit_message(
            _message(0.1, -0.1, left_tracking=True, right_tracking=True)
        )
        time.sleep(0.05)
    finally:
        service.stop()

    assert coordinator.commands
    assert set(coordinator.commands[-1][0]) == {"left", "right"}
