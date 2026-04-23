import pytest

from vt_franka_shared.models import ResetCommand, parse_unity_teleop_message


def test_parse_unity_teleop_message_accepts_nested_payload():
    message = parse_unity_teleop_message(
        {
            "timestamp": 1.0,
            "leftHand": {
                "wristPos": [0.1, 0.2, 0.3],
                "wristQuat": [1.0, 0.0, 0.0, 0.0],
                "triggerState": 0.4,
                "buttonState": [False, False, False, False, True],
            },
            "rightHand": {
                "wristPos": [0.0, 0.0, 0.0],
                "wristQuat": [1.0, 0.0, 0.0, 0.0],
                "triggerState": 0.0,
                "buttonState": [False, False, False, False, False],
            },
        }
    )

    assert message.leftHand.wristPos == [0.1, 0.2, 0.3]
    assert message.leftHand.buttonState[4] is True


def test_parse_unity_teleop_message_accepts_flat_payload():
    message = parse_unity_teleop_message(
        {
            "timestamp": 2.0,
            "leftHandPose": [0.4, 0.5, 0.6, 1.0, 0.0, 0.0, 0.0],
            "leftGripperState": 0.9,
            "buttonStates": {"button_0": True, "button_4": True},
        }
    )

    assert message.leftHand.wristPos == [0.4, 0.5, 0.6]
    assert message.leftHand.triggerState == 0.9
    assert message.leftHand.buttonState == [True, False, False, False, True]
    assert message.rightHand.wristQuat == [1.0, 0.0, 0.0, 0.0]


def test_reset_command_accepts_valid_payload():
    command = ResetCommand(
        profile="ready",
        joint_positions=[0.0] * 7,
        joint_duration_sec=1.0,
        eef_pose_xyz_rpy_deg=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        eef_duration_sec=2.0,
        gripper_target="open",
        gripper_width=0.078,
        gripper_velocity=0.1,
        gripper_force_limit=7.0,
        source="test",
    )

    assert command.profile == "ready"
    assert command.gripper_target == "open"


def test_reset_command_rejects_invalid_lengths_and_negative_values():
    with pytest.raises(ValueError):
        ResetCommand(joint_positions=[0.0] * 6)
    with pytest.raises(ValueError):
        ResetCommand(eef_pose_xyz_rpy_deg=[0.0] * 5)
    with pytest.raises(ValueError):
        ResetCommand(joint_duration_sec=-1.0)
