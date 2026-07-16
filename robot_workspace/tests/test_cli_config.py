from __future__ import annotations

import sys

import pytest
from pydantic import ValidationError

from vt_dual_franka_workspace import cli
from vt_dual_franka_workspace.config import (
    ArmEndpointSettings,
    WorkspaceSettings,
    load_inference_config,
    load_policy_config,
    load_task_config,
    load_workspace_config,
)


def test_workspace_has_only_two_explicit_controller_endpoints():
    workspace = load_workspace_config("robot_workspace/config/workspace.yaml")

    assert not hasattr(workspace, "controller")
    assert workspace.arms == {
        "left": ArmEndpointSettings(
            arm_id="left",
            host="10.0.0.1",
            port=8092,
            request_timeout_sec=0.1,
        ),
        "right": ArmEndpointSettings(
            arm_id="right",
            host="10.0.0.1",
            port=8093,
            request_timeout_sec=0.1,
        ),
    }
    assert workspace.operator_ui.preview_camera_role == "left_wrist"


def test_workspace_rejects_missing_mislabeled_or_duplicate_arm_endpoints():
    left = {
        "arm_id": "left",
        "host": "10.0.0.1",
        "port": 8092,
        "request_timeout_sec": 0.1,
    }
    right = {
        "arm_id": "right",
        "host": "10.0.0.1",
        "port": 8093,
        "request_timeout_sec": 0.1,
    }
    with pytest.raises(ValidationError, match="exactly left and right"):
        WorkspaceSettings(arms={"left": left})
    with pytest.raises(ValidationError, match="arm_id must be"):
        WorkspaceSettings(arms={"left": {**left, "arm_id": "right"}, "right": right})
    with pytest.raises(ValidationError, match="distinct"):
        WorkspaceSettings(arms={"left": left, "right": {**right, "port": 8092}})


def test_bimanual_demo_contains_requested_left_and_right_initial_poses():
    task = load_task_config("robot_workspace/config/tasks/bimanual_demo.yaml")

    assert task.task_name == "bimanual_demo"
    assert task.initial_poses["left"].eef_pose_xyz_rpy_deg == [
        0.0,
        0.3,
        0.4,
        -180.0,
        0.0,
        45.0,
    ]
    assert task.initial_poses["right"].eef_pose_xyz_rpy_deg == [
        0.0,
        -0.3,
        0.4,
        -180.0,
        0.0,
        -145.0,
    ]
    assert task.initial_poses["left"].joint_positions_rad == [
        1.3922,
        -0.7209,
        0.1799,
        -2.8098,
        0.1171,
        2.0371,
        0.7,
    ]
    assert task.initial_poses["right"].joint_positions_rad == [
        -1.5857,
        -0.7485,
        0.0049,
        -2.8095,
        0.0293,
        2.0184,
        0.9069,
    ]


def test_only_bimanual_policy_and_inference_configs_remain():
    inference = load_inference_config(
        "robot_workspace/config/inference/bimanual_demo_dp.yaml"
    )
    policy = load_policy_config(
        "robot_workspace/config/policies/dp_bimanual_demo.yaml"
    )

    assert inference.initial_poses is not None
    assert set(inference.initial_poses) == {"left", "right"}
    assert policy.type == "dp_bimanual"
    assert policy.config["model"] == "dp_bimanual"


def test_cli_exposes_only_dual_platform_commands(capsys):
    old_argv = sys.argv
    try:
        sys.argv = ["vt-dual-franka-workspace", "--help"]
        with pytest.raises(SystemExit):
            cli.main()
    finally:
        sys.argv = old_argv
    help_text = capsys.readouterr().out
    for command in ("collect", "teleop", "make-dataset", "train", "run-policy"):
        assert command in help_text
    for removed in (
        "dual-teleop",
        "make-bimanual-dataset",
        "gripper-testbed",
        "prepare-visuotactile",
        "remote-train-visuotactile",
    ):
        assert removed not in help_text


def test_merge_task_into_inference_copies_paired_pose_and_sensors():
    task = load_task_config("robot_workspace/config/tasks/bimanual_demo.yaml")
    inference = load_inference_config(
        "robot_workspace/config/inference/bimanual_demo_dp.yaml"
    )

    merged = cli._merge_task_into_inference(inference, task)

    assert merged.initial_poses == task.initial_poses
    assert merged.rgb_cameras == task.rgb_cameras
    assert merged.gelsights == task.gelsights
