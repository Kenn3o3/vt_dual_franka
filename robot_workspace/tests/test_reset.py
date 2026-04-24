from __future__ import annotations

import sys

import pytest
import requests

from vt_franka_workspace import cli
from vt_franka_workspace.controller.client import ControllerClient, ControllerClientError
from vt_franka_workspace.reset import build_reset_command
from vt_franka_shared.config import load_yaml_model
from vt_franka_workspace.settings import WorkspaceSettings


def test_build_reset_command_ready_profile_is_eef_only():
    settings = WorkspaceSettings(
        teleop={"max_gripper_width": 0.078, "gripper_velocity": 0.1, "grasp_force": 7.0},
        reset={
            "default_profile": "ready",
            "profiles": {
                "ready": {
                    "eef_pose_xyz_rpy_deg": [0.0, 0.45, 0.5, 180.0, 0.0, 45.0],
                    "eef_duration_sec": 2.0,
                    "gripper_target": "open",
                }
            },
        },
    )

    command = build_reset_command(settings, source="test")

    assert command.profile == "ready"
    assert command.joint_positions is None
    assert command.eef_pose_xyz_rpy_deg == [0.0, 0.45, 0.5, 180.0, 0.0, 45.0]
    assert command.gripper_target == "open"
    assert command.gripper_width == 0.078


def test_controller_client_post_error_includes_api_detail(monkeypatch):
    client = ControllerClient(host="10.0.0.1", port=8092)

    class FakeResponse:
        status_code = 400

        @staticmethod
        def raise_for_status():
            error = requests.HTTPError("400 Client Error")
            error.response = FakeResponse()
            raise error

        @staticmethod
        def json():
            return {"detail": "Reset target did not settle within timeout"}

    class FakeSession:
        @staticmethod
        def post(*args, **kwargs):
            del args, kwargs
            return FakeResponse()

        @staticmethod
        def close():
            return None

    client._local.session = FakeSession()

    with pytest.raises(ControllerClientError) as exc_info:
        client._post_json("/api/v1/actions/reset", {})

    assert "Reset target did not settle within timeout" in str(exc_info.value)


def test_cli_reset_uses_workspace_profile(monkeypatch, capsys):
    calls: dict[str, object] = {}

    class FakeController:
        def __init__(self, host: str, port: int, request_timeout_sec: float = 1.0) -> None:
            calls["controller_init"] = {
                "host": host,
                "port": port,
                "request_timeout_sec": request_timeout_sec,
            }

        def reset(self, command):
            calls["reset_command"] = command
            return {
                "status": "ok",
                "profile": command.profile,
                "path": "slow",
                "gripper_target": command.gripper_target,
            }

    monkeypatch.setattr(cli, "ControllerClient", FakeController)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vt-franka-workspace",
            "reset",
            "--config",
            "/home/zhenya/kenny/visuotact/vt_franka/robot_workspace/config/workspace.yaml",
        ],
    )

    cli.main()

    command = calls["reset_command"]
    assert calls["controller_init"] == {
        "host": "10.0.0.1",
        "port": 8092,
        "request_timeout_sec": 0.1,
    }
    assert command.profile == "ready"
    assert command.eef_pose_xyz_rpy_deg == [0.0, 0.45, 0.5, 180.0, 0.0, 45.0]
    assert command.gripper_target == "open"
    assert "Workspace reset completed" in capsys.readouterr().out


def test_cli_reset_controller_ready_calls_ready(monkeypatch, capsys):
    calls: dict[str, int] = {"ready": 0}

    class FakeController:
        def __init__(self, host: str, port: int, request_timeout_sec: float = 1.0) -> None:
            del host, port, request_timeout_sec

        def ready(self) -> None:
            calls["ready"] += 1

    monkeypatch.setattr(cli, "ControllerClient", FakeController)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "vt-franka-workspace",
            "reset",
            "--config",
            "/home/zhenya/kenny/visuotact/vt_franka/robot_workspace/config/workspace.yaml",
            "--controller-ready",
        ],
    )

    cli.main()

    assert calls["ready"] == 1
    assert "Controller ready action completed." in capsys.readouterr().out


def test_workspace_config_joint_ready_profile_matches_controller_joint_reset():
    settings = load_yaml_model(
        "/home/zhenya/kenny/visuotact/vt_franka/robot_workspace/config/workspace.yaml",
        WorkspaceSettings,
    )

    command = build_reset_command(settings, source="test", profile_name="joint_ready")

    assert command.joint_positions == [1.7026, -0.0901, -0.1763, -1.8991, -0.0258, 1.791, 0.7442]
    assert command.joint_duration_sec == 4.0
    assert command.eef_pose_xyz_rpy_deg is None
    assert command.gripper_target == "unchanged"
