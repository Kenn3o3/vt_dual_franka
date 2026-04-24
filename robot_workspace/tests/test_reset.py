from __future__ import annotations

import pytest
import requests

from vt_franka_workspace.controller.client import ControllerClient, ControllerClientError
from vt_franka_workspace.reset import build_reset_command
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
