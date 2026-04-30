from __future__ import annotations

import pytest
import requests

from vt_franka_workspace.controller.client import ControllerClient, ControllerClientError
from vt_franka_workspace.runtime import eef_xyz_rpy_deg_to_tcp_pose


def test_eef_xyz_rpy_deg_to_tcp_pose_returns_wxyz_quaternion():
    pose = eef_xyz_rpy_deg_to_tcp_pose([0.1, 0.2, 0.3, 0.0, 0.0, 0.0])

    assert pose == [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]


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
            return {"detail": "target did not settle"}

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
        client._post_json("/api/v1/commands/tcp", {})

    assert "target did not settle" in str(exc_info.value)
