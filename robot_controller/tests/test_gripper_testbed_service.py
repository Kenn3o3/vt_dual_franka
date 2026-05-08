import time

from fastapi.testclient import TestClient

from vt_franka_controller.api.gripper_testbed_app import create_gripper_testbed_app
from vt_franka_controller.backends.mock import MockFrankaBackend
from vt_franka_controller.control.gripper_service import GripperTestbedService
from vt_franka_controller.settings import BackendSettings, ControlSettings, ControllerSettings, ServerSettings


def _settings():
    return ControllerSettings(
        server=ServerSettings(host="127.0.0.1", port=18093),
        backend=BackendSettings(kind="mock"),
        control=ControlSettings(control_frequency_hz=50.0),
    )


def test_gripper_testbed_api_accepts_latest_width_target():
    service = GripperTestbedService(_settings(), MockFrankaBackend())
    app = create_gripper_testbed_app(service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/gripper/target",
            json={"target_width": 0.03, "velocity": 0.1, "force_limit": 5.0, "trigger_depth": 0.6},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "queued"

        deadline = time.time() + 1.0
        state = None
        while time.time() < deadline:
            state = client.get("/api/v1/state").json()
            if state["gripper_width"] == 0.03:
                break
            time.sleep(0.01)
        assert state is not None
        assert state["gripper_width"] == 0.03


def test_gripper_testbed_replaces_pending_commands():
    class SlowMockBackend(MockFrankaBackend):
        def move_gripper(self, width, velocity, force_limit):
            time.sleep(0.05)
            super().move_gripper(width, velocity, force_limit)

    service = GripperTestbedService(_settings(), SlowMockBackend())
    app = create_gripper_testbed_app(service)

    with TestClient(app) as client:
        for width in [0.07, 0.06, 0.05, 0.04]:
            response = client.post(
                "/api/v1/gripper/target",
                json={"target_width": width, "velocity": 0.1, "force_limit": 5.0},
            )
            assert response.status_code == 200

        deadline = time.time() + 1.0
        state = None
        while time.time() < deadline:
            state = client.get("/api/v1/state").json()
            status = client.get("/api/v1/gripper/status").json()
            if state["gripper_width"] == 0.04 and not status["in_flight"]:
                break
            time.sleep(0.01)

        assert state is not None
        assert state["gripper_width"] == 0.04
        status = client.get("/api/v1/gripper/status").json()
        assert status["replaced_command_count"] >= 1

