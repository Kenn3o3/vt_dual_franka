from fastapi.testclient import TestClient

from vt_franka_controller.api.app import create_app
from vt_franka_controller.api.demo_state_app import create_demo_state_app
from vt_franka_controller.backends.mock import MockFrankaBackend
from vt_franka_controller.control.service import ControllerService
from vt_franka_controller.settings import BackendSettings, ControlSettings, ControllerSettings, ServerSettings
from vt_franka_shared.models import ControllerState, HealthStatus


def test_mock_controller_api_accepts_waypoint_and_reports_state():
    settings = ControllerSettings(
        server=ServerSettings(host="127.0.0.1", port=18092),
        backend=BackendSettings(kind="mock"),
        control=ControlSettings(
            control_frequency_hz=50.0,
            teleop_command_hz=10.0,
            ready_ee_pose=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        ),
    )
    service = ControllerService(settings, MockFrankaBackend())
    app = create_app(service)

    with TestClient(app) as client:
        health = client.get("/api/v1/health")
        assert health.status_code == 200

        response = client.post(
            "/api/v1/commands/tcp",
            json={"target_tcp": [0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0], "source": "test"},
        )
        assert response.status_code == 200

        state = client.get("/api/v1/state")
        assert state.status_code == 200
        assert len(state.json()["tcp_pose"]) == 7

        ready = client.post("/api/v1/actions/ready")
        assert ready.status_code == 200


def test_mock_controller_api_accepts_reset_command():
    settings = ControllerSettings(
        server=ServerSettings(host="127.0.0.1", port=18092),
        backend=BackendSettings(kind="mock"),
        control=ControlSettings(
            control_frequency_hz=50.0,
            teleop_command_hz=10.0,
            ready_ee_pose=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        ),
    )
    service = ControllerService(settings, MockFrankaBackend())
    app = create_app(service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/actions/reset",
            json={
                "profile": "ready",
                "joint_positions": [0.0] * 7,
                "joint_duration_sec": 0.1,
                "eef_pose_xyz_rpy_deg": [0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
                "eef_duration_sec": 0.1,
                "gripper_target": "open",
                "gripper_width": 0.078,
                "gripper_velocity": 0.1,
                "gripper_force_limit": 7.0,
                "source": "test",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["profile"] == "ready"
        assert payload["gripper_target"] == "open"
        assert payload["path"] == "slow"


def test_mock_controller_api_accepts_explicit_waypoint_duration():
    settings = ControllerSettings(
        server=ServerSettings(host="127.0.0.1", port=18092),
        backend=BackendSettings(kind="mock"),
        control=ControlSettings(control_frequency_hz=50.0, teleop_command_hz=60.0),
    )
    service = ControllerService(settings, MockFrankaBackend())
    app = create_app(service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/commands/tcp",
            json={
                "target_tcp": [0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0],
                "target_duration_sec": 0.1,
                "source": "test",
            },
        )
        assert response.status_code == 200


def test_mock_controller_api_can_block_on_gripper_command():
    settings = ControllerSettings(
        server=ServerSettings(host="127.0.0.1", port=18092),
        backend=BackendSettings(kind="mock"),
        control=ControlSettings(control_frequency_hz=50.0, teleop_command_hz=60.0),
    )
    service = ControllerService(settings, MockFrankaBackend())
    app = create_app(service)

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/commands/gripper/width",
            json={
                "width": 0.078,
                "velocity": 0.1,
                "force_limit": 7.0,
                "blocking": True,
                "source": "test",
            },
        )

        assert response.status_code == 200
        assert response.json()["status"] == "completed"
        state = client.get("/api/v1/state")
        assert state.json()["gripper_width"] == 0.078


def test_mock_controller_api_rejects_streaming_commands_during_reset():
    class BlockingResetBackend(MockFrankaBackend):
        def go_home(self, ee_pose, duration_sec):
            super().go_home(ee_pose, duration_sec)
            import time

            time.sleep(0.2)

    settings = ControllerSettings(
        server=ServerSettings(host="127.0.0.1", port=18092),
        backend=BackendSettings(kind="mock"),
        control=ControlSettings(control_frequency_hz=50.0, teleop_command_hz=60.0),
    )
    service = ControllerService(settings, BlockingResetBackend())
    app = create_app(service)

    with TestClient(app) as client:
        import threading

        result = {}

        def run_reset():
            result["reset"] = client.post(
                "/api/v1/actions/reset",
                json={
                    "profile": "ready",
                    "eef_pose_xyz_rpy_deg": [0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
                    "eef_duration_sec": 0.1,
                    "gripper_target": "unchanged",
                    "source": "test",
                },
            )

        thread = threading.Thread(target=run_reset)
        thread.start()
        import time

        time.sleep(0.05)
        response = client.post(
            "/api/v1/commands/tcp",
            json={"target_tcp": [0.4, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0], "source": "test"},
        )
        thread.join(timeout=1.0)

        assert response.status_code == 409
        assert result["reset"].status_code == 200


def test_demo_state_app_reports_state_and_legacy_payload():
    class FakeService:
        def get_health(self):
            return HealthStatus(ok=True, backend="polymetis-demo", message="running")

        def get_state(self):
            return ControllerState(
                tcp_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                tcp_velocity=[0.0] * 6,
                tcp_wrench=[0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                joint_positions=[0.0] * 7,
                joint_velocities=[0.0] * 7,
                gripper_width=0.01,
                gripper_force=5.0,
                control_frequency_hz=60.0,
                backend="polymetis-demo",
            )

    app = create_demo_state_app(FakeService())
    with TestClient(app) as client:
        state = client.get("/api/v1/state")
        assert state.status_code == 200
        assert state.json()["backend"] == "polymetis-demo"

        legacy = client.get("/get_current_robot_states")
        assert legacy.status_code == 200
        payload = legacy.json()
        assert payload["leftRobotTCP"] == [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
        assert payload["leftRobotTCPWrench"] == [0.5, 0.0, 0.0, 0.0, 0.0, 0.0]
