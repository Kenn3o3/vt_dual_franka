import time
from types import SimpleNamespace

from fastapi.testclient import TestClient

from vt_dual_franka_controller.api.gripper_testbed_app import create_gripper_testbed_app
from vt_dual_franka_controller.backends.franka_ros_gripper import FrankaRosGripperOnlyBackend, RosGripperState
from vt_dual_franka_controller.backends.gripper_only import PolymetisGripperOnlyBackend
from vt_dual_franka_controller.backends.mock import MockFrankaBackend
from vt_dual_franka_controller.control.gripper_service import GripperTestbedService
from vt_dual_franka_controller.settings import BackendSettings, ControlSettings, ControllerSettings, RosGripperActionSettings, ServerSettings


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


def test_gripper_only_backend_accepts_dict_state_without_polymetis_constructor():
    backend = PolymetisGripperOnlyBackend.__new__(PolymetisGripperOnlyBackend)
    backend._gripper = SimpleNamespace(get_state=lambda: {"width": [0.042], "force": [3.5]})
    backend._last_width = 0.078
    backend._last_force = 0.0
    backend._state_warning_logged = False

    state = backend.get_controller_state(50.0)

    assert state.gripper_width == 0.042
    assert state.gripper_force == 3.5


def test_gripper_only_backend_falls_back_when_width_missing():
    backend = PolymetisGripperOnlyBackend.__new__(PolymetisGripperOnlyBackend)
    backend._gripper = SimpleNamespace(get_state=lambda: SimpleNamespace(other=1.0))
    backend._last_width = 0.055
    backend._last_force = 2.0
    backend._state_warning_logged = False

    state = backend.get_controller_state(50.0)

    assert state.gripper_width == 0.055
    assert state.gripper_force == 2.0


class FakeRosGripperDriver:
    def __init__(self):
        self.calls = []
        self.state = RosGripperState(width=0.078, force=0.0, wall_time=time.time())

    def move(self, *, width, speed):
        self.calls.append(("move", width, speed))
        self.state = RosGripperState(width=width, force=self.state.force, wall_time=time.time())

    def grasp(self, *, width, speed, force, epsilon_inner, epsilon_outer):
        self.calls.append(("grasp", width, speed, force, epsilon_inner, epsilon_outer))
        self.state = RosGripperState(width=width, force=force, wall_time=time.time())

    def stop(self):
        self.calls.append(("stop",))

    def get_state(self):
        return self.state

    def home(self):
        self.calls.append(("home",))

    def shutdown(self):
        self.calls.append(("shutdown",))


def test_franka_ros_gripper_backend_maps_close_target_to_grasp_action():
    driver = FakeRosGripperDriver()
    settings = RosGripperActionSettings(close_width_threshold=0.001, grasp_epsilon_inner=0.002, grasp_epsilon_outer=0.08)
    backend = FrankaRosGripperOnlyBackend(settings, driver=driver)

    backend.move_gripper(width=0.0, velocity=0.02, force_limit=7.0)

    assert driver.calls == [("grasp", 0.0, 0.02, 7.0, 0.002, 0.08)]


def test_franka_ros_gripper_backend_maps_open_target_to_move_action():
    driver = FakeRosGripperDriver()
    settings = RosGripperActionSettings(max_gripper_width=0.078, close_width_threshold=0.001)
    backend = FrankaRosGripperOnlyBackend(settings, driver=driver)

    backend.move_gripper(width=0.078, velocity=0.05, force_limit=7.0)

    assert driver.calls == [("move", 0.078, 0.05)]


def test_franka_ros_gripper_backend_stop_uses_driver_stop():
    driver = FakeRosGripperDriver()
    backend = FrankaRosGripperOnlyBackend(RosGripperActionSettings(), driver=driver)

    backend.stop_gripper()

    assert driver.calls == [("stop",)]
