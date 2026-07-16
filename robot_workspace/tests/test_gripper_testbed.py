import time

from fastapi.testclient import TestClient

from vt_dual_franka_shared.models import ControllerState
from vt_dual_franka_workspace.gripper_testbed.report import write_gripper_testbed_report
from vt_dual_franka_workspace.gripper_testbed.replay import create_gripper_testbed_replay_app
from vt_dual_franka_workspace.gripper_testbed.service import GripperTestbedService, GripperTestbedSettings, map_trigger_to_width, create_gripper_testbed_app


class FakeController:
    def __init__(self):
        self.targets = []
        self.open_calls = []
        self.stop_calls = 0
        self.state = ControllerState(
            tcp_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            tcp_velocity=[0.0] * 6,
            tcp_wrench=[0.0] * 6,
            joint_positions=[0.0] * 7,
            joint_velocities=[0.0] * 7,
            gripper_width=0.078,
            gripper_force=0.0,
            control_frequency_hz=60.0,
            backend="mock",
        )

    def get_state(self):
        return self.state

    def send_target(self, command):
        self.targets.append(command)
        self.state = self.state.model_copy(update={"gripper_width": command.target_width, "gripper_force": command.force_limit})
        return {"status": "queued", "sequence": command.sequence}

    def open_gripper(self, *args, **kwargs):
        del args
        self.open_calls.append(kwargs)
        width = float(kwargs.get("width", 0.078))
        force = float(kwargs.get("force_limit", 0.0))
        self.state = self.state.model_copy(update={"gripper_width": width, "gripper_force": force})
        return {"status": "queued"}

    def stop_gripper(self):
        self.stop_calls += 1
        return {"status": "stopped"}


def test_trigger_to_width_mapping_is_monotonic():
    open_width = map_trigger_to_width(0.0, min_width=0.0, max_width=0.078, gamma=1.5)
    mid_width = map_trigger_to_width(0.5, min_width=0.0, max_width=0.078, gamma=1.5)
    close_width = map_trigger_to_width(1.0, min_width=0.0, max_width=0.078, gamma=1.5)
    assert open_width > mid_width > close_width


def test_gripper_testbed_endpoint_updates_state_and_records_samples():
    controller = FakeController()
    service = GripperTestbedService(GripperTestbedSettings(control_hand="left"), controller)
    app = create_gripper_testbed_app(service)

    payload = {
        "timestamp": 123.0,
        "leftHandPose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        "leftGripperState": 0.9,
        "buttonStates": {"button_4": True},
    }

    with TestClient(app) as client:
        client.post("/api/v1/enable?enabled=true")
        client.post("/api/v1/arm")
        client.post("/unity", json=payload)
        time.sleep(0.1)
        status = client.get("/api/v1/status").json()
        samples = client.get("/api/v1/samples").json()["samples"]

    assert status["enabled"] is True
    assert status["armed"] is True
    assert len(controller.targets) >= 1
    assert samples
    assert samples[-1]["trigger_depth"] == 0.9
    assert samples[-1]["gripper_command"] == -1
    assert samples[-1]["target_width"] < 0.078


def test_gripper_testbed_start_stop_records_latest_with_replacement(tmp_path):
    controller = FakeController()
    settings = GripperTestbedSettings(collect_root=tmp_path, require_enable_button=False, control_hand="left")
    service = GripperTestbedService(settings, controller)
    app = create_gripper_testbed_app(service)
    payload = {
        "timestamp": 123.0,
        "leftHandPose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        "leftGripperState": 0.8,
        "buttonStates": {},
    }

    with TestClient(app) as client:
        response = client.post("/api/v1/test/start")
        assert response.status_code == 200
        client.post("/unity", json=payload)
        time.sleep(0.1)
        status = client.get("/api/v1/status").json()
        assert status["enabled"] is True
        assert status["armed"] is True
        assert status["command_allowed"] is True
        assert status["active_run_dir"].endswith("/latest")
        first_target_count = len(controller.targets)

        response = client.post("/api/v1/test/stop")
        assert response.status_code == 200
        status = client.get("/api/v1/status").json()
        assert status["enabled"] is False
        assert status["armed"] is False
        assert (tmp_path / "latest" / "episodes" / "episode_0000" / "episode_manifest.json").exists()

        response = client.post("/api/v1/test/start")
        assert response.status_code == 200
        assert len(list((tmp_path / "latest" / "episodes").glob("episode_*"))) == 1
        client.post("/unity", json=payload)
        time.sleep(0.1)
        assert len(controller.targets) > first_target_count


def test_gripper_testbed_manual_open_and_close_use_configured_widths(tmp_path):
    controller = FakeController()
    settings = GripperTestbedSettings(collect_root=tmp_path, min_gripper_width=0.01, max_gripper_width=0.07)
    app = create_gripper_testbed_app(GripperTestbedService(settings, controller))

    with TestClient(app) as client:
        response = client.post("/api/v1/gripper/open-max")
        assert response.status_code == 200
        assert response.json()["target_width"] == 0.07
        assert controller.targets[-1].target_width == 0.07

        response = client.post("/api/v1/gripper/close-min")
        assert response.status_code == 200
        assert response.json()["target_width"] == 0.01
        assert controller.targets[-1].target_width == 0.01

        response = client.post("/api/v1/gripper/stop")
        assert response.status_code == 200
        assert response.json()["gripper_command"] == 0
        assert controller.stop_calls == 1


def test_gripper_testbed_report_and_replay(tmp_path):
    stream_dir = tmp_path / "run" / "episodes" / "episode_0000" / "streams"
    stream_dir.mkdir(parents=True)
    (stream_dir / "gripper_telemetry.jsonl").write_text(
        '{"wall_time": 1.0, "trigger_depth": 0.0, "target_width": 0.078, "force_limit": 5.0}\n'
        '{"wall_time": 2.0, "trigger_depth": 1.0, "target_width": 0.0, "force_limit": 7.0}\n',
        encoding="utf-8",
    )
    (stream_dir / "gripper_states.jsonl").write_text(
        '{"wall_time": 1.0, "trigger_depth": 0.0, "target_width": 0.078, "measured_width": 0.078, "measured_force": 0.0, "width_error": 0.0}\n'
        '{"wall_time": 2.0, "trigger_depth": 1.0, "target_width": 0.0, "measured_width": 0.002, "measured_force": 7.0, "width_error": -0.002}\n',
        encoding="utf-8",
    )

    report = write_gripper_testbed_report(tmp_path / "run")
    assert report.exists()

    app = create_gripper_testbed_replay_app(tmp_path / "run")
    with TestClient(app) as client:
        samples = client.get("/api/v1/samples").json()["samples"]
        status = client.get("/api/v1/status").json()

    assert len(samples) == 2
    assert status["active_episode_name"] == "replay"
