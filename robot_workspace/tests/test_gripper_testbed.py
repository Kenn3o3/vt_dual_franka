import time

from fastapi.testclient import TestClient

from vt_franka_shared.models import ControllerState
from vt_franka_workspace.gripper_testbed.report import write_gripper_testbed_report
from vt_franka_workspace.gripper_testbed.replay import create_gripper_testbed_replay_app
from vt_franka_workspace.gripper_testbed.service import GripperTestbedService, GripperTestbedSettings, map_trigger_to_width, create_gripper_testbed_app


class FakeController:
    def __init__(self):
        self.targets = []
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
        del args, kwargs
        self.state = self.state.model_copy(update={"gripper_width": 0.078, "gripper_force": 0.0})
        return {"status": "queued"}

    def stop_gripper(self):
        return {"status": "stopped"}


def test_trigger_to_width_mapping_is_monotonic():
    open_width = map_trigger_to_width(0.0, min_width=0.0, max_width=0.078, gamma=1.5)
    mid_width = map_trigger_to_width(0.5, min_width=0.0, max_width=0.078, gamma=1.5)
    close_width = map_trigger_to_width(1.0, min_width=0.0, max_width=0.078, gamma=1.5)
    assert open_width > mid_width > close_width


def test_gripper_testbed_endpoint_updates_state_and_records_samples():
    controller = FakeController()
    service = GripperTestbedService(GripperTestbedSettings(), controller)
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
    assert samples[-1]["target_width"] < 0.078


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
