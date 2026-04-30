from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from vt_franka_shared.models import ControllerState
from vt_franka_workspace.collection import DataCollector
from vt_franka_workspace.config import CollectionRuntimeSettings, ModalitySettings, RgbCameraSettings, TaskConfig, WorkspaceSettings
from vt_franka_workspace.operator import OperatorActionError
from vt_franka_workspace.recording import RunSessionManager
from vt_franka_workspace.runtime import LiveSampleBuffer, eef_xyz_rpy_deg_to_tcp_pose


class FakeController:
    def __init__(self):
        self.tcp_targets = []
        self.state = ControllerState(
            tcp_pose=eef_xyz_rpy_deg_to_tcp_pose([0.4, 0.0, 0.3, 180.0, 0.0, 0.0]),
            tcp_velocity=[0.0] * 6,
            tcp_wrench=[0.0] * 6,
            joint_positions=[0.0] * 7,
            joint_velocities=[0.0] * 7,
            gripper_width=0.078,
            gripper_force=0.0,
        )

    def queue_tcp(self, target_tcp, source="workspace", target_duration_sec=None):
        self.tcp_targets.append((list(target_tcp), source, target_duration_sec))
        self.state = self.state.model_copy(update={"tcp_pose": list(target_tcp)})


class FakeStateMonitor:
    def __init__(self, controller: FakeController):
        self.controller = controller

    def start(self):
        return None

    def stop(self):
        return None

    def get_state(self, max_age_sec=None):
        del max_age_sec
        return self.controller.state

    def is_healthy(self, max_age_sec=2.0):
        del max_age_sec
        return True

    def snapshot(self):
        return {"healthy": True, "age_sec": 0.0, "sample_count": 1, "failure_count": 0, "max_gap_sec": 0.0, "last_error": None}


class FakeTeleopService:
    def __init__(self):
        self.enabled = False

    def set_teleop_enabled(self, enabled: bool):
        self.enabled = enabled

    def has_recent_message(self, timeout_sec: float):
        del timeout_sec
        return True

    def is_teleop_enabled(self):
        return self.enabled


class FakeServer:
    def is_alive(self):
        return True


def make_collector(tmp_path: Path) -> DataCollector:
    workspace = WorkspaceSettings(
        recording={"collect_root": tmp_path / "collect", "eval_root": tmp_path / "eval", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    task = TaskConfig(
        task_name="put_cup_on_plate",
        initial_eef_pose_xyz_rpy_deg=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        initial_move_duration_sec=1.0,
        collection=CollectionRuntimeSettings(start_countdown_sec=0.0, require_quest_connection=True),
        modality=ModalitySettings(proprioception=True, rgb_cameras=["third_person"]),
        rgb_cameras={"third_person": RgbCameraSettings(stream_name="rgb_third_person")},
    )
    controller = FakeController()
    collector = DataCollector(workspace, task, controller, calibration=None)
    collector.sessions = RunSessionManager(tmp_path / "collect")
    collector.sessions.start_run(task.task_name)
    collector.state_monitor = FakeStateMonitor(controller)
    collector.teleop_service = FakeTeleopService()
    collector.teleop_server = FakeServer()
    collector.rgb_camera_buffers["third_person"] = LiveSampleBuffer("rgb_third_person")
    return collector


def test_data_collector_requires_initial_pose_before_start(tmp_path: Path):
    collector = make_collector(tmp_path)

    with pytest.raises(OperatorActionError):
        collector.operator_start_episode()

    collector.operator_reset_ready_pose()
    collector.operator_start_episode()

    assert collector._current_episode_dir is not None
    assert collector.teleop_service.is_teleop_enabled() is True
    assert collector.controller.tcp_targets[-1][1] == "data_collector_initial_pose"


def test_data_collector_freezes_idle_snapshot_when_ready(tmp_path: Path):
    collector = make_collector(tmp_path)
    collector.rgb_camera_buffers["third_person"].update(
        np.zeros((6, 7, 3), dtype=np.uint8),
        metadata={"camera_name": "third_person"},
        captured_wall_time=time.time(),
    )

    collector.operator_reset_ready_pose()
    status = collector.get_operator_status()

    assert status["ready"] is True
    assert status["snapshots"]["third_person"]["available"] is True
    snapshot = collector.get_operator_snapshot("third_person")
    assert snapshot is not None
    assert snapshot.image.shape == (6, 7, 3)


def test_data_collector_starts_only_modality_requested_cameras(tmp_path: Path, monkeypatch):
    collector = make_collector(tmp_path)
    collector.task.modality.rgb_cameras = ["third_person"]
    collector.task.rgb_cameras["unused"] = RgbCameraSettings(stream_name="rgb_unused")
    started = []

    monkeypatch.setattr(collector.state_monitor, "start", lambda: None)

    class FakeBridge:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self):
            return None

    class FakeTeleopServer:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def start(self):
            return None

        def is_alive(self):
            return True

    class FakeTeleopService:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def set_teleop_enabled(self, enabled):
            del enabled

        def has_recent_message(self, timeout_sec):
            del timeout_sec
            return True

        def get_gripper_status(self):
            return {}

    monkeypatch.setattr("vt_franka_workspace.collection.data_collector.StateBridge", FakeBridge)
    monkeypatch.setattr("vt_franka_workspace.collection.data_collector.ManagedUvicornServer", FakeTeleopServer)
    monkeypatch.setattr("vt_franka_workspace.collection.data_collector.QuestTeleopService", FakeTeleopService)
    monkeypatch.setattr(
        "vt_franka_workspace.collection.data_collector.build_rgb_camera_recorder",
        lambda spec, **kwargs: type("Service", (), {"run": lambda self, stop_event=None: None})(),
    )
    monkeypatch.setattr(
        "vt_franka_workspace.collection.data_collector.start_thread_worker",
        lambda workers, name, target, required: started.append(name),
    )

    collector._start_workers()

    assert started == ["rgb_camera:third_person"]
