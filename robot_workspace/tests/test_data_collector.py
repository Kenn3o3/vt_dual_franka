from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from vt_dual_franka_shared.models import ControllerState
from vt_dual_franka_workspace.collection import DataCollector
from vt_dual_franka_workspace.config import CollectionRuntimeSettings, ModalitySettings, RgbCameraSettings, TaskConfig, WorkspaceSettings
from vt_dual_franka_workspace.operator import OperatorActionError
from vt_dual_franka_workspace.recording import EpisodeImageStreamRecorder, RunSessionManager
from vt_dual_franka_workspace.runtime import LiveSampleBuffer, eef_xyz_rpy_deg_to_tcp_pose


class FakeController:
    def __init__(self):
        self.tcp_targets = []
        self.reset_commands = []
        self.gripper_moves = []
        self.gripper_grasps = []
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

    def reset(self, command):
        self.reset_commands.append(command)
        if command.joint_positions is not None:
            self.state = self.state.model_copy(update={"joint_positions": list(command.joint_positions)})
        return {"status": "ok", "profile": command.profile}

    def grasp_gripper(self, velocity, force_limit, source="workspace", blocking=False):
        self.gripper_grasps.append((velocity, force_limit, source, blocking))
        self.state = self.state.model_copy(update={"gripper_width": 0.0, "gripper_force": float(force_limit)})

    def move_gripper(self, width, velocity, force_limit, source="workspace", blocking=False):
        self.gripper_moves.append((width, velocity, force_limit, source, blocking))
        self.state = self.state.model_copy(update={"gripper_width": float(width), "gripper_force": float(force_limit)})


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
        home_joint_positions_rad=[0.0] * 7,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
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
    assert collector.controller.reset_commands == []
    assert collector.controller.tcp_targets[-1][1] == "data_collector_initial_pose"


def test_data_collector_home_joint_reset_is_optional_but_invalidates_initial_pose(tmp_path: Path):
    collector = make_collector(tmp_path)

    collector.operator_reset_ready_pose()
    assert collector.get_operator_status()["ready"] is True

    collector.operator_reset_home_joints()
    status = collector.get_operator_status()

    assert status["ready"] is False
    assert "robot has not been moved to the task initial pose with H" in status["reasons"]
    assert collector._initial_pose_completed is False
    assert collector._current_initial_pose is None
    with pytest.raises(OperatorActionError):
        collector.operator_start_episode()

    collector.operator_reset_ready_pose()
    collector.operator_start_episode()

    assert collector._current_episode_dir is not None
    assert collector.controller.reset_commands[-1].source == "data_collector_home_joints"
    assert collector.controller.tcp_targets[-1][1] == "data_collector_initial_pose"


def test_data_collector_keeps_teleop_enabled_after_save_until_next_initial_pose(tmp_path: Path):
    collector = make_collector(tmp_path)

    collector.operator_reset_home_joints()
    collector.operator_reset_ready_pose()
    assert collector.teleop_service.is_teleop_enabled() is False
    collector.operator_start_episode()
    assert collector.teleop_service.is_teleop_enabled() is True

    collector.operator_stop_episode()

    assert collector._current_episode_dir is None
    assert collector.teleop_service.is_teleop_enabled() is True
    assert collector._home_joint_completed is False

    collector.operator_reset_home_joints()
    collector.operator_reset_ready_pose()

    assert collector.teleop_service.is_teleop_enabled() is False


def test_data_collector_operator_ui_contract_has_unsupported_outcome_actions(tmp_path: Path):
    collector = make_collector(tmp_path)

    status = collector.get_operator_status()

    assert status["allowed_actions"]["mark_success"] is False
    assert status["allowed_actions"]["mark_fail"] is False
    with pytest.raises(OperatorActionError):
        collector.operator_mark_episode_success()
    with pytest.raises(OperatorActionError):
        collector.operator_mark_episode_fail()


def test_data_collector_forever_closed_always_waits_for_confirm_after_initial_pose(tmp_path: Path):
    collector = make_collector(tmp_path)
    collector.task.gripper_forever_closed = True
    collector.controller.state = collector.controller.state.model_copy(update={"gripper_width": 0.0, "gripper_force": 7.0})

    collector.operator_reset_home_joints()
    collector.operator_reset_ready_pose()

    assert collector._pending_initial_gripper_close is True
    assert collector._initial_pose_completed is False
    with pytest.raises(OperatorActionError):
        collector.operator_start_episode()

    collector.operator_confirm_gripper_closed()
    collector.operator_start_episode()

    assert collector._current_episode_dir is not None
    assert collector.controller.gripper_grasps[-1] == (
        collector.workspace.teleop.gripper_velocity,
        collector.workspace.teleop.grasp_force,
        "data_collector_initial_gripper_close",
        True,
    )
    manifest = collector._current_episode_dir.joinpath("episode_manifest.json").read_text(encoding="utf-8")
    assert '"gripper_forever_closed": true' in manifest


def test_data_collector_forever_closed_confirm_always_sends_grasp(tmp_path: Path):
    collector = make_collector(tmp_path)
    collector.task.gripper_forever_closed = True
    collector.controller.state = collector.controller.state.model_copy(update={"gripper_width": 0.0, "gripper_force": 7.0})

    collector.operator_reset_home_joints()
    collector.operator_reset_ready_pose()
    collector.operator_confirm_gripper_closed()

    assert len(collector.controller.gripper_grasps) == 1
    collector.operator_confirm_gripper_closed()
    assert len(collector.controller.gripper_grasps) == 2


def test_data_collector_forever_closed_open_requires_reclose_before_start(tmp_path: Path):
    collector = make_collector(tmp_path)
    collector.task.gripper_forever_closed = True

    collector.operator_reset_home_joints()
    collector.operator_reset_ready_pose()
    collector.operator_confirm_gripper_closed()
    assert collector.get_operator_status()["ready"] is True

    collector.operator_open_gripper()

    status = collector.get_operator_status()
    assert status["ready"] is False
    assert status["allowed_actions"]["confirm_gripper_closed"] is True
    assert status["allowed_actions"]["open_gripper"] is True
    assert "initial gripper close is pending; press C to confirm" in status["reasons"]
    with pytest.raises(OperatorActionError):
        collector.operator_start_episode()

    collector.operator_confirm_gripper_closed()
    collector.operator_start_episode()

    assert collector._current_episode_dir is not None
    assert collector.controller.gripper_moves[-1] == (
        collector.workspace.teleop.max_gripper_width,
        collector.workspace.teleop.gripper_velocity,
        collector.workspace.teleop.grasp_force,
        "data_collector_gripper_adjustment_open",
        True,
    )


def test_data_collector_records_randomized_initial_pose_metadata(tmp_path: Path):
    collector = make_collector(tmp_path)
    collector.task.rand_init_pose = [0.01, 0.0, 0.0]

    collector.operator_reset_home_joints()
    collector.operator_reset_ready_pose()
    collector.operator_start_episode()

    assert collector._current_episode_dir is not None
    manifest = collector._current_episode_dir.joinpath("episode_manifest.json").read_text(encoding="utf-8")
    assert '"initial_pose"' in manifest
    assert '"delta_xyz_m"' in manifest


def test_data_collector_serves_live_wrist_preview_before_start_and_stops_while_active(tmp_path: Path):
    collector = make_collector(tmp_path)
    collector.rgb_camera_buffers["wrist"] = LiveSampleBuffer("rgb_wrist")
    collector.rgb_camera_buffers["wrist"].update(
        np.zeros((6, 7, 3), dtype=np.uint8),
        metadata={"camera_name": "wrist"},
        captured_wall_time=time.time(),
    )

    status = collector.get_operator_status()

    assert status["ready"] is False
    assert status["preview"]["role"] == "wrist"
    assert status["preview"]["streaming"] is True
    assert status["snapshots"]["wrist"]["available"] is True
    snapshot = collector.get_operator_snapshot("wrist")
    assert snapshot is not None
    assert snapshot.image.shape == (6, 7, 3)

    collector.rgb_camera_buffers["wrist"].update(
        np.ones((6, 7, 3), dtype=np.uint8),
        metadata={"camera_name": "wrist"},
        captured_wall_time=time.time(),
    )
    updated_snapshot = collector.get_operator_snapshot("wrist")
    assert updated_snapshot is not None
    assert int(updated_snapshot.image[0, 0, 0]) == 1

    collector.operator_reset_home_joints()
    collector.operator_reset_ready_pose()
    assert collector.get_operator_status()["ready"] is True
    collector.operator_start_episode()
    active_status = collector.get_operator_status()
    assert active_status["preview"]["streaming"] is False
    assert collector.get_operator_snapshot("wrist") is None


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

    monkeypatch.setattr("vt_dual_franka_workspace.collection.data_collector.StateBridge", FakeBridge)
    monkeypatch.setattr("vt_dual_franka_workspace.collection.data_collector.ManagedUvicornServer", FakeTeleopServer)
    monkeypatch.setattr("vt_dual_franka_workspace.collection.data_collector.QuestTeleopService", FakeTeleopService)
    monkeypatch.setattr(
        "vt_dual_franka_workspace.collection.data_collector.build_rgb_camera_recorder",
        lambda spec, **kwargs: type("Service", (), {"run": lambda self, stop_event=None: None})(),
    )
    monkeypatch.setattr(
        "vt_dual_franka_workspace.collection.data_collector.start_thread_worker",
        lambda workers, name, target, required: started.append(name),
    )

    collector._start_workers()

    assert started == ["rgb_camera:third_person"]


def test_data_collector_flushes_buffered_gelsight_on_stop(tmp_path: Path):
    collector = make_collector(tmp_path)
    collector.task.modality = ModalitySettings(proprioception=True, gelsight_frame=True)
    collector.task.gelsight.enabled = True
    collector.task.gelsight.buffered_recording = True
    collector.image_stream_recorders["tactile_left"] = EpisodeImageStreamRecorder(
        collector.sessions,
        "tactile_left",
        image_format="jpg",
        max_frames=4,
    )

    collector.operator_reset_home_joints()
    collector.operator_reset_ready_pose()
    collector.operator_start_episode()
    assert collector._current_episode_dir is not None
    episode_dir = collector._current_episode_dir
    collector.image_stream_recorders["tactile_left"].record_frame(
        np.zeros((3, 4, 3), dtype=np.uint8),
        captured_wall_time=20.0,
        sequence_id=0,
    )

    collector.operator_stop_episode()

    assert (episode_dir / "streams" / "tactile_left" / "index.jsonl").exists()
    assert (episode_dir / "episode_qc.json").exists()
    manifest = (episode_dir / "episode_manifest.json").read_text(encoding="utf-8")
    assert '"tactile_left"' in manifest
    assert '"frame_count": 1' in manifest
    assert '"episode_qc"' in manifest
