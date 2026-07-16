from __future__ import annotations

from pathlib import Path

import pytest

from vt_dual_franka_shared.models import ControllerState, DualArmControllerState
from vt_dual_franka_workspace.collection import BimanualDataCollector
from vt_dual_franka_workspace.config import (
    ArmEndpointSettings,
    ArmInitialPoseSettings,
    CollectionRuntimeSettings,
    ModalitySettings,
    TaskConfig,
    WorkspaceSettings,
)
from vt_dual_franka_workspace.operator import OperatorActionError
from vt_dual_franka_workspace.recording import RunSessionManager
from vt_dual_franka_workspace.runtime.motion import eef_xyz_rpy_deg_to_tcp_pose


LEFT_JOINTS = [1.3922, -0.7209, 0.1799, -2.8098, 0.1171, 2.0371, 0.7]
RIGHT_JOINTS = [-1.5857, -0.7485, 0.0049, -2.8095, 0.0293, 2.0184, 0.9069]


def _state(arm_id: str, pose: list[float], joints: list[float]) -> ControllerState:
    return ControllerState(
        arm_id=arm_id,
        tcp_pose=eef_xyz_rpy_deg_to_tcp_pose(pose),
        joint_positions=joints,
        gripper_width=0.078,
    )


class FakeCoordinator:
    def __init__(self) -> None:
        self.states = {
            "left": _state("left", [0.0, 0.3, 0.4, -180.0, 0.0, 45.0], LEFT_JOINTS),
            "right": _state("right", [0.0, -0.3, 0.4, -180.0, 0.0, -145.0], RIGHT_JOINTS),
        }
        self.reset_calls = []
        self.grasp_calls = []
        self.move_calls = []
        self.fail_reset = False

    def reset_pair(self, commands, command_id=None):
        del command_id
        if self.fail_reset:
            raise RuntimeError("right reset failed")
        self.reset_calls.append(commands)
        for arm_id, command in commands.items():
            updates = {}
            if command.eef_pose_xyz_rpy_deg is not None:
                updates["tcp_pose"] = eef_xyz_rpy_deg_to_tcp_pose(
                    command.eef_pose_xyz_rpy_deg
                )
            if command.joint_positions is not None:
                updates["joint_positions"] = list(command.joint_positions)
            self.states[arm_id] = self.states[arm_id].model_copy(update=updates)
        return {"left": {"status": "ok"}, "right": {"status": "ok"}}

    def get_state(self, max_age_sec=None):
        del max_age_sec
        return DualArmControllerState(
            left=self.states["left"],
            right=self.states["right"],
        )

    def is_healthy(self, max_age_sec=2.0):
        del max_age_sec
        return True

    def snapshot(self):
        return {
            "left": {"healthy": True, "age_sec": 0.0},
            "right": {"healthy": True, "age_sec": 0.0},
        }

    def grasp_grippers(self, **kwargs):
        self.grasp_calls.append(kwargs)

    def move_grippers(self, widths, **kwargs):
        self.move_calls.append((widths, kwargs))

    def start(self):
        return None

    def stop(self):
        return None


class FakeTeleopService:
    def __init__(self) -> None:
        self.enabled = False

    def set_teleop_enabled(self, enabled: bool):
        self.enabled = enabled

    def is_teleop_enabled(self):
        return self.enabled

    def has_recent_message(self, timeout_sec: float):
        del timeout_sec
        return True


class FakeServer:
    def is_alive(self):
        return True


def _collector(tmp_path: Path) -> tuple[BimanualDataCollector, FakeCoordinator]:
    workspace = WorkspaceSettings(
        arms={
            "left": ArmEndpointSettings(arm_id="left", port=8092),
            "right": ArmEndpointSettings(arm_id="right", port=8093),
        },
        recording={
            "collect_root": tmp_path / "collect",
            "eval_root": tmp_path / "eval",
        },
        operator_ui={"enabled": False},
    )
    task = TaskConfig(
        task_name="bimanual_demo",
        initial_poses={
            "left": ArmInitialPoseSettings(
                eef_pose_xyz_rpy_deg=[0.0, 0.3, 0.4, -180.0, 0.0, 45.0],
                joint_positions_rad=LEFT_JOINTS,
            ),
            "right": ArmInitialPoseSettings(
                eef_pose_xyz_rpy_deg=[0.0, -0.3, 0.4, -180.0, 0.0, -145.0],
                joint_positions_rad=RIGHT_JOINTS,
            ),
        },
        collection=CollectionRuntimeSettings(
            require_quest_connection=False,
            start_countdown_sec=0.0,
        ),
        modality=ModalitySettings(
            proprioception=True,
            rgb_cameras=[],
            gelsight_frame=False,
        ),
    )
    coordinator = FakeCoordinator()
    collector = BimanualDataCollector(
        workspace,
        task,
        coordinator,
        calibration=object(),
    )
    collector.sessions = RunSessionManager(tmp_path / "collect")
    collector.sessions.start_run("bimanual_demo")
    collector.teleop_service = FakeTeleopService()
    collector.teleop_server = FakeServer()
    return collector, coordinator


def test_initial_pose_reset_always_sends_left_and_right_pair(tmp_path: Path):
    collector, coordinator = _collector(tmp_path)

    collector.operator_reset_ready_pose()

    commands = coordinator.reset_calls[-1]
    assert set(commands) == {"left", "right"}
    assert commands["left"].eef_pose_xyz_rpy_deg == [
        0.0,
        0.3,
        0.4,
        -180.0,
        0.0,
        45.0,
    ]
    assert commands["right"].eef_pose_xyz_rpy_deg == [
        0.0,
        -0.3,
        0.4,
        -180.0,
        0.0,
        -145.0,
    ]
    collector.operator_start_episode()
    assert collector._current_episode_dir is not None
    assert collector.teleop_service.is_teleop_enabled()


def test_joint_reset_always_sends_requested_left_and_right_joints(tmp_path: Path):
    collector, coordinator = _collector(tmp_path)

    collector.operator_reset_home_joints()

    commands = coordinator.reset_calls[-1]
    assert commands["left"].joint_positions == LEFT_JOINTS
    assert commands["right"].joint_positions == RIGHT_JOINTS
    assert collector._home_joint_completed
    assert not collector._initial_pose_completed


def test_paired_reset_failure_does_not_mark_collector_ready(tmp_path: Path):
    collector, coordinator = _collector(tmp_path)
    coordinator.fail_reset = True

    with pytest.raises(OperatorActionError, match="both arms"):
        collector.operator_reset_ready_pose()

    assert not collector._initial_pose_completed
    with pytest.raises(OperatorActionError):
        collector.operator_start_episode()


def test_forever_closed_task_operates_both_grippers(tmp_path: Path):
    collector, coordinator = _collector(tmp_path)
    collector.task.gripper_forever_closed = True

    collector.operator_reset_ready_pose()
    assert collector._pending_initial_gripper_close
    collector.operator_confirm_gripper_closed()

    assert len(coordinator.grasp_calls) == 1
    assert collector._initial_pose_completed
