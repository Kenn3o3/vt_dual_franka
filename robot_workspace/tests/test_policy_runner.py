from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from vt_dual_franka_shared.models import ControllerState, DualArmControllerState
from vt_dual_franka_workspace.config import (
    ArmEndpointSettings,
    ArmInitialPoseSettings,
    InferenceRuntimeSettings,
    ModalitySettings,
    WorkspaceSettings,
)
from vt_dual_franka_workspace.inference import (
    Action,
    BimanualObservationAssembler,
    BimanualPolicyRunner,
    DualActionExecutor,
    ObservationHistory,
)
from vt_dual_franka_workspace.policies.base import Policy
from vt_dual_franka_workspace.runtime.live_buffer import LiveSampleBuffer
from vt_dual_franka_workspace.runtime.motion import eef_xyz_rpy_deg_to_tcp_pose


class FakeController:
    def __init__(self):
        self.grasps = []
        self.moves = []

    def grasp_gripper(self, *args, **kwargs):
        self.grasps.append((args, kwargs))

    def move_gripper(self, *args, **kwargs):
        self.moves.append((args, kwargs))


class Context:
    def __init__(self):
        self.controller = FakeController()


class FakeCoordinator:
    def __init__(self):
        self.arms = {"left": Context(), "right": Context()}
        self.paired_targets = []
        self.resets = []
        self.states = {
            "left": ControllerState(
                arm_id="left",
                tcp_pose=eef_xyz_rpy_deg_to_tcp_pose(
                    [0.0, 0.3, 0.4, -180.0, 0.0, 45.0]
                ),
                joint_positions=[0.0] * 7,
            ),
            "right": ControllerState(
                arm_id="right",
                tcp_pose=eef_xyz_rpy_deg_to_tcp_pose(
                    [0.0, -0.3, 0.4, -180.0, 0.0, -145.0]
                ),
                joint_positions=[0.0] * 7,
            ),
        }

    def queue_tcp_pair(self, targets, **kwargs):
        self.paired_targets.append((targets, kwargs))
        return "pair"

    def reset_pair(self, commands, command_id=None):
        del command_id
        self.resets.append(commands)
        for arm, command in commands.items():
            update = {}
            if command.eef_pose_xyz_rpy_deg is not None:
                update["tcp_pose"] = eef_xyz_rpy_deg_to_tcp_pose(
                    command.eef_pose_xyz_rpy_deg
                )
            if command.joint_positions is not None:
                update["joint_positions"] = list(command.joint_positions)
            self.states[arm] = self.states[arm].model_copy(update=update)
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

    def move_grippers(self, widths, **kwargs):
        for arm, width in widths.items():
            self.arms[arm].controller.move_gripper(width, **kwargs)

    def grasp_grippers(self, **kwargs):
        for arm in ("left", "right"):
            self.arms[arm].controller.grasp_gripper(**kwargs)

    def start(self):
        return None

    def stop(self):
        return None


class NoopPolicy(Policy):
    def predict(self, observation_window):
        del observation_window
        return []


def _fresh_buffer(name: str, value: int) -> LiveSampleBuffer:
    buffer = LiveSampleBuffer(name)
    buffer.update(
        np.full((4, 5, 3), value, dtype=np.uint8),
        captured_wall_time=time.time(),
    )
    return buffer


def test_dual_action_executor_issues_one_paired_target_and_two_grippers():
    coordinator = FakeCoordinator()
    executor = DualActionExecutor(coordinator)
    action = Action(
        target_tcp_by_arm={
            "left": [0.0, 0.3, 0.4, 1.0, 0.0, 0.0, 0.0],
            "right": [0.0, -0.3, 0.4, 1.0, 0.0, 0.0, 0.0],
        },
        target_duration_sec=0.1,
        gripper_closed_by_arm={"left": True, "right": False},
        gripper_width_by_arm={"left": 0.0, "right": 0.078},
    )

    executor.execute_normalized(action)

    assert len(coordinator.paired_targets) == 1
    assert len(coordinator.arms["left"].controller.grasps) == 1
    assert len(coordinator.arms["right"].controller.moves) == 1


def test_dual_action_executor_rejects_single_arm_action():
    with pytest.raises(ValueError, match="target_tcp_by_arm"):
        DualActionExecutor(FakeCoordinator()).execute_normalized(
            Action(target_tcp=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        )


def test_bimanual_observation_assembler_requires_and_returns_four_images():
    coordinator = FakeCoordinator()
    assembler = BimanualObservationAssembler(
        coordinator=coordinator,
        rgb_camera_buffers={
            "rgb_wrist_left": _fresh_buffer("rgb_wrist_left", 1),
            "rgb_wrist_right": _fresh_buffer("rgb_wrist_right", 2),
        },
        tactile_buffers={
            "left": _fresh_buffer("tactile_left", 3),
            "right": _fresh_buffer("tactile_right", 4),
        },
    )

    observation, recorded = assembler.assemble()

    assert set(observation["images"]) == {"left_wrist", "right_wrist"}
    assert set(observation["tactile"]) == {"left", "right"}
    assert set(
        observation["proprioception"]["controller_state_by_arm"]
    ) == {"left", "right"}
    assert set(recorded["proprioception"]["controller_state_by_arm"]) == {
        "left",
        "right",
    }


def test_observation_history_pads_bimanual_observation():
    history = ObservationHistory(2)
    observation = {"proprioception": {"controller_state_by_arm": {}}}
    history.initialize_with_padding(observation)
    assert len(history.window()) == 2


def test_policy_runner_initial_reset_contains_both_arms(tmp_path: Path):
    left_joints = [1.3922, -0.7209, 0.1799, -2.8098, 0.1171, 2.0371, 0.7]
    right_joints = [-1.5857, -0.7485, 0.0049, -2.8095, 0.0293, 2.0184, 0.9069]
    workspace = WorkspaceSettings(
        arms={
            "left": ArmEndpointSettings(arm_id="left", port=8092),
            "right": ArmEndpointSettings(arm_id="right", port=8093),
        },
        recording={"eval_root": tmp_path / "eval"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="bimanual_demo",
        initial_poses={
            "left": ArmInitialPoseSettings(
                eef_pose_xyz_rpy_deg=[0.0, 0.3, 0.4, -180.0, 0.0, 45.0],
                joint_positions_rad=left_joints,
            ),
            "right": ArmInitialPoseSettings(
                eef_pose_xyz_rpy_deg=[
                    0.0,
                    -0.3,
                    0.4,
                    -180.0,
                    0.0,
                    -145.0,
                ],
                joint_positions_rad=right_joints,
            ),
        },
        modality=ModalitySettings(
            rgb_cameras=[],
            gelsight_frame=False,
        ),
    )
    coordinator = FakeCoordinator()
    runner = BimanualPolicyRunner(
        workspace,
        inference,
        coordinator,
        calibration=object(),
        policy=NoopPolicy(),
    )

    runner.operator_reset_ready_pose()

    assert set(coordinator.resets[-1]) == {"left", "right"}
    assert coordinator.resets[-1]["left"].arm_id is None
    assert runner._initial_pose_completed
