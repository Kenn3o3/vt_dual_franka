from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from vt_franka_shared.models import ControllerState
from vt_franka_workspace.config import InferenceRuntimeSettings, ModalitySettings, WorkspaceSettings
from vt_franka_workspace.inference import ObservationAssembler, ObservationHistory, PolicyRunner
from vt_franka_workspace.inference.actions import ActionExecutor
from vt_franka_workspace.inference.policy_runner import GripperStatusEstimator
from vt_franka_workspace.policies.base import Policy
from vt_franka_workspace.recording import RunSessionManager
from vt_franka_workspace.runtime import eef_xyz_rpy_deg_to_tcp_pose


class FakeController:
    def __init__(self):
        self.tcp_targets = []
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

    def queue_tcp(self, target_tcp, source="policy_runner", target_duration_sec=None):
        self.tcp_targets.append((list(target_tcp), source, target_duration_sec))
        self.state = self.state.model_copy(update={"tcp_pose": list(target_tcp)})

    def move_gripper(self, width, velocity, force_limit, source="policy_runner"):
        self.gripper_moves.append((width, velocity, force_limit, source))

    def grasp_gripper(self, velocity, force_limit, source="policy_runner"):
        self.gripper_grasps.append((velocity, force_limit, source))


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


class TwoStepThenTerminatePolicy(Policy):
    def __init__(self):
        self.reset_calls = 0
        self.windows: list[list[dict[str, Any]]] = []
        self.started_windows: list[list[dict[str, Any]]] = []
        self.executed_chunks: list[list[dict[str, Any]]] = []
        self.target_1 = eef_xyz_rpy_deg_to_tcp_pose([0.41, 0.01, 0.31, 180.0, 0.0, 0.0])
        self.target_2 = eef_xyz_rpy_deg_to_tcp_pose([0.42, 0.02, 0.32, 180.0, 0.0, 0.0])
        self.extra_target = eef_xyz_rpy_deg_to_tcp_pose([0.43, 0.03, 0.33, 180.0, 0.0, 0.0])

    def reset(self):
        self.reset_calls += 1

    def start_episode(self, observation_window):
        self.started_windows.append(observation_window)

    def observe_executed_actions(self, actions):
        self.executed_chunks.append(actions)

    def predict(self, observation_window):
        self.windows.append(observation_window)
        if len(self.windows) > 1:
            return [{"terminate": True}]
        return [
            {"target_tcp": self.target_1, "target_duration_sec": 0.1},
            {"target_tcp": self.target_2, "target_duration_sec": 0.2},
            {"target_tcp": self.extra_target, "target_duration_sec": 0.3},
        ]


def test_observation_history_initial_padding_repeats_first_observation():
    history = ObservationHistory(3)
    first = {"proprioception": {"controller_state": {"tcp_pose": [1, 2, 3, 1, 0, 0, 0]}}}

    history.initialize_with_padding(first)
    first["proprioception"]["controller_state"]["tcp_pose"][0] = 99

    window = history.window()
    assert len(window) == 3
    assert [item["proprioception"]["controller_state"]["tcp_pose"][0] for item in window] == [1, 1, 1]


def test_observation_assembler_uses_semantic_keys(tmp_path: Path):
    state = ControllerState(tcp_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    assembler = ObservationAssembler(
        modality=ModalitySettings(proprioception=True),
        state_provider=lambda max_age_sec=None: state,
    )

    observation, recorded = assembler.assemble(tmp_path, 0)

    assert observation["proprioception"]["controller_state"]["tcp_pose"] == state.tcp_pose
    assert observation["images"] == {}
    assert observation["tactile"] == {}
    assert recorded["proprioception"]["controller_state"]["tcp_pose"] == state.tcp_pose


def test_policy_runner_executes_only_exe_horizon_and_records(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        obs_horizon=2,
        exe_horizon=2,
        control_hz=100.0,
        max_duration_sec=1.0,
        start_countdown_sec=0.0,
        initial_eef_pose_xyz_rpy_deg=None,
        modality=ModalitySettings(proprioception=True),
    )
    controller = FakeController()
    policy = TwoStepThenTerminatePolicy()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=policy)
    runner.sessions = RunSessionManager(tmp_path / "eval")
    run_dir = runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        image_format="jpg",
    )
    runner._initial_pose_completed = True

    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()

    assert policy.reset_calls == 1
    assert len(policy.started_windows) == 1
    assert len(policy.started_windows[0]) == 2
    assert len(controller.tcp_targets) == 2
    assert [target[2] for target in controller.tcp_targets] == [0.1, 0.2]
    assert len(policy.windows[0]) == 2
    assert len(policy.windows) == 2
    second_window_tcp_x = [
        item["proprioception"]["controller_state"]["tcp_pose"][0]
        for item in policy.windows[1]
    ]
    assert second_window_tcp_x == [policy.target_1[0], policy.target_2[0]]
    assert len(policy.executed_chunks) == 2
    assert len(policy.executed_chunks[0]) == 2
    episode_dir = run_dir / "episodes" / "episode_0000"
    lines = (episode_dir / "streams" / "policy_steps.jsonl").read_text(encoding="utf-8").strip().splitlines()
    events = [json.loads(line) for line in lines]
    assert any(event["phase"] == "initial_padding" for event in events)
    first_chunk = next(event for event in events if event.get("phase") == "policy_chunk")
    assert len(first_chunk["actions_returned"]) == 3
    assert len(first_chunk["actions_executed"]) == 2
    assert len(first_chunk["observations_after_actions"]) == 2
    assert [
        item["observation"]["proprioception"]["controller_state"]["tcp_pose"][0]
        for item in first_chunk["observations_after_actions"]
    ] == [policy.target_1[0], policy.target_2[0]]
    manifest = json.loads((episode_dir / "episode_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["termination_reason"] == "policy_terminate"


def test_action_executor_sends_gripper_state_changes_immediately():
    controller = FakeController()
    executor = ActionExecutor(controller)

    from vt_franka_workspace.inference.actions import Action

    executor.execute(Action(gripper_closed=True))
    executor.execute(Action(gripper_closed=True))
    executor.execute(Action(gripper_width=0.078))
    assert len(controller.gripper_grasps) == 1
    assert len(controller.gripper_moves) == 1

    executor.execute(Action(gripper_width=0.078))
    assert len(controller.gripper_moves) == 1


def test_gripper_status_estimator_derives_stability():
    estimator = GripperStatusEstimator(
        type(
            "Settings",
            (),
            {
                "gripper_stability_window": 3,
                "gripper_force_close_threshold": 15.0,
                "gripper_force_open_threshold": 5.0,
                "gripper_width_vis_precision": 0.001,
            },
        )()
    )
    for width in (0.078, 0.0782, 0.0781):
        estimator.update(ControllerState(gripper_width=width, gripper_force=1.0))
    status = estimator.get_status()
    assert status["left_gripper_stable_open"] is True
    assert status["left_gripper_stable_closed"] is False
