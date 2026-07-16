from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vt_dual_franka_shared.models import ControllerState
from vt_dual_franka_workspace.config import InferenceRuntimeSettings, ModalitySettings, RgbCameraSettings, WorkspaceSettings
from vt_dual_franka_workspace.inference import ObservationAssembler, ObservationHistory, PolicyRunner
from vt_dual_franka_workspace.inference.actions import ActionExecutor
from vt_dual_franka_workspace.inference.policy_runner import GripperStatusEstimator
from vt_dual_franka_workspace.operator import OperatorActionError
from vt_dual_franka_workspace.policies.base import Policy
from vt_dual_franka_workspace.runtime import LiveSampleBuffer, eef_xyz_rpy_deg_to_tcp_pose


TEST_HOME_JOINTS = [0.0] * 7


class FakeController:
    def __init__(self):
        self.tcp_targets = []
        self.reset_commands = []
        self.gripper_moves = []
        self.gripper_grasps = []
        self.events = []
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
        self.events.append(("tcp", list(target_tcp), source, target_duration_sec))
        self.state = self.state.model_copy(update={"tcp_pose": list(target_tcp)})

    def reset(self, command):
        self.reset_commands.append(command)
        self.events.append(("reset", command))
        if command.joint_positions is not None:
            self.state = self.state.model_copy(update={"joint_positions": list(command.joint_positions)})
        return {"status": "ok", "profile": command.profile}

    def move_gripper(self, width, velocity, force_limit, source="policy_runner", blocking=False):
        self.gripper_moves.append((width, velocity, force_limit, source, blocking))
        self.events.append(("gripper_width", width, velocity, force_limit, source, blocking))
        self.state = self.state.model_copy(update={"gripper_width": float(width), "gripper_force": 0.0})

    def grasp_gripper(self, velocity, force_limit, source="policy_runner", blocking=False):
        self.gripper_grasps.append((velocity, force_limit, source, blocking))
        self.events.append(("gripper_grasp", velocity, force_limit, source, blocking))
        self.state = self.state.model_copy(update={"gripper_width": 0.0, "gripper_force": float(force_limit)})


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


class SlowSecondPredictionPolicy(Policy):
    def __init__(self):
        self.windows: list[list[dict[str, Any]]] = []
        self.target_1 = eef_xyz_rpy_deg_to_tcp_pose([0.41, 0.01, 0.31, 180.0, 0.0, 0.0])
        self.target_2 = eef_xyz_rpy_deg_to_tcp_pose([0.42, 0.02, 0.32, 180.0, 0.0, 0.0])

    def reset(self):
        return None

    def start_episode(self, observation_window):
        return None

    def observe_executed_actions(self, actions):
        return None

    def predict(self, observation_window):
        self.windows.append(observation_window)
        if len(self.windows) > 1:
            time.sleep(0.05)
            return [{"terminate": True}]
        return [
            {"target_tcp": self.target_1, "target_duration_sec": 0.1},
            {"target_tcp": self.target_2, "target_duration_sec": 0.2},
        ]


class FourStepThenTerminatePolicy(Policy):
    def __init__(self):
        self.windows: list[list[dict[str, Any]]] = []
        self.executed_chunks: list[list[dict[str, Any]]] = []
        self.targets = [
            eef_xyz_rpy_deg_to_tcp_pose([0.41, 0.01, 0.31, 180.0, 0.0, 0.0]),
            eef_xyz_rpy_deg_to_tcp_pose([0.42, 0.02, 0.32, 180.0, 0.0, 0.0]),
            eef_xyz_rpy_deg_to_tcp_pose([0.43, 0.03, 0.33, 180.0, 0.0, 0.0]),
            eef_xyz_rpy_deg_to_tcp_pose([0.44, 0.04, 0.34, 180.0, 0.0, 0.0]),
        ]

    def reset(self):
        return None

    def start_episode(self, observation_window):
        return None

    def observe_executed_actions(self, actions):
        self.executed_chunks.append(actions)

    def predict(self, observation_window):
        self.windows.append(observation_window)
        if len(self.windows) > 1:
            return [{"terminate": True}]
        return [{"target_tcp": target, "target_duration_sec": 0.01} for target in self.targets]


class ModelInputPolicy(Policy):
    def __init__(self):
        self.built_windows: list[list[dict[str, Any]]] = []
        self.predicted_inputs: list[dict[str, np.ndarray]] = []
        self.target = eef_xyz_rpy_deg_to_tcp_pose([0.41, 0.01, 0.31, 180.0, 0.0, 0.0])

    def build_model_inputs(self, observation_window):
        self.built_windows.append(observation_window)
        return {
            "rgb_wrist": np.stack(
                [
                    np.full((4, 4, 3), 0.25, dtype=np.float32),
                    np.full((4, 4, 3), 0.50, dtype=np.float32),
                ],
                axis=0,
            ),
            "gelsight": np.stack(
                [
                    np.full((4, 4, 3), 0.75, dtype=np.float32),
                    np.full((4, 4, 3), 1.00, dtype=np.float32),
                ],
                axis=0,
            ),
            "qpos": np.zeros((2, 10), dtype=np.float32),
        }

    def predict_from_model_inputs(self, inputs):
        self.predicted_inputs.append(inputs)
        if len(self.predicted_inputs) > 1:
            return [{"terminate": True}]
        return [{"target_tcp": self.target, "target_duration_sec": 0.01}]

    def predict(self, observation_window):
        raise AssertionError("model_input_recording should use predict_from_model_inputs")


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


def test_observation_assembler_can_skip_gelsight_frame_disk_write(tmp_path: Path):
    state = ControllerState(tcp_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])
    gelsight_buffer = LiveSampleBuffer("gelsight_frame")
    gelsight_buffer.update(np.zeros((4, 5, 3), dtype=np.uint8), captured_wall_time=time.time())
    assembler = ObservationAssembler(
        modality=ModalitySettings(proprioception=True, gelsight_frame=True),
        state_provider=lambda max_age_sec=None: state,
        gelsight_frame_buffer=gelsight_buffer,
        record_gelsight_frames=False,
    )

    observation, recorded = assembler.assemble(tmp_path, 0)

    assert observation["tactile"]["gelsight_frame"]["image"].shape == (4, 5, 3)
    assert "frame_path" not in recorded["tactile"]["gelsight_frame"]
    assert not (tmp_path / "streams" / "gelsight_frame").exists()


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
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        modality=ModalitySettings(proprioception=True),
        eval={"enabled": False},
    )
    controller = FakeController()
    policy = TwoStepThenTerminatePolicy()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=policy)
    run_dir = runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.rgb_camera_buffers["wrist"] = LiveSampleBuffer("rgb_wrist")
    runner.gelsight_frame_buffer = LiveSampleBuffer("gelsight_frame")
    runner.rgb_camera_buffers["wrist"].update(np.full((4, 4, 3), 40, dtype=np.uint8), captured_wall_time=time.time())
    runner.gelsight_frame_buffer.update(np.full((4, 4, 3), 60, dtype=np.uint8), captured_wall_time=time.time())
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        rgb_camera_buffers=runner.rgb_camera_buffers,
        gelsight_frame_buffer=runner.gelsight_frame_buffer,
        image_format="jpg",
    )
    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()

    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()

    assert policy.reset_calls == 1
    assert len(policy.started_windows) == 1
    assert len(policy.started_windows[0]) == 2
    policy_tcp_targets = [target for target in controller.tcp_targets if target[1] == "policy_runner"]
    assert len(policy_tcp_targets) == 2
    assert [target[2] for target in policy_tcp_targets] == [0.1, 0.2]
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
    inference_events = [
        json.loads(line)
        for line in (episode_dir / "streams" / "policy_inference.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    first_inference = inference_events[0]
    assert len(first_inference["observation_window"]) == 2
    assert "raw_observation_window" not in first_inference
    assert len(first_inference["raw_policy_output"]) == 3
    assert len(first_inference["actions_returned"]) == 3
    assert first_inference["raw_action_vectors_10d"] == [None, None, None]
    manifest = json.loads((episode_dir / "episode_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["termination_reason"] == "policy_terminate"


def test_policy_runner_collects_after_each_action_and_reuses_latest_obs_horizon(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        obs_horizon=2,
        exe_horizon=4,
        control_hz=100.0,
        max_duration_sec=1.0,
        start_countdown_sec=0.0,
        initial_eef_pose_xyz_rpy_deg=None,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        modality=ModalitySettings(proprioception=True),
        eval={"enabled": False},
    )
    controller = FakeController()
    policy = FourStepThenTerminatePolicy()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=policy)
    run_dir = runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        image_format="jpg",
    )
    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()

    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()

    policy_tcp_targets = [target for target in controller.tcp_targets if target[1] == "policy_runner"]
    assert len(policy_tcp_targets) == 4
    assert len(policy.windows) == 2
    second_window_tcp_x = [
        item["proprioception"]["controller_state"]["tcp_pose"][0]
        for item in policy.windows[1]
    ]
    assert second_window_tcp_x == [policy.targets[2][0], policy.targets[3][0]]
    episode_dir = run_dir / "episodes" / "episode_0000"
    events = [
        json.loads(line)
        for line in (episode_dir / "streams" / "policy_steps.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    first_chunk = next(event for event in events if event.get("phase") == "policy_chunk")
    assert len(first_chunk["actions_executed"]) == 4
    assert [item["chunk_action_index"] for item in first_chunk["observations_after_actions"]] == [0, 1, 2, 3]
    assert [item["step_index"] for item in first_chunk["observations_after_actions"]] == [1, 2, 3, 4]
    assert [
        item["observation"]["proprioception"]["controller_state"]["tcp_pose"][0]
        for item in first_chunk["observations_after_actions"]
    ] == [target[0] for target in policy.targets]


def test_policy_runner_streams_last_valid_action_during_slow_inference(tmp_path: Path):
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
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        modality=ModalitySettings(proprioception=True),
        eval={"enabled": False},
    )
    controller = FakeController()
    policy = SlowSecondPredictionPolicy()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=policy)
    run_dir = runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        image_format="jpg",
    )
    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()

    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()

    hold_targets = [target for target in controller.tcp_targets if target[1] == "policy_runner_inference_hold"]
    assert any(np.allclose(target_tcp, policy.target_2) for target_tcp, _, _ in hold_targets)
    episode_dir = run_dir / "episodes" / "episode_0000"
    inference_events = [
        json.loads(line)
        for line in (episode_dir / "streams" / "policy_inference.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert any(event["timing"]["inference_hold_command_count"] > 0 for event in inference_events)


def test_policy_runner_records_exact_model_inputs(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        obs_horizon=2,
        exe_horizon=1,
        control_hz=100.0,
        max_duration_sec=1.0,
        start_countdown_sec=0.0,
        initial_eef_pose_xyz_rpy_deg=None,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        modality=ModalitySettings(proprioception=True),
        eval={"enabled": False},
        model_input_recording={
            "enabled": True,
            "streams": ["rgb_wrist", "gelsight"],
            "format": "png",
            "save_npz": True,
        },
    )
    controller = FakeController()
    policy = ModelInputPolicy()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=policy)
    run_dir = runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        image_format="jpg",
    )
    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()

    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()

    assert policy.predicted_inputs
    assert policy.predicted_inputs[0]["rgb_wrist"].shape == (2, 4, 4, 3)
    episode_dir = run_dir / "episodes" / "episode_0000"
    model_inputs_path = episode_dir / "streams" / "model_inputs.jsonl"
    records = [json.loads(line) for line in model_inputs_path.read_text(encoding="utf-8").splitlines()]
    first_record = records[0]
    assert first_record["streams"]["rgb_wrist"]["shape"] == [2, 4, 4, 3]
    assert first_record["streams"]["gelsight"]["shape"] == [2, 4, 4, 3]
    assert (episode_dir / first_record["streams"]["rgb_wrist"]["frame_paths"][0]).is_file()
    assert (episode_dir / first_record["streams"]["gelsight"]["frame_paths"][1]).is_file()
    assert (episode_dir / first_record["npz_path"]).is_file()
    with np.load(episode_dir / first_record["npz_path"]) as data:
        np.testing.assert_allclose(data["rgb_wrist"], policy.predicted_inputs[0]["rgb_wrist"])
        np.testing.assert_allclose(data["gelsight"], policy.predicted_inputs[0]["gelsight"])
        np.testing.assert_allclose(data["qpos"], policy.predicted_inputs[0]["qpos"])
    inference_records = [
        json.loads(line)
        for line in (episode_dir / "streams" / "policy_inference.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert inference_records[0]["model_input_record"]["npz_path"] == first_record["npz_path"]


def test_policy_runner_writes_clean_eval_videos_for_rgb_gelsight_and_third(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        eval={"enabled": True, "cameras": ["gelsight", "wrist"], "stream_cameras": ["third"], "video_hz": 10.0},
    )
    runner = PolicyRunner(workspace, inference, FakeController(), calibration=None, policy=TwoStepThenTerminatePolicy())
    runner.sessions.start_run("policy_test")
    episode_dir = runner.sessions.start_episode("episode_0000")

    class FakeRolloutRecorder:
        def __init__(self, stream_name, output_name, fps):
            self.stream_name = stream_name
            self.output_name = output_name
            self.fps = fps
            self.frames: list[tuple[Path, tuple[int, ...], float | None]] = []

        def record_frame(self, episode_dir, frame, *, event_time=None):
            self.frames.append((Path(episode_dir), tuple(np.asarray(frame).shape), event_time))

        def flush_episode(self, episode_dir):
            output_path = Path(episode_dir) / self.output_name
            output_path.write_bytes(b"fake mp4")
            return {"output_path": str(output_path), "dropped_due_to_backpressure": 0, "write_errors": 0}

        def close(self):
            return None

    runner.eval_video_recorders = {
        "gelsight": FakeRolloutRecorder("gelsight", "rollout_gelsight.mp4", 10.0),
        "wrist": FakeRolloutRecorder("rgb_wrist", "rollout_wrist.mp4", 10.0),
    }
    runner.eval_stream_video_recorders = {
        "third_person": FakeRolloutRecorder("rgb_third_person", "rollout_third_person.mp4", 10.0),
    }
    runner.rgb_camera_buffers["wrist"] = LiveSampleBuffer("rgb_wrist")
    runner.eval_rgb_camera_buffers["third_person"] = LiveSampleBuffer("rgb_third_person")
    runner.gelsight_frame_buffer = LiveSampleBuffer("gelsight")

    runner.rgb_camera_buffers["wrist"].update(np.full((4, 5, 3), 11, dtype=np.uint8), captured_wall_time=time.time())
    runner.eval_rgb_camera_buffers["third_person"].update(np.full((4, 5, 3), 22, dtype=np.uint8), captured_wall_time=time.time())
    runner.gelsight_frame_buffer.update(np.full((4, 5, 3), 33, dtype=np.uint8), captured_wall_time=time.time())

    runner._record_eval_video_frames(
        episode_dir=episode_dir,
        event_time=1.0,
        observation={
            "images": {"wrist": {"image": np.full((4, 5, 3), 11, dtype=np.uint8)}},
            "tactile": {"gelsight_frame": {"image": np.full((4, 5, 3), 33, dtype=np.uint8)}},
        },
    )

    assert len(runner.eval_video_recorders["wrist"].frames) == 1
    assert len(runner.eval_video_recorders["gelsight"].frames) == 1
    assert runner.eval_stream_video_recorders["third_person"].frames == []
    runner._write_eval_videos(episode_dir)
    assert (episode_dir / "rollout_third_person.mp4").is_file()
    assert (episode_dir / "rollout_gelsight.mp4").is_file()
    assert (episode_dir / "rollout_wrist.mp4").is_file()


def test_policy_runner_requires_outcome_mark_before_next_reset(tmp_path: Path):
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
        initial_eef_pose_xyz_rpy_deg=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        initial_move_duration_sec=0.01,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        initial_pose_settle_timeout_sec=0.2,
        initial_pose_settle_dwell_sec=0.0,
        modality=ModalitySettings(proprioception=True),
        eval={"enabled": False},
    )
    controller = FakeController()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=TwoStepThenTerminatePolicy())
    run_dir = runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        image_format="jpg",
    )
    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()

    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()

    assert runner._pending_outcome_episode_dir is not None
    with pytest.raises(OperatorActionError):
        runner.operator_reset_home_joints()
    with pytest.raises(OperatorActionError):
        runner.operator_reset_ready_pose()

    runner.operator_mark_episode_success()

    assert runner._pending_outcome_episode_dir is None
    assert (run_dir / "episode_outcomes.csv").read_text(encoding="utf-8").splitlines() == [
        "outcome,episode",
        "success,episode_0000",
    ]
    manifest = json.loads((run_dir / "episodes" / "episode_0000" / "episode_manifest.json").read_text(encoding="utf-8"))
    assert manifest["metadata"]["operator_outcome"] == "success"
    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()
    assert runner._initial_pose_completed is True


def test_policy_runner_discard_removes_episode_outcome_row(tmp_path: Path):
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
        initial_eef_pose_xyz_rpy_deg=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        initial_move_duration_sec=0.01,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        modality=ModalitySettings(proprioception=True),
        eval={"enabled": False},
    )
    controller = FakeController()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=TwoStepThenTerminatePolicy())
    run_dir = runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        image_format="jpg",
    )
    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()

    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()
    runner.operator_mark_episode_success()
    runner.operator_discard_latest_episode()

    assert not (run_dir / "episodes" / "episode_0000").exists()
    assert (run_dir / "episode_outcomes.csv").read_text(encoding="utf-8").splitlines() == [
        "outcome,episode",
    ]


def test_policy_runner_groups_eval_runs_by_policy(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(task_name="put_cup_on_plate", eval={"enabled": False})

    runner = PolicyRunner(workspace, inference, FakeController(), calibration=None, policy=TwoStepThenTerminatePolicy())

    assert runner.sessions.root_dir == tmp_path / "eval" / "put_cup_on_plate" / "twostepthenterminate"
    assert runner.run_name.count("_") == 1


def test_policy_runner_groups_visuotactile_eval_runs_by_task_and_model(tmp_path: Path):
    class FakeSettings:
        task_name = "pencil_insertion"
        model = "vista_so3"
        policy_name = "ignored_policy_name"

    class FakeVisuotactilePolicy(TwoStepThenTerminatePolicy):
        settings = FakeSettings()

    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(task_name="wrong_default", eval={"enabled": False})

    runner = PolicyRunner(workspace, inference, FakeController(), calibration=None, policy=FakeVisuotactilePolicy())

    assert runner.sessions.root_dir == tmp_path / "eval" / "pencil_insertion" / "vista_so3"


def test_policy_runner_reset_initial_pose_opens_gripper(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        initial_eef_pose_xyz_rpy_deg=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        initial_move_duration_sec=0.01,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        initial_pose_settle_timeout_sec=0.2,
        initial_pose_settle_dwell_sec=0.0,
        eval={"enabled": False},
    )
    controller = FakeController()
    controller.state = controller.state.model_copy(update={"gripper_width": 0.0, "gripper_force": 7.0})
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=TwoStepThenTerminatePolicy())
    runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)

    runner.operator_reset_ready_pose()

    assert runner._initial_pose_completed is True
    assert controller.reset_commands == []
    assert controller.gripper_moves[-1] == (
        workspace.teleop.max_gripper_width,
        workspace.teleop.gripper_velocity,
        workspace.teleop.grasp_force,
        "policy_runner_initial_pose",
        True,
    )
    assert controller.state.gripper_width == workspace.teleop.max_gripper_width


def test_policy_runner_home_joint_reset_is_optional_but_invalidates_initial_pose(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        initial_eef_pose_xyz_rpy_deg=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        initial_move_duration_sec=0.01,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        initial_pose_settle_timeout_sec=0.2,
        initial_pose_settle_dwell_sec=0.0,
        modality=ModalitySettings(proprioception=True),
        eval={"enabled": False},
    )
    controller = FakeController()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=TwoStepThenTerminatePolicy())
    runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        image_format="jpg",
    )

    runner.operator_reset_ready_pose()
    assert runner.get_operator_status()["ready"] is True

    runner.operator_reset_home_joints()
    status = runner.get_operator_status()

    assert status["ready"] is False
    assert "robot has not been moved to the policy initial pose with H" in status["reasons"]
    assert runner._initial_pose_completed is False
    assert runner._current_initial_pose is None
    with pytest.raises(OperatorActionError):
        runner.operator_start_episode()

    runner.operator_reset_ready_pose()
    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()

    assert controller.reset_commands[-1].source == "policy_runner_home_joints"


def test_policy_runner_forever_closed_always_waits_for_confirm_and_skips_open(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        initial_eef_pose_xyz_rpy_deg=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        initial_move_duration_sec=0.01,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        initial_pose_settle_timeout_sec=0.2,
        initial_pose_settle_dwell_sec=0.0,
        gripper_forever_closed=True,
        eval={"enabled": False},
    )
    controller = FakeController()
    controller.state = controller.state.model_copy(update={"gripper_width": 0.0, "gripper_force": 7.0})
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=TwoStepThenTerminatePolicy())
    runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)

    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()

    assert runner._pending_initial_gripper_close is True
    assert runner._initial_pose_completed is False
    assert controller.gripper_moves == []
    with pytest.raises(OperatorActionError):
        runner.operator_start_episode()

    runner.operator_confirm_gripper_closed()

    assert runner._initial_pose_completed is True
    assert controller.gripper_grasps[-1] == (
        workspace.teleop.gripper_velocity,
        workspace.teleop.grasp_force,
        "policy_runner_initial_gripper_close",
        True,
    )


def test_policy_runner_forever_closed_confirm_always_sends_grasp(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        initial_eef_pose_xyz_rpy_deg=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        initial_move_duration_sec=0.01,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        initial_pose_settle_timeout_sec=0.2,
        initial_pose_settle_dwell_sec=0.0,
        gripper_forever_closed=True,
        eval={"enabled": False},
    )
    controller = FakeController()
    controller.state = controller.state.model_copy(update={"gripper_width": 0.0, "gripper_force": 7.0})
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=TwoStepThenTerminatePolicy())
    runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)

    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()
    runner.operator_confirm_gripper_closed()

    assert len(controller.gripper_grasps) == 1
    runner.operator_confirm_gripper_closed()
    assert len(controller.gripper_grasps) == 2


def test_policy_runner_forever_closed_can_open_after_episode_before_marking_outcome(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        obs_horizon=2,
        exe_horizon=1,
        control_hz=100.0,
        max_duration_sec=1.0,
        start_countdown_sec=0.0,
        initial_eef_pose_xyz_rpy_deg=[0.4, 0.0, 0.3, 180.0, 0.0, 0.0],
        initial_move_duration_sec=0.01,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        initial_pose_settle_timeout_sec=0.2,
        initial_pose_settle_dwell_sec=0.0,
        gripper_forever_closed=True,
        modality=ModalitySettings(proprioception=True),
        eval={"enabled": False},
    )
    controller = FakeController()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=TwoStepThenTerminatePolicy())
    runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        image_format="jpg",
    )

    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()
    runner.operator_confirm_gripper_closed()
    initial_pose = runner._current_initial_pose
    initial_target_tcp = runner._current_initial_target_tcp

    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()

    assert runner._pending_outcome_episode_dir is not None
    assert runner._current_initial_pose is initial_pose
    assert runner._current_initial_target_tcp == initial_target_tcp

    runner.operator_open_gripper()

    assert runner._pending_initial_gripper_close is True
    assert runner._initial_pose_completed is False
    assert controller.gripper_moves[-1] == (
        workspace.teleop.max_gripper_width,
        workspace.teleop.gripper_velocity,
        workspace.teleop.grasp_force,
        "policy_runner_gripper_adjustment_open",
        True,
    )
    status_after_open = runner.get_operator_status()
    assert status_after_open["allowed_actions"]["confirm_gripper_closed"] is True
    assert status_after_open["allowed_actions"]["open_gripper"] is True
    assert status_after_open["allowed_actions"]["start"] is False

    runner.operator_confirm_gripper_closed()

    assert runner._pending_initial_gripper_close is False
    assert runner._initial_pose_completed is True
    assert controller.gripper_grasps[-1] == (
        workspace.teleop.gripper_velocity,
        workspace.teleop.grasp_force,
        "policy_runner_initial_gripper_close",
        True,
    )
    with pytest.raises(OperatorActionError):
        runner.operator_start_episode()

    runner.operator_mark_episode_success()
    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()
    runner.operator_confirm_gripper_closed()
    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()


def test_policy_runner_forever_closed_records_executed_actions_as_closed(tmp_path: Path):
    class OpenThenTerminatePolicy(Policy):
        def __init__(self):
            self.executed_chunks = []

        def predict(self, observation_window):
            del observation_window
            return [{"gripper_width": 0.078, "terminate": True}]

        def observe_executed_actions(self, actions):
            self.executed_chunks.append(actions)

    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        obs_horizon=2,
        exe_horizon=1,
        control_hz=100.0,
        max_duration_sec=1.0,
        start_countdown_sec=0.0,
        initial_eef_pose_xyz_rpy_deg=None,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        gripper_forever_closed=True,
        modality=ModalitySettings(proprioception=True),
        eval={"enabled": False},
    )
    controller = FakeController()
    policy = OpenThenTerminatePolicy()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=policy)
    run_dir = runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        image_format="jpg",
    )
    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()

    runner.operator_start_episode()
    runner._wait_for_episode_finish_locked()

    assert controller.gripper_moves == []
    assert len(controller.gripper_grasps) == 1
    assert policy.executed_chunks[0][0]["gripper_closed"] is True
    assert "gripper_width" not in policy.executed_chunks[0][0]
    episode_dir = run_dir / "episodes" / "episode_0000"
    events = [
        json.loads(line)
        for line in (episode_dir / "streams" / "policy_steps.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    first_chunk = next(event for event in events if event.get("phase") == "policy_chunk")
    assert first_chunk["actions_returned"][0]["gripper_width"] == 0.078
    assert first_chunk["actions_executed"][0]["gripper_closed"] is True
    assert "gripper_width" not in first_chunk["actions_executed"][0]


def test_policy_runner_accepts_realistic_open_gripper_width(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(eval={"enabled": False})
    controller = FakeController()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=TwoStepThenTerminatePolicy())
    runner.state_monitor = FakeStateMonitor(controller)
    controller.state = controller.state.model_copy(update={"gripper_width": 0.07384854555130005})

    runner._wait_for_gripper_width_locked(target_width=0.078, tolerance_m=0.006, timeout_sec=0.01)


def test_policy_runner_serves_live_wrist_preview_before_start_and_stops_while_active(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="policy_test",
        start_countdown_sec=0.0,
        initial_eef_pose_xyz_rpy_deg=None,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        modality=ModalitySettings(proprioception=True, rgb_cameras=["wrist"]),
        eval={"enabled": False},
    )
    controller = FakeController()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=TwoStepThenTerminatePolicy())
    runner.sessions.start_run("policy_test")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.rgb_camera_buffers["wrist"] = LiveSampleBuffer("rgb_wrist")
    runner.rgb_camera_buffers["wrist"].update(
        np.zeros((6, 7, 3), dtype=np.uint8),
        metadata={"camera_name": "wrist"},
        captured_wall_time=time.time(),
    )
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        rgb_camera_buffers=runner.rgb_camera_buffers,
        image_format="jpg",
    )
    status = runner.get_operator_status()

    assert status["ready"] is False
    assert status["preview"]["role"] == "wrist"
    assert status["preview"]["streaming"] is True
    snapshot = runner.get_operator_snapshot("wrist")
    assert snapshot is not None
    assert snapshot.image.shape == (6, 7, 3)

    runner.rgb_camera_buffers["wrist"].update(
        np.ones((6, 7, 3), dtype=np.uint8),
        metadata={"camera_name": "wrist"},
        captured_wall_time=time.time(),
    )
    updated_snapshot = runner.get_operator_snapshot("wrist")
    assert updated_snapshot is not None
    assert int(updated_snapshot.image[0, 0, 0]) == 1

    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()
    assert runner.get_operator_status()["ready"] is True
    runner.operator_start_episode()
    active_status = runner.get_operator_status()
    assert active_status["preview"]["streaming"] is False
    assert runner.get_operator_snapshot("wrist") is None
    runner._wait_for_episode_finish_locked()


def test_policy_runner_starts_eval_camera_without_policy_camera(monkeypatch, tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        modality=ModalitySettings(proprioception=True, rgb_cameras=[]),
        eval={"enabled": True, "cameras": [], "stream_cameras": ["third"], "video_hz": 10.0},
        rgb_cameras={"third_person": RgbCameraSettings(stream_name="rgb_third_person")},
    )
    runner = PolicyRunner(workspace, inference, FakeController(), calibration=None, policy=TwoStepThenTerminatePolicy())
    runner.state_monitor = FakeStateMonitor(FakeController())
    started = {}

    class FakeRecorder:
        def __init__(self, spec, recorder, live_buffer, quest_publisher, image_format, **kwargs):
            del quest_publisher, image_format
            started["role"] = spec.role
            started["stream_name"] = spec.stream_name
            started["recorder"] = recorder
            started["live_buffer"] = live_buffer
            started["stream_video_recorder"] = kwargs.get("stream_video_recorder")

        def run(self, stop_event=None):
            del stop_event

    def fake_start_worker(workers, name, target, required, startup_delay_sec=0.2):
        del startup_delay_sec
        target(None)
        workers[name] = type("FakeWorker", (), {"required": required, "error": None, "is_alive": lambda self: False})()

    monkeypatch.setattr("vt_dual_franka_workspace.inference.policy_runner.build_rgb_camera_recorder", FakeRecorder)
    monkeypatch.setattr("vt_dual_franka_workspace.inference.policy_runner.start_thread_worker", fake_start_worker)

    runner._start_workers()

    assert runner.rgb_camera_buffers == {}
    assert list(runner.eval_rgb_camera_buffers) == ["third_person"]
    assert runner.eval_camera_stream_names == {"third_person": "rgb_third_person"}
    assert started["role"] == "third_person"
    assert started["recorder"] is None
    assert runner.eval_video_recorders == {}
    assert runner.eval_stream_video_recorders["third_person"].__class__.__name__ == "AsyncStreamVideoRecorder"
    assert started["stream_video_recorder"] is runner.eval_stream_video_recorders["third_person"]


def test_policy_runner_does_not_preview_third_person_when_wrist_is_not_running(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={"eval_root": tmp_path / "eval", "collect_root": tmp_path / "collect", "image_format": "jpg"},
        operator_ui={"enabled": False},
    )
    inference = InferenceRuntimeSettings(
        task_name="state_only",
        initial_eef_pose_xyz_rpy_deg=None,
        home_joint_positions_rad=TEST_HOME_JOINTS,
        home_joint_duration_sec=0.01,
        home_joint_settle_timeout_sec=0.01,
        modality=ModalitySettings(proprioception=True, rgb_cameras=[]),
        eval={"enabled": True, "cameras": [], "stream_cameras": ["third"], "video_hz": 10.0},
    )
    controller = FakeController()
    runner = PolicyRunner(workspace, inference, controller, calibration=None, policy=TwoStepThenTerminatePolicy())
    runner.sessions.start_run("state_only")
    runner.state_monitor = FakeStateMonitor(controller)
    runner.eval_rgb_camera_buffers["third_person"] = LiveSampleBuffer("rgb_third_person")
    runner.eval_rgb_camera_buffers["third_person"].update(
        np.zeros((6, 7, 3), dtype=np.uint8),
        metadata={"camera_name": "third_person"},
        captured_wall_time=time.time(),
    )
    runner.assembler = ObservationAssembler(
        modality=inference.modality,
        state_provider=lambda max_age_sec=None: controller.state,
        image_format="jpg",
    )
    runner.operator_reset_home_joints()
    runner.operator_reset_ready_pose()

    status = runner.get_operator_status()

    assert status["ready"] is True
    assert status["preview"]["role"] == "wrist"
    assert status["preview"]["streaming"] is False
    assert runner.get_operator_snapshot("third_person") is None


def test_action_executor_sends_gripper_state_changes_immediately():
    controller = FakeController()
    executor = ActionExecutor(controller)

    from vt_dual_franka_workspace.inference.actions import Action

    executor.execute(Action(gripper_closed=True))
    executor.execute(Action(gripper_closed=True))
    executor.execute(Action(gripper_width=0.078))
    assert len(controller.gripper_grasps) == 1
    assert len(controller.gripper_moves) == 1

    executor.execute(Action(gripper_width=0.078))
    assert len(controller.gripper_moves) == 1
    assert controller.gripper_grasps[0][3] is True
    assert controller.gripper_moves[0][4] is True


def test_action_executor_blocks_gripper_transition_before_tcp_waypoint():
    controller = FakeController()
    executor = ActionExecutor(controller)

    from vt_dual_franka_workspace.inference.actions import Action

    target = eef_xyz_rpy_deg_to_tcp_pose([0.41, 0.01, 0.31, 180.0, 0.0, 0.0])
    executor.execute(Action(target_tcp=target, target_duration_sec=0.1, gripper_closed=True))

    assert controller.events[0][0] == "gripper_grasp"
    assert controller.events[0][-1] is True
    assert controller.events[1][0] == "tcp"


def test_action_executor_force_gripper_closed_suppresses_open_width():
    controller = FakeController()
    executor = ActionExecutor(controller, force_gripper_closed=True)

    from vt_dual_franka_workspace.inference.actions import Action

    action = Action(gripper_width=0.078)
    executed = executor.normalize_for_execution(action)
    executor.execute_normalized(executed)

    assert executed.gripper_closed is True
    assert executed.gripper_width is None
    assert len(controller.gripper_grasps) == 1
    assert controller.gripper_moves == []


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
