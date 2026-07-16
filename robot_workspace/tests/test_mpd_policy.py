from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vt_dual_franka_workspace.config import InferenceRuntimeSettings, PolicyConfig, WorkspaceSettings
from vt_dual_franka_workspace.policies import resolve_policy
from vt_dual_franka_workspace.policies.mpd.config import (
    ACTION_CONVENTION_OPEN_FRACTION,
    checkpoint_run_dir,
    default_checkpoint_path,
    get_policy_spec,
    normalize_algorithm_name,
)
from vt_dual_franka_workspace.policies.mpd.data import PrepareMPDDatasetConfig, prepare_mpd_dataset
from vt_dual_franka_workspace.policies.mpd.math import pose7d_and_gripper_to_tcp_state, tcp_state_to_pose7d_and_gripper
from vt_dual_franka_workspace.policies.mpd.policy import MPDPolicy, MPDRuntimeSpec
from vt_dual_franka_workspace.policies.mpd.smooth_gripper_dataset import (
    SmoothGripperDatasetConfig,
    build_transition_ramp,
    smooth_gripper_dataset,
)
from vt_dual_franka_workspace.policies.mpd.train import build_train_command, build_train_config_from_workspace


class FakeMPDBackend:
    def __init__(self, required_history_keys=("action", "action_vel"), action_convention="tcp_xyz_rot6d_gripper_closedness"):
        self.runtime_spec = MPDRuntimeSpec(
            obs_horizon=3,
            prediction_horizon=2,
            action_dim=10,
            observation_keys=("agent_pos",),
            required_history_keys=tuple(required_history_keys),
            dt=0.1,
        )
        self.action_convention = action_convention
        self.inputs: list[dict[str, np.ndarray]] = []

    def predict_action_chunk(self, inputs):
        self.inputs.append(inputs)
        return np.stack([inputs["agent_pos"][-1], inputs["agent_pos"][-1]], axis=0)

    def close(self):
        return None


def test_mpd_algorithm_mapping_rejects_retired_variants():
    assert normalize_algorithm_name("prodmp_diffusion") == "mpd"
    assert normalize_algorithm_name("freqpolicy_official") == "freqpolicy"
    assert get_policy_spec("motif").upstream_config_name("put_cup_on_plate").endswith("train_motif_transformer")
    assert get_policy_spec("freqpolicy").upstream_config_name("put_cup_on_plate").endswith("train_freqpolicy")
    with pytest.raises(ValueError):
        normalize_algorithm_name("prodmp_fm")
    with pytest.raises(ValueError):
        normalize_algorithm_name("motif_fm")


def test_mpd_checkpoint_path_layout(tmp_path: Path):
    workspace = WorkspaceSettings(recording={"checkpoints_root": tmp_path / "checkpoints"})

    run_dir = checkpoint_run_dir(workspace, task_name="put_cup_on_plate", algorithm="dp")
    checkpoint = default_checkpoint_path(workspace, task_name="put_cup_on_plate", algorithm="dp")

    assert run_dir == tmp_path / "checkpoints" / "put_cup_on_plate" / "mpd" / "dp" / "dp_state"
    assert checkpoint == run_dir / "best_model.pth"

    freqpolicy_run_dir = checkpoint_run_dir(workspace, task_name="put_cup_on_plate", algorithm="freqpolicy")
    assert freqpolicy_run_dir == tmp_path / "checkpoints" / "put_cup_on_plate" / "mpd" / "freqpolicy" / "freqpolicy_state"


def test_mpd_vector_round_trip():
    pose = [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    vector = pose7d_and_gripper_to_tcp_state(pose, 0.75)
    restored_pose, restored_gripper = tcp_state_to_pose7d_and_gripper(vector)

    assert vector.shape == (10,)
    assert np.allclose(restored_pose, pose)
    assert restored_gripper == pytest.approx(0.75)


def test_mpd_policy_requires_explicit_start_episode_for_action_history():
    workspace = WorkspaceSettings()
    inference = InferenceRuntimeSettings(obs_horizon=3, exe_horizon=2, control_hz=10.0)
    policy_config = PolicyConfig(type="mpd", config={"algorithm": "mpd", "task_name": "put_cup_on_plate"})
    settings, checkpoint = policy_config_to_settings(policy_config, workspace, inference)
    policy = MPDPolicy(settings, checkpoint, inference, workspace, backend=FakeMPDBackend())
    window = [_observation([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])] * 3

    with pytest.raises(RuntimeError):
        policy.predict(window)

    policy.start_episode(window)
    actions = policy.predict(window)

    assert len(actions) == 2
    assert "action" in policy.backend.inputs[-1]
    assert policy.backend.inputs[-1]["action"].shape == (3, 10)


def test_mpd_policy_records_executed_actions_for_next_prediction():
    workspace = WorkspaceSettings()
    inference = InferenceRuntimeSettings(obs_horizon=3, exe_horizon=2, control_hz=10.0)
    policy_config = PolicyConfig(type="mpd", config={"algorithm": "mpd", "task_name": "put_cup_on_plate"})
    settings, checkpoint = policy_config_to_settings(policy_config, workspace, inference)
    backend = FakeMPDBackend()
    policy = MPDPolicy(settings, checkpoint, inference, workspace, backend=backend)
    window = [_observation([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])] * 3
    policy.start_episode(window)
    policy.observe_executed_actions(
        [
            {
                "target_tcp": [0.4, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                "gripper_width": 0.078,
            }
        ]
    )

    policy.predict(window)

    pose, _ = tcp_state_to_pose7d_and_gripper(backend.inputs[-1]["action"][-1])
    assert np.allclose(pose[:3], [0.4, 0.2, 0.3])


def test_mpd_policy_uses_temporary_gripper_hysteresis():
    workspace = WorkspaceSettings()
    inference = InferenceRuntimeSettings(obs_horizon=3, exe_horizon=2, control_hz=10.0)
    policy_config = PolicyConfig(type="mpd", config={"algorithm": "fm", "task_name": "put_cup_on_plate"})
    settings, checkpoint = policy_config_to_settings(policy_config, workspace, inference)
    policy = MPDPolicy(settings, checkpoint, inference, workspace, backend=FakeMPDBackend(required_history_keys=()))

    assert policy._gripper_mode_from_closedness(0.2) == "open"
    assert policy._gripper_mode_from_closedness(0.6) == "close"
    assert policy._gripper_mode_from_closedness(0.4) == "close"
    assert policy._gripper_mode_from_closedness(0.2) == "open"


def test_mpd_policy_locks_gripper_mode_after_switch():
    workspace = WorkspaceSettings()
    inference = InferenceRuntimeSettings(obs_horizon=3, exe_horizon=2, control_hz=10.0)
    policy_config = PolicyConfig(
        type="mpd",
        config={
            "algorithm": "motif",
            "task_name": "put_cup_on_plate",
            "gripper_switch_lockout_actions": 3,
        },
    )
    settings, checkpoint = policy_config_to_settings(policy_config, workspace, inference)
    policy = MPDPolicy(settings, checkpoint, inference, workspace, backend=FakeMPDBackend(required_history_keys=()))
    open_state = pose7d_and_gripper_to_tcp_state([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], 0.0)
    close_state = pose7d_and_gripper_to_tcp_state([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0], 1.0)

    assert policy._row_to_action(open_state)["gripper_width"] == pytest.approx(0.078)
    assert policy._row_to_action(close_state)["gripper_closed"] is True
    assert policy._row_to_action(open_state)["gripper_closed"] is True
    assert policy._row_to_action(open_state)["gripper_closed"] is True
    assert policy._row_to_action(open_state)["gripper_closed"] is True
    assert policy._row_to_action(open_state)["gripper_width"] == pytest.approx(0.078)


def test_mpd_policy_supports_open_fraction_gripper_convention():
    workspace = WorkspaceSettings()
    inference = InferenceRuntimeSettings(obs_horizon=3, exe_horizon=2, control_hz=10.0)
    policy_config = PolicyConfig(type="mpd", config={"algorithm": "motif", "task_name": "put_cup_on_plate"})
    settings, checkpoint = policy_config_to_settings(policy_config, workspace, inference)
    policy = MPDPolicy(
        settings,
        checkpoint,
        inference,
        workspace,
        backend=FakeMPDBackend(required_history_keys=(), action_convention=ACTION_CONVENTION_OPEN_FRACTION),
    )

    open_state = policy._state_from_observation(_observation([0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]))
    closed_state = policy._state_from_observation(
        {
            "proprioception": {
                "controller_state": {
                    "tcp_pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                    "gripper_width": 0.0,
                }
            },
            "images": {},
            "tactile": {},
        }
    )

    assert open_state[9] == pytest.approx(1.0)
    assert closed_state[9] == pytest.approx(0.0)
    assert policy._row_to_action(np.r_[open_state[:9], 1.0])["gripper_width"] == pytest.approx(0.078)
    assert policy._row_to_action(np.r_[open_state[:9], 0.5])["gripper_closed"] is True
    assert policy._row_to_action(np.r_[open_state[:9], 0.0])["gripper_closed"] is True
    assert policy._row_to_action(np.r_[open_state[:9], 0.5])["gripper_width"] == pytest.approx(0.078)


def test_prepare_mpd_dataset_from_raw_streams(tmp_path: Path):
    raw_run = tmp_path / "collect" / "put_cup_on_plate"
    for episode_index in range(2):
        episode_dir = raw_run / "episodes" / f"episode_{episode_index:04d}" / "streams"
        episode_dir.mkdir(parents=True)
        controller_records = []
        teleop_records = []
        for step in range(20):
            t = 100.0 + episode_index * 10.0 + step * 0.05
            pose = [0.1 + 0.001 * step, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
            controller_records.append(
                {
                    "state": {
                        "wall_time": t,
                        "tcp_pose": pose,
                        "gripper_width": 0.078,
                    }
                }
            )
            teleop_records.append(
                {
                    "source_wall_time": t + 0.02,
                    "target_tcp": [pose[0] + 0.01, pose[1], pose[2], 1.0, 0.0, 0.0, 0.0],
                    "gripper_closed": False,
                }
            )
        (episode_dir / "controller_state.jsonl").write_text(
            "\n".join(json.dumps(record) for record in controller_records) + "\n",
            encoding="utf-8",
        )
        (episode_dir / "teleop_commands.jsonl").write_text(
            "\n".join(json.dumps(record) for record in teleop_records) + "\n",
            encoding="utf-8",
        )

    result = prepare_mpd_dataset(
        PrepareMPDDatasetConfig(
            task_name="put_cup_on_plate",
            raw_run_dir=raw_run,
            output_dir=tmp_path / "prepared" / "mpd" / "put_cup_on_plate" / "vt_franka_mpd_v1",
            target_hz=10.0,
            min_steps=5,
        )
    )

    assert result.train_episodes == 1
    assert result.val_episodes == 1
    demo = result.output_dir / "train" / "demo_000"
    assert np.load(demo / "agent_pos.npz")["arr_0"].shape[1] == 10
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["action_alignment"] == "causal_future_command"


def test_build_transition_ramp_sets_switch_threshold_at_transition():
    smoothed = build_transition_ramp(
        np.asarray([1, 1, 1, 0, 0, 0, 0], dtype=np.float64),
        switch_threshold=0.5,
        pre_switch_ramp_steps=2,
        post_switch_ramp_steps=2,
    )

    assert smoothed.tolist() == pytest.approx([1.0, 5.0 / 6.0, 2.0 / 3.0, 0.5, 1.0 / 3.0, 1.0 / 6.0, 0.0])


def test_smooth_gripper_dataset_writes_open_fraction_variant(tmp_path: Path):
    source = tmp_path / "prepared" / "mpd" / "put_cup_on_plate" / "vt_franka_mpd_v1"
    for split_name in ("train", "val"):
        demo_dir = source / split_name / "demo_000"
        demo_dir.mkdir(parents=True)
        values = np.zeros((7, 10), dtype=np.float32)
        values[:, 3] = 1.0
        values[:, 9] = np.asarray([0, 0, 0, 1, 1, 1, 1], dtype=np.float32)
        np.savez_compressed(demo_dir / "agent_pos.npz", values)
        np.savez_compressed(demo_dir / "agent_vel.npz", values * 0.0)
        np.savez_compressed(demo_dir / "action.npz", values)
        np.savez_compressed(demo_dir / "action_vel.npz", values * 0.0)
        np.savez_compressed(demo_dir / "timestamps.npz", np.arange(7, dtype=np.float64) * 0.1)
        (demo_dir / "dataset_manifest.json").write_text(json.dumps({"num_steps": 7}), encoding="utf-8")
    (source / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "vt_franka_mpd_v1",
                "dt": 0.1,
                "target_hz": 10.0,
                "action_convention": "tcp_xyz_rot6d_gripper_closedness",
            }
        ),
        encoding="utf-8",
    )
    np.savez_compressed(source / "scaler_values.npz", action_min=np.zeros(10, dtype=np.float32))

    output = tmp_path / "prepared" / "mpd" / "put_cup_on_plate" / "smooth"
    result = smooth_gripper_dataset(
        SmoothGripperDatasetConfig(
            source_dataset_dir=source,
            output_dataset_dir=output,
            pre_switch_ramp_steps=2,
            post_switch_ramp_steps=2,
            plot_first_demo=False,
        )
    )

    action = np.load(output / "train" / "demo_000" / "action.npz")["arr_0"]
    agent_pos = np.load(output / "train" / "demo_000" / "agent_pos.npz")["arr_0"]
    action_vel = np.load(output / "train" / "demo_000" / "action_vel.npz")["arr_0"]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["action_convention"] == ACTION_CONVENTION_OPEN_FRACTION
    assert action[:, 9].tolist() == pytest.approx([1.0, 5.0 / 6.0, 2.0 / 3.0, 0.5, 1.0 / 3.0, 1.0 / 6.0, 0.0])
    assert agent_pos[:, 9].tolist() == pytest.approx([1, 1, 1, 0, 0, 0, 0])
    assert action_vel[3, 9] == pytest.approx((0.5 - 2.0 / 3.0) / 0.1)
    assert (output / "scaler_values.npz").exists()


def test_train_command_uses_checkpoint_root_and_disables_sim_eval(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )
    prepared = tmp_path / "prepared" / "mpd" / "put_cup_on_plate" / "vt_franka_mpd_v1"
    prepared.mkdir(parents=True)
    (prepared / "dataset_manifest.json").write_text(json.dumps({"dt": 0.1}), encoding="utf-8")
    config = build_train_config_from_workspace(
        workspace,
        task_name="put_cup_on_plate",
        algorithm="dp",
        prepared_dataset_dir=prepared,
        python="python",
        epochs=1,
    )

    command = build_train_command(config)

    assert "workspace_config._target_=movement_primitive_diffusion.workspaces.dummy_workspace.DummyWorkspace" in command
    assert "eval_in_env_after_epochs=0" in command
    assert f"+train_trajectory_dir={(prepared / 'train').resolve()}" in command
    assert f"+val_trajectory_dir={(prepared / 'val').resolve()}" in command
    assert f"hydra.run.dir={(tmp_path / 'checkpoints' / 'put_cup_on_plate' / 'mpd' / 'dp' / 'dp_state').resolve()}" in command


def test_train_command_supports_freqpolicy(tmp_path: Path):
    workspace = WorkspaceSettings(
        recording={
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )
    prepared = tmp_path / "prepared" / "mpd" / "put_cup_on_plate" / "vt_franka_mpd_v1"
    prepared.mkdir(parents=True)
    (prepared / "dataset_manifest.json").write_text(json.dumps({"dt": 0.1}), encoding="utf-8")
    config = build_train_config_from_workspace(
        workspace,
        task_name="put_cup_on_plate",
        algorithm="freqpolicy",
        prepared_dataset_dir=prepared,
        python="python",
        epochs=1,
    )

    command = build_train_command(config)

    assert "--config-name=experiments/put_cup_on_plate/train_freqpolicy" in command
    assert "method_name=freqpolicy" in command
    assert f"hydra.run.dir={(tmp_path / 'checkpoints' / 'put_cup_on_plate' / 'mpd' / 'freqpolicy' / 'freqpolicy_state').resolve()}" in command


def test_registry_resolves_mpd_policy(monkeypatch):
    calls = {}

    class FakePolicy:
        @classmethod
        def from_config(cls, policy_config, inference_config, workspace):
            calls["args"] = (policy_config, inference_config, workspace)
            return "mpd-policy"

    monkeypatch.setattr("vt_dual_franka_workspace.policies.mpd.policy.MPDPolicy", FakePolicy)

    policy = resolve_policy(
        PolicyConfig(type="mpd", config={"algorithm": "dp", "task_name": "put_cup_on_plate"}),
        InferenceRuntimeSettings(),
        WorkspaceSettings(),
    )

    assert policy == "mpd-policy"


def policy_config_to_settings(policy_config: PolicyConfig, workspace: WorkspaceSettings, inference: InferenceRuntimeSettings):
    from vt_dual_franka_workspace.policies.mpd.config import MPDPolicySettings

    return MPDPolicySettings.from_policy_config(policy_config, workspace, fallback_task_name=inference.task_name)


def _observation(tcp_pose: list[float]) -> dict[str, Any]:
    return {
        "proprioception": {
            "controller_state": {
                "tcp_pose": tcp_pose,
                "gripper_width": 0.078,
            }
        },
        "images": {},
        "tactile": {},
    }
