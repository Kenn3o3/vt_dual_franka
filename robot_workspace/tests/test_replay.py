from __future__ import annotations

from pathlib import Path

import numpy as np

from vt_franka_workspace.config import InferenceRuntimeSettings, PolicyConfig, WorkspaceSettings
from vt_franka_workspace.policies import resolve_policy
from vt_franka_workspace.policies.replay.policy import ReplayPolicy, load_replay_episode


def write_aligned_episode(episode_dir: Path) -> Path:
    episode_dir.mkdir(parents=True)
    path = episode_dir / "aligned_episode.npz"
    np.savez(
        path,
        timestamps=np.asarray([1.0, 1.1, 1.2], dtype=np.float64),
        teleop_target_tcp=np.asarray(
            [
                [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [0.2, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                [0.3, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
        teleop_gripper_closed=np.asarray([False, True, True], dtype=bool),
    )
    return path


def test_load_replay_episode_reads_aligned_npz(tmp_path: Path):
    episode_dir = tmp_path / "episode_0000"
    write_aligned_episode(episode_dir)

    episode = load_replay_episode(episode_dir)

    assert episode.target_tcp.shape == (3, 7)
    assert episode.gripper_closed.tolist() == [False, True, True]
    assert episode.timestamps.tolist() == [1.0, 1.1, 1.2]


def test_replay_policy_returns_unified_action_dicts(tmp_path: Path):
    episode_dir = tmp_path / "episode_0000"
    write_aligned_episode(episode_dir)
    policy = ReplayPolicy(load_replay_episode(episode_dir), chunk_size=2, gripper_open_width=0.078)

    actions0 = policy.predict([])
    actions1 = policy.predict([])

    assert actions0[0]["target_tcp"] == [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    assert actions0[0]["gripper_width"] == 0.078
    assert actions0[1]["gripper_closed"] is True
    assert actions1[0]["target_tcp"] == [0.3, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    assert actions1[0]["terminate"] is True


def test_registry_resolves_replay_policy(tmp_path: Path):
    episode_dir = tmp_path / "episode_0000"
    write_aligned_episode(episode_dir)
    policy_config = PolicyConfig(type="replay", config={"episode_dir": str(episode_dir), "chunk_size": 1})

    policy = resolve_policy(policy_config, InferenceRuntimeSettings(), WorkspaceSettings())

    assert isinstance(policy, ReplayPolicy)
