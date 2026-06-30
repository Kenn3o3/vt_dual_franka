from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.episode_alignment import align_episode


def test_align_episode_uses_gelsight_frame_refs_without_markers(tmp_path: Path):
    episode_dir = tmp_path / "episode_0000"
    streams = episode_dir / "streams"
    streams.mkdir(parents=True)
    controller_records = []
    teleop_records = []
    gelsight_records = []
    for index in range(5):
        timestamp = 10.0 + index * 0.1
        controller_records.append(
            {
                "received_wall_time": timestamp,
                "state": {
                    "tcp_pose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                    "tcp_velocity": [0.0] * 6,
                    "tcp_wrench": [0.0] * 6,
                    "joint_positions": [0.0] * 7,
                    "joint_velocities": [0.0] * 7,
                    "gripper_width": 0.05,
                    "gripper_force": 0.0,
                },
            }
        )
        teleop_records.append(
            {
                "source_wall_time": timestamp + 0.05,
                "target_tcp": [0.2, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                "gripper_closed": False,
            }
        )
        gelsight_records.append(
            {
                "frame_path": "streams/gelsight_frames/chunk_000000.npz",
                "index_in_chunk": index,
                "captured_wall_time": timestamp,
            }
        )

    _write_jsonl(streams / "controller_state.jsonl", controller_records)
    _write_jsonl(streams / "teleop_commands.jsonl", teleop_records)
    _write_jsonl(streams / "gelsight_frames.jsonl", gelsight_records)

    output_path = align_episode(episode_dir, target_hz=10.0, overwrite=True)

    manifest = json.loads((episode_dir / "aligned_episode_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "vt_franka_aligned_episode_v2"
    assert "gelsight_frames" in manifest["streams_used"]
    with np.load(output_path, allow_pickle=True) as data:
        assert "gelsight_marker_locations" not in data.files
        assert data["gelsight_frame_paths"][0] == "streams/gelsight_frames/chunk_000000.npz"
        assert data["gelsight_frame_indices"][0] == 0


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
