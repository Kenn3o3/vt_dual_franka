from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from vt_franka_workspace.datasets import MakeDatasetConfig, make_common_dataset
from vt_franka_workspace.recording.image_io import read_rgb_image, write_rgb_jpeg


def test_make_common_dataset_causal_aligns_and_copies_standard_images(tmp_path: Path) -> None:
    collect_task_dir = tmp_path / "data" / "collect" / "usb_insertion"
    episode_dir = collect_task_dir / "episodes" / "episode_0000"
    _write_standard_episode_streams(episode_dir)

    result = make_common_dataset(
        MakeDatasetConfig(
            collect_task_dir=collect_task_dir,
            dataset_name="real_640x480_v1",
            target_hz=10.0,
        )
    )

    assert result.episode_count == 1
    assert result.step_count == 4
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "vt_franka_common_dataset_v1"
    assert manifest["streams"]["tactile_left"]["shape"] == [480, 640, 3]
    assert manifest["standardized_image_jpeg_quality"] == 90
    steps_path = result.output_dir / "episodes" / "episode_0000" / "steps.jsonl"
    steps = [json.loads(line) for line in steps_path.read_text(encoding="utf-8").splitlines()]
    assert steps[0]["images"]["rgb_wrist"] == "episodes/episode_0000/images/rgb_wrist/000000.jpg"
    assert steps[0]["source"]["rgb_wrist"]["frame_index"] == 0
    assert steps[0]["source"]["teleop_command"]["lead_sec"] > 0.0
    copied = read_rgb_image(result.output_dir / steps[0]["images"]["tactile_left"])
    assert copied.shape == (480, 640, 3)


def _write_standard_episode_streams(episode_dir: Path) -> None:
    streams = episode_dir / "streams"
    rgb_records = []
    tactile_records = []
    for index, timestamp in enumerate([0.0, 0.1, 0.2, 0.3]):
        rgb = np.full((480, 640, 3), [10 + index, 20, 30], dtype=np.uint8)
        tactile = np.full((480, 640, 3), [40, 50 + index, 60], dtype=np.uint8)
        rgb_path = streams / "rgb_wrist" / "frames" / f"{index:06d}.jpg"
        tactile_path = streams / "tactile_left" / "frames" / f"{index:06d}.jpg"
        write_rgb_jpeg(rgb_path, rgb, quality=90)
        write_rgb_jpeg(tactile_path, tactile, quality=90)
        rgb_records.append(
            {
                "frame_index": index,
                "frame_path": rgb_path.relative_to(episode_dir).as_posix(),
                "captured_wall_time": timestamp,
                "sequence_id": index,
                "frame_width": 640,
                "frame_height": 480,
                "metadata": {"color_space": "RGB"},
            }
        )
        tactile_records.append(
            {
                "frame_index": index,
                "frame_path": tactile_path.relative_to(episode_dir).as_posix(),
                "captured_wall_time": timestamp,
                "sequence_id": index,
                "frame_width": 640,
                "frame_height": 480,
                "metadata": {"color_space": "RGB"},
            }
        )
    _write_jsonl(streams / "rgb_wrist" / "index.jsonl", rgb_records)
    _write_jsonl(streams / "tactile_left" / "index.jsonl", tactile_records)
    controller_records = []
    teleop_records = []
    for index, timestamp in enumerate([0.0, 0.1, 0.2, 0.3]):
        controller_records.append(
            {
                "received_wall_time": timestamp,
                "state": {
                    "tcp_pose": [0.3 + index * 0.01, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0],
                    "gripper_width": 0.078,
                },
            }
        )
        teleop_records.append(
            {
                "source_wall_time": timestamp + 0.05,
                "target_tcp": [0.31 + index * 0.01, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0],
                "gripper_closed": index >= 2,
            }
        )
    _write_jsonl(streams / "controller_state.jsonl", controller_records)
    _write_jsonl(streams / "teleop_commands.jsonl", teleop_records)
    episode_dir.mkdir(parents=True, exist_ok=True)
    (episode_dir / "episode_manifest.json").write_text(json.dumps({"outcome": "saved"}), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
