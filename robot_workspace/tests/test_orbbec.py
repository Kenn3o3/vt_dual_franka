import json
from pathlib import Path

import numpy as np
import pytest

from vt_franka_workspace.inference.eval_video import write_rollout_video
from vt_franka_workspace.recording.raw_recorder import JsonlStreamRecorder
from vt_franka_workspace.recording.session import RunSessionManager
from vt_franka_workspace.sensors.orbbec.frame_decoder import decode_color_buffer


def test_decode_rgb_buffer_to_bgr():
    pytest.importorskip("cv2")
    rgb = np.array([[[255, 0, 0], [0, 255, 0]]], dtype=np.uint8)
    image = decode_color_buffer(rgb.tobytes(), width=2, height=1, color_format="RGB")
    assert image.tolist() == [[[0, 0, 255], [0, 255, 0]]]


def test_rgb_camera_stream_recorder_writes_raw_frame_event(tmp_path: Path):
    pytest.importorskip("cv2")
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("orbbec")
    sessions.start_episode("orbbec")

    rgb_camera = JsonlStreamRecorder(sessions, "rgb_third_person")

    frame_path = rgb_camera.record_frame(
        np.zeros((8, 9, 3), dtype=np.uint8),
        frame_id="000001",
        extra_event_fields={"captured_wall_time": 1.0, "frame_width": 9, "frame_height": 8},
        event_time=1.0,
    )

    assert frame_path is not None
    assert frame_path.name == "000001.jpg"
    event_path = sessions.get_active_episode_dir() / "streams" / "rgb_third_person.jsonl"
    assert "streams/rgb_third_person/000001.jpg" in event_path.read_text(encoding="utf-8")


def test_write_rollout_video_from_rgb_stream(tmp_path: Path):
    pytest.importorskip("cv2")
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("eval_video")
    episode_dir = sessions.start_episode("eval_video")
    rgb_camera = JsonlStreamRecorder(sessions, "rgb_third_person")

    for index in range(3):
        frame = np.full((8, 9, 3), index * 80, dtype=np.uint8)
        rgb_camera.record_frame(frame, frame_id=f"{index:06d}", event_time=float(index))

    output_path = write_rollout_video(episode_dir, stream_name="rgb_third_person", output_name="rollout.mp4", fps=10.0)

    assert output_path == episode_dir / "rollout.mp4"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_write_rollout_video_draws_policy_action_overlay(tmp_path: Path):
    cv2 = pytest.importorskip("cv2")
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("eval_video")
    episode_dir = sessions.start_episode("eval_video")
    rgb_camera = JsonlStreamRecorder(sessions, "rgb_third_person")

    for index in range(3):
        frame = np.full((64, 96, 3), 255, dtype=np.uint8)
        rgb_camera.record_frame(frame, frame_id=f"{index:06d}", event_time=float(index))
    streams = episode_dir / "streams"
    policy_step = {
        "step_index": 1,
        "phase": "policy_chunk",
        "policy_wall_time": 0.0,
        "actions_executed": [
            {
                "target_tcp": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
                "gripper_closed": True,
                "metadata": {
                    "mpd_tcp_state": [0.1, 0.2, 0.3, 1, 0, 0, 0, 1, 0, 0.25],
                    "mpd_algorithm": "motif",
                    "mpd_action_convention": "tcp_xyz_rot6d_gripper_open_fraction",
                },
            }
        ],
        "observations_after_actions": [
            {
                "chunk_action_index": 0,
                "observation": {
                    "assembled_wall_time": 0.0,
                    "proprioception": {"controller_state": {"wall_time": 0.0}},
                },
            }
        ],
    }
    (streams / "policy_steps.jsonl").write_text(json.dumps(policy_step) + "\n", encoding="utf-8")

    output_path = write_rollout_video(episode_dir, stream_name="rgb_third_person", output_name="rollout.mp4", fps=10.0)

    assert output_path == episode_dir / "rollout.mp4"
    cap = cv2.VideoCapture(str(output_path))
    ok, frame = cap.read()
    cap.release()
    assert ok
    assert frame[:20].mean() < 250
