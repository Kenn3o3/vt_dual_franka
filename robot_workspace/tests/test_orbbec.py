from pathlib import Path

import numpy as np
import pytest

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
