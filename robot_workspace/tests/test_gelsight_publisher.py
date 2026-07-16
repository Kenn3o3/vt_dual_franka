from __future__ import annotations

from pathlib import Path

import numpy as np

from vt_dual_franka_workspace.sensors.gelsight import publisher
from vt_dual_franka_workspace.settings import GelsightSettings


def test_filter_candidates_matches_name_and_serial():
    candidates = [
        {"name": "Orbbec Gemini", "serial_number": "AAA", "device_path": "/dev/video0"},
        {"name": "GelSight Mini R0B", "serial_number": "2BUUCE3E", "device_path": "/dev/video12"},
    ]

    matched = publisher._filter_candidates(
        candidates,
        name_contains="GelSight Mini",
        serial_number="2BUUCE3E",
    )

    assert matched == [{"name": "GelSight Mini R0B", "serial_number": "2BUUCE3E", "device_path": "/dev/video12"}]


def test_query_v4l_device_detects_metadata_only_node(monkeypatch, tmp_path: Path):
    device = tmp_path / "video13"
    device.write_text("", encoding="utf-8")

    class FakeCompletedProcess:
        def __init__(self, stdout: str):
            self.returncode = 0
            self.stdout = stdout

    stdout = "\n".join(
        [
            "Card type        : GelSight Mini R0B 2BUU-CE3E: Ge",
            "Serial           : 2BUUCE3E",
            "Device Caps      : 0x04a00000",
            "\tMetadata Capture",
            "\tStreaming",
        ]
    )
    monkeypatch.setattr(
        publisher.subprocess,
        "run",
        lambda *args, **kwargs: FakeCompletedProcess(stdout),
    )

    info = publisher._query_v4l_device(device)

    assert info is not None
    assert info["video_capture"] is False
    assert info["serial_number"] == "2BUUCE3E"


def test_gelsight_settings_exposes_buffered_recording_options():
    gel_publisher = publisher.GelsightPublisher(
        settings=GelsightSettings(buffered_recording=True, buffer_max_frames=12, buffer_chunk_frames=3),
        quest_publisher=_DummyQuestPublisher(),
    )

    assert gel_publisher.settings.buffered_recording is True
    assert gel_publisher.settings.buffer_max_frames == 12
    assert gel_publisher.settings.buffer_chunk_frames == 3


def test_gelsight_configure_capture_skips_camera_controls_by_default(monkeypatch):
    calls = []

    class FakeCv2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        CAP_PROP_EXPOSURE = 15
        CAP_PROP_CONTRAST = 11

    class FakeCap:
        def set(self, prop, value):
            calls.append((prop, value))

    monkeypatch.setitem(__import__("sys").modules, "cv2", FakeCv2())
    gel_publisher = publisher.GelsightPublisher(
        settings=GelsightSettings(width=3280, height=2464, fps=25, exposure=-6, contrast=100),
        quest_publisher=_DummyQuestPublisher(),
    )

    gel_publisher._configure_capture(FakeCap())

    assert calls == [(3, 3280), (4, 2464), (5, 25)]


def test_gelsight_configure_capture_applies_camera_controls_when_enabled(monkeypatch):
    calls = []

    class FakeCv2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        CAP_PROP_EXPOSURE = 15
        CAP_PROP_CONTRAST = 11

    class FakeCap:
        def set(self, prop, value):
            calls.append((prop, value))

    monkeypatch.setitem(__import__("sys").modules, "cv2", FakeCv2())
    gel_publisher = publisher.GelsightPublisher(
        settings=GelsightSettings(
            width=3280,
            height=2464,
            fps=25,
            apply_controls=True,
            exposure=-6,
            contrast=32,
        ),
        quest_publisher=_DummyQuestPublisher(),
    )

    gel_publisher._configure_capture(FakeCap())

    assert calls == [(3, 3280), (4, 2464), (5, 25), (15, -6), (11, 32)]


def test_gelsight_direct_frame_recorder_receives_standardized_rgb(monkeypatch):
    recorded = {}

    class FakeFrameRecorder:
        def record_frame(self, frame, frame_id, metadata=None, image_format="jpg", extra_event_fields=None, event_time=None):
            del frame_id, image_format, extra_event_fields, event_time
            recorded["frame"] = np.asarray(frame)
            recorded["metadata"] = dict(metadata or {})

    class FakeCv2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        COLOR_BGR2RGB = 9

        class VideoCapture:
            def __init__(self, source):
                del source
                self.read_count = 0

            def isOpened(self):
                return True

            def set(self, prop, value):
                del prop, value

            def read(self):
                if self.read_count > 0:
                    raise StopIteration
                self.read_count += 1
                frame_bgr = np.zeros((4, 5, 3), dtype=np.uint8)
                frame_bgr[:, :, 0] = 10
                frame_bgr[:, :, 2] = 200
                return True, frame_bgr

            def release(self):
                return None

        @staticmethod
        def cvtColor(frame, code):
            del code
            return frame[:, :, ::-1].copy()

        @staticmethod
        def resize(frame, size, interpolation=None):
            del interpolation
            width, height = size
            return np.resize(frame, (height, width, 3)).astype(np.uint8)

        INTER_AREA = 0

    monkeypatch.setitem(__import__("sys").modules, "cv2", FakeCv2())
    monkeypatch.setattr(publisher.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(publisher, "precise_sleep", lambda seconds: None)
    gel_publisher = publisher.GelsightPublisher(
        settings=GelsightSettings(save_frames=True, width=5, height=4, fps=30),
        quest_publisher=_DummyQuestPublisher(),
        frame_recorder=FakeFrameRecorder(),
    )

    try:
        gel_publisher.run()
    except StopIteration:
        pass

    assert recorded["frame"].shape == (480, 640, 3)
    assert recorded["frame"][0, 0, 0] == 200
    assert recorded["frame"][0, 0, 2] == 10
    assert recorded["metadata"]["color_space"] == "RGB"
    assert recorded["metadata"]["standard_shape"] == [480, 640, 3]


class _DummyQuestPublisher:
    def publish_image(self, image, stream_settings) -> None:
        return None
