from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from vt_franka_workspace.sensors.gelsight import publisher
from vt_franka_workspace.settings import GelsightSettings


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


def test_apply_marker_safety_mask_zeros_index_8():
    gel_publisher = publisher.GelsightPublisher(
        settings=GelsightSettings(),
        quest_publisher=_DummyQuestPublisher(),
    )
    offsets = np.zeros((12, 2), dtype=np.float32)
    offsets[8] = [12.0, -5.0]
    offsets[7] = [0.1, 0.2]

    masked = gel_publisher._apply_marker_safety_mask(offsets)

    assert np.allclose(masked[8], np.zeros(2, dtype=np.float32))
    assert np.allclose(masked[7], np.array([0.1, 0.2], dtype=np.float32))


class _DummyQuestPublisher:
    def publish_tactile(self, message) -> None:
        return None

    def publish_image(self, image, stream_settings) -> None:
        return None
