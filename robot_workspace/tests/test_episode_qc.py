from __future__ import annotations

import json
from pathlib import Path

from vt_franka_workspace.recording import analyze_episode_quality, format_episode_qc_summary


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")


def test_episode_qc_reports_hz_gaps_and_missing_frames(tmp_path: Path):
    episode_dir = tmp_path / "episode_0000"
    streams = episode_dir / "streams"
    episode_dir.mkdir()
    (episode_dir / "episode_manifest.json").write_text(
        json.dumps(
            {
                "episode_id": "episode_0000",
                "episode_name": "episode_0000",
                "outcome": "saved",
                "started_at_wall_time": 10.0,
                "stopped_at_wall_time": 14.0,
            }
        ),
        encoding="utf-8",
    )
    _write_jsonl(
        streams / "controller_state.jsonl",
        [
            {"source_wall_time": 10.0, "received_wall_time": 10.01},
            {"source_wall_time": 10.5, "received_wall_time": 10.51},
            {"source_wall_time": 11.0, "received_wall_time": 11.01},
        ],
    )
    _write_jsonl(
        streams / "rgb_wrist.jsonl",
        [
            {"captured_wall_time": 10.0, "frame_path": "streams/rgb_wrist/frame0.jpg", "sequence_id": 2},
            {"captured_wall_time": 10.1, "frame_path": "streams/rgb_wrist/frame1.jpg", "sequence_id": 3},
            {"captured_wall_time": 10.4, "frame_path": "streams/rgb_wrist/missing.jpg", "sequence_id": 4},
        ],
    )
    frame_dir = streams / "rgb_wrist"
    frame_dir.mkdir()
    (frame_dir / "frame0.jpg").write_bytes(b"0")
    (frame_dir / "frame1.jpg").write_bytes(b"1")

    report = analyze_episode_quality(episode_dir, expected_hz={"rgb_wrist": 15.0}, write=True)

    assert (episode_dir / "episode_qc.json").exists()
    assert report["streams"]["controller_state"]["effective_hz"] == 2.0
    assert report["streams"]["rgb_wrist"]["record_count"] == 3
    assert report["streams"]["rgb_wrist"]["missing_frame_file_count"] == 1
    assert report["streams"]["rgb_wrist"]["max_gap_sec"] == 0.3000000000000007
    assert any("rgb_wrist:" in warning for warning in report["warnings"])
    assert "rgb_wrist:" in format_episode_qc_summary(report)
