from __future__ import annotations

import json
from pathlib import Path

from vt_franka_workspace.recording import JsonlStreamRecorder, RunSessionManager


def test_run_session_manager_creates_nested_raw_episode(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    run_dir = sessions.start_run("task_demo", metadata={"operator": "tester"})
    episode_dir = sessions.start_episode()
    sessions.stop_episode(outcome="saved")
    sessions.stop_run()

    run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    episode_manifest = json.loads((episode_dir / "episode_manifest.json").read_text(encoding="utf-8"))
    assert run_manifest["run_name"] == "task_demo"
    assert run_dir == tmp_path / "runs" / "task_demo"
    assert episode_dir.parent.name == "episodes"
    assert episode_manifest["outcome"] == "saved"


def test_run_session_manager_resumes_and_increments_episode_index(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    run_dir = sessions.start_run("task_demo")
    episode_dir = sessions.start_episode()
    sessions.stop_episode(outcome="saved")
    sessions.stop_run()

    resumed = RunSessionManager(tmp_path / "runs")
    resumed_run_dir = resumed.start_run("task_demo")
    next_episode_dir = resumed.start_episode()

    assert resumed_run_dir == run_dir
    assert episode_dir.name == "episode_0000"
    assert next_episode_dir.name == "episode_0001"


def test_jsonl_stream_recorder_respects_record_hz(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("rate_limited")
    sessions.start_episode("rate_limited")
    recorder = JsonlStreamRecorder(sessions, "controller_state", record_hz=2.0)

    recorder.record_event({"source_wall_time": 1.0, "state": {"tcp_pose": [0.0] * 7}}, event_time=1.0)
    recorder.record_event({"source_wall_time": 1.1, "state": {"tcp_pose": [0.0] * 7}}, event_time=1.1)
    recorder.record_event({"source_wall_time": 1.6, "state": {"tcp_pose": [0.0] * 7}}, event_time=1.6)

    path = sessions.get_active_episode_dir() / "streams" / "controller_state.jsonl"
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
