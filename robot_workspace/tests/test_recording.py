from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vt_franka_workspace.recording import (
    AsyncImageStreamRecorder,
    AsyncRolloutVideoRecorder,
    AsyncStreamVideoRecorder,
    BufferedGelsightFrameRecorder,
    CanonicalPreprocess1StreamRecorder,
    CanonicalPreprocessBackpressure,
    GelsightBufferOverflow,
    JsonlStreamRecorder,
    RunSessionManager,
    default_canonical_stream_specs,
)
from vt_franka_workspace.recording.episode_streams import EpisodeImageStreamRecorder
from vt_franka_workspace.recording.image_io import read_rgb_image


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


def test_async_image_stream_recorder_flushes_frames(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("async_eval")
    episode_dir = sessions.start_episode("async_eval")
    recorder = AsyncImageStreamRecorder(sessions, "rgb_wrist", record_hz=10.0, queue_size=4)
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    frame[:, :, 0] = 210

    recorder.record_frame(
        frame,
        frame_id="000001",
        metadata={"color_space": "RGB"},
        extra_event_fields={"captured_wall_time": 1.0},
        event_time=1.0,
    )
    sessions.stop_episode(outcome="saved")
    summary = recorder.flush_episode(episode_dir)
    recorder.close()

    assert summary["enqueued"] == 1
    assert summary["written"] == 1
    event_path = episode_dir / "streams" / "rgb_wrist.jsonl"
    record = json.loads(event_path.read_text(encoding="utf-8").strip())
    decoded = read_rgb_image(episode_dir / record["frame_path"])
    assert decoded.shape == (8, 10, 3)
    assert decoded[:, :, 0].mean() > decoded[:, :, 2].mean()


def test_async_image_stream_recorder_drops_without_blocking_when_queue_is_full(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("async_eval")
    sessions.start_episode("async_eval")
    recorder = AsyncImageStreamRecorder(sessions, "rgb_wrist", queue_size=1)
    recorder._started = True
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    recorder.record_frame(frame, frame_id="000001", event_time=1.0)
    recorder.record_frame(frame, frame_id="000002", event_time=1.1)

    snapshot = recorder.snapshot()
    assert snapshot["total_enqueued"] == 1
    assert snapshot["dropped_due_to_backpressure"] == 1


def test_async_rollout_video_recorder_writes_mp4_from_action_steps(tmp_path: Path):
    pytest.importorskip("cv2")
    recorder = AsyncRolloutVideoRecorder(stream_name="rgb_wrist", output_name="rollout_wrist.mp4", fps=10.0, queue_size=4)
    episode_dir = tmp_path / "runs" / "task" / "episodes" / "episode_0000"
    episode_dir.mkdir(parents=True)

    recorder.record_frame(episode_dir, np.full((8, 12, 3), 10, dtype=np.uint8), event_time=1.0)
    recorder.record_frame(episode_dir, np.full((8, 12, 3), 20, dtype=np.uint8), event_time=1.1)
    summary = recorder.flush_episode(episode_dir)
    recorder.close()

    assert summary is not None
    assert summary["written"] == 2
    assert (episode_dir / "rollout_wrist.mp4").is_file()
    assert (episode_dir / "rollout_wrist.mp4").stat().st_size > 0


def test_async_stream_video_recorder_writes_rate_limited_mp4_from_active_episode(tmp_path: Path):
    pytest.importorskip("cv2")
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("stream")
    episode_dir = sessions.start_episode("episode_0000")
    recorder = AsyncStreamVideoRecorder(
        sessions,
        stream_name="rgb_third_person",
        output_name="rollout_third_person.mp4",
        fps=10.0,
        queue_size=4,
    )

    recorder.record_frame(np.full((8, 12, 3), 10, dtype=np.uint8), event_time=1.0)
    recorder.record_frame(np.full((8, 12, 3), 20, dtype=np.uint8), event_time=1.05)
    recorder.record_frame(np.full((8, 12, 3), 30, dtype=np.uint8), event_time=1.11)
    sessions.stop_episode(outcome="saved")
    summary = recorder.flush_episode(episode_dir)
    recorder.close()

    assert summary is not None
    assert summary["recording_mode"] == "stream"
    assert summary["written"] == 2
    assert summary["skipped_due_to_rate_limit"] == 1
    assert (episode_dir / "rollout_third_person.mp4").is_file()
    assert (episode_dir / "rollout_third_person.mp4").stat().st_size > 0


def test_buffered_gelsight_recorder_writes_chunked_npz_without_resizing(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("gelsight")
    episode_dir = sessions.start_episode("gelsight")
    recorder = BufferedGelsightFrameRecorder(sessions, max_frames=4, chunk_frames=2)

    frame0 = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
    frame1 = np.full((3, 4, 3), 7, dtype=np.uint8)
    frame2 = np.full((3, 4, 3), 9, dtype=np.uint8)
    recorder.record_frame(frame0, captured_wall_time=10.0, sequence_id=0, metadata={"camera_name": "gel"})
    recorder.record_frame(frame1, captured_wall_time=10.1, sequence_id=1, metadata={"camera_name": "gel"})
    recorder.record_frame(frame2, captured_wall_time=10.2, sequence_id=2, metadata={"camera_name": "gel"})
    sessions.stop_episode()

    summary = recorder.flush_episode(episode_dir)

    assert summary["frame_count"] == 3
    assert summary["chunk_count"] == 2
    index_path = episode_dir / "streams" / "gelsight_frames.jsonl"
    records = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines()]
    assert records[0]["frame_shape"] == [3, 4, 3]
    assert records[0]["dtype"] == "uint8"
    with np.load(episode_dir / records[0]["chunk_path"]) as chunk:
        assert chunk["frames"].shape == (2, 3, 4, 3)
        assert np.array_equal(chunk["frames"][0], frame0)
        assert chunk["captured_wall_times"].tolist() == [10.0, 10.1]


def test_buffered_gelsight_recorder_overflow_fails_episode(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("gelsight")
    sessions.start_episode("gelsight")
    recorder = BufferedGelsightFrameRecorder(sessions, max_frames=1, chunk_frames=1)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    recorder.record_frame(frame, captured_wall_time=1.0, sequence_id=0)
    with pytest.raises(GelsightBufferOverflow):
        recorder.record_frame(frame, captured_wall_time=1.1, sequence_id=1)


def test_episode_image_stream_recorder_buffers_standard_rgb_and_flushes_jpeg(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("standard_stream")
    episode_dir = sessions.start_episode("standard_stream")
    recorder = EpisodeImageStreamRecorder(sessions, "tactile_left", record_hz=10.0, jpeg_quality=90)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = 200
    frame[:, :, 1] = 10

    assert recorder.record_frame(frame, captured_wall_time=1.0, sequence_id=0, metadata={"color_space": "RGB"})
    assert not recorder.record_frame(frame, captured_wall_time=1.05, sequence_id=1, metadata={"color_space": "RGB"})
    assert recorder.record_frame(frame, captured_wall_time=1.12, sequence_id=2, metadata={"color_space": "RGB"})
    sessions.stop_episode()
    summary = recorder.flush_episode(episode_dir)

    assert summary is not None
    assert summary["stream_name"] == "tactile_left"
    assert summary["frame_count"] == 2
    records = [
        json.loads(line)
        for line in (episode_dir / "streams" / "tactile_left" / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["color_space"] == "RGB"
    decoded = read_rgb_image(episode_dir / records[0]["frame_path"])
    assert decoded.shape == (480, 640, 3)
    assert decoded[:, :, 0].mean() > decoded[:, :, 2].mean()


def test_canonical_preprocess1_stream_recorder_writes_small_rgb_chunks(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("preprocess1")
    episode_dir = sessions.start_episode("preprocess1")
    spec = default_canonical_stream_specs(
        canonical_size=4,
        gelsight_crop_box=None,
        gelsight_margin_fraction=0.0,
        wrist_raw_jpeg_compat=False,
    )["gelsight"]
    recorder = CanonicalPreprocess1StreamRecorder(sessions, spec, queue_size=2, chunk_frames=2)

    frame_bgr = np.zeros((6, 8, 3), dtype=np.uint8)
    frame_bgr[:, :, 0] = 10
    frame_bgr[:, :, 1] = 20
    frame_bgr[:, :, 2] = 30
    recorder.record_frame(frame_bgr, captured_wall_time=1.0, sequence_id=0)
    recorder.record_frame(frame_bgr, captured_wall_time=1.1, sequence_id=1)
    sessions.stop_episode()
    summary = recorder.flush_episode(episode_dir)
    recorder.close()

    assert summary is not None
    assert summary["frame_count"] == 2
    assert summary["canonical_shape"] == [4, 4, 3]
    records = [
        json.loads(line)
        for line in (episode_dir / "streams" / "preprocess1_gelsight.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    with np.load(episode_dir / records[0]["chunk_path"]) as chunk:
        assert chunk["frames"].shape == (2, 4, 4, 3)
        assert np.all(chunk["frames"][0] == [30, 20, 10])


def test_canonical_preprocess1_stream_recorder_updates_live_buffer_with_canonical_frame(tmp_path: Path):
    from vt_franka_workspace.runtime import LiveSampleBuffer

    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("preprocess1")
    episode_dir = sessions.start_episode("preprocess1")
    spec = default_canonical_stream_specs(
        canonical_size=4,
        gelsight_crop_box=None,
        gelsight_margin_fraction=0.0,
        wrist_raw_jpeg_compat=False,
    )["gelsight"]
    live_buffer = LiveSampleBuffer("gelsight_frame")
    recorder = CanonicalPreprocess1StreamRecorder(sessions, spec, queue_size=2, chunk_frames=2)

    recorder.record_frame(
        np.zeros((6, 8, 3), dtype=np.uint8),
        captured_wall_time=1.0,
        sequence_id=0,
        live_buffer=live_buffer,
    )
    recorder._queue.join()
    sample = live_buffer.get_latest(max_age_sec=None)
    recorder.flush_episode(episode_dir)
    recorder.close()

    assert sample.data.shape == (4, 4, 3)
    assert sample.metadata["canonical_shape"] == [4, 4, 3]


def test_canonical_preprocess1_stream_recorder_rate_limits_before_enqueue(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("preprocess1")
    sessions.start_episode("preprocess1")
    spec = default_canonical_stream_specs(
        canonical_size=4,
        gelsight_crop_box=None,
        gelsight_margin_fraction=0.0,
        wrist_raw_jpeg_compat=False,
    )["gelsight"]
    recorder = CanonicalPreprocess1StreamRecorder(sessions, spec, queue_size=1, chunk_frames=100, record_hz=10.0)
    recorder._started = True

    frame = np.zeros((6, 8, 3), dtype=np.uint8)
    recorder.record_frame(frame, captured_wall_time=1.0, sequence_id=0)
    recorder.record_frame(frame, captured_wall_time=1.03, sequence_id=1)
    recorder.record_frame(frame, captured_wall_time=1.06, sequence_id=2)

    snapshot = recorder.snapshot()
    assert snapshot["skipped_due_to_rate_limit"] == 2
    assert snapshot["total_enqueued"] == 1
    assert snapshot["dropped_due_to_backpressure"] == 0
    with pytest.raises(CanonicalPreprocessBackpressure):
        recorder.record_frame(frame, captured_wall_time=1.1, sequence_id=3)


def test_canonical_preprocess1_stream_recorder_rate_limit_uses_target_grid(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("preprocess1")
    sessions.start_episode("preprocess1")
    spec = default_canonical_stream_specs(
        canonical_size=4,
        gelsight_crop_box=None,
        gelsight_margin_fraction=0.0,
        wrist_raw_jpeg_compat=False,
    )["gelsight"]
    recorder = CanonicalPreprocess1StreamRecorder(sessions, spec, queue_size=20, chunk_frames=100, record_hz=10.0)
    recorder._started = True
    frame = np.zeros((6, 8, 3), dtype=np.uint8)

    for sequence_id in range(15):
        recorder.record_frame(frame, captured_wall_time=1.0 + sequence_id / 15.0, sequence_id=sequence_id)

    snapshot = recorder.snapshot()
    assert snapshot["total_enqueued"] == 10
    assert snapshot["skipped_due_to_rate_limit"] == 5


def test_canonical_preprocess1_stream_recorder_backpressure_is_explicit(tmp_path: Path):
    sessions = RunSessionManager(tmp_path / "runs")
    sessions.start_run("preprocess1")
    sessions.start_episode("preprocess1")
    spec = default_canonical_stream_specs(
        canonical_size=4,
        gelsight_crop_box=None,
        gelsight_margin_fraction=0.0,
        wrist_raw_jpeg_compat=False,
    )["gelsight"]
    recorder = CanonicalPreprocess1StreamRecorder(sessions, spec, queue_size=1, chunk_frames=100)
    recorder._started = True
    recorder._queue.put(object())  # type: ignore[arg-type]

    with pytest.raises(CanonicalPreprocessBackpressure):
        recorder.record_frame(np.zeros((6, 8, 3), dtype=np.uint8), captured_wall_time=1.0, sequence_id=0)
