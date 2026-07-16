from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .image_io import ensure_hwc_uint8_rgb

LOGGER = logging.getLogger(__name__)


class SupportsActiveEpisodeDir(Protocol):
    def get_active_episode_dir(self) -> Path | None: ...


@dataclass(frozen=True)
class _VideoFrameSample:
    episode_dir: Path
    frame: np.ndarray
    event_time: float


@dataclass
class _EpisodeVideoStats:
    enqueued: int = 0
    written: int = 0
    dropped_due_to_backpressure: int = 0
    write_errors: int = 0
    first_event_time: float | None = None
    last_event_time: float | None = None
    output_path: Path | None = None
    temp_path: Path | None = None
    frame_size: tuple[int, int] | None = None
    last_error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "enqueued": self.enqueued,
            "written": self.written,
            "dropped_due_to_backpressure": self.dropped_due_to_backpressure,
            "write_errors": self.write_errors,
            "first_event_time": self.first_event_time,
            "last_event_time": self.last_event_time,
            "output_path": None if self.output_path is None else str(self.output_path),
            "temp_path": None if self.temp_path is None else str(self.temp_path),
            "frame_size": None if self.frame_size is None else [int(value) for value in self.frame_size],
            "last_error": self.last_error,
        }


class AsyncRolloutVideoRecorder:
    """Background MP4 writer for lightweight action-step eval videos."""

    def __init__(
        self,
        *,
        stream_name: str,
        output_name: str,
        fps: float,
        queue_size: int = 128,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self.stream_name = stream_name
        self.output_name = output_name
        self.fps = float(fps)
        self.queue_size = int(queue_size)
        self._queue: queue.Queue[_VideoFrameSample | None] = queue.Queue(maxsize=self.queue_size)
        self._lock = threading.Lock()
        self._started = False
        self._closed = False
        self._worker = threading.Thread(target=self._run_worker, name=f"eval-rollout-video:{stream_name}", daemon=True)
        self._stats_by_episode: dict[Path, _EpisodeVideoStats] = {}
        self._current_episode_dir: Path | None = None
        self._current_writer = None
        self._current_output_path: Path | None = None
        self._current_temp_path: Path | None = None
        self._current_frame_size: tuple[int, int] | None = None
        self._total_enqueued = 0
        self._total_written = 0
        self._total_dropped_due_to_backpressure = 0
        self._total_write_errors = 0

    def start(self) -> None:
        with self._lock:
            if self._closed or self._started:
                return
            self._started = True
            self._worker.start()

    def record_frame(self, episode_dir: str | Path, frame: np.ndarray, *, event_time: float | None = None) -> None:
        episode_path = Path(episode_dir)
        event_time = float(event_time if event_time is not None else time.time())
        self.start()
        sample = _VideoFrameSample(
            episode_dir=episode_path,
            frame=np.ascontiguousarray(frame),
            event_time=event_time,
        )
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            with self._lock:
                stats = self._stats_by_episode.setdefault(episode_path, _EpisodeVideoStats())
                stats.dropped_due_to_backpressure += 1
                self._total_dropped_due_to_backpressure += 1
            return
        with self._lock:
            stats = self._stats_by_episode.setdefault(episode_path, _EpisodeVideoStats())
            stats.enqueued += 1
            if stats.first_event_time is None:
                stats.first_event_time = event_time
            stats.last_event_time = event_time
            self._total_enqueued += 1

    def flush_episode(self, episode_dir: str | Path, *, settle_sec: float = 0.05) -> dict[str, Any] | None:
        episode_path = Path(episode_dir)
        self._queue.join()
        if settle_sec > 0.0:
            time.sleep(settle_sec)
        with self._lock:
            stats = self._stats_by_episode.get(episode_path)
            if stats is None or (stats.enqueued == 0 and stats.written == 0 and stats.dropped_due_to_backpressure == 0 and stats.write_errors == 0):
                return None
            if self._current_episode_dir == episode_path:
                self._close_current_writer_locked()
            payload = stats.to_json()
            payload.update({"stream_name": self.stream_name, "fps": self.fps, "queue_size": self.queue_size})
            return payload

    def close(self, timeout_sec: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
        if started:
            try:
                self._queue.put(None, timeout=timeout_sec)
            except queue.Full:
                LOGGER.warning("Timed out closing rollout video recorder for %s", self.stream_name)
                return
            self._worker.join(timeout=timeout_sec)
            if self._worker.is_alive():
                LOGGER.warning("Timed out closing rollout video recorder for %s", self.stream_name)
        with self._lock:
            self._close_current_writer_locked()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stream_name": self.stream_name,
                "fps": self.fps,
                "queue_size": self.queue_size,
                "queued_frames": self._queue.qsize(),
                "total_enqueued": self._total_enqueued,
                "total_written": self._total_written,
                "dropped_due_to_backpressure": self._total_dropped_due_to_backpressure,
                "write_errors": self._total_write_errors,
                "closed": self._closed,
                "active_episode": None if self._current_episode_dir is None else str(self._current_episode_dir),
                "output_path": None if self._current_output_path is None else str(self._current_output_path),
                "current_frame_size": None
                if self._current_frame_size is None
                else [int(value) for value in self._current_frame_size],
            }

    def _run_worker(self) -> None:
        while True:
            sample = self._queue.get()
            try:
                if sample is None:
                    return
                self._write_frame_sample(sample)
            except Exception as exc:  # pragma: no cover - codec/disk dependent
                episode_dir = getattr(sample, "episode_dir", None)
                if episode_dir is not None:
                    with self._lock:
                        stats = self._stats_by_episode.setdefault(episode_dir, _EpisodeVideoStats())
                        stats.write_errors += 1
                        stats.last_error = str(exc)
                        self._total_write_errors += 1
                LOGGER.warning("Failed to write rollout video frame for %s: %s", self.stream_name, exc)
            finally:
                self._queue.task_done()

    def _write_frame_sample(self, sample: _VideoFrameSample) -> None:
        frame_rgb = ensure_hwc_uint8_rgb(sample.frame)
        frame_bgr = self._ensure_writer_and_maybe_resize(sample.episode_dir, frame_rgb)
        with self._lock:
            writer = self._current_writer
            stats = self._stats_by_episode.setdefault(sample.episode_dir, _EpisodeVideoStats())
            stats.last_event_time = sample.event_time
        if writer is None:
            raise RuntimeError(f"Video writer is not available for {self.stream_name}")
        writer.write(frame_bgr)
        with self._lock:
            stats = self._stats_by_episode.setdefault(sample.episode_dir, _EpisodeVideoStats())
            stats.written += 1
            self._total_written += 1

    def _ensure_writer_and_maybe_resize(self, episode_dir: Path, frame_rgb: np.ndarray) -> np.ndarray:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("OpenCV is required to write rollout videos") from exc

        frame_height, frame_width = frame_rgb.shape[:2]
        with self._lock:
            if self._current_episode_dir != episode_dir:
                self._close_current_writer_locked()
                self._current_episode_dir = episode_dir
            if self._current_writer is None:
                output_path = episode_dir / self.output_name
                temp_path = episode_dir / f".{output_path.stem}.tmp{output_path.suffix}"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if temp_path.exists():
                    temp_path.unlink()
                writer = cv2.VideoWriter(
                    str(temp_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(self.fps),
                    (frame_width, frame_height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Failed to open rollout video writer: {temp_path}")
                self._current_writer = writer
                self._current_output_path = output_path
                self._current_temp_path = temp_path
                self._current_frame_size = (frame_width, frame_height)
                stats = self._stats_by_episode.setdefault(episode_dir, _EpisodeVideoStats())
                stats.output_path = output_path
                stats.temp_path = temp_path
                stats.frame_size = self._current_frame_size
            target_width, target_height = self._current_frame_size or (frame_width, frame_height)
        if (frame_width, frame_height) != (target_width, target_height):
            frame_rgb = cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

    def _close_current_writer_locked(self) -> None:
        writer = self._current_writer
        output_path = self._current_output_path
        temp_path = self._current_temp_path
        episode_dir = self._current_episode_dir
        self._current_writer = None
        self._current_output_path = None
        self._current_temp_path = None
        self._current_frame_size = None
        self._current_episode_dir = None
        if writer is not None:
            writer.release()
        if writer is not None and output_path is not None and temp_path is not None and temp_path.exists():
            os.replace(temp_path, output_path)
            if episode_dir is not None:
                stats = self._stats_by_episode.setdefault(episode_dir, _EpisodeVideoStats())
                stats.output_path = output_path


class AsyncStreamVideoRecorder(AsyncRolloutVideoRecorder):
    """Background MP4 writer fed directly by a live camera stream during active episodes."""

    def __init__(
        self,
        session_manager: SupportsActiveEpisodeDir,
        *,
        stream_name: str,
        output_name: str,
        fps: float,
        queue_size: int = 128,
    ) -> None:
        super().__init__(stream_name=stream_name, output_name=output_name, fps=fps, queue_size=queue_size)
        self.session_manager = session_manager
        self._stream_ingest_lock = threading.Lock()
        self._last_stream_episode_dir: Path | None = None
        self._last_stream_record_time: float | None = None
        self._skipped_due_to_rate_limit_by_episode: dict[Path, int] = {}
        self._total_skipped_due_to_rate_limit = 0

    def record_frame(self, frame: np.ndarray, *, event_time: float | None = None) -> None:  # type: ignore[override]
        episode_dir = self.session_manager.get_active_episode_dir()
        if episode_dir is None:
            return
        episode_path = Path(episode_dir)
        event_time = float(event_time if event_time is not None else time.time())
        with self._stream_ingest_lock:
            if self.session_manager.get_active_episode_dir() != episode_path:
                return
            if not self._reserve_stream_record_slot(episode_path, event_time):
                return
            super().record_frame(episode_path, frame, event_time=event_time)

    def flush_episode(self, episode_dir: str | Path, *, settle_sec: float = 0.05) -> dict[str, Any] | None:
        episode_path = Path(episode_dir)
        payload = super().flush_episode(episode_path, settle_sec=settle_sec)
        if payload is None:
            return None
        with self._lock:
            skipped = self._skipped_due_to_rate_limit_by_episode.get(episode_path, 0)
        payload.update(
            {
                "recording_mode": "stream",
                "skipped_due_to_rate_limit": skipped,
            }
        )
        return payload

    def snapshot(self) -> dict[str, Any]:
        payload = super().snapshot()
        with self._lock:
            payload.update(
                {
                    "recording_mode": "stream",
                    "skipped_due_to_rate_limit": self._total_skipped_due_to_rate_limit,
                }
            )
        return payload

    def _reserve_stream_record_slot(self, episode_dir: Path, event_time: float) -> bool:
        with self._lock:
            if self._closed:
                return False
            if self._last_stream_episode_dir != episode_dir:
                self._last_stream_episode_dir = episode_dir
                self._last_stream_record_time = None
            if self._last_stream_record_time is not None:
                min_period = 1.0 / max(self.fps, 1e-6)
                if event_time - self._last_stream_record_time < min_period:
                    self._skipped_due_to_rate_limit_by_episode[episode_dir] = (
                        self._skipped_due_to_rate_limit_by_episode.get(episode_dir, 0) + 1
                    )
                    self._total_skipped_due_to_rate_limit += 1
                    return False
            self._last_stream_record_time = event_time
            return True
