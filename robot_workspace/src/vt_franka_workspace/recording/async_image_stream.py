from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .image_io import write_rgb_image
from .raw_recorder import _json_default

LOGGER = logging.getLogger(__name__)


class SupportsActiveEpisodeDir(Protocol):
    def get_active_episode_dir(self) -> Path | None: ...


@dataclass
class _AsyncEpisodeStats:
    enqueued: int = 0
    written: int = 0
    skipped_due_to_rate_limit: int = 0
    dropped_due_to_backpressure: int = 0
    write_errors: int = 0

    def to_json(self) -> dict[str, int]:
        return {
            "enqueued": self.enqueued,
            "written": self.written,
            "skipped_due_to_rate_limit": self.skipped_due_to_rate_limit,
            "dropped_due_to_backpressure": self.dropped_due_to_backpressure,
            "write_errors": self.write_errors,
        }


@dataclass(frozen=True)
class _AsyncFrameSample:
    episode_dir: Path
    frame: np.ndarray
    frame_id: str
    image_format: str
    metadata: dict[str, Any]
    extra_event_fields: dict[str, Any]
    event_time: float


@dataclass(frozen=True)
class _AsyncEventSample:
    episode_dir: Path
    payload: dict[str, Any]
    event_time: float


class AsyncImageStreamRecorder:
    """Non-blocking image stream recorder for eval videos.

    Camera threads only rate-limit and enqueue frames. JPEG encoding, directory
    creation, and JSONL appends happen in one background worker. If the queue is
    full, recording drops the frame instead of slowing camera capture.
    """

    def __init__(
        self,
        session_manager: SupportsActiveEpisodeDir,
        stream_name: str,
        record_hz: float = 0.0,
        *,
        queue_size: int = 128,
        jpeg_quality: int = 90,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if record_hz < 0.0:
            raise ValueError("record_hz must be non-negative")
        self.session_manager = session_manager
        self.stream_name = stream_name
        self.record_hz = float(record_hz)
        self.queue_size = int(queue_size)
        self.jpeg_quality = int(jpeg_quality)
        self._queue: queue.Queue[_AsyncFrameSample | _AsyncEventSample | None] = queue.Queue(maxsize=self.queue_size)
        self._ingest_lock = threading.Lock()
        self._lock = threading.Lock()
        self._file_lock = threading.Lock()
        self._last_record_time: float | None = None
        self._last_episode_dir: Path | None = None
        self._stats_by_episode: dict[Path, _AsyncEpisodeStats] = {}
        self._total_enqueued = 0
        self._total_written = 0
        self._total_skipped_due_to_rate_limit = 0
        self._total_dropped_due_to_backpressure = 0
        self._total_write_errors = 0
        self._started = False
        self._closed = False
        self._worker = threading.Thread(target=self._run_worker, name=f"eval-video:{stream_name}", daemon=True)

    def start(self) -> None:
        with self._lock:
            if self._closed or self._started:
                return
            self._started = True
            self._worker.start()

    def close(self, timeout_sec: float = 5.0) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            started = self._started
        if not started:
            return
        self._queue.put(None, timeout=timeout_sec)
        self._worker.join(timeout=timeout_sec)
        if self._worker.is_alive():
            LOGGER.warning("Timed out closing eval video recorder for %s", self.stream_name)

    def record_event(self, payload: dict[str, Any], event_time: float | None = None) -> None:
        episode_dir = self.session_manager.get_active_episode_dir()
        event_time = float(event_time if event_time is not None else payload.get("recorded_at_wall_time", time.time()))
        if episode_dir is None:
            return
        sample = _AsyncEventSample(episode_dir=episode_dir, payload=dict(payload), event_time=event_time)
        with self._ingest_lock:
            if self.session_manager.get_active_episode_dir() != episode_dir:
                return
            if not self._reserve_record_slot(episode_dir, event_time):
                return
            self.start()
            self._enqueue_sample(sample, episode_dir=episode_dir)

    def record_frame(
        self,
        frame: np.ndarray,
        frame_id: str,
        metadata: dict[str, Any] | None = None,
        image_format: str = "jpg",
        extra_event_fields: dict[str, Any] | None = None,
        event_time: float | None = None,
    ) -> None:
        episode_dir = self.session_manager.get_active_episode_dir()
        event_time = float(event_time if event_time is not None else time.time())
        if episode_dir is None:
            return
        with self._ingest_lock:
            if self.session_manager.get_active_episode_dir() != episode_dir:
                return
            if not self._reserve_record_slot(episode_dir, event_time):
                return
            self.start()
            sample = _AsyncFrameSample(
                episode_dir=episode_dir,
                frame=np.ascontiguousarray(frame),
                frame_id=str(frame_id),
                image_format=image_format.lower().lstrip(".") or "jpg",
                metadata=dict(metadata or {}),
                extra_event_fields=dict(extra_event_fields or {}),
                event_time=event_time,
            )
            self._enqueue_sample(sample, episode_dir=episode_dir)

    def flush_episode(self, episode_dir: Path, *, settle_sec: float = 0.1) -> dict[str, Any]:
        episode_dir = Path(episode_dir)
        with self._ingest_lock:
            previous_counts: tuple[int, int, int, int, int] | None = None
            stable_since: float | None = None
            while True:
                self._queue.join()
                counts = self._episode_counts(episode_dir)
                now = time.time()
                if counts == previous_counts:
                    if stable_since is None:
                        stable_since = now
                    if now - stable_since >= settle_sec:
                        break
                else:
                    previous_counts = counts
                    stable_since = now
                time.sleep(min(settle_sec, 0.02))

            with self._lock:
                stats = self._stats_by_episode.pop(episode_dir, _AsyncEpisodeStats())
        payload = stats.to_json()
        payload.update(
            {
                "stream_name": self.stream_name,
                "record_hz": self.record_hz,
                "queue_size": self.queue_size,
            }
        )
        return payload

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stream_name": self.stream_name,
                "record_hz": self.record_hz,
                "queue_size": self.queue_size,
                "queued_frames": self._queue.qsize(),
                "total_enqueued": self._total_enqueued,
                "total_written": self._total_written,
                "skipped_due_to_rate_limit": self._total_skipped_due_to_rate_limit,
                "dropped_due_to_backpressure": self._total_dropped_due_to_backpressure,
                "write_errors": self._total_write_errors,
                "closed": self._closed,
            }

    def _reserve_record_slot(self, episode_dir: Path, event_time: float) -> bool:
        with self._lock:
            if self._closed:
                return False
            if self._last_episode_dir != episode_dir:
                self._last_episode_dir = episode_dir
                self._last_record_time = None
            stats = self._stats_by_episode.setdefault(episode_dir, _AsyncEpisodeStats())
            if self.record_hz > 0.0 and self._last_record_time is not None:
                if event_time - self._last_record_time < 1.0 / self.record_hz:
                    stats.skipped_due_to_rate_limit += 1
                    self._total_skipped_due_to_rate_limit += 1
                    return False
            self._last_record_time = event_time
            return True

    def _enqueue_sample(self, sample: _AsyncFrameSample | _AsyncEventSample, *, episode_dir: Path) -> None:
        try:
            self._queue.put_nowait(sample)
        except queue.Full:
            with self._lock:
                stats = self._stats_by_episode.setdefault(episode_dir, _AsyncEpisodeStats())
                stats.dropped_due_to_backpressure += 1
                self._total_dropped_due_to_backpressure += 1
            return
        with self._lock:
            stats = self._stats_by_episode.setdefault(episode_dir, _AsyncEpisodeStats())
            stats.enqueued += 1
            self._total_enqueued += 1

    def _run_worker(self) -> None:
        while True:
            sample = self._queue.get()
            try:
                if sample is None:
                    return
                if isinstance(sample, _AsyncFrameSample):
                    self._write_frame_sample(sample)
                else:
                    self._write_event(sample.episode_dir, sample.payload, event_time=sample.event_time)
                with self._lock:
                    stats = self._stats_by_episode.setdefault(sample.episode_dir, _AsyncEpisodeStats())
                    stats.written += 1
                    self._total_written += 1
            except Exception as exc:  # pragma: no cover - disk/codec dependent
                episode_dir = getattr(sample, "episode_dir", None)
                if episode_dir is not None:
                    with self._lock:
                        stats = self._stats_by_episode.setdefault(episode_dir, _AsyncEpisodeStats())
                        stats.write_errors += 1
                        self._total_write_errors += 1
                LOGGER.warning("Failed to write eval video frame for %s: %s", self.stream_name, exc)
            finally:
                self._queue.task_done()

    def _write_frame_sample(self, sample: _AsyncFrameSample) -> None:
        frame_dir = sample.episode_dir / "streams" / self.stream_name
        frame_path = frame_dir / f"{sample.frame_id}.{sample.image_format}"
        write_rgb_image(frame_path, sample.frame, quality=self.jpeg_quality)
        payload: dict[str, Any] = {"frame_path": frame_path.relative_to(sample.episode_dir).as_posix()}
        if sample.metadata:
            payload["metadata"] = sample.metadata
        if sample.extra_event_fields:
            payload.update(sample.extra_event_fields)
        self._write_event(sample.episode_dir, payload, event_time=sample.event_time)

    def _write_event(self, episode_dir: Path, payload: dict[str, Any], *, event_time: float) -> None:
        stream_dir = episode_dir / "streams"
        stream_dir.mkdir(parents=True, exist_ok=True)
        record = dict(payload)
        record.setdefault("recorded_at_wall_time", time.time())
        record.setdefault("event_wall_time", event_time)
        with self._file_lock:
            with (stream_dir / f"{self.stream_name}.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=_json_default))
                handle.write("\n")

    def _episode_counts(self, episode_dir: Path) -> tuple[int, int, int, int, int]:
        with self._lock:
            stats = self._stats_by_episode.get(episode_dir, _AsyncEpisodeStats())
            return (
                stats.enqueued,
                stats.written,
                stats.skipped_due_to_rate_limit,
                stats.dropped_due_to_backpressure,
                stats.write_errors,
            )
