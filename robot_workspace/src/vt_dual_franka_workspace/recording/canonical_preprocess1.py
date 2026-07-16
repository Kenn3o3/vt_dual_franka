from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..policies.visuotactile.image_preprocess import (
    ImagePreprocessSpec,
    bgr_to_rgb,
    default_preprocess1_specs,
    preprocess_image_rgb,
    rgb_to_bgr,
)
from .raw_recorder import _json_default


class SupportsActiveEpisodeDir(Protocol):
    def get_active_episode_dir(self) -> Path | None: ...


class CanonicalPreprocessBackpressure(RuntimeError):
    pass


@dataclass(frozen=True)
class CanonicalStreamSpec:
    stream_name: str
    canonical_key: str
    preprocess: ImagePreprocessSpec
    source_color_order: str = "bgr_opencv"
    canonical_color_order: str = "rgb"
    raw_jpeg_compat: bool = False

    def manifest_payload(self) -> dict[str, Any]:
        transforms: list[dict[str, Any]] = []
        if self.raw_jpeg_compat:
            transforms.append({"name": "raw_recording_jpeg_roundtrip", "quality": None})
        transforms.extend(
            [
                {"name": "bgr_to_rgb"},
                {"name": "preprocess1", "preprocess": self.preprocess.to_json()},
            ]
        )
        return {
            "stream_name": self.stream_name,
            "canonical_key": self.canonical_key,
            "source_stage": "collection_live_bgr",
            "source_color_order": self.source_color_order,
            "canonical_color_order": self.canonical_color_order,
            "compatibility_transforms": transforms,
            "preprocess": self.preprocess.to_json(),
        }


@dataclass
class _CanonicalSample:
    episode_dir: Path
    captured_wall_time: float
    sequence_id: int
    frame: np.ndarray
    metadata: dict[str, Any]
    live_buffer: Any | None = None


@dataclass
class _EpisodeBuffer:
    episode_dir: Path
    frames: list[np.ndarray] = field(default_factory=list)
    captured_wall_times: list[float] = field(default_factory=list)
    sequence_ids: list[int] = field(default_factory=list)
    source_shapes: list[list[int]] = field(default_factory=list)
    records: list[dict[str, Any]] = field(default_factory=list)
    chunk_count: int = 0
    frame_count: int = 0
    skipped_due_to_rate_limit: int = 0
    started_at_wall_time: float = field(default_factory=time.time)
    preprocess_durations_sec: list[float] = field(default_factory=list)
    write_durations_sec: list[float] = field(default_factory=list)


class CanonicalPreprocess1StreamRecorder:
    def __init__(
        self,
        session_manager: SupportsActiveEpisodeDir,
        spec: CanonicalStreamSpec,
        *,
        queue_size: int = 4,
        chunk_frames: int = 64,
        record_hz: float = 0.0,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        if record_hz < 0.0:
            raise ValueError("record_hz must be non-negative")
        self.session_manager = session_manager
        self.spec = spec
        self.queue_size = int(queue_size)
        self.chunk_frames = int(chunk_frames)
        self.record_hz = float(record_hz)
        self._queue: queue.Queue[_CanonicalSample | None] = queue.Queue(maxsize=self.queue_size)
        self._lock = threading.Lock()
        self._buffers: dict[Path, _EpisodeBuffer] = {}
        self._next_record_time_by_episode: dict[Path, float] = {}
        self._skipped_due_to_rate_limit_by_episode: dict[Path, int] = {}
        self._worker = threading.Thread(target=self._run_worker, name=f"preprocess1:{spec.stream_name}", daemon=True)
        self._started = False
        self._closed = False
        self._error: Exception | None = None
        self._total_enqueued = 0
        self._total_processed = 0
        self._skipped_due_to_rate_limit = 0
        self._dropped_due_to_backpressure = 0

    def start(self) -> None:
        with self._lock:
            if self._started:
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
        self._queue.put(None)
        self._worker.join(timeout=timeout_sec)
        if self._worker.is_alive():
            raise RuntimeError(f"Timed out closing preprocess1 recorder for {self.spec.stream_name}")

    def record_frame(
        self,
        frame_bgr: np.ndarray,
        *,
        captured_wall_time: float,
        sequence_id: int,
        metadata: dict[str, Any] | None = None,
        live_buffer: Any | None = None,
    ) -> None:
        episode_dir = self.session_manager.get_active_episode_dir()
        if episode_dir is None:
            return
        captured_wall_time = float(captured_wall_time)
        if not self._should_record(episode_dir, captured_wall_time):
            return
        self.start()
        self._raise_if_failed()
        sample = _CanonicalSample(
            episode_dir=episode_dir,
            captured_wall_time=captured_wall_time,
            sequence_id=int(sequence_id),
            frame=np.ascontiguousarray(frame_bgr),
            metadata=dict(metadata or {}),
            live_buffer=live_buffer,
        )
        try:
            self._queue.put_nowait(sample)
        except queue.Full as exc:
            with self._lock:
                self._dropped_due_to_backpressure += 1
                self._error = CanonicalPreprocessBackpressure(
                    f"preprocess1 queue overflow for {self.spec.stream_name}; "
                    f"queue_size={self.queue_size}"
                )
            raise self._error from exc
        with self._lock:
            self._total_enqueued += 1

    def flush_episode(self, episode_dir: Path) -> dict[str, Any] | None:
        self._queue.join()
        self._raise_if_failed()
        with self._lock:
            episode_dir = Path(episode_dir)
            buffer = self._buffers.pop(episode_dir, None)
            self._next_record_time_by_episode.pop(episode_dir, None)
            skipped_due_to_rate_limit = self._skipped_due_to_rate_limit_by_episode.pop(episode_dir, 0)
        if buffer is None:
            return None
        buffer.skipped_due_to_rate_limit = skipped_due_to_rate_limit
        self._flush_buffer(buffer, final=True)
        return self._write_stream_manifest(buffer)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "stream_name": self.spec.stream_name,
                "canonical_key": self.spec.canonical_key,
                "record_hz": self.record_hz,
                "queue_size": self.queue_size,
                "queued_frames": self._queue.qsize(),
                "total_enqueued": self._total_enqueued,
                "total_processed": self._total_processed,
                "skipped_due_to_rate_limit": self._skipped_due_to_rate_limit,
                "dropped_due_to_backpressure": self._dropped_due_to_backpressure,
                "error": None if self._error is None else str(self._error),
            }

    @property
    def error(self) -> Exception | None:
        with self._lock:
            return self._error

    def _run_worker(self) -> None:
        while True:
            sample = self._queue.get()
            try:
                if sample is None:
                    return
                self._process_sample(sample)
            except Exception as exc:
                with self._lock:
                    if self._error is None:
                        self._error = exc
            finally:
                self._queue.task_done()

    def _process_sample(self, sample: _CanonicalSample) -> None:
        started = time.time()
        canonical = preprocess_bgr_to_canonical_rgb(sample.frame, self.spec)
        duration = time.time() - started
        with self._lock:
            buffer = self._buffers.get(sample.episode_dir)
            if buffer is None:
                buffer = _EpisodeBuffer(sample.episode_dir)
                self._buffers[sample.episode_dir] = buffer
            buffer.frames.append(canonical)
            buffer.captured_wall_times.append(sample.captured_wall_time)
            buffer.sequence_ids.append(sample.sequence_id)
            buffer.source_shapes.append(list(sample.frame.shape))
            buffer.preprocess_durations_sec.append(duration)
            self._total_processed += 1
            should_flush = len(buffer.frames) >= self.chunk_frames
        if should_flush:
            self._flush_buffer(buffer, final=False)
        if sample.live_buffer is not None:
            live_metadata = dict(sample.metadata)
            live_metadata["canonical_shape"] = list(canonical.shape)
            sample.live_buffer.update(
                canonical.copy(),
                metadata=live_metadata,
                captured_wall_time=sample.captured_wall_time,
            )

    def _flush_buffer(self, buffer: _EpisodeBuffer, *, final: bool) -> None:
        del final
        with self._lock:
            if not buffer.frames:
                return
            frames = np.stack(buffer.frames, axis=0).astype(np.uint8)
            captured_wall_times = np.asarray(buffer.captured_wall_times, dtype=np.float64)
            sequence_ids = np.asarray(buffer.sequence_ids, dtype=np.int64)
            source_shapes = list(buffer.source_shapes)
            buffer.frames.clear()
            buffer.captured_wall_times.clear()
            buffer.sequence_ids.clear()
            buffer.source_shapes.clear()
            chunk_index = buffer.chunk_count
            buffer.chunk_count += 1
        streams_dir = buffer.episode_dir / "streams"
        frame_dir = streams_dir / self.spec.stream_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        chunk_name = f"chunk_{chunk_index:06d}.npz"
        chunk_path = frame_dir / chunk_name
        tmp_chunk_path = frame_dir / f".{chunk_name}.tmp"
        write_started = time.time()
        with tmp_chunk_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                frames=frames,
                captured_wall_times=captured_wall_times,
                sequence_ids=sequence_ids,
            )
        os.replace(tmp_chunk_path, chunk_path)
        write_duration = time.time() - write_started
        rel_chunk_path = chunk_path.relative_to(buffer.episode_dir).as_posix()
        records: list[dict[str, Any]] = []
        for index_in_chunk, captured_wall_time in enumerate(captured_wall_times):
            record = {
                "frame_path": rel_chunk_path,
                "chunk_path": rel_chunk_path,
                "chunk_index": int(chunk_index),
                "index_in_chunk": int(index_in_chunk),
                "captured_wall_time": float(captured_wall_time),
                "sequence_id": int(sequence_ids[index_in_chunk]),
                "frame_width": int(frames.shape[2]),
                "frame_height": int(frames.shape[1]),
                "frame_shape": [int(value) for value in frames[index_in_chunk].shape],
                "source_frame_shape": source_shapes[index_in_chunk],
                "dtype": str(frames.dtype),
                "canonical_key": self.spec.canonical_key,
                "recorded_at_wall_time": time.time(),
            }
            records.append(record)
        index_path = streams_dir / f"{self.spec.stream_name}.jsonl"
        with index_path.open("a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, default=_json_default))
                handle.write("\n")
        with self._lock:
            buffer.records.extend(records)
            buffer.frame_count += len(records)
            buffer.write_durations_sec.append(write_duration)

    def _write_stream_manifest(self, buffer: _EpisodeBuffer) -> dict[str, Any]:
        frame_dir = buffer.episode_dir / "streams" / self.spec.stream_name
        summary = {
            "schema_version": "vt_franka_preprocess1_stream_v1",
            "stream_name": self.spec.stream_name,
            "canonical_key": self.spec.canonical_key,
            "collection_time_preprocess1": True,
            "frame_count": int(buffer.frame_count),
            "skipped_due_to_rate_limit": int(buffer.skipped_due_to_rate_limit),
            "chunk_count": int(buffer.chunk_count),
            "chunk_frames": int(self.chunk_frames),
            "queue_size": int(self.queue_size),
            "record_hz": float(self.record_hz),
            "index_path": f"streams/{self.spec.stream_name}.jsonl",
            "chunks_dir": f"streams/{self.spec.stream_name}",
            "first_captured_wall_time": None if not buffer.records else float(buffer.records[0]["captured_wall_time"]),
            "last_captured_wall_time": None if not buffer.records else float(buffer.records[-1]["captured_wall_time"]),
            "source_shape": None if not buffer.records else buffer.records[0].get("source_frame_shape"),
            "canonical_shape": None if not buffer.records else buffer.records[0].get("frame_shape"),
            "preprocess": self.spec.preprocess.to_json(),
            "compatibility_transforms": self.spec.manifest_payload()["compatibility_transforms"],
            "mean_preprocess_duration_sec": _mean(buffer.preprocess_durations_sec),
            "max_preprocess_duration_sec": max(buffer.preprocess_durations_sec) if buffer.preprocess_durations_sec else None,
            "mean_write_duration_sec": _mean(buffer.write_durations_sec),
            "max_write_duration_sec": max(buffer.write_durations_sec) if buffer.write_durations_sec else None,
        }
        manifest_path = frame_dir / "manifest.json"
        tmp_manifest_path = frame_dir / ".manifest.json.tmp"
        tmp_manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        os.replace(tmp_manifest_path, manifest_path)
        return summary

    def _raise_if_failed(self) -> None:
        error = self.error
        if error is not None:
            raise error

    def _should_record(self, episode_dir: Path, captured_wall_time: float) -> bool:
        with self._lock:
            if self.record_hz <= 0.0:
                return True
            period = 1.0 / self.record_hz
            next_record_time = self._next_record_time_by_episode.get(episode_dir)
            if next_record_time is None:
                self._next_record_time_by_episode[episode_dir] = captured_wall_time + period
                return True
            if captured_wall_time >= next_record_time - 1e-9:
                if captured_wall_time <= next_record_time + 1e-9:
                    self._next_record_time_by_episode[episode_dir] = next_record_time + period
                else:
                    skipped_slots = int((captured_wall_time - next_record_time) // period)
                    self._next_record_time_by_episode[episode_dir] = next_record_time + (skipped_slots + 1) * period
                return True
            self._skipped_due_to_rate_limit += 1
            self._skipped_due_to_rate_limit_by_episode[episode_dir] = (
                self._skipped_due_to_rate_limit_by_episode.get(episode_dir, 0) + 1
            )
            return False


def default_canonical_stream_specs(
    *,
    canonical_size: int,
    gelsight_crop_box: tuple[int, int, int, int] | None,
    gelsight_margin_fraction: float,
    wrist_raw_jpeg_compat: bool,
) -> dict[str, CanonicalStreamSpec]:
    preprocess = default_preprocess1_specs(
        canonical_size=canonical_size,
        gelsight_crop_box=gelsight_crop_box,
        gelsight_margin_fraction=gelsight_margin_fraction,
    )
    return {
        "rgb_wrist": CanonicalStreamSpec(
            stream_name="preprocess1_rgb_wrist",
            canonical_key="rgb_wrist",
            preprocess=preprocess["rgb_wrist"],
            raw_jpeg_compat=bool(wrist_raw_jpeg_compat),
        ),
        "gelsight": CanonicalStreamSpec(
            stream_name="preprocess1_gelsight",
            canonical_key="gelsight",
            preprocess=preprocess["gelsight"],
            raw_jpeg_compat=False,
        ),
    }


def preprocess_bgr_to_canonical_rgb(frame_bgr: np.ndarray, spec: CanonicalStreamSpec) -> np.ndarray:
    image_bgr = np.asarray(frame_bgr, dtype=np.uint8)
    if spec.raw_jpeg_compat:
        image_bgr = _jpeg_roundtrip_bgr(image_bgr)
    image_rgb = bgr_to_rgb(image_bgr)
    return preprocess_image_rgb(image_rgb, spec.preprocess)


def read_canonical_stream_frame(episode_dir: Path, rel_path: str, index_in_chunk: int) -> np.ndarray:
    if index_in_chunk < 0:
        raise ValueError(f"Canonical stream chunk frame requires index_in_chunk: {rel_path}")
    with np.load(Path(episode_dir) / rel_path) as chunk:
        frames = chunk["frames"]
        if index_in_chunk >= len(frames):
            raise IndexError(f"Canonical stream index {index_in_chunk} out of range for {rel_path}")
        return np.asarray(frames[index_in_chunk], dtype=np.uint8)


def _jpeg_roundtrip_bgr(image_bgr: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for collection-time preprocess1 JPEG compatibility") from exc
    ok, payload = cv2.imencode(".jpg", np.asarray(image_bgr, dtype=np.uint8))
    if not ok:
        raise RuntimeError("OpenCV failed to JPEG-encode collection-time preprocess1 image")
    decoded = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("OpenCV failed to JPEG-decode collection-time preprocess1 image")
    return np.ascontiguousarray(decoded)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))
