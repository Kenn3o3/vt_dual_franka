from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .image_io import ensure_hwc_uint8_rgb, write_rgb_jpeg
from .raw_recorder import _json_default


class SupportsActiveEpisodeDir(Protocol):
    def get_active_episode_dir(self) -> Path | None: ...


@dataclass
class BufferedImageFrame:
    image_rgb: np.ndarray
    captured_wall_time: float
    sequence_id: int | None
    metadata: dict[str, Any]


class EpisodeImageStreamRecorder:
    def __init__(
        self,
        session_manager: SupportsActiveEpisodeDir,
        stream_name: str,
        *,
        record_hz: float = 0.0,
        image_format: str = "jpg",
        jpeg_quality: int = 90,
        max_frames: int | None = None,
    ) -> None:
        self.session_manager = session_manager
        self.stream_name = stream_name
        self.record_hz = float(record_hz)
        self.image_format = image_format.lower().lstrip(".")
        self.jpeg_quality = int(jpeg_quality)
        self.max_frames = None if max_frames is None else int(max_frames)
        self._lock = threading.Lock()
        self._buffers: dict[Path, list[BufferedImageFrame]] = {}
        self._last_episode_dir: Path | None = None
        self._last_record_time: float | None = None
        self._frames_seen = 0
        self._frames_buffered = 0
        self._frames_skipped_rate = 0
        self._frames_dropped_capacity = 0
        self._last_flush_summary: dict[str, Any] | None = None

    def record_frame(
        self,
        image_rgb: np.ndarray,
        *,
        captured_wall_time: float,
        sequence_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        episode_dir = self.session_manager.get_active_episode_dir()
        with self._lock:
            self._frames_seen += 1
            if episode_dir is None:
                self._last_episode_dir = None
                self._last_record_time = None
                return False
            event_time = float(captured_wall_time)
            if not self._should_record_locked(episode_dir, event_time):
                self._frames_skipped_rate += 1
                return False
            buffer = self._buffers.setdefault(Path(episode_dir), [])
            if self.max_frames is not None and len(buffer) >= self.max_frames:
                self._frames_dropped_capacity += 1
                raise RuntimeError(
                    f"Episode image stream buffer overflow for {self.stream_name}; "
                    f"max_frames={self.max_frames}"
                )
            buffer.append(
                BufferedImageFrame(
                    image_rgb=ensure_hwc_uint8_rgb(image_rgb).copy(),
                    captured_wall_time=event_time,
                    sequence_id=sequence_id,
                    metadata=dict(metadata or {}),
                )
            )
            self._frames_buffered += 1
            return True

    def flush_episode(self, episode_dir: str | Path) -> dict[str, Any] | None:
        episode_dir = Path(episode_dir)
        with self._lock:
            frames = self._buffers.pop(episode_dir, [])
        if not frames:
            return None

        start = time.time()
        stream_dir = episode_dir / "streams" / self.stream_name
        frames_dir = stream_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        index_path = stream_dir / "index.jsonl"
        legacy_index_path = episode_dir / "streams" / f"{self.stream_name}.jsonl"
        records: list[dict[str, Any]] = []
        for frame_index, item in enumerate(frames):
            frame_path = frames_dir / f"{frame_index:06d}.{self.image_format}"
            if self.image_format not in {"jpg", "jpeg"}:
                raise ValueError(f"EpisodeImageStreamRecorder currently supports JPEG output, got {self.image_format!r}")
            write_rgb_jpeg(frame_path, item.image_rgb, quality=self.jpeg_quality)
            rel_path = frame_path.relative_to(episode_dir).as_posix()
            record = {
                "frame_index": frame_index,
                "frame_path": rel_path,
                "captured_wall_time": float(item.captured_wall_time),
                "sequence_id": item.sequence_id,
                "frame_width": int(item.image_rgb.shape[1]),
                "frame_height": int(item.image_rgb.shape[0]),
                "frame_shape": [int(v) for v in item.image_rgb.shape],
                "dtype": str(item.image_rgb.dtype),
                "color_space": "RGB",
                "metadata": item.metadata,
            }
            records.append(record)

        _write_jsonl(index_path, records)
        _write_jsonl(legacy_index_path, records)
        captured_times = [float(item.captured_wall_time) for item in frames]
        duration = max(captured_times) - min(captured_times) if len(captured_times) > 1 else 0.0
        effective_hz = (len(captured_times) - 1) / duration if duration > 0.0 and len(captured_times) > 1 else None
        manifest = {
            "schema_version": "vt_franka_standardized_image_stream_v1",
            "stream_name": self.stream_name,
            "image_format": self.image_format,
            "jpeg_quality": self.jpeg_quality,
            "frame_count": len(frames),
            "frame_width": 640,
            "frame_height": 480,
            "color_space": "RGB",
            "first_captured_wall_time": min(captured_times),
            "last_captured_wall_time": max(captured_times),
            "duration_sec": duration,
            "effective_hz": effective_hz,
            "flush_duration_sec": time.time() - start,
            "index_path": index_path.relative_to(episode_dir).as_posix(),
        }
        (stream_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
        with self._lock:
            self._last_flush_summary = manifest
        return manifest

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active_counts = {str(path): len(frames) for path, frames in self._buffers.items()}
            return {
                "stream_name": self.stream_name,
                "record_hz": self.record_hz,
                "frames_seen": self._frames_seen,
                "frames_buffered": self._frames_buffered,
                "frames_skipped_due_to_rate_limit": self._frames_skipped_rate,
                "frames_dropped_due_to_capacity": self._frames_dropped_capacity,
                "active_episode_frames": active_counts,
                "last_flush_summary": self._last_flush_summary,
            }

    def _should_record_locked(self, episode_dir: Path, event_time: float) -> bool:
        if self._last_episode_dir != episode_dir:
            self._last_episode_dir = episode_dir
            self._last_record_time = None
        if self.record_hz <= 0.0:
            self._last_record_time = event_time
            return True
        if self._last_record_time is None or event_time - self._last_record_time >= 1.0 / self.record_hz:
            self._last_record_time = event_time
            return True
        return False


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=_json_default))
            handle.write("\n")
