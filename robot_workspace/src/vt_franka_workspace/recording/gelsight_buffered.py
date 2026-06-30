from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .raw_recorder import _json_default


class SupportsActiveEpisodeDir(Protocol):
    def get_active_episode_dir(self) -> Path | None: ...


class GelsightBufferOverflow(RuntimeError):
    pass


@dataclass
class GelsightFrameSample:
    episode_dir: Path
    captured_wall_time: float
    sequence_id: int
    frame: np.ndarray
    metadata: dict[str, Any]


class BufferedGelsightFrameRecorder:
    def __init__(
        self,
        session_manager: SupportsActiveEpisodeDir,
        *,
        stream_name: str = "gelsight_frames",
        max_frames: int = 900,
        chunk_frames: int = 100,
    ) -> None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        self.session_manager = session_manager
        self.stream_name = stream_name
        self.max_frames = int(max_frames)
        self.chunk_frames = int(chunk_frames)
        self._lock = threading.Lock()
        self._episode_dir: Path | None = None
        self._samples: list[GelsightFrameSample] = []
        self._overflow_error: GelsightBufferOverflow | None = None
        self._total_recorded = 0

    def record_frame(
        self,
        frame: np.ndarray,
        *,
        captured_wall_time: float,
        sequence_id: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        episode_dir = self.session_manager.get_active_episode_dir()
        if episode_dir is None:
            return

        sample = GelsightFrameSample(
            episode_dir=episode_dir,
            captured_wall_time=float(captured_wall_time),
            sequence_id=int(sequence_id),
            frame=np.ascontiguousarray(frame.copy()),
            metadata=dict(metadata or {}),
        )
        with self._lock:
            if self._episode_dir != episode_dir:
                self._episode_dir = episode_dir
                self._samples.clear()
                self._overflow_error = None
            if self._overflow_error is not None:
                raise self._overflow_error
            if len(self._samples) >= self.max_frames:
                self._overflow_error = GelsightBufferOverflow(
                    f"GelSight frame buffer overflow for {episode_dir.name}: "
                    f"max_frames={self.max_frames}"
                )
                raise self._overflow_error
            self._samples.append(sample)
            self._total_recorded += 1

    def pop_overflow_error(self) -> GelsightBufferOverflow | None:
        with self._lock:
            error = self._overflow_error
            self._overflow_error = None
        return error

    def freeze_episode(self, episode_dir: Path) -> list[GelsightFrameSample]:
        with self._lock:
            samples = [sample for sample in self._samples if sample.episode_dir == episode_dir]
            if self._episode_dir == episode_dir:
                self._samples.clear()
                self._episode_dir = None
                self._overflow_error = None
        return samples

    def flush_episode(self, episode_dir: Path, samples: list[GelsightFrameSample] | None = None) -> dict[str, Any]:
        flush_start = time.time()
        samples = self.freeze_episode(episode_dir) if samples is None else list(samples)
        streams_dir = episode_dir / "streams"
        frame_dir = streams_dir / self.stream_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        index_path = streams_dir / f"{self.stream_name}.jsonl"
        tmp_index_path = streams_dir / f".{self.stream_name}.jsonl.tmp"

        chunk_count = 0
        frame_count = len(samples)
        with tmp_index_path.open("w", encoding="utf-8") as handle:
            for chunk_start in range(0, frame_count, self.chunk_frames):
                chunk_samples = samples[chunk_start : chunk_start + self.chunk_frames]
                chunk_path = frame_dir / f"chunk_{chunk_count:06d}.npz"
                tmp_chunk_path = frame_dir / f".chunk_{chunk_count:06d}.npz.tmp"
                frames = np.stack([sample.frame for sample in chunk_samples], axis=0) if chunk_samples else np.empty((0,))
                captured_wall_times = np.asarray(
                    [sample.captured_wall_time for sample in chunk_samples],
                    dtype=np.float64,
                )
                sequence_ids = np.asarray([sample.sequence_id for sample in chunk_samples], dtype=np.int64)
                with tmp_chunk_path.open("wb") as chunk_handle:
                    np.savez(
                        chunk_handle,
                        frames=frames,
                        captured_wall_times=captured_wall_times,
                        sequence_ids=sequence_ids,
                    )
                os.replace(tmp_chunk_path, chunk_path)
                rel_chunk_path = chunk_path.relative_to(episode_dir).as_posix()
                for index_in_chunk, sample in enumerate(chunk_samples):
                    record = {
                        "frame_path": rel_chunk_path,
                        "chunk_path": rel_chunk_path,
                        "chunk_index": chunk_count,
                        "index_in_chunk": index_in_chunk,
                        "captured_wall_time": sample.captured_wall_time,
                        "sequence_id": sample.sequence_id,
                        "frame_width": int(sample.frame.shape[1]) if sample.frame.ndim >= 2 else 0,
                        "frame_height": int(sample.frame.shape[0]) if sample.frame.ndim >= 2 else 0,
                        "frame_shape": list(sample.frame.shape),
                        "dtype": str(sample.frame.dtype),
                        "metadata": sample.metadata,
                        "recorded_at_wall_time": time.time(),
                    }
                    handle.write(json.dumps(record, default=_json_default))
                    handle.write("\n")
                chunk_count += 1
        os.replace(tmp_index_path, index_path)

        summary = {
            "stream_name": self.stream_name,
            "buffered_recording": True,
            "frame_count": frame_count,
            "chunk_count": chunk_count,
            "chunk_frames": self.chunk_frames,
            "max_frames": self.max_frames,
            "flush_duration_sec": time.time() - flush_start,
            "index_path": index_path.relative_to(episode_dir).as_posix(),
            "chunks_dir": frame_dir.relative_to(episode_dir).as_posix(),
        }
        manifest_path = frame_dir / "manifest.json"
        tmp_manifest_path = frame_dir / ".manifest.json.tmp"
        tmp_manifest_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        os.replace(tmp_manifest_path, manifest_path)
        return summary

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "active_episode": None if self._episode_dir is None else str(self._episode_dir),
                "queued_frames": len(self._samples),
                "max_frames": self.max_frames,
                "total_recorded": self._total_recorded,
                "overflow_error": None if self._overflow_error is None else str(self._overflow_error),
            }
