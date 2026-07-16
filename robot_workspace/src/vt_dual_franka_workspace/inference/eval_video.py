from __future__ import annotations

import json
from pathlib import Path


def write_rollout_video(
    episode_dir: str | Path,
    *,
    stream_name: str,
    output_name: str,
    fps: float,
) -> Path | None:
    episode_path = Path(episode_dir)
    event_path = episode_path / "streams" / f"{stream_name}.jsonl"
    if not event_path.exists():
        return None

    frame_paths = _frame_paths_from_events(episode_path, event_path)
    if not frame_paths:
        return None

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenCV is required to write eval rollout videos") from exc

    first_frame = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first_frame is None:
        return None
    height, width = first_frame.shape[:2]
    output_path = episode_path / output_name
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")
    try:
        writer.write(first_frame)
        for frame_path in frame_paths[1:]:
            frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
    finally:
        writer.release()
    return output_path


def _frame_paths_from_events(episode_dir: Path, event_path: Path) -> list[Path]:
    frame_paths: list[Path] = []
    seen: set[Path] = set()
    for line in event_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        frame_path = _frame_path_from_event(event)
        if frame_path is None:
            continue
        absolute_path = episode_dir / frame_path
        if absolute_path in seen or not absolute_path.exists():
            continue
        seen.add(absolute_path)
        frame_paths.append(absolute_path)
    return frame_paths


def _frame_path_from_event(event: dict) -> Path | None:
    frame_path = event.get("frame_path")
    if isinstance(frame_path, str) and frame_path:
        return Path(frame_path)
    metadata = event.get("metadata")
    if isinstance(metadata, dict):
        nested_frame_path = metadata.get("frame_path")
        if isinstance(nested_frame_path, str) and nested_frame_path:
            return Path(nested_frame_path)
    return None
