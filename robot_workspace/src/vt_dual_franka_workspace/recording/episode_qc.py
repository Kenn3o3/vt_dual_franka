from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping


TIMESTAMP_KEYS_BY_STREAM = {
    "controller_state_by_arm": ("received_wall_time", "source_wall_time", "recorded_at_wall_time"),
    "quest_messages": ("source_wall_time", "recorded_at_wall_time"),
    "teleop_commands": ("source_wall_time", "recorded_at_wall_time"),
    "gelsight_frames": ("captured_wall_time", "recorded_at_wall_time"),
}
FALLBACK_TIMESTAMP_KEYS = (
    "captured_wall_time",
    "source_wall_time",
    "received_wall_time",
    "recorded_at_wall_time",
    "wall_time",
    "timestamp",
)
GAP_THRESHOLDS_SEC = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0)


def build_expected_episode_hz(workspace: Any, task: Any) -> dict[str, float]:
    expected: dict[str, float] = {}
    if getattr(task.modality, "proprioception", False):
        expected["controller_state_by_arm"] = float(task.collection.controller_state_poll_hz)
    if getattr(task.collection, "record_raw_quest_messages", False):
        quest_hz = float(getattr(workspace.teleop, "quest_message_record_hz", 0.0))
        expected["quest_messages"] = quest_hz if quest_hz > 0.0 else float(workspace.teleop.loop_hz)
    command_hz = float(getattr(workspace.teleop, "command_record_hz", 0.0))
    if command_hz > 0.0:
        expected["teleop_commands"] = command_hz
    for role in getattr(task.modality, "rgb_cameras", []):
        settings = task.rgb_cameras.get(role)
        if settings is None:
            continue
        stream_name = settings.stream_name or f"rgb_{role}"
        record_hz = float(settings.record_hz)
        expected[stream_name] = record_hz if record_hz > 0.0 else float(settings.color_fps)
        preprocess1 = getattr(getattr(task, "collection", None), "preprocess1_recording", None)
        if role == "wrist" and preprocess1 is not None and getattr(preprocess1, "enabled", False):
            expected["preprocess1_rgb_wrist"] = float(getattr(preprocess1, "target_hz", settings.color_fps))
    if task.modality.needs_gelsight():
        for arm_id, settings in task.gelsights.items():
            if not settings.enabled:
                continue
            record_hz = float(settings.record_hz)
            expected[f"tactile_{arm_id}"] = record_hz if record_hz > 0.0 else float(settings.fps)
    return expected


def analyze_episode_quality(
    episode_dir: str | Path,
    *,
    expected_hz: Mapping[str, float] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    episode_dir = Path(episode_dir)
    streams_dir = episode_dir / "streams"
    expected_hz = dict(expected_hz or {})
    manifest = _read_json(episode_dir / "episode_manifest.json")
    episode_start = _finite_float(manifest.get("started_at_wall_time"))
    episode_stop = _finite_float(manifest.get("stopped_at_wall_time"))
    episode_duration = (
        episode_stop - episode_start
        if episode_start is not None and episode_stop is not None and episode_stop >= episode_start
        else None
    )

    streams: dict[str, dict[str, Any]] = {}
    if streams_dir.exists():
        for path in sorted(streams_dir.glob("*.jsonl")):
            stream_name = path.stem
            streams[stream_name] = _analyze_stream(
                episode_dir,
                path,
                stream_name,
                expected_hz=expected_hz.get(stream_name),
                episode_start=episode_start,
                episode_stop=episode_stop,
                episode_duration=episode_duration,
            )

    warnings: list[str] = []
    for stream_name, stats in streams.items():
        for warning in stats["warnings"]:
            warnings.append(f"{stream_name}: {warning}")

    summary = _build_summary(streams, warnings)
    report = {
        "schema_version": "vt_dual_franka_episode_qc_v1",
        "generated_at_wall_time": time.time(),
        "episode_dir": str(episode_dir),
        "episode": {
            "episode_id": manifest.get("episode_id"),
            "episode_name": manifest.get("episode_name"),
            "outcome": manifest.get("outcome"),
            "started_at_wall_time": episode_start,
            "stopped_at_wall_time": episode_stop,
            "duration_sec": episode_duration,
        },
        "expected_hz": {key: value for key, value in expected_hz.items() if key in streams},
        "streams": streams,
        "warnings": warnings,
        "summary": summary,
    }
    if write:
        (episode_dir / "episode_qc.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def episode_qc_manifest_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    streams = {}
    for stream_name, stats in report.get("streams", {}).items():
        streams[stream_name] = {
            "record_count": stats.get("record_count"),
            "effective_hz": stats.get("effective_hz"),
            "max_gap_sec": stats.get("max_gap_sec"),
            "p95_gap_sec": stats.get("p95_gap_sec"),
            "expected_hz": stats.get("expected_hz"),
            "warning_count": len(stats.get("warnings", [])),
        }
    return {
        "path": "episode_qc.json",
        "warning_count": len(report.get("warnings", [])),
        "streams": streams,
    }


def format_episode_qc_summary(report: Mapping[str, Any]) -> str:
    parts = []
    for stream_name, stats in sorted(report.get("streams", {}).items()):
        count = stats.get("record_count")
        hz = _format_float(stats.get("effective_hz"), digits=2)
        max_gap = _format_float(stats.get("max_gap_sec"), digits=3)
        if hz == "n/a":
            parts.append(f"{stream_name}: n={count}")
        else:
            parts.append(f"{stream_name}: {hz}Hz max_gap={max_gap}s n={count}")
    warning_count = len(report.get("warnings", []))
    suffix = f"; warnings={warning_count}" if warning_count else ""
    return "; ".join(parts) + suffix


def _analyze_stream(
    episode_dir: Path,
    path: Path,
    stream_name: str,
    *,
    expected_hz: float | None,
    episode_start: float | None,
    episode_stop: float | None,
    episode_duration: float | None,
) -> dict[str, Any]:
    records, malformed_line_count = _read_jsonl_records(path)
    timestamp_key = _select_timestamp_key(stream_name, records)
    timestamps: list[float] = []
    missing_timestamp_count = 0
    if timestamp_key is None:
        missing_timestamp_count = len(records)
    else:
        for record in records:
            value = _finite_float(record.get(timestamp_key))
            if value is None:
                missing_timestamp_count += 1
                continue
            timestamps.append(value)
    in_episode_timestamps = _filter_episode_window(timestamps, episode_start, episode_stop)

    stats = _timestamp_stats(in_episode_timestamps)
    stats["raw_timestamp_count"] = len(timestamps)
    stats["timestamp_count_outside_episode"] = len(timestamps) - len(in_episode_timestamps)
    warnings = _stream_warnings(
        record_count=len(records),
        malformed_line_count=malformed_line_count,
        missing_timestamp_count=missing_timestamp_count,
        stats=stats,
        expected_hz=expected_hz,
        episode_duration=episode_duration,
    )
    frame_stats = _frame_path_stats(episode_dir, records)
    if frame_stats["missing_frame_file_count"] > 0:
        warnings.append(f"{frame_stats['missing_frame_file_count']} referenced frame files are missing")

    coverage = _coverage_stats(in_episode_timestamps, episode_start, episode_stop, episode_duration)
    if expected_hz is not None and coverage.get("coverage_fraction") is not None and coverage["coverage_fraction"] < 0.9:
        warnings.append(f"stream covers only {coverage['coverage_fraction']:.1%} of episode duration")

    return {
        "path": path.relative_to(episode_dir).as_posix(),
        "record_count": len(records),
        "malformed_line_count": malformed_line_count,
        "timestamp_key": timestamp_key,
        "missing_timestamp_count": missing_timestamp_count,
        "expected_hz": expected_hz,
        **stats,
        **coverage,
        **_sequence_stats(records),
        **frame_stats,
        "warnings": warnings,
    }


def _timestamp_stats(timestamps: list[float]) -> dict[str, Any]:
    if len(timestamps) < 2:
        return {
            "timestamp_count": len(timestamps),
            "start_wall_time": timestamps[0] if timestamps else None,
            "end_wall_time": timestamps[-1] if timestamps else None,
            "span_sec": 0.0 if timestamps else None,
            "effective_hz": None,
            "min_gap_sec": None,
            "median_gap_sec": None,
            "p95_gap_sec": None,
            "p99_gap_sec": None,
            "max_gap_sec": None,
            "max_gap_start_wall_time": None,
            "max_gap_end_wall_time": None,
            "nonmonotonic_timestamp_count": 0,
            "nonpositive_gap_count": 0,
            "gap_counts_over_sec": {str(threshold): 0 for threshold in GAP_THRESHOLDS_SEC},
        }

    deltas = [right - left for left, right in zip(timestamps, timestamps[1:])]
    positive_deltas = [delta for delta in deltas if delta > 0.0]
    span = timestamps[-1] - timestamps[0]
    max_gap_index = max(range(len(deltas)), key=lambda index: deltas[index])
    return {
        "timestamp_count": len(timestamps),
        "start_wall_time": timestamps[0],
        "end_wall_time": timestamps[-1],
        "span_sec": span if span >= 0.0 else None,
        "effective_hz": (len(timestamps) - 1) / span if span > 0.0 else None,
        "min_gap_sec": min(positive_deltas) if positive_deltas else None,
        "median_gap_sec": _quantile(positive_deltas, 0.5),
        "p95_gap_sec": _quantile(positive_deltas, 0.95),
        "p99_gap_sec": _quantile(positive_deltas, 0.99),
        "max_gap_sec": max(positive_deltas) if positive_deltas else None,
        "max_gap_start_wall_time": timestamps[max_gap_index],
        "max_gap_end_wall_time": timestamps[max_gap_index + 1],
        "nonmonotonic_timestamp_count": sum(1 for delta in deltas if delta < 0.0),
        "nonpositive_gap_count": sum(1 for delta in deltas if delta <= 0.0),
        "gap_counts_over_sec": {str(threshold): sum(1 for delta in positive_deltas if delta > threshold) for threshold in GAP_THRESHOLDS_SEC},
    }


def _coverage_stats(
    timestamps: list[float],
    episode_start: float | None,
    episode_stop: float | None,
    episode_duration: float | None,
) -> dict[str, Any]:
    if not timestamps or episode_start is None or episode_stop is None or episode_duration in (None, 0.0):
        return {
            "coverage_start_lag_sec": None,
            "coverage_end_lag_sec": None,
            "coverage_fraction": None,
        }
    stream_span = max(0.0, timestamps[-1] - timestamps[0])
    return {
        "coverage_start_lag_sec": timestamps[0] - episode_start,
        "coverage_end_lag_sec": episode_stop - timestamps[-1],
        "coverage_fraction": stream_span / episode_duration,
    }


def _filter_episode_window(
    timestamps: list[float],
    episode_start: float | None,
    episode_stop: float | None,
) -> list[float]:
    if episode_start is None or episode_stop is None or episode_stop < episode_start:
        return timestamps
    return [timestamp for timestamp in timestamps if episode_start <= timestamp <= episode_stop]


def _sequence_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    sequence_ids = [_finite_int(record.get("sequence_id")) for record in records if record.get("sequence_id") is not None]
    if len(sequence_ids) < 2:
        return {
            "sequence_count": len(sequence_ids),
            "sequence_first": sequence_ids[0] if sequence_ids else None,
            "sequence_last": sequence_ids[-1] if sequence_ids else None,
            "sequence_step_median": None,
            "nonmonotonic_sequence_count": 0,
        }
    deltas = [right - left for left, right in zip(sequence_ids, sequence_ids[1:])]
    return {
        "sequence_count": len(sequence_ids),
        "sequence_first": sequence_ids[0],
        "sequence_last": sequence_ids[-1],
        "sequence_step_median": _quantile([float(delta) for delta in deltas], 0.5),
        "nonmonotonic_sequence_count": sum(1 for delta in deltas if delta <= 0),
    }


def _frame_path_stats(episode_dir: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    frame_paths = [record.get("frame_path") or record.get("chunk_path") for record in records]
    frame_paths = [str(path) for path in frame_paths if path]
    unique_paths = sorted(set(frame_paths))
    missing = 0
    total_bytes = 0
    for rel_path in unique_paths:
        path = episode_dir / rel_path
        if not path.exists():
            missing += 1
            continue
        total_bytes += path.stat().st_size
    chunk_indices = [_finite_int(record.get("index_in_chunk")) for record in records if record.get("index_in_chunk") is not None]
    return {
        "frame_reference_count": len(frame_paths),
        "unique_frame_file_count": len(unique_paths),
        "missing_frame_file_count": missing,
        "unique_frame_file_bytes": total_bytes,
        "chunk_index_count": len(chunk_indices),
        "invalid_chunk_index_count": sum(1 for index in chunk_indices if index is None or index < 0),
    }


def _stream_warnings(
    *,
    record_count: int,
    malformed_line_count: int,
    missing_timestamp_count: int,
    stats: Mapping[str, Any],
    expected_hz: float | None,
    episode_duration: float | None,
) -> list[str]:
    warnings: list[str] = []
    if record_count == 0:
        warnings.append("stream has no records")
    if malformed_line_count:
        warnings.append(f"{malformed_line_count} malformed JSONL lines")
    if missing_timestamp_count:
        warnings.append(f"{missing_timestamp_count} records missing the selected timestamp")
    if stats.get("nonmonotonic_timestamp_count"):
        warnings.append(f"{stats['nonmonotonic_timestamp_count']} timestamp order violations")
    if expected_hz is not None and expected_hz > 0.0 and stats.get("effective_hz") is not None:
        ratio = float(stats["effective_hz"]) / expected_hz
        if ratio < 0.75:
            warnings.append(f"effective_hz is {ratio:.1%} of expected {expected_hz:.2f}Hz")
        max_gap = stats.get("max_gap_sec")
        if max_gap is not None and max_gap > max(0.5, 3.0 / expected_hz):
            warnings.append(f"max_gap_sec {max_gap:.3f}s is high for expected {expected_hz:.2f}Hz")
    if episode_duration is not None and record_count == 1:
        warnings.append("only one timestamped sample; Hz/gap cannot be estimated")
    return warnings


def _select_timestamp_key(stream_name: str, records: list[dict[str, Any]]) -> str | None:
    candidates = TIMESTAMP_KEYS_BY_STREAM.get(stream_name, FALLBACK_TIMESTAMP_KEYS)
    for key in candidates:
        if any(record.get(key) is not None for record in records):
            return key
    for key in FALLBACK_TIMESTAMP_KEYS:
        if any(record.get(key) is not None for record in records):
            return key
    return None


def _read_jsonl_records(path: Path) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    malformed = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(record, dict):
                records.append(record)
            else:
                malformed += 1
    return records, malformed


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _build_summary(streams: Mapping[str, Mapping[str, Any]], warnings: list[str]) -> dict[str, Any]:
    return {
        "stream_count": len(streams),
        "total_record_count": sum(int(stats.get("record_count", 0)) for stats in streams.values()),
        "warning_count": len(warnings),
        "streams_with_warnings": [name for name, stats in streams.items() if stats.get("warnings")],
    }


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _finite_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _format_float(value: Any, *, digits: int) -> str:
    number = _finite_float(value)
    if number is None:
        return "n/a"
    return f"{number:.{digits}f}"
