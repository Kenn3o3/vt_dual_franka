#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any


STATE_NAMES = ("x", "y", "z", "r0", "r1", "r2", "r3", "r4", "r5", "gripper")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect policy action chunks recorded in policy_steps.jsonl.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="policy_steps.jsonl files, episode directories, or run directories containing episodes/.",
    )
    parser.add_argument(
        "--source",
        choices=("actions_returned", "actions_executed"),
        default="actions_returned",
        help="Which action list to inspect. Defaults to raw policy outputs.",
    )
    parser.add_argument("--summary-only", action="store_true", help="Only print per-chunk summaries.")
    parser.add_argument("--chunk", type=int, default=None, help="Only print one zero-based chunk index.")
    parser.add_argument("--precision", type=int, default=4, help="Number of decimals to print.")
    parser.add_argument("--close-threshold", type=float, default=0.5)
    parser.add_argument("--open-threshold", type=float, default=0.3)
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV export path.")
    args = parser.parse_args()

    jsonl_paths = collect_policy_step_paths([Path(path) for path in args.paths])
    if not jsonl_paths:
        raise SystemExit("No policy_steps.jsonl files found.")

    csv_rows: list[dict[str, Any]] = []
    for path in jsonl_paths:
        rows = inspect_path(
            path,
            source=args.source,
            summary_only=args.summary_only,
            chunk_filter=args.chunk,
            precision=args.precision,
            close_threshold=args.close_threshold,
            open_threshold=args.open_threshold,
        )
        csv_rows.extend(rows)

    if args.csv is not None:
        write_csv(args.csv, csv_rows)
        print(f"\n[ok] wrote {args.csv}")


def inspect_path(
    path: Path,
    *,
    source: str,
    summary_only: bool,
    chunk_filter: int | None,
    precision: int,
    close_threshold: float,
    open_threshold: float,
) -> list[dict[str, Any]]:
    chunks = load_chunks(path, source)
    if chunk_filter is not None:
        chunks = [chunk for chunk in chunks if chunk["chunk_index"] == chunk_filter]

    rows = [row for chunk in chunks for row in chunk["rows"]]
    gripper_values = [float(row["gripper"]) for row in rows]

    print(f"\n{path}")
    print(
        f"source={source} chunks={len(chunks)} actions={len(rows)} "
        f"gripper={format_stats(gripper_values, precision)} "
        f">={close_threshold:g}:{count_ge(gripper_values, close_threshold)} "
        f"<={open_threshold:g}:{count_le(gripper_values, open_threshold)}"
    )

    for chunk in chunks:
        chunk_rows = chunk["rows"]
        values = [float(row["gripper"]) for row in chunk_rows]
        close_cmds = sum(1 for row in chunk_rows if row["command"] == "close")
        open_cmds = sum(1 for row in chunk_rows if row["command"] == "open")
        print(
            f"\nchunk={chunk['chunk_index']:03d} step={chunk['step_index']} "
            f"t={format_optional_float(chunk['episode_elapsed_sec'], precision)}s "
            f"infer={format_optional_float(chunk['inference_duration_sec'], precision)}s "
            f"n={len(chunk_rows)} gripper={format_stats(values, precision)} "
            f"cmd close/open={close_cmds}/{open_cmds}"
        )
        if not summary_only:
            print(format_header())
            for row in chunk_rows:
                print(format_row(row, precision))
    return rows


def load_chunks(path: Path, source: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for line_number, event in iter_json_events(path):
        if event.get("phase") != "policy_chunk":
            continue
        chunk_index = len(chunks)
        timing = event.get("timing") or {}
        elapsed = maybe_float(event.get("episode_elapsed_sec"))
        rows = []
        for action_index, action in enumerate(event.get(source, [])):
            state = extract_mpd_tcp_state(action)
            if state is None:
                continue
            duration = maybe_float(action.get("target_duration_sec")) or 0.0
            rows.append(
                {
                    "policy_steps_path": str(path),
                    "line_number": line_number,
                    "chunk_index": chunk_index,
                    "step_index": event.get("step_index"),
                    "chunk_action_index": action_index,
                    "episode_elapsed_sec": elapsed,
                    "action_time_sec": None if elapsed is None else elapsed + action_index * duration,
                    "target_duration_sec": duration,
                    "command": gripper_command(action),
                    **{name: float(state[index]) for index, name in enumerate(STATE_NAMES)},
                }
            )
        chunks.append(
            {
                "chunk_index": chunk_index,
                "step_index": event.get("step_index"),
                "episode_elapsed_sec": elapsed,
                "inference_duration_sec": maybe_float(timing.get("inference_duration_sec")),
                "rows": rows,
            }
        )
    return chunks


def iter_json_events(path: Path):
    buffer = ""
    start_line = 0
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip() and not buffer:
                continue
            if not buffer:
                start_line = line_number
            buffer += line
            while buffer:
                stripped = buffer.lstrip()
                if not stripped:
                    buffer = ""
                    break
                try:
                    event, end = decoder.raw_decode(stripped)
                except JSONDecodeError:
                    break
                yield start_line, event
                buffer = stripped[end:]
                start_line = line_number
    if buffer.strip():
        raise RuntimeError(f"Trailing incomplete JSON event in {path} starting at line {start_line}")


def collect_policy_step_paths(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.is_file():
            found.append(path)
            continue
        direct = path / "streams" / "policy_steps.jsonl"
        if direct.exists():
            found.append(direct)
            continue
        episodes_dir = path / "episodes"
        if episodes_dir.exists():
            found.extend(sorted(episodes_dir.glob("episode_*/streams/policy_steps.jsonl")))
            continue
        found.extend(sorted(path.glob("**/streams/policy_steps.jsonl")))
    return dedupe_paths(found)


def dedupe_paths(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        result.append(path)
    return result


def extract_mpd_tcp_state(action: dict[str, Any]) -> list[float] | None:
    metadata = action.get("metadata") or {}
    state = metadata.get("mpd_tcp_state")
    if not isinstance(state, list) or len(state) != 10:
        return None
    return [float(value) for value in state]


def gripper_command(action: dict[str, Any]) -> str:
    if action.get("gripper_closed") is True:
        return "close"
    if "gripper_width" in action:
        return "open"
    return "none"


def format_header() -> str:
    names = ("idx", "time", *STATE_NAMES, "cmd")
    widths = (4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7)
    return " ".join(name.rjust(width) for name, width in zip(names, widths, strict=True))


def format_row(row: dict[str, Any], precision: int) -> str:
    values: list[str] = [
        str(row["chunk_action_index"]),
        format_optional_float(row["action_time_sec"], precision),
        *(format_float(row[name], precision) for name in STATE_NAMES),
        str(row["command"]),
    ]
    widths = (4, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 7)
    return " ".join(value.rjust(width) for value, width in zip(values, widths, strict=True))


def format_stats(values: list[float], precision: int) -> str:
    if not values:
        return "n=0"
    return (
        f"n={len(values)} "
        f"min={format_float(min(values), precision)} "
        f"max={format_float(max(values), precision)} "
        f"mean={format_float(sum(values) / len(values), precision)}"
    )


def count_ge(values: list[float], threshold: float) -> int:
    return sum(1 for value in values if value >= threshold)


def count_le(values: list[float], threshold: float) -> int:
    return sum(1 for value in values if value <= threshold)


def format_float(value: float, precision: int) -> str:
    return f"{float(value):.{precision}f}"


def format_optional_float(value: float | None, precision: int) -> str:
    if value is None:
        return "nan"
    return format_float(value, precision)


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy_steps_path",
        "line_number",
        "chunk_index",
        "step_index",
        "chunk_action_index",
        "episode_elapsed_sec",
        "action_time_sec",
        "target_duration_sec",
        *STATE_NAMES,
        "command",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
