#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Render videos from aligned_episode.npz files.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Episode directories, aligned_episode.npz files, a run directory containing episodes/, or an episodes/ directory.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None, help="Override output FPS. Defaults to aligned manifest/timestamps.")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for episode_dir, npz_path in _collect_aligned_paths([Path(path) for path in args.paths]):
        output_path = _resolve_output_path(episode_dir, args.output_dir)
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists")
            continue
        render_video(
            episode_dir,
            npz_path,
            output_path,
            fps=args.fps,
            width=args.width,
            height=args.height,
        )
        print(f"[ok] {output_path}")


def render_video(
    episode_dir: Path,
    npz_path: Path,
    output_path: Path,
    *,
    fps: float | None = None,
    width: int = 1920,
    height: int = 1080,
) -> Path:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to render aligned episode videos") from exc

    data = np.load(npz_path, allow_pickle=True)
    timestamps = np.asarray(data["timestamps"], dtype=np.float64)
    if timestamps.size == 0:
        raise RuntimeError(f"No aligned timestamps found in {npz_path}")

    manifest = _load_json(episode_dir / "aligned_episode_manifest.json")
    episode_manifest = _load_json(episode_dir / "episode_manifest.json")
    export_fps = float(fps if fps is not None else _infer_aligned_fps(timestamps, manifest))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), export_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        robot_pose = _pose7_to_xyz_rpy(np.asarray(data["robot_tcp_pose"], dtype=np.float64))
        action_pose = _pose7_to_xyz_rpy(np.asarray(data["teleop_target_tcp"], dtype=np.float64))
        image_streams = _load_rgb_streams(data)
        image_streams.extend(_load_raw_image_streams(episode_dir, timestamps, existing_names={name for name, _ in image_streams}))
        gripper_width = np.asarray(data["gripper_width"], dtype=np.float64) if "gripper_width" in data else None
        teleop_gripper_closed = (
            np.asarray(data["teleop_gripper_closed"], dtype=bool) if "teleop_gripper_closed" in data else None
        )
        for index in range(len(timestamps)):
            writer.write(
                _render_frame(
                    cv2=cv2,
                    episode_dir=episode_dir,
                    episode_name=episode_dir.name,
                    episode_manifest=episode_manifest,
                    manifest=manifest,
                    index=index,
                    timestamps=timestamps,
                    rgb_streams=image_streams,
                    robot_pose=robot_pose,
                    action_pose=action_pose,
                    gripper_width=gripper_width,
                    teleop_gripper_closed=teleop_gripper_closed,
                    width=width,
                    height=height,
                    fps=export_fps,
                )
            )
    finally:
        writer.release()
    return output_path


def _collect_aligned_paths(paths: list[Path]) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for path in paths:
        if path.name == "aligned_episode.npz":
            if not path.exists():
                raise FileNotFoundError(path)
            pairs.append((path.parent, path))
            continue
        if path.name.startswith("episode_") and path.is_dir():
            pairs.append((path, path / "aligned_episode.npz"))
            continue
        episodes_dir = path / "episodes"
        if episodes_dir.exists():
            pairs.extend((episode_dir, episode_dir / "aligned_episode.npz") for episode_dir in sorted(episodes_dir.glob("episode_*")))
            continue
        if path.is_dir():
            pairs.extend((episode_dir, episode_dir / "aligned_episode.npz") for episode_dir in sorted(path.glob("episode_*")))
            continue
        raise FileNotFoundError(f"Cannot resolve path: {path}")

    resolved = []
    for episode_dir, npz_path in pairs:
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing aligned_episode.npz: {npz_path}")
        resolved.append((episode_dir, npz_path))
    return sorted(dict.fromkeys(resolved))


def _resolve_output_path(episode_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return episode_dir / "aligned_episode_video.mp4"
    if output_dir.suffix.lower() == ".mp4":
        return output_dir
    return output_dir / f"{episode_dir.name}_aligned.mp4"


def _load_rgb_streams(data: np.lib.npyio.NpzFile) -> list[tuple[str, np.ndarray]]:
    streams = []
    for key in sorted(data.files):
        if key.endswith("_frame_paths"):
            name = key[: -len("_frame_paths")]
            frame_paths = np.asarray(data[key], dtype=object)
            index_key = f"{name}_frame_indices"
            if index_key in data:
                frame_indices = np.asarray(data[index_key], dtype=np.int64)
                frame_refs = np.asarray(
                    [
                        "" if not str(path) else {"frame_path": str(path), "index_in_chunk": int(frame_indices[index])}
                        for index, path in enumerate(frame_paths)
                    ],
                    dtype=object,
                )
                streams.append((name, frame_refs))
            else:
                streams.append((name, frame_paths))
    preferred = ["rgb_third_person", "rgb_wrist"]
    ordered = [item for name in preferred for item in streams if item[0] == name]
    ordered.extend(item for item in streams if item[0] not in preferred)
    return ordered


def _load_raw_image_streams(
    episode_dir: Path,
    timestamps: np.ndarray,
    *,
    existing_names: set[str],
) -> list[tuple[str, np.ndarray]]:
    streams: list[tuple[str, np.ndarray]] = []
    candidates = [
        ("gelsight", "gelsight_frames"),
    ]
    for display_name, stream_name in candidates:
        if display_name in existing_names or stream_name in existing_names:
            continue
        stream_path = episode_dir / "streams" / f"{stream_name}.jsonl"
        if not stream_path.exists():
            continue
        stream = _align_raw_frame_stream(stream_path, timestamps)
        if stream is not None:
            streams.append((display_name, stream))
    return streams


def _align_raw_frame_stream(stream_path: Path, timestamps: np.ndarray) -> np.ndarray | None:
    frame_times: list[float] = []
    frame_refs: list[dict[str, object]] = []
    with stream_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            frame_path = record.get("frame_path")
            if not frame_path:
                continue
            metadata = record.get("metadata") or {}
            timestamp = metadata.get("captured_wall_time", record.get("captured_wall_time", record.get("recorded_at_wall_time")))
            if timestamp is None:
                continue
            frame_times.append(float(timestamp))
            frame_refs.append({"frame_path": str(frame_path), "index_in_chunk": int(record.get("index_in_chunk", -1))})
    if not frame_times:
        return None

    order = np.argsort(np.asarray(frame_times, dtype=np.float64))
    sorted_times = np.asarray(frame_times, dtype=np.float64)[order]
    sorted_refs = np.asarray(frame_refs, dtype=object)[order]
    aligned_refs: list[dict[str, object] | str] = []
    for timestamp in timestamps:
        right = int(np.searchsorted(sorted_times, float(timestamp), side="left"))
        candidates = []
        if right > 0:
            candidates.append(right - 1)
        if right < len(sorted_times):
            candidates.append(right)
        if not candidates:
            aligned_refs.append("")
            continue
        best = min(candidates, key=lambda idx: abs(float(sorted_times[idx]) - float(timestamp)))
        aligned_refs.append(sorted_refs[best])
    return np.asarray(aligned_refs, dtype=object)


def _infer_aligned_fps(timestamps: np.ndarray, manifest: dict) -> float:
    target_hz = manifest.get("target_hz")
    if target_hz is not None:
        return float(target_hz)
    if len(timestamps) < 2:
        return 10.0
    dt = np.diff(timestamps)
    dt = dt[dt > 0.0]
    return 10.0 if dt.size == 0 else float(1.0 / np.median(dt))


def _pose7_to_xyz_rpy(pose7: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    output = np.zeros((pose7.shape[0], 6), dtype=np.float64)
    output[:, :3] = pose7[:, :3]
    quat_xyzw = np.column_stack([pose7[:, 4], pose7[:, 5], pose7[:, 6], pose7[:, 3]])
    output[:, 3:] = Rotation.from_quat(quat_xyzw).as_euler("xyz", degrees=True)
    return output


def _render_frame(
    *,
    cv2,
    episode_dir: Path,
    episode_name: str,
    episode_manifest: dict,
    manifest: dict,
    index: int,
    timestamps: np.ndarray,
    rgb_streams: list[tuple[str, np.ndarray]],
    robot_pose: np.ndarray,
    action_pose: np.ndarray,
    gripper_width: np.ndarray | None,
    teleop_gripper_closed: np.ndarray | None,
    width: int,
    height: int,
    fps: float,
) -> np.ndarray:
    canvas = np.full((height, width, 3), 246, dtype=np.uint8)
    _draw_header(
        cv2,
        canvas,
        episode_name=episode_name,
        index=index,
        timestamps=timestamps,
        episode_manifest=episode_manifest,
        manifest=manifest,
        fps=fps,
    )

    margin = 32
    gap = 28
    top = 140
    left_w = int(width * 0.42)
    right_x = margin + left_w + gap
    right_w = width - right_x - margin
    content_h = height - top - margin

    _draw_rgb_stack(
        cv2,
        canvas,
        episode_dir=episode_dir,
        rgb_streams=rgb_streams,
        index=index,
        x=margin,
        y=top,
        width=left_w,
        height=content_h,
    )
    plot_h = (content_h - 18) // 2
    times = timestamps - timestamps[0]
    labels = ["x", "y", "z", "roll", "pitch", "yaw"]
    _draw_pose_grid(
        cv2,
        canvas,
        title="Teleop Target",
        labels=labels,
        values=action_pose,
        times=times,
        current_index=index,
        x=right_x,
        y=top,
        width=right_w,
        height=plot_h,
    )
    _draw_pose_grid(
        cv2,
        canvas,
        title="Robot State",
        labels=labels,
        values=robot_pose,
        times=times,
        current_index=index,
        x=right_x,
        y=top + plot_h + 18,
        width=right_w,
        height=plot_h,
    )
    _draw_gripper_status(
        cv2,
        canvas,
        x=margin,
        y=height - margin - 48,
        gripper_width=gripper_width,
        teleop_gripper_closed=teleop_gripper_closed,
        index=index,
    )
    return canvas


def _draw_header(cv2, canvas: np.ndarray, *, episode_name: str, index: int, timestamps: np.ndarray, episode_manifest: dict, manifest: dict, fps: float) -> None:
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 118), (231, 236, 243), thickness=-1)
    elapsed = float(timestamps[index] - timestamps[0])
    title = f"{episode_name}  frame={index:04d}/{len(timestamps)-1:04d}  t={elapsed:.2f}s  fps={fps:.2f}"
    cv2.putText(canvas, title, (32, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.86, (34, 40, 49), 2, cv2.LINE_AA)
    task_name = episode_manifest.get("metadata", {}).get("task_name") or episode_manifest.get("run_id", "")
    subtitle = f"task={task_name}  aligned_hz={manifest.get('target_hz', 'unknown')}  mode={manifest.get('alignment_mode', '')}"
    cv2.putText(canvas, subtitle[:170], (32, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (70, 78, 89), 1, cv2.LINE_AA)


def _draw_rgb_stack(cv2, canvas: np.ndarray, *, episode_dir: Path, rgb_streams: list[tuple[str, np.ndarray]], index: int, x: int, y: int, width: int, height: int) -> None:
    if not rgb_streams:
        _draw_panel(cv2, canvas, x, y, width, height, "Images")
        cv2.putText(canvas, "No image streams found", (x + 20, y + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (80, 80, 80), 2, cv2.LINE_AA)
        return
    panel_gap = 14
    panel_h = (height - panel_gap * (len(rgb_streams) - 1)) // len(rgb_streams)
    for slot, (stream_name, frame_paths) in enumerate(rgb_streams):
        py = y + slot * (panel_h + panel_gap)
        _draw_panel(cv2, canvas, x, py, width, panel_h, _stream_label(stream_name))
        image = _read_frame(cv2, episode_dir, frame_paths, index)
        inner_x, inner_y = x + 16, py + 48
        inner_w, inner_h = width - 32, panel_h - 66
        if image is None:
            cv2.putText(canvas, "No frame", (inner_x + 8, inner_y + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2, cv2.LINE_AA)
            continue
        fitted = _fit_image(cv2, image, inner_w, inner_h)
        ih, iw = fitted.shape[:2]
        ox = inner_x + (inner_w - iw) // 2
        oy = inner_y + (inner_h - ih) // 2
        canvas[oy : oy + ih, ox : ox + iw] = fitted


def _draw_pose_grid(cv2, canvas: np.ndarray, *, title: str, labels: list[str], values: np.ndarray, times: np.ndarray, current_index: int, x: int, y: int, width: int, height: int) -> None:
    _draw_panel(cv2, canvas, x, y, width, height, title)
    cols, rows = 3, 2
    inner_x, inner_y = x + 16, y + 48
    inner_w, inner_h = width - 32, height - 64
    gap_x, gap_y = 14, 14
    plot_w = (inner_w - gap_x * (cols - 1)) // cols
    plot_h = (inner_h - gap_y * (rows - 1)) // rows
    colors = [(74, 73, 209), (73, 174, 237), (140, 121, 0), (142, 99, 48), (64, 15, 95), (143, 157, 42)]
    for idx, label in enumerate(labels):
        px = inner_x + (idx % cols) * (plot_w + gap_x)
        py = inner_y + (idx // cols) * (plot_h + gap_y)
        _draw_series(cv2, canvas, label=label, series=values[:, idx], times=times, current_index=current_index, x=px, y=py, width=plot_w, height=plot_h, color=colors[idx])


def _draw_series(cv2, canvas: np.ndarray, *, label: str, series: np.ndarray, times: np.ndarray, current_index: int, x: int, y: int, width: int, height: int, color: tuple[int, int, int]) -> None:
    del times
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (251, 251, 248), thickness=-1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (210, 210, 210), thickness=1)
    cv2.putText(canvas, label, (x + 8, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (36, 36, 36), 2, cv2.LINE_AA)
    plot_x, plot_y = x + 8, y + 30
    plot_w, plot_h = width - 16, height - 44
    ymin, ymax = float(np.min(series)), float(np.max(series))
    if np.isclose(ymin, ymax):
        pad = 1.0 if np.isclose(ymin, 0.0) else abs(ymin) * 0.1
        ymin -= pad
        ymax += pad
    pad = 0.12 * (ymax - ymin)
    ymin -= pad
    ymax += pad
    xs = np.linspace(plot_x + 4, plot_x + plot_w - 4, len(series))
    pts = []
    for idx, value in enumerate(series):
        ratio = 0.5 if np.isclose(ymax, ymin) else (float(value) - ymin) / (ymax - ymin)
        pts.append([int(round(xs[idx])), int(round(plot_y + plot_h - 4 - ratio * (plot_h - 8)))])
    cv2.polylines(canvas, [np.asarray(pts, dtype=np.int32).reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
    current = tuple(pts[current_index])
    cv2.circle(canvas, current, 4, color, -1, cv2.LINE_AA)
    cv2.line(canvas, (current[0], plot_y + 2), (current[0], plot_y + plot_h - 2), (185, 185, 185), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{series[current_index]: .3f}", (x + 8, y + height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (80, 80, 80), 1, cv2.LINE_AA)


def _draw_gripper_status(cv2, canvas: np.ndarray, *, x: int, y: int, gripper_width: np.ndarray | None, teleop_gripper_closed: np.ndarray | None, index: int) -> None:
    parts = []
    if gripper_width is not None:
        parts.append(f"robot gripper width={float(gripper_width[index]):.4f}m")
    if teleop_gripper_closed is not None:
        parts.append(f"teleop gripper={'closed' if bool(teleop_gripper_closed[index]) else 'open'}")
    if parts:
        cv2.putText(canvas, "  ".join(parts), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (45, 52, 61), 2, cv2.LINE_AA)


def _draw_panel(cv2, canvas: np.ndarray, x: int, y: int, width: int, height: int, title: str) -> None:
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (236, 240, 245), thickness=-1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), (205, 212, 220), thickness=2)
    cv2.putText(canvas, title, (x + 14, y + 31), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (36, 42, 49), 2, cv2.LINE_AA)


def _read_frame(cv2, episode_dir: Path, frame_paths: np.ndarray, index: int) -> np.ndarray | None:
    frame_ref = frame_paths[index]
    if isinstance(frame_ref, dict):
        rel_path = str(frame_ref.get("frame_path", ""))
        index_in_chunk = int(frame_ref.get("index_in_chunk", -1))
    else:
        rel_path = str(frame_ref)
        index_in_chunk = -1
    if not rel_path:
        return None
    path = episode_dir / rel_path
    if path.suffix.lower() == ".npz":
        if index_in_chunk < 0:
            return None
        with np.load(path) as chunk:
            frames = chunk["frames"]
            if index_in_chunk >= len(frames):
                return None
            return np.asarray(frames[index_in_chunk])
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _fit_image(cv2, image: np.ndarray, width: int, height: int) -> np.ndarray:
    ih, iw = image.shape[:2]
    scale = min(width / max(iw, 1), height / max(ih, 1))
    target_size = (max(1, int(round(iw * scale))), max(1, int(round(ih * scale))))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def _stream_label(stream_name: str) -> str:
    if stream_name == "gelsight":
        return "GelSight"
    label = stream_name.removeprefix("rgb_").replace("_", " ").strip()
    return label.title() if label else stream_name


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
