#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_TASKS = (
    "cleaning",
    "erasing",
    "pencil_insertion",
    "usb_insertion",
    "writing",
)


@dataclass(frozen=True)
class LayoutTile:
    x: int
    y: int
    width: int
    height: int
    image_x: int
    image_y: int
    image_width: int
    image_height: int
    overlay_width: int
    overlay_height: int


@dataclass
class TaskClip:
    task_name: str
    episode_dir: Path
    wrist_paths: np.ndarray
    gelsight_paths: np.ndarray
    gelsight_indices: np.ndarray
    start_index: int
    end_index: int
    tactile_cache: dict[tuple[str, int], np.ndarray]

    @property
    def frame_count(self) -> int:
        return self.end_index - self.start_index + 1

    @property
    def label(self) -> str:
        return self.task_name.replace("_", " ").title().replace("Usb", "USB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render a 5-task grid demo from collected aligned_episode.npz files. "
            "Each tile shows wrist RGB with a synchronized GelSight inset."
        )
    )
    parser.add_argument("--data-root", type=Path, default=Path("robot_workspace/data/collect"))
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--episode", default="episode_0000")
    parser.add_argument("--output", type=Path, default=Path("robot_workspace/data/collect/demo_5tasks_grid.mp4"))
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--timeline",
        choices=("stretch", "hold"),
        default="stretch",
        help=(
            "stretch maps each task's full episode onto the full output duration; "
            "hold preserves each aligned 10 Hz index and holds shorter tasks on their final frame."
        ),
    )
    parser.add_argument(
        "--include-pre-tactile",
        action="store_true",
        help="Start each task at aligned index 0. By default, tiny pre-GelSight gaps are trimmed.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional debug/export cap.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")
    if args.fps <= 0:
        raise ValueError("fps must be positive")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to replace it")

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to render the demo video") from exc

    clips = [
        load_clip(args.data_root / task / "episodes" / args.episode, task, trim_pre_tactile=not args.include_pre_tactile)
        for task in args.tasks
    ]
    layout = build_layout(args.width, args.height, len(clips))
    overlay_size = (layout[0].overlay_width, layout[0].overlay_height)
    for clip in clips:
        preload_tactile_cache(cv2, clip, overlay_size=overlay_size)

    total_frames = max(clip.frame_count for clip in clips)
    if args.max_frames is not None:
        total_frames = min(total_frames, int(args.max_frames))
    if total_frames <= 0:
        raise RuntimeError("No frames to render")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (args.width, args.height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {args.output}")

    print(f"[render] {args.output}  frames={total_frames}  fps={args.fps:g}  timeline={args.timeline}")
    try:
        for frame_index in range(total_frames):
            frame = render_frame(cv2, clips, layout, frame_index, total_frames, args.timeline, args.width, args.height)
            writer.write(frame)
            if frame_index == 0 or (frame_index + 1) % 50 == 0 or frame_index + 1 == total_frames:
                print(f"[render] {frame_index + 1}/{total_frames}")
    finally:
        writer.release()
    print(f"[ok] wrote {args.output}")


def load_clip(episode_dir: Path, task_name: str, *, trim_pre_tactile: bool) -> TaskClip:
    npz_path = episode_dir / "aligned_episode.npz"
    if not npz_path.exists():
        raise FileNotFoundError(npz_path)

    data = np.load(npz_path, allow_pickle=True)
    wrist_paths = np.asarray(data["rgb_wrist_frame_paths"], dtype=object)
    gelsight_paths = np.asarray(data["gelsight_frame_paths"], dtype=object)
    gelsight_indices = np.asarray(data["gelsight_frame_indices"], dtype=np.int64)

    if not (len(wrist_paths) == len(gelsight_paths) == len(gelsight_indices)):
        raise RuntimeError(f"Mismatched aligned stream lengths in {npz_path}")

    valid = [
        index
        for index, (wrist_path, gelsight_path, gelsight_index) in enumerate(
            zip(wrist_paths, gelsight_paths, gelsight_indices, strict=True)
        )
        if str(wrist_path) and str(gelsight_path) and int(gelsight_index) >= 0
    ]
    if not valid:
        raise RuntimeError(f"No aligned wrist/GelSight frames found in {npz_path}")

    start_index = 0 if trim_pre_tactile is False else valid[0]
    end_index = valid[-1]
    if start_index > end_index:
        raise RuntimeError(f"Invalid frame span for {npz_path}: {start_index}>{end_index}")

    return TaskClip(
        task_name=task_name,
        episode_dir=episode_dir,
        wrist_paths=wrist_paths,
        gelsight_paths=gelsight_paths,
        gelsight_indices=gelsight_indices,
        start_index=start_index,
        end_index=end_index,
        tactile_cache={},
    )


def build_layout(width: int, height: int, tile_count: int) -> list[LayoutTile]:
    if tile_count != 5:
        raise ValueError("This renderer is designed for exactly 5 task tiles")

    margin_x = max(24, width // 64)
    margin_y = max(24, height // 36)
    gap_x = max(18, width // 80)
    gap_y = max(18, height // 45)
    tile_w = (width - 2 * margin_x - 2 * gap_x) // 3
    tile_h = (height - 2 * margin_y - gap_y) // 2

    top_xs = [margin_x + col * (tile_w + gap_x) for col in range(3)]
    bottom_total_w = 2 * tile_w + gap_x
    bottom_start_x = (width - bottom_total_w) // 2
    bottom_xs = [bottom_start_x, bottom_start_x + tile_w + gap_x]
    positions = [(x, margin_y) for x in top_xs] + [(x, margin_y + tile_h + gap_y) for x in bottom_xs]

    tiles = []
    for x, y in positions:
        inner_pad = max(10, tile_w // 60)
        title_h = max(38, tile_h // 11)
        available_w = tile_w - 2 * inner_pad
        available_h = tile_h - title_h - inner_pad - max(8, tile_h // 60)
        image_w = available_w
        image_h = int(round(image_w * 3.0 / 4.0))
        if image_h > available_h:
            image_h = available_h
            image_w = int(round(image_h * 4.0 / 3.0))
        image_x = x + (tile_w - image_w) // 2
        image_y = y + title_h
        overlay_w = max(140, int(round(image_w * 0.34)))
        overlay_h = int(round(overlay_w * 3.0 / 4.0))
        tiles.append(
            LayoutTile(
                x=x,
                y=y,
                width=tile_w,
                height=tile_h,
                image_x=image_x,
                image_y=image_y,
                image_width=image_w,
                image_height=image_h,
                overlay_width=overlay_w,
                overlay_height=overlay_h,
            )
        )
    return tiles


def preload_tactile_cache(cv2, clip: TaskClip, *, overlay_size: tuple[int, int]) -> None:
    refs_by_chunk: dict[str, set[int]] = {}
    last_ref: tuple[str, int] | None = None
    for index in range(clip.start_index, clip.end_index + 1):
        ref = gelsight_ref(clip, index)
        if ref is None:
            ref = last_ref
        else:
            last_ref = ref
        if ref is None:
            continue
        refs_by_chunk.setdefault(ref[0], set()).add(ref[1])

    for rel_path, indices in sorted(refs_by_chunk.items()):
        chunk_path = clip.episode_dir / rel_path
        with np.load(chunk_path) as chunk:
            frames = np.asarray(chunk["frames"])
            for frame_index in sorted(indices):
                if frame_index < 0 or frame_index >= len(frames):
                    continue
                tactile = cv2.rotate(frames[frame_index], cv2.ROTATE_90_CLOCKWISE)
                tactile = resize_exact(cv2, tactile, overlay_size[0], overlay_size[1])
                clip.tactile_cache[(rel_path, frame_index)] = tactile
        gc.collect()
    print(f"[cache] {clip.task_name}: {len(clip.tactile_cache)} GelSight frames")


def render_frame(
    cv2,
    clips: list[TaskClip],
    layout: list[LayoutTile],
    frame_index: int,
    total_frames: int,
    timeline: str,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.full((height, width, 3), (18, 21, 26), dtype=np.uint8)
    for clip, tile in zip(clips, layout, strict=True):
        aligned_index = map_frame_index(clip, frame_index, total_frames, timeline)
        draw_tile(cv2, canvas, clip, tile, aligned_index, frame_index, total_frames)
    return canvas


def map_frame_index(clip: TaskClip, frame_index: int, total_frames: int, timeline: str) -> int:
    if timeline == "hold":
        local = min(frame_index, clip.frame_count - 1)
    else:
        if total_frames <= 1 or clip.frame_count <= 1:
            local = 0
        else:
            local = int(round(frame_index * (clip.frame_count - 1) / (total_frames - 1)))
    return clip.start_index + local


def draw_tile(cv2, canvas: np.ndarray, clip: TaskClip, tile: LayoutTile, aligned_index: int, frame_index: int, total_frames: int) -> None:
    panel_bg = (31, 36, 44)
    panel_border = (79, 88, 102)
    image_bg = (10, 12, 15)
    text_color = (238, 241, 245)
    subtext_color = (155, 164, 177)
    accent = (81, 175, 232)

    cv2.rectangle(canvas, (tile.x, tile.y), (tile.x + tile.width, tile.y + tile.height), panel_bg, thickness=-1)
    cv2.rectangle(canvas, (tile.x, tile.y), (tile.x + tile.width, tile.y + tile.height), panel_border, thickness=2)

    label_y = tile.y + max(29, tile.height // 14)
    cv2.putText(canvas, clip.label, (tile.x + 16, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.78, text_color, 2, cv2.LINE_AA)
    elapsed = aligned_index - clip.start_index
    total = max(1, clip.frame_count - 1)
    cv2.putText(
        canvas,
        f"{elapsed:03d}/{total:03d}",
        (tile.x + tile.width - 102, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        subtext_color,
        1,
        cv2.LINE_AA,
    )

    cv2.rectangle(
        canvas,
        (tile.image_x - 1, tile.image_y - 1),
        (tile.image_x + tile.image_width + 1, tile.image_y + tile.image_height + 1),
        (4, 5, 7),
        thickness=-1,
    )
    wrist = read_wrist_frame(cv2, clip, aligned_index)
    if wrist is None:
        cv2.rectangle(
            canvas,
            (tile.image_x, tile.image_y),
            (tile.image_x + tile.image_width, tile.image_y + tile.image_height),
            image_bg,
            thickness=-1,
        )
    else:
        wrist = cv2.rotate(wrist, cv2.ROTATE_180)
        wrist = resize_exact(cv2, wrist, tile.image_width, tile.image_height)
        canvas[tile.image_y : tile.image_y + tile.image_height, tile.image_x : tile.image_x + tile.image_width] = wrist

    tactile = read_tactile_frame(clip, aligned_index)
    overlay_x = tile.image_x + tile.image_width - tile.overlay_width - 12
    overlay_y = tile.image_y + 12
    draw_tactile_overlay(cv2, canvas, tactile, overlay_x, overlay_y, tile.overlay_width, tile.overlay_height)

    progress = 0.0 if total_frames <= 1 else frame_index / float(total_frames - 1)
    progress_y = tile.y + tile.height - 11
    cv2.line(canvas, (tile.x + 14, progress_y), (tile.x + tile.width - 14, progress_y), (62, 68, 79), 3, cv2.LINE_AA)
    progress_x = tile.x + 14 + int(round(progress * (tile.width - 28)))
    cv2.line(canvas, (tile.x + 14, progress_y), (progress_x, progress_y), accent, 3, cv2.LINE_AA)


def read_wrist_frame(cv2, clip: TaskClip, aligned_index: int) -> np.ndarray | None:
    rel_path = str(clip.wrist_paths[aligned_index])
    if not rel_path:
        return None
    image = cv2.imread(str(clip.episode_dir / rel_path), cv2.IMREAD_COLOR)
    return image


def read_tactile_frame(clip: TaskClip, aligned_index: int) -> np.ndarray | None:
    ref = gelsight_ref(clip, aligned_index)
    if ref is not None and ref in clip.tactile_cache:
        return clip.tactile_cache[ref]

    for fallback_index in range(aligned_index - 1, clip.start_index - 1, -1):
        ref = gelsight_ref(clip, fallback_index)
        if ref is not None and ref in clip.tactile_cache:
            return clip.tactile_cache[ref]
    return None


def gelsight_ref(clip: TaskClip, aligned_index: int) -> tuple[str, int] | None:
    rel_path = str(clip.gelsight_paths[aligned_index])
    frame_index = int(clip.gelsight_indices[aligned_index])
    if not rel_path or frame_index < 0:
        return None
    return rel_path, frame_index


def draw_tactile_overlay(
    cv2,
    canvas: np.ndarray,
    tactile: np.ndarray | None,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    shadow = 5
    cv2.rectangle(canvas, (x - shadow, y - shadow), (x + width + shadow, y + height + shadow), (0, 0, 0), thickness=-1)
    cv2.rectangle(canvas, (x - 2, y - 2), (x + width + 2, y + height + 2), (238, 241, 245), thickness=2)
    if tactile is None:
        cv2.rectangle(canvas, (x, y), (x + width, y + height), (18, 21, 26), thickness=-1)
        return
    canvas[y : y + height, x : x + width] = tactile


def resize_exact(cv2, image: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_AREA)


if __name__ == "__main__":
    main()
