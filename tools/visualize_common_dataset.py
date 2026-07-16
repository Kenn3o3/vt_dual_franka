#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Render videos from VT Dual Franka common datasets.")
    parser.add_argument("dataset_or_episode", type=Path, help="Common dataset root or one common dataset episode directory.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--keep-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for dataset_root, episode_dir, steps_path in _resolve_episodes(args.dataset_or_episode):
        output_path = _resolve_output_path(dataset_root, episode_dir, args.output_dir)
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {output_path} exists")
            continue
        render_episode(
            dataset_root=dataset_root,
            episode_dir=episode_dir,
            steps_path=steps_path,
            output_path=output_path,
            fps=args.fps,
            width=args.width,
            height=args.height,
            max_frames=args.max_frames,
            keep_frames=args.keep_frames,
        )
        print(f"[ok] {output_path}")


def render_episode(
    *,
    dataset_root: Path,
    episode_dir: Path,
    steps_path: Path,
    output_path: Path,
    fps: float | None,
    width: int,
    height: int,
    max_frames: int | None,
    keep_frames: bool,
) -> Path:
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("OpenCV and NumPy are required to render common dataset videos") from exc

    steps = _read_jsonl(steps_path)
    if not steps:
        raise RuntimeError(f"No steps found in {steps_path}")
    if max_frames is not None:
        steps = steps[: max(0, int(max_frames))]
    if not steps:
        raise RuntimeError("max_frames removed all frames")

    target_fps = float(fps if fps is not None else _infer_fps(steps))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output_path.with_suffix("").parent / f"{output_path.with_suffix('').name}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)
    for index, step in enumerate(steps):
        frame = _render_frame(cv2, np, dataset_root, step, index, len(steps), width, height)
        frame_path = frames_dir / f"frame_{index:06d}.png"
        if not cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Failed to write preview frame: {frame_path}")
        if index == 0:
            preview_path = output_path.with_suffix(".preview.png")
            if not cv2.imwrite(str(preview_path), frame):
                raise RuntimeError(f"Failed to write preview image: {preview_path}")
        if index == 0 or (index + 1) % 50 == 0 or index + 1 == len(steps):
            print(f"[render] {episode_dir.name}: {index + 1}/{len(steps)}")
    _encode_with_ffmpeg(frames_dir, output_path, target_fps)
    if not keep_frames:
        shutil.rmtree(frames_dir)
    return output_path


def _encode_with_ffmpeg(frames_dir: Path, output_path: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print(f"[warn] ffmpeg not found; frames left in {frames_dir}")
        return
    command = [
        ffmpeg,
        "-y",
        "-framerate",
        f"{fps:.6f}",
        "-i",
        str(frames_dir / "frame_%06d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def _render_frame(cv2, np, dataset_root: Path, step: dict[str, Any], index: int, total: int, width: int, height: int):
    canvas = np.full((height, width, 3), (24, 26, 30), dtype=np.uint8)
    gap = 12
    header_h = 42
    panel_w = (width - gap * 3) // 2
    panel_h = height - header_h - gap * 2
    y0 = header_h + gap
    rgb = _read_image(cv2, dataset_root / step["images"]["rgb_wrist"])
    tactile = _read_image(cv2, dataset_root / step["images"]["tactile_left"])
    _draw_panel(cv2, canvas, rgb, x=gap, y=y0, w=panel_w, h=panel_h, label="rgb_wrist")
    _draw_panel(cv2, canvas, tactile, x=gap * 2 + panel_w, y=y0, w=panel_w, h=panel_h, label="tactile_left")

    episode_id = step.get("episode_id", "")
    step_index = int(step.get("step_index", index))
    ts = float(step.get("timestamp", 0.0))
    rgb_age = _source_age(step, "rgb_wrist")
    tactile_age = _source_age(step, "tactile_left")
    action_lead = step.get("source", {}).get("teleop_command", {}).get("lead_sec")
    title = f"{episode_id}  step={step_index:04d}/{total - 1:04d}  t={ts:.3f}"
    meta = f"rgb_age={rgb_age:+.3f}s  tactile_age={tactile_age:+.3f}s"
    if action_lead is not None:
        meta += f"  action_lead={float(action_lead):+.3f}s"
    cv2.putText(canvas, title, (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (238, 241, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, meta, (14, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (168, 176, 190), 1, cv2.LINE_AA)
    return canvas


def _draw_panel(cv2, canvas, image, *, x: int, y: int, w: int, h: int, label: str) -> None:
    cv2.rectangle(canvas, (x - 1, y - 1), (x + w + 1, y + h + 1), (76, 84, 98), 1)
    resized = _letterbox(cv2, image, w, h)
    canvas[y : y + h, x : x + w] = resized
    cv2.rectangle(canvas, (x, y), (x + 128, y + 25), (0, 0, 0), -1)
    cv2.putText(canvas, label, (x + 8, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (245, 247, 250), 1, cv2.LINE_AA)


def _letterbox(cv2, image, width: int, height: int):
    import numpy as np

    canvas = np.full((height, width, 3), (8, 10, 14), dtype=np.uint8)
    src_h, src_w = image.shape[:2]
    scale = min(width / max(src_w, 1), height / max(src_h, 1))
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_AREA)
    x0 = (width - out_w) // 2
    y0 = (height - out_h) // 2
    canvas[y0 : y0 + out_h, x0 : x0 + out_w] = resized
    return canvas


def _read_image(cv2, path: Path):
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return image


def _source_age(step: dict[str, Any], stream: str) -> float:
    source = step.get("source", {}).get(stream, {})
    return float(step.get("timestamp", 0.0)) - float(source.get("timestamp", step.get("timestamp", 0.0)))


def _infer_fps(steps: list[dict[str, Any]]) -> float:
    if len(steps) < 2:
        return 10.0
    first = float(steps[0]["timestamp"])
    last = float(steps[-1]["timestamp"])
    duration = last - first
    if duration <= 0.0:
        return 10.0
    return max(1.0, min(60.0, (len(steps) - 1) / duration))


def _resolve_episodes(path: Path) -> list[tuple[Path, Path, Path]]:
    path = Path(path)
    if (path / "dataset_manifest.json").exists():
        dataset_root = path
        manifest = json.loads((dataset_root / "dataset_manifest.json").read_text(encoding="utf-8"))
        episodes = []
        for entry in manifest.get("episodes", []):
            steps_rel = entry.get("steps_path")
            if not steps_rel:
                continue
            steps_path = dataset_root / steps_rel
            episodes.append((dataset_root, steps_path.parent, steps_path))
        if episodes:
            return episodes
    if (path / "steps.jsonl").exists():
        episode_dir = path
        dataset_root = _infer_dataset_root_from_episode_dir(episode_dir)
        return [(dataset_root, episode_dir, episode_dir / "steps.jsonl")]
    raise FileNotFoundError(f"Expected a common dataset root or episode directory: {path}")


def _infer_dataset_root_from_episode_dir(episode_dir: Path) -> Path:
    if episode_dir.parent.name == "episodes":
        return episode_dir.parent.parent
    return episode_dir.parent


def _resolve_output_path(dataset_root: Path, episode_dir: Path, output_dir: Path | None) -> Path:
    if output_dir is None:
        return dataset_root / f"{episode_dir.name}_common_dataset.mp4"
    if output_dir.suffix.lower() == ".mp4":
        return output_dir
    return output_dir / f"{episode_dir.name}_common_dataset.mp4"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


if __name__ == "__main__":
    main()
