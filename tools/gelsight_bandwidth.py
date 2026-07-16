#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_SRC = REPO_ROOT / "robot_workspace" / "src"
if str(ROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROBOT_SRC))
SHARED_SRC = REPO_ROOT / "shared" / "src"
if str(SHARED_SRC) not in sys.path:
    sys.path.insert(0, str(SHARED_SRC))

from vt_dual_franka_workspace.config import GelsightSettings, load_task_config
from vt_dual_franka_workspace.sensors.gelsight.publisher import GelsightPublisher

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously read GelSight frames and print rolling bandwidth/FPS diagnostics. "
            "This opens the camera exclusively; do not run it during data collection."
        )
    )
    parser.add_argument("--task-config", default="robot_workspace/config/tasks/pencil_insertion.yaml")
    parser.add_argument("--camera-path", default=None, help="Override the GelSight device path.")
    parser.add_argument("--camera-index", type=int, default=None, help="Override the GelSight camera index.")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--exposure", type=int, default=None)
    parser.add_argument("--contrast", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["collection", "decoded-mjpg", "mjpeg-payload"],
        default="collection",
        help=(
            "collection matches the current data-collection OpenCV path and reads decoded frames. "
            "decoded-mjpg forces MJPG then reads decoded frames. "
            "mjpeg-payload disables OpenCV RGB conversion and measures compressed camera payload throughput."
        ),
    )
    parser.add_argument(
        "--unpaced",
        action="store_true",
        help="Do not sleep to the configured FPS. Useful for measuring maximum host-side read throughput.",
    )
    parser.add_argument("--window-sec", type=float, default=5.0, help="Rolling stats window.")
    parser.add_argument("--print-every-sec", type=float, default=1.0)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 means run until Ctrl+C.")
    parser.add_argument("--warmup-sec", type=float, default=1.0)
    parser.add_argument("--jsonl", default=None, help="Optional path to append one JSON stats record per print.")
    parser.add_argument(
        "--copy-work",
        action="store_true",
        help="Also copy each decoded frame twice, approximating current live-buffer + recorder copy cost.",
    )
    parser.add_argument(
        "--apply-controls",
        action="store_true",
        help="Set exposure/contrast controls before streaming. Default matches the safer collection path and leaves them untouched.",
    )
    parser.add_argument(
        "--skip-controls",
        action="store_true",
        help="Deprecated alias for the default behavior. Controls are skipped unless --apply-controls is set.",
    )
    parser.add_argument(
        "--backend",
        choices=["collector", "v4l2", "any", "v4l2-ctl"],
        default="collector",
        help=(
            "Capture backend. collector uses the same VideoCapture(source) constructor as data collection. "
            "v4l2-ctl bypasses OpenCV and streams with v4l2-ctl --stream-mmap."
        ),
    )
    args = parser.parse_args()

    settings = _load_settings(args)
    source = _resolve_source(settings)
    _print_device_info(source)
    if args.backend == "v4l2-ctl":
        run_v4l2_ctl_probe(
            source,
            settings=settings,
            window_sec=args.window_sec,
            print_every_sec=args.print_every_sec,
            duration_sec=args.duration_sec,
            jsonl_path=Path(args.jsonl) if args.jsonl else None,
        )
        return
    run_bandwidth_probe(
        source,
        settings=settings,
        mode=args.mode,
        window_sec=args.window_sec,
        print_every_sec=args.print_every_sec,
        duration_sec=args.duration_sec,
        warmup_sec=args.warmup_sec,
        jsonl_path=Path(args.jsonl) if args.jsonl else None,
        copy_work=args.copy_work,
        paced=not args.unpaced and args.mode in {"collection", "decoded-mjpg"},
        apply_controls=args.apply_controls and not args.skip_controls,
        backend=args.backend,
    )


def run_bandwidth_probe(
    source: str | int,
    *,
    settings: GelsightSettings,
    mode: str,
    window_sec: float,
    print_every_sec: float,
    duration_sec: float,
    warmup_sec: float,
    jsonl_path: Path | None,
    copy_work: bool,
    paced: bool,
    apply_controls: bool,
    backend: str,
) -> None:
    import cv2

    if backend == "collector":
        cap = cv2.VideoCapture(source)
    elif backend == "v4l2":
        cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(source, cv2.CAP_ANY)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open GelSight camera source {source!r}")
    _configure_capture(cap, settings, mode=mode, apply_controls=apply_controls)
    actual = _capture_properties(cap)
    if warmup_sec > 0.0:
        time.sleep(warmup_sec)

    print(
        "mode={mode} source={source!r} requested={width}x{height}@{fps} paced={paced}".format(
            mode=mode,
            source=source,
            width=settings.width,
            height=settings.height,
            fps=settings.fps,
            paced=paced,
        ),
        flush=True,
    )
    print(f"actual={actual}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)

    samples: collections.deque[dict[str, Any]] = collections.deque()
    total_frames = 0
    total_bytes = 0
    start = time.time()
    last_print = start
    last_frame_time: float | None = None
    jsonl_handle = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("a", encoding="utf-8")

    try:
        while True:
            now = time.time()
            if duration_sec > 0.0 and now - start >= duration_sec:
                break
            loop_start = time.time()
            ok, frame = cap.read()
            read_time = time.time()
            if not ok:
                time.sleep(0.001)
                continue

            if copy_work:
                _ = frame.copy()
                _ = frame.copy()
            frame_bytes = int(getattr(frame, "nbytes", 0))
            sample = {
                "wall_time": read_time,
                "frame_bytes": frame_bytes,
                "shape": list(frame.shape),
                "dtype": str(frame.dtype),
                "gap_sec": None if last_frame_time is None else read_time - last_frame_time,
            }
            last_frame_time = read_time
            samples.append(sample)
            total_frames += 1
            total_bytes += frame_bytes
            while samples and read_time - float(samples[0]["wall_time"]) > window_sec:
                samples.popleft()

            if read_time - last_print >= print_every_sec:
                stats = _rolling_stats(samples, total_frames, total_bytes, start, mode, source, settings)
                print(_format_stats(stats), flush=True)
                if jsonl_handle is not None:
                    jsonl_handle.write(json.dumps(stats))
                    jsonl_handle.write("\n")
                    jsonl_handle.flush()
                last_print = read_time
            if paced and settings.fps > 0:
                remaining = (1.0 / float(settings.fps)) - (time.time() - loop_start)
                if remaining > 0.0:
                    time.sleep(remaining)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    finally:
        cap.release()
        if jsonl_handle is not None:
            jsonl_handle.close()


def run_v4l2_ctl_probe(
    source: str | int,
    *,
    settings: GelsightSettings,
    window_sec: float,
    print_every_sec: float,
    duration_sec: float,
    jsonl_path: Path | None,
) -> None:
    device = _source_to_device_path(source)
    command = [
        "v4l2-ctl",
        "-d",
        device,
        f"--set-fmt-video=width={int(settings.width)},height={int(settings.height)},pixelformat=MJPG",
        f"--set-parm={float(settings.fps)}",
        "--stream-mmap=4",
        "--stream-to=/dev/null",
        "--verbose",
    ]
    if duration_sec > 0.0:
        command = ["timeout", f"{duration_sec + 3.0:.3f}s", *command]

    print("backend=v4l2-ctl command=" + " ".join(command), flush=True)
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    samples: collections.deque[dict[str, Any]] = collections.deque()
    total_frames = 0
    total_bytes = 0
    start = time.time()
    last_print = start
    last_timestamp: float | None = None
    jsonl_handle = None
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_handle = jsonl_path.open("a", encoding="utf-8")

    try:
        assert process.stdout is not None
        for line in process.stdout:
            parsed = _parse_v4l2_ctl_stream_line(line)
            if parsed is None:
                continue
            timestamp = parsed["timestamp"]
            frame_bytes = parsed["frame_bytes"]
            sample = {
                "wall_time": timestamp,
                "frame_bytes": frame_bytes,
                "shape": ["mjpeg"],
                "dtype": "mjpeg",
                "gap_sec": None if last_timestamp is None else timestamp - last_timestamp,
            }
            last_timestamp = timestamp
            samples.append(sample)
            total_frames += 1
            total_bytes += frame_bytes
            while samples and timestamp - float(samples[0]["wall_time"]) > window_sec:
                samples.popleft()

            now = time.time()
            if now - last_print >= print_every_sec:
                stats = _rolling_stats(samples, total_frames, total_bytes, start, "v4l2-ctl", device, settings)
                print(_format_stats(stats), flush=True)
                if jsonl_handle is not None:
                    jsonl_handle.write(json.dumps(stats))
                    jsonl_handle.write("\n")
                    jsonl_handle.flush()
                last_print = now
        return_code = process.wait()
        if return_code not in (0, 124):
            raise RuntimeError(f"v4l2-ctl exited with status {return_code}")
    except KeyboardInterrupt:
        process.terminate()
        print("\nStopped.", flush=True)
    finally:
        if process.poll() is None:
            process.terminate()
        if jsonl_handle is not None:
            jsonl_handle.close()


def _parse_v4l2_ctl_stream_line(line: str) -> dict[str, Any] | None:
    if "cap dqbuf:" not in line or "bytesused:" not in line or " ts:" not in line:
        return None
    parts = line.replace("(", " ").split()
    try:
        bytes_index = parts.index("bytesused:")
        ts_index = parts.index("ts:")
        seq_index = parts.index("seq:")
        return {
            "sequence_id": int(parts[seq_index + 1]),
            "frame_bytes": int(parts[bytes_index + 1]),
            "timestamp": float(parts[ts_index + 1]),
        }
    except (ValueError, IndexError):
        return None


def _rolling_stats(
    samples: collections.deque[dict[str, Any]],
    total_frames: int,
    total_bytes: int,
    start_time: float,
    mode: str,
    source: str | int,
    settings: GelsightSettings,
) -> dict[str, Any]:
    now = time.time()
    if len(samples) >= 2:
        window_span = float(samples[-1]["wall_time"]) - float(samples[0]["wall_time"])
        window_frames = len(samples)
        window_bytes = sum(int(sample["frame_bytes"]) for sample in samples)
        gaps = [float(sample["gap_sec"]) for sample in samples if sample["gap_sec"] is not None and sample["gap_sec"] > 0.0]
    else:
        window_span = 0.0
        window_frames = len(samples)
        window_bytes = sum(int(sample["frame_bytes"]) for sample in samples)
        gaps = []
    elapsed = max(0.0, now - start_time)
    return {
        "wall_time": now,
        "mode": mode,
        "source": str(source),
        "requested_width": settings.width,
        "requested_height": settings.height,
        "requested_fps": settings.fps,
        "window_sec": window_span,
        "window_frames": window_frames,
        "window_hz": (window_frames - 1) / window_span if window_span > 0.0 and window_frames >= 2 else None,
        "window_mib_per_sec": window_bytes / 1024**2 / window_span if window_span > 0.0 else None,
        "frame_bytes_min": min((int(sample["frame_bytes"]) for sample in samples), default=None),
        "frame_bytes_median": _median([int(sample["frame_bytes"]) for sample in samples]),
        "frame_bytes_max": max((int(sample["frame_bytes"]) for sample in samples), default=None),
        "gap_median_ms": _median(gaps, scale=1000.0),
        "gap_p95_ms": _quantile(gaps, 0.95, scale=1000.0),
        "gap_max_ms": max(gaps) * 1000.0 if gaps else None,
        "total_frames": total_frames,
        "total_mib": total_bytes / 1024**2,
        "total_hz": total_frames / elapsed if elapsed > 0.0 else None,
        "total_mib_per_sec": total_bytes / 1024**2 / elapsed if elapsed > 0.0 else None,
        "last_shape": list(samples[-1]["shape"]) if samples else None,
        "last_dtype": samples[-1]["dtype"] if samples else None,
    }


def _format_stats(stats: dict[str, Any]) -> str:
    return (
        "hz={hz} mib/s={mib_s} gap_med={gap_med}ms gap_p95={gap_p95}ms gap_max={gap_max}ms "
        "frame={frame_med}B range=[{frame_min},{frame_max}] total_hz={total_hz} shape={shape}"
    ).format(
        hz=_fmt(stats["window_hz"], 2),
        mib_s=_fmt(stats["window_mib_per_sec"], 2),
        gap_med=_fmt(stats["gap_median_ms"], 1),
        gap_p95=_fmt(stats["gap_p95_ms"], 1),
        gap_max=_fmt(stats["gap_max_ms"], 1),
        frame_med=_fmt(stats["frame_bytes_median"], 0),
        frame_min=stats["frame_bytes_min"],
        frame_max=stats["frame_bytes_max"],
        total_hz=_fmt(stats["total_hz"], 2),
        shape=stats["last_shape"],
    )


def _configure_capture(cap: Any, settings: GelsightSettings, *, mode: str, apply_controls: bool) -> None:
    import cv2

    if mode in {"decoded-mjpg", "mjpeg-payload"}:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if mode == "mjpeg-payload":
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
    cap.set(cv2.CAP_PROP_FPS, settings.fps)
    if apply_controls and settings.exposure is not None:
        cap.set(cv2.CAP_PROP_EXPOSURE, settings.exposure)
    if apply_controls and settings.contrast is not None:
        cap.set(cv2.CAP_PROP_CONTRAST, settings.contrast)


def _capture_properties(cap: Any) -> dict[str, Any]:
    import cv2

    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    return {
        "width": cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        "height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "fourcc": _fourcc_to_str(fourcc),
        "convert_rgb": cap.get(cv2.CAP_PROP_CONVERT_RGB),
    }


def _fourcc_to_str(value: int) -> str:
    chars = []
    for shift in (0, 8, 16, 24):
        code = (value >> shift) & 0xFF
        chars.append(chr(code) if 32 <= code <= 126 else "?")
    return "".join(chars)


def _load_settings(args: argparse.Namespace) -> GelsightSettings:
    if args.task_config:
        settings = load_task_config(args.task_config).gelsight
    else:
        settings = GelsightSettings(enabled=True)
    updates = {}
    for key in ("camera_path", "camera_index", "width", "height", "fps", "exposure", "contrast"):
        value = getattr(args, key)
        if value is not None:
            updates[key] = value
    if updates:
        settings = settings.model_copy(update=updates)
    return settings


def _resolve_source(settings: GelsightSettings) -> str | int:
    if settings.camera_path:
        return settings.camera_path
    return GelsightPublisher(settings, quest_publisher=None)._resolve_capture_source()


def _source_to_device_path(source: str | int) -> str:
    if isinstance(source, int):
        return f"/dev/video{source}"
    return os.path.realpath(source)


def _print_device_info(source: str | int) -> None:
    if isinstance(source, str):
        resolved = os.path.realpath(source)
        print(f"device={source} resolved={resolved}", flush=True)
        video_name = Path(resolved).name
        sys_video = Path("/sys/class/video4linux") / video_name / "device"
        try:
            usb_device = _usb_device_from_video_device(sys_video.resolve())
        except Exception:
            usb_device = None
        if usb_device is not None:
            speed = _read_text(usb_device / "speed")
            product = _read_text(usb_device / "product")
            manufacturer = _read_text(usb_device / "manufacturer")
            print(f"usb={usb_device.name} speed={speed or 'unknown'}M product={product} manufacturer={manufacturer}", flush=True)
        _run_quiet(["v4l2-ctl", "-d", resolved, "--get-fmt-video", "--get-parm"])
        return
    print(f"device_index={source}", flush=True)


def _usb_device_from_video_device(path: Path) -> Path | None:
    for parent in [path, *path.parents]:
        speed_path = parent / "speed"
        product_path = parent / "product"
        if speed_path.exists() and product_path.exists():
            return parent
    return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _run_quiet(command: list[str]) -> None:
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True, timeout=2.0)
    except Exception as exc:
        print(f"failed to run {' '.join(command)}: {exc}", flush=True)
        return
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), flush=True)


def _median(values: list[float | int], *, scale: float = 1.0) -> float | None:
    if not values:
        return None
    return float(statistics.median(values)) * scale


def _quantile(values: list[float], q: float, *, scale: float = 1.0) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0] * scale
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return (ordered[lower] * (1.0 - weight) + ordered[upper] * weight) * scale


def _fmt(value: Any, digits: int) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


if __name__ == "__main__":
    main()
