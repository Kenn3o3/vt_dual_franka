from __future__ import annotations

import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..config import GelsightSettings, RgbCameraSettings, TaskConfig
from ..recording.raw_recorder import _json_default
from .gelsight.publisher import _filter_candidates, _list_v4l_video_capture_candidates
from .orbbec.frame_decoder import decode_color_frame
from .rgb_camera import resolve_rgb_camera_specs
from .standardization import STANDARDIZATION_VERSION, standardize_camera_frame


@dataclass(frozen=True)
class CameraDiagnosticsConfig:
    task: TaskConfig
    output_root: Path
    duration_sec: float = 10.0
    include_rgb: bool = True
    include_gelsight: bool = True


@dataclass(frozen=True)
class CameraDiagnosticsResult:
    report_path: Path
    report: dict[str, Any]


def diagnose_task_cameras(config: CameraDiagnosticsConfig) -> CameraDiagnosticsResult:
    output_root = Path(config.output_root)
    run_dir = output_root / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": "vt_franka_camera_diagnostics_v1",
        "created_at_wall_time": time.time(),
        "task_name": config.task.task_name,
        "duration_sec": float(config.duration_sec),
        "standardization": STANDARDIZATION_VERSION,
        "streams": {},
    }
    if config.include_rgb:
        rgb_specs = {spec.role: spec for spec in resolve_rgb_camera_specs(config.task.rgb_cameras)}
        for role in config.task.modality.rgb_cameras:
            spec = rgb_specs.get(role)
            if spec is None:
                report["streams"][f"rgb:{role}"] = {"ok": False, "error": f"RGB camera role is not configured: {role}"}
                continue
            report["streams"][f"rgb:{role}"] = _diagnose_orbbec(spec.settings, duration_sec=config.duration_sec)
    if config.include_gelsight and config.task.modality.needs_gelsight():
        report["streams"]["tactile_left"] = _diagnose_gelsight(config.task.gelsight, duration_sec=config.duration_sec)
    report["ok"] = all(stream.get("ok", False) for stream in report["streams"].values())
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    return CameraDiagnosticsResult(report_path=report_path, report=report)


def _diagnose_gelsight(settings: GelsightSettings, *, duration_sec: float) -> dict[str, Any]:
    start = time.monotonic()
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        return _error_payload(start, exc)
    try:
        source = _resolve_gelsight_capture_source(settings)
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open GelSight camera source {source!r}")
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.height)
            cap.set(cv2.CAP_PROP_FPS, settings.fps)
            frames = 0
            standardize_times: list[float] = []
            source_shapes: set[tuple[int, ...]] = set()
            first_wall_time: float | None = None
            last_wall_time: float | None = None
            deadline = time.monotonic() + float(duration_sec)
            while time.monotonic() < deadline:
                ok, frame = cap.read()
                if not ok:
                    continue
                captured = time.time()
                t0 = time.monotonic()
                standardized = standardize_camera_frame(
                    frame,
                    stream_name="tactile_left",
                    camera_name=settings.camera_name,
                    source_color="BGR",
                    captured_wall_time=captured,
                    sequence_id=frames,
                )
                standardize_times.append(time.monotonic() - t0)
                source_shapes.add(tuple(int(v) for v in frame.shape))
                first_wall_time = captured if first_wall_time is None else first_wall_time
                last_wall_time = captured
                _assert_standardized_output(standardized.image_rgb)
                frames += 1
        finally:
            cap.release()
        return _success_payload(
            start,
            frames=frames,
            first_wall_time=first_wall_time,
            last_wall_time=last_wall_time,
            source_shapes=source_shapes,
            standardize_times=standardize_times,
            requested_fps=settings.fps,
            target_min_hz=9.5,
            camera_type="gelsight",
        )
    except Exception as exc:  # pragma: no cover - hardware dependent
        return _error_payload(start, exc)


def _diagnose_orbbec(settings: RgbCameraSettings, *, duration_sec: float) -> dict[str, Any]:
    start = time.monotonic()
    try:
        sdk = importlib.import_module("pyorbbecsdk")
        pipeline = _open_orbbec_pipeline(settings, sdk)
        try:
            frames = 0
            standardize_times: list[float] = []
            source_shapes: set[tuple[int, ...]] = set()
            first_wall_time: float | None = None
            last_wall_time: float | None = None
            deadline = time.monotonic() + float(duration_sec)
            while time.monotonic() < deadline:
                frameset = pipeline.wait_for_frames(settings.frame_timeout_ms)
                if frameset is None:
                    continue
                color_frame = frameset.get_color_frame()
                if color_frame is None:
                    continue
                image = decode_color_frame(color_frame)
                captured = time.time()
                t0 = time.monotonic()
                standardized = standardize_camera_frame(
                    image,
                    stream_name=settings.stream_name or "rgb_wrist",
                    camera_name=settings.camera_name or "orbbec",
                    source_color="BGR",
                    captured_wall_time=captured,
                    sequence_id=frames,
                )
                standardize_times.append(time.monotonic() - t0)
                source_shapes.add(tuple(int(v) for v in image.shape))
                first_wall_time = captured if first_wall_time is None else first_wall_time
                last_wall_time = captured
                _assert_standardized_output(standardized.image_rgb)
                frames += 1
        finally:
            pipeline.stop()
        return _success_payload(
            start,
            frames=frames,
            first_wall_time=first_wall_time,
            last_wall_time=last_wall_time,
            source_shapes=source_shapes,
            standardize_times=standardize_times,
            requested_fps=settings.color_fps,
            target_min_hz=None,
            camera_type="orbbec_rgb",
        )
    except Exception as exc:  # pragma: no cover - hardware dependent
        return _error_payload(start, exc)


def _open_orbbec_pipeline(settings: RgbCameraSettings, sdk: Any) -> Any:
    context = sdk.Context()
    device_list = context.query_devices()
    if int(device_list.get_count()) <= 0:
        raise RuntimeError("No Orbbec devices found")
    selected = None
    for index in range(int(device_list.get_count())):
        device = _device_at_index(device_list, index)
        serial = _safe_call(device.get_device_info(), "get_serial_number")
        if settings.serial_number and serial != settings.serial_number:
            continue
        selected = device
        break
    if selected is None:
        raise RuntimeError(f"Configured Orbbec serial_number was not found: {settings.serial_number}")
    pipeline = sdk.Pipeline(selected)
    config = sdk.Config()
    profile_list = pipeline.get_stream_profile_list(sdk.OBSensorType.COLOR_SENSOR)
    requested_format = getattr(sdk.OBFormat, settings.color_format.upper(), None)
    if requested_format is None:
        raise RuntimeError(f"Unsupported Orbbec color_format in config: {settings.color_format}")
    try:
        profile = profile_list.get_video_stream_profile(
            settings.color_width,
            settings.color_height,
            requested_format,
            settings.color_fps,
        )
    except Exception:
        profile = profile_list.get_default_video_stream_profile()
    config.enable_stream(profile)
    pipeline.start(config)
    return pipeline


def _resolve_gelsight_capture_source(settings: GelsightSettings) -> str | int:
    if settings.camera_path:
        return settings.camera_path
    matched = _filter_candidates(
        _list_v4l_video_capture_candidates(),
        name_contains=settings.device_name_contains,
        serial_number=settings.device_serial_number,
    )
    if matched:
        return matched[0]["device_path"]
    return settings.camera_index


def _success_payload(
    start: float,
    *,
    frames: int,
    first_wall_time: float | None,
    last_wall_time: float | None,
    source_shapes: set[tuple[int, ...]],
    standardize_times: list[float],
    requested_fps: float,
    target_min_hz: float | None,
    camera_type: str,
) -> dict[str, Any]:
    wall_duration = 0.0 if first_wall_time is None or last_wall_time is None else max(0.0, last_wall_time - first_wall_time)
    effective_hz = (frames - 1) / wall_duration if frames > 1 and wall_duration > 0.0 else 0.0
    standardize_ms = np.asarray(standardize_times, dtype=np.float64) * 1000.0
    meets_target = True if target_min_hz is None else effective_hz >= float(target_min_hz)
    return {
        "ok": bool(frames > 0 and meets_target),
        "camera_type": camera_type,
        "frames": int(frames),
        "requested_fps": float(requested_fps),
        "target_min_hz": target_min_hz,
        "effective_hz": float(effective_hz),
        "meets_target": bool(meets_target),
        "elapsed_sec": float(time.monotonic() - start),
        "source_shapes": [list(shape) for shape in sorted(source_shapes)],
        "standardize_ms_mean": float(standardize_ms.mean()) if len(standardize_ms) else None,
        "standardize_ms_p95": float(np.percentile(standardize_ms, 95)) if len(standardize_ms) else None,
    }


def _error_payload(start: float, exc: Exception) -> dict[str, Any]:
    return {
        "ok": False,
        "error": str(exc),
        "error_type": type(exc).__name__,
        "elapsed_sec": float(time.monotonic() - start),
    }


def _assert_standardized_output(image: np.ndarray) -> None:
    if image.shape != (480, 640, 3):
        raise RuntimeError(f"Standardized frame has wrong shape: {image.shape}")
    if image.dtype != np.uint8:
        raise RuntimeError(f"Standardized frame has wrong dtype: {image.dtype}")


def _device_at_index(device_list: Any, index: int) -> Any:
    try:
        return device_list[index]
    except Exception:
        return device_list.get_device_by_index(index)


def _safe_call(obj: Any, method_name: str) -> Any | None:
    method = getattr(obj, method_name, None)
    if method is None:
        return None
    try:
        return method()
    except Exception:
        return None
