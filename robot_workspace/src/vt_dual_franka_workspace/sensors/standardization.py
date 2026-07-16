from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from ..recording.image_io import ensure_hwc_uint8_rgb


STANDARD_CAMERA_WIDTH = 640
STANDARD_CAMERA_HEIGHT = 480
STANDARD_CAMERA_SHAPE = (STANDARD_CAMERA_HEIGHT, STANDARD_CAMERA_WIDTH, 3)
STANDARDIZATION_VERSION = "vt_franka_camera_standard_rgb_640x480_v1"

ColorSpace = Literal["BGR", "RGB"]


@dataclass(frozen=True)
class StandardizedFrame:
    image_rgb: np.ndarray
    metadata: dict[str, Any]
    captured_wall_time: float
    sequence_id: int | None = None


def standardize_camera_frame(
    frame: np.ndarray,
    *,
    stream_name: str,
    camera_name: str,
    source_color: ColorSpace = "BGR",
    captured_wall_time: float | None = None,
    sequence_id: int | None = None,
    metadata: dict[str, Any] | None = None,
    output_size: tuple[int, int] = (STANDARD_CAMERA_WIDTH, STANDARD_CAMERA_HEIGHT),
) -> StandardizedFrame:
    cv2 = _require_cv2()
    captured = float(captured_wall_time if captured_wall_time is not None else time.time())
    source = np.asarray(frame)
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 camera frame for {stream_name}, got shape {source.shape}")
    if source.dtype != np.uint8:
        source = np.clip(source, 0, 255).astype(np.uint8)
    if source_color == "BGR":
        image_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    elif source_color == "RGB":
        image_rgb = np.ascontiguousarray(source)
    else:
        raise ValueError(f"Unsupported source_color={source_color!r}")
    target_w, target_h = int(output_size[0]), int(output_size[1])
    if image_rgb.shape[:2] != (target_h, target_w):
        image_rgb = cv2.resize(image_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    image_rgb = ensure_hwc_uint8_rgb(image_rgb)
    payload = dict(metadata or {})
    payload.update(
        {
            "stream_name": stream_name,
            "camera_name": camera_name,
            "captured_wall_time": captured,
            "sequence_id": sequence_id,
            "source_shape": [int(v) for v in source.shape],
            "standard_shape": [int(v) for v in image_rgb.shape],
            "frame_width": int(image_rgb.shape[1]),
            "frame_height": int(image_rgb.shape[0]),
            "color_space": "RGB",
            "standardization": STANDARDIZATION_VERSION,
        }
    )
    return StandardizedFrame(
        image_rgb=image_rgb,
        metadata=payload,
        captured_wall_time=captured,
        sequence_id=sequence_id,
    )


def assert_standardized_rgb_frame(image: np.ndarray, metadata: dict[str, Any] | None = None) -> np.ndarray:
    array = ensure_hwc_uint8_rgb(image)
    if array.shape != STANDARD_CAMERA_SHAPE:
        raise ValueError(f"Expected standardized camera frame shape {STANDARD_CAMERA_SHAPE}, got {array.shape}")
    if metadata is not None:
        color_space = metadata.get("color_space")
        if color_space is not None and color_space != "RGB":
            raise ValueError(f"Expected standardized camera metadata color_space='RGB', got {color_space!r}")
    return array


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenCV is required for camera standardization") from exc
    return cv2
