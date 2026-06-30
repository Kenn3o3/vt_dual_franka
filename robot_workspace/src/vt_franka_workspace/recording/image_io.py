from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_hwc_uint8_rgb(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def write_rgb_jpeg(path: str | Path, image_rgb: np.ndarray, *, quality: int = 90) -> Path:
    cv2 = _require_cv2()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = ensure_hwc_uint8_rgb(image_rgb)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError(f"Failed to JPEG-encode RGB image: {output_path}")
    output_path.write_bytes(encoded.tobytes())
    return output_path


def read_rgb_image(path: str | Path) -> np.ndarray:
    cv2 = _require_cv2()
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return np.ascontiguousarray(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))


def write_rgb_image(path: str | Path, image_rgb: np.ndarray, *, quality: int = 90) -> Path:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return write_rgb_jpeg(path, image_rgb, quality=quality)
    cv2 = _require_cv2()
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = ensure_hwc_uint8_rgb(image_rgb)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(output_path), image_bgr):
        raise RuntimeError(f"Failed to write RGB image: {output_path}")
    return output_path


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenCV is required for RGB image IO") from exc
    return cv2
