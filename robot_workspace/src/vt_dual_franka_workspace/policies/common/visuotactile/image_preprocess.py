from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


CropMode = Literal["none", "center_square", "box"]


@dataclass(frozen=True)
class CropSpec:
    mode: CropMode = "center_square"
    box_xyxy: tuple[int, int, int, int] | None = None
    margin_fraction: float = 0.0

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "box_xyxy": None if self.box_xyxy is None else list(self.box_xyxy),
            "margin_fraction": float(self.margin_fraction),
        }


@dataclass(frozen=True)
class ImagePreprocessSpec:
    output_size: tuple[int, int]
    crop: CropSpec = CropSpec()
    interpolation: str = "area"

    def to_json(self) -> dict[str, Any]:
        return {
            "output_size": list(self.output_size),
            "crop": self.crop.to_json(),
            "interpolation": self.interpolation,
        }


def default_preprocess1_specs(
    *,
    canonical_size: int = 480,
    gelsight_crop_box: tuple[int, int, int, int] | None = None,
    gelsight_margin_fraction: float = 0.0,
) -> dict[str, ImagePreprocessSpec]:
    gelsight_crop = (
        CropSpec(mode="box", box_xyxy=gelsight_crop_box)
        if gelsight_crop_box is not None
        else CropSpec(mode="center_square", margin_fraction=gelsight_margin_fraction)
    )
    return {
        "rgb_wrist": ImagePreprocessSpec(
            output_size=(canonical_size, canonical_size),
            crop=CropSpec(mode="center_square", margin_fraction=0.0),
        ),
        "gelsight": ImagePreprocessSpec(
            output_size=(canonical_size, canonical_size),
            crop=gelsight_crop,
        ),
    }


def parse_crop_box(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None or not value.strip():
        return None
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("Crop box must be formatted as x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Crop box must satisfy x1>x0 and y1>y0")
    return x0, y0, x1, y1


def preprocess_image_rgb(image_rgb: np.ndarray, spec: ImagePreprocessSpec) -> np.ndarray:
    cv2 = _require_cv2()
    image = _ensure_hwc_uint8(image_rgb)
    cropped = crop_image(image, spec.crop)
    target_h, target_w = spec.output_size
    if cropped.shape[:2] != (target_h, target_w):
        cropped = cv2.resize(
            cropped,
            (target_w, target_h),
            interpolation=_cv2_interpolation(cv2, spec.interpolation),
        )
    return np.ascontiguousarray(cropped)


def crop_image(image: np.ndarray, spec: CropSpec) -> np.ndarray:
    height, width = image.shape[:2]
    if spec.mode == "none":
        cropped = image
    elif spec.mode == "center_square":
        margin = max(0, int(round(min(height, width) * float(spec.margin_fraction))))
        x0, y0 = margin, margin
        x1, y1 = width - margin, height - margin
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid margin_fraction={spec.margin_fraction} for image shape {image.shape}")
        size = min(x1 - x0, y1 - y0)
        x0 = x0 + (x1 - x0 - size) // 2
        y0 = y0 + (y1 - y0 - size) // 2
        x1 = x0 + size
        y1 = y0 + size
        cropped = image[y0:y1, x0:x1]
    elif spec.mode == "box":
        if spec.box_xyxy is None:
            raise ValueError("CropSpec mode='box' requires box_xyxy")
        x0, y0, x1, y1 = spec.box_xyxy
        x0 = int(np.clip(x0, 0, width - 1))
        y0 = int(np.clip(y0, 0, height - 1))
        x1 = int(np.clip(x1, x0 + 1, width))
        y1 = int(np.clip(y1, y0 + 1, height))
        cropped = image[y0:y1, x0:x1]
    else:
        raise ValueError(f"Unsupported crop mode: {spec.mode}")
    if cropped.size == 0:
        raise ValueError(f"Crop produced an empty image for shape {image.shape}: {spec}")
    return cropped


def read_image_file_as_rgb(path: Path) -> np.ndarray:
    cv2 = _require_cv2()
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Failed to read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def bgr_to_rgb(image_bgr: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    return cv2.cvtColor(_ensure_hwc_uint8(image_bgr), cv2.COLOR_BGR2RGB)


def rgb_to_bgr(image_rgb: np.ndarray) -> np.ndarray:
    cv2 = _require_cv2()
    return cv2.cvtColor(_ensure_hwc_uint8(image_rgb), cv2.COLOR_RGB2BGR)


def make_contact_sheet_rgb(
    panels: list[tuple[str, np.ndarray]],
    *,
    panel_size: int = 180,
    columns: int = 3,
) -> np.ndarray:
    cv2 = _require_cv2()
    if not panels:
        return np.full((panel_size, panel_size, 3), 255, dtype=np.uint8)
    columns = max(1, int(columns))
    rows = int(np.ceil(len(panels) / columns))
    label_h = 28
    gap = 8
    width = columns * panel_size + (columns + 1) * gap
    height = rows * (panel_size + label_h) + (rows + 1) * gap
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    for idx, (label, image) in enumerate(panels):
        row, col = divmod(idx, columns)
        x = gap + col * (panel_size + gap)
        y = gap + row * (panel_size + label_h + gap)
        thumb = cv2.resize(_ensure_hwc_uint8(image), (panel_size, panel_size), interpolation=cv2.INTER_AREA)
        canvas[y : y + panel_size, x : x + panel_size] = thumb
        cv2.putText(
            canvas,
            label[:32],
            (x, y + panel_size + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    return canvas


def _ensure_hwc_uint8(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 image, got shape {array.shape}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def _cv2_interpolation(cv2, name: str) -> int:
    key = name.strip().lower()
    if key == "area":
        return cv2.INTER_AREA
    if key == "linear":
        return cv2.INTER_LINEAR
    if key == "nearest":
        return cv2.INTER_NEAREST
    if key == "cubic":
        return cv2.INTER_CUBIC
    raise ValueError(f"Unsupported interpolation: {name}")


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenCV is required for visuotactile image preprocessing") from exc
    return cv2
