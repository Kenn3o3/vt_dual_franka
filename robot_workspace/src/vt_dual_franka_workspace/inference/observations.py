from __future__ import annotations

import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from vt_dual_franka_shared.models import ControllerState

from ..config import ModalitySettings
from ..recording.image_io import write_rgb_image
from ..runtime.live_buffer import LiveSample, LiveSampleBuffer


class ObservationAssembler:
    def __init__(
        self,
        *,
        modality: ModalitySettings,
        state_provider,
        rgb_camera_buffers: dict[str, LiveSampleBuffer] | None = None,
        gelsight_frame_buffer: LiveSampleBuffer | None = None,
        image_format: str = "jpg",
        record_rgb_frames: bool = False,
        record_gelsight_frames: bool = True,
    ) -> None:
        self.modality = modality
        self.state_provider = state_provider
        self.rgb_camera_buffers = dict(rgb_camera_buffers or {})
        self.gelsight_frame_buffer = gelsight_frame_buffer
        self.image_format = image_format
        self.record_rgb_frames = bool(record_rgb_frames)
        self.record_gelsight_frames = bool(record_gelsight_frames)

    def assert_ready(self) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if self.modality.proprioception:
            try:
                self.state_provider(max_age_sec=self.modality.controller_state_max_age_sec)
            except Exception as exc:
                reasons.append(f"proprioception unavailable: {exc}")
        for role in self.modality.rgb_cameras:
            self._check_buffer(self.rgb_camera_buffers.get(role), self.modality.rgb_camera_max_age_sec, f"images.{role}", reasons)
        if self.modality.gelsight_frame:
            self._check_buffer(self.gelsight_frame_buffer, self.modality.gelsight_max_age_sec, "tactile.gelsight_frame", reasons)
        return not reasons, reasons

    def assemble(self, episode_dir: Path | None = None, step_index: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        observation: dict[str, Any] = {
            "proprioception": {},
            "images": {},
            "tactile": {},
            "timestamps": {"assembled_wall_time": time.time()},
        }
        recorded: dict[str, Any] = {"assembled_wall_time": observation["timestamps"]["assembled_wall_time"]}

        if self.modality.proprioception:
            state = self.state_provider(max_age_sec=self.modality.controller_state_max_age_sec)
            state_payload = state.model_dump(mode="json") if isinstance(state, ControllerState) else dict(state)
            observation["proprioception"]["controller_state"] = state_payload
            recorded["proprioception"] = {"controller_state": state_payload}

        image_records: dict[str, Any] = {}
        for role in self.modality.rgb_cameras:
            sample = self._required_sample(self.rgb_camera_buffers.get(role), self.modality.rgb_camera_max_age_sec, f"images.{role}")
            rel_path = (
                self._write_frame(episode_dir, sample.name, sample, step_index)
                if self.record_rgb_frames and episode_dir is not None and step_index is not None
                else None
            )
            observation["images"][role] = {
                "image": sample.data,
                "metadata": dict(sample.metadata),
                "captured_wall_time": sample.captured_wall_time,
            }
            image_records[role] = self._recorded_image_sample(sample, rel_path)
        if image_records:
            recorded["images"] = image_records

        tactile_records: dict[str, Any] = {}
        if self.modality.gelsight_frame:
            sample = self._required_sample(self.gelsight_frame_buffer, self.modality.gelsight_max_age_sec, "tactile.gelsight_frame")
            rel_path = (
                self._write_frame(episode_dir, "gelsight_frame", sample, step_index)
                if self.record_gelsight_frames and episode_dir is not None and step_index is not None
                else None
            )
            tactile_item = {
                "image": sample.data,
                "metadata": dict(sample.metadata),
                "captured_wall_time": sample.captured_wall_time,
            }
            observation["tactile"]["tactile_left"] = tactile_item
            observation["tactile"]["gelsight_frame"] = tactile_item
            tactile_records["gelsight_frame"] = self._recorded_image_sample(sample, rel_path)
        if tactile_records:
            recorded["tactile"] = tactile_records

        return observation, recorded

    @staticmethod
    def _check_buffer(buffer: LiveSampleBuffer | None, max_age_sec: float, name: str, reasons: list[str]) -> None:
        if buffer is None:
            reasons.append(f"{name} buffer is not configured")
            return
        try:
            buffer.get_latest(max_age_sec=max_age_sec)
        except RuntimeError as exc:
            reasons.append(str(exc))

    @staticmethod
    def _required_sample(buffer: LiveSampleBuffer | None, max_age_sec: float, name: str) -> LiveSample:
        if buffer is None:
            raise RuntimeError(f"{name} buffer is not configured")
        return buffer.get_latest(max_age_sec=max_age_sec)

    def _write_frame(self, episode_dir: Path, stream_name: str, sample: LiveSample, step_index: int) -> str:
        frame_dir = episode_dir / "streams" / stream_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"step_{step_index:06d}.{self.image_format}"
        write_rgb_image(frame_path, sample.data, quality=90)
        return frame_path.relative_to(episode_dir).as_posix()

    @staticmethod
    def _recorded_image_sample(sample: LiveSample, rel_path: str | None) -> dict[str, Any]:
        image = sample.data
        height = int(image.shape[0]) if hasattr(image, "shape") and len(image.shape) >= 2 else 0
        width = int(image.shape[1]) if hasattr(image, "shape") and len(image.shape) >= 2 else 0
        payload = {
            "captured_wall_time": sample.captured_wall_time,
            "frame_width": width,
            "frame_height": height,
            "metadata": _json_safe(sample.metadata),
        }
        if rel_path is not None:
            payload["frame_path"] = rel_path
        return payload


class ObservationHistory:
    def __init__(self, horizon: int) -> None:
        if horizon <= 0:
            raise ValueError("Observation horizon must be positive")
        self.horizon = int(horizon)
        self._items: deque[dict[str, Any]] = deque(maxlen=self.horizon)

    def initialize_with_padding(self, observation: dict[str, Any]) -> None:
        self._items.clear()
        for _ in range(self.horizon):
            self._items.append(deepcopy(observation))

    def append(self, observation: dict[str, Any]) -> None:
        self._items.append(deepcopy(observation))

    def window(self) -> list[dict[str, Any]]:
        if len(self._items) != self.horizon:
            raise RuntimeError(f"Observation history is not initialized: {len(self._items)}/{self.horizon}")
        return list(self._items)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value
