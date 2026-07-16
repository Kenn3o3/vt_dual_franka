from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from vt_dual_franka_shared.models import ArmId, DualArmControllerState

from ..recording.image_io import write_rgb_image
from ..runtime.dual_arm import ARM_ORDER, DualArmCoordinator
from ..runtime.live_buffer import LiveSampleBuffer


class BimanualObservationAssembler:
    def __init__(
        self,
        *,
        coordinator: DualArmCoordinator,
        rgb_camera_buffers: dict[str, LiveSampleBuffer],
        tactile_buffers: dict[ArmId, LiveSampleBuffer],
        image_format: str = "jpg",
        state_max_age_sec: float = 2.0,
        image_max_age_sec: float = 2.0,
        tactile_max_age_sec: float = 2.0,
        record_frames: bool = False,
    ) -> None:
        self.coordinator = coordinator
        self.rgb_camera_buffers = rgb_camera_buffers
        self.tactile_buffers = tactile_buffers
        self.image_format = image_format
        self.state_max_age_sec = float(state_max_age_sec)
        self.image_max_age_sec = float(image_max_age_sec)
        self.tactile_max_age_sec = float(tactile_max_age_sec)
        self.record_frames = bool(record_frames)

    def assemble(self, episode_dir: Path | None = None, step_index: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        dual_state = self.coordinator.get_state(max_age_sec=self.state_max_age_sec)
        observation: dict[str, Any] = {
            "proprioception": {"controller_state_by_arm": _state_payload(dual_state)},
            "images": {},
            "tactile": {},
            "timestamps": {"assembled_wall_time": time.time()},
        }
        recorded: dict[str, Any] = {
            "assembled_wall_time": observation["timestamps"]["assembled_wall_time"],
            "proprioception": {"controller_state_by_arm": observation["proprioception"]["controller_state_by_arm"]},
        }
        for arm_id in ARM_ORDER:
            image_key = f"{arm_id}_wrist"
            stream_key = f"rgb_wrist_{arm_id}"
            sample = self.rgb_camera_buffers[stream_key].get_latest(max_age_sec=self.image_max_age_sec)
            rel_path = self._write_frame(episode_dir, stream_key, sample.data, step_index) if self._should_write(episode_dir, step_index) else None
            observation["images"][image_key] = {"image": sample.data, "metadata": dict(sample.metadata), "captured_wall_time": sample.captured_wall_time}
            recorded.setdefault("images", {})[image_key] = _recorded_sample(sample, rel_path)

            tactile = self.tactile_buffers[arm_id].get_latest(max_age_sec=self.tactile_max_age_sec)
            tactile_key = arm_id
            tactile_stream = f"tactile_{arm_id}"
            tactile_rel = self._write_frame(episode_dir, tactile_stream, tactile.data, step_index) if self._should_write(episode_dir, step_index) else None
            observation["tactile"][tactile_key] = {"image": tactile.data, "metadata": dict(tactile.metadata), "captured_wall_time": tactile.captured_wall_time}
            recorded.setdefault("tactile", {})[tactile_key] = _recorded_sample(tactile, tactile_rel)
        return observation, recorded

    def _should_write(self, episode_dir: Path | None, step_index: int | None) -> bool:
        return self.record_frames and episode_dir is not None and step_index is not None

    def _write_frame(self, episode_dir: Path | None, stream_name: str, image, step_index: int | None) -> str:
        assert episode_dir is not None and step_index is not None
        frame_dir = episode_dir / "streams" / stream_name
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"step_{step_index:06d}.{self.image_format}"
        write_rgb_image(frame_path, image, quality=90)
        return frame_path.relative_to(episode_dir).as_posix()


def _state_payload(dual_state: DualArmControllerState) -> dict[ArmId, dict[str, Any]]:
    return {
        "left": dual_state.left.model_dump(mode="json"),
        "right": dual_state.right.model_dump(mode="json"),
    }


def _recorded_sample(sample, rel_path: str | None) -> dict[str, Any]:
    payload = {
        "captured_wall_time": sample.captured_wall_time,
        "metadata": dict(sample.metadata),
    }
    if rel_path is not None:
        payload["frame_path"] = rel_path
    return payload
