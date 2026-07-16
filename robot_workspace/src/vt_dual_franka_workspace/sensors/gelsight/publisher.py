from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from threading import Event

from vt_dual_franka_shared.timing import precise_sleep

from ...publishers.quest_udp import QuestUdpPublisher
from ...recording.async_image_stream import AsyncImageStreamRecorder
from ...recording.canonical_preprocess1 import CanonicalPreprocess1StreamRecorder
from ...recording.episode_streams import EpisodeImageStreamRecorder
from ...recording.gelsight_buffered import BufferedGelsightFrameRecorder
from ...runtime.live_buffer import LiveSampleBuffer
from ...recording.raw_recorder import JsonlStreamRecorder
from ...settings import GelsightSettings
from ..standardization import standardize_camera_frame


class GelsightPublisher:
    def __init__(
        self,
        settings: GelsightSettings,
        quest_publisher: QuestUdpPublisher,
        frame_recorder: JsonlStreamRecorder | AsyncImageStreamRecorder | BufferedGelsightFrameRecorder | None = None,
        canonical_recorder: CanonicalPreprocess1StreamRecorder | None = None,
        episode_image_recorder: EpisodeImageStreamRecorder | None = None,
        frame_buffer: LiveSampleBuffer | None = None,
        image_format: str = "jpg",
    ) -> None:
        self.settings = settings
        self.quest_publisher = quest_publisher
        self.frame_recorder = frame_recorder
        self.canonical_recorder = canonical_recorder
        self.episode_image_recorder = episode_image_recorder
        self.frame_buffer = frame_buffer
        self.image_format = image_format
        self._frames_seen = 0

    def run(self, stop_event: Event | None = None) -> None:
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("OpenCV is required for the GelSight publisher") from exc

        source = self._resolve_capture_source()
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open GelSight camera source {source!r}")
        self._configure_capture(cap)
        time.sleep(1.0)

        period = 1.0 / self.settings.fps
        try:
            while stop_event is None or not stop_event.is_set():
                loop_start = time.time()
                ok, frame = cap.read()
                if not ok:
                    precise_sleep(period)
                    continue

                captured_wall_time = time.time()
                sequence_id = self._frames_seen
                metadata = {
                    "camera_name": self.settings.camera_name,
                    "captured_wall_time": captured_wall_time,
                    "sequence_id": sequence_id,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                    "dtype": str(frame.dtype),
                }
                standardized = standardize_camera_frame(
                    frame,
                    stream_name="tactile_left",
                    camera_name=self.settings.camera_name,
                    source_color="BGR",
                    captured_wall_time=captured_wall_time,
                    sequence_id=sequence_id,
                    metadata=metadata,
                )
                if self.frame_buffer is not None and self.canonical_recorder is None:
                    self.frame_buffer.update(
                        standardized.image_rgb.copy(),
                        metadata=standardized.metadata,
                        captured_wall_time=captured_wall_time,
                    )
                if self.episode_image_recorder is not None:
                    self.episode_image_recorder.record_frame(
                        standardized.image_rgb,
                        captured_wall_time=captured_wall_time,
                        sequence_id=sequence_id,
                        metadata=standardized.metadata,
                    )
                if self.frame_recorder is not None and self.settings.save_frames:
                    if isinstance(self.frame_recorder, BufferedGelsightFrameRecorder):
                        self.frame_recorder.record_frame(
                            frame,
                            captured_wall_time=captured_wall_time,
                            sequence_id=sequence_id,
                            metadata=metadata,
                        )
                    else:
                        self.frame_recorder.record_frame(
                            standardized.image_rgb,
                            frame_id=f"{captured_wall_time:.6f}".replace(".", "_"),
                            metadata=standardized.metadata,
                            image_format=self.image_format,
                            event_time=captured_wall_time,
                        )
                if self.canonical_recorder is not None:
                    self.canonical_recorder.record_frame(
                        frame,
                        captured_wall_time=captured_wall_time,
                        sequence_id=sequence_id,
                        metadata=metadata,
                        live_buffer=self.frame_buffer,
                    )

                self._frames_seen += 1
                precise_sleep(max(0.0, period - (time.time() - loop_start)))
        finally:
            cap.release()

    def _resolve_capture_source(self) -> str | int:
        if self.settings.camera_path:
            return self.settings.camera_path

        candidates = _list_v4l_video_capture_candidates()
        matched = _filter_candidates(
            candidates,
            name_contains=self.settings.device_name_contains,
            serial_number=self.settings.device_serial_number,
        )
        if matched:
            return matched[0]["device_path"]
        return self.settings.camera_index

    def _configure_capture(self, cap) -> None:
        try:
            import cv2
        except ImportError:
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings.height)
        cap.set(cv2.CAP_PROP_FPS, self.settings.fps)
        if not self.settings.apply_controls:
            return
        if self.settings.exposure is not None:
            cap.set(cv2.CAP_PROP_EXPOSURE, self.settings.exposure)
        if self.settings.contrast is not None:
            cap.set(cv2.CAP_PROP_CONTRAST, self.settings.contrast)

    @property
    def frames_seen(self) -> int:
        return self._frames_seen


def _filter_candidates(
    candidates: list[dict[str, str]],
    *,
    name_contains: str,
    serial_number: str,
) -> list[dict[str, str]]:
    filtered = candidates
    if name_contains:
        needle = name_contains.lower()
        filtered = [candidate for candidate in filtered if needle in candidate["name"].lower()]
    if serial_number:
        filtered = [candidate for candidate in filtered if candidate["serial_number"] == serial_number]
    return filtered


def _list_v4l_video_capture_candidates() -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    by_id_dir = Path("/dev/v4l/by-id")
    if by_id_dir.exists():
        for symlink in sorted(by_id_dir.iterdir()):
            try:
                resolved = symlink.resolve(strict=True)
            except FileNotFoundError:
                continue
            if not resolved.name.startswith("video"):
                continue
            info = _query_v4l_device(resolved)
            if info is None or not info["video_capture"]:
                continue
            info["stable_path"] = str(symlink)
            info["device_path"] = str(symlink)
            candidates.append(info)

    if candidates:
        return _deduplicate_candidates(candidates)

    for device in sorted(Path("/dev").glob("video*")):
        info = _query_v4l_device(device)
        if info is None or not info["video_capture"]:
            continue
        info["stable_path"] = str(device)
        info["device_path"] = str(device)
        candidates.append(info)
    return _deduplicate_candidates(candidates)


def _deduplicate_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    deduped: list[dict[str, str]] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.realpath(candidate["device_path"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _query_v4l_device(device: Path) -> dict[str, str] | None:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", str(device), "--all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None

    stdout = result.stdout
    video_capture = "Video Capture" in stdout and "Metadata Capture" not in _device_caps_section(stdout)
    card_type = _extract_v4l_value(stdout, "Card type") or _extract_v4l_value(stdout, "Model") or device.name
    serial_number = _extract_v4l_value(stdout, "Serial") or ""
    return {
        "name": card_type,
        "serial_number": serial_number,
        "video_capture": video_capture,
    }


def _device_caps_section(text: str) -> str:
    marker = "Device Caps"
    index = text.find(marker)
    if index == -1:
        return text
    return text[index : index + 256]


def _extract_v4l_value(text: str, key: str) -> str | None:
    prefix = f"{key:<17}:"
    for line in text.splitlines():
        if line.strip().startswith(f"{key: <17}:".strip()):
            _, _, value = line.partition(":")
            return value.strip()
        if line.startswith(prefix):
            _, _, value = line.partition(":")
            return value.strip()
    return None
