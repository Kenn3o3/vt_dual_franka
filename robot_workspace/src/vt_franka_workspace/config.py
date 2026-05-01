from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from vt_franka_shared.config import load_yaml_model


class ControllerClientSettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8092
    request_timeout_sec: float = 1.0


class TeleopSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8082
    loop_hz: float = 60.0
    tracking_button_index: int = 4
    trigger_close_threshold: float = 0.5
    relative_translation_scale: float = 1.0
    relative_rotation_scale: float = 0.3
    max_tracking_position_error_m: float = 0.3
    operator_yaw_offset_deg: float = 180.0
    use_force_control_for_gripper: bool = True
    max_gripper_width: float = 0.078
    min_gripper_width: float = 0.0
    grasp_force: float = 7.0
    gripper_velocity: float = 0.1
    gripper_stability_window: int = 30
    gripper_force_close_threshold: float = 15.0
    gripper_force_open_threshold: float = 5.0
    gripper_width_vis_precision: float = 0.001
    command_record_hz: float = 0.0
    quest_message_record_hz: float = 0.0


class QuestFeedbackSettings(BaseModel):
    quest_ip: str = "127.0.0.1"
    robot_state_udp_port: int = 10001
    tactile_udp_port: int = 10002
    image_udp_port: int = 10004
    force_udp_port: int = 10005
    state_publish_hz: float = 60.0
    force_scale_factor: float = 0.025
    record_hz: float = 0.0


class QuestImageStreamSettings(BaseModel):
    enabled: bool = False
    image_id: str = ""
    in_head_space: bool = False
    left_or_right: bool = False
    position: list[float] = Field(default_factory=lambda: [0.0, 0.4, 0.5])
    rotation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = Field(default_factory=lambda: [0.002, 0.0015, 0.001])
    max_width: int = 320
    max_height: int = 240
    quality: int = 30
    chunk_size: int = 1024
    max_publish_hz: float = 12.0


class GelsightSettings(BaseModel):
    enabled: bool = False
    camera_name: str = "left_gripper_camera_1"
    camera_index: int = 0
    camera_path: str = ""
    device_name_contains: str = "GelSight Mini"
    device_serial_number: str = ""
    fps: int = 15
    width: int = 640
    height: int = 480
    exposure: int | None = -6
    contrast: int | None = 100
    marker_dimension: int = 2
    marker_vis_rotation_deg: float = 0.0
    vis_latency_steps: int = 5
    teleop_status_host: str = "127.0.0.1"
    teleop_status_port: int = 8082
    save_frames: bool = False
    record_hz: float = 0.0
    quest_stream: QuestImageStreamSettings = Field(default_factory=QuestImageStreamSettings)
    quest_overlay_arrow_scale: float = 12.0


class RgbCameraSettings(BaseModel):
    enabled: bool = True
    backend: Literal["orbbec"] = "orbbec"
    stream_name: str = ""
    camera_name: str = ""
    serial_number: str = ""
    color_width: int = 640
    color_height: int = 0
    color_format: str = "RGB"
    color_fps: int = 30
    frame_timeout_ms: int = 200
    save_frames: bool = True
    record_hz: float = 0.0
    quest_stream: QuestImageStreamSettings = Field(default_factory=QuestImageStreamSettings)


class RecordingSettings(BaseModel):
    enabled: bool = True
    collect_root: Path = Path("./data/collect")
    prepared_root: Path = Path("./data/prepared")
    train_root: Path = Path("./data/train")
    eval_root: Path = Path("./data/eval")
    checkpoints_root: Path = Path("./data/checkpoints")
    image_format: str = "jpg"


class OperatorUiSettings(BaseModel):
    enabled: bool = True
    host: str = "127.0.0.1"
    port: int = 8083
    log_buffer_size: int = 1000
    snapshot_max_age_sec: float = 1.5


class CalibrationSettings(BaseModel):
    calibration_dir: Path = Path("config/calibration/v6")


class WorkspaceSettings(BaseModel):
    controller: ControllerClientSettings = Field(default_factory=ControllerClientSettings)
    teleop: TeleopSettings = Field(default_factory=TeleopSettings)
    quest_feedback: QuestFeedbackSettings = Field(default_factory=QuestFeedbackSettings)
    recording: RecordingSettings = Field(default_factory=RecordingSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    operator_ui: OperatorUiSettings = Field(default_factory=OperatorUiSettings)


class ModalitySettings(BaseModel):
    proprioception: bool = True
    rgb_cameras: list[str] = Field(default_factory=list)
    gelsight_markers: bool = False
    gelsight_frame: bool = False
    controller_state_max_age_sec: float = 2.0
    rgb_camera_max_age_sec: float = 2.0
    gelsight_max_age_sec: float = 2.0

    def needs_gelsight(self) -> bool:
        return self.gelsight_markers or self.gelsight_frame


class EvalRuntimeSettings(BaseModel):
    enabled: bool = True
    cameras: list[str] = Field(default_factory=lambda: ["third_person"])
    video_hz: float = 10.0

    @field_validator("cameras")
    @classmethod
    def _normalize_cameras(cls, value: list[str]) -> list[str]:
        aliases = {"third": "third_person", "third_person": "third_person", "wrist": "wrist"}
        normalized: list[str] = []
        for camera in value:
            for item in camera.replace(",", "+").split("+"):
                key = item.strip().lower()
                if not key:
                    continue
                if key not in aliases:
                    supported = ", ".join(sorted([*aliases, "wrist+third"]))
                    raise ValueError(f"Unsupported eval camera {camera!r}. Supported cameras: {supported}")
                role = aliases[key]
                if role not in normalized:
                    normalized.append(role)
        return normalized

    @field_validator("video_hz")
    @classmethod
    def _validate_video_hz(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("eval.video_hz must be positive")
        return value


class CollectionRuntimeSettings(BaseModel):
    controller_state_poll_hz: float = 60.0
    quest_message_timeout_sec: float = 2.0
    require_quest_connection: bool = True
    start_countdown_sec: float = 2.0
    status_print_hz: float = 1.0
    record_raw_quest_messages: bool = False
    initial_pose_tolerance_m: float = 0.015
    initial_pose_tolerance_deg: float = 10.0
    initial_pose_settle_timeout_sec: float = 8.0
    initial_pose_settle_dwell_sec: float = 0.3


class TaskConfig(BaseModel):
    task_name: str
    initial_eef_pose_xyz_rpy_deg: list[float]
    initial_move_duration_sec: float = 2.0
    modality: ModalitySettings = Field(default_factory=ModalitySettings)
    collection: CollectionRuntimeSettings = Field(default_factory=CollectionRuntimeSettings)
    rgb_cameras: dict[str, RgbCameraSettings] = Field(default_factory=dict)
    gelsight: GelsightSettings = Field(default_factory=GelsightSettings)

    @field_validator("initial_eef_pose_xyz_rpy_deg")
    @classmethod
    def _validate_initial_pose(cls, value: list[float]) -> list[float]:
        if len(value) != 6:
            raise ValueError("initial_eef_pose_xyz_rpy_deg must contain exactly 6 values")
        return value


class InferenceRuntimeSettings(BaseModel):
    obs_horizon: int = 2
    exe_horizon: int = 1
    control_hz: float = 10.0
    max_duration_sec: float = 30.0
    start_countdown_sec: float = 2.0
    status_print_hz: float = 1.0
    task_name: str = "policy_run"
    initial_eef_pose_xyz_rpy_deg: list[float] | None = None
    initial_move_duration_sec: float = 2.0
    modality: ModalitySettings = Field(default_factory=ModalitySettings)
    eval: EvalRuntimeSettings = Field(default_factory=EvalRuntimeSettings)
    rgb_cameras: dict[str, RgbCameraSettings] = Field(default_factory=dict)
    gelsight: GelsightSettings = Field(default_factory=GelsightSettings)
    controller_state_poll_hz: float = 60.0
    initial_pose_tolerance_m: float = 0.015
    initial_pose_tolerance_deg: float = 10.0
    initial_pose_settle_timeout_sec: float = 8.0
    initial_pose_settle_dwell_sec: float = 0.3

    @field_validator("obs_horizon", "exe_horizon")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("horizon values must be positive")
        return value

    @field_validator("initial_eef_pose_xyz_rpy_deg")
    @classmethod
    def _validate_optional_initial_pose(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 6:
            raise ValueError("initial_eef_pose_xyz_rpy_deg must contain exactly 6 values")
        return value


class PolicyConfig(BaseModel):
    type: str
    checkpoint_path: Path | None = None
    config: dict[str, Any] = Field(default_factory=dict)


def load_workspace_config(path: str | Path) -> WorkspaceSettings:
    model = load_yaml_model(path, WorkspaceSettings)
    _resolve_model_paths(model, Path(path).resolve().parent)
    return model


def load_task_config(path: str | Path, *, task_name_override: str | None = None) -> TaskConfig:
    model = load_yaml_model(path, TaskConfig)
    _resolve_model_paths(model, Path(path).resolve().parent)
    if task_name_override:
        model.task_name = task_name_override
    return model


def load_inference_config(path: str | Path) -> InferenceRuntimeSettings:
    model = load_yaml_model(path, InferenceRuntimeSettings)
    _resolve_model_paths(model, Path(path).resolve().parent)
    return model


def load_policy_config(path: str | Path) -> PolicyConfig:
    model = load_yaml_model(path, PolicyConfig)
    _resolve_model_paths(model, Path(path).resolve().parent)
    return model


def _resolve_model_paths(value: Any, base_dir: Path) -> None:
    if hasattr(type(value), "model_fields"):
        for name in type(value).model_fields:
            item = getattr(value, name)
            if isinstance(item, Path) and not item.is_absolute():
                setattr(value, name, (base_dir / item).resolve())
            else:
                _resolve_model_paths(item, base_dir)
    elif isinstance(value, dict):
        for item in value.values():
            _resolve_model_paths(item, base_dir)
    elif isinstance(value, list):
        for item in value:
            _resolve_model_paths(item, base_dir)
