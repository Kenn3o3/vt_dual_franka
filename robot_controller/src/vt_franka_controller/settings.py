from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8092


class BackendSettings(BaseModel):
    kind: Literal["polymetis", "mock"] = "polymetis"
    robot_ip: str = "127.0.0.1"
    robot_port: int = 50051
    gripper_ip: str = "127.0.0.1"
    gripper_port: int = 50052


class ControlSettings(BaseModel):
    control_frequency_hz: float = 300.0
    teleop_command_hz: float = 60.0
    state_cache_hz: float = 60.0
    cartesian_stiffness: List[float] = Field(default_factory=lambda: [750.0, 750.0, 750.0, 15.0, 15.0, 15.0])
    cartesian_damping: List[float] = Field(default_factory=lambda: [37.0, 37.0, 37.0, 2.0, 2.0, 2.0])
    home_ee_pose: List[float] = Field(default_factory=lambda: [0.4, 0.0, 0.3, 180.0, 0.0, 0.0])
    home_duration_sec: float = 8.0
    ready_ee_pose: Optional[List[float]] = None
    ready_duration_sec: float = 5.0
    ready_joint_positions: Optional[List[float]] = None
    ready_joint_duration_sec: Optional[float] = None
    reset_fast_path_position_threshold_m: float = 0.06
    reset_fast_path_rotation_threshold_deg: float = 12.0
    reset_settle_position_threshold_m: float = 0.01
    reset_settle_rotation_threshold_deg: float = 3.0
    reset_settle_timeout_sec: float = 3.0
    reset_settle_dwell_sec: float = 0.3
    max_queue_size: int = 256


class ControllerSettings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    control: ControlSettings = Field(default_factory=ControlSettings)
