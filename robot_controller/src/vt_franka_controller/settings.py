from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ServerSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8092


class TeleopGripperDefaults(BaseModel):
    max_gripper_width: float = 0.078
    gripper_velocity: float = 0.1
    grasp_force: float = 7.0


class BackendSettings(BaseModel):
    kind: Literal["polymetis", "mock"] = "polymetis"
    robot_ip: str = "127.0.0.1"
    robot_port: int = 50051
    gripper_ip: str = "127.0.0.1"
    gripper_port: int = 50052


class RosGripperActionSettings(BaseModel):
    action_namespace: str = "/franka_gripper"
    joint_states_topic: str = "/franka_gripper/joint_states"
    max_gripper_width: float = 0.078
    close_width_threshold: float = 0.001
    default_velocity: float = 0.05
    default_force_limit: float = 7.0
    grasp_epsilon_inner: float = 0.001
    grasp_epsilon_outer: float = 0.08
    action_server_timeout_sec: float = 2.0
    action_result_timeout_sec: float = 10.0
    state_stale_after_sec: float = 1.0
    home_on_start: bool = False


class RosGripperTestbedSettings(BaseModel):
    server: ServerSettings = Field(default_factory=lambda: ServerSettings(port=8094))
    ros: RosGripperActionSettings = Field(default_factory=RosGripperActionSettings)
    control_frequency_hz: float = 60.0


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
    max_policy_pos_speed_m_s: float = float("inf")
    max_policy_rot_speed_rad_s: float = float("inf")
    max_queue_size: int = 256


class ControllerSettings(BaseModel):
    server: ServerSettings = Field(default_factory=ServerSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    control: ControlSettings = Field(default_factory=ControlSettings)
    teleop: TeleopGripperDefaults = Field(default_factory=TeleopGripperDefaults)
