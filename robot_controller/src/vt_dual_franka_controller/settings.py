from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from vt_dual_franka_shared.models import ArmId


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
    expected_physical_robot_ip: Optional[str] = None


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
    arm_id: ArmId = "left"
    server: ServerSettings = Field(default_factory=ServerSettings)
    backend: BackendSettings = Field(default_factory=BackendSettings)
    control: ControlSettings = Field(default_factory=ControlSettings)
    teleop: TeleopGripperDefaults = Field(default_factory=TeleopGripperDefaults)

    @model_validator(mode="after")
    def _validate_fixed_dual_endpoint(self) -> "ControllerSettings":
        if self.backend.kind != "polymetis":
            return self
        expected = {
            "left": {
                "server_port": 8092,
                "robot_port": 50051,
                "gripper_port": 50052,
                "physical_robot_ip": "172.16.0.2",
            },
            "right": {
                "server_port": 8093,
                "robot_port": 50061,
                "gripper_port": 50062,
                "physical_robot_ip": "172.16.1.2",
            },
        }[self.arm_id]
        actual = {
            "server_port": self.server.port,
            "robot_port": self.backend.robot_port,
            "gripper_port": self.backend.gripper_port,
            "physical_robot_ip": self.backend.expected_physical_robot_ip,
        }
        if actual != expected:
            raise ValueError(
                f"{self.arm_id} controller endpoint mismatch: expected={expected}, actual={actual}"
            )
        if self.backend.robot_ip not in {"127.0.0.1", "localhost"}:
            raise ValueError("Controller must connect to its local Polymetis robot server")
        if self.backend.gripper_ip not in {"127.0.0.1", "localhost"}:
            raise ValueError("Controller must connect to its local Polymetis gripper server")
        return self
