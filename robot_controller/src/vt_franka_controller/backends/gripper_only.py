from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

from vt_franka_shared.models import ControllerState

from .base import FrankaBackend

LOGGER = logging.getLogger(__name__)


class PolymetisGripperOnlyBackend(FrankaBackend):
    name = "polymetis-gripper-only"

    def __init__(self, gripper_ip: str, gripper_port: int) -> None:
        try:
            from polymetis import GripperInterface
        except ImportError as exc:
            raise RuntimeError("Polymetis gripper-only backend requires the polymetis Python package") from exc
        self._gripper = GripperInterface(ip_address=gripper_ip, port=gripper_port)

    def get_tcp_pose(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def get_controller_state(self, control_frequency_hz: float) -> ControllerState:
        gripper_state = self._gripper.get_state()
        try:
            gripper_force = float(gripper_state.force)
        except (AttributeError, TypeError):
            gripper_force = 0.0
        return ControllerState(
            tcp_pose=self.get_tcp_pose().tolist(),
            tcp_velocity=[0.0] * 6,
            tcp_wrench=[0.0] * 6,
            joint_positions=[0.0] * 7,
            joint_velocities=[0.0] * 7,
            gripper_width=float(gripper_state.width),
            gripper_force=gripper_force,
            control_frequency_hz=control_frequency_hz,
            backend=self.name,
        )

    def start_cartesian_impedance(self, stiffness: Sequence[float], damping: Sequence[float]) -> None:
        raise RuntimeError("Gripper-only backend does not support arm impedance control")

    def update_desired_tcp(self, target_pose6d: np.ndarray) -> None:
        raise RuntimeError("Gripper-only backend does not support arm motion")

    def move_gripper(self, width: float, velocity: float, force_limit: float) -> None:
        self._gripper.goto(width=width, speed=velocity, force=force_limit)

    def grasp(self, velocity: float, force_limit: float) -> None:
        self._gripper.grasp(speed=velocity, force=force_limit)

    def stop_gripper(self) -> None:
        LOGGER.info("Polymetis gripper stop is not exposed; keeping current state")

    def go_home(self, ee_pose: Sequence[float], duration_sec: float) -> None:
        raise RuntimeError("Gripper-only backend does not support arm motion")

    def move_to_joints(self, joint_positions: Sequence[float], duration_sec: float | None = None) -> None:
        raise RuntimeError("Gripper-only backend does not support arm motion")

    def shutdown(self) -> None:
        return None

