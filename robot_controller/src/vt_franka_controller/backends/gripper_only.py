from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence as SequenceABC
from typing import Any
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
        self._last_width = 0.078
        self._last_force = 0.0
        self._state_warning_logged = False

    def get_tcp_pose(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def get_controller_state(self, control_frequency_hz: float) -> ControllerState:
        gripper_state = self._gripper.get_state()
        gripper_width = self._read_state_float(
            gripper_state,
            ("width", "gripper_width", "q", "position", "positions"),
            fallback=self._last_width,
            label="width",
        )
        gripper_force = self._read_state_float(
            gripper_state,
            ("force", "gripper_force", "grasp_force", "max_force"),
            fallback=self._last_force,
            label="force",
            warn_on_missing=False,
        )
        self._last_width = gripper_width
        self._last_force = gripper_force
        return ControllerState(
            tcp_pose=self.get_tcp_pose().tolist(),
            tcp_velocity=[0.0] * 6,
            tcp_wrench=[0.0] * 6,
            joint_positions=[0.0] * 7,
            joint_velocities=[0.0] * 7,
            gripper_width=gripper_width,
            gripper_force=gripper_force,
            control_frequency_hz=control_frequency_hz,
            backend=self.name,
        )

    def _read_state_float(
        self,
        state: Any,
        names: Sequence[str],
        *,
        fallback: float,
        label: str,
        warn_on_missing: bool = True,
    ) -> float:
        for name in names:
            value = self._read_attr_or_key(state, name)
            number = self._coerce_state_float(value)
            if number is not None:
                return number
        number = self._coerce_state_float(state)
        if number is not None and label == "width":
            return number
        if warn_on_missing and not self._state_warning_logged:
            LOGGER.warning(
                "Could not read gripper %s from Polymetis state type=%s fields=%s; using fallback %.4f",
                label,
                type(state).__name__,
                self._describe_state_fields(state),
                fallback,
            )
            self._state_warning_logged = True
        return float(fallback)

    @staticmethod
    def _read_attr_or_key(state: Any, name: str) -> Any:
        if isinstance(state, Mapping) and name in state:
            return state[name]
        if hasattr(state, name):
            return getattr(state, name)
        return None

    @classmethod
    def _coerce_state_float(cls, value: Any) -> float | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return None
            value = value.reshape(-1)[0]
        elif isinstance(value, SequenceABC) and not isinstance(value, (str, bytes, bytearray)):
            if len(value) == 0:
                return None
            value = value[0]
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        return number

    @staticmethod
    def _describe_state_fields(state: Any) -> list[str]:
        if isinstance(state, Mapping):
            return [str(key) for key in state.keys()]
        if hasattr(state, "_fields"):
            return [str(field) for field in state._fields]
        if hasattr(state, "__dict__"):
            return sorted(str(key) for key in vars(state).keys())
        return [name for name in dir(state) if not name.startswith("_")][:20]

    def start_cartesian_impedance(self, stiffness: Sequence[float], damping: Sequence[float]) -> None:
        raise RuntimeError("Gripper-only backend does not support arm impedance control")

    def update_desired_tcp(self, target_pose6d: np.ndarray) -> None:
        raise RuntimeError("Gripper-only backend does not support arm motion")

    def move_gripper(self, width: float, velocity: float, force_limit: float) -> None:
        self._last_width = float(width)
        self._last_force = float(force_limit)
        self._gripper.goto(width=width, speed=velocity, force=force_limit)

    def grasp(self, velocity: float, force_limit: float) -> None:
        self._last_width = 0.0
        self._last_force = float(force_limit)
        self._gripper.grasp(speed=velocity, force=force_limit)

    def stop_gripper(self) -> None:
        stop = getattr(self._gripper, "stop", None)
        if stop is None:
            LOGGER.warning("Polymetis GripperInterface has no stop(); cannot interrupt gripper motion")
            return
        stop()

    def go_home(self, ee_pose: Sequence[float], duration_sec: float) -> None:
        raise RuntimeError("Gripper-only backend does not support arm motion")

    def move_to_joints(self, joint_positions: Sequence[float], duration_sec: float | None = None) -> None:
        raise RuntimeError("Gripper-only backend does not support arm motion")

    def shutdown(self) -> None:
        return None
