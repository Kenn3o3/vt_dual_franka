from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator


class Action(BaseModel):
    target_tcp: list[float] | None = None
    target_duration_sec: float | None = None
    gripper_width: float | None = None
    gripper_closed: bool | None = None
    gripper_velocity: float = 0.1
    gripper_force_limit: float = 5.0
    terminate: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("target_tcp")
    @classmethod
    def _validate_target_tcp(cls, value: list[float] | None) -> list[float] | None:
        if value is not None and len(value) != 7:
            raise ValueError("target_tcp must contain exactly 7 values")
        return value

    @field_validator("target_duration_sec")
    @classmethod
    def _validate_duration(cls, value: float | None) -> float | None:
        if value is not None and value <= 0.0:
            raise ValueError("target_duration_sec must be positive")
        return value


class ActionController(Protocol):
    def queue_tcp(self, target_tcp: list[float], source: str = "workspace", target_duration_sec: float | None = None) -> None:
        ...

    def move_gripper(
        self,
        width: float,
        velocity: float,
        force_limit: float,
        source: str = "workspace",
        blocking: bool = False,
    ) -> None:
        ...

    def grasp_gripper(
        self,
        velocity: float,
        force_limit: float,
        source: str = "workspace",
        blocking: bool = False,
    ) -> None:
        ...


def normalize_action_chunk(raw_actions: Any) -> list[Action]:
    if raw_actions is None:
        raise ValueError("Policy returned None; expected a list of action dictionaries")
    if isinstance(raw_actions, dict):
        raw_actions = [raw_actions]
    if not isinstance(raw_actions, list):
        raise TypeError(f"Policy must return list[dict], got {type(raw_actions).__name__}")
    return [Action.model_validate(action) for action in raw_actions]


class ActionExecutor:
    def __init__(self, controller: ActionController, *, blocking_gripper: bool = True) -> None:
        self.controller = controller
        self.blocking_gripper = blocking_gripper
        self._last_gripper_closed: bool | None = None
        self._last_gripper_width: float | None = None

    def reset(self) -> None:
        self._last_gripper_closed = None
        self._last_gripper_width = None

    def execute(self, action: Action, *, source: str = "policy_runner") -> None:
        if action.target_tcp is not None:
            self.controller.queue_tcp(
                list(action.target_tcp),
                source=source,
                target_duration_sec=action.target_duration_sec,
            )
        if action.gripper_closed is True and self._last_gripper_closed is not True:
            self.controller.grasp_gripper(
                velocity=action.gripper_velocity,
                force_limit=action.gripper_force_limit,
                source=source,
                blocking=self.blocking_gripper,
            )
            self._last_gripper_closed = True
            self._last_gripper_width = None
        elif action.gripper_width is not None:
            width = float(action.gripper_width)
            if self._last_gripper_closed is False and self._last_gripper_width == width:
                return
            self.controller.move_gripper(
                width=width,
                velocity=action.gripper_velocity,
                force_limit=action.gripper_force_limit,
                source=source,
                blocking=self.blocking_gripper,
            )
            self._last_gripper_closed = False
            self._last_gripper_width = width
        elif action.gripper_closed is False and self._last_gripper_closed is None:
            self._last_gripper_closed = False


def action_to_json(action: Action) -> dict[str, Any]:
    return action.model_dump(mode="json", exclude_none=True)
