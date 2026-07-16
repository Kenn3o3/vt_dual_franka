from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field, field_validator

from vt_dual_franka_shared.models import ArmId


class Action(BaseModel):
    target_tcp: list[float] | None = None
    target_tcp_by_arm: dict[ArmId, list[float]] | None = None
    target_duration_sec: float | None = None
    gripper_width: float | None = None
    gripper_closed: bool | None = None
    gripper_width_by_arm: dict[ArmId, float] | None = None
    gripper_closed_by_arm: dict[ArmId, bool] | None = None
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

    @field_validator("target_tcp_by_arm")
    @classmethod
    def _validate_target_tcp_by_arm(cls, value: dict[ArmId, list[float]] | None) -> dict[ArmId, list[float]] | None:
        if value is None:
            return value
        for arm_id, pose in value.items():
            if len(pose) != 7:
                raise ValueError(f"target_tcp_by_arm[{arm_id!r}] must contain exactly 7 values")
        return value

    @field_validator("target_duration_sec")
    @classmethod
    def _validate_duration(cls, value: float | None) -> float | None:
        if value is not None and value <= 0.0:
            raise ValueError("target_duration_sec must be positive")
        return value

    @field_validator("gripper_width_by_arm", "gripper_closed_by_arm")
    @classmethod
    def _validate_dual_gripper_mapping(cls, value, info):
        if value is not None and set(value) != {"left", "right"}:
            raise ValueError(f"{info.field_name} must contain exactly left and right")
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
    def __init__(
        self,
        controller: ActionController,
        *,
        blocking_gripper: bool = True,
        force_gripper_closed: bool = False,
    ) -> None:
        self.controller = controller
        self.blocking_gripper = blocking_gripper
        self.force_gripper_closed = bool(force_gripper_closed)
        self._last_gripper_closed: bool | None = None
        self._last_gripper_width: float | None = None

    def reset(self) -> None:
        self._last_gripper_closed = None
        self._last_gripper_width = None

    def execute(self, action: Action, *, source: str = "policy_runner") -> None:
        executed_action = self.normalize_for_execution(action)
        self.execute_normalized(executed_action, source=source)

    def execute_normalized(self, action: Action, *, source: str = "policy_runner") -> None:
        self._execute_gripper(action, source=source)
        if action.target_tcp is not None:
            self.controller.queue_tcp(
                list(action.target_tcp),
                source=source,
                target_duration_sec=action.target_duration_sec,
            )

    def normalize_for_execution(self, action: Action) -> Action:
        if not self.force_gripper_closed:
            return action
        metadata = dict(action.metadata)
        metadata["force_gripper_closed"] = True
        return action.model_copy(
            update={
                "gripper_closed": True,
                "gripper_width": None,
                "metadata": metadata,
            }
        )

    def _execute_gripper(self, action: Action, *, source: str) -> None:
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


class DualActionExecutor:
    def __init__(self, coordinator, *, gripper_close_threshold: float = 0.5) -> None:
        self.coordinator = coordinator
        self.gripper_close_threshold = float(gripper_close_threshold)
        self._last_gripper_closed: dict[ArmId, bool | None] = {"left": None, "right": None}

    def reset(self) -> None:
        self._last_gripper_closed = {"left": None, "right": None}

    @staticmethod
    def normalize_for_execution(action: Action) -> Action:
        return action

    def execute_normalized(self, action: Action, *, source: str = "dual_policy_runner") -> None:
        if action.target_tcp_by_arm is None:
            raise ValueError("DualActionExecutor requires action.target_tcp_by_arm")
        self._execute_grippers(action, source=source)
        target_duration_sec = action.target_duration_sec or 0.1
        self.coordinator.queue_tcp_pair(
            action.target_tcp_by_arm,
            source=source,
            target_duration_sec=target_duration_sec,
        )

    def _execute_grippers(self, action: Action, *, source: str) -> None:
        closed_by_arm = action.gripper_closed_by_arm
        width_by_arm = action.gripper_width_by_arm
        for arm_id in ("left", "right"):
            controller = self.coordinator.arms[arm_id].controller
            if closed_by_arm is not None:
                wants_closed = bool(closed_by_arm[arm_id])
                if self._last_gripper_closed[arm_id] == wants_closed:
                    continue
                if wants_closed:
                    controller.grasp_gripper(
                        action.gripper_velocity,
                        action.gripper_force_limit,
                        source=source,
                        blocking=False,
                    )
                else:
                    width = 0.078 if width_by_arm is None else float(width_by_arm[arm_id])
                    controller.move_gripper(
                        width,
                        action.gripper_velocity,
                        action.gripper_force_limit,
                        source=source,
                        blocking=False,
                    )
                self._last_gripper_closed[arm_id] = wants_closed
            elif width_by_arm is not None:
                controller.move_gripper(
                    float(width_by_arm[arm_id]),
                    action.gripper_velocity,
                    action.gripper_force_limit,
                    source=source,
                    blocking=False,
                )


def action_to_json(action: Action) -> dict[str, Any]:
    return action.model_dump(mode="json", exclude_none=True)
