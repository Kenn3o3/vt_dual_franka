from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from uuid import uuid4

from vt_dual_franka_shared.models import ArmId, ControllerState, DualArmControllerState

from ..collection.controller_state import ControllerStateMonitor
from ..config import ArmEndpointSettings, WorkspaceSettings
from ..controller.client import ControllerClient

ARM_ORDER: tuple[ArmId, ArmId] = ("left", "right")


@dataclass
class ArmRuntimeContext:
    arm_id: ArmId
    controller: ControllerClient
    state_monitor: ControllerStateMonitor

    def get_state(self, max_age_sec: float | None = None) -> ControllerState:
        state = self.state_monitor.get_state(max_age_sec=max_age_sec)
        return state.model_copy(update={"arm_id": self.arm_id})


class DualArmCoordinator:
    def __init__(self, arms: dict[ArmId, ArmRuntimeContext]) -> None:
        missing = [arm for arm in ARM_ORDER if arm not in arms]
        if missing:
            raise ValueError(f"Missing arm contexts: {missing}")
        self.arms = arms

    @classmethod
    def from_workspace(cls, workspace: WorkspaceSettings, *, poll_hz: float = 60.0) -> "DualArmCoordinator":
        contexts: dict[ArmId, ArmRuntimeContext] = {}
        for arm_id in ARM_ORDER:
            settings = workspace.arms[arm_id]
            controller = _build_arm_controller(settings)
            monitor = ControllerStateMonitor(controller, poll_hz=poll_hz)
            contexts[arm_id] = ArmRuntimeContext(arm_id=arm_id, controller=controller, state_monitor=monitor)
        return cls(contexts)

    def start(self) -> None:
        for context in self.arms.values():
            context.controller.assert_identity()
            context.state_monitor.start()

    def stop(self) -> None:
        for context in self.arms.values():
            context.state_monitor.stop()

    def get_state(self, max_age_sec: float | None = None) -> DualArmControllerState:
        return DualArmControllerState(
            left=self.arms["left"].get_state(max_age_sec=max_age_sec),
            right=self.arms["right"].get_state(max_age_sec=max_age_sec),
        )

    def queue_tcp_pair(
        self,
        targets: dict[ArmId, list[float]],
        *,
        source: str,
        target_duration_sec: float,
        command_id: str | None = None,
        start_delay_sec: float = 0.02,
    ) -> str:
        missing = [arm for arm in ARM_ORDER if arm not in targets]
        if missing:
            raise ValueError(f"Missing paired target(s): {missing}")
        command_id = command_id or f"dual-{uuid4().hex}"
        target_monotonic_time = time.monotonic() + max(float(start_delay_sec), 0.0) + float(target_duration_sec)
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-tcp") as pool:
            futures = {
                pool.submit(
                    self.arms[arm_id].controller.queue_tcp,
                    list(targets[arm_id]),
                    source,
                    target_duration_sec,
                    command_id=command_id,
                    target_monotonic_time=target_monotonic_time,
                ): arm_id
                for arm_id in ARM_ORDER
            }
            errors: dict[ArmId, str] = {}
            for future in as_completed(futures):
                arm_id = futures[future]
                try:
                    future.result()
                except Exception as exc:  # pragma: no cover - exercised in integration tests
                    errors[arm_id] = str(exc)
            if errors:
                self._best_effort_hold_current(source=f"{source}_fail_closed")
                raise RuntimeError(f"Dual-arm command {command_id} failed: {errors}")
        return command_id

    def hold_current(self, *, source: str = "dual_arm_hold", duration_sec: float = 0.1) -> str:
        state = self.get_state(max_age_sec=None)
        return self.queue_tcp_pair(
            {"left": state.left.tcp_pose, "right": state.right.tcp_pose},
            source=source,
            target_duration_sec=duration_sec,
        )

    def _best_effort_hold_current(self, *, source: str, duration_sec: float = 0.1) -> None:
        try:
            state = self.get_state(max_age_sec=None)
        except Exception:
            return
        for arm_id, pose in (("left", state.left.tcp_pose), ("right", state.right.tcp_pose)):
            try:
                self.arms[arm_id].controller.queue_tcp(list(pose), source=source, target_duration_sec=duration_sec)
            except Exception:
                pass


def _build_arm_controller(settings: ArmEndpointSettings) -> ControllerClient:
    return ControllerClient(
        settings.host,
        settings.port,
        settings.request_timeout_sec,
        arm_id=settings.arm_id,
    )
