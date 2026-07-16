from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from uuid import uuid4

from vt_dual_franka_shared.models import ArmId, ControllerState, DualArmControllerState, ResetCommand

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
        for arm_id in ARM_ORDER:
            self.arms[arm_id].controller.assert_identity()
        for arm_id in ARM_ORDER:
            self.arms[arm_id].state_monitor.start()

    def stop(self) -> None:
        for arm_id in ARM_ORDER:
            self.arms[arm_id].state_monitor.stop()

    def is_healthy(self, max_age_sec: float = 2.0) -> bool:
        return all(self.arms[arm_id].state_monitor.is_healthy(max_age_sec=max_age_sec) for arm_id in ARM_ORDER)

    def snapshot(self) -> dict[ArmId, dict]:
        return {arm_id: self.arms[arm_id].state_monitor.snapshot() for arm_id in ARM_ORDER}

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

    def reset_pair(
        self,
        commands: dict[ArmId, ResetCommand],
        *,
        command_id: str | None = None,
    ) -> dict[ArmId, dict]:
        missing = [arm for arm in ARM_ORDER if arm not in commands]
        if missing:
            raise ValueError(f"Missing paired reset command(s): {missing}")
        command_id = command_id or f"dual-reset-{uuid4().hex}"
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-reset") as pool:
            futures = {
                pool.submit(
                    self.arms[arm_id].controller.reset,
                    commands[arm_id].model_copy(update={"arm_id": arm_id, "command_id": command_id}),
                ): arm_id
                for arm_id in ARM_ORDER
            }
            results: dict[ArmId, dict] = {}
            errors: dict[ArmId, str] = {}
            for future in as_completed(futures):
                arm_id = futures[future]
                try:
                    results[arm_id] = future.result()
                except Exception as exc:
                    errors[arm_id] = str(exc)
            if errors:
                self._best_effort_hold_current(source="dual_reset_fail_closed")
                raise RuntimeError(f"Dual-arm reset {command_id} failed: {errors}")
        return results

    def move_grippers(
        self,
        widths: dict[ArmId, float],
        *,
        velocity: float,
        force_limit: float,
        source: str,
        blocking: bool = True,
    ) -> None:
        missing = [arm for arm in ARM_ORDER if arm not in widths]
        if missing:
            raise ValueError(f"Missing paired gripper width(s): {missing}")
        self._run_paired(
            lambda arm_id: self.arms[arm_id].controller.move_gripper(
                widths[arm_id],
                velocity,
                force_limit,
                source=source,
                blocking=blocking,
            ),
            label="gripper move",
        )

    def grasp_grippers(
        self,
        *,
        velocity: float,
        force_limit: float,
        source: str,
        blocking: bool = True,
    ) -> None:
        self._run_paired(
            lambda arm_id: self.arms[arm_id].controller.grasp_gripper(
                velocity,
                force_limit,
                source=source,
                blocking=blocking,
            ),
            label="gripper grasp",
        )

    def _run_paired(self, callback, *, label: str) -> None:
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix="dual-arm") as pool:
            futures = {pool.submit(callback, arm_id): arm_id for arm_id in ARM_ORDER}
            errors: dict[ArmId, str] = {}
            for future in as_completed(futures):
                arm_id = futures[future]
                try:
                    future.result()
                except Exception as exc:
                    errors[arm_id] = str(exc)
            if errors:
                raise RuntimeError(f"Dual-arm {label} failed: {errors}")

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
