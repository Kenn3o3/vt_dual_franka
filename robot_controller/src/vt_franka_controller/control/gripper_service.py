from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from vt_franka_shared.models import ControllerState, GripperTestbedTargetCommand, HealthStatus

from ..backends.base import FrankaBackend
from ..settings import ControllerSettings

LOGGER = logging.getLogger(__name__)


@dataclass
class GripperServiceSnapshot:
    active: bool = False
    command_sequence: int = 0
    replaced_command_count: int = 0
    last_target_width: float | None = None
    last_trigger_depth: float | None = None
    last_force_limit: float | None = None
    last_command_source: str | None = None
    last_issue_wall_time: float | None = None
    last_complete_wall_time: float | None = None
    last_error: str | None = None
    in_flight: bool = False


@dataclass
class _QueuedGripperCommand:
    sequence: int
    command: GripperTestbedTargetCommand


class GripperTestbedService:
    def __init__(self, settings: ControllerSettings, backend: FrankaBackend) -> None:
        self.settings = settings
        self.backend = backend
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._running = threading.Event()
        self._worker: threading.Thread | None = None
        self._pending: _QueuedGripperCommand | None = None
        self._latest_sequence = 0
        self._snapshot = GripperServiceSnapshot()
        self._cached_state: ControllerState | None = None
        self._state_lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._running.is_set():
                return
            self._running.set()
            self._worker = threading.Thread(target=self._run, name="gripper-testbed-worker", daemon=True)
            self._worker.start()

    def shutdown(self) -> None:
        with self._lock:
            self._running.clear()
            self._condition.notify_all()
        if self._worker is not None:
            self._worker.join(timeout=2.0)

    def get_state(self) -> ControllerState:
        state = self.backend.get_controller_state(self.settings.control.control_frequency_hz)
        with self._state_lock:
            self._cached_state = state
        return state

    def get_health(self) -> HealthStatus:
        with self._lock:
            running = self._running.is_set()
            pending = self._pending is not None
        return HealthStatus(
            ok=running and self._snapshot.last_error is None,
            backend=self.backend.name,
            message="running" if running else "stopped",
            queue_depth=1 if pending else 0,
            control_loop_running=running,
            last_state_monotonic_time=self._snapshot.last_complete_wall_time,
        )

    def get_snapshot(self) -> dict[str, object]:
        with self._lock:
            snapshot = self._snapshot
        return {
            "active": snapshot.active,
            "command_sequence": snapshot.command_sequence,
            "replaced_command_count": snapshot.replaced_command_count,
            "last_target_width": snapshot.last_target_width,
            "last_trigger_depth": snapshot.last_trigger_depth,
            "last_force_limit": snapshot.last_force_limit,
            "last_command_source": snapshot.last_command_source,
            "last_issue_wall_time": snapshot.last_issue_wall_time,
            "last_complete_wall_time": snapshot.last_complete_wall_time,
            "last_error": snapshot.last_error,
            "in_flight": snapshot.in_flight,
        }

    def queue_target_command(self, command: GripperTestbedTargetCommand) -> dict[str, object]:
        with self._lock:
            if not self._running.is_set():
                raise RuntimeError("Gripper testbed service is not running")
            sequence = self._latest_sequence + 1
            self._latest_sequence = sequence
            replaced = self._pending is not None
            self._pending = _QueuedGripperCommand(sequence=sequence, command=command)
            self._snapshot.active = True
            self._snapshot.command_sequence = sequence
            self._snapshot.replaced_command_count += 1 if replaced else 0
            self._snapshot.last_target_width = float(command.target_width)
            self._snapshot.last_trigger_depth = command.trigger_depth
            self._snapshot.last_force_limit = float(command.force_limit)
            self._snapshot.last_command_source = command.source
            self._snapshot.last_issue_wall_time = command.issued_at_wall_time
            self._snapshot.last_error = None
            self._condition.notify_all()
        return {"status": "queued", "sequence": sequence, "replaced": replaced}

    def open_gripper(self, *, width: float | None = None, velocity: float | None = None, force_limit: float | None = None) -> dict[str, object]:
        command = GripperTestbedTargetCommand(
            target_width=float(width if width is not None else self.settings.teleop.max_gripper_width),
            velocity=float(velocity if velocity is not None else self.settings.teleop.gripper_velocity),
            force_limit=float(force_limit if force_limit is not None else self.settings.teleop.grasp_force),
            trigger_depth=0.0,
            source="gripper_testbed_open",
        )
        return self.queue_target_command(command)

    def stop_gripper(self) -> dict[str, object]:
        with self._lock:
            self._pending = None
            self._snapshot.active = False
            self._snapshot.in_flight = False
            self._condition.notify_all()
        self.backend.stop_gripper()
        return {"status": "stopped"}

    def _run(self) -> None:
        while True:
            with self._lock:
                while self._running.is_set() and self._pending is None:
                    self._condition.wait(timeout=0.1)
                if not self._running.is_set():
                    return
                pending = self._pending
                self._pending = None
                self._snapshot.in_flight = True
            if pending is None:
                continue
            command = pending.command
            start_wall = time.time()
            try:
                self.backend.move_gripper(command.target_width, command.velocity, command.force_limit)
                end_wall = time.time()
                with self._lock:
                    self._snapshot.last_complete_wall_time = end_wall
                    self._snapshot.in_flight = False
                    self._snapshot.last_error = None
            except Exception as exc:  # pragma: no cover - hardware failure path
                LOGGER.exception("Gripper testbed command failed")
                with self._lock:
                    self._snapshot.in_flight = False
                    self._snapshot.last_error = str(exc)
                    self._snapshot.last_complete_wall_time = start_wall
