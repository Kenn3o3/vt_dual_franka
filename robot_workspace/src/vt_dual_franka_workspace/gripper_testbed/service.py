from __future__ import annotations

import json
import logging
import shutil
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from vt_dual_franka_shared.models import ControllerState, GripperTestbedTargetCommand, UnityTeleopMessage, parse_unity_teleop_message
from vt_dual_franka_shared.timing import precise_sleep

from ..operator.logs import OperatorLogBuffer
from ..recording import JsonlStreamRecorder, RunSessionManager
from .client import GripperTestbedControllerClient

LOGGER = logging.getLogger(__name__)


def map_trigger_to_width(
    trigger_depth: float,
    *,
    min_width: float,
    max_width: float,
    gamma: float = 1.5,
) -> float:
    trigger = float(np.clip(trigger_depth, 0.0, 1.0))
    min_width = float(max(min_width, 0.0))
    max_width = float(max(max_width, min_width))
    return float(max_width - (trigger**gamma) * (max_width - min_width))


class GripperTestbedSettings(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8084
    controller_host: str = "127.0.0.1"
    controller_port: int = 8092
    controller_request_timeout_sec: float = 0.3
    loop_hz: float = 60.0
    command_hz: float = 20.0
    quest_message_timeout_sec: float = 1.0
    trigger_enable_button_index: int = 4
    min_gripper_width: float = 0.0
    max_gripper_width: float = 0.078
    gripper_velocity: float = 0.1
    force_limit: float = 7.0
    open_velocity: float = 0.05
    close_velocity: float = 0.02
    force_threshold: float = 7.0
    trigger_pressed_threshold: float = 0.2
    control_hand: Literal["left", "right"] = "right"
    width_gamma: float = 1.5
    width_deadband_m: float = 0.0015
    command_record_hz: float = 0.0
    quest_message_record_hz: float = 0.0
    state_record_hz: float = 30.0
    status_history_len: int = 240
    collect_root: Path = Path("./data/gripper_testbed")
    require_enable_button: bool = False

    @field_validator(
        "loop_hz",
        "command_hz",
        "quest_message_timeout_sec",
        "controller_request_timeout_sec",
        "gripper_velocity",
        "force_limit",
        "open_velocity",
        "close_velocity",
        "force_threshold",
        "trigger_pressed_threshold",
        "width_gamma",
        "status_history_len",
    )
    @classmethod
    def _validate_positive(cls, value: float, info) -> float:
        if value <= 0:
            raise ValueError(f"{info.field_name} must be positive")
        return value


@dataclass
class GripperTestbedSample:
    wall_time: float
    trigger_depth: float
    trigger_pressed: bool
    gripper_command: int
    target_width: float
    measured_width: float
    measured_force: float
    width_error: float
    command_latency_sec: float | None
    command_sequence: int | None
    in_flight: bool


class GripperTestbedRuntimeUpdate(BaseModel):
    min_gripper_width: float | None = None
    max_gripper_width: float | None = None
    gripper_velocity: float | None = None
    force_limit: float | None = None
    open_velocity: float | None = None
    close_velocity: float | None = None
    force_threshold: float | None = None
    trigger_pressed_threshold: float | None = None
    control_hand: Literal["left", "right"] | None = None
    width_gamma: float | None = None
    width_deadband_m: float | None = None
    command_hz: float | None = None
    require_enable_button: bool | None = None


class GripperTestbedService:
    def __init__(
        self,
        settings: GripperTestbedSettings,
        controller: GripperTestbedControllerClient,
        *,
        operator_log_buffer: OperatorLogBuffer | None = None,
    ) -> None:
        self.settings = settings
        self.controller = controller
        self.operator_log_buffer = operator_log_buffer

        self._run_sessions = RunSessionManager(settings.collect_root)
        self.telemetry_recorder = JsonlStreamRecorder(self._run_sessions, "gripper_telemetry", record_hz=settings.command_record_hz)
        self.state_recorder = JsonlStreamRecorder(self._run_sessions, "gripper_states", record_hz=settings.state_record_hz)
        self.quest_message_recorder = JsonlStreamRecorder(self._run_sessions, "quest_messages", record_hz=settings.quest_message_record_hz)
        self._running = threading.Event()
        self._loop_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._message_lock = threading.Lock()
        self._latest_message: UnityTeleopMessage | None = None
        self._latest_wall_time: float | None = None
        self._enabled = False
        self._armed = False
        self._command_counter = 0
        self._last_sent_width: float | None = None
        self._last_sent_wall_time: float | None = None
        self._last_status: dict[str, Any] = {}
        self._last_trigger_pressed: bool | None = None
        self._gripper_command_state = 0
        self._stop_latched = False
        self._samples: deque[GripperTestbedSample] = deque(maxlen=settings.status_history_len)
        self._command_events: deque[dict[str, Any]] = deque(maxlen=500)
        self._control_error: str | None = None
        self._last_runtime_error: str | None = None
        self._last_runtime_error_wall_time: float = 0.0
        self._active_run_dir: Path | None = None
        self._active_episode_name: str | None = None
        self._last_state: ControllerState | None = None

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._loop_thread = threading.Thread(target=self._run_loop, name="gripper-testbed-loop", daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)

    def submit_message(self, message: UnityTeleopMessage) -> None:
        wall_time = time.time()
        with self._message_lock:
            self._latest_message = message
            self._latest_wall_time = wall_time
        if self.quest_message_recorder is not None:
            self.quest_message_recorder.record_event(
                {
                    "quest_timestamp": message.timestamp,
                    "source_wall_time": wall_time,
                    "message": message.model_dump(mode="json"),
                },
                event_time=wall_time,
            )

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._armed = False
            self._command_events.append({"event_type": "enabled_changed", "wall_time": time.time(), "enabled": self._enabled})

    def arm(self) -> None:
        with self._lock:
            self._armed = True
            self._command_events.append({"event_type": "armed_changed", "wall_time": time.time(), "armed": True})

    def disarm(self) -> None:
        with self._lock:
            self._armed = False
            self._command_events.append({"event_type": "armed_changed", "wall_time": time.time(), "armed": False})

    def update_runtime_settings(self, update: GripperTestbedRuntimeUpdate) -> dict[str, Any]:
        values = update.model_dump(exclude_none=True)
        with self._lock:
            for key, value in values.items():
                if key in {
                    "min_gripper_width",
                    "max_gripper_width",
                    "gripper_velocity",
                    "force_limit",
                    "open_velocity",
                    "close_velocity",
                    "force_threshold",
                    "trigger_pressed_threshold",
                    "width_gamma",
                    "width_deadband_m",
                    "command_hz",
                }:
                    value = float(value)
                    if key != "min_gripper_width" and value <= 0.0:
                        raise ValueError(f"{key} must be positive")
                    if key == "min_gripper_width" and value < 0.0:
                        raise ValueError("min_gripper_width must be non-negative")
                setattr(self.settings, key, value)
            if self.settings.max_gripper_width < self.settings.min_gripper_width:
                raise ValueError("max_gripper_width must be >= min_gripper_width")
            self.settings.gripper_velocity = self.settings.close_velocity
            self.settings.force_limit = self.settings.force_threshold
            event = {"event_type": "settings_updated", "wall_time": time.time(), "settings": values}
            self._command_events.append(event)
        return self.settings.model_dump(mode="json")

    def start_run(self, run_name: str) -> dict[str, Any]:
        run_dir = self._run_sessions.start_run(run_name, metadata={"mode": "gripper_testbed"}, resume=True)
        self._active_run_dir = run_dir
        return {"run_dir": str(run_dir)}

    def start_live_session(self) -> dict[str, Any]:
        with self._lock:
            self._enabled = False
            self._armed = False
        if self._run_sessions.get_active_episode_dir() is not None:
            self._run_sessions.stop_episode(outcome="replaced")
        self._run_sessions.stop_run(metadata_updates={"replaced_by": "latest_start"})

        run_dir_path = self.settings.collect_root / "latest"
        if run_dir_path.exists():
            if run_dir_path.is_dir():
                shutil.rmtree(run_dir_path)
            else:
                run_dir_path.unlink()
        self._run_sessions = RunSessionManager(self.settings.collect_root)
        self._rebind_recorders_to_current_session()

        run_dir = self._run_sessions.start_run("latest", metadata={"mode": "gripper_testbed", "surface": "quest_trigger", "replacement": True}, resume=False)
        episode_dir = self._run_sessions.start_episode("current", metadata={"mode": "gripper_testbed", "surface": "quest_trigger"})
        with self._lock:
            self._enabled = True
            self._armed = True
            self._control_error = None
            self._samples.clear()
            self._command_events.clear()
            self._command_counter = 0
            self._last_sent_width = None
            self._last_sent_wall_time = None
            self._last_trigger_pressed = None
            self._gripper_command_state = 0
            self._stop_latched = False
            self._active_run_dir = run_dir
            self._active_episode_name = "current"
            self._command_events.append({"event_type": "session_started", "wall_time": time.time(), "run_dir": str(run_dir), "episode_dir": str(episode_dir)})
        try:
            self._send_gripper_motion(command_state=1, trigger_depth=0.0, source="session_start_open")
        except Exception as exc:  # pragma: no cover - hardware/runtime failures
            LOGGER.exception("Failed to open gripper at session start")
            with self._lock:
                self._control_error = str(exc)
        return {"run_dir": str(run_dir), "episode_dir": str(episode_dir), "status": "started"}

    def start_episode(self, episode_name: str | None = None) -> dict[str, Any]:
        episode_dir = self._run_sessions.start_episode(episode_name, metadata={"mode": "gripper_testbed"})
        self._active_episode_name = episode_name
        return {"episode_dir": str(episode_dir)}

    def stop_episode(self, outcome: str = "saved") -> dict[str, Any]:
        episode_dir = self._run_sessions.stop_episode(outcome=outcome)
        self._active_episode_name = None
        return {"episode_dir": None if episode_dir is None else str(episode_dir), "outcome": outcome}

    def stop_live_session(self) -> dict[str, Any]:
        with self._lock:
            self._enabled = False
            self._armed = False
            self._command_events.append({"event_type": "session_stopped", "wall_time": time.time()})
        episode_dir = self._run_sessions.stop_episode(outcome="saved")
        self._run_sessions.stop_run(metadata_updates={"stopped_by": "ui"})
        try:
            self.controller.stop_gripper()
        except Exception as exc:  # pragma: no cover - hardware/runtime failures
            LOGGER.exception("Failed to stop gripper on session stop")
            with self._lock:
                self._control_error = str(exc)
        with self._lock:
            self._active_episode_name = None
        return {"episode_dir": None if episode_dir is None else str(episode_dir), "status": "stopped"}

    def open_gripper_max(self) -> dict[str, Any]:
        return self._send_gripper_motion(command_state=1, trigger_depth=0.0, source="manual_open_max")

    def close_gripper_min(self) -> dict[str, Any]:
        return self._send_gripper_motion(command_state=-1, trigger_depth=1.0, source="manual_close_min")

    def hold_gripper(self) -> dict[str, Any]:
        result = self.controller.stop_gripper()
        with self._lock:
            now = time.time()
            self._gripper_command_state = 0
            self._stop_latched = True
            self._last_sent_wall_time = now
            self._command_events.append({"event_type": "gripper_hold", "wall_time": now, "gripper_command": 0, "result": result})
        return {"status": "stopped", "gripper_command": 0, "result": result}

    def _send_gripper_motion(self, *, command_state: int, trigger_depth: float, source: str) -> dict[str, Any]:
        if command_state not in {-1, 1}:
            raise ValueError("command_state must be -1 or +1 for gripper motion")
        width = float(self.settings.min_gripper_width if command_state < 0 else self.settings.max_gripper_width)
        velocity = float(self.settings.close_velocity if command_state < 0 else self.settings.open_velocity)
        force_threshold = float(self.settings.force_threshold)
        command = GripperTestbedTargetCommand(
            target_width=width,
            velocity=velocity,
            force_limit=force_threshold,
            trigger_depth=trigger_depth,
            sequence=self._command_counter + 1,
            source=source,
        )
        result = self.controller.send_target(command)
        now = time.time()
        with self._lock:
            self._command_counter += 1
            self._last_sent_width = width
            self._last_sent_wall_time = now
            self._gripper_command_state = command_state
            if command_state > 0:
                self._stop_latched = False
            event = {
                "event_type": "gripper_command",
                "wall_time": now,
                "gripper_command": command_state,
                "trigger_depth": trigger_depth,
                "target_width": width,
                "velocity": velocity,
                "force_threshold": force_threshold,
                "result": result,
            }
            self._command_events.append(event)
            if self.telemetry_recorder is not None:
                self.telemetry_recorder.record_event(
                    {
                        **event,
                        "sequence": self._command_counter,
                        "controller_result": result,
                    },
                    event_time=now,
                )
        return {"status": "queued", "gripper_command": command_state, "target_width": width, "result": result}

    def _rebind_recorders_to_current_session(self) -> None:
        for recorder in (self.telemetry_recorder, self.state_recorder, self.quest_message_recorder):
            recorder.session_manager = self._run_sessions
            recorder._last_episode_dir = None
            recorder._last_record_time = None

    def get_state(self) -> ControllerState:
        state = self.controller.get_state()
        with self._lock:
            self._last_state = state
        return state

    def get_status(self) -> dict[str, Any]:
        now = time.time()
        with self._message_lock:
            message = self._latest_message.model_copy(deep=True) if self._latest_message is not None else None
            message_wall_time = self._latest_wall_time
        with self._lock:
            latest_state = self._last_state
            latest_sample = self._samples[-1].__dict__.copy() if self._samples else None
            settings = self.settings
            enabled = self._enabled
            armed = self._armed
            control_error = self._control_error
            gripper_command_state = self._gripper_command_state

            message_age_sec = None if message_wall_time is None else max(0.0, now - message_wall_time)
            trigger_depth = None if message is None else self._trigger_depth(message)
            trigger_pressed = False if trigger_depth is None else self._trigger_pressed(trigger_depth)
            enable_button_pressed = False if message is None else self._button_pressed(message, settings.trigger_enable_button_index)
            deadman_ok = enable_button_pressed or not settings.require_enable_button
            quest_message_fresh = message_age_sec is not None and message_age_sec <= settings.quest_message_timeout_sec
            measured_width = None if latest_state is None else float(latest_state.gripper_width)
            target_width_preview = self._target_width_for_command(gripper_command_state, measured_width)
            width_error_preview = (
                None if target_width_preview is None or measured_width is None else float(target_width_preview - measured_width)
            )
            command_allowed = bool(enabled and armed and deadman_ok and quest_message_fresh and control_error is None)

            blockers: list[str] = []
            if latest_state is None:
                blockers.append("controller state not received")
            if control_error:
                blockers.append(control_error)
            if message is None:
                blockers.append("Quest message not received")
            elif not quest_message_fresh:
                blockers.append(f"Quest message stale: {message_age_sec:.2f}s old")
            if not enabled:
                blockers.append("Enable is off")
            if not armed:
                blockers.append("Arm is off")
            if settings.require_enable_button and not enable_button_pressed:
                blockers.append(f"Quest enable/deadman button {settings.trigger_enable_button_index} is not pressed")

            return {
                "enabled": enabled,
                "armed": armed,
                "command_counter": self._command_counter,
                "last_sent_width": self._last_sent_width,
                "last_sent_wall_time": self._last_sent_wall_time,
                "control_error": control_error,
                "active_run_dir": None if self._active_run_dir is None else str(self._active_run_dir),
                "active_episode_name": self._active_episode_name,
                "latest_state": None if latest_state is None else latest_state.model_dump(mode="json"),
                "latest_sample": latest_sample,
                "latest_quest_message_wall_time": message_wall_time,
                "quest_message_age_sec": message_age_sec,
                "quest_message_fresh": quest_message_fresh,
                "latest_trigger_depth": trigger_depth,
                "trigger_pressed": trigger_pressed,
                "gripper_command": gripper_command_state,
                "gripper_command_label": self._command_label(gripper_command_state),
                "enable_button_pressed": enable_button_pressed,
                "deadman_ok": deadman_ok,
                "command_allowed": command_allowed,
                "command_blockers": blockers,
                "target_width_preview": target_width_preview,
                "measured_width": measured_width,
                "measured_force": None if latest_state is None else float(latest_state.gripper_force),
                "width_error_preview": width_error_preview,
                "settings": settings.model_dump(mode="json"),
            }

    def get_samples(self) -> list[dict[str, Any]]:
        return [sample.__dict__ for sample in list(self._samples)]

    def get_events(self) -> list[dict[str, Any]]:
        return list(self._command_events)

    def get_recent_message(self) -> dict[str, Any] | None:
        with self._message_lock:
            message = self._latest_message
            wall_time = self._latest_wall_time
        if message is None:
            return None
        return {
            "wall_time": wall_time,
            "message": message.model_dump(mode="json"),
        }

    def _run_loop(self) -> None:
        loop_period = 1.0 / max(self.settings.loop_hz, 1e-6)
        last_command_wall_time = 0.0
        while self._running.is_set():
            try:
                command_period = 1.0 / max(self.settings.command_hz, 1e-6)
                state = self.controller.get_state()
                with self._lock:
                    self._last_state = state
                message, message_wall_time = self._latest_message_snapshot()
                if message is None:
                    self._record_sample(
                        state=state,
                        trigger_depth=0.0,
                        trigger_pressed=False,
                        gripper_command=self._gripper_command_state,
                        target_width=state.gripper_width,
                        command_latency_sec=None,
                        command_sequence=None,
                        in_flight=False,
                    )
                    precise_sleep(loop_period)
                    continue
                trigger_depth = self._trigger_depth(message)
                trigger_pressed = self._trigger_pressed(trigger_depth)
                enabled = self._enabled
                armed = self._armed
                enable_pressed = self._button_pressed(message, self.settings.trigger_enable_button_index)
                deadman_ok = enable_pressed or not self.settings.require_enable_button
                message_age_sec = None if message_wall_time is None else time.time() - message_wall_time
                quest_message_fresh = message_age_sec is not None and message_age_sec <= self.settings.quest_message_timeout_sec
                command_allowed = enabled and armed and deadman_ok and quest_message_fresh

                command_state = self._next_command_state(trigger_pressed)
                should_command = (
                    command_allowed
                    and command_state in {-1, 1}
                    and time.time() - last_command_wall_time >= command_period
                )
                if should_command:
                    self._send_gripper_motion(
                        command_state=command_state,
                        trigger_depth=trigger_depth,
                        source="trigger_close" if command_state < 0 else "trigger_open",
                    )
                    last_command_wall_time = time.time()
                latency = None
                if self._last_sent_wall_time is not None:
                    latency = time.time() - self._last_sent_wall_time
                with self._lock:
                    gripper_command = self._gripper_command_state
                    target_width = self._target_width_for_command(gripper_command, float(state.gripper_width))
                self._record_sample(
                    state=state,
                    trigger_depth=trigger_depth,
                    trigger_pressed=trigger_pressed,
                    gripper_command=gripper_command,
                    target_width=target_width,
                    command_latency_sec=latency,
                    command_sequence=self._command_counter if self._last_sent_width is not None else None,
                    in_flight=bool(command_allowed and should_command),
                )
                self._control_error = None
            except Exception as exc:  # pragma: no cover - hardware/runtime failures
                error_text = str(exc)
                should_log = False
                with self._lock:
                    self._control_error = error_text
                    if error_text != self._last_runtime_error or time.time() - self._last_runtime_error_wall_time >= 5.0:
                        self._last_runtime_error = error_text
                        self._last_runtime_error_wall_time = time.time()
                        should_log = True
                if should_log:
                    LOGGER.exception("Gripper testbed iteration failed")
            precise_sleep(loop_period)

    def _latest_message_copy(self) -> UnityTeleopMessage | None:
        with self._message_lock:
            return self._latest_message.model_copy(deep=True) if self._latest_message is not None else None

    def _latest_message_snapshot(self) -> tuple[UnityTeleopMessage | None, float | None]:
        with self._message_lock:
            message = self._latest_message.model_copy(deep=True) if self._latest_message is not None else None
            return message, self._latest_wall_time

    def _record_sample(
        self,
        *,
        state: ControllerState,
        trigger_depth: float,
        trigger_pressed: bool,
        gripper_command: int,
        target_width: float,
        command_latency_sec: float | None,
        command_sequence: int | None,
        in_flight: bool,
    ) -> None:
        sample = GripperTestbedSample(
            wall_time=time.time(),
            trigger_depth=float(trigger_depth),
            trigger_pressed=bool(trigger_pressed),
            gripper_command=int(gripper_command),
            target_width=float(target_width),
            measured_width=float(state.gripper_width),
            measured_force=float(state.gripper_force),
            width_error=float(target_width - float(state.gripper_width)),
            command_latency_sec=command_latency_sec,
            command_sequence=command_sequence,
            in_flight=in_flight,
        )
        self._samples.append(sample)
        self.state_recorder.record_event(sample.__dict__, event_time=sample.wall_time)

    def _next_command_state(self, trigger_pressed: bool) -> int | None:
        with self._lock:
            previous = self._last_trigger_pressed
            self._last_trigger_pressed = trigger_pressed

        if previous is None:
            if trigger_pressed:
                return -1
            return 1 if self._last_sent_width is None else None
        if trigger_pressed and not previous:
            with self._lock:
                self._stop_latched = False
            return -1
        if not trigger_pressed and previous:
            with self._lock:
                self._stop_latched = False
            return 1
        return None

    def _trigger_depth(self, message: UnityTeleopMessage) -> float:
        hand = message.rightHand if self.settings.control_hand == "right" else message.leftHand
        return float(hand.triggerState)

    def _trigger_pressed(self, trigger_depth: float) -> bool:
        return float(trigger_depth) >= float(self.settings.trigger_pressed_threshold)

    def _target_width_for_command(self, command_state: int, measured_width: float | None) -> float:
        if command_state < 0:
            return float(self.settings.min_gripper_width)
        if command_state > 0:
            return float(self.settings.max_gripper_width)
        if self._last_sent_width is not None:
            return float(self._last_sent_width)
        if measured_width is not None:
            return float(measured_width)
        return float(self.settings.max_gripper_width)

    @staticmethod
    def _command_label(command_state: int) -> str:
        if command_state < 0:
            return "close"
        if command_state > 0:
            return "open"
        return "hold"

    @staticmethod
    def _button_pressed_for_hand(message: UnityTeleopMessage, index: int, hand: Literal["left", "right"]) -> bool:
        hand_state = message.rightHand if hand == "right" else message.leftHand
        if index >= len(hand_state.buttonState):
            return False
        return bool(hand_state.buttonState[index])

    def _button_pressed(self, message: UnityTeleopMessage, index: int) -> bool:
        return self._button_pressed_for_hand(message, index, self.settings.control_hand)


def create_gripper_testbed_app(
    service: GripperTestbedService,
    *,
    operator_log_buffer: OperatorLogBuffer | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        try:
            yield
        finally:
            service.stop()

    app = FastAPI(title="VT Dual Franka Gripper Testbed", version="0.1.0", lifespan=lifespan)

    @app.post("/unity")
    async def unity(request: Request):
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid Quest JSON") from exc
        message = parse_unity_teleop_message(payload)
        service.submit_message(message)
        return {"status": "ok"}

    @app.get("/api/v1/status")
    def status():
        return service.get_status()

    @app.get("/api/v1/state")
    def state():
        return service.get_state()

    @app.get("/api/v1/samples")
    def samples():
        return {"samples": service.get_samples()}

    @app.get("/api/v1/events")
    def events():
        return {"events": service.get_events()}

    @app.post("/api/v1/enable")
    def enable(enabled: bool = True):
        service.set_enabled(enabled)
        return {"enabled": enabled}

    @app.post("/api/v1/settings")
    def update_settings(update: GripperTestbedRuntimeUpdate):
        try:
            return service.update_runtime_settings(update)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/test/start")
    def test_start():
        return service.start_live_session()

    @app.post("/api/v1/test/stop")
    def test_stop():
        return service.stop_live_session()

    @app.post("/api/v1/arm")
    def arm():
        service.arm()
        return {"armed": True}

    @app.post("/api/v1/disarm")
    def disarm():
        service.disarm()
        return {"armed": False}

    @app.post("/api/v1/run/start")
    def start_run(run_name: str):
        return service.start_run(run_name)

    @app.post("/api/v1/run/episode/start")
    def start_episode(episode_name: str | None = None):
        return service.start_episode(episode_name)

    @app.post("/api/v1/run/episode/stop")
    def stop_episode(outcome: str = "saved"):
        return service.stop_episode(outcome=outcome)

    @app.post("/api/v1/gripper/open")
    def open_gripper():
        return service.open_gripper_max()

    @app.post("/api/v1/gripper/open-max")
    def open_gripper_max():
        return service.open_gripper_max()

    @app.post("/api/v1/gripper/close-min")
    def close_gripper_min():
        return service.close_gripper_min()

    @app.post("/api/v1/gripper/stop")
    def stop_gripper():
        return service.hold_gripper()

    @app.get("/operator", response_class=HTMLResponse)
    def operator_page() -> str:
        return _OPERATOR_PAGE

    @app.get("/operator/api/status")
    def operator_status():
        return service.get_status()

    @app.get("/operator/api/logs")
    def operator_logs(limit: int = 200):
        entries = []
        if operator_log_buffer is not None:
            entries = operator_log_buffer.get_entries(limit=max(1, min(int(limit), operator_log_buffer.max_records)))
        return {"entries": entries}

    return app


_OPERATOR_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Panda Hand Testbed</title>
  <style>
    :root {
      --bg: #f4f7fb;
      --band: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d9e2ec;
      --good: #167047;
      --good-bg: #e8f6ee;
      --warn: #9a5a1e;
      --warn-bg: #fff4df;
      --bad: #b42318;
      --bad-bg: #feeceb;
      --accent: #2457c5;
      --accent-bg: #e9efff;
      --mono: "IBM Plex Mono", "SFMono-Regular", monospace;
      --ui: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: var(--ui);
    }
    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 18px;
      display: grid;
      gap: 14px;
    }
    .band {
      background: var(--band);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      display: grid;
      gap: 14px;
    }
    .top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      flex-wrap: wrap;
    }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 28px; line-height: 1.1; }
    h2 { font-size: 16px; }
    h3 { font-size: 13px; color: var(--muted); text-transform: uppercase; }
    .mono { font-family: var(--mono); }
    .small { font-size: 12px; color: var(--muted); }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 10px;
      font-size: 13px;
      font-weight: 700;
      background: #fff;
      white-space: nowrap;
    }
    .chip.good { color: var(--good); background: var(--good-bg); border-color: #b7e0c8; }
    .chip.warn { color: var(--warn); background: var(--warn-bg); border-color: #f2d2a0; }
    .chip.bad { color: var(--bad); background: var(--bad-bg); border-color: #f7b5b2; }
    .dashboard {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 14px;
    }
    .stack { display: grid; gap: 14px; align-content: start; }
    .row {
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr);
      align-items: center;
      gap: 10px;
      padding: 9px 0;
      border-bottom: 1px solid #eef2f6;
    }
    .row:last-child { border-bottom: 0; }
    .row .name { color: var(--muted); font-size: 13px; }
    .row .value { font-weight: 700; min-width: 0; overflow-wrap: anywhere; }
    .status-line {
      border-left: 4px solid var(--line);
      padding: 10px 12px;
      background: #f8fafc;
      display: grid;
      gap: 4px;
    }
    .status-line.good { border-left-color: var(--good); background: var(--good-bg); }
    .status-line.warn { border-left-color: var(--warn); background: var(--warn-bg); }
    .status-line.bad { border-left-color: var(--bad); background: var(--bad-bg); }
    .status-title { font-weight: 800; }
    .blockers { margin: 0; padding-left: 18px; color: var(--bad); font-size: 13px; }
    .actions { display: flex; gap: 8px; flex-wrap: wrap; }
    button, input, select { font: inherit; }
    button {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 9px 12px;
      cursor: pointer;
      font-weight: 750;
    }
    button.primary { color: #fff; background: var(--accent); border-color: var(--accent); }
    button.good { color: #fff; background: var(--good); border-color: var(--good); }
    button.warn { color: #fff; background: var(--warn); border-color: var(--warn); }
    button.bad { color: #fff; background: var(--bad); border-color: var(--bad); }
    button:disabled { opacity: 0.5; cursor: not-allowed; }
    label {
      display: grid;
      gap: 5px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    input[type="number"], input[type="text"], select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 9px;
      background: #fff;
      color: var(--ink);
    }
    input[type="range"] { width: 100%; }
    .settings {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
    }
    .readouts {
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
    }
    .readout {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: #fff;
      min-width: 0;
    }
    .readout .label { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .readout .number { font-size: 22px; font-weight: 850; line-height: 1.1; overflow-wrap: anywhere; }
    .chart-wrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: #fff;
    }
    canvas {
      width: 100%;
      height: 260px;
      display: block;
    }
    .legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; font-size: 12px; color: var(--muted); }
    .swatch { display: inline-block; width: 12px; height: 3px; vertical-align: middle; margin-right: 5px; }
    .events {
      max-height: 260px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
    }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #eef2f6; text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: #f8fafc; color: var(--muted); }
    details { border-top: 1px solid #eef2f6; padding-top: 10px; }
    summary { cursor: pointer; font-weight: 800; color: var(--muted); }
    @media (max-width: 980px) {
      .dashboard { grid-template-columns: 1fr; }
      .settings, .readouts { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .row { grid-template-columns: 1fr; gap: 4px; }
    }
    @media (max-width: 620px) {
      .settings, .readouts { grid-template-columns: 1fr; }
      canvas { height: 220px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="band">
      <div class="top">
        <div>
          <h1>Panda Hand Trigger Testbed</h1>
          <div class="small mono" id="quest-url">Quest endpoint: /unity</div>
        </div>
        <div class="chips">
          <div class="chip warn" id="chip-controller">controller: waiting</div>
          <div class="chip warn" id="chip-quest">quest: waiting</div>
          <div class="chip bad" id="chip-command">command: blocked</div>
          <div class="chip" id="chip-recording">recording: off</div>
        </div>
      </div>
      <div id="gate" class="status-line warn">
        <div class="status-title">Waiting for status</div>
        <div class="small">Refresh is running at 5 Hz.</div>
      </div>
    </section>

    <section class="dashboard">
      <div class="stack">
        <section class="band">
          <h2>Run Checklist</h2>
          <div id="checklist"></div>
        </section>

        <section class="band">
          <h2>Control</h2>
          <div class="actions">
            <button class="primary" onclick="apiPost('/api/v1/enable?enabled=true')">Enable</button>
            <button onclick="apiPost('/api/v1/enable?enabled=false')">Disable</button>
            <button class="good" onclick="apiPost('/api/v1/arm')">Arm</button>
            <button onclick="apiPost('/api/v1/disarm')">Disarm</button>
          </div>
          <div class="actions">
            <button onclick="apiPost('/api/v1/gripper/open')">Open Hand</button>
            <button class="warn" onclick="apiPost('/api/v1/gripper/stop')">Stop Commands</button>
          </div>
          <div class="settings">
            <label>Force limit N
              <select id="force-limit">
                <option>3</option><option selected>5</option><option>7</option><option>10</option><option>15</option>
              </select>
            </label>
            <label>Velocity m/s
              <select id="velocity">
                <option>0.02</option><option selected>0.05</option><option>0.08</option><option>0.10</option>
              </select>
            </label>
            <label>Command Hz
              <select id="command-hz">
                <option>10</option><option selected>20</option><option>30</option>
              </select>
            </label>
          </div>
          <details>
            <summary>Advanced mapping</summary>
            <div class="settings" style="margin-top: 10px;">
              <label>Min width m
                <input id="min-width" type="number" min="0" max="0.078" step="0.001" value="0.000">
              </label>
              <label>Max width m
                <input id="max-width" type="number" min="0" max="0.09" step="0.001" value="0.078">
              </label>
              <label>Gamma
                <input id="gamma" type="range" min="0.8" max="3.0" step="0.1" value="1.5">
              </label>
            </div>
            <div class="small mono" id="gamma-val">gamma 1.5</div>
          </details>
        </section>

        <section class="band">
          <h2>Recording</h2>
          <label>Run name
            <input id="run-name" type="text" value="gripper_testbed_run">
          </label>
          <div class="actions">
            <button onclick="apiPost('/api/v1/run/start?run_name=' + encodeURIComponent(document.getElementById('run-name').value))">Start Run</button>
            <button onclick="apiPost('/api/v1/run/episode/start')">Start Episode</button>
            <button onclick="apiPost('/api/v1/run/episode/stop?outcome=saved')">Stop Episode</button>
          </div>
          <div class="small mono" id="record-path">run: none</div>
        </section>
      </div>

      <div class="stack">
        <section class="band">
          <h2>Live Readout</h2>
          <div class="readouts">
            <div class="readout"><div class="label">Trigger</div><div class="number mono" id="trigger">n/a</div></div>
            <div class="readout"><div class="label">Target width m</div><div class="number mono" id="target-width">n/a</div></div>
            <div class="readout"><div class="label">Measured width m</div><div class="number mono" id="measured-width">n/a</div></div>
            <div class="readout"><div class="label">Width error m</div><div class="number mono" id="width-error">n/a</div></div>
            <div class="readout"><div class="label">Force N</div><div class="number mono" id="force">n/a</div></div>
          </div>
        </section>

        <section class="band">
          <div class="top">
            <h2>Trigger to Width</h2>
            <div class="small mono" id="sample-age">samples: 0</div>
          </div>
          <div class="chart-wrap">
            <canvas id="chart"></canvas>
            <div class="legend">
              <span><span class="swatch" style="background:#2457c5"></span>trigger</span>
              <span><span class="swatch" style="background:#167047"></span>target width</span>
              <span><span class="swatch" style="background:#9a5a1e"></span>measured width</span>
              <span><span class="swatch" style="background:#b42318"></span>width error</span>
            </div>
          </div>
        </section>

        <section class="band">
          <h2>Recent Events</h2>
          <div class="events">
            <table>
              <thead><tr><th>Time</th><th>Event</th><th>Details</th></tr></thead>
              <tbody id="events"></tbody>
            </table>
          </div>
        </section>
      </div>
    </section>
  </div>
  <script>
    const chart = document.getElementById('chart');
    const eventsTbody = document.getElementById('events');
    const maxSamples = 240;
    let samples = [];

    function fmt(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
      return Number(value).toFixed(digits);
    }

    function cssStatus(ok, warn = false) {
      if (ok) return 'good';
      return warn ? 'warn' : 'bad';
    }

    async function apiPost(path) {
      await fetch(path, {method: 'POST'});
      await refresh();
    }

    async function syncSettings() {
      const payload = {
        force_limit: Number(document.getElementById('force-limit').value),
        gripper_velocity: Number(document.getElementById('velocity').value),
        command_hz: Number(document.getElementById('command-hz').value),
        min_gripper_width: Number(document.getElementById('min-width').value),
        max_gripper_width: Number(document.getElementById('max-width').value),
        width_gamma: Number(document.getElementById('gamma').value),
      };
      await fetch('/api/v1/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      });
      await refresh();
    }

    ['force-limit', 'velocity', 'command-hz', 'min-width', 'max-width', 'gamma'].forEach(id => {
      document.getElementById(id).addEventListener('change', syncSettings);
    });
    document.getElementById('gamma').addEventListener('input', (event) => {
      document.getElementById('gamma-val').textContent = `gamma ${Number(event.target.value).toFixed(1)}`;
    });

    function setChip(id, text, status) {
      const node = document.getElementById(id);
      node.textContent = text;
      node.className = `chip ${status}`;
    }

    function updateSettingsInputs(settings) {
      if (!settings) return;
      document.getElementById('force-limit').value = String(settings.force_limit);
      document.getElementById('velocity').value = String(settings.gripper_velocity);
      document.getElementById('command-hz').value = String(settings.command_hz);
      document.getElementById('min-width').value = Number(settings.min_gripper_width).toFixed(3);
      document.getElementById('max-width').value = Number(settings.max_gripper_width).toFixed(3);
      document.getElementById('gamma').value = String(settings.width_gamma);
      document.getElementById('gamma-val').textContent = `gamma ${Number(settings.width_gamma).toFixed(1)}`;
    }

    function updateStatus(data) {
      const controllerOk = Boolean(data.latest_state) && !data.control_error;
      const questOk = Boolean(data.quest_message_fresh);
      const commandOk = Boolean(data.command_allowed);
      setChip('chip-controller', controllerOk ? `controller: ${data.latest_state?.backend || 'ok'}` : 'controller: error', cssStatus(controllerOk));
      setChip('chip-quest', questOk ? `quest: live ${fmt(data.quest_message_age_sec, 2)}s` : 'quest: waiting', cssStatus(questOk, !data.latest_quest_message_wall_time));
      setChip('chip-command', commandOk ? 'command: ready' : 'command: blocked', cssStatus(commandOk));
      setChip('chip-recording', data.active_episode_name ? 'recording: episode' : 'recording: off', data.active_episode_name ? 'good' : '');

      const gate = document.getElementById('gate');
      const blockers = data.command_blockers || [];
      gate.className = `status-line ${commandOk ? 'good' : (blockers.length ? 'bad' : 'warn')}`;
      gate.innerHTML = commandOk
        ? '<div class="status-title">Ready: trigger changes will command the Panda Hand.</div><div class="small">Squeeze slowly and watch target width versus measured width.</div>'
        : `<div class="status-title">Blocked: trigger is not commanding the Panda Hand.</div>${blockers.length ? `<ul class="blockers">${blockers.map(item => `<li>${item}</li>`).join('')}</ul>` : '<div class="small">Waiting for fresh status.</div>'}`;

      const checklist = [
        ['Controller state', controllerOk ? `OK, width ${fmt(data.measured_width, 4)} m` : (data.control_error || 'not received'), controllerOk],
        ['Quest stream', questOk ? `OK, trigger ${fmt(data.latest_trigger_depth, 3)}` : 'not fresh at /unity', questOk],
        ['Enable', data.enabled ? 'on' : 'off', Boolean(data.enabled)],
        ['Arm', data.armed ? 'on' : 'off', Boolean(data.armed)],
        ['Deadman button', data.deadman_ok ? 'OK' : `hold button ${data.settings?.trigger_enable_button_index}`, Boolean(data.deadman_ok)],
        ['Command gate', commandOk ? 'ready' : 'blocked', commandOk],
      ];
      document.getElementById('checklist').innerHTML = checklist.map(([name, value, ok]) => `
        <div class="row">
          <div class="name">${name}</div>
          <div class="value ${ok ? 'status-good' : 'status-bad'}">${value}</div>
        </div>
      `).join('');

      document.getElementById('trigger').textContent = fmt(data.latest_trigger_depth, 3);
      document.getElementById('target-width').textContent = fmt(data.target_width_preview, 4);
      document.getElementById('measured-width').textContent = fmt(data.measured_width, 4);
      document.getElementById('width-error').textContent = fmt(data.width_error_preview, 4);
      document.getElementById('force').textContent = fmt(data.measured_force, 2);
      document.getElementById('record-path').textContent = `run: ${data.active_run_dir || 'none'}`;
      updateSettingsInputs(data.settings);
    }

    function drawChart(rows) {
      const dpr = window.devicePixelRatio || 1;
      const width = chart.clientWidth;
      const height = chart.clientHeight;
      chart.width = width * dpr;
      chart.height = height * dpr;
      const ctx = chart.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#eef2f6';
      ctx.lineWidth = 1;
      for (let i = 0; i < 5; i++) {
        const y = 18 + (height - 42) * i / 4;
        ctx.beginPath();
        ctx.moveTo(38, y);
        ctx.lineTo(width - 10, y);
        ctx.stroke();
      }
      if (!rows.length) return;
      const points = rows.map(row => ({
        t: row.wall_time,
        trigger: row.trigger_depth,
        target: row.target_width,
        measured: row.measured_width,
        error: row.width_error,
      }));
      const t0 = points[0].t;
      const t1 = points[points.length - 1].t || (t0 + 1);
      const values = points.flatMap(p => [p.trigger, p.target, p.measured, p.error]).filter(v => Number.isFinite(Number(v)));
      const minY = Math.min(0, ...values);
      const maxY = Math.max(1, ...values);
      const spanY = Math.max(maxY - minY, 1e-6);
      const xy = (t, value) => [
        38 + (t - t0) / Math.max(t1 - t0, 1e-6) * (width - 50),
        18 + (1 - (value - minY) / spanY) * (height - 42),
      ];
      [
        ['trigger', '#2457c5'],
        ['target', '#167047'],
        ['measured', '#9a5a1e'],
        ['error', '#b42318'],
      ].forEach(([key, color]) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = key === 'error' ? 1.5 : 2;
        ctx.beginPath();
        let started = false;
        points.forEach(point => {
          const value = point[key];
          if (!Number.isFinite(Number(value))) return;
          const [x, y] = xy(point.t, value);
          if (!started) {
            ctx.moveTo(x, y);
            started = true;
          } else {
            ctx.lineTo(x, y);
          }
        });
        ctx.stroke();
      });
    }

    function updateEvents(rows) {
      eventsTbody.innerHTML = rows.slice(-18).reverse().map(event => `
        <tr>
          <td class="mono">${new Date((event.wall_time || 0) * 1000).toLocaleTimeString()}</td>
          <td>${event.event_type || event.type || 'event'}</td>
          <td class="mono">${JSON.stringify(event).slice(0, 180)}</td>
        </tr>
      `).join('');
    }

    async function fetchJson(path) {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`${path} returned ${response.status}`);
      return response.json();
    }

    async function refresh() {
      try {
        const [status, sampleData, eventData] = await Promise.all([
          fetchJson('/api/v1/status'),
          fetchJson('/api/v1/samples'),
          fetchJson('/api/v1/events'),
        ]);
        updateStatus(status);
        samples = (sampleData.samples || []).slice(-maxSamples);
        drawChart(samples);
        updateEvents(eventData.events || []);
        document.getElementById('sample-age').textContent = `samples: ${samples.length}`;
      } catch (error) {
        setChip('chip-controller', 'workspace: error', 'bad');
        const gate = document.getElementById('gate');
        gate.className = 'status-line bad';
        gate.innerHTML = `<div class="status-title">Workspace API refresh failed.</div><div class="small mono">${error.message}</div>`;
      }
    }

    document.getElementById('quest-url').textContent = `Quest endpoint: ${window.location.origin}/unity`;
    refresh();
    setInterval(refresh, 200);
  </script>
</body>
</html>
"""

_OPERATOR_PAGE = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Panda Hand Trigger Test</title>
  <style>
    :root {
      --bg: #f5f7fa;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #667085;
      --line: #d8dee8;
      --good: #157347;
      --bad: #b42318;
      --warn: #9a5a1e;
      --accent: #2457c5;
      --mono: "IBM Plex Mono", "SFMono-Regular", monospace;
      --ui: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--ui); }
    main { max-width: 1180px; margin: 0 auto; padding: 20px; display: grid; gap: 14px; }
    section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 16px; }
    h1, h2 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 28px; }
    h2 { font-size: 16px; margin-bottom: 10px; }
    .muted { color: var(--muted); }
    .mono { font-family: var(--mono); }
    .top { display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; flex-wrap: wrap; }
    .chips { display: flex; gap: 8px; flex-wrap: wrap; }
    .chip { border: 1px solid var(--line); border-radius: 999px; padding: 7px 10px; font-size: 13px; font-weight: 800; background: #fff; }
    .chip.good { color: var(--good); background: #e7f5ed; border-color: #b9dfc9; }
    .chip.bad { color: var(--bad); background: #feeceb; border-color: #f3bab6; }
    .chip.warn { color: var(--warn); background: #fff4df; border-color: #efd19b; }
    .gate { border-left: 4px solid var(--warn); background: #fff8e8; padding: 11px 12px; display: grid; gap: 6px; }
    .gate.good { border-left-color: var(--good); background: #e7f5ed; }
    .gate.bad { border-left-color: var(--bad); background: #feeceb; }
    .grid { display: grid; grid-template-columns: 390px minmax(0, 1fr); gap: 14px; align-items: start; }
    .actions { display: grid; gap: 10px; }
    .action { display: grid; grid-template-columns: 130px minmax(0, 1fr); gap: 10px; align-items: center; border-bottom: 1px solid #edf1f6; padding-bottom: 10px; }
    .action:last-child { border-bottom: 0; padding-bottom: 0; }
    button { border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; font: inherit; font-weight: 850; cursor: pointer; background: #fff; }
    button.primary { color: #fff; background: var(--accent); border-color: var(--accent); }
    button.bad { color: #fff; background: var(--bad); border-color: var(--bad); }
    button.good { color: #fff; background: var(--good); border-color: var(--good); }
    button.warn { color: #fff; background: var(--warn); border-color: var(--warn); }
    .desc { color: var(--muted); font-size: 13px; line-height: 1.45; }
    .settings { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 12px; }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; font-weight: 800; }
    select { border: 1px solid var(--line); border-radius: 8px; padding: 9px; font: inherit; background: #fff; }
    .readouts { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .readout { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #fff; min-width: 0; }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 6px; }
    .value { font: 850 22px/1.1 var(--mono); overflow-wrap: anywhere; }
    .steps { margin: 0; padding-left: 20px; color: var(--ink); line-height: 1.55; }
    .steps code { background: #eef2f7; padding: 1px 4px; border-radius: 4px; }
    canvas { width: 100%; height: 250px; display: block; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    .legend { display: flex; gap: 14px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: var(--muted); }
    .swatch { display: inline-block; width: 14px; height: 3px; margin-right: 5px; vertical-align: middle; }
    .path { margin-top: 8px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    @media (max-width: 900px) { .grid, .readouts { grid-template-columns: 1fr; } .action { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <section>
      <div class="top">
        <div>
          <h1>Panda Hand Trigger Test</h1>
          <div class="muted">只测试 Franka Panda Hand，不动机械臂。Start 后 trigger 闭合，按 B 停住，松开 trigger 张开。</div>
        </div>
        <div class="chips">
          <div id="controller-chip" class="chip warn">controller: waiting</div>
          <div id="quest-chip" class="chip warn">quest: waiting</div>
          <div id="command-chip" class="chip bad">command: stopped</div>
        </div>
      </div>
      <div id="gate" class="gate bad" style="margin-top: 12px;">
        <strong>还没有开始。</strong>
        <span class="muted">确认 Quest 数据进来后，按 Start 开始测试。</span>
      </div>
    </section>

    <section>
      <h2>Meta Quest 连接方式</h2>
      <ol class="steps">
        <li>Quest app 里的 workstation IP 填 <code>127.0.0.1</code>。</li>
        <li>在 workstation 上运行：<code id="reverse-cmd">adb reverse tcp:8082 tcp:PORT</code></li>
        <li>页面里的 Quest 状态变成 live 后，按 <strong>Start</strong>。</li>
        <li>trigger 从松开变成按下 = 以 close velocity 闭合；键盘 <strong>B</strong> = 停住；trigger 松开 = 以 open velocity 张开。</li>
      </ol>
    </section>

    <section>
      <h2>Step-by-step Procedure</h2>
      <ol class="steps">
        <li>确认 controller chip 显示 <strong>controller: ok</strong>，Quest chip 显示 live。</li>
        <li>在参数区选择 <strong>Close velocity</strong>、<strong>Open velocity</strong>、<strong>Force threshold</strong> 和 control hand。</li>
        <li>先点 <strong>Open Max</strong>，确认夹爪可以张开且周围没有障碍物。</li>
        <li>点 <strong>Start</strong> 开始记录；Start 会先自动发送一次打开命令。</li>
        <li>按下 Quest trigger：夹爪以 close velocity 开始闭合，页面 command 显示 <code>-1 close</code>。</li>
        <li>到达想要的接触/宽度时按键盘 <strong>B</strong>：夹爪 stop，页面 command 显示 <code>0 hold</code>。</li>
        <li>松开 trigger：夹爪以 open velocity 张开，页面 command 显示 <code>+1 open</code>。</li>
        <li>实验结束点 <strong>Stop</strong>，记录会保存到 <code>data/gripper_testbed/latest</code>。</li>
      </ol>
    </section>

    <div class="grid">
      <section>
        <h2>四个按钮</h2>
        <div class="actions">
          <div class="action">
            <button class="good" onclick="postAction('/api/v1/gripper/open-max')">Open Max</button>
            <div class="desc">手动把夹爪张到最大宽度。建议 Start 前先按一次，确认夹爪能动。</div>
          </div>
          <div class="action">
            <button class="warn" onclick="postAction('/api/v1/gripper/close-min')">Close Min</button>
            <div class="desc">手动把夹爪闭到最小宽度。不要夹易碎物体时直接按这个。</div>
          </div>
          <div class="action">
            <button class="warn" onclick="postAction('/api/v1/gripper/stop')">Hold / B</button>
            <div class="desc">中断当前夹爪运动并保持当前位置；键盘按 B 等价于这个按钮。</div>
          </div>
          <div class="action">
            <button class="primary" onclick="postAction('/api/v1/test/start')">Start</button>
            <div class="desc">开始测试并记录到 <span class="mono">data/gripper_testbed/latest</span>。每次 Start 会替换上一段记录。</div>
          </div>
          <div class="action">
            <button class="bad" onclick="postAction('/api/v1/test/stop')">Stop</button>
            <div class="desc">停止 Quest trigger 控制，并保存当前这段记录。Stop 后 trigger 不再发夹爪命令。</div>
          </div>
        </div>
        <div class="settings">
          <label>Force threshold N
            <select id="force-limit">
              <option>3</option><option>5</option><option selected>7</option><option>10</option><option>15</option>
            </select>
          </label>
          <label>Close velocity m/s
            <select id="close-velocity">
              <option selected>0.02</option><option>0.03</option><option>0.05</option><option>0.08</option>
            </select>
          </label>
          <label>Open velocity m/s
            <select id="open-velocity">
              <option>0.02</option><option selected>0.05</option><option>0.08</option><option>0.10</option>
            </select>
          </label>
          <label>Control hand
            <select id="control-hand">
              <option>left</option><option selected>right</option>
            </select>
          </label>
        </div>
      </section>

      <section>
        <h2>现在发生了什么</h2>
        <div class="readouts">
          <div class="readout"><div class="label">Trigger depth</div><div id="trigger" class="value">n/a</div></div>
          <div class="readout"><div class="label">Command -1/0/+1</div><div id="command-state" class="value">n/a</div></div>
          <div class="readout"><div class="label">Target width m</div><div id="target" class="value">n/a</div></div>
          <div class="readout"><div class="label">Measured width m</div><div id="measured" class="value">n/a</div></div>
          <div class="readout"><div class="label">Width error m</div><div id="error" class="value">n/a</div></div>
        </div>
        <div class="path mono" id="record-path">record: none</div>
      </section>
    </div>

    <section>
      <h2>实时曲线</h2>
      <canvas id="chart"></canvas>
      <div class="legend">
        <span><span class="swatch" style="background:#2457c5"></span>trigger</span>
        <span><span class="swatch" style="background:#b42318"></span>command (-1/0/+1)</span>
        <span><span class="swatch" style="background:#157347"></span>target width</span>
        <span><span class="swatch" style="background:#9a5a1e"></span>measured width</span>
      </div>
    </section>
  </main>

  <script>
    const chart = document.getElementById('chart');
    const maxSamples = 240;

    function fmt(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
      return Number(value).toFixed(digits);
    }

    function setChip(id, text, state) {
      const node = document.getElementById(id);
      node.textContent = text;
      node.className = `chip ${state}`;
    }

    async function postAction(path) {
      await syncSettings();
      const response = await fetch(path, {method: 'POST'});
      if (!response.ok) {
        const text = await response.text();
        alert(`${path} failed: ${text}`);
      }
      await refresh();
    }

    async function syncSettings() {
      await fetch('/api/v1/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
          force_threshold: Number(document.getElementById('force-limit').value),
          close_velocity: Number(document.getElementById('close-velocity').value),
          open_velocity: Number(document.getElementById('open-velocity').value),
          control_hand: document.getElementById('control-hand').value,
        }),
      });
    }

    document.addEventListener('keydown', (event) => {
      if (event.key && event.key.toLowerCase() === 'b' && !event.repeat) {
        event.preventDefault();
        postAction('/api/v1/gripper/stop');
      }
    });

    function updateStatus(status) {
      const controllerOk = Boolean(status.latest_state) && !status.control_error;
      const questOk = Boolean(status.quest_message_fresh);
      const running = Boolean(status.enabled && status.armed);
      const commandOk = Boolean(status.command_allowed);

      setChip('controller-chip', controllerOk ? 'controller: ok' : 'controller: error', controllerOk ? 'good' : 'bad');
      setChip('quest-chip', questOk ? `quest: live ${fmt(status.quest_message_age_sec, 2)}s` : 'quest: no message', questOk ? 'good' : 'warn');
      setChip('command-chip', commandOk ? 'command: active' : (running ? 'command: blocked' : 'command: stopped'), commandOk ? 'good' : (running ? 'bad' : 'warn'));

      const gate = document.getElementById('gate');
      if (commandOk) {
        gate.className = 'gate good';
        gate.innerHTML = '<strong>正在运行。</strong><span>trigger 按下闭合，键盘 B 停住，trigger 松开张开。</span>';
      } else {
        const blockers = status.command_blockers || [];
        gate.className = `gate ${running ? 'bad' : 'bad'}`;
        gate.innerHTML = running
          ? `<strong>Start 已按，但当前不能发命令。</strong><span class="muted">${blockers.join(' | ') || 'waiting'}</span>`
          : '<strong>已停止。</strong><span class="muted">按 Start 后才会让 Quest trigger 控制夹爪。</span>';
      }

      document.getElementById('trigger').textContent = fmt(status.latest_trigger_depth, 3);
      document.getElementById('command-state').textContent = `${status.gripper_command ?? 0} ${status.gripper_command_label || ''}`;
      document.getElementById('target').textContent = fmt(status.target_width_preview, 4);
      document.getElementById('measured').textContent = fmt(status.measured_width, 4);
      document.getElementById('error').textContent = fmt(status.width_error_preview, 4);
      document.getElementById('record-path').textContent = `record: ${status.active_run_dir || 'none'}`;

      if (status.settings) {
        document.getElementById('force-limit').value = String(status.settings.force_threshold);
        document.getElementById('close-velocity').value = String(status.settings.close_velocity);
        document.getElementById('open-velocity').value = String(status.settings.open_velocity);
        document.getElementById('control-hand').value = String(status.settings.control_hand || 'right');
      }
    }

    function drawChart(samples) {
      const dpr = window.devicePixelRatio || 1;
      const width = chart.clientWidth;
      const height = chart.clientHeight;
      chart.width = width * dpr;
      chart.height = height * dpr;
      const ctx = chart.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = '#edf1f6';
      for (let i = 0; i < 5; i += 1) {
        const y = 18 + (height - 40) * i / 4;
        ctx.beginPath();
        ctx.moveTo(36, y);
        ctx.lineTo(width - 10, y);
        ctx.stroke();
      }
      if (!samples.length) return;
      const points = samples.map(row => ({
        t: row.wall_time,
        trigger: row.trigger_depth,
        command: row.gripper_command,
        target: row.target_width,
        measured: row.measured_width,
      }));
      const t0 = points[0].t;
      const t1 = points[points.length - 1].t || (t0 + 1);
      const minY = 0;
      const maxY = 1;
      const xy = (t, value) => [
        36 + (t - t0) / Math.max(t1 - t0, 1e-6) * (width - 48),
        18 + (1 - (value - minY) / (maxY - minY)) * (height - 40),
      ];
      [['trigger', '#2457c5'], ['command', '#b42318'], ['target', '#157347'], ['measured', '#9a5a1e']].forEach(([key, color]) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        let started = false;
        points.forEach(point => {
          let value = Number(point[key]);
          if (!Number.isFinite(value)) return;
          if (key === 'command') value = (value + 1) / 2;
          else if (key !== 'trigger') value = value / 0.078;
          const [x, y] = xy(point.t, value);
          if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
        });
        ctx.stroke();
      });
    }

    async function fetchJson(path) {
      const response = await fetch(path);
      if (!response.ok) throw new Error(`${path}: ${response.status}`);
      return response.json();
    }

    async function refresh() {
      try {
        const [status, sampleData] = await Promise.all([
          fetchJson('/api/v1/status'),
          fetchJson('/api/v1/samples'),
        ]);
        updateStatus(status);
        drawChart((sampleData.samples || []).slice(-maxSamples));
      } catch (error) {
        setChip('controller-chip', 'workspace: error', 'bad');
        document.getElementById('gate').className = 'gate bad';
        document.getElementById('gate').innerHTML = `<strong>页面无法读取 testbed 状态。</strong><span class="mono">${error.message}</span>`;
      }
    }

    const localPort = window.location.port || (window.location.protocol === 'https:' ? '443' : '80');
    document.getElementById('reverse-cmd').textContent = `adb reverse --remove tcp:8082; adb reverse tcp:8082 tcp:${localPort}`;
    refresh();
    setInterval(refresh, 200);
  </script>
</body>
</html>
"""
