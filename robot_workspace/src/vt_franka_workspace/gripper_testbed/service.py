from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse

from vt_franka_shared.models import ControllerState, GripperTestbedTargetCommand, UnityTeleopMessage, parse_unity_teleop_message
from vt_franka_shared.timing import precise_sleep

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
    trigger_enable_button_index: int = 4
    min_gripper_width: float = 0.0
    max_gripper_width: float = 0.078
    gripper_velocity: float = 0.1
    force_limit: float = 7.0
    width_gamma: float = 1.5
    width_deadband_m: float = 0.0015
    command_record_hz: float = 0.0
    quest_message_record_hz: float = 0.0
    state_record_hz: float = 30.0
    status_history_len: int = 240
    collect_root: Path = Path("./data/gripper_testbed")
    require_enable_button: bool = True

    @field_validator(
        "loop_hz",
        "command_hz",
        "controller_request_timeout_sec",
        "gripper_velocity",
        "force_limit",
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
        self._samples: deque[GripperTestbedSample] = deque(maxlen=settings.status_history_len)
        self._command_events: deque[dict[str, Any]] = deque(maxlen=500)
        self._control_error: str | None = None
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
                if key in {"min_gripper_width", "max_gripper_width", "gripper_velocity", "force_limit", "width_gamma", "width_deadband_m", "command_hz"}:
                    value = float(value)
                    if key != "min_gripper_width" and value <= 0.0:
                        raise ValueError(f"{key} must be positive")
                    if key == "min_gripper_width" and value < 0.0:
                        raise ValueError("min_gripper_width must be non-negative")
                setattr(self.settings, key, value)
            if self.settings.max_gripper_width < self.settings.min_gripper_width:
                raise ValueError("max_gripper_width must be >= min_gripper_width")
            event = {"event_type": "settings_updated", "wall_time": time.time(), "settings": values}
            self._command_events.append(event)
        return self.settings.model_dump(mode="json")

    def start_run(self, run_name: str) -> dict[str, Any]:
        run_dir = self._run_sessions.start_run(run_name, metadata={"mode": "gripper_testbed"}, resume=True)
        self._active_run_dir = run_dir
        return {"run_dir": str(run_dir)}

    def start_episode(self, episode_name: str | None = None) -> dict[str, Any]:
        episode_dir = self._run_sessions.start_episode(episode_name, metadata={"mode": "gripper_testbed"})
        self._active_episode_name = episode_name
        return {"episode_dir": str(episode_dir)}

    def stop_episode(self, outcome: str = "saved") -> dict[str, Any]:
        episode_dir = self._run_sessions.stop_episode(outcome=outcome)
        self._active_episode_name = None
        return {"episode_dir": None if episode_dir is None else str(episode_dir), "outcome": outcome}

    def get_state(self) -> ControllerState:
        state = self.controller.get_state()
        with self._lock:
            self._last_state = state
        return state

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "armed": self._armed,
                "command_counter": self._command_counter,
                "last_sent_width": self._last_sent_width,
                "last_sent_wall_time": self._last_sent_wall_time,
                "control_error": self._control_error,
                "active_run_dir": None if self._active_run_dir is None else str(self._active_run_dir),
                "active_episode_name": self._active_episode_name,
                "latest_state": None if self._last_state is None else self._last_state.model_dump(mode="json"),
                "settings": self.settings.model_dump(mode="json"),
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
                message = self._latest_message_copy()
                if message is None:
                    self._record_sample(state=state, trigger_depth=0.0, target_width=state.gripper_width, command_latency_sec=None, command_sequence=None, in_flight=False)
                    precise_sleep(loop_period)
                    continue
                trigger_depth = float(message.leftHand.triggerState)
                enabled = self._enabled
                armed = self._armed
                enable_pressed = self._button_pressed(message, self.settings.trigger_enable_button_index)
                deadman_ok = enable_pressed or not self.settings.require_enable_button
                command_allowed = enabled and armed and deadman_ok

                target_width = map_trigger_to_width(
                    trigger_depth,
                    min_width=self.settings.min_gripper_width,
                    max_width=self.settings.max_gripper_width,
                    gamma=self.settings.width_gamma,
                )
                width_changed = self._last_sent_width is None or abs(target_width - self._last_sent_width) >= self.settings.width_deadband_m
                should_command = command_allowed and width_changed and (time.time() - last_command_wall_time >= command_period)
                if should_command:
                    command = GripperTestbedTargetCommand(
                        target_width=target_width,
                        velocity=self.settings.gripper_velocity,
                        force_limit=self.settings.force_limit,
                        trigger_depth=trigger_depth,
                        sequence=self._command_counter + 1,
                    )
                    result = self.controller.send_target(command)
                    self._command_counter += 1
                    last_command_wall_time = time.time()
                    self._last_sent_width = target_width
                    self._last_sent_wall_time = last_command_wall_time
                    self._command_events.append(
                        {
                            "event_type": "gripper_target",
                            "wall_time": last_command_wall_time,
                            "trigger_depth": trigger_depth,
                            "target_width": target_width,
                            "force_limit": self.settings.force_limit,
                            "result": result,
                        }
                    )
                    if self.telemetry_recorder is not None:
                        self.telemetry_recorder.record_event(
                            {
                                "event_type": "gripper_target",
                                "wall_time": last_command_wall_time,
                                "trigger_depth": trigger_depth,
                                "target_width": target_width,
                                "force_limit": self.settings.force_limit,
                                "sequence": self._command_counter,
                                "controller_result": result,
                            },
                            event_time=last_command_wall_time,
                        )
                latency = None
                if self._last_sent_wall_time is not None:
                    latency = time.time() - self._last_sent_wall_time
                self._record_sample(
                    state=state,
                    trigger_depth=trigger_depth,
                    target_width=target_width,
                    command_latency_sec=latency,
                    command_sequence=self._command_counter if self._last_sent_width is not None else None,
                    in_flight=bool(command_allowed and should_command),
                )
                self._control_error = None
            except Exception as exc:  # pragma: no cover - hardware/runtime failures
                LOGGER.exception("Gripper testbed iteration failed")
                with self._lock:
                    self._control_error = str(exc)
            precise_sleep(loop_period)

    def _latest_message_copy(self) -> UnityTeleopMessage | None:
        with self._message_lock:
            return self._latest_message.model_copy(deep=True) if self._latest_message is not None else None

    def _record_sample(
        self,
        *,
        state: ControllerState,
        trigger_depth: float,
        target_width: float,
        command_latency_sec: float | None,
        command_sequence: int | None,
        in_flight: bool,
    ) -> None:
        sample = GripperTestbedSample(
            wall_time=time.time(),
            trigger_depth=float(trigger_depth),
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

    @staticmethod
    def _button_pressed(message: UnityTeleopMessage, index: int) -> bool:
        if index >= len(message.leftHand.buttonState):
            return False
        return bool(message.leftHand.buttonState[index])


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

    app = FastAPI(title="VT Franka Gripper Testbed", version="0.1.0", lifespan=lifespan)

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
        return service.controller.open_gripper(
            width=service.settings.max_gripper_width,
            velocity=service.settings.gripper_velocity,
            force_limit=service.settings.force_limit,
        )

    @app.post("/api/v1/gripper/stop")
    def stop_gripper():
        return service.controller.stop_gripper()

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
  <title>VT Franka Gripper Testbed</title>
  <style>
    :root {
      --bg: #eef3f6;
      --panel: rgba(255, 255, 255, 0.92);
      --ink: #18212b;
      --muted: #5f6b77;
      --line: rgba(24, 33, 43, 0.12);
      --good: #1d6b44;
      --warn: #9a5a1e;
      --bad: #a12f2f;
      --accent: #2563eb;
      --shadow: 0 18px 40px rgba(33, 47, 67, 0.12);
      --mono: "IBM Plex Mono", "SFMono-Regular", monospace;
      --ui: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--ui);
      color: var(--ink);
      background: linear-gradient(180deg, #f7fbfd, var(--bg));
      min-height: 100vh;
    }
    .shell {
      max-width: 1480px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      gap: 16px;
    }
    .hero, .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
      backdrop-filter: blur(12px);
    }
    .hero { padding: 18px 20px; display: grid; gap: 12px; }
    .top { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 12px; }
    h1 { margin: 0; font-size: 30px; line-height: 1.05; }
    .sub { color: var(--muted); font-size: 14px; }
    .pillrow { display: flex; gap: 8px; flex-wrap: wrap; }
    .pill { padding: 8px 12px; border-radius: 999px; border: 1px solid var(--line); background: rgba(255,255,255,0.75); font-size: 13px; font-weight: 600; }
    .layout { display: grid; grid-template-columns: 300px minmax(0, 1fr) 380px; gap: 16px; }
    .panel { padding: 16px; display: grid; gap: 12px; min-height: 180px; }
    .panel h2 { margin: 0; font-size: 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
    .metric { border-radius: 14px; border: 1px solid var(--line); background: rgba(255,255,255,0.82); padding: 12px; }
    .metric .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
    .metric .value { font-size: 18px; font-weight: 700; word-break: break-word; }
    .controls { display: grid; gap: 10px; }
    .row { display: grid; gap: 6px; }
    .row label { font-size: 12px; color: var(--muted); }
    input, button, select { font: inherit; }
    input[type="number"], input[type="text"], select {
      width: 100%; border: 1px solid var(--line); border-radius: 12px; padding: 9px 10px; background: white;
    }
    input[type="range"] { width: 100%; }
    .actions { display: flex; flex-wrap: wrap; gap: 8px; }
    button {
      border: 1px solid var(--line); background: white; border-radius: 12px; padding: 10px 14px; cursor: pointer; font-weight: 600;
    }
    button.primary { background: var(--accent); color: white; border-color: var(--accent); }
    button.good { background: var(--good); color: white; border-color: var(--good); }
    button.warn { background: var(--warn); color: white; border-color: var(--warn); }
    button.bad { background: var(--bad); color: white; border-color: var(--bad); }
    .chart {
      border: 1px solid var(--line); border-radius: 14px; background: rgba(255,255,255,0.88);
      min-height: 220px; padding: 10px; display: grid; gap: 10px;
    }
    canvas { width: 100%; height: 180px; display: block; }
    .table { max-height: 280px; overflow: auto; border: 1px solid var(--line); border-radius: 14px; background: rgba(255,255,255,0.88); }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid rgba(24,33,43,0.08); text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: #f8fafc; }
    .mono { font-family: var(--mono); font-size: 11px; }
    .status-good { color: var(--good); }
    .status-warn { color: var(--warn); }
    .status-bad { color: var(--bad); }
    @media (max-width: 1180px) { .layout { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="top">
        <div>
          <h1>VT Franka Gripper Testbed</h1>
          <div class="sub">Quest trigger depth to Panda Hand width with latest-target-wins command flow.</div>
        </div>
        <div class="pillrow">
          <div class="pill" id="pill-enabled">enabled: false</div>
          <div class="pill" id="pill-armed">armed: false</div>
          <div class="pill" id="pill-conn">controller: unknown</div>
        </div>
      </div>
      <div class="grid" id="metrics"></div>
    </section>

    <section class="layout">
      <div class="panel">
        <h2>Controls</h2>
        <div class="controls">
          <div class="row">
            <label>Run name</label>
            <input id="run-name" type="text" value="gripper_testbed_run">
          </div>
          <div class="actions">
            <button class="primary" onclick="apiPost('/api/v1/enable?enabled=true')">Enable</button>
            <button onclick="apiPost('/api/v1/enable?enabled=false')">Disable</button>
            <button class="good" onclick="apiPost('/api/v1/arm')">Arm</button>
            <button onclick="apiPost('/api/v1/disarm')">Disarm</button>
          </div>
          <div class="actions">
            <button onclick="apiPost('/api/v1/gripper/open')">Open</button>
            <button class="warn" onclick="apiPost('/api/v1/gripper/stop')">Stop</button>
          </div>
          <div class="actions">
            <button onclick="apiPost('/api/v1/run/start?run_name=' + encodeURIComponent(document.getElementById('run-name').value))">Start Run</button>
            <button onclick="apiPost('/api/v1/run/episode/start')">Start Episode</button>
            <button onclick="apiPost('/api/v1/run/episode/stop?outcome=saved')">Stop Episode</button>
          </div>
          <div class="row">
            <label>Force limit</label>
            <select id="force-limit">
              <option>3</option><option selected>5</option><option>7</option><option>10</option><option>15</option>
            </select>
          </div>
          <div class="row">
            <label>Gamma</label>
            <input id="gamma" type="range" min="0.8" max="3.0" step="0.1" value="1.5">
            <div class="mono" id="gamma-val">1.5</div>
          </div>
          <div class="row">
            <label>Trigger depth</label>
            <div class="mono" id="trigger-val">0.000</div>
          </div>
          <div class="row">
            <label>Target width</label>
            <div class="mono" id="target-width-val">0.0000</div>
          </div>
        </div>
      </div>

      <div class="panel">
        <h2>Live Charts</h2>
        <div class="chart">
          <canvas id="chart"></canvas>
        </div>
        <div class="chart">
          <canvas id="chart2"></canvas>
        </div>
      </div>

      <div class="panel">
        <h2>Events</h2>
        <div class="table">
          <table>
            <thead><tr><th>Time</th><th>Event</th><th>Details</th></tr></thead>
            <tbody id="events"></tbody>
          </table>
        </div>
      </div>
    </section>
  </div>
  <script>
    const metrics = document.getElementById('metrics');
    const eventsTbody = document.getElementById('events');
    const chart = document.getElementById('chart');
    const chart2 = document.getElementById('chart2');
    const ctx = chart.getContext('2d');
    const ctx2 = chart2.getContext('2d');
    const samples = [];
    const maxSamples = 180;

    function fmt(v, digits=3) {
      if (v === null || v === undefined || Number.isNaN(v)) return 'n/a';
      return Number(v).toFixed(digits);
    }

    function apiPost(path) {
      fetch(path, {method:'POST'}).then(refresh).catch(console.error);
    }

    document.getElementById('gamma').addEventListener('input', (e) => {
      document.getElementById('gamma-val').textContent = Number(e.target.value).toFixed(1);
    });
    document.getElementById('force-limit').addEventListener('change', syncSettings);
    document.getElementById('gamma').addEventListener('change', syncSettings);

    function syncSettings() {
      const payload = {
        force_limit: Number(document.getElementById('force-limit').value),
        width_gamma: Number(document.getElementById('gamma').value),
      };
      fetch('/api/v1/settings', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload),
      }).then(refresh).catch(console.error);
    }

    function updateMetrics(data) {
      const rows = [
        ['Trigger', fmt(data.latest_trigger_depth, 3)],
        ['Target width', fmt(data.last_sent_width, 4)],
        ['Measured width', fmt(data.latest_state?.gripper_width, 4)],
        ['Measured force', fmt(data.latest_state?.gripper_force, 2)],
        ['Latency', fmt(data.last_command_latency_sec, 3)],
        ['Command count', String(data.command_counter ?? 0)],
        ['Control error', data.control_error || 'none'],
        ['Active run', data.active_run_dir || 'none'],
      ];
      metrics.innerHTML = rows.map(([label, value]) => `
        <div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>
      `).join('');
      document.getElementById('pill-enabled').textContent = `enabled: ${data.enabled}`;
      document.getElementById('pill-armed').textContent = `armed: ${data.armed}`;
      document.getElementById('pill-conn').textContent = `controller: ${data.latest_state?.backend || 'unknown'}`;
      if (data.settings) {
        document.getElementById('force-limit').value = String(data.settings.force_limit);
        document.getElementById('gamma').value = String(data.settings.width_gamma);
        document.getElementById('gamma-val').textContent = Number(data.settings.width_gamma).toFixed(1);
      }
    }

    function drawCanvas(canvas, points, seriesNames, colors) {
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.clientWidth;
      const height = canvas.clientHeight;
      canvas.width = width * dpr;
      canvas.height = height * dpr;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr,0,0,dpr,0,0);
      ctx.clearRect(0,0,width,height);
      ctx.fillStyle = '#fff';
      ctx.fillRect(0,0,width,height);
      ctx.strokeStyle = 'rgba(24,33,43,0.08)';
      for (let i=0;i<5;i++) {
        const y = 20 + (height-40) * i / 4;
        ctx.beginPath(); ctx.moveTo(40,y); ctx.lineTo(width-12,y); ctx.stroke();
      }
      if (!points.length) return;
      const xs = points.map(p => p.t);
      const xMin = xs[0];
      const xMax = xs[xs.length-1] || (xMin + 1);
      const vals = seriesNames.flatMap((_, idx) => points.map(p => p[idx]));
      const vMin = Math.min(...vals, 0);
      const vMax = Math.max(...vals, 1);
      const span = Math.max(vMax - vMin, 1e-6);
      const plot = (x, y) => [
        40 + (x - xMin) / Math.max(xMax - xMin, 1e-6) * (width - 56),
        20 + (1 - (y - vMin) / span) * (height - 40)
      ];
      seriesNames.forEach((name, idx) => {
        ctx.strokeStyle = colors[idx];
        ctx.lineWidth = 2;
        ctx.beginPath();
        points.forEach((p, i) => {
          const [x, y] = plot(p.t, p[idx]);
          if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.stroke();
      });
    }

    function updateCharts(samples) {
      const points = samples.map(s => ({t: s.wall_time, 0: s.trigger_depth, 1: s.target_width, 2: s.measured_width, 3: s.width_error}));
      drawCanvas(chart, points, ['trigger', 'target', 'measured'], ['#2563eb', '#1d6b44', '#9a5a1e']);
      drawCanvas(chart2, points.map(p => ({t: p.t, 0: p[3], 1: p[2]})), ['error', 'measured'], ['#a12f2f', '#2563eb']);
    }

    function updateEvents(rows) {
      eventsTbody.innerHTML = rows.slice(-30).reverse().map(ev => `
        <tr>
          <td class="mono">${new Date((ev.wall_time || 0) * 1000).toLocaleTimeString()}</td>
          <td>${ev.event_type || ev.type || 'event'}</td>
          <td class="mono">${JSON.stringify(ev).slice(0, 220)}</td>
        </tr>
      `).join('');
    }

    async function refresh() {
      const status = await fetch('/api/v1/status').then(r => r.json());
      const state = await fetch('/api/v1/state').then(r => r.json());
      const sampleData = await fetch('/api/v1/samples').then(r => r.json());
      const eventData = await fetch('/api/v1/events').then(r => r.json());
      const latest = sampleData.samples?.at(-1) || {};
      status.latest_state = state;
      status.latest_trigger_depth = latest.trigger_depth;
      status.last_sent_width = latest.target_width;
      status.last_command_latency_sec = latest.command_latency_sec;
      status.command_counter = status.command_counter ?? 0;
      updateMetrics(status);
      samples.splice(0, samples.length, ...(sampleData.samples || []).slice(-maxSamples));
      updateCharts(samples);
      updateEvents(eventData.events || []);
      document.getElementById('trigger-val').textContent = fmt(latest.trigger_depth, 3);
      document.getElementById('target-width-val').textContent = fmt(latest.target_width, 4);
    }

    refresh();
    setInterval(refresh, 300);
  </script>
</body>
</html>
"""
