from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from vt_franka_shared.models import ControllerState

from .report import load_jsonl
from .service import _OPERATOR_PAGE


class GripperTestbedReplay:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.samples = self._load_stream("gripper_states")
        self.events = self._load_stream("gripper_telemetry")

    def status(self) -> dict[str, Any]:
        latest_state = None
        if self.samples:
            sample = self.samples[-1]
            latest_state = ControllerState(
                tcp_pose=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                tcp_velocity=[0.0] * 6,
                tcp_wrench=[0.0] * 6,
                joint_positions=[0.0] * 7,
                joint_velocities=[0.0] * 7,
                gripper_width=float(sample.get("measured_width", 0.0)),
                gripper_force=float(sample.get("measured_force", 0.0)),
                backend="replay",
            ).model_dump(mode="json")
        settings = {
            "force_limit": float(self.events[-1].get("force_limit", 5.0)) if self.events else 5.0,
            "width_gamma": 1.5,
            "min_gripper_width": 0.0,
            "max_gripper_width": 0.078,
            "gripper_velocity": 0.1,
            "command_hz": 20.0,
            "width_deadband_m": 0.0015,
        }
        return {
            "enabled": False,
            "armed": False,
            "command_counter": len(self.events),
            "last_sent_width": self.samples[-1].get("target_width") if self.samples else None,
            "last_sent_wall_time": self.samples[-1].get("wall_time") if self.samples else None,
            "control_error": None,
            "active_run_dir": str(self.run_dir),
            "active_episode_name": "replay",
            "latest_state": latest_state,
            "settings": settings,
        }

    def state(self) -> dict[str, Any]:
        latest = self.status().get("latest_state")
        if latest is None:
            return ControllerState(backend="replay").model_dump(mode="json")
        return latest

    def _load_stream(self, name: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.run_dir.glob(f"episodes/episode_*/streams/{name}.jsonl")):
            rows.extend(load_jsonl(path))
        return rows


def create_gripper_testbed_replay_app(run_dir: str | Path) -> FastAPI:
    replay = GripperTestbedReplay(run_dir)
    app = FastAPI(title="VT Franka Gripper Testbed Replay", version="0.1.0")

    @app.get("/operator", response_class=HTMLResponse)
    def operator_page() -> str:
        return _OPERATOR_PAGE

    @app.get("/api/v1/status")
    def status():
        return replay.status()

    @app.get("/api/v1/state")
    def state():
        return replay.state()

    @app.get("/api/v1/samples")
    def samples():
        return {"samples": replay.samples}

    @app.get("/api/v1/events")
    def events():
        return {"events": replay.events}

    @app.post("/api/v1/enable")
    def enable(enabled: bool = True):
        return {"enabled": False, "replay": True, "requested": enabled}

    @app.post("/api/v1/settings")
    def settings():
        return {"replay": True}

    @app.post("/api/v1/arm")
    def arm():
        return {"armed": False, "replay": True}

    @app.post("/api/v1/disarm")
    def disarm():
        return {"armed": False, "replay": True}

    @app.post("/api/v1/run/start")
    def start_run(run_name: str):
        return {"replay": True, "run_name": run_name}

    @app.post("/api/v1/run/episode/start")
    def start_episode(episode_name: str | None = None):
        return {"replay": True, "episode_name": episode_name}

    @app.post("/api/v1/run/episode/stop")
    def stop_episode(outcome: str = "saved"):
        return {"replay": True, "outcome": outcome}

    @app.post("/api/v1/gripper/open")
    def open_gripper():
        return {"status": "ignored", "replay": True}

    @app.post("/api/v1/gripper/stop")
    def stop_gripper():
        return {"status": "ignored", "replay": True}

    return app
