from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from vt_dual_franka_shared.models import GripperTestbedTargetCommand

from ..control.gripper_service import GripperTestbedService


def create_gripper_testbed_app(service: GripperTestbedService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(title="VT Dual Franka Gripper Test Controller", version="0.1.0", lifespan=lifespan)

    @app.get("/api/v1/health")
    def health():
        return service.get_health()

    @app.get("/api/v1/state")
    def state():
        return service.get_state()

    @app.get("/api/v1/gripper/status")
    def gripper_status():
        return service.get_snapshot()

    @app.post("/api/v1/gripper/target")
    def gripper_target(command: GripperTestbedTargetCommand):
        try:
            return service.queue_target_command(command)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/gripper/open")
    def open_gripper(width: float | None = None, velocity: float | None = None, force_limit: float | None = None):
        try:
            return service.open_gripper(width=width, velocity=velocity, force_limit=force_limit)
        except RuntimeError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/api/v1/gripper/stop")
    def stop_gripper():
        return service.stop_gripper()

    return app

