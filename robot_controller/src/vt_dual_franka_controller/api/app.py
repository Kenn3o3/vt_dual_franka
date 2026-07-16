from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from vt_dual_franka_shared.models import GripperGraspCommand, GripperWidthCommand, ResetCommand, TcpTargetCommand

from ..control.service import ControllerBusyError, ControllerService


def create_app(service: ControllerService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        try:
            yield
        finally:
            service.shutdown()

    app = FastAPI(title="VT Dual Franka Controller", version="0.1.0", lifespan=lifespan)

    @app.get("/api/v1/health")
    def health():
        return service.get_health()

    @app.get("/api/v1/state")
    def state():
        return service.get_state()

    @app.get("/api/v1/tcp")
    def tcp():
        return service.get_state().tcp_pose

    @app.post("/api/v1/commands/tcp")
    def move_tcp(command: TcpTargetCommand):
        try:
            service.queue_tcp_command(command)
        except ControllerBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "queued"}

    @app.post("/api/v1/commands/gripper/width")
    def move_gripper(command: GripperWidthCommand):
        try:
            service.queue_gripper_width_command(command)
        except ControllerBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "completed" if command.blocking else "queued"}

    @app.post("/api/v1/commands/gripper/grasp")
    def grasp_gripper(command: GripperGraspCommand):
        try:
            service.queue_gripper_grasp_command(command)
        except ControllerBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "completed" if command.blocking else "queued"}

    @app.post("/api/v1/commands/gripper/stop")
    def stop_gripper():
        try:
            service.stop_gripper()
        except ControllerBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.post("/api/v1/actions/reset")
    def reset(command: ResetCommand):
        try:
            return service.run_reset(command)
        except ControllerBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/v1/actions/home")
    def go_home():
        try:
            service.go_home()
        except ControllerBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"status": "ok"}

    @app.post("/api/v1/actions/ready")
    def go_ready():
        try:
            service.go_ready()
        except ControllerBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "ok"}

    return app
