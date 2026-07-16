from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict

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

    @app.get("/get_current_tcp/{robot_side}")
    def legacy_get_current_tcp(robot_side: str):
        _ensure_local_arm(service, robot_side)
        return service.get_state().tcp_pose

    @app.get("/get_current_robot_states")
    def legacy_get_current_robot_states():
        state = service.get_state()
        return _legacy_state_payload(state)

    @app.post("/move_tcp/{robot_side}")
    def legacy_move_tcp(robot_side: str, command: TcpTargetCommand):
        _ensure_local_arm(service, robot_side)
        service.queue_tcp_command(command.model_copy(update={"arm_id": service.settings.arm_id}))
        return {"message": "Waypoint added for franka robot"}

    @app.post("/move_gripper/{robot_side}")
    def legacy_move_gripper(robot_side: str, command: GripperWidthCommand):
        _ensure_local_arm(service, robot_side)
        service.queue_gripper_width_command(command)
        return {"message": f"Gripper moving to width {command.width}"}

    @app.post("/move_gripper_force/{robot_side}")
    def legacy_grasp_gripper(robot_side: str, command: GripperWidthCommand):
        _ensure_local_arm(service, robot_side)
        grasp_command = GripperGraspCommand(
            velocity=command.velocity,
            force_limit=command.force_limit,
            source=command.source,
        )
        service.queue_gripper_grasp_command(grasp_command)
        return {"message": f"Gripper grasping with force {command.force_limit}"}

    @app.post("/stop_gripper/{robot_side}")
    def legacy_stop_gripper(robot_side: str):
        _ensure_local_arm(service, robot_side)
        service.stop_gripper()
        return {"message": "Gripper stopped"}

    @app.post("/birobot_go_home")
    def legacy_home():
        service.go_home()
        return {"message": "Robot moved to home position"}

    return app


def _ensure_local_arm(service: ControllerService, robot_side: str) -> None:
    if robot_side != service.settings.arm_id:
        raise HTTPException(
            status_code=400,
            detail=f"This controller serves {service.settings.arm_id!r}, not {robot_side!r}",
        )


def _legacy_state_payload(state) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "leftRobotTCP": [0.0] * 7,
        "leftRobotTCPVel": [0.0] * 6,
        "leftRobotTCPWrench": [0.0] * 6,
        "leftGripperState": [0.0, 0.0],
        "rightRobotTCP": [0.0] * 7,
        "rightRobotTCPVel": [0.0] * 6,
        "rightRobotTCPWrench": [0.0] * 6,
        "rightGripperState": [0.0, 0.0],
    }
    prefix = "left" if state.arm_id != "right" else "right"
    payload[f"{prefix}RobotTCP"] = state.tcp_pose
    payload[f"{prefix}RobotTCPVel"] = state.tcp_velocity
    payload[f"{prefix}RobotTCPWrench"] = state.tcp_wrench
    payload[f"{prefix}GripperState"] = [state.gripper_width, state.gripper_force]
    return payload
