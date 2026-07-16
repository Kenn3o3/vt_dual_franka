from __future__ import annotations

import argparse
import logging

from vt_dual_franka_shared.config import load_yaml_model

from .backends.mock import MockFrankaBackend
from .backends.gripper_only import PolymetisGripperOnlyBackend
from .backends.polymetis import PolymetisFrankaBackend
from .control.gripper_service import GripperTestbedService
from .control.service import ControllerService
from .settings import ControlSettings, ControllerSettings, RosGripperTestbedSettings, TeleopGripperDefaults


def build_backend(settings: ControllerSettings):
    if settings.backend.kind == "mock":
        return MockFrankaBackend()
    if settings.backend.kind == "polymetis":
        return PolymetisFrankaBackend(
            robot_ip=settings.backend.robot_ip,
            robot_port=settings.backend.robot_port,
            gripper_ip=settings.backend.gripper_ip,
            gripper_port=settings.backend.gripper_port,
        )
    raise ValueError(f"Unsupported backend kind: {settings.backend.kind}")


def main() -> None:
    parser = argparse.ArgumentParser(description="VT Dual Franka controller CLI")
    parser.add_argument(
        "command",
        choices=["run", "home", "ready", "gripper-testbed", "ros-gripper-testbed"],
        help="Command to execute",
    )
    parser.add_argument("--config", default="config/controller.yaml", help="Path to YAML config")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.command == "ros-gripper-testbed":
        try:
            import uvicorn
            from .api.gripper_testbed_app import create_gripper_testbed_app
            from .backends.franka_ros_gripper import FrankaRosGripperOnlyBackend
        except ImportError as exc:
            raise RuntimeError("Failed to import FastAPI/uvicorn for 'vt-dual-franka-controller ros-gripper-testbed'.") from exc
        settings = load_yaml_model(args.config, RosGripperTestbedSettings)
        backend = FrankaRosGripperOnlyBackend(settings.ros)
        service_settings = ControllerSettings(
            server=settings.server,
            control=ControlSettings(control_frequency_hz=settings.control_frequency_hz),
            teleop=TeleopGripperDefaults(
                max_gripper_width=settings.ros.max_gripper_width,
                gripper_velocity=settings.ros.default_velocity,
                grasp_force=settings.ros.default_force_limit,
            ),
        )
        service = GripperTestbedService(service_settings, backend)
        app = create_gripper_testbed_app(service)
        uvicorn.run(app, host=settings.server.host, port=settings.server.port)
        return

    settings = load_yaml_model(args.config, ControllerSettings)
    if args.command == "gripper-testbed":
        try:
            import uvicorn
            from .api.gripper_testbed_app import create_gripper_testbed_app
        except ImportError as exc:
            raise RuntimeError("Failed to import FastAPI/uvicorn for 'vt-dual-franka-controller gripper-testbed'.") from exc
        if settings.backend.kind == "mock":
            backend = MockFrankaBackend()
        else:
            backend = PolymetisGripperOnlyBackend(
                gripper_ip=settings.backend.gripper_ip,
                gripper_port=settings.backend.gripper_port,
            )
        service = GripperTestbedService(settings, backend)
        app = create_gripper_testbed_app(service)
        uvicorn.run(app, host=settings.server.host, port=settings.server.port)
        return

    backend = build_backend(settings)

    if args.command == "home":
        backend.go_home(settings.control.home_ee_pose, settings.control.home_duration_sec)
        backend.shutdown()
        return

    if args.command == "ready":
        if settings.control.ready_ee_pose is None:
            raise SystemExit("ready_ee_pose is not configured in the controller config")
        backend.go_home(settings.control.ready_ee_pose, settings.control.ready_duration_sec)
        backend.shutdown()
        return

    try:
        import uvicorn
        from .api.app import create_app
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import FastAPI/uvicorn for 'vt-dual-franka-controller run'. "
        ) from exc

    service = ControllerService(settings, backend)
    app = create_app(service)
    uvicorn.run(app, host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
