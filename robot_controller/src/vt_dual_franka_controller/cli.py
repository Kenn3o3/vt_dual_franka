from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vt_dual_franka_shared.config import load_yaml_model

from .backends.mock import MockFrankaBackend
from .backends.polymetis import PolymetisFrankaBackend
from .control.service import ControllerService
from .settings import ControllerSettings


def build_backend(settings: ControllerSettings):
    if settings.backend.kind == "mock":
        return MockFrankaBackend()
    return PolymetisFrankaBackend(
        robot_ip=settings.backend.robot_ip,
        robot_port=settings.backend.robot_port,
        gripper_ip=settings.backend.gripper_ip,
        gripper_port=settings.backend.gripper_port,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One arm process of the fixed VT Dual Franka controller pair"
    )
    parser.add_argument("command", choices=["run"])
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Explicit controller_left.yaml or controller_right.yaml; no single-arm default exists",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = load_yaml_model(args.config, ControllerSettings)
    if args.config.name not in {"controller_left.yaml", "controller_right.yaml"}:
        raise SystemExit("Use the canonical controller_left.yaml or controller_right.yaml config")

    try:
        import uvicorn
        from .api.app import create_app
    except ImportError as exc:
        raise RuntimeError("FastAPI and uvicorn are required to run the dual controller pair") from exc

    service = ControllerService(settings, build_backend(settings))
    uvicorn.run(create_app(service), host=settings.server.host, port=settings.server.port)


if __name__ == "__main__":
    main()
