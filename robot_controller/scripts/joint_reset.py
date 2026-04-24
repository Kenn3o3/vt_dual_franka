#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from polymetis import RobotInterface

from vt_franka_shared.config import load_yaml_model
from vt_franka_controller.settings import ControllerSettings


def format_array(values) -> str:
    return np.array2string(np.asarray(values, dtype=np.float64), precision=4, suppress_small=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move Franka to configured ready joint positions only.")
    parser.add_argument(
        "--config",
        default="/home/medair/vt_franka/robot_controller/config/controller.yaml",
        help="Path to controller.yaml",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip interactive confirmation",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    settings = load_yaml_model(config_path, ControllerSettings)

    joint_positions = settings.control.ready_joint_positions
    if joint_positions is None:
        raise SystemExit("ready_joint_positions is not configured in controller.yaml")
    duration_sec = settings.control.ready_joint_duration_sec or settings.control.ready_duration_sec

    print(f"Config: {config_path}")
    print(f"Robot endpoint: {settings.backend.robot_ip}:{settings.backend.robot_port}")
    print(f"Target ready joints (rad): {format_array(joint_positions)}")
    print(f"Move duration (s): {duration_sec:.2f}")

    if not args.yes:
        response = input("Type 'yes' to execute joint reset: ").strip().lower()
        if response != "yes":
            raise SystemExit("Aborted.")

    robot = RobotInterface(
        ip_address=settings.backend.robot_ip,
        port=settings.backend.robot_port,
    )
    try:
        robot.move_to_joint_positions(
            positions=torch.tensor(np.asarray(joint_positions, dtype=np.float32)),
            time_to_go=float(duration_sec),
        )
        print("Joint reset completed.")
    finally:
        try:
            robot.terminate_current_policy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
