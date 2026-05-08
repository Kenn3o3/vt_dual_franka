from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vt_franka_shared.transforms import SingleArmCalibration
from vt_franka_shared.config import load_yaml_model

from .collection import DataCollector
from .config import (
    load_inference_config,
    load_policy_config,
    load_task_config,
    load_workspace_config,
)
from .controller.client import ControllerClient
from .gripper_testbed import GripperTestbedControllerClient, GripperTestbedService, GripperTestbedSettings, create_gripper_testbed_app
from .gripper_testbed.report import write_gripper_testbed_report
from .gripper_testbed.replay import create_gripper_testbed_replay_app
from .inference import PolicyRunner
from .operator import OperatorLogBuffer, install_operator_logging
from .policies import resolve_policy


def main() -> None:
    parser = argparse.ArgumentParser(description="VT Franka workspace CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect = subparsers.add_parser("collect", help="Run the data collection pipeline")
    collect.add_argument("--workspace-config", default="config/workspace.yaml")
    collect.add_argument("--task", default=None)
    collect.add_argument("--task-config", default=None)
    collect.add_argument("--task-name", default=None)

    run_policy = subparsers.add_parser("run-policy", help="Run a policy on the robot")
    run_policy.add_argument("--workspace-config", default="config/workspace.yaml")
    run_policy.add_argument("--policy-config", required=True)
    run_policy.add_argument("--inference-config", required=True)
    run_policy.add_argument("--run-name", default=None)
    run_policy.add_argument("--no-resume", action="store_true")

    gripper_testbed = subparsers.add_parser("gripper-testbed", help="Run the standalone Panda Hand gripper testbed")
    gripper_testbed.add_argument("--config", default="config/gripper_testbed.yaml")
    gripper_report = subparsers.add_parser("gripper-testbed-report", help="Generate a static HTML report from a gripper testbed run")
    gripper_report.add_argument("--run-dir", required=True)
    gripper_report.add_argument("--output", default=None)
    gripper_replay = subparsers.add_parser("gripper-testbed-replay", help="Replay a gripper testbed run in the dashboard")
    gripper_replay.add_argument("--run-dir", required=True)
    gripper_replay.add_argument("--host", default="127.0.0.1")
    gripper_replay.add_argument("--port", type=int, default=8085)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.command == "collect":
        workspace = load_workspace_config(args.workspace_config)
        task_config_path = _resolve_task_config_path(args.workspace_config, args.task_config, args.task)
        task = load_task_config(task_config_path, task_name_override=args.task_name or args.task)
        log_buffer = OperatorLogBuffer(workspace.operator_ui.log_buffer_size)
        install_operator_logging(log_buffer, suppress_console_noise=workspace.operator_ui.enabled)
        DataCollector(
            workspace,
            task,
            _build_controller(workspace),
            _load_calibration(workspace),
            log_buffer=log_buffer,
        ).run()
        return

    if args.command == "gripper-testbed":
        settings = load_yaml_model(args.config, GripperTestbedSettings)
        log_buffer = OperatorLogBuffer()
        install_operator_logging(log_buffer, suppress_console_noise=False)
        controller = GripperTestbedControllerClient(
            settings.controller_host,
            settings.controller_port,
            settings.controller_request_timeout_sec,
        )
        service = GripperTestbedService(settings, controller, operator_log_buffer=log_buffer)
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Failed to import uvicorn for gripper-testbed") from exc
        print(f"Gripper testbed UI: http://{settings.host}:{settings.port}/operator", flush=True)
        uvicorn.run(create_gripper_testbed_app(service, operator_log_buffer=log_buffer), host=settings.host, port=settings.port)
        return

    if args.command == "gripper-testbed-report":
        report = write_gripper_testbed_report(args.run_dir, args.output)
        print(str(report), flush=True)
        return

    if args.command == "gripper-testbed-replay":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Failed to import uvicorn for gripper-testbed-replay") from exc
        print(f"Gripper testbed replay UI: http://{args.host}:{args.port}/operator", flush=True)
        uvicorn.run(create_gripper_testbed_replay_app(args.run_dir), host=args.host, port=args.port)
        return

    if args.command == "run-policy":
        workspace = load_workspace_config(args.workspace_config)
        inference = load_inference_config(args.inference_config)
        policy_config = load_policy_config(args.policy_config)
        policy = resolve_policy(policy_config, inference, workspace)
        log_buffer = OperatorLogBuffer(workspace.operator_ui.log_buffer_size)
        install_operator_logging(log_buffer, suppress_console_noise=workspace.operator_ui.enabled)
        PolicyRunner(
            workspace,
            inference,
            _build_controller(workspace),
            _load_calibration(workspace),
            policy,
            run_name=args.run_name,
            log_buffer=log_buffer,
            resume_run=not args.no_resume,
        ).run()
        return


def _build_controller(workspace):
    return ControllerClient(
        workspace.controller.host,
        workspace.controller.port,
        workspace.controller.request_timeout_sec,
    )


def _load_calibration(workspace) -> SingleArmCalibration:
    return SingleArmCalibration.from_dir(workspace.calibration.calibration_dir)


def _resolve_task_config_path(workspace_config_path: str, task_config_path: str | None, task_name: str | None) -> Path:
    if task_config_path is not None:
        return Path(task_config_path)
    if task_name is None:
        raise SystemExit("collect requires either --task or --task-config")
    return Path(workspace_config_path).resolve().parent / "tasks" / f"{task_name}.yaml"


if __name__ == "__main__":
    main()
