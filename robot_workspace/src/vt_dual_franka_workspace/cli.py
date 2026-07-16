from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from vt_dual_franka_shared.transforms import BimanualCalibration

from .collection import BimanualDataCollector
from .config import (
    InferenceRuntimeSettings,
    PolicyConfig,
    TaskConfig,
    WorkspaceSettings,
    load_inference_config,
    load_policy_config,
    load_task_config,
    load_workspace_config,
)
from .inference import BimanualPolicyRunner
from .operator import OperatorLogBuffer, install_operator_logging
from .policies import resolve_policy
from .runtime.dual_arm import DualArmCoordinator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="VT Dual Franka: dual-arm-only collection, training, teleoperation, and inference"
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    collect = subparsers.add_parser("collect", help="Collect synchronized bimanual demonstrations")
    _add_workspace_and_task_args(collect)

    teleop = subparsers.add_parser("teleop", help="Control left/right Franka with left/right Quest hands")
    teleop.add_argument("--workspace-config", default="robot_workspace/config/workspace.yaml")
    teleop.add_argument("--poll-hz", type=float, default=60.0)

    make_dataset = subparsers.add_parser(
        "make-dataset",
        help="Build the 20D commanded-action bimanual common dataset",
    )
    make_dataset.add_argument("collect_task_dir", type=Path)
    make_dataset.add_argument("--output-dir", type=Path, default=None)
    make_dataset.add_argument("--name", "--dataset-name", dest="dataset_name", default="real_bimanual_v1")
    make_dataset.add_argument("--target-hz", type=float, default=10.0)
    make_dataset.add_argument("--max-action-lead-sec", type=float, default=None)
    make_dataset.add_argument("--overwrite", action="store_true")

    train = subparsers.add_parser("train", help="Train the dual-arm 20D Diffusion Policy")
    _add_train_args(train)

    run_policy = subparsers.add_parser("run-policy", help="Run the dual-arm 20D policy")
    _add_workspace_and_task_args(run_policy, task_required=False)
    run_policy.add_argument("--policy-config", default="robot_workspace/config/policies/dp_bimanual_demo.yaml")
    run_policy.add_argument("--checkpoint", type=Path, default=None)
    run_policy.add_argument(
        "--inference-config",
        default="robot_workspace/config/inference/bimanual_demo_dp.yaml",
    )
    run_policy.add_argument("--run-name", default=None)
    run_policy.add_argument("--resume", action="store_true")
    run_policy.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    if args.subcommand == "collect":
        workspace = load_workspace_config(args.workspace_config)
        task = _load_requested_task(args)
        log_buffer = _install_operator_logging(workspace)
        BimanualDataCollector(
            workspace,
            task,
            DualArmCoordinator.from_workspace(
                workspace,
                poll_hz=task.collection.controller_state_poll_hz,
            ),
            BimanualCalibration.from_dir(workspace.calibration.calibration_dir),
            log_buffer=log_buffer,
        ).run()
        return

    if args.subcommand == "teleop":
        _run_teleop(args)
        return

    if args.subcommand == "make-dataset":
        from .datasets.bimanual_common import MakeBimanualDatasetConfig, make_bimanual_common_dataset

        manifest_path = make_bimanual_common_dataset(
            MakeBimanualDatasetConfig(
                collect_task_dir=args.collect_task_dir,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name,
                target_hz=args.target_hz,
                max_action_lead_sec=args.max_action_lead_sec,
                overwrite=args.overwrite,
            )
        )
        print(f"Bimanual dataset manifest: {manifest_path}", flush=True)
        return

    if args.subcommand == "train":
        from .policies.common.visuotactile.train import config_from_args, train_visuotactile

        args.model = "dp_bimanual"
        result = train_visuotactile(config_from_args(args))
        print(f"checkpoint_dir={result.checkpoint_dir}", flush=True)
        print(f"dataset_dir={result.dataset_dir}", flush=True)
        print(f"manifest={result.manifest_path}", flush=True)
        return

    if args.subcommand == "run-policy":
        workspace = load_workspace_config(args.workspace_config)
        inference = load_inference_config(args.inference_config)
        if args.task or args.task_config:
            inference = _merge_task_into_inference(inference, _load_requested_task(args))
        policy_config = _resolve_bimanual_policy_config(
            checkpoint=args.checkpoint,
            policy_config_path=args.policy_config,
            fallback_task_name=inference.task_name,
        )
        policy = resolve_policy(policy_config, inference, workspace)
        log_buffer = _install_operator_logging(workspace)
        run_name = args.run_name
        if run_name is None and args.resume:
            if args.checkpoint is None:
                raise SystemExit("run-policy --resume requires --checkpoint")
            run_name = _checkpoint_resume_run_name(args.checkpoint)
        BimanualPolicyRunner(
            workspace,
            inference,
            DualArmCoordinator.from_workspace(
                workspace,
                poll_hz=max(inference.controller_state_poll_hz, inference.control_hz),
            ),
            BimanualCalibration.from_dir(workspace.calibration.calibration_dir),
            policy,
            run_name=run_name,
            log_buffer=log_buffer,
            resume_run=not args.no_resume,
        ).run()
        return


def _add_workspace_and_task_args(parser: argparse.ArgumentParser, *, task_required: bool = True) -> None:
    parser.add_argument("--workspace-config", default="robot_workspace/config/workspace.yaml")
    parser.add_argument("--task", default="bimanual_demo" if task_required else None)
    parser.add_argument("--task-config", default=None)
    parser.add_argument("--task-name", default=None)


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace-config", default="robot_workspace/config/workspace.yaml")
    parser.add_argument("--task-name", default="bimanual_demo")
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--backend-dataset-root", type=Path, default=None)
    parser.add_argument("--raw-run-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="real_bimanual_v1")
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--extra-arg", dest="extra_args", action="append", default=[])
    parser.add_argument("--command", dest="command_override", nargs=argparse.REMAINDER, default=None)


def _load_requested_task(args: argparse.Namespace) -> TaskConfig:
    task_path = _resolve_task_config_path(
        args.workspace_config,
        getattr(args, "task_config", None),
        getattr(args, "task", None),
    )
    task = load_task_config(
        task_path,
        task_name_override=getattr(args, "task_name", None) or getattr(args, "task", None),
    )
    if task.task_name != "bimanual_demo":
        raise SystemExit(
            f"VT Dual Franka supports only the dual-arm task 'bimanual_demo'; got {task.task_name!r}"
        )
    return task


def _resolve_task_config_path(
    workspace_config_path: str,
    task_config_path: str | None,
    task_name: str | None,
) -> Path:
    if task_config_path is not None:
        return Path(task_config_path)
    if task_name is None:
        raise SystemExit("command requires --task bimanual_demo or --task-config")
    return Path(workspace_config_path).resolve().parent / "tasks" / f"{task_name}.yaml"


def _merge_task_into_inference(
    inference: InferenceRuntimeSettings,
    task: TaskConfig,
) -> InferenceRuntimeSettings:
    return inference.model_copy(
        update={
            "task_name": task.task_name,
            "modality": task.modality,
            "rgb_cameras": task.rgb_cameras,
            "gelsights": task.gelsights,
            "initial_poses": task.initial_poses,
            "initial_move_duration_sec": task.initial_move_duration_sec,
            "home_joint_duration_sec": task.home_joint_duration_sec,
            "home_joint_tolerance_rad": task.home_joint_tolerance_rad,
            "home_joint_settle_timeout_sec": task.home_joint_settle_timeout_sec,
            "gripper_forever_closed": task.gripper_forever_closed,
        }
    )


def _resolve_bimanual_policy_config(
    *,
    checkpoint: Path | None,
    policy_config_path: str,
    fallback_task_name: str,
) -> PolicyConfig:
    if checkpoint is None:
        config = load_policy_config(policy_config_path)
    else:
        checkpoint_dir, checkpoint_file = _resolve_checkpoint_reference(checkpoint)
        manifest_path = checkpoint_dir / "policy_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Missing policy_manifest.json: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        model = str(manifest.get("model") or "")
        if model != "dp_bimanual":
            raise ValueError(f"Expected model='dp_bimanual', got {model!r}")
        payload = {
            "model": model,
            "task_name": str(manifest.get("task_name") or fallback_task_name),
        }
        if checkpoint_file is not None:
            payload["checkpoint_file"] = _checkpoint_file_reference(checkpoint_dir, checkpoint_file)
        config = PolicyConfig(type="dp_bimanual", checkpoint_path=checkpoint_dir, config=payload)
    if config.type != "dp_bimanual" or config.config.get("model") != "dp_bimanual":
        raise ValueError("Only type/model 'dp_bimanual' is supported in vt_dual_franka")
    return config


def _resolve_checkpoint_reference(checkpoint: Path) -> tuple[Path, Path | None]:
    path = checkpoint.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        for parent in [path.parent, *path.parents]:
            if (parent / "policy_manifest.json").is_file():
                return parent, path
        return path.parent, path
    return path, None


def _checkpoint_file_reference(checkpoint_dir: Path, checkpoint_file: Path) -> str:
    try:
        return checkpoint_file.resolve().relative_to(checkpoint_dir.resolve()).as_posix()
    except ValueError:
        return checkpoint_file.name


def _checkpoint_resume_run_name(checkpoint: Path) -> str:
    path = checkpoint.expanduser()
    name = path.stem if path.suffix else path.name
    return re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower() or "checkpoint"


def _install_operator_logging(workspace: WorkspaceSettings) -> OperatorLogBuffer:
    log_buffer = OperatorLogBuffer(workspace.operator_ui.log_buffer_size)
    install_operator_logging(log_buffer, suppress_console_noise=workspace.operator_ui.enabled)
    return log_buffer


def _run_teleop(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required for teleop") from exc
    from .teleop.dual_quest_server import DualQuestTeleopService, create_dual_teleop_app

    workspace = load_workspace_config(args.workspace_config)
    coordinator = DualArmCoordinator.from_workspace(workspace, poll_hz=args.poll_hz)
    service = DualQuestTeleopService(
        workspace.teleop,
        coordinator,
        BimanualCalibration.from_dir(workspace.calibration.calibration_dir),
    )
    coordinator.start()
    service.set_teleop_enabled(True)
    try:
        print(f"Bimanual teleop API: http://{workspace.teleop.host}:{workspace.teleop.port}/unity", flush=True)
        uvicorn.run(
            create_dual_teleop_app(service),
            host=workspace.teleop.host,
            port=workspace.teleop.port,
        )
    finally:
        service.stop()
        coordinator.stop()


if __name__ == "__main__":
    main()
