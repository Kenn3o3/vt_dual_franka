from __future__ import annotations

import argparse
import json
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
    PolicyConfig,
    WorkspaceSettings,
)
from .controller.client import ControllerClient
from .gripper_testbed import GripperTestbedControllerClient, GripperTestbedService, GripperTestbedSettings, create_gripper_testbed_app
from .gripper_testbed.report import write_gripper_testbed_report
from .gripper_testbed.replay import create_gripper_testbed_replay_app
from .inference import PolicyRunner
from .operator import OperatorLogBuffer, install_operator_logging
from .policies import resolve_policy
from .policies.visuotactile.config import DEFAULT_DATASET_NAME, DEFAULT_PREPROCESS1_PROFILE, MODEL_SPECS


def main() -> None:
    parser = argparse.ArgumentParser(description="VT Franka workspace CLI")
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    collect = subparsers.add_parser("collect", help="Run the data collection pipeline")
    collect.add_argument("--workspace-config", default="config/workspace.yaml")
    collect.add_argument("--task", default=None)
    collect.add_argument("--task-config", default=None)
    collect.add_argument("--task-name", default=None)

    make_dataset = subparsers.add_parser("make-dataset", help="Build the synchronized model-agnostic common dataset")
    make_dataset.add_argument("collect_task_dir", type=Path)
    make_dataset.add_argument("--output-dir", type=Path, default=None)
    make_dataset.add_argument("--name", "--dataset-name", dest="dataset_name", default="real_640x480_v1")
    make_dataset.add_argument("--target-hz", type=float, default=10.0)
    make_dataset.add_argument("--max-action-lead-sec", type=float, default=None)
    make_dataset.add_argument("--overwrite", action="store_true")

    diagnose = subparsers.add_parser("diagnose-cameras", help="Benchmark task cameras through camera standardization")
    diagnose.add_argument("--workspace-config", default="config/workspace.yaml")
    diagnose.add_argument("--task", default=None)
    diagnose.add_argument("--task-config", default=None)
    diagnose.add_argument("--duration-sec", type=float, default=10.0)
    diagnose.add_argument("--output-root", type=Path, default=Path("analysis/camera_diagnostics"))
    diagnose.add_argument("--no-rgb", action="store_true")
    diagnose.add_argument("--no-gelsight", action="store_true")

    run_policy = subparsers.add_parser("run-policy", help="Run a policy on the robot")
    run_policy.add_argument("--workspace-config", default="config/workspace.yaml")
    run_policy.add_argument("--task", default=None)
    run_policy.add_argument("--task-config", default=None)
    run_policy.add_argument("--policy-config", default=None)
    run_policy.add_argument("--checkpoint", type=Path, default=None)
    run_policy.add_argument("--inference-config", required=True)
    run_policy.add_argument("--run-name", default=None)
    run_policy.add_argument(
        "--resume",
        action="store_true",
        help="Resume the eval run for this checkpoint by deriving a stable run name from --checkpoint.",
    )
    run_policy.add_argument("--no-resume", action="store_true")

    prepare_vt = subparsers.add_parser("prepare-visuotactile", help="Prepare aligned episodes for visuotactile training")
    prepare_vt.add_argument("--workspace-config", default="config/workspace.yaml")
    prepare_vt.add_argument("--task-name", required=True)
    prepare_vt.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    prepare_vt.add_argument("--raw-run-dir", type=Path, default=None)
    prepare_vt.add_argument("--output-dir", type=Path, default=None)
    prepare_vt.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    prepare_vt.add_argument("--preprocess1-profile", default=DEFAULT_PREPROCESS1_PROFILE)
    prepare_vt.add_argument("--target-hz", type=float, default=10.0)
    prepare_vt.add_argument("--image-size", type=int, default=None)
    prepare_vt.add_argument("--val-ratio", type=float, default=0.2)
    prepare_vt.add_argument("--val-episodes", type=int, default=None)
    prepare_vt.add_argument("--canonical-size", type=int, default=480)
    prepare_vt.add_argument("--gelsight-crop-box", default=None, help="Optional x0,y0,x1,y1 crop before canonical resize")
    prepare_vt.add_argument("--gelsight-margin-fraction", type=float, default=0.0)
    prepare_vt.add_argument("--source", choices=["raw", "preprocess1", "common"], default="raw")
    prepare_vt.add_argument("--source-root", type=Path, default=None)
    prepare_vt.add_argument("--no-build-preprocess1", action="store_true")
    prepare_vt.add_argument("--overwrite", action="store_true")

    train_vt = subparsers.add_parser("train-visuotactile", help="Train a visuotactile policy locally")
    train_vt.add_argument("--workspace-config", default="config/workspace.yaml")
    train_vt.add_argument("--task-name", required=True)
    train_vt.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    train_vt.add_argument("--dataset-dir", type=Path, default=None)
    train_vt.add_argument("--backend-dataset-root", type=Path, default=None)
    train_vt.add_argument("--raw-run-dir", type=Path, default=None)
    train_vt.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    train_vt.add_argument("--checkpoint-dir", type=Path, default=None)
    train_vt.add_argument("--run-name", default=None)
    train_vt.add_argument("--seed", type=int, default=0)
    train_vt.add_argument("--device", default="cuda")
    train_vt.add_argument("--batch-size", type=int, default=None)
    train_vt.add_argument("--epochs", type=int, default=None)
    train_vt.add_argument("--learning-rate", type=float, default=None)
    train_vt.add_argument("--wandb-mode", default=None)
    train_vt.add_argument("--no-prepare", action="store_true")
    train_vt.add_argument("--overwrite", action="store_true")
    train_vt.add_argument("--resume", action="store_true", help="Resume from the latest milestone/best checkpoint.")
    train_vt.add_argument("--dry-run", action="store_true")
    train_vt.add_argument("--extra-arg", dest="extra_args", action="append", default=[])
    train_vt.add_argument("--command", dest="command_override", nargs=argparse.REMAINDER, default=None)

    remote_vt = subparsers.add_parser("remote-train-visuotactile", help="Sync and train a visuotactile policy on a remote PC")
    remote_vt.add_argument("--workspace-config", default="config/workspace.yaml")
    remote_vt.add_argument("--task-name", required=True)
    remote_vt.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    remote_vt.add_argument("--dataset-dir", type=Path, default=None)
    remote_vt.add_argument("--backend-dataset-root", type=Path, default=None)
    remote_vt.add_argument("--raw-run-dir", type=Path, default=None)
    remote_vt.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    remote_vt.add_argument("--checkpoint-dir", type=Path, default=None)
    remote_vt.add_argument("--run-name", default=None)
    remote_vt.add_argument("--seed", type=int, default=0)
    remote_vt.add_argument("--device", default="cuda")
    remote_vt.add_argument("--batch-size", type=int, default=None)
    remote_vt.add_argument("--epochs", type=int, default=None)
    remote_vt.add_argument("--learning-rate", type=float, default=None)
    remote_vt.add_argument("--wandb-mode", default=None)
    remote_vt.add_argument("--overwrite", action="store_true")
    remote_vt.add_argument("--resume", action="store_true", help="Resume from the latest milestone/best checkpoint.")
    remote_vt.add_argument("--remote", required=True)
    remote_vt.add_argument("--remote-root", required=True)
    remote_vt.add_argument("--ssh-port", type=int, default=None)
    remote_vt.add_argument("--ssh-key", type=Path, default=None)
    remote_vt.add_argument("--remote-python", default="python")
    remote_vt.add_argument("--no-sync-code", action="store_true")
    remote_vt.add_argument("--no-sync-dataset", action="store_true")
    remote_vt.add_argument("--no-download", action="store_true")
    remote_vt.add_argument("--dry-run", action="store_true")
    remote_vt.add_argument("--extra-arg", dest="extra_args", action="append", default=[])
    remote_vt.add_argument("--command", dest="command_override", nargs=argparse.REMAINDER, default=None)

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

    if args.subcommand == "collect":
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

    if args.subcommand == "make-dataset":
        from .datasets import MakeDatasetConfig, make_common_dataset

        result = make_common_dataset(
            MakeDatasetConfig(
                collect_task_dir=args.collect_task_dir,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name,
                target_hz=args.target_hz,
                max_action_lead_sec=args.max_action_lead_sec,
                overwrite=args.overwrite,
            )
        )
        print(f"Common dataset: {result.output_dir}", flush=True)
        print(f"Episodes: {result.episode_count}", flush=True)
        print(f"Steps: {result.step_count}", flush=True)
        print(f"Manifest: {result.manifest_path}", flush=True)
        return

    if args.subcommand == "diagnose-cameras":
        from .sensors.diagnostics import CameraDiagnosticsConfig, diagnose_task_cameras

        task_config_path = _resolve_task_config_path(args.workspace_config, args.task_config, args.task)
        task = load_task_config(task_config_path, task_name_override=args.task)
        result = diagnose_task_cameras(
            CameraDiagnosticsConfig(
                task=task,
                output_root=args.output_root,
                duration_sec=args.duration_sec,
                include_rgb=not args.no_rgb,
                include_gelsight=not args.no_gelsight,
            )
        )
        print(f"Camera diagnostics report: {result.report_path}", flush=True)
        for name, stream in result.report.get("streams", {}).items():
            if stream.get("ok"):
                print(f"{name}: {stream.get('effective_hz', 0.0):.2f} Hz", flush=True)
            else:
                print(f"{name}: FAILED {stream.get('error', '')}", flush=True)
        return

    if args.subcommand == "gripper-testbed":
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

    if args.subcommand == "gripper-testbed-report":
        report = write_gripper_testbed_report(args.run_dir, args.output)
        print(str(report), flush=True)
        return

    if args.subcommand == "gripper-testbed-replay":
        try:
            import uvicorn
        except ImportError as exc:
            raise RuntimeError("Failed to import uvicorn for gripper-testbed-replay") from exc
        print(f"Gripper testbed replay UI: http://{args.host}:{args.port}/operator", flush=True)
        uvicorn.run(create_gripper_testbed_replay_app(args.run_dir), host=args.host, port=args.port)
        return

    if args.subcommand == "prepare-visuotactile":
        from .policies.visuotactile.image_preprocess import parse_crop_box
        from .policies.visuotactile.prepare import build_prepare_config_from_workspace, prepare_visuotactile_dataset

        workspace = load_workspace_config(args.workspace_config)
        config = build_prepare_config_from_workspace(
            workspace,
            task_name=args.task_name,
            model=args.model,
            raw_run_dir=args.raw_run_dir,
            output_dir=args.output_dir,
            dataset_name=args.dataset_name,
            preprocess1_profile=args.preprocess1_profile,
            target_hz=args.target_hz,
            image_size=args.image_size,
            val_ratio=args.val_ratio,
            val_episodes=args.val_episodes,
            overwrite=args.overwrite,
            build_preprocess1_if_missing=not args.no_build_preprocess1,
            canonical_size=args.canonical_size,
            gelsight_crop_box=parse_crop_box(args.gelsight_crop_box),
            gelsight_margin_fraction=args.gelsight_margin_fraction,
            source=args.source,
            source_root=args.source_root,
        )
        result = prepare_visuotactile_dataset(config)
        print(f"Prepared visuotactile dataset: {result.output_dir}", flush=True)
        print(f"Train episodes: {result.train_episodes}", flush=True)
        print(f"Val episodes: {result.val_episodes}", flush=True)
        print(f"Total steps: {result.total_steps}", flush=True)
        print(f"Manifest: {result.manifest_path}", flush=True)
        return

    if args.subcommand == "train-visuotactile":
        from .policies.visuotactile.train import config_from_args, train_visuotactile

        result = train_visuotactile(config_from_args(args))
        print(f"checkpoint_dir={result.checkpoint_dir}", flush=True)
        print(f"dataset_dir={result.dataset_dir}", flush=True)
        if result.backend_dataset_root is not None:
            print(f"backend_dataset_root={result.backend_dataset_root}", flush=True)
        print(f"manifest={result.manifest_path}", flush=True)
        return

    if args.subcommand == "remote-train-visuotactile":
        import json
        import shlex

        from .policies.visuotactile.remote import config_from_args, remote_train_visuotactile

        result = remote_train_visuotactile(config_from_args(args))
        print(
            json.dumps(
                {
                    "local_checkpoint_dir": str(result.local_checkpoint_dir),
                    "remote_checkpoint_dir": result.remote_checkpoint_dir,
                    "commands": [" ".join(shlex.quote(item) for item in command) for command in result.commands],
                },
                indent=2,
            ),
            flush=True,
        )
        return

    if args.subcommand == "run-policy":
        workspace = load_workspace_config(args.workspace_config)
        inference = load_inference_config(args.inference_config)
        if args.task or args.task_config:
            task_config_path = _resolve_task_config_path(args.workspace_config, args.task_config, args.task)
            task = load_task_config(task_config_path, task_name_override=args.task)
            inference = _merge_task_camera_config_into_inference(inference, task)
        policy_config = _resolve_run_policy_config(
            checkpoint=args.checkpoint,
            policy_config_path=args.policy_config,
            workspace=workspace,
            fallback_task_name=inference.task_name,
        )
        manifest_task_name = str(policy_config.config.get("task_name") or "")
        if manifest_task_name and inference.task_name == "policy_run" and not (args.task or args.task_config):
            inference = inference.model_copy(update={"task_name": manifest_task_name})
        policy = resolve_policy(policy_config, inference, workspace)
        log_buffer = OperatorLogBuffer(workspace.operator_ui.log_buffer_size)
        install_operator_logging(log_buffer, suppress_console_noise=workspace.operator_ui.enabled)
        run_name = args.run_name
        if run_name is None and args.resume:
            if args.checkpoint is None:
                raise SystemExit("run-policy --resume requires --checkpoint")
            run_name = _checkpoint_resume_run_name(args.checkpoint)
        PolicyRunner(
            workspace,
            inference,
            _build_controller(workspace),
            _load_calibration(workspace),
            policy,
            run_name=run_name,
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
        raise SystemExit("command requires either --task or --task-config")
    return Path(workspace_config_path).resolve().parent / "tasks" / f"{task_name}.yaml"


def _merge_task_camera_config_into_inference(inference, task):
    return inference.model_copy(
        update={
            "task_name": task.task_name,
            "modality": task.modality,
            "rgb_cameras": task.rgb_cameras,
            "gelsight": task.gelsight,
            "initial_eef_pose_xyz_rpy_deg": task.initial_eef_pose_xyz_rpy_deg,
            "initial_move_duration_sec": task.initial_move_duration_sec,
            "home_joint_positions_rad": task.home_joint_positions_rad,
            "home_joint_duration_sec": task.home_joint_duration_sec,
            "home_joint_tolerance_rad": task.home_joint_tolerance_rad,
            "home_joint_settle_timeout_sec": task.home_joint_settle_timeout_sec,
            "gripper_forever_closed": task.gripper_forever_closed,
            "rand_init_pose": task.rand_init_pose,
        }
    )


def _resolve_run_policy_config(
    *,
    checkpoint: Path | None,
    policy_config_path: str | None,
    workspace: WorkspaceSettings,
    fallback_task_name: str,
) -> PolicyConfig:
    if checkpoint is None:
        if policy_config_path is None:
            raise SystemExit("run-policy requires --checkpoint. --policy-config remains supported for legacy policies.")
        return load_policy_config(policy_config_path)

    checkpoint_dir, checkpoint_file = _resolve_checkpoint_reference(Path(checkpoint))
    manifest_path = checkpoint_dir / "policy_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint-first run-policy requires policy_manifest.json next to the checkpoint root: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_name = manifest.get("model")
    if not model_name:
        raise ValueError(f"Checkpoint manifest is missing model: {manifest_path}")
    task_name = str(manifest.get("task_name") or fallback_task_name)
    config = {
        "model": str(model_name),
        "task_name": task_name,
    }
    if checkpoint_file is not None:
        config["checkpoint_file"] = _checkpoint_file_reference(checkpoint_dir, checkpoint_file)

    if policy_config_path is not None:
        policy_config = load_policy_config(policy_config_path)
        merged = dict(policy_config.config)
        merged.update(config)
        return policy_config.model_copy(update={"checkpoint_path": checkpoint_dir, "config": merged})

    return PolicyConfig(type="visuotactile", checkpoint_path=checkpoint_dir, config=config)


def _resolve_checkpoint_reference(checkpoint: Path) -> tuple[Path, Path | None]:
    checkpoint = checkpoint.expanduser()
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint}")
    checkpoint = checkpoint.resolve()
    if checkpoint.is_file():
        for parent in [checkpoint.parent, *checkpoint.parents]:
            if (parent / "policy_manifest.json").is_file():
                return parent, checkpoint
        if checkpoint.parent.name == "checkpoints":
            return checkpoint.parent.parent, checkpoint
        return checkpoint.parent, checkpoint
    return checkpoint, None


def _checkpoint_file_reference(checkpoint_dir: Path, checkpoint_file: Path) -> str:
    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint_file = checkpoint_file.resolve()
    try:
        return checkpoint_file.relative_to(checkpoint_dir).as_posix()
    except ValueError:
        return checkpoint_file.name


def _checkpoint_resume_run_name(checkpoint: Path) -> str:
    checkpoint = checkpoint.expanduser()
    name = checkpoint.stem if checkpoint.suffix else checkpoint.name
    return _slugify_cli_run_name(name) or "checkpoint"


def _slugify_cli_run_name(value: str) -> str:
    import re

    return re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()


if __name__ == "__main__":
    main()
