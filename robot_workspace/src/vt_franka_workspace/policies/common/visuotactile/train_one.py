from __future__ import annotations

import argparse
import os
from pathlib import Path

from ....config import load_workspace_config
from .config import DEFAULT_DATASET_NAME, get_model_spec
from .train import TrainVisuotactileConfig, train_visuotactile


REPO_ROOT = Path(__file__).resolve().parents[6]
ROBOT_WORKSPACE_ROOT = REPO_ROOT / "robot_workspace"
DATA_ROOT = ROBOT_WORKSPACE_ROOT / "data"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Short launcher for one real VT_Franka visuotactile training run.")
    parser.add_argument("task_name")
    parser.add_argument("--model", default=None)
    parser.add_argument("--config-name", default=None, help="Alias for --model, accepted for UniVTAC-style command shape.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-id", default=None, help="Recorded for naming parity; checkpoints still use the fixed model dir.")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-batch-size", type=int, default=None, help="Accepted for command parity; backend uses the same batch size.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--wandb-mode", default="offline")
    parser.add_argument("--workspace-config", default="robot_workspace/config/workspace.yaml")
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest milestone/best checkpoint.")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    workspace = load_workspace_config(args.workspace_config)
    model_name = args.config_name or args.model
    if not model_name:
        raise SystemExit("one of --config-name or --model is required")
    model = get_model_spec(model_name).name
    dataset_dir = DATA_ROOT / "prepared" / args.task_name / "visuotactile" / args.dataset_name / model
    checkpoint_dir = DATA_ROOT / "checkpoints" / args.task_name / model
    source_root = args.source_root or DATA_ROOT / "preprocess1" / args.task_name / args.dataset_name
    prepare_command = [
        "-m",
        "vt_franka_workspace.policies.visuotactile.prepare",
        "--workspace-config",
        args.workspace_config,
        "--task-name",
        args.task_name,
        "--model",
        model,
        "--output-dir",
        str(dataset_dir),
        "--source",
        "preprocess1",
        "--source-root",
        str(source_root),
    ]
    if args.overwrite and not args.resume:
        prepare_command.append("--overwrite")
    if args.dry_run:
        print("prepare_command=python " + " ".join(prepare_command))
    else:
        import subprocess
        import sys

        subprocess.run([sys.executable, *prepare_command], check=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    result = train_visuotactile(
        TrainVisuotactileConfig(
            workspace=workspace,
            task_name=args.task_name,
            model=model,
            dataset_dir=dataset_dir,
            backend_dataset_root=checkpoint_dir / "backend_dataset",
            dataset_name=args.dataset_name,
            checkpoint_dir=checkpoint_dir,
            seed=args.seed,
            device="cuda:0",
            batch_size=args.batch_size,
            epochs=args.epochs,
            wandb_mode=args.wandb_mode,
            overwrite=args.overwrite and not args.resume,
            resume=args.resume,
            prepare_if_missing=False,
            dry_run=args.dry_run,
        )
    )
    print(f"checkpoint_dir={result.checkpoint_dir}")
    print(f"dataset_dir={result.dataset_dir}")
    if result.backend_dataset_root is not None:
        print(f"backend_dataset_root={result.backend_dataset_root}")
    print("command=" + " ".join(str(part) for part in result.command))


if __name__ == "__main__":
    main()
