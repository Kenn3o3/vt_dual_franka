from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from ...config import WorkspaceSettings, load_workspace_config
from .config import (
    DEFAULT_DATASET_NAME,
    DEFAULT_UPSTREAM_REPO,
    checkpoint_run_dir,
    default_prepared_dataset_dir,
    get_policy_spec,
    normalize_algorithm_name,
)


@dataclass(frozen=True)
class MPDTrainConfig:
    task_name: str
    algorithm: str
    prepared_dataset_dir: Path
    checkpoint_dir: Path
    upstream_repo_dir: Path = DEFAULT_UPSTREAM_REPO
    upstream_experiment: str | None = None
    device: str = "cuda"
    python: str = sys.executable
    epochs: int | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    seed: int | None = None
    num_modes: int | None = None
    alpha_vel: float | None = None
    swanlab_mode: str = "disabled"
    swanlab_entity: str = "motif"
    swanlab_project: str = "baseline-comparison"
    swanlab_group: str | None = None
    dry_run: bool = False
    extra_overrides: tuple[str, ...] = field(default_factory=tuple)


def build_train_config_from_workspace(
    workspace: WorkspaceSettings,
    *,
    task_name: str,
    algorithm: str,
    prepared_dataset_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    upstream_repo_dir: Path = DEFAULT_UPSTREAM_REPO,
    upstream_experiment: str | None = None,
    device: str = "cuda",
    python: str = sys.executable,
    epochs: int | None = None,
    batch_size: int | None = None,
    learning_rate: float | None = None,
    seed: int | None = None,
    num_modes: int | None = None,
    alpha_vel: float | None = None,
    swanlab_mode: str = "disabled",
    swanlab_entity: str = "motif",
    swanlab_project: str = "baseline-comparison",
    swanlab_group: str | None = None,
    dry_run: bool = False,
    extra_overrides: tuple[str, ...] = (),
) -> MPDTrainConfig:
    normalized_algorithm = normalize_algorithm_name(algorithm)
    spec = get_policy_spec(normalized_algorithm)
    if num_modes is not None and normalized_algorithm != "motif":
        raise ValueError("--num-modes only applies to motif")
    if alpha_vel is not None and normalized_algorithm != "motif":
        raise ValueError("--alpha-vel only applies to motif")
    return MPDTrainConfig(
        task_name=task_name,
        algorithm=normalized_algorithm,
        prepared_dataset_dir=prepared_dataset_dir or default_prepared_dataset_dir(workspace, task_name, DEFAULT_DATASET_NAME),
        checkpoint_dir=checkpoint_dir
        or checkpoint_run_dir(workspace, task_name=task_name, algorithm=normalized_algorithm, policy_name=spec.policy_name),
        upstream_repo_dir=upstream_repo_dir,
        upstream_experiment=upstream_experiment,
        device=device,
        python=python,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        num_modes=num_modes,
        alpha_vel=alpha_vel,
        swanlab_mode=swanlab_mode,
        swanlab_entity=swanlab_entity,
        swanlab_project=swanlab_project,
        swanlab_group=swanlab_group or spec.policy_name,
        dry_run=dry_run,
        extra_overrides=extra_overrides,
    )


def build_train_command(config: MPDTrainConfig) -> list[str]:
    spec = get_policy_spec(config.algorithm)
    experiment = config.upstream_experiment or config.task_name
    prepared_dataset_dir = Path(config.prepared_dataset_dir).expanduser().resolve()
    checkpoint_dir = Path(config.checkpoint_dir).expanduser().resolve()
    upstream_repo_dir = Path(config.upstream_repo_dir).expanduser().resolve()
    manifest = _load_manifest(prepared_dataset_dir)
    dt = float(manifest.get("dt", 0.1))
    command = [
        config.python,
        str(upstream_repo_dir / "scripts" / "train.py"),
        f"--config-name={spec.upstream_config_name(experiment)}",
        f"device={config.device}",
        "+fixed_split=True",
        f"+train_trajectory_dir={prepared_dataset_dir / 'train'}",
        f"+val_trajectory_dir={prepared_dataset_dir / 'val'}",
        f"hydra.run.dir={checkpoint_dir}",
        f"task_name={config.task_name}",
        f"method_name={spec.method_names[0]}",
        f"dataset_config.dt={dt}",
        "performance_metric=val_loss",
        "performance_direction=min",
        "workspace_config._target_=movement_primitive_diffusion.workspaces.dummy_workspace.DummyWorkspace",
        "eval_in_env_after_epochs=0",
        "num_trajectories_in_env=0",
        f"swanlab.mode={config.swanlab_mode}",
        f"swanlab.entity={config.swanlab_entity}",
        f"swanlab.project={config.swanlab_project}",
    ]
    if config.epochs is not None:
        command.append(f"epochs={config.epochs}")
    if config.batch_size is not None:
        command.append(f"data_loader_config.batch_size={config.batch_size}")
    if config.learning_rate is not None:
        command.append(f"agent_config.lr={config.learning_rate}")
    if config.seed is not None:
        command.append(f"seed={config.seed}")
    if config.swanlab_group:
        command.append(f"swanlab.group={config.swanlab_group}")
    if config.algorithm == "motif":
        if config.num_modes is not None:
            command.append(f"agent_config.process_batch_config.motif_handler_config.num_modes={config.num_modes}")
            command.append(f"agent_config.model_config.inner_model_config.motif_handler_config.num_modes={config.num_modes}")
        if config.alpha_vel is not None:
            command.append(f"agent_config.model_config.alpha_vel={config.alpha_vel}")
    command.extend(config.extra_overrides)
    return command


def run_training(config: MPDTrainConfig) -> int:
    command = build_train_command(config)
    print(" ".join(str(part) for part in command))
    if config.dry_run:
        return 0
    env = os.environ.copy()
    upstream_dir = Path(config.upstream_repo_dir).resolve()
    python_paths = [
        str(upstream_dir),
        str(upstream_dir / "dependencies" / "MP_PyTorch"),
    ]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return subprocess.run(command, cwd=upstream_dir, env=env, check=False).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MPD-family policies on prepared VT Franka datasets")
    parser.add_argument("--workspace-config", default="robot_workspace/config/workspace.yaml")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--algorithm", required=True, choices=["dp", "fm", "sfp", "mpd", "motif", "freqpolicy"])
    parser.add_argument("--prepared-dataset-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--upstream-repo-dir", type=Path, default=DEFAULT_UPSTREAM_REPO)
    parser.add_argument("--upstream-experiment", default=None)
    parser.add_argument("--device", default="cuda", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-modes", type=int, default=None)
    parser.add_argument("--alpha-vel", type=float, default=None)
    parser.add_argument("--swanlab-mode", default="disabled")
    parser.add_argument("--swanlab-entity", default="motif")
    parser.add_argument("--swanlab-project", default="baseline-comparison")
    parser.add_argument("--swanlab-group", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    workspace = load_workspace_config(args.workspace_config)
    config = build_train_config_from_workspace(
        workspace,
        task_name=args.task_name,
        algorithm=args.algorithm,
        prepared_dataset_dir=args.prepared_dataset_dir,
        checkpoint_dir=args.checkpoint_dir,
        upstream_repo_dir=args.upstream_repo_dir,
        upstream_experiment=args.upstream_experiment,
        device=args.device,
        python=args.python,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        num_modes=args.num_modes,
        alpha_vel=args.alpha_vel,
        swanlab_mode=args.swanlab_mode,
        swanlab_entity=args.swanlab_entity,
        swanlab_project=args.swanlab_project,
        swanlab_group=args.swanlab_group,
        dry_run=args.dry_run,
        extra_overrides=tuple(args.override),
    )
    raise SystemExit(run_training(config))


def _load_manifest(prepared_dataset_dir: Path) -> dict:
    path = Path(prepared_dataset_dir) / "dataset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Prepared MPD dataset manifest does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
