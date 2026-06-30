from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


POLICY_NAME = "DP"
CONFIG_GROUP = "dp"
REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = POLICY_ROOT

VALID_TASKS = {
    "grasp_classify",
    "insert_HDMI",
    "insert_HDMI_D1",
    "insert_HDMI_D2",
    "insert_hole",
    "insert_tube",
    "insert_tube_D1",
    "insert_tube_D2",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
    "put_bottle_in_shelf_D1",
    "put_bottle_in_shelf_D2",
}

_VAL_LOSS_CKPT_RE = re.compile(r"val_loss=([0-9]+(?:\.[0-9]+)?)\.ckpt$")
_CROSS_ENV_COMPILER_VARS = (
    "CC",
    "CXX",
    "CPPFLAGS",
    "CFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "LD",
)
_CROSS_ENV_CONDA_VARS = (
    "CONDA_DEFAULT_ENV",
    "CONDA_SHLVL",
    "CONDA_PROMPT_MODIFIER",
    "_CE_M",
    "_CE_CONDA",
)


def _is_hydra_invocation(argv: list[str]) -> bool:
    return any(arg.startswith("--config-name=") for arg in argv) or any(
        "=" in arg and not arg.startswith("--") for arg in argv
    )


def _line_buffer_stdio() -> None:
    sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
    sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)


def _register_hydra_resolvers() -> None:
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("get_ws_x_center", lambda task_name: 0.0, replace=True)
    OmegaConf.register_new_resolver("get_ws_y_center", lambda task_name: 0.0, replace=True)
    OmegaConf.register_new_resolver("get_ws_z_center", lambda task_name: 0.8, replace=True)
    OmegaConf.register_new_resolver("eval", eval, replace=True)


def run_hydra_backend() -> None:
    """Run the local DP Hydra backend from policy/DP/dp/config."""
    _line_buffer_stdio()
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))

    import hydra
    from omegaconf import OmegaConf
    from dp.workspace.base_workspace import BaseWorkspace

    _register_hydra_resolvers()

    @hydra.main(
        version_base=None,
        config_path=str(BACKEND_ROOT / "dp" / "config"),
    )
    def hydra_main(cfg: OmegaConf) -> None:
        OmegaConf.resolve(cfg)
        cls = hydra.utils.get_class(cfg._target_)
        workspace: BaseWorkspace = cls(cfg)
        workspace.run()

    hydra_main()


def timestamp_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def require_experiment_name(experiment_name: str | None = None) -> str:
    experiment = experiment_name or os.environ.get("EXPERIMENT_NAME")
    if not experiment:
        raise ValueError("experiment_name is required. Pass --experiment-name or set EXPERIMENT_NAME.")
    return experiment


def require_known_task(task_name: str) -> str:
    return task_name


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(payload)!r}")
    return payload


def load_named_config(group: str, name: str) -> dict[str, Any]:
    return load_yaml(REPO_ROOT / "configs" / group / f"{name}.yaml")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_run_metadata(path: Path, payload: dict[str, Any]) -> None:
    write_json(path / "run_metadata.json", payload)


def ensure_absent(path: Path, *, label: str) -> None:
    if path.exists():
        raise FileExistsError(f"{label} already exists: {path}")


def run_checked(command: list[str], *, env: dict[str, str | None] | None = None, cwd: Path | None = None) -> None:
    merged_env = os.environ.copy()
    if env:
        for key, value in env.items():
            if value is None:
                merged_env.pop(key, None)
            else:
                merged_env[key] = str(value)
    subprocess.run(command, cwd=str(cwd or REPO_ROOT), env=merged_env, check=True)


def cleanup_prepared_artifact(path: str | os.PathLike, *, label: str = "prepared data") -> bool:
    keep_value = os.environ.get("UNIVTAC_KEEP_PREPARED_DATA", "").strip().lower()
    if keep_value in {"1", "true", "yes", "on"}:
        print(f"[cleanup] keep {label}: {path} (UNIVTAC_KEEP_PREPARED_DATA={keep_value})")
        return False
    target = Path(path).expanduser()
    if not target.exists():
        return False
    resolved = target.resolve()
    if "prepared_data" not in resolved.parts:
        raise ValueError(f"Refusing to delete non-prepared-data path: {resolved}")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    print(f"[cleanup] removed {label}: {resolved}")
    return True


def conda_executable() -> str:
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).is_file():
        return conda_exe
    resolved = shutil.which("conda")
    if resolved:
        return resolved

    candidates: list[Path] = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        prefix = Path(conda_prefix)
        candidates.extend(
            [
                prefix.parent.parent / "bin" / "conda",
                prefix.parent / "condabin" / "conda",
            ]
        )
    executable = Path(sys.executable).resolve()
    conda_root_from_executable = executable.parents[3] if len(executable.parents) > 3 else executable.parent
    candidates.extend(
        [
            conda_root_from_executable / "bin" / "conda",
            conda_root_from_executable / "condabin" / "conda",
            Path.home() / "miniconda3" / "bin" / "conda",
            Path.home() / "miniconda3" / "condabin" / "conda",
            Path.home() / "miniforge3" / "bin" / "conda",
            Path.home() / "miniforge3" / "condabin" / "conda",
            Path.home() / "anaconda3" / "bin" / "conda",
            Path.home() / "anaconda3" / "condabin" / "conda",
            Path.home() / ".conda" / "bin" / "conda",
            Path.home() / ".conda" / "condabin" / "conda",
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return "conda"


def raw_data_root(experiment_name: str) -> Path:
    return REPO_ROOT / "data" / experiment_name


def config_name_with_demo(config_name: str, n_demo: int) -> str:
    return f"{config_name}_demo{n_demo}"


def policy_cache_dir(experiment_name: str, task_name: str, full_config_name: str) -> Path:
    return REPO_ROOT / "prepared_data" / experiment_name / task_name / POLICY_NAME / full_config_name / "cache"


def checkpoint_run_dir(
    experiment_name: str,
    task_name: str,
    full_config_name: str,
    run_id: str,
) -> Path:
    return REPO_ROOT / "checkpoints" / experiment_name / task_name / POLICY_NAME / full_config_name / run_id


def tactile_encoder_ckpt() -> Path:
    override = os.environ.get("UNIVTAC_TACTILE_ENCODER_CKPT")
    if override:
        path = Path(override).expanduser()
        return path if path.is_absolute() else REPO_ROOT / path
    return REPO_ROOT / "checkpoints" / "UniVTAC_encoder" / "best.pth"


def resolve_repo_path(path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def best_checkpoint(run_dir: Path) -> Path | None:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    direct_best = ckpt_dir / "best.ckpt"
    if direct_best.is_file():
        return direct_best
    candidates: list[tuple[float, Path]] = []
    for path in ckpt_dir.glob("*.ckpt"):
        if path.name in {"latest.ckpt", "best.ckpt"}:
            continue
        match = _VAL_LOSS_CKPT_RE.search(path.name)
        if match is not None:
            candidates.append((float(match.group(1)), path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1].name))
    return candidates[0][1]


def normalization_mode_from_config(config: dict[str, Any]) -> tuple[str, str]:
    normalization_value = config.get("normalization", "default")
    normalization = ("on" if normalization_value else "off") if isinstance(normalization_value, bool) else str(normalization_value).lower()
    aliases = {
        "default": "default",
        "on": "default",
        "legacy": "legacy",
        "off": "off",
        "identity": "off",
    }
    if normalization not in aliases:
        raise ValueError(f"Unsupported normalization mode: {normalization}")
    return normalization, aliases[normalization]


def horizon_overrides_from_config(config: dict[str, Any]) -> tuple[int | None, int | None, int | None, int | None]:
    action_horizon = config.get("action_horizon")
    n_obs_steps = config.get("n_obs_steps")
    horizon = config.get("horizon")
    n_action_steps = config.get("n_action_steps")
    if action_horizon is not None:
        action_horizon = int(action_horizon)
        n_obs_steps = int(n_obs_steps) if n_obs_steps is not None else 2
        n_action_steps = int(n_action_steps) if n_action_steps is not None else action_horizon
        horizon = int(horizon) if horizon is not None else n_obs_steps - 1 + n_action_steps
    return (
        action_horizon,
        None if n_obs_steps is None else int(n_obs_steps),
        None if horizon is None else int(horizon),
        None if n_action_steps is None else int(n_action_steps),
    )


def clean_cross_conda_env(conda_env: str, gpu: str | None) -> dict[str, str | None]:
    env: dict[str, str | None] = {}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    current_env = os.environ.get("CONDA_DEFAULT_ENV")
    if current_env and current_env != conda_env:
        for key in _CROSS_ENV_COMPILER_VARS:
            env[key] = None
        for key in _CROSS_ENV_CONDA_VARS:
            env[key] = None
        for key in tuple(os.environ):
            if key.startswith("CONDA_PREFIX"):
                env[key] = None
    return env


def train_dp(
    *,
    experiment_name: str,
    task_name: str,
    config_name: str,
    n_demo: int,
    policy_name: str = POLICY_NAME,
    config_group: str = CONFIG_GROUP,
    run_id: str | None = None,
    seed: int = 0,
    gpu: str | None = None,
    conda_env: str = "isp",
    batch_size: int | None = None,
    val_batch_size: int | None = None,
    num_epochs: int | None = None,
    wandb_mode: str | None = None,
    allow_existing_run_dir: bool = False,
) -> Path:
    require_known_task(task_name)
    config = load_named_config(config_group, config_name)
    hydra_config = str(config["hydra_config"])
    normalization, normalization_mode = normalization_mode_from_config(config)
    action_horizon, n_obs_steps, horizon, n_action_steps = horizon_overrides_from_config(config)
    hydra_overrides = config.get("hydra_overrides", []) or []
    if not isinstance(hydra_overrides, list):
        raise TypeError(f"Expected hydra_overrides list in configs/{CONFIG_GROUP}/{config_name}.yaml")

    dataset_root = raw_data_root(experiment_name)
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Raw dataset root not found: {dataset_root}")

    resolved_run_id = run_id or timestamp_id()
    full_config_name = config_name_with_demo(config_name, n_demo)
    run_dir = (
        REPO_ROOT
        / "checkpoints"
        / experiment_name
        / task_name
        / policy_name
        / full_config_name
        / resolved_run_id
    )
    if not allow_existing_run_dir:
        ensure_absent(run_dir, label=f"{policy_name} checkpoint run directory")
    cache_dir = REPO_ROOT / "prepared_data" / experiment_name / task_name / policy_name / full_config_name / "cache"

    env = clean_cross_conda_env(conda_env, gpu)
    effective_wandb_mode = wandb_mode or os.environ.get("WANDB_MODE", "offline")
    command = [
        conda_executable(),
        "run",
        "--live-stream",
        "-n",
        conda_env,
        "python",
        "train.py",
        f"--config-name={hydra_config}",
        f"task_name={task_name}",
        f"n_demo={n_demo}",
        f"dataset_root={dataset_root}",
        "data_split=clean",
        f"exp_name={experiment_name}",
        f"+task.dataset.cache_dir={cache_dir}",
        "training.device=cuda:0",
        f"training.seed={seed}",
        f"hydra.run.dir={run_dir}",
        f"hydra.sweep.dir={run_dir}",
        f"multi_run.run_dir={run_dir}",
        f"task.dataset.normalization_mode={normalization_mode}",
    ]
    if n_obs_steps is not None:
        command.extend([f"n_obs_steps={n_obs_steps}", f"dataset_obs_steps={n_obs_steps}"])
    if horizon is not None:
        command.append(f"horizon={horizon}")
    if n_action_steps is not None:
        command.append(f"n_action_steps={n_action_steps}")
    if batch_size is not None:
        command.extend(
            [
                f"dataloader.batch_size={batch_size}",
                f"val_dataloader.batch_size={val_batch_size or batch_size}",
            ]
        )
    elif val_batch_size is not None:
        command.append(f"val_dataloader.batch_size={val_batch_size}")
    if num_epochs is not None:
        command.append(f"training.num_epochs={num_epochs}")
    command.extend(
        [
            "training.checkpoint_every=1",
            "checkpoint.topk.k=1",
            "checkpoint.topk.format_str=best.ckpt",
            "checkpoint.save_last_ckpt=False",
            "checkpoint.save_last_snapshot=False",
        ]
    )
    command.extend(
        [
            f"logging.mode={effective_wandb_mode}",
            f"logging.project={task_name}",
            f"logging.name={full_config_name}",
        ]
    )

    if "vision_backbone_path" in config:
        vision_path = resolve_repo_path(str(config["vision_backbone_path"]))
        if not vision_path.is_file():
            raise FileNotFoundError(f"ViTAL vision encoder checkpoint not found: {vision_path}")
        command.append(f"policy.obs_encoder.vision_backbone_path={vision_path}")
    if "gelsight_backbone_path" in config:
        gelsight_path = resolve_repo_path(str(config["gelsight_backbone_path"]))
        if not gelsight_path.is_file():
            raise FileNotFoundError(f"ViTAL tactile encoder checkpoint not found: {gelsight_path}")
        command.append(f"policy.obs_encoder.gelsight_backbone_path={gelsight_path}")

    command.extend(str(override) for override in hydra_overrides)

    run_checked(command, env=env, cwd=BACKEND_ROOT)
    best_ckpt_path = None
    best_ckpt = best_checkpoint(run_dir)
    if best_ckpt is not None:
        best_ckpt_path = run_dir / "checkpoints" / "best.ckpt"
        if best_ckpt.resolve() != best_ckpt_path.resolve():
            shutil.copy2(best_ckpt, best_ckpt_path)

    metadata = {
            "experiment_name": experiment_name,
            "policy_name": policy_name,
            "task_name": task_name,
            "config_name": full_config_name,
            "config_base_name": config_name,
            "run_id": resolved_run_id,
            "hydra_config": hydra_config,
            "config_group": config_group,
            "n_demo": n_demo,
            "normalization": normalization,
            "normalization_mode": normalization_mode,
            "action_horizon": action_horizon,
            "horizon": horizon,
            "n_action_steps": n_action_steps,
            "n_obs_steps": n_obs_steps,
            "hydra_overrides": hydra_overrides,
            "seed": seed,
            "batch_size": batch_size,
            "val_batch_size": val_batch_size or batch_size,
            "num_epochs": num_epochs,
            "wandb_mode": effective_wandb_mode,
            "wandb_project": task_name,
            "wandb_run_name": full_config_name,
            "dataset_root": str(dataset_root),
            "cache_dir": str(cache_dir),
            "checkpoint_dir": str(run_dir),
            "best_ckpt_path": str(best_ckpt_path) if best_ckpt_path is not None else None,
    }
    write_run_metadata(run_dir, metadata)
    if best_ckpt_path is not None:
        cleanup_prepared_artifact(cache_dir.parent, label=f"{policy_name} prepared cache")
    return run_dir


def clean_main() -> None:
    parser = argparse.ArgumentParser(description="Train standalone DP-family policies")
    parser.add_argument("task_name", type=str)
    parser.add_argument("--experiment-name", type=str, default=None)
    parser.add_argument("--config-name", type=str, required=True)
    parser.add_argument("--n-demo", type=int, required=True)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=str, default=None)
    parser.add_argument("--conda-env", type=str, default="isp")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--val-batch-size", type=int, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--allow-existing-run-dir", action="store_true")
    args = parser.parse_args()

    train_dp(
        experiment_name=require_experiment_name(args.experiment_name),
        task_name=args.task_name,
        config_name=args.config_name,
        n_demo=args.n_demo,
        run_id=args.run_id,
        seed=args.seed,
        gpu=args.gpu,
        conda_env=args.conda_env,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        num_epochs=args.num_epochs,
        wandb_mode=args.wandb_mode,
        allow_existing_run_dir=args.allow_existing_run_dir,
    )


main = clean_main


if __name__ == "__main__":
    if _is_hydra_invocation(sys.argv[1:]):
        run_hydra_backend()
    else:
        clean_main()
