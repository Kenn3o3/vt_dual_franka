from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


POLICY_NAME = "ACT"
CONFIG_GROUP = "act"
REPO_ROOT = Path(__file__).resolve().parents[2]
POLICY_ROOT = Path(__file__).resolve().parent

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


def raw_task_dir(experiment_name: str, task_name: str) -> Path:
    return REPO_ROOT / "data" / experiment_name / task_name


def raw_hdf5_dir(experiment_name: str, task_name: str) -> Path:
    return raw_task_dir(experiment_name, task_name) / "hdf5"


def prepared_data_dir(experiment_name: str, task_name: str, full_config_name: str) -> Path:
    return REPO_ROOT / "prepared_data" / experiment_name / task_name / POLICY_NAME / full_config_name


def checkpoint_run_dir(
    experiment_name: str,
    task_name: str,
    full_config_name: str,
    run_id: str,
) -> Path:
    return REPO_ROOT / "checkpoints" / experiment_name / task_name / POLICY_NAME / full_config_name / run_id


def config_name_with_demo(config_name: str, n_demo: int) -> str:
    return f"{config_name}_demo{n_demo}"


def count_hdf5_files(hdf5_dir: Path) -> int:
    return len(sorted(hdf5_dir.glob("*.hdf5"), key=lambda path: int(path.stem)))


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


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def prepare_act_data(
    *,
    experiment_name: str,
    task_name: str,
    config_name: str,
    config: dict[str, Any],
    n_demo: int,
) -> Path:
    representation = str(config.get("representation", "ee"))
    if representation != "ee":
        raise ValueError(f"Final ACT pipeline supports EE representation only, got {representation!r}")

    input_dir = raw_task_dir(experiment_name, task_name)
    hdf5_dir = raw_hdf5_dir(experiment_name, task_name)
    available = count_hdf5_files(hdf5_dir)
    resolved_n_demo = n_demo if n_demo > 0 else available
    if resolved_n_demo > available:
        raise ValueError(f"Requested n_demo={resolved_n_demo}, but only found {available} episodes in {hdf5_dir}")

    full_config_name = config_name_with_demo(config_name, resolved_n_demo)
    output_dir = prepared_data_dir(experiment_name, task_name, full_config_name)
    if output_dir.is_dir() and len(list(output_dir.glob("episode_*.hdf5"))) >= resolved_n_demo:
        return output_dir
    ensure_absent(output_dir, label="prepared ACT data directory")

    run_checked(
        [
            sys.executable,
            str(POLICY_ROOT / "process_data_ee.py"),
            task_name,
            "clean",
            str(resolved_n_demo),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--skip-sim-task-configs",
        ],
        cwd=REPO_ROOT,
    )
    write_json(
        output_dir / "prepared_data.json",
        {
            "experiment_name": experiment_name,
            "policy_name": POLICY_NAME,
            "task_name": task_name,
            "config_name": full_config_name,
            "config_base_name": config_name,
            "representation": representation,
            "n_demo": resolved_n_demo,
            "num_episodes": resolved_n_demo,
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
        },
    )
    return output_dir


def write_train_config(
    path: Path,
    *,
    base_config_name: str,
    clean_config: dict[str, Any],
    batch_size: int | None,
    num_steps: int | None,
    gpu: str | None,
) -> dict[str, Any]:
    base_path = POLICY_ROOT / f"{base_config_name}.yml"
    if not base_path.is_file():
        raise FileNotFoundError(f"ACT base train config not found: {base_path}")
    payload = load_yaml(base_path)
    payload.update(
        {
            "temporal_agg": False,
            "state_dim": int(clean_config.get("state_dim", payload.get("state_dim", 8))),
            "chunk_size": int(clean_config.get("chunk_size", payload.get("chunk_size", 8))),
            "camera_names": list(clean_config.get("camera_names", payload.get("camera_names", ["cam_wrist"]))),
            "tactile_names": list(clean_config.get("tactile_names", payload.get("tactile_names", []))),
            "device": "cuda:0" if gpu is not None else str(clean_config.get("device", payload.get("device", "cuda:0"))),
        }
    )
    payload["batch_size"] = int(batch_size or clean_config.get("batch_size", payload.get("batch_size", 64)))
    payload["num_steps"] = int(num_steps or clean_config.get("num_steps", payload.get("num_steps", 4000)))

    tactile_ckpt = resolve_path(clean_config.get("tactile_ckpt"))
    if payload["tactile_names"]:
        if tactile_ckpt is None:
            raise ValueError("ACT/univtac requires tactile_ckpt")
        if not tactile_ckpt.is_file():
            raise FileNotFoundError(f"UniVTAC tactile encoder checkpoint not found: {tactile_ckpt}")
        payload["tactile_ckpt"] = str(tactile_ckpt)
    else:
        payload["tactile_ckpt"] = ""
        payload["lr_tactile_backbone"] = 0.0

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return payload


def train_act(
    *,
    experiment_name: str,
    task_name: str,
    config_name: str,
    n_demo: int,
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
    del conda_env, wandb_mode
    require_known_task(task_name)
    config = load_named_config(CONFIG_GROUP, config_name)
    full_config_name = config_name_with_demo(config_name, n_demo)
    dataset_dir = prepare_act_data(
        experiment_name=experiment_name,
        task_name=task_name,
        config_name=config_name,
        config=config,
        n_demo=n_demo,
    )
    num_episodes = len(sorted(dataset_dir.glob("episode_*.hdf5")))
    if num_episodes <= 0:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in prepared dataset: {dataset_dir}")

    resolved_run_id = run_id or timestamp_id()
    run_dir = checkpoint_run_dir(experiment_name, task_name, full_config_name, resolved_run_id)
    if not allow_existing_run_dir:
        ensure_absent(run_dir, label="ACT checkpoint run directory")
    run_dir.mkdir(parents=True, exist_ok=True)

    env: dict[str, str | None] = {}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu

    requested_steps = num_epochs
    base_config_name = str(config.get("train_config", "train_config_ee"))
    with tempfile.TemporaryDirectory(prefix="train_act_cfg_") as temp_dir:
        config_path = Path(temp_dir) / f"{config_name}.yml"
        train_config = write_train_config(
            config_path,
            base_config_name=base_config_name,
            clean_config=config,
            batch_size=batch_size,
            num_steps=requested_steps,
            gpu=gpu,
        )
        run_checked(
            [
                sys.executable,
                "-m",
                "policy.ACT.imitate_episodes",
                "--ckpt_dir",
                str(run_dir),
                "--task_name",
                task_name,
                "--config_path",
                str(config_path),
                "--seed",
                str(seed),
                "--dataset_dir",
                str(dataset_dir),
                "--num_episodes",
                str(num_episodes),
            ],
            env=env,
            cwd=REPO_ROOT,
        )

    best_src = run_dir / "policy_best.ckpt"
    if best_src.is_file():
        link_or_copy(best_src, run_dir / "best.ckpt")
    elif not (run_dir / "best.ckpt").is_file():
        raise FileNotFoundError(f"ACT training did not produce policy_best.ckpt in {run_dir}")

    write_json(
        run_dir / "args.json",
        {
            "policy_config": train_config,
            "policy_name": POLICY_NAME,
            "train_config": base_config_name,
            "config_name": full_config_name,
        },
    )
    write_json(run_dir / "clean_train_config.json", train_config)
    write_json(
        run_dir / "run_metadata.json",
        {
            "experiment_name": experiment_name,
            "policy_name": POLICY_NAME,
            "task_name": task_name,
            "config_name": full_config_name,
            "config_base_name": config_name,
            "run_id": resolved_run_id,
            "n_demo": n_demo,
            "num_episodes": num_episodes,
            "seed": seed,
            "batch_size": int(train_config.get("batch_size", batch_size or 64)),
            "val_batch_size": val_batch_size,
            "num_epochs": requested_steps,
            "num_steps": int(train_config.get("num_steps", requested_steps or 4000)),
            "chunk_size": int(train_config.get("chunk_size", 8)),
            "train_config": base_config_name,
            "deploy_config": str(config.get("deploy_config", "policy/ACT/deploy_ee")),
            "dataset_dir": str(dataset_dir),
            "checkpoint_dir": str(run_dir),
            "best_ckpt_path": str(run_dir / "best.ckpt"),
            "temporal_agg": False,
        },
    )
    cleanup_prepared_artifact(dataset_dir, label="ACT prepared dataset")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Train original UniVTAC ACT architecture in policy/ACT")
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
    parser.add_argument("--num-epochs", type=int, default=None, help="Mapped to ACT num_steps.")
    parser.add_argument("--wandb-mode", type=str, default=None)
    parser.add_argument("--allow-existing-run-dir", action="store_true")
    args = parser.parse_args()

    train_act(
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


if __name__ == "__main__":
    main()
