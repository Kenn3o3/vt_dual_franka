from __future__ import annotations

import argparse
import os
import json
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ....config import load_workspace_config
from .config import (
    DEFAULT_DATASET_NAME,
    MODEL_SPECS,
    default_checkpoint_dir,
    default_prepared_dataset_dir,
    get_model_spec,
    normalize_model_name,
)
from .train import TrainVisuotactileConfig, train_visuotactile


REPO_ROOT = Path(__file__).resolve().parents[6]
WORKSPACE_ROOT = REPO_ROOT
DEFAULT_MODELSCOPE_REPO_TYPE = "dataset"
DEFAULT_MODELSCOPE_REVISION = "master"
INFERENCE_BUNDLE_PREFIX = "inference_bundle"
DEFAULT_INFERENCE_BUNDLE_EXCLUDES = (
    "backend_dataset/***",
    "backend_dataset/**",
    "wandb/***",
    "wandb/**",
    "*.hdf5",
    "*.zarr",
    "*.zarr/***",
    "*.zarr.zip",
    "*.mp4",
    "__pycache__/***",
    "__pycache__/**",
)


@dataclass(frozen=True)
class RemoteTrainConfig:
    local_train: TrainVisuotactileConfig
    remote: str
    remote_root: str
    ssh_port: int | None = None
    ssh_key: Path | None = None
    remote_python: str = "python"
    sync_code: bool = True
    sync_dataset: bool = True
    download: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class RemoteTrainResult:
    local_checkpoint_dir: Path
    remote_checkpoint_dir: str
    commands: list[list[str]]


def remote_train_visuotactile(config: RemoteTrainConfig) -> RemoteTrainResult:
    local_train = config.local_train
    spec = get_model_spec(local_train.model)
    dataset_dir = _remote_train_dataset_source(local_train)
    remote_prepare = _remote_should_prepare_from_dataset(local_train, dataset_dir)
    checkpoint_dir = local_train.checkpoint_dir or default_checkpoint_dir(
        local_train.workspace,
        task_name=local_train.task_name,
        model=spec.name,
        run_name=local_train.run_name,
    )
    remote_root = config.remote_root.rstrip("/")
    rel_dataset = _relative_to_workspace(dataset_dir)
    rel_checkpoint = _relative_to_workspace(checkpoint_dir)
    remote_dataset = f"{remote_root}/{rel_dataset.as_posix()}"
    remote_checkpoint = f"{remote_root}/{rel_checkpoint.as_posix()}"
    remote_backend_dataset = f"{remote_checkpoint.rstrip('/')}/backend_dataset"

    commands: list[list[str]] = []
    if config.sync_code:
        commands.append(_rsync_code_command(config))
    if config.sync_dataset:
        commands.append(_rsync_path_command(config, dataset_dir, remote_dataset))

    remote_pythonpath = "robot_workspace/src:shared/src:${PYTHONPATH:-}"
    remote_train_command = [
        "cd",
        remote_root,
        "&&",
        f"PYTHONPATH={remote_pythonpath}",
        config.remote_python,
        "-m",
        "vt_franka_workspace.policies.visuotactile.train",
        "--workspace-config",
        "robot_workspace/config/workspace.yaml",
        "--task-name",
        local_train.task_name,
        "--model",
        spec.name,
        "--dataset-dir",
        remote_dataset,
        "--checkpoint-dir",
        remote_checkpoint,
        "--backend-dataset-root",
        remote_backend_dataset,
        "--seed",
        str(local_train.seed),
        "--device",
        local_train.device,
    ]
    if not remote_prepare:
        remote_train_command.append("--no-prepare")
    if local_train.batch_size is not None:
        remote_train_command.extend(["--batch-size", str(local_train.batch_size)])
    if local_train.epochs is not None:
        remote_train_command.extend(["--epochs", str(local_train.epochs)])
    if local_train.learning_rate is not None:
        remote_train_command.extend(["--learning-rate", str(local_train.learning_rate)])
    if local_train.wandb_mode is not None:
        remote_train_command.extend(["--wandb-mode", str(local_train.wandb_mode)])
    if local_train.overwrite:
        remote_train_command.append("--overwrite")
    if local_train.resume:
        remote_train_command.append("--resume")
    if local_train.command_override:
        remote_train_command.extend(["--command", "--", *local_train.command_override])
    commands.append([*_ssh_base(config), config.remote, _shell_join_remote(remote_train_command)])

    if config.download:
        commands.append(_rsync_download_command(config, remote_checkpoint, checkpoint_dir))

    if config.dry_run:
        return RemoteTrainResult(
            local_checkpoint_dir=checkpoint_dir,
            remote_checkpoint_dir=remote_checkpoint,
            commands=commands,
        )

    if config.sync_dataset and not remote_prepare and not (dataset_dir / "dataset_manifest.json").exists():
        train_visuotactile(
            TrainVisuotactileConfig(
                workspace=local_train.workspace,
                task_name=local_train.task_name,
                model=spec.name,
                dataset_dir=dataset_dir,
                backend_dataset_root=local_train.backend_dataset_root,
                raw_run_dir=local_train.raw_run_dir,
                dataset_name=local_train.dataset_name,
                checkpoint_dir=checkpoint_dir,
                run_name=local_train.run_name,
                seed=local_train.seed,
                device=local_train.device,
                batch_size=local_train.batch_size,
                epochs=local_train.epochs,
                learning_rate=local_train.learning_rate,
                wandb_mode=local_train.wandb_mode,
                overwrite=local_train.overwrite,
                prepare_if_missing=local_train.prepare_if_missing,
                dry_run=False,
                command_override=["true"],
            )
        )

    for command in commands:
        subprocess.run(command, check=True)

    return RemoteTrainResult(
        local_checkpoint_dir=checkpoint_dir,
        remote_checkpoint_dir=remote_checkpoint,
        commands=commands,
    )


def _remote_train_dataset_source(local_train: TrainVisuotactileConfig) -> Path:
    if local_train.dataset_dir is not None:
        return Path(local_train.dataset_dir)
    return _default_common_dataset_dir(local_train)


def _default_common_dataset_dir(local_train: TrainVisuotactileConfig) -> Path:
    collect_root = Path(local_train.workspace.recording.collect_root)
    data_root = collect_root.parent if collect_root.name == "collect" else collect_root.parent
    return data_root / "datasets" / local_train.task_name / local_train.dataset_name


def _remote_should_prepare_from_dataset(local_train: TrainVisuotactileConfig, dataset_dir: Path) -> bool:
    if not local_train.prepare_if_missing:
        return False
    if local_train.dataset_dir is None and _looks_like_common_dataset_path(dataset_dir):
        return True
    return _is_common_dataset_dir(dataset_dir)


def _looks_like_common_dataset_path(path: Path) -> bool:
    parts = Path(path).parts
    return "datasets" in parts


def _is_common_dataset_dir(path: Path) -> bool:
    manifest_path = Path(path) / "dataset_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return str(payload.get("schema_version", "")).startswith("vt_franka_common_dataset")


def _relative_to_workspace(path: Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = (Path.cwd() / path)
    try:
        return path.absolute().relative_to(WORKSPACE_ROOT.absolute())
    except ValueError:
        pass
    try:
        return path.absolute().relative_to(WORKSPACE_ROOT.resolve())
    except ValueError:
        pass
    path = path.resolve()
    try:
        return path.relative_to(WORKSPACE_ROOT)
    except ValueError:
        return Path("external") / path.name


def _ssh_base(config: RemoteTrainConfig) -> list[str]:
    command = ["ssh", "-o", "StrictHostKeyChecking=accept-new", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=6"]
    if config.ssh_port is not None:
        command.extend(["-p", str(config.ssh_port)])
    if config.ssh_key is not None:
        command.extend(["-i", str(config.ssh_key.expanduser()), "-o", "IdentitiesOnly=yes"])
    return command


def _rsync_ssh(config: RemoteTrainConfig) -> str:
    return " ".join(shlex.quote(item) for item in _ssh_base(config))


def _rsync_code_command(config: RemoteTrainConfig) -> list[str]:
    remote_root = config.remote_root.rstrip("/")
    return [
        "rsync",
        "-a",
        "--info=progress2",
        "--partial",
        "--exclude",
        "__pycache__",
        "--exclude",
        "*.pyc",
        "--exclude",
        ".pytest_cache",
        "--exclude",
        "robot_workspace/data/collect",
        "--exclude",
        "robot_workspace/data/datasets",
        "--exclude",
        "robot_workspace/data/prepared",
        "--exclude",
        "robot_workspace/data/checkpoints",
        "--exclude",
        "data/collect",
        "--exclude",
        "data/prepared",
        "--exclude",
        "data/checkpoints",
        "-e",
        _rsync_ssh(config),
        f"{WORKSPACE_ROOT}/",
        f"{config.remote}:{remote_root}/",
    ]


def _rsync_path_command(config: RemoteTrainConfig, local_path: Path, remote_path: str) -> list[str]:
    return [
        "rsync",
        "-a",
        "--info=progress2",
        "--partial",
        "-e",
        _rsync_ssh(config),
        f"{Path(local_path).resolve()}/",
        f"{config.remote}:{remote_path.rstrip('/')}/",
    ]


def _rsync_download_command(config: RemoteTrainConfig, remote_path: str, local_path: Path) -> list[str]:
    return [
        "rsync",
        "-a",
        "--info=progress2",
        "--partial",
        "-e",
        _rsync_ssh(config),
        f"{config.remote}:{remote_path.rstrip('/')}/",
        f"{Path(local_path).resolve()}/",
    ]


def _shell_join_remote(tokens: list[str]) -> str:
    joined: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "&&":
            joined.append(token)
        elif token.startswith("PYTHONPATH="):
            joined.append(token)
        else:
            joined.append(shlex.quote(token))
        index += 1
    return " ".join(joined)


def _repo_checkpoint_task_root(repo_root: Path, task_name: str) -> Path:
    return Path(repo_root).expanduser().resolve() / "robot_workspace" / "data" / "checkpoints" / task_name


def _normalize_models(models: list[str]) -> list[str]:
    normalized = [normalize_model_name(model) for model in models]
    seen: set[str] = set()
    unique: list[str] = []
    for model in normalized:
        if model in seen:
            continue
        seen.add(model)
        unique.append(model)
    return unique


def _model_dir(checkpoint_task_root: Path, model: str) -> Path:
    return Path(checkpoint_task_root) / model


def _require_model_bundle(checkpoint_task_root: Path, model: str) -> Path:
    model_dir = _model_dir(checkpoint_task_root, model)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Missing checkpoint directory for {model}: {model_dir}")
    if not _has_model_checkpoint(model_dir):
        raise FileNotFoundError(f"Missing checkpoint artifact for {model}: {model_dir}")
    return model_dir


def _checkpoint_candidates(model_dir: Path) -> list[Path]:
    checkpoints_dir = Path(model_dir) / "checkpoints"
    candidates = [
        checkpoints_dir / "best.ckpt",
        Path(model_dir) / "best.ckpt",
        Path(model_dir) / "policy_best.ckpt",
        checkpoints_dir / "latest.ckpt",
    ]
    if checkpoints_dir.is_dir():
        candidates.extend(sorted(checkpoints_dir.glob("epoch=*.ckpt")))
    return [path for path in candidates if path.is_file()]


def _has_model_checkpoint(model_dir: Path) -> bool:
    return bool(_checkpoint_candidates(model_dir))


def _file_identity(path: Path) -> tuple[int, int]:
    stat = Path(path).stat()
    return stat.st_size, stat.st_mtime_ns


def _rsync_with_excludes(source: Path, destination: Path, *, excludes: tuple[str, ...], delete: bool = True) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = ["rsync", "-a", "--partial", "--info=progress2"]
    if delete:
        command.append("--delete")
    for pattern in excludes:
        command.extend(["--exclude", pattern])
    command.extend([f"{source.resolve()}/", f"{destination.resolve()}/"])
    subprocess.run(command, check=True)


def _build_stage_manifest(
    *,
    task_name: str,
    models: list[str],
    source_root: Path,
    stage_task_root: Path,
    path_in_repo: str,
    excludes: tuple[str, ...],
) -> Path:
    files: list[dict[str, object]] = []
    for path in sorted(stage_task_root.rglob("*")):
        if path.is_file():
            stat = path.stat()
            files.append(
                {
                    "path": path.relative_to(stage_task_root).as_posix(),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task": task_name,
        "models": models,
        "source_root": str(source_root),
        "path_in_repo": path_in_repo,
        "excludes": list(excludes),
        "files": files,
    }
    manifest_path = stage_task_root / "inference_bundle_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def _require_modelscope_token(token: str | None) -> str:
    resolved = token or os.environ.get("MODELSCOPE_TOKEN")
    if not resolved:
        raise RuntimeError("ModelScope token is required. Set MODELSCOPE_TOKEN or pass --token.")
    return resolved


@dataclass(frozen=True)
class PublishInferenceBundleConfig:
    task_name: str
    models: list[str]
    repo_id: str
    repo_type: str = DEFAULT_MODELSCOPE_REPO_TYPE
    revision: str = DEFAULT_MODELSCOPE_REVISION
    token: str | None = None
    repo_root: Path = REPO_ROOT
    checkpoints_root: Path | None = None
    stage_root: Path | None = None
    max_workers: int = 5
    create_repo: bool = True
    visibility: str = "private"
    endpoint: str | None = None
    dry_run: bool = False


@dataclass(frozen=True)
class FetchInferenceBundleConfig:
    task_name: str
    models: list[str]
    repo_id: str
    repo_type: str = DEFAULT_MODELSCOPE_REPO_TYPE
    revision: str = DEFAULT_MODELSCOPE_REVISION
    token: str | None = None
    repo_root: Path = REPO_ROOT
    ssd_root: Path = Path("/mnt/kenny_ssd/vt_franka")
    download_dir: Path | None = None
    destination_root: Path | None = None
    max_workers: int = 5
    endpoint: str | None = None
    link_repo_checkpoints: bool = True
    replace_existing_repo_path: bool = False
    delete_destination_extra_files: bool = True
    dry_run: bool = False


def publish_inference_bundle(config: PublishInferenceBundleConfig) -> dict[str, object]:
    models = _normalize_models(config.models)
    repo_root = Path(config.repo_root).expanduser().resolve()
    checkpoint_task_root = (
        Path(config.checkpoints_root).expanduser().resolve()
        if config.checkpoints_root is not None
        else _repo_checkpoint_task_root(repo_root, config.task_name)
    )
    path_in_repo = f"{INFERENCE_BUNDLE_PREFIX}/{config.task_name}"
    source_model_dirs = {model: _require_model_bundle(checkpoint_task_root, model) for model in models}
    upload_root: Path

    stage_task_root = (
        Path(config.stage_root).expanduser().resolve() / f"{config.task_name}_inference_bundle" / config.task_name
        if config.stage_root is not None
        else checkpoint_task_root
    )
    payload = {
        "task": config.task_name,
        "models": models,
        "source_root": str(checkpoint_task_root),
        "stage_root": str(stage_task_root),
        "path_in_repo": path_in_repo,
        "manifest": str(stage_task_root / "inference_bundle_manifest.json") if config.stage_root is not None else None,
        "repo_id": config.repo_id,
        "repo_type": config.repo_type,
        "revision": config.revision,
    }
    if config.dry_run:
        return {**payload, "uploaded": False}

    if config.stage_root is not None:
        for model, source_model_dir in source_model_dirs.items():
            before = {
                path.relative_to(source_model_dir).as_posix(): _file_identity(path)
                for path in _checkpoint_candidates(source_model_dir)
            }
            _rsync_with_excludes(
                source_model_dir,
                stage_task_root / model,
                excludes=DEFAULT_INFERENCE_BUNDLE_EXCLUDES,
                delete=True,
            )
            after = {
                path.relative_to(stage_task_root / model).as_posix(): _file_identity(path)
                for path in _checkpoint_candidates(stage_task_root / model)
            }
            if before != after:
                raise RuntimeError(
                    f"{source_model_dir} checkpoint files changed while staging. Stop the writer or rerun after training finishes."
                )
        manifest_path = _build_stage_manifest(
            task_name=config.task_name,
            models=models,
            source_root=checkpoint_task_root,
            stage_task_root=stage_task_root,
            path_in_repo=path_in_repo,
            excludes=DEFAULT_INFERENCE_BUNDLE_EXCLUDES,
        )
        upload_root = stage_task_root
    else:
        manifest_path = None
        upload_root = checkpoint_task_root

    token = _require_modelscope_token(config.token)
    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(token, endpoint=config.endpoint)
    if config.create_repo and not api.repo_exists(
        config.repo_id,
        repo_type=config.repo_type,
        token=token,
        endpoint=config.endpoint,
    ):
        api.create_repo(
            config.repo_id,
            repo_type=config.repo_type,
            token=token,
            visibility=config.visibility,
            exist_ok=True,
            endpoint=config.endpoint,
        )

    if config.stage_root is not None:
        api.upload_folder(
            repo_id=config.repo_id,
            repo_type=config.repo_type,
            folder_path=upload_root,
            path_in_repo=path_in_repo,
            token=token,
            revision=config.revision,
            max_workers=config.max_workers,
            ignore_patterns=list(DEFAULT_INFERENCE_BUNDLE_EXCLUDES),
            commit_message=f"publish {config.task_name} inference bundle",
        )
    else:
        for model, source_model_dir in source_model_dirs.items():
            api.upload_folder(
                repo_id=config.repo_id,
                repo_type=config.repo_type,
                folder_path=source_model_dir,
                path_in_repo=f"{path_in_repo}/{model}",
                token=token,
                revision=config.revision,
                max_workers=config.max_workers,
                ignore_patterns=list(DEFAULT_INFERENCE_BUNDLE_EXCLUDES),
                commit_message=f"publish {config.task_name} {model} inference bundle",
            )

    return {**payload, "uploaded": True}


def _find_downloaded_task_root(download_root: Path, task_name: str) -> Path:
    candidates = [
        download_root / INFERENCE_BUNDLE_PREFIX / task_name,
        download_root / task_name,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Downloaded bundle for {task_name!r} was not found under {download_root}. "
        f"Expected {candidates[0]}."
    )


def _install_repo_checkpoint_link(
    *,
    repo_root: Path,
    task_name: str,
    destination_task_root: Path,
    replace_existing_repo_path: bool,
) -> dict[str, str | None]:
    repo_checkpoint_parent = Path(repo_root).expanduser().resolve() / "robot_workspace" / "data" / "checkpoints"
    repo_checkpoint_parent.mkdir(parents=True, exist_ok=True)
    repo_task_path = repo_checkpoint_parent / task_name
    destination_task_root = destination_task_root.resolve()
    backup_path: Path | None = None

    if repo_task_path.is_symlink() or not repo_task_path.exists():
        if repo_task_path.is_symlink():
            repo_task_path.unlink()
        repo_task_path.symlink_to(destination_task_root, target_is_directory=True)
    elif repo_task_path.resolve() == destination_task_root:
        pass
    elif replace_existing_repo_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = repo_checkpoint_parent / f"{task_name}.backup.{timestamp}"
        repo_task_path.rename(backup_path)
        repo_task_path.symlink_to(destination_task_root, target_is_directory=True)
    else:
        print(
            f"WARNING: {repo_task_path} exists and is not a symlink. "
            "Leaving it unchanged; pass --replace-existing-repo-path to move it aside and link SSD checkpoints.",
            file=sys.stderr,
        )

    return {
        "repo_checkpoint_path": str(repo_task_path),
        "backup_path": str(backup_path) if backup_path is not None else None,
    }


def _default_destination_task_root(*, repo_root: Path, ssd_root: Path, task_name: str) -> Path:
    repo_checkpoint_parent = Path(repo_root).expanduser().resolve() / "robot_workspace" / "data" / "checkpoints"
    if repo_checkpoint_parent.exists():
        resolved_parent = repo_checkpoint_parent.resolve()
        try:
            resolved_parent.relative_to(Path(ssd_root).expanduser().resolve())
            return resolved_parent / task_name
        except ValueError:
            pass
    return Path(ssd_root).expanduser().resolve() / "data" / "checkpoints" / task_name


def fetch_inference_bundle(config: FetchInferenceBundleConfig) -> dict[str, object]:
    models = _normalize_models(config.models)
    repo_root = Path(config.repo_root).expanduser().resolve()
    ssd_root = Path(config.ssd_root).expanduser().resolve()
    download_dir = (
        Path(config.download_dir).expanduser().resolve()
        if config.download_dir is not None
        else ssd_root / "modelscope_downloads" / config.repo_id.split("/")[-1]
    )
    destination_task_root = (
        Path(config.destination_root).expanduser().resolve() / config.task_name
        if config.destination_root is not None
        else _default_destination_task_root(repo_root=repo_root, ssd_root=ssd_root, task_name=config.task_name)
    )
    allow_patterns = [f"{INFERENCE_BUNDLE_PREFIX}/{config.task_name}/inference_bundle_manifest.json"]
    allow_patterns.extend(f"{INFERENCE_BUNDLE_PREFIX}/{config.task_name}/{model}/**" for model in models)

    payload: dict[str, object] = {
        "task": config.task_name,
        "models": models,
        "repo_id": config.repo_id,
        "repo_type": config.repo_type,
        "revision": config.revision,
        "download_dir": str(download_dir),
        "destination": str(destination_task_root),
        "allow_patterns": allow_patterns,
    }
    if config.dry_run:
        return {**payload, "downloaded": False}

    token = _require_modelscope_token(config.token)
    from modelscope.hub.snapshot_download import dataset_snapshot_download

    download_root = Path(
        dataset_snapshot_download(
            config.repo_id,
            revision=config.revision,
            local_dir=str(download_dir),
            allow_patterns=allow_patterns,
            max_workers=config.max_workers,
            token=token,
            endpoint=config.endpoint,
        )
    )
    downloaded_task_root = _find_downloaded_task_root(download_root, config.task_name)
    for model in models:
        _require_model_bundle(downloaded_task_root, model)
        _rsync_with_excludes(
            downloaded_task_root / model,
            destination_task_root / model,
            excludes=(),
            delete=config.delete_destination_extra_files,
        )

    link_payload: dict[str, str | None] = {"repo_checkpoint_path": None, "backup_path": None}
    if config.link_repo_checkpoints:
        link_payload = _install_repo_checkpoint_link(
            repo_root=repo_root,
            task_name=config.task_name,
            destination_task_root=destination_task_root,
            replace_existing_repo_path=config.replace_existing_repo_path,
        )

    return {
        **payload,
        "download_root": str(download_root),
        "downloaded_task_root": str(downloaded_task_root),
        **link_payload,
        "downloaded": True,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sync and train a VT Franka visuotactile policy on a remote PC")
    parser.add_argument("--workspace-config", default="robot_workspace/config/workspace.yaml")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--backend-dataset-root", type=Path, default=None)
    parser.add_argument("--raw-run-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--wandb-mode", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest milestone/best checkpoint.")
    parser.add_argument("--remote", required=True, help="SSH target, e.g. user@host")
    parser.add_argument("--remote-root", required=True, help="Remote robot_workspace root")
    parser.add_argument("--ssh-port", type=int, default=None)
    parser.add_argument("--ssh-key", type=Path, default=None)
    parser.add_argument("--remote-python", default="python")
    parser.add_argument("--no-sync-code", action="store_true")
    parser.add_argument("--no-sync-dataset", action="store_true")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=None)
    return parser


def build_pipeline_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish/fetch VT Franka visuotactile inference bundles")
    subparsers = parser.add_subparsers(dest="command", required=True)

    publish = subparsers.add_parser(
        "publish-inference-bundle",
        help="Upload checkpoint folders needed for local inference to ModelScope",
    )
    publish.add_argument("task_name")
    publish.add_argument("--models", nargs="+", required=True, choices=sorted(MODEL_SPECS))
    publish.add_argument("--repo-id", required=True)
    publish.add_argument("--repo-type", default=DEFAULT_MODELSCOPE_REPO_TYPE)
    publish.add_argument("--revision", default=DEFAULT_MODELSCOPE_REVISION)
    publish.add_argument("--token", default=None)
    publish.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    publish.add_argument("--checkpoints-root", type=Path, default=None)
    publish.add_argument("--stage-root", type=Path, default=None)
    publish.add_argument("--max-workers", type=int, default=5)
    publish.add_argument("--visibility", default="private")
    publish.add_argument("--endpoint", default=None)
    publish.add_argument("--no-create-repo", action="store_true")
    publish.add_argument("--dry-run", action="store_true")

    fetch = subparsers.add_parser(
        "fetch-inference-bundle",
        help="Download ModelScope inference bundles to SSD and install repo checkpoint link",
    )
    fetch.add_argument("task_name")
    fetch.add_argument("--models", nargs="+", required=True, choices=sorted(MODEL_SPECS))
    fetch.add_argument("--repo-id", required=True)
    fetch.add_argument("--repo-type", default=DEFAULT_MODELSCOPE_REPO_TYPE)
    fetch.add_argument("--revision", default=DEFAULT_MODELSCOPE_REVISION)
    fetch.add_argument("--token", default=None)
    fetch.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    fetch.add_argument("--ssd-root", type=Path, default=Path("/mnt/kenny_ssd/vt_franka"))
    fetch.add_argument("--download-dir", type=Path, default=None)
    fetch.add_argument("--destination-root", type=Path, default=None)
    fetch.add_argument("--max-workers", type=int, default=5)
    fetch.add_argument("--endpoint", default=None)
    fetch.add_argument("--no-link-repo-checkpoints", action="store_true")
    fetch.add_argument("--replace-existing-repo-path", action="store_true")
    fetch.add_argument("--no-delete-destination-extra-files", action="store_true")
    fetch.add_argument("--dry-run", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> RemoteTrainConfig:
    workspace = load_workspace_config(args.workspace_config)
    command_override = getattr(args, "command_override", None)
    if command_override is None:
        command_override = getattr(args, "command", None)
    if command_override and command_override[0] == "--":
        command_override = command_override[1:]
    local_train = TrainVisuotactileConfig(
        workspace=workspace,
        task_name=args.task_name,
        model=args.model,
        dataset_dir=args.dataset_dir,
        backend_dataset_root=args.backend_dataset_root,
        raw_run_dir=args.raw_run_dir,
        dataset_name=args.dataset_name,
        checkpoint_dir=args.checkpoint_dir,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        wandb_mode=args.wandb_mode,
        overwrite=args.overwrite,
        resume=args.resume,
        dry_run=args.dry_run,
        command_override=command_override or None,
    )
    return RemoteTrainConfig(
        local_train=local_train,
        remote=args.remote,
        remote_root=args.remote_root,
        ssh_port=args.ssh_port,
        ssh_key=args.ssh_key,
        remote_python=args.remote_python,
        sync_code=not args.no_sync_code,
        sync_dataset=not args.no_sync_dataset,
        download=not args.no_download,
        dry_run=args.dry_run,
    )


def pipeline_config_from_args(args: argparse.Namespace) -> PublishInferenceBundleConfig | FetchInferenceBundleConfig:
    if args.command == "publish-inference-bundle":
        return PublishInferenceBundleConfig(
            task_name=args.task_name,
            models=args.models,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            token=args.token,
            repo_root=args.repo_root,
            checkpoints_root=args.checkpoints_root,
            stage_root=args.stage_root,
            max_workers=args.max_workers,
            create_repo=not args.no_create_repo,
            visibility=args.visibility,
            endpoint=args.endpoint,
            dry_run=args.dry_run,
        )
    if args.command == "fetch-inference-bundle":
        return FetchInferenceBundleConfig(
            task_name=args.task_name,
            models=args.models,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            revision=args.revision,
            token=args.token,
            repo_root=args.repo_root,
            ssd_root=args.ssd_root,
            download_dir=args.download_dir,
            destination_root=args.destination_root,
            max_workers=args.max_workers,
            endpoint=args.endpoint,
            link_repo_checkpoints=not args.no_link_repo_checkpoints,
            replace_existing_repo_path=args.replace_existing_repo_path,
            delete_destination_extra_files=not args.no_delete_destination_extra_files,
            dry_run=args.dry_run,
        )
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in {"publish-inference-bundle", "fetch-inference-bundle"}:
        args = build_pipeline_arg_parser().parse_args(argv)
        config = pipeline_config_from_args(args)
        if isinstance(config, PublishInferenceBundleConfig):
            payload = publish_inference_bundle(config)
        else:
            payload = fetch_inference_bundle(config)
        print(json.dumps(payload, indent=2))
        return

    args = build_arg_parser().parse_args(argv)
    result = remote_train_visuotactile(config_from_args(args))
    payload = {
        "local_checkpoint_dir": str(result.local_checkpoint_dir),
        "remote_checkpoint_dir": result.remote_checkpoint_dir,
        "commands": [" ".join(shlex.quote(item) for item in command) for command in result.commands],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
