from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ....config import WorkspaceSettings, load_workspace_config
from .config import (
    DEFAULT_DATASET_NAME,
    MODEL_SPECS,
    VisuotactileModelSpec,
    default_checkpoint_dir,
    default_prepared_dataset_dir,
    get_model_spec,
)
from .export_backend import BackendExportResult, export_prepared_dataset_for_backend
from .prepare import build_prepare_config_from_workspace, prepare_visuotactile_dataset
from .runtime import RuntimeManifests, write_runtime_manifests


REPO_ROOT = Path(__file__).resolve().parents[6]
WORKSPACE_SRC_ROOT = REPO_ROOT / "robot_workspace" / "src"
POLICIES_ROOT = WORKSPACE_SRC_ROOT / "vt_franka_workspace" / "policies"
CHECKPOINT_ASSET_ROOT = POLICIES_ROOT / "common" / "visuotactile" / "checkpoints"
UNIVTAC_ENCODER_CKPT = CHECKPOINT_ASSET_ROOT / "UniVTAC_encoder" / "best.pth"
VITAL_VISION_ENCODER_CKPT = CHECKPOINT_ASSET_ROOT / "VITAL_encoder" / "best_vision_encoder.pth"
VITAL_GELSIGHT_ENCODER_CKPT = CHECKPOINT_ASSET_ROOT / "VITAL_encoder" / "best_gelsight_encoder.pth"


@dataclass(frozen=True)
class TrainVisuotactileConfig:
    workspace: WorkspaceSettings
    task_name: str
    model: str
    dataset_dir: Path | None = None
    backend_dataset_root: Path | None = None
    raw_run_dir: Path | None = None
    dataset_name: str = DEFAULT_DATASET_NAME
    checkpoint_dir: Path | None = None
    run_name: str | None = None
    seed: int = 0
    device: str = "cuda"
    batch_size: int | None = None
    epochs: int | None = None
    learning_rate: float | None = None
    wandb_mode: str | None = None
    overwrite: bool = False
    resume: bool = False
    prepare_if_missing: bool = True
    dry_run: bool = False
    command_override: list[str] | None = None
    extra_args: list[str] | None = None


@dataclass(frozen=True)
class TrainVisuotactileResult:
    checkpoint_dir: Path
    dataset_dir: Path
    backend_dataset_root: Path | None
    command: list[str]
    manifest_path: Path
    dry_run: bool
    cwd: Path | None = None


def train_visuotactile(config: TrainVisuotactileConfig) -> TrainVisuotactileResult:
    spec = get_model_spec(config.model)
    dataset_dir = _resolve_dataset_dir(config)
    checkpoint_dir = _resolve_checkpoint_dir(config, spec)
    if checkpoint_dir.exists() and not config.dry_run:
        if config.resume:
            pass
        elif not config.overwrite:
            raise FileExistsError(f"Checkpoint directory already exists: {checkpoint_dir}")
        else:
            shutil.rmtree(checkpoint_dir)
    if not config.dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    backend_export = _resolve_backend_export(config, spec, dataset_dir, checkpoint_dir)

    backend_dataset_root = backend_export.backend_dataset_root if backend_export is not None else dataset_dir
    backend_episode_count = 1 if backend_export is None else max(int(backend_export.num_episodes), 1)
    command, command_cwd, command_env = _resolve_train_invocation(
        config,
        spec,
        backend_dataset_root,
        checkpoint_dir,
        n_demo=backend_episode_count,
    )
    if config.extra_args:
        command = [*command, *config.extra_args]

    manifests = _build_runtime_manifest_bundle(
        spec=spec,
        task_name=config.task_name,
        dataset_dir=dataset_dir,
        backend_export=backend_export,
        checkpoint_dir=checkpoint_dir,
        command=command,
        seed=config.seed,
        device=config.device,
        dry_run=config.dry_run,
    )
    if not config.dry_run:
        write_runtime_manifests(checkpoint_dir, manifests)
        (checkpoint_dir / "train_command.sh").write_text(_shell_join(command) + "\n", encoding="utf-8")
        (checkpoint_dir / "train_config.json").write_text(
            json.dumps(
                _train_config_payload(config, dataset_dir, checkpoint_dir, command, command_cwd),
                indent=2,
            ),
            encoding="utf-8",
        )
        _run_command(command, cwd=command_cwd or Path.cwd(), extra_env=command_env)
    return TrainVisuotactileResult(
        checkpoint_dir=checkpoint_dir,
        dataset_dir=dataset_dir,
        backend_dataset_root=None if backend_export is None else backend_export.backend_dataset_root,
        command=command,
        manifest_path=checkpoint_dir / "policy_manifest.json",
        dry_run=config.dry_run,
        cwd=command_cwd,
    )


def _resolve_dataset_dir(config: TrainVisuotactileConfig) -> Path:
    spec = get_model_spec(config.model)
    requested_dataset_dir = Path(config.dataset_dir) if config.dataset_dir is not None else None
    if requested_dataset_dir is not None and _is_common_dataset_dir(requested_dataset_dir):
        prepared_dir = _prepared_dir_for_common_dataset(requested_dataset_dir, config, spec)
        if (prepared_dir / "dataset_manifest.json").exists():
            return prepared_dir
        if config.dry_run:
            return prepared_dir
        if not config.prepare_if_missing:
            raise FileNotFoundError(f"Prepared visuotactile dataset is missing: {prepared_dir}")
        prepare_config = build_prepare_config_from_workspace(
            config.workspace,
            task_name=config.task_name,
            model=config.model,
            raw_run_dir=requested_dataset_dir,
            output_dir=prepared_dir,
            dataset_name=config.dataset_name,
            overwrite=config.overwrite,
            source="common",
            source_root=requested_dataset_dir,
        )
        result = prepare_visuotactile_dataset(prepare_config)
        return result.output_dir
    dataset_dir = requested_dataset_dir or default_prepared_dataset_dir(
        config.workspace,
        config.task_name,
        config.dataset_name,
        model=spec.name,
    )
    dataset_dir = Path(dataset_dir)
    if (dataset_dir / "dataset_manifest.json").exists():
        return dataset_dir
    if config.dry_run:
        return dataset_dir
    if not config.prepare_if_missing:
        raise FileNotFoundError(f"Prepared visuotactile dataset is missing: {dataset_dir}")
    prepare_config = build_prepare_config_from_workspace(
        config.workspace,
        task_name=config.task_name,
        model=config.model,
        raw_run_dir=config.raw_run_dir,
        output_dir=dataset_dir,
        dataset_name=config.dataset_name,
        overwrite=config.overwrite,
    )
    result = prepare_visuotactile_dataset(prepare_config)
    return result.output_dir


def _is_common_dataset_dir(path: Path) -> bool:
    manifest_path = Path(path) / "dataset_manifest.json"
    if not manifest_path.exists():
        return False
    try:
        payload = _read_json(manifest_path)
    except Exception:
        return False
    return str(payload.get("schema_version", "")).startswith("vt_franka_common_dataset")


def _prepared_dir_for_common_dataset(common_dataset_dir: Path, config: TrainVisuotactileConfig, spec: VisuotactileModelSpec) -> Path:
    explicit_prepared = default_prepared_dataset_dir(
        config.workspace,
        config.task_name,
        config.dataset_name,
        model=spec.name,
    )
    common_dataset_dir = Path(common_dataset_dir)
    parts = common_dataset_dir.parts
    if "datasets" in parts:
        datasets_index = parts.index("datasets")
        data_root = Path(*parts[:datasets_index]) if datasets_index > 0 else Path("/")
        if data_root != Path("/"):
            return data_root / "prepared" / config.task_name / "visuotactile" / config.dataset_name / spec.name
    return explicit_prepared


def _resolve_checkpoint_dir(config: TrainVisuotactileConfig, spec: VisuotactileModelSpec) -> Path:
    if config.checkpoint_dir is not None:
        return Path(config.checkpoint_dir)
    return default_checkpoint_dir(config.workspace, task_name=config.task_name, model=spec.name, run_name=config.run_name)


def _resolve_backend_export(
    config: TrainVisuotactileConfig,
    spec: VisuotactileModelSpec,
    dataset_dir: Path,
    checkpoint_dir: Path,
) -> BackendExportResult | None:
    if config.dry_run:
        output_root = config.backend_dataset_root or (checkpoint_dir / "backend_dataset")
        return BackendExportResult(
            backend_dataset_root=Path(output_root),
            task_dir=Path(output_root) / config.task_name,
            hdf5_dir=Path(output_root) / config.task_name / "hdf5",
            act_hdf5_dir=Path(output_root) / config.task_name / "act_hdf5",
            num_episodes=0,
            manifest_path=Path(output_root) / config.task_name / "backend_dataset_manifest.json",
        )
    output_root = config.backend_dataset_root or (checkpoint_dir / "backend_dataset")
    return export_prepared_dataset_for_backend(
        dataset_dir,
        output_root,
        model=spec.name,
        task_name=config.task_name,
        overwrite=config.overwrite,
    )


def _default_train_command(
    config: TrainVisuotactileConfig,
    spec: VisuotactileModelSpec,
    dataset_dir: Path,
    checkpoint_dir: Path,
    *,
    n_demo: int,
) -> list[str]:
    backend = spec.train_backend
    if backend in {"diffusion_policy", "vista", "vital_dp"}:
        return _hydra_train_command(config, spec, dataset_dir, checkpoint_dir, n_demo=n_demo)
    if backend in {"act", "vital_act"}:
        return _act_train_command(config, spec, dataset_dir, checkpoint_dir, n_demo=n_demo)
    raise ValueError(f"Unsupported visuotactile training backend: {backend}")


def _resolve_train_invocation(
    config: TrainVisuotactileConfig,
    spec: VisuotactileModelSpec,
    backend_dataset_root: Path,
    checkpoint_dir: Path,
    *,
    n_demo: int,
) -> tuple[list[str], Path | None, dict[str, str] | None]:
    if config.command_override:
        return config.command_override, None, None
    command = _default_train_command(config, spec, backend_dataset_root, checkpoint_dir, n_demo=n_demo)
    cwd: Path | None = None
    env: dict[str, str] | None = None
    if spec.name == "act_univtac":
        cwd = Path.cwd()
        env = _pythonpath_env(WORKSPACE_SRC_ROOT)
    elif spec.name == "vital_dp":
        cwd = Path.cwd()
        env = _pythonpath_env(WORKSPACE_SRC_ROOT, POLICIES_ROOT / "DP")
    elif spec.name == "vital_act":
        vital_root = POLICIES_ROOT / "ViTAL"
        cwd = vital_root
        env = {"PYTHONPATH": f"{vital_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"}
    return command, cwd, env


def _pythonpath_env(*paths: Path) -> dict[str, str]:
    existing = os.environ.get("PYTHONPATH", "")
    values = [str(path) for path in paths]
    if existing:
        values.append(existing)
    return {"PYTHONPATH": os.pathsep.join(values)}


def _hydra_train_command(
    config: TrainVisuotactileConfig,
    spec: VisuotactileModelSpec,
    dataset_dir: Path,
    checkpoint_dir: Path,
    *,
    n_demo: int,
) -> list[str]:
    backend_root = POLICIES_ROOT / spec.vendor_subdir
    module_or_script = {
        "dp_manifeel": str(backend_root / "train.py"),
        "dp_equidiff_tact": str(backend_root / "train.py"),
        "vital_dp": str(POLICIES_ROOT / "DP" / "train.py"),
        "vista_so2": str(backend_root / "train.py"),
        "vista_so3": str(backend_root / "train.py"),
    }[spec.name]
    hydra_config = {
        "dp_manifeel": "train_diffusion_unet_manifeel",
        "dp_equidiff_tact": "train_diffusion_unet_equidiff_tact",
        "vital_dp": "train_diffusion_unet_vital",
        "vista_so2": "train_vista_so2",
        "vista_so3": "train_vista",
    }[spec.name]
    command = [
        sys.executable,
        module_or_script,
        f"--config-name={hydra_config}",
        f"task_name={config.task_name}",
        f"dataset_root={dataset_dir}",
        "data_split=clean",
        f"n_demo={n_demo}",
        f"training.seed={config.seed}",
        f"training.device={config.device}",
        f"hydra.run.dir={checkpoint_dir}",
        f"hydra.sweep.dir={checkpoint_dir}",
        f"multi_run.run_dir={checkpoint_dir}",
        f"logging.mode={config.wandb_mode or 'disabled'}",
    ]
    if spec.name == "dp_equidiff_tact":
        command.extend(_shape_meta_overrides(spec, include_linked_paths=False))
        command.extend(
            [
                f"policy.resize_shape=[3,{spec.wrist_image_size},{spec.wrist_image_size}]",
                f"policy.crop_shape=[{spec.wrist_image_size},{spec.wrist_image_size}]",
            ]
        )
    elif spec.name in {"vista_so2", "vista_so3"}:
        command.extend(
            [
                "task=univtac_vista",
                "policy.crop_shape=null",
                f"policy.tactile_shape=[3,{spec.tactile_image_size},{spec.tactile_image_size}]",
                f"task.shape_meta.obs.robot0_eye_in_hand_image.shape=[3,{spec.wrist_image_size},{spec.wrist_image_size}]",
                f"task.dataset.shape_meta.obs.robot0_eye_in_hand_image.shape=[3,{spec.wrist_image_size},{spec.wrist_image_size}]",
                f"task.shape_meta.obs.robot0_tactile_left_image.shape=[3,{spec.tactile_image_size},{spec.tactile_image_size}]",
                f"task.dataset.shape_meta.obs.robot0_tactile_left_image.shape=[3,{spec.tactile_image_size},{spec.tactile_image_size}]",
            ]
        )
    else:
        command.extend(_shape_meta_overrides(spec, include_linked_paths=False))
    if config.batch_size is not None:
        command.extend([f"dataloader.batch_size={config.batch_size}", f"val_dataloader.batch_size={config.batch_size}"])
    if config.epochs is not None:
        command.append(f"training.num_epochs={config.epochs}")
    if config.learning_rate is not None:
        command.append(f"optimizer.learning_rate={config.learning_rate}")
    if spec.name in {"dp_manifeel", "dp_equidiff_tact", "vital_dp", "vista_so2", "vista_so3"}:
        if config.epochs is None:
            command.append("training.num_epochs=300")
        command.extend(
            [
                f"training.resume={str(config.resume).lower()}",
                "training.checkpoint_mode=milestone_train_loss",
                "training.checkpoint_every=30",
                "training.val_every=1000000000",
                "task.dataset.val_ratio=0.0",
                "checkpoint.topk.k=0",
                "checkpoint.save_last_ckpt=False",
                "checkpoint.save_last_snapshot=False",
            ]
        )
    if spec.name == "vital_dp":
        command.extend(
            [
                f"policy.obs_encoder.vision_backbone_path={VITAL_VISION_ENCODER_CKPT.resolve()}",
                f"policy.obs_encoder.gelsight_backbone_path={VITAL_GELSIGHT_ENCODER_CKPT.resolve()}",
            ]
        )
    return command


def _act_train_command(
    config: TrainVisuotactileConfig,
    spec: VisuotactileModelSpec,
    dataset_dir: Path,
    checkpoint_dir: Path,
    *,
    n_demo: int,
) -> list[str]:
    if spec.name == "act_univtac":
        return _univtac_act_train_command(config, dataset_dir, checkpoint_dir, n_demo=n_demo)
    if spec.name == "vital_act":
        return _vital_act_train_command(config, dataset_dir, checkpoint_dir, n_demo=n_demo)
    raise ValueError(f"Unsupported ACT-style visuotactile model: {spec.name}")


def _univtac_act_train_command(
    config: TrainVisuotactileConfig,
    dataset_dir: Path,
    checkpoint_dir: Path,
    *,
    n_demo: int,
) -> list[str]:
    train_config_path = checkpoint_dir / "act_train_config.yml"
    if not config.dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_config_path.write_text(
            yaml.safe_dump(_univtac_act_config(config), sort_keys=False),
            encoding="utf-8",
        )
    return [
        sys.executable,
        "-m",
        "vt_franka_workspace.policies.ACT.imitate_episodes",
        "--ckpt_dir",
        str(checkpoint_dir),
        "--task_name",
        config.task_name,
        "--config_path",
        str(train_config_path.resolve()),
        "--dataset_dir",
        str(_act_dataset_dir(dataset_dir, config.task_name).resolve()),
        "--num_episodes",
        str(n_demo),
        "--seed",
        str(config.seed),
    ]


def _vital_act_train_command(
    config: TrainVisuotactileConfig,
    dataset_dir: Path,
    checkpoint_dir: Path,
    *,
    n_demo: int,
) -> list[str]:
    del n_demo
    train_config_path = checkpoint_dir / "vital_act_train_config.json"
    if not config.dry_run:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        train_config_path.write_text(
            json.dumps(_vital_act_config(config, dataset_dir, checkpoint_dir), indent=2),
            encoding="utf-8",
        )
    return [
        sys.executable,
        str(POLICIES_ROOT / "ViTAL" / "imitate_episodes.py"),
        "--config",
        str(train_config_path.resolve()),
    ]


def _act_dataset_dir(backend_dataset_root: Path, task_name: str) -> Path:
    return Path(backend_dataset_root) / task_name / "act_hdf5"


def _univtac_act_config(config: TrainVisuotactileConfig) -> dict[str, Any]:
    batch_size = int(config.batch_size or 64)
    return {
        "state_dim": 8,
        "kl_weight": 10.0,
        "chunk_size": 50,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "temporal_agg": False,
        "device": config.device if ":" in config.device else f"{config.device}:0",
        "ckpt_dir": None,
        "policy_class": "ACT",
        "num_steps": int(config.epochs or 4000),
        "num_epochs": None,
        "batch_size": batch_size,
        "save_freq": 1000,
        "position_embedding": "sine",
        "lr_vision_backbone": 1e-5,
        "lr_tactile_backbone": 1e-5,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
        "lr": float(config.learning_rate or 1e-5),
        "masks": False,
        "dilation": False,
        "backbone": "resnet18",
        "nheads": 8,
        "enc_layers": 4,
        "dec_layers": 7,
        "pre_norm": False,
        "dropout": 0.1,
        "camera_names": ["cam_wrist"],
        "tactile_names": ["tac_left", "tac_right"],
        "tactile_masks": False,
        "tactile_backbone": "resnet18",
        "tactile_ckpt": str(UNIVTAC_ENCODER_CKPT),
        "tactile_dilation": False,
    }


def _vital_act_config(
    config: TrainVisuotactileConfig,
    dataset_dir: Path,
    checkpoint_dir: Path,
) -> dict[str, Any]:
    gpu = _device_to_gpu_index(config.device)
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        gpu = -1
    return {
        "eval": False,
        "checkpoint": "policy_best.ckpt",
        "onscreen_render": False,
        "save_dir": str(_act_dataset_dir(dataset_dir, config.task_name).resolve()),
        "name": str((checkpoint_dir / "vendor_vital_act_run").resolve()),
        "policy_class": "ACT",
        "batch_size": int(config.batch_size or 32),
        "seed": int(config.seed),
        "num_epochs": int(config.epochs or 6000),
        "lr": float(config.learning_rate or 1e-5),
        "kl_weight": 10.0,
        "start_kl_epoch": 0,
        "kl_scale_epochs": 0,
        "chunk_size": 20,
        "hidden_dim": 512,
        "dim_feedforward": 3200,
        "temporal_agg": True,
        "z_dimension": 32,
        "gpu": gpu,
        "lr_backbone": 1e-5,
        "weight_decay": 1e-4,
        "backbone": "clip_backbone",
        "dilation": False,
        "position_embedding": "sine",
        "enc_layers": 4,
        "dec_layers": 7,
        "dropout": 0.025,
        "nheads": 8,
        "pre_norm": False,
        "masks": False,
        "gelsight_backbone_path": str(VITAL_GELSIGHT_ENCODER_CKPT.resolve()),
        "vision_backbone_path": str(VITAL_VISION_ENCODER_CKPT.resolve()),
    }


def _device_to_gpu_index(device: str) -> int:
    value = str(device).strip().lower()
    if not value.startswith("cuda"):
        return -1
    if ":" not in value:
        return -1
    suffix = value.split(":", 1)[1]
    try:
        return int(suffix)
    except ValueError:
        return -1


def _build_runtime_manifest_bundle(
    *,
    spec: VisuotactileModelSpec,
    task_name: str,
    dataset_dir: Path,
    backend_export: BackendExportResult | None,
    checkpoint_dir: Path,
    command: list[str],
    seed: int,
    device: str,
    dry_run: bool = False,
) -> RuntimeManifests:
    dataset_manifest = _read_dataset_manifest(dataset_dir, spec=spec, task_name=task_name, dry_run=dry_run)
    normalizer_stats = _read_normalizer_stats(dataset_dir, spec=spec, dry_run=dry_run)
    common_dataset = _common_dataset_manifest(dataset_dir, dataset_manifest, dry_run=dry_run)
    preprocess2 = {
        "schema_version": "vt_franka_visuotactile_preprocess2_v1",
        "source_dataset_manifest": str(dataset_dir / "dataset_manifest.json"),
        "preprocess2": dataset_manifest["preprocess2"],
    }
    policy = {
        "schema_version": "vt_franka_visuotactile_policy_v1",
        "model": spec.name,
        "family": spec.family,
        "task_name": task_name,
        "dataset_dir": str(dataset_dir),
        "backend_dataset_root": None if backend_export is None else str(backend_export.backend_dataset_root),
        "backend_dataset_manifest": None if backend_export is None else str(backend_export.manifest_path),
        "checkpoint_dir": str(checkpoint_dir),
        "action_representation": spec.action_representation,
        "action_dim": spec.action_dim,
        "qpos_dim": spec.qpos_dim,
        "model_input": dataset_manifest.get("model_input", {"shape_meta": spec.backend_shape_meta()}),
        "obs_horizon": int(dataset_manifest.get("obs_horizon", spec.obs_horizon)),
        "action_horizon": int(dataset_manifest.get("action_horizon", spec.action_horizon)),
        "train_backend": spec.train_backend,
        "train_command": command,
        "seed": int(seed),
        "device": device,
        "created_at_wall_time": time.time(),
        "runtime_artifact": "checkpoints/epoch=*.ckpt",
        "runtime_note": (
            "The VT_Franka runtime loads DP/VISTA checkpoints from best.ckpt, latest.ckpt, "
            "or the latest checkpoints/epoch=*.ckpt milestone "
            "and ACT/ViTAL ACT checkpoints from policy_best.ckpt or best.ckpt. "
            "model_torchscript.pt remains supported for exported TorchScript backends."
        ),
    }
    return RuntimeManifests(
        policy=policy,
        preprocess1=common_dataset,
        preprocess2=preprocess2,
        normalizer_stats=normalizer_stats,
    )


def _common_dataset_manifest(dataset_dir: Path, dataset_manifest: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    source_root = dataset_manifest.get("source_root")
    if source_root:
        source_manifest = Path(source_root) / "dataset_manifest.json"
        if source_manifest.exists():
            return _read_json(source_manifest)
    if dataset_manifest.get("source") == "common":
        manifest_path = dataset_dir / "dataset_manifest.json"
        if manifest_path.exists():
            return {
                "schema_version": "vt_franka_runtime_standardized_dataset_reference_v1",
                "source_dataset_manifest": str(manifest_path),
                "camera_standardization": "RGB uint8 640x480",
                "streams": {
                    "rgb_wrist": {"shape": [480, 640, 3], "color_space": "RGB"},
                    "gelsight": {"shape": [480, 640, 3], "color_space": "RGB", "source_stream": "tactile_left"},
                },
            }
    return _first_preprocess1_manifest(dataset_manifest, dry_run=dry_run)


def _first_preprocess1_manifest(dataset_manifest: dict[str, Any], *, dry_run: bool = False) -> dict[str, Any]:
    for entry in dataset_manifest.get("episodes", []):
        preprocess_manifest = entry.get("preprocess1_manifest")
        if preprocess_manifest:
            path = Path(preprocess_manifest)
            if path.exists():
                return _read_json(path)
        preprocess_dir = entry.get("preprocess1_dir")
        if preprocess_dir:
            path = Path(preprocess_dir) / "preprocess1_manifest.json"
            if path.exists():
                return _read_json(path)
    if dry_run:
        return {
            "schema_version": "vt_franka_preprocess1_dry_run_placeholder",
            "profile_name": dataset_manifest.get("dataset_name"),
            "streams": {},
        }
    raise FileNotFoundError("Could not locate a preprocess1_manifest.json from dataset_manifest episodes")


def _read_dataset_manifest(
    dataset_dir: Path,
    *,
    spec: VisuotactileModelSpec,
    task_name: str,
    dry_run: bool,
) -> dict[str, Any]:
    path = dataset_dir / "dataset_manifest.json"
    if path.exists():
        return _read_json(path)
    if not dry_run:
        raise FileNotFoundError(path)
    return {
        "schema_version": "vt_franka_visuotactile_dataset_dry_run_placeholder",
        "task_name": task_name,
        "model": spec.name,
        "dataset_name": "dry_run",
        "preprocess2": spec.preprocess2_specs(),
        "model_input": {
            "camera_names": list(spec.camera_names),
            "tactile_names": list(spec.tactile_names),
            "shape_meta": spec.backend_shape_meta(),
        },
        "obs_horizon": int(spec.obs_horizon),
        "action_horizon": int(spec.action_horizon),
        "episodes": [],
    }


def _read_normalizer_stats(
    dataset_dir: Path,
    *,
    spec: VisuotactileModelSpec,
    dry_run: bool,
) -> dict[str, Any]:
    path = dataset_dir / "normalizer_stats.json"
    if path.exists():
        return _read_json(path)
    if not dry_run:
        raise FileNotFoundError(path)
    preferred_action = "action_pose10_rot6d_gripper" if spec.action_representation == "pose10_rot6d_gripper" else "action_pose7_gripper"
    preferred_qpos = "qpos_pose10_rot6d_gripper" if spec.action_representation == "pose10_rot6d_gripper" else "qpos_pose7_gripper"
    return {
        "schema_version": "vt_franka_visuotactile_normalizer_dry_run_placeholder",
        "preferred_action_key": preferred_action,
        "preferred_qpos_key": preferred_qpos,
    }


def _train_config_payload(
    config: TrainVisuotactileConfig,
    dataset_dir: Path,
    checkpoint_dir: Path,
    command: list[str],
    command_cwd: Path | None,
) -> dict[str, Any]:
    return {
        "schema_version": "vt_franka_visuotactile_train_config_v1",
        "task_name": config.task_name,
        "model": get_model_spec(config.model).name,
        "dataset_dir": str(dataset_dir),
        "backend_dataset_root": None if config.backend_dataset_root is None else str(config.backend_dataset_root),
        "checkpoint_dir": str(checkpoint_dir),
        "seed": config.seed,
        "device": config.device,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "learning_rate": config.learning_rate,
        "resume": config.resume,
        "command": command,
        "command_cwd": None if command_cwd is None else str(command_cwd),
    }


def _run_command(command: list[str], *, cwd: Path, extra_env: dict[str, str] | None = None) -> None:
    env = None
    if extra_env:
        env = os.environ.copy()
        env.update(extra_env)
    try:
        subprocess.run(command, cwd=str(cwd), env=env, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(f"Failed to start visuotactile training command: {_shell_join(command)}") from exc


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp_run_name(model: str) -> str:
    return f"{model}_{time.strftime('%Y%m%d_%H%M%S')}"


def _shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(str(item)) for item in command)


def _shape_meta_overrides(
    spec: VisuotactileModelSpec,
    *,
    include_linked_paths: bool = True,
) -> list[str]:
    if not spec.uses_pose10_rot6d:
        return []
    overrides = [
        f"shape_meta.obs.robot0_eye_in_hand_image.shape=[3,{spec.wrist_image_size},{spec.wrist_image_size}]",
        f"shape_meta.obs.robot0_tactile_left_image.shape=[3,{spec.tactile_image_size},{spec.tactile_image_size}]",
    ]
    if include_linked_paths:
        overrides.extend(
            [
                f"task.shape_meta.obs.robot0_eye_in_hand_image.shape=[3,{spec.wrist_image_size},{spec.wrist_image_size}]",
                f"task.dataset.shape_meta.obs.robot0_eye_in_hand_image.shape=[3,{spec.wrist_image_size},{spec.wrist_image_size}]",
                f"task.shape_meta.obs.robot0_tactile_left_image.shape=[3,{spec.tactile_image_size},{spec.tactile_image_size}]",
                f"task.dataset.shape_meta.obs.robot0_tactile_left_image.shape=[3,{spec.tactile_image_size},{spec.tactile_image_size}]",
            ]
        )
    return overrides


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a VT Franka visuotactile policy")
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
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--resume", action="store_true", help="Resume from the latest milestone/best checkpoint.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--extra-arg",
        dest="extra_args",
        action="append",
        default=[],
        help="Append one backend training override argument. Can be repeated.",
    )
    parser.add_argument("--command", nargs=argparse.REMAINDER, default=None, help="Override backend train command")
    return parser


def config_from_args(args: argparse.Namespace) -> TrainVisuotactileConfig:
    workspace = load_workspace_config(args.workspace_config)
    command_override = getattr(args, "command_override", None)
    if command_override is None:
        command_override = getattr(args, "command", None)
    if command_override and command_override[0] == "--":
        command_override = command_override[1:]
    return TrainVisuotactileConfig(
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
        prepare_if_missing=not args.no_prepare,
        dry_run=args.dry_run,
        command_override=command_override or None,
        extra_args=args.extra_args or None,
    )


def main() -> None:
    args = build_arg_parser().parse_args()
    result = train_visuotactile(config_from_args(args))
    print(f"checkpoint_dir={result.checkpoint_dir}")
    print(f"dataset_dir={result.dataset_dir}")
    if result.backend_dataset_root is not None:
        print(f"backend_dataset_root={result.backend_dataset_root}")
    print(f"manifest={result.manifest_path}")
    print(f"command={_shell_join(result.command)}")


if __name__ == "__main__":
    main()
