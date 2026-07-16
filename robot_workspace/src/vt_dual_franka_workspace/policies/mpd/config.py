from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ...config import PolicyConfig, WorkspaceSettings

AlgorithmName = Literal["dp", "fm", "sfp", "mpd", "motif", "freqpolicy"]
PolicyFamily = Literal["chunk", "prodmp", "motif", "freqpolicy"]

DEFAULT_DATASET_NAME = "vt_franka_mpd_v1"
DEFAULT_UPSTREAM_REPO = Path("/home/zhenya/kenny/visuotact/vt_dual_franka/robot_workspace/third_parties/mpd")
DEFAULT_VECTOR_DIM = 10
ACTION_CONVENTION_CLOSEDNESS = "tcp_xyz_rot6d_gripper_closedness"
ACTION_CONVENTION_OPEN_FRACTION = "tcp_xyz_rot6d_gripper_open_fraction"


@dataclass(frozen=True)
class MPDPolicySpec:
    algorithm: AlgorithmName
    policy_name: str
    family: PolicyFamily
    reference_model_name: str
    method_names: tuple[str, ...]

    def upstream_config_name(self, experiment: str) -> str:
        config_leaf = {
            "dp": "train_dp_transformer",
            "fm": "train_fm_transformer",
            "sfp": "train_sfp_transformer",
            "mpd": "train_prodmp_transformer",
            "motif": "train_motif_transformer",
            "freqpolicy": "train_freqpolicy",
        }[self.algorithm]
        return f"experiments/{experiment}/{config_leaf}"


_POLICY_SPECS: dict[str, MPDPolicySpec] = {
    "dp": MPDPolicySpec(
        algorithm="dp",
        policy_name="dp_state",
        family="chunk",
        reference_model_name="dp",
        method_names=("dp_transformer",),
    ),
    "fm": MPDPolicySpec(
        algorithm="fm",
        policy_name="fm_state",
        family="chunk",
        reference_model_name="fm",
        method_names=("fm_transformer",),
    ),
    "sfp": MPDPolicySpec(
        algorithm="sfp",
        policy_name="sfp_state",
        family="chunk",
        reference_model_name="sfp",
        method_names=("sfp_transformer",),
    ),
    "mpd": MPDPolicySpec(
        algorithm="mpd",
        policy_name="mpd_state",
        family="prodmp",
        reference_model_name="mpd",
        method_names=("mpd-transformer", "prodmp_transformer"),
    ),
    "motif": MPDPolicySpec(
        algorithm="motif",
        policy_name="motif_state",
        family="motif",
        reference_model_name="motif",
        method_names=("motif", "motif_transformer"),
    ),
    "freqpolicy": MPDPolicySpec(
        algorithm="freqpolicy",
        policy_name="freqpolicy_state",
        family="freqpolicy",
        reference_model_name="freqpolicy",
        method_names=("freqpolicy", "freqpolicy_official"),
    ),
}

_ALIASES = {
    "prodmp_diffusion": "mpd",
    "motif_diffusion": "motif",
    "freqpolicy_official": "freqpolicy",
    "freq_policy": "freqpolicy",
}

_REJECTED = {
    "prodmp_fm": "prodmp_fm is intentionally unsupported; run mpd instead.",
    "motif_fm": "motif_fm is intentionally unsupported; run motif instead.",
}


def normalize_algorithm_name(value: str) -> AlgorithmName:
    key = value.strip().lower()
    if key in _REJECTED:
        raise ValueError(_REJECTED[key])
    key = _ALIASES.get(key, key)
    if key not in _POLICY_SPECS:
        supported = ", ".join(_POLICY_SPECS)
        raise ValueError(f"Unsupported MPD algorithm {value!r}. Supported algorithms: {supported}")
    return key  # type: ignore[return-value]


def get_policy_spec(value: str) -> MPDPolicySpec:
    return _POLICY_SPECS[normalize_algorithm_name(value)]


def default_prepared_dataset_dir(workspace: WorkspaceSettings, task_name: str, dataset_name: str = DEFAULT_DATASET_NAME) -> Path:
    return Path(workspace.recording.prepared_root) / "mpd" / task_name / dataset_name


def checkpoint_run_dir(
    workspace: WorkspaceSettings,
    *,
    task_name: str,
    algorithm: str,
    policy_name: str | None = None,
) -> Path:
    spec = get_policy_spec(algorithm)
    run_name = policy_name or spec.policy_name
    return Path(workspace.recording.checkpoints_root) / task_name / "mpd" / spec.algorithm / run_name


def default_checkpoint_path(
    workspace: WorkspaceSettings,
    *,
    task_name: str,
    algorithm: str,
    policy_name: str | None = None,
) -> Path:
    return checkpoint_run_dir(
        workspace,
        task_name=task_name,
        algorithm=algorithm,
        policy_name=policy_name,
    ) / "best_model.pth"


class MPDPolicySettings(BaseModel):
    algorithm: str
    task_name: str | None = None
    policy_name: str | None = None
    checkpoint_dir: Path | None = None
    upstream_repo_dir: Path = DEFAULT_UPSTREAM_REPO
    upstream_experiment: str | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    action_dim: int = DEFAULT_VECTOR_DIM
    gripper_close_threshold: float = 0.5
    gripper_open_width_m: float | None = None
    target_duration_sec: float | None = None
    gripper_switch_lockout_actions: int = 0

    @field_validator("action_dim")
    @classmethod
    def _validate_action_dim(cls, value: int) -> int:
        if value != DEFAULT_VECTOR_DIM:
            raise ValueError("The VT Dual Franka MPD adapter currently uses the 10D state/action vector")
        return value

    @field_validator("gripper_close_threshold")
    @classmethod
    def _validate_gripper_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("gripper_close_threshold must be in [0, 1]")
        return value

    @field_validator("gripper_switch_lockout_actions")
    @classmethod
    def _validate_gripper_switch_lockout_actions(cls, value: int) -> int:
        if value < 0:
            raise ValueError("gripper_switch_lockout_actions must be non-negative")
        return value

    @model_validator(mode="after")
    def _normalize_algorithm(self) -> "MPDPolicySettings":
        self.algorithm = normalize_algorithm_name(self.algorithm)
        if self.policy_name is None:
            self.policy_name = get_policy_spec(self.algorithm).policy_name
        return self

    @classmethod
    def from_policy_config(
        cls,
        policy_config: PolicyConfig,
        workspace: WorkspaceSettings,
        *,
        fallback_task_name: str,
    ) -> tuple["MPDPolicySettings", Path]:
        settings = cls.model_validate(policy_config.config)
        task_name = settings.task_name or fallback_task_name
        checkpoint_path = policy_config.checkpoint_path
        if checkpoint_path is None and settings.checkpoint_dir is not None:
            checkpoint_path = settings.checkpoint_dir / "best_model.pth"
        if checkpoint_path is None:
            checkpoint_path = default_checkpoint_path(
                workspace,
                task_name=task_name,
                algorithm=settings.algorithm,
                policy_name=settings.policy_name,
            )
        settings.task_name = task_name
        return settings, Path(checkpoint_path)
