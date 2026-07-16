from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from ....config import PolicyConfig, WorkspaceSettings


VisuotactileModelName = Literal["dp_bimanual"]
ActionRepresentation = Literal["pose10_rot6d_gripper", "pose7_gripper", "bimanual_pose20_rot6d_gripper"]
ModelFamily = Literal["dp", "act", "vital", "vista"]

DEFAULT_DATASET_NAME = "real_canonical_v1"
DEFAULT_PREPROCESS1_PROFILE = "real_canonical_v1"
DEFAULT_POLICY_FAMILY = "visuotactile"


@dataclass(frozen=True)
class VisuotactileModelSpec:
    name: VisuotactileModelName
    family: ModelFamily
    action_representation: ActionRepresentation
    default_image_size: int
    wrist_image_size: int
    tactile_image_size: int
    obs_horizon: int
    action_horizon: int
    vendor_subdir: str
    train_backend: str
    camera_names: tuple[str, ...] = ("rgb_wrist",)
    tactile_names: tuple[str, ...] = ("gelsight",)

    @property
    def action_dim(self) -> int:
        if self.action_representation == "bimanual_pose20_rot6d_gripper":
            return 20
        return 10 if self.action_representation == "pose10_rot6d_gripper" else 8

    @property
    def qpos_dim(self) -> int:
        return self.action_dim

    @property
    def image_size(self) -> int:
        if self.wrist_image_size != self.tactile_image_size:
            raise ValueError(f"{self.name} uses different wrist/tactile sizes")
        return self.wrist_image_size

    @property
    def uses_pose10_rot6d(self) -> bool:
        return self.action_representation == "pose10_rot6d_gripper"

    def preprocess2_specs(self) -> dict[str, dict[str, object]]:
        from .image_preprocess import CropSpec, ImagePreprocessSpec

        if self.action_representation == "bimanual_pose20_rot6d_gripper":
            return {
                "rgb_wrist_left": ImagePreprocessSpec(
                    output_size=(self.wrist_image_size, self.wrist_image_size),
                    crop=CropSpec(mode="none"),
                    interpolation="area",
                ).to_json(),
                "rgb_wrist_right": ImagePreprocessSpec(
                    output_size=(self.wrist_image_size, self.wrist_image_size),
                    crop=CropSpec(mode="none"),
                    interpolation="area",
                ).to_json(),
                "tactile_left": ImagePreprocessSpec(
                    output_size=(self.tactile_image_size, self.tactile_image_size),
                    crop=CropSpec(mode="none"),
                    interpolation="area",
                ).to_json(),
                "tactile_right": ImagePreprocessSpec(
                    output_size=(self.tactile_image_size, self.tactile_image_size),
                    crop=CropSpec(mode="none"),
                    interpolation="area",
                ).to_json(),
            }
        return {
            "rgb_wrist": ImagePreprocessSpec(
                output_size=(self.wrist_image_size, self.wrist_image_size),
                crop=CropSpec(mode="none"),
                interpolation="area",
            ).to_json(),
            "gelsight": ImagePreprocessSpec(
                output_size=(self.tactile_image_size, self.tactile_image_size),
                crop=CropSpec(mode="none"),
                interpolation="area",
            ).to_json(),
        }

    def backend_shape_meta(self) -> dict[str, object]:
        if self.action_representation == "bimanual_pose20_rot6d_gripper":
            image_shape = [3, self.wrist_image_size, self.wrist_image_size]
            tactile_shape = [3, self.tactile_image_size, self.tactile_image_size]
            return {
                "obs": {
                    "robot0_eye_in_hand_image": {"shape": image_shape, "type": "rgb"},
                    "robot1_eye_in_hand_image": {"shape": image_shape, "type": "rgb"},
                    "robot0_tactile_left_image": {"shape": tactile_shape, "type": "tactile_rgb"},
                    "robot1_tactile_right_image": {"shape": tactile_shape, "type": "tactile_rgb"},
                    "qpos": {"shape": [20], "type": "low_dim"},
                },
                "action": {"shape": [20]},
            }
        if self.action_representation == "pose10_rot6d_gripper":
            obs: dict[str, object] = {
                "robot0_eye_in_hand_image": {
                    "shape": [3, self.wrist_image_size, self.wrist_image_size],
                    "type": "rgb",
                },
                "robot0_eef_pos": {"shape": [3]},
                "robot0_eef_quat": {"shape": [4]},
                "robot0_gripper_qpos": {"shape": [2]},
            }
            obs["robot0_tactile_left_image"] = {
                "shape": [3, self.tactile_image_size, self.tactile_image_size],
                "type": "tactile_rgb",
            }
            return {"obs": obs, "action": {"shape": [10]}}
        return {
            "obs": {
                "cam_wrist": {
                    "shape": [3, self.wrist_image_size, self.wrist_image_size],
                    "type": "rgb",
                },
                "tac_left": {
                    "shape": [3, self.tactile_image_size, self.tactile_image_size],
                    "type": "tactile_rgb",
                },
                "tac_right": {
                    "shape": [3, self.tactile_image_size, self.tactile_image_size],
                    "type": "tactile_rgb",
                },
                "qpos": {"shape": [8]},
            },
            "action": {"shape": [8]},
        }


MODEL_SPECS: dict[str, VisuotactileModelSpec] = {
    "dp_bimanual": VisuotactileModelSpec(
        name="dp_bimanual",
        family="dp",
        action_representation="bimanual_pose20_rot6d_gripper",
        default_image_size=224,
        wrist_image_size=224,
        tactile_image_size=224,
        obs_horizon=2,
        action_horizon=8,
        vendor_subdir="DP",
        train_backend="diffusion_policy",
        camera_names=("rgb_wrist_left", "rgb_wrist_right"),
        tactile_names=("tactile_left", "tactile_right"),
    ),
}


def normalize_model_name(value: str) -> VisuotactileModelName:
    key = value.strip().lower().replace("-", "_")
    if key not in MODEL_SPECS:
        supported = ", ".join(sorted(MODEL_SPECS))
        raise ValueError(f"Unsupported visuotactile model {value!r}. Supported models: {supported}")
    return key  # type: ignore[return-value]


def get_model_spec(value: str) -> VisuotactileModelSpec:
    return MODEL_SPECS[normalize_model_name(value)]


def default_prepared_dataset_dir(
    workspace: WorkspaceSettings,
    task_name: str,
    dataset_name: str = DEFAULT_DATASET_NAME,
    model: str | None = None,
) -> Path:
    base = Path(workspace.recording.prepared_root) / task_name / "visuotactile" / dataset_name
    return base if model is None else base / get_model_spec(model).name


def default_preprocess1_dir(
    workspace: WorkspaceSettings,
    task_name: str,
    profile_name: str = DEFAULT_PREPROCESS1_PROFILE,
) -> Path:
    root = getattr(workspace.recording, "preprocess1_root", Path(workspace.recording.prepared_root).parent / "preprocess1")
    return Path(root) / task_name / profile_name


def default_checkpoint_dir(
    workspace: WorkspaceSettings,
    *,
    task_name: str,
    model: str,
    run_name: str | None = None,
) -> Path:
    spec = get_model_spec(model)
    base = Path(workspace.recording.checkpoints_root) / task_name / spec.name
    return base if run_name is None else base / run_name


class VisuotactilePolicySettings(BaseModel):
    model: str
    family: str = DEFAULT_POLICY_FAMILY
    policy_name: str | None = None
    task_name: str | None = None
    checkpoint_dir: Path | None = None
    checkpoint_file: Path | None = None
    device: Literal["auto", "cpu", "cuda"] = "auto"
    preprocess1_profile: str = DEFAULT_PREPROCESS1_PROFILE
    target_duration_sec: float | None = None
    gripper_open_width_m: float | None = None
    gripper_close_threshold: float = 0.5
    temporal_agg: bool = False
    temporal_agg_k: float = 0.01
    act_temporal_agg: bool = False
    act_temporal_agg_k: float = 0.01
    sampling_scheduler: Literal["checkpoint", "ddpm", "ddim"] = "checkpoint"
    num_inference_steps: int | None = None

    @field_validator("gripper_close_threshold")
    @classmethod
    def _validate_gripper_close_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("gripper_close_threshold must be in [0, 1]")
        return value

    @field_validator("target_duration_sec")
    @classmethod
    def _validate_target_duration_sec(cls, value: float | None) -> float | None:
        if value is not None and value <= 0.0:
            raise ValueError("target_duration_sec must be positive")
        return value

    @field_validator("temporal_agg_k", "act_temporal_agg_k")
    @classmethod
    def _validate_temporal_agg_k(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("temporal aggregation k must be non-negative")
        return value

    @field_validator("num_inference_steps")
    @classmethod
    def _validate_num_inference_steps(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("num_inference_steps must be positive")
        return value

    @field_validator("sampling_scheduler", mode="before")
    @classmethod
    def _normalize_sampling_scheduler(cls, value: str) -> str:
        return str(value or "checkpoint").strip().lower()

    @model_validator(mode="after")
    def _normalize_model(self) -> "VisuotactilePolicySettings":
        self.model = normalize_model_name(self.model)
        self.family = str(self.family or DEFAULT_POLICY_FAMILY).strip().lower()
        if not self.policy_name:
            self.policy_name = self.model
        return self

    @classmethod
    def from_policy_config(
        cls,
        policy_config: PolicyConfig,
        workspace: WorkspaceSettings,
        *,
        fallback_task_name: str,
    ) -> tuple["VisuotactilePolicySettings", Path]:
        settings = cls.model_validate(policy_config.config)
        task_name = settings.task_name or fallback_task_name
        checkpoint_path = policy_config.checkpoint_path
        if checkpoint_path is None and settings.checkpoint_dir is not None:
            checkpoint_path = settings.checkpoint_dir
        if checkpoint_path is None:
            checkpoint_path = default_checkpoint_dir(
                workspace,
                task_name=task_name,
                model=settings.model,
            )
        settings.task_name = task_name
        return settings, Path(checkpoint_path)
