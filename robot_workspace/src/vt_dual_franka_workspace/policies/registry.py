from __future__ import annotations

from ..config import InferenceRuntimeSettings, PolicyConfig, WorkspaceSettings
from .base import Policy


def resolve_policy(
    policy_config: PolicyConfig,
    inference_config: InferenceRuntimeSettings,
    workspace: WorkspaceSettings,
) -> Policy:
    policy_type = policy_config.type.strip().lower()
    if policy_type == "replay":
        from .replay.policy import ReplayPolicy

        return ReplayPolicy.from_config(policy_config, inference_config, workspace)
    if policy_type == "mpd":
        from .mpd.policy import MPDPolicy

        return MPDPolicy.from_config(policy_config, inference_config, workspace)
    if policy_type in {"visuotactile", "vt", "univtac"}:
        from .visuotactile.policy import VisuotactilePolicy

        return VisuotactilePolicy.from_config(policy_config, inference_config, workspace)
    if policy_type in {"bimanual_visuotactile", "bimanual_vt", "dp_bimanual"}:
        from .common.visuotactile.bimanual_policy import BimanualVisuotactilePolicy
        from .common.visuotactile.config import VisuotactilePolicySettings

        settings, checkpoint_path = VisuotactilePolicySettings.from_policy_config(
            policy_config,
            workspace,
            fallback_task_name=inference_config.task_name,
        )
        return BimanualVisuotactilePolicy(settings, checkpoint_path, gripper_open_width_m=workspace.teleop.max_gripper_width)
    raise ValueError(f"Unsupported policy type: {policy_config.type!r}")
