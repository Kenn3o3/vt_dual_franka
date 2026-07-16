from __future__ import annotations

from ..config import InferenceRuntimeSettings, PolicyConfig, WorkspaceSettings
from .base import Policy


def resolve_policy(
    policy_config: PolicyConfig,
    inference_config: InferenceRuntimeSettings,
    workspace: WorkspaceSettings,
) -> Policy:
    policy_type = policy_config.type.strip().lower()
    if policy_type == "dp_bimanual":
        from .common.visuotactile.bimanual_policy import BimanualVisuotactilePolicy
        from .common.visuotactile.config import VisuotactilePolicySettings

        settings, checkpoint_path = VisuotactilePolicySettings.from_policy_config(
            policy_config,
            workspace,
            fallback_task_name=inference_config.task_name,
        )
        return BimanualVisuotactilePolicy(settings, checkpoint_path, gripper_open_width_m=workspace.teleop.max_gripper_width)
    raise ValueError(
        f"Unsupported policy type {policy_config.type!r}; vt_dual_franka supports only 'dp_bimanual'"
    )
