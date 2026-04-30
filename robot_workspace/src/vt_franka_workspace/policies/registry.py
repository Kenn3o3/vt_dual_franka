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
    raise ValueError(f"Unsupported policy type: {policy_config.type!r}")
