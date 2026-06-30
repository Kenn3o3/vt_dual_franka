from __future__ import annotations

from pathlib import Path
from typing import Any

from ....config import InferenceRuntimeSettings, PolicyConfig, WorkspaceSettings
from ...base import Policy
from .config import VisuotactilePolicySettings, get_model_spec
from .runtime import (
    RuntimeManifests,
    RuntimePreprocessor,
    TorchScriptVisuotactileBackend,
    VisuotactileBackend,
    action_row_to_vt_action,
    load_runtime_manifests,
    resolve_runtime_checkpoint_dir,
)


class VisuotactilePolicy(Policy):
    def __init__(
        self,
        settings: VisuotactilePolicySettings,
        checkpoint_dir: Path,
        inference_config: InferenceRuntimeSettings,
        workspace: WorkspaceSettings,
        *,
        backend: VisuotactileBackend | None = None,
    ) -> None:
        self.settings = settings
        self.checkpoint_dir = resolve_runtime_checkpoint_dir(Path(checkpoint_dir))
        self.inference_config = inference_config
        self.workspace = workspace
        self.model_spec = get_model_spec(settings.model)
        self.checkpoint_path = self.checkpoint_dir
        self.manifests = _resolve_manifests(self.checkpoint_dir, backend=backend)
        self.backend = backend or _build_backend(self.checkpoint_dir, settings=settings, manifests=self.manifests)
        self.target_duration_sec = settings.target_duration_sec or (1.0 / max(inference_config.control_hz, 1e-8))
        self.gripper_open_width_m = (
            float(settings.gripper_open_width_m)
            if settings.gripper_open_width_m is not None
            else float(workspace.teleop.max_gripper_width)
        )
        self.preprocessor = RuntimePreprocessor(
            self.manifests,
            gripper_open_width_m=self.gripper_open_width_m,
            force_gripper_closedness=inference_config.gripper_forever_closed,
        )

    def ensure_loaded(self) -> None:
        self.backend.ensure_loaded()

    def reset(self) -> None:
        reset = getattr(self.backend, "reset", None)
        if reset is not None:
            reset()

    @classmethod
    def from_config(
        cls,
        policy_config: PolicyConfig,
        inference_config: InferenceRuntimeSettings,
        workspace: WorkspaceSettings,
    ) -> "VisuotactilePolicy":
        settings, checkpoint_dir = VisuotactilePolicySettings.from_policy_config(
            policy_config,
            workspace,
            fallback_task_name=inference_config.task_name,
        )
        return cls(settings, checkpoint_dir, inference_config, workspace)

    def predict(self, observation_window: list[dict[str, Any]]) -> list[dict[str, Any]]:
        inputs = self.build_model_inputs(observation_window)
        return self.predict_from_model_inputs(inputs)

    def build_model_inputs(self, observation_window: list[dict[str, Any]]) -> dict[str, Any]:
        return self.preprocessor.observation_window_to_model_inputs(
            observation_window,
            model_spec=self.model_spec,
        )

    def predict_from_model_inputs(self, inputs: dict[str, Any]) -> list[dict[str, Any]]:
        action_chunk = self.backend.predict_action_chunk(inputs)
        return [
            action_row_to_vt_action(
                row,
                model_spec=self.model_spec,
                target_duration_sec=self.target_duration_sec,
                gripper_open_width_m=self.gripper_open_width_m,
                gripper_close_threshold=self.settings.gripper_close_threshold,
            )
            for row in action_chunk
        ]

    def close(self) -> None:
        self.backend.close()


def _resolve_manifests(
    checkpoint_dir: Path,
    *,
    backend: VisuotactileBackend | None,
) -> RuntimeManifests:
    manifests = getattr(backend, "manifests", None) if backend is not None else None
    if manifests is not None:
        return manifests
    return load_runtime_manifests(checkpoint_dir)


def _build_backend(
    checkpoint_dir: Path,
    *,
    settings: VisuotactilePolicySettings,
    manifests: RuntimeManifests,
) -> VisuotactileBackend:
    torchscript_artifact = Path(checkpoint_dir) / "model_torchscript.pt"
    if torchscript_artifact.exists():
        return TorchScriptVisuotactileBackend(
            checkpoint_dir,
            device=settings.device,
            manifests=manifests,
        )
    from .vendor_dp_runtime import VendorDPCheckpointBackend, can_load_vendor_dp_checkpoint

    if can_load_vendor_dp_checkpoint(checkpoint_dir, manifests, checkpoint_file=settings.checkpoint_file):
        return VendorDPCheckpointBackend(
            checkpoint_dir,
            device=settings.device,
            manifests=manifests,
            checkpoint_file=settings.checkpoint_file,
            temporal_agg=settings.temporal_agg,
            temporal_agg_k=settings.temporal_agg_k,
        )
    from .vendor_act_runtime import VendorACTCheckpointBackend, can_load_vendor_act_checkpoint

    if can_load_vendor_act_checkpoint(checkpoint_dir, manifests):
        return VendorACTCheckpointBackend(
            checkpoint_dir,
            device=settings.device,
            manifests=manifests,
            temporal_agg=settings.act_temporal_agg,
            temporal_agg_k=settings.act_temporal_agg_k,
        )
    from .vendor_vista_runtime import VendorVISTACheckpointBackend, can_load_vendor_vista_checkpoint

    if can_load_vendor_vista_checkpoint(checkpoint_dir, manifests, checkpoint_file=settings.checkpoint_file):
        return VendorVISTACheckpointBackend(
            checkpoint_dir,
            device=settings.device,
            manifests=manifests,
            checkpoint_file=settings.checkpoint_file,
            temporal_agg=settings.temporal_agg,
            temporal_agg_k=settings.temporal_agg_k,
            sampling_scheduler=settings.sampling_scheduler,
            num_inference_steps=settings.num_inference_steps,
        )
    return TorchScriptVisuotactileBackend(
        checkpoint_dir,
        device=settings.device,
        manifests=manifests,
    )
