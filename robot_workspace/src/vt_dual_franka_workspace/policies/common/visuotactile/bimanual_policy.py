from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from vt_dual_franka_shared.models import ControllerState

from ....policies.base import Policy
from .bimanual_runtime import ARM_ORDER, bimanual_states_to_20d, decode_bimanual_20d_action
from .config import VisuotactilePolicySettings, get_model_spec
from .runtime import TorchScriptVisuotactileBackend, load_runtime_manifests
from .vendor_dp_runtime import VendorDPCheckpointBackend, can_load_vendor_dp_checkpoint


class BimanualVisuotactilePolicy(Policy):
    def __init__(
        self,
        settings: VisuotactilePolicySettings,
        checkpoint_dir: Path,
        *,
        gripper_open_width_m: float = 0.078,
    ) -> None:
        self.settings = settings
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_spec = get_model_spec(settings.model)
        if self.model_spec.name != "dp_bimanual":
            raise ValueError("BimanualVisuotactilePolicy currently supports only model='dp_bimanual'")
        self.manifests = load_runtime_manifests(self.checkpoint_dir)
        if can_load_vendor_dp_checkpoint(
            self.checkpoint_dir,
            manifests=self.manifests,
            checkpoint_file=settings.checkpoint_file,
        ):
            self.backend = VendorDPCheckpointBackend(
                self.checkpoint_dir,
                device=settings.device,
                manifests=self.manifests,
                checkpoint_file=settings.checkpoint_file,
                temporal_agg=settings.temporal_agg,
                temporal_agg_k=settings.temporal_agg_k,
            )
        else:
            self.backend = TorchScriptVisuotactileBackend(
                self.checkpoint_dir,
                device=settings.device,
                manifests=self.manifests,
            )
        self.gripper_open_width_m = float(gripper_open_width_m)
        self.target_duration_sec = float(settings.target_duration_sec or 0.1)
        self.gripper_close_threshold = float(settings.gripper_close_threshold)

    def predict(self, observation_window: list[dict[str, Any]]) -> list[dict[str, Any]]:
        inputs = self._build_inputs(observation_window)
        chunk = self.backend.predict_action_chunk(inputs)
        actions: list[dict[str, Any]] = []
        for row in chunk:
            decoded = decode_bimanual_20d_action(np.asarray(row, dtype=np.float64))
            gripper_closed = {
                arm: decoded.gripper_closedness[arm] >= self.gripper_close_threshold for arm in ARM_ORDER
            }
            actions.append(
                {
                    "target_tcp_by_arm": decoded.target_tcp,
                    "target_duration_sec": self.target_duration_sec,
                    "gripper_closed_by_arm": gripper_closed,
                    "gripper_width_by_arm": {
                        arm: 0.0 if gripper_closed[arm] else self.gripper_open_width_m for arm in ARM_ORDER
                    },
                    "metadata": {
                        **decoded.metadata,
                        "visuotactile_model": self.model_spec.name,
                        "gripper_closedness": decoded.gripper_closedness,
                        "gripper_closed": gripper_closed,
                    },
                }
            )
        return actions

    def ensure_loaded(self) -> None:
        self.backend.ensure_loaded()

    def close(self) -> None:
        self.backend.close()

    def _build_inputs(self, observation_window: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        rgb_left = []
        rgb_right = []
        tactile_left = []
        tactile_right = []
        qpos = []
        for observation in observation_window:
            rgb_left.append(_image_array(observation, "images", "left_wrist"))
            rgb_right.append(_image_array(observation, "images", "right_wrist"))
            tactile_left.append(_image_array(observation, "tactile", "left"))
            tactile_right.append(_image_array(observation, "tactile", "right"))
            states = observation.get("proprioception", {}).get("controller_state_by_arm")
            if not isinstance(states, dict):
                raise ValueError("Bimanual policy requires proprioception.controller_state_by_arm")
            state_by_arm = {arm: ControllerState.model_validate(states[arm]) for arm in ARM_ORDER}
            qpos.append(bimanual_states_to_20d(state_by_arm, gripper_open_width_m=self.gripper_open_width_m))
        return {
            "rgb_wrist_left": np.stack(rgb_left, axis=0).astype(np.float32) / 255.0,
            "rgb_wrist_right": np.stack(rgb_right, axis=0).astype(np.float32) / 255.0,
            "tactile_left": np.stack(tactile_left, axis=0).astype(np.float32) / 255.0,
            "tactile_right": np.stack(tactile_right, axis=0).astype(np.float32) / 255.0,
            "qpos": np.stack(qpos, axis=0).astype(np.float32),
        }


def _image_array(observation: dict[str, Any], group: str, key: str) -> np.ndarray:
    item = observation.get(group, {}).get(key)
    if not isinstance(item, dict) or "image" not in item:
        raise ValueError(f"Observation missing {group}.{key}.image")
    return np.asarray(item["image"])
