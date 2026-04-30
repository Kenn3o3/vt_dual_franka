from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ...config import InferenceRuntimeSettings, PolicyConfig, WorkspaceSettings
from ..base import Policy
from .config import MPDPolicySettings, get_policy_spec
from .math import (
    gripper_width_to_closedness,
    pose7d_and_gripper_to_tcp_state,
    tcp_state_to_pose7d_and_gripper,
)


@dataclass(frozen=True)
class MPDRuntimeSpec:
    obs_horizon: int
    prediction_horizon: int
    action_dim: int
    observation_keys: tuple[str, ...]
    required_history_keys: tuple[str, ...]
    dt: float


class MPDBackend(Protocol):
    runtime_spec: MPDRuntimeSpec

    def predict_action_chunk(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        ...

    def close(self) -> None:
        ...


class MPDPolicy(Policy):
    def __init__(
        self,
        settings: MPDPolicySettings,
        checkpoint_path: Path,
        inference_config: InferenceRuntimeSettings,
        workspace: WorkspaceSettings,
        *,
        backend: MPDBackend | None = None,
    ) -> None:
        self.settings = settings
        self.checkpoint_path = Path(checkpoint_path)
        self.inference_config = inference_config
        self.workspace = workspace
        self.backend = backend or HydraMPDBackend(settings, self.checkpoint_path)
        self.gripper_open_width_m = (
            float(settings.gripper_open_width_m)
            if settings.gripper_open_width_m is not None
            else float(workspace.teleop.max_gripper_width)
        )
        self.target_duration_sec = settings.target_duration_sec or (1.0 / max(inference_config.control_hz, 1e-8))
        self._executed_action_history: deque[np.ndarray] = deque(maxlen=max(self.backend.runtime_spec.obs_horizon, 1))
        self._action_history_initialized = False

    @classmethod
    def from_config(
        cls,
        policy_config: PolicyConfig,
        inference_config: InferenceRuntimeSettings,
        workspace: WorkspaceSettings,
    ) -> "MPDPolicy":
        settings, checkpoint_path = MPDPolicySettings.from_policy_config(
            policy_config,
            workspace,
            fallback_task_name=inference_config.task_name,
        )
        return cls(settings, checkpoint_path, inference_config, workspace)

    def reset(self) -> None:
        self._executed_action_history.clear()
        self._action_history_initialized = False

    def start_episode(self, observation_window: list[dict[str, Any]]) -> None:
        required = set(self.backend.runtime_spec.required_history_keys)
        if "action" not in required and "action_vel" not in required:
            self._action_history_initialized = True
            return
        states = self._states_from_observation_window(observation_window)
        self._executed_action_history.clear()
        for state in states:
            self._executed_action_history.append(state)
        self._action_history_initialized = True

    def close(self) -> None:
        self.backend.close()

    def predict(self, observation_window: list[dict[str, Any]]) -> list[dict[str, Any]]:
        states = self._states_from_observation_window(observation_window)
        required = set(self.backend.runtime_spec.required_history_keys)
        model_inputs: dict[str, np.ndarray] = {"agent_pos": states}
        if "action" in required or "action_vel" in required:
            actions = self._action_history_for_window(states)
            model_inputs["action"] = actions
        action_chunk = self.backend.predict_action_chunk(model_inputs)
        if action_chunk.ndim != 2 or action_chunk.shape[1] != self.backend.runtime_spec.action_dim:
            raise ValueError(
                f"MPD backend returned {action_chunk.shape}; expected [T, {self.backend.runtime_spec.action_dim}]"
            )
        return [self._row_to_action(row) for row in action_chunk]

    def observe_executed_actions(self, actions: list[dict[str, Any]]) -> None:
        for action in actions:
            metadata = action.get("metadata") or {}
            if "mpd_tcp_state" in metadata:
                self._executed_action_history.append(np.asarray(metadata["mpd_tcp_state"], dtype=np.float64))
                continue
            target_tcp = action.get("target_tcp")
            if target_tcp is None:
                continue
            closedness = 1.0 if action.get("gripper_closed") is True else 0.0
            if action.get("gripper_width") is not None:
                closedness = gripper_width_to_closedness(
                    float(action["gripper_width"]),
                    open_width_m=self.gripper_open_width_m,
                )
            self._executed_action_history.append(pose7d_and_gripper_to_tcp_state(target_tcp, closedness))

    def _states_from_observation_window(self, observation_window: list[dict[str, Any]]) -> np.ndarray:
        if not observation_window:
            raise ValueError("MPD policy requires a non-empty observation window")
        horizon = self.backend.runtime_spec.obs_horizon
        padded = list(observation_window)
        while len(padded) < horizon:
            padded.insert(0, padded[0])
        padded = padded[-horizon:]
        return np.stack([self._state_from_observation(item) for item in padded], axis=0).astype(np.float64)

    def _state_from_observation(self, observation: dict[str, Any]) -> np.ndarray:
        controller_state = observation.get("proprioception", {}).get("controller_state")
        if not isinstance(controller_state, dict):
            raise ValueError("MPD policy requires observation['proprioception']['controller_state']")
        tcp_pose = controller_state.get("tcp_pose")
        if tcp_pose is None:
            raise ValueError("MPD policy requires controller_state.tcp_pose")
        closedness = gripper_width_to_closedness(
            float(controller_state.get("gripper_width", self.gripper_open_width_m)),
            open_width_m=self.gripper_open_width_m,
        )
        return pose7d_and_gripper_to_tcp_state(tcp_pose, closedness)

    def _action_history_for_window(self, states: np.ndarray) -> np.ndarray:
        if not self._action_history_initialized:
            raise RuntimeError("MPD action history is not initialized. PolicyRunner must call policy.start_episode().")
        horizon = states.shape[0]
        history = list(self._executed_action_history)[-horizon:]
        if len(history) != horizon:
            raise RuntimeError(f"MPD action history has {len(history)} steps; expected {horizon}")
        return np.stack(history[-horizon:], axis=0).astype(np.float64)

    def _row_to_action(self, row: np.ndarray) -> dict[str, Any]:
        pose7d, closedness = tcp_state_to_pose7d_and_gripper(row)
        action: dict[str, Any] = {
            "target_tcp": pose7d.astype(float).tolist(),
            "target_duration_sec": self.target_duration_sec,
            "metadata": {
                "mpd_tcp_state": np.asarray(row, dtype=np.float64).tolist(),
                "mpd_algorithm": self.settings.algorithm,
            },
        }
        if closedness >= self.settings.gripper_close_threshold:
            action["gripper_closed"] = True
            action["gripper_velocity"] = self.workspace.teleop.gripper_velocity
            action["gripper_force_limit"] = self.workspace.teleop.grasp_force
        else:
            action["gripper_width"] = self.gripper_open_width_m
            action["gripper_velocity"] = self.workspace.teleop.gripper_velocity
            action["gripper_force_limit"] = self.workspace.teleop.grasp_force
        return action


class HydraMPDBackend:
    def __init__(self, settings: MPDPolicySettings, checkpoint_path: Path) -> None:
        self.settings = settings
        self.checkpoint_path = Path(checkpoint_path)
        self._cfg = None
        self._agent = None
        self._device = "cpu"
        self._scaler_values = None
        self._runtime_spec: MPDRuntimeSpec | None = None

    @property
    def runtime_spec(self) -> MPDRuntimeSpec:
        if self._runtime_spec is None:
            cfg = self._load_config(require_checkpoint_artifacts=True)
            _patch_config_for_inference(cfg)
            self._runtime_spec = self._runtime_spec_from_config(cfg)
        return self._runtime_spec

    def predict_action_chunk(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        self._ensure_loaded()
        import torch
        from movement_primitive_diffusion.datasets.scalers import denormalize, normalize

        assert self._agent is not None
        assert self._cfg is not None
        assert self._scaler_values is not None
        observation_buffer: dict[str, Any] = {}
        required = set(self.runtime_spec.required_history_keys)
        for key, values in inputs.items():
            tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(self._device)
            if key in {"agent_pos", "action"}:
                tensor = normalize(tensor, self._scaler_values[key], symmetric=True)
            observation_buffer[key] = tensor.unsqueeze(0)
        if "agent_vel" in required and "agent_vel" not in observation_buffer:
            observation_buffer["agent_vel"] = _finite_difference_tensor(
                observation_buffer["agent_pos"],
                dt=self.runtime_spec.dt,
            )
        if "action_vel" in required and "action_vel" not in observation_buffer:
            observation_buffer["action_vel"] = _finite_difference_tensor(
                observation_buffer["action"],
                dt=self.runtime_spec.dt,
            )
        observation, extra_inputs = self._agent.process_batch.process_env_observation(observation_buffer)
        observation = {key: value.to(self._device) for key, value in observation.items()}
        for key, value in extra_inputs.items():
            if isinstance(value, torch.Tensor):
                extra_inputs[key] = value.to(self._device)
        with torch.no_grad():
            action = self._agent.predict(observation, extra_inputs=extra_inputs)[0].detach()
        if "action" in self._scaler_values:
            action = denormalize(action, self._scaler_values["action"], symmetric=True)
        return action.cpu().numpy().astype(np.float64)

    def close(self) -> None:
        self._agent = None

    def _ensure_loaded(self) -> None:
        if self._agent is not None:
            return
        _ensure_upstream_repo_on_path(self.settings.upstream_repo_dir)
        import hydra
        import torch
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        cfg = self._load_config(require_checkpoint_artifacts=True)
        _patch_config_for_inference(cfg)
        self._validate_checkpoint_config(cfg)
        self._device = _resolve_device(self.settings.device, torch)
        _patch_config_device(cfg, self._device)
        if hasattr(cfg.agent_config.lr_scheduler_config, "num_training_steps"):
            cfg.agent_config.lr_scheduler_config.num_training_steps = 1
        self._cfg = cfg
        self._runtime_spec = self._runtime_spec_from_config(cfg)
        self._scaler_values = self._load_scaler_values(torch)
        self._agent = hydra.utils.instantiate(cfg.agent_config)
        _patch_process_batch_runtime(self._agent.process_batch, cfg.agent_config.process_batch_config)
        _load_pretrained_for_inference(self._agent, self.checkpoint_path, torch)
        self._agent.model.eval()
        if hasattr(self._agent, "encoder"):
            self._agent.encoder.eval()

    def _runtime_spec_from_config(self, cfg: Any) -> MPDRuntimeSpec:
        process_config = cfg.agent_config.process_batch_config
        observation_keys = tuple(str(item.observation_key) for item in cfg.agent_config.encoder_config.network_configs)
        required_history_keys = set(observation_keys)
        required_history_keys.update(_as_tuple(getattr(process_config, "initial_position_keys", ())))
        required_history_keys.update(_as_tuple(getattr(process_config, "initial_velocity_keys", ())))
        action_dim = int(sum(int(size) for item in process_config.action_keys for size in item.feature_size))
        return MPDRuntimeSpec(
            obs_horizon=int(cfg.t_obs),
            prediction_horizon=int(cfg.t_pred),
            action_dim=action_dim,
            observation_keys=observation_keys,
            required_history_keys=tuple(sorted(required_history_keys)),
            dt=float(cfg.dataset_config.dt),
        )

    def _load_config(self, *, require_checkpoint_artifacts: bool):
        run_dir = self.checkpoint_path if self.checkpoint_path.is_dir() else self.checkpoint_path.parent
        resolved_path = run_dir / "resolved_config.yaml"
        if resolved_path.exists():
            from omegaconf import OmegaConf

            cfg = OmegaConf.load(resolved_path)
            OmegaConf.resolve(cfg)
            return cfg
        if require_checkpoint_artifacts:
            raise FileNotFoundError(f"Missing MPD checkpoint artifact: {resolved_path}")
        return _compose_upstream_config(self.settings)

    def _load_scaler_values(self, torch_module: Any) -> dict[str, dict[str, Any]]:
        run_dir = self.checkpoint_path if self.checkpoint_path.is_dir() else self.checkpoint_path.parent
        scaler_path = run_dir / "scaler_values.npz"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing MPD checkpoint artifact: {scaler_path}")
        raw = np.load(scaler_path)
        values: dict[str, dict[str, Any]] = {}
        for full_key in raw.files:
            key_name, stat_name = full_key.rsplit("_", 1)
            values.setdefault(key_name, {})[stat_name] = torch_module.from_numpy(raw[full_key].astype(np.float32)).to(self._device)
        for key in ("agent_pos", "action"):
            if key not in values:
                raise RuntimeError(f"Missing scaler values for MPD key: {key}")
        return values

    def _validate_checkpoint_config(self, cfg: Any) -> None:
        spec = get_policy_spec(self.settings.algorithm)
        method_name = str(getattr(cfg, "method_name", "") or "")
        if method_name and method_name not in spec.method_names:
            expected = ", ".join(spec.method_names)
            raise RuntimeError(
                f"Policy/checkpoint mismatch: algorithm={self.settings.algorithm} expects method_name in [{expected}], "
                f"but resolved_config.yaml has {method_name!r}"
            )
        run_dir = self.checkpoint_path if self.checkpoint_path.is_dir() else self.checkpoint_path.parent
        manifest_path = run_dir / "dataset_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("action_convention") != "tcp_xyz_rot6d_gripper_closedness":
                raise RuntimeError("MPD checkpoint dataset manifest uses an unsupported action_convention")


def _ensure_upstream_repo_on_path(upstream_repo_dir: Path) -> None:
    repo_path = str(Path(upstream_repo_dir).resolve())
    if sys.path and sys.path[0] == repo_path:
        return
    sys.path = [path for path in sys.path if path != repo_path]
    sys.path.insert(0, repo_path)


def _compose_upstream_config(settings: MPDPolicySettings):
    _ensure_upstream_repo_on_path(settings.upstream_repo_dir)
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    experiment = settings.upstream_experiment or settings.task_name
    if not experiment:
        raise ValueError("MPD upstream_experiment or task_name is required to compose an upstream config")
    config_name = get_policy_spec(settings.algorithm).upstream_config_name(experiment)
    config_dir = Path(settings.upstream_repo_dir) / "conf"
    with hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = hydra.compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def _patch_config_for_inference(cfg: Any) -> None:
    cfg.agent_config.process_batch_config.observation_keys = [
        item.observation_key for item in cfg.agent_config.encoder_config.network_configs
    ]
    for item in cfg.agent_config.process_batch_config.action_keys:
        if item.feature_size is None:
            item.feature_size = [10]
    for network_config in cfg.agent_config.encoder_config.network_configs:
        if network_config.observation_key == "agent_pos" and network_config.feature_size is None:
            network_config.feature_size = [10]
        if hasattr(network_config.network_config, "feature_size") and network_config.network_config.feature_size is None:
            network_config.network_config.feature_size = network_config.feature_size


def _patch_config_device(cfg: Any, device: str) -> None:
    cfg.device = device
    cfg.agent_config.device = device
    process_config = cfg.agent_config.process_batch_config
    for attr in ("prodmp_handler_config", "motif_handler_config"):
        if hasattr(process_config, attr):
            getattr(process_config, attr).device = device
    inner_model_config = cfg.agent_config.model_config.inner_model_config
    for attr in ("prodmp_handler_config", "motif_handler_config"):
        if hasattr(inner_model_config, attr):
            getattr(inner_model_config, attr).device = device


def _patch_process_batch_runtime(process_batch: Any, process_config: Any) -> None:
    if hasattr(process_batch, "initial_position_keys"):
        process_batch.initial_position_keys = list(_as_tuple(getattr(process_config, "initial_position_keys", ())))
    if hasattr(process_batch, "initial_velocity_keys"):
        process_batch.initial_velocity_keys = list(_as_tuple(getattr(process_config, "initial_velocity_keys", ())))
    if hasattr(process_batch, "initial_values_come_from_action_data"):
        process_batch.initial_values_come_from_action_data = bool(
            getattr(process_config, "initial_values_come_from_action_data", False)
        )
        process_batch.initial_value_index = process_batch.t_obs - (
            2 if process_batch.initial_values_come_from_action_data else 1
        )


def _resolve_device(configured: str, torch_module: Any) -> str:
    if configured == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("MPD policy configured for CUDA, but torch.cuda.is_available() is false")
    return configured


def _load_pretrained_for_inference(agent: Any, checkpoint_path: Path, torch_module: Any) -> None:
    path = checkpoint_path / "best_model.pth" if checkpoint_path.is_dir() else checkpoint_path
    if not path.exists():
        raise FileNotFoundError(f"MPD checkpoint does not exist: {path}")
    state_dict = torch_module.load(path, map_location="cpu")
    agent.model.load_state_dict(_strip_compile_prefix(state_dict["model"]))
    agent.encoder.load_state_dict(_strip_compile_prefix(state_dict["encoder"]))
    if getattr(agent, "use_ema", False):
        agent.ema_model.load_state_dict(_strip_compile_prefix(state_dict["ema_model"]))
        agent.ema_encoder.load_state_dict(_strip_compile_prefix(state_dict["ema_encoder"]))


def _strip_compile_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key.removeprefix(prefix): value for key, value in state_dict.items()}


def _finite_difference_tensor(values: Any, *, dt: float):
    velocity = values.new_zeros(values.shape)
    if values.shape[1] > 1:
        velocity[:, 1:] = (values[:, 1:] - values[:, :-1]) / float(dt)
    return velocity


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(str(item) for item in value)
