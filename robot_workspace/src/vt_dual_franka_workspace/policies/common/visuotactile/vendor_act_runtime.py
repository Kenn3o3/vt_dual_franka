from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import VisuotactileModelSpec, get_model_spec
from .runtime import RuntimeManifests


_ACT_CAMERA_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_ACT_CAMERA_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def can_load_vendor_act_checkpoint(checkpoint_dir: Path, manifests: RuntimeManifests) -> bool:
    model = manifests.policy.get("model")
    if model not in {"act_univtac", "vital_act"}:
        return False
    return _resolve_act_artifact_dir(Path(checkpoint_dir), str(model), missing_ok=True) is not None


class VendorACTCheckpointBackend:
    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        device: str = "auto",
        manifests: RuntimeManifests | None = None,
        temporal_agg: bool = False,
        temporal_agg_k: float = 0.01,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        if manifests is None:
            from .runtime import load_runtime_manifests

            manifests = load_runtime_manifests(self.checkpoint_dir)
        self.manifests = manifests
        self.model_spec = get_model_spec(self.manifests.policy["model"])
        self.artifact_dir = _resolve_act_artifact_dir(self.checkpoint_dir, self.model_spec.name)
        self.obs_horizon = int(self.manifests.policy.get("obs_horizon", self.model_spec.obs_horizon))
        self.action_horizon = int(self.manifests.policy.get("action_horizon", self.model_spec.action_horizon))
        self.action_dim = int(self.manifests.policy.get("action_dim", self.model_spec.action_dim))
        self._device_name = device
        self.temporal_agg = bool(temporal_agg)
        self.temporal_agg_k = float(temporal_agg_k)
        self._model = None

    def ensure_loaded(self) -> None:
        self._ensure_loaded()

    def close(self) -> None:
        self._model = None

    def reset(self) -> None:
        if self._model is not None:
            self._model.reset()

    def predict_action_chunk(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        self._ensure_loaded()
        assert self._model is not None
        obs = self._inputs_to_act_obs(inputs)
        if self.temporal_agg:
            return self._predict_temporal_agg_action(obs)
        # Query a fresh chunk for the current real robot observation window.
        self._model.reset()
        action = self._model.get_action(obs)
        normalized_actions = getattr(self._model, "all_actions", None)
        actions = (
            _postprocess_action_chunk(self._model, normalized_actions)
            if normalized_actions is not None
            else np.empty((0, self.action_dim), dtype=np.float64)
        )
        if actions.ndim != 2 or actions.shape[1] != self.action_dim:
            fallback = np.asarray(action, dtype=np.float64).reshape(1, -1)
            if fallback.shape[1] != self.action_dim:
                raise ValueError(f"Expected ACT action dim {self.action_dim}, got {fallback.shape}")
            actions = fallback
        actions = actions[: self.action_horizon].astype(np.float64)
        return _normalize_pose7_gripper_actions(actions, obs["qpos"])

    def _predict_temporal_agg_action(self, obs: dict[str, Any]) -> np.ndarray:
        assert self._model is not None
        action = np.asarray(self._model.get_action(obs), dtype=np.float64).reshape(1, -1)
        if action.shape[1] != self.action_dim:
            raise ValueError(f"Expected ACT action dim {self.action_dim}, got {action.shape}")
        normalized = _normalize_pose7_gripper_actions(action, obs["qpos"])
        raw_actions = getattr(self._model, "all_actions", None)
        metadata: dict[str, Any] = {
            "mode": "temporal_agg",
            "timestep": int(getattr(self._model, "t", 0)) - 1,
        }
        if raw_actions is not None:
            postprocessed = _postprocess_action_chunk(self._model, raw_actions)
            metadata["last_query_chunk_shape"] = list(postprocessed.shape)
        self._last_temporal_agg_metadata = metadata
        return normalized

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Torch is required to run ACT visuotactile checkpoints.") from exc
        args = self._load_policy_config()
        args["device"] = _resolve_device_name(self._device_name, torch)
        args.setdefault("task_name", self.manifests.policy.get("task_name", "task"))
        args["temporal_agg"] = self.temporal_agg
        args["temporal_agg_k"] = self.temporal_agg_k
        if self.model_spec.name == "vital_act":
            from vt_dual_franka_workspace.policies.ViTAL.clip_pretraining import modified_resnet18
            from vt_dual_franka_workspace.policies.ViTAL.policy import ACT

            args["ckpt_dir"] = str(self.artifact_dir)
            _fill_vital_act_runtime_args(args)
            self._model = ACT(args, [modified_resnet18(), modified_resnet18()])
            if self.temporal_agg:
                setattr(self._model, "temporal_agg_k", self.temporal_agg_k)
        else:
            from vt_dual_franka_workspace.policies.ACT.act_policy import ACT

            args["ckpt_dir"] = str(self.artifact_dir)
            args["ckpt_path"] = str(self._checkpoint_path())
            self._model = ACT(args)
            if self.temporal_agg:
                setattr(self._model, "temporal_agg_k", self.temporal_agg_k)

    def _load_policy_config(self) -> dict[str, Any]:
        args_path = self.artifact_dir / "args.json"
        if args_path.is_file():
            payload = json.loads(args_path.read_text(encoding="utf-8"))
            policy_config = payload.get("policy_config")
            if isinstance(policy_config, dict):
                return dict(policy_config)
            if isinstance(payload, dict):
                return dict(payload)
        if self.model_spec.name == "vital_act":
            config_path = self.checkpoint_dir / "vital_act_train_config.json"
            if config_path.is_file():
                payload = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    return dict(payload)
        config_path = self.checkpoint_dir / "act_train_config.yml"
        if config_path.is_file():
            import yaml

            payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return dict(payload)
        raise FileNotFoundError(f"Missing ACT runtime config in {self.checkpoint_dir}")

    def _checkpoint_path(self) -> Path:
        policy_best = self.artifact_dir / "policy_best.ckpt"
        if policy_best.is_file():
            return policy_best
        best = self.artifact_dir / "best.ckpt"
        if best.is_file():
            return best
        raise FileNotFoundError(f"Missing ACT checkpoint in {self.artifact_dir}")

    def _inputs_to_act_obs(self, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        rgb = np.asarray(inputs["rgb_wrist"], dtype=np.float32)
        gelsight = np.asarray(inputs["gelsight"], dtype=np.float32)
        qpos = np.asarray(inputs["qpos"], dtype=np.float32)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb_wrist must be [T,H,W,3], got {rgb.shape}")
        if gelsight.ndim != 4 or gelsight.shape[-1] != 3:
            raise ValueError(f"gelsight must be [T,H,W,3], got {gelsight.shape}")
        if qpos.ndim != 2 or qpos.shape[1] != self.action_dim:
            raise ValueError(f"qpos must be [T,{self.action_dim}] for ACT checkpoints, got {qpos.shape}")
        if self.model_spec.name == "vital_act":
            return _inputs_to_vital_act_obs(self._model, rgb[-1], gelsight[-1], qpos[-1])
        return _inputs_to_univtac_act_obs(rgb[-1], gelsight[-1], qpos[-1])


def _resolve_act_artifact_dir(checkpoint_dir: Path, model: str, *, missing_ok: bool = False) -> Path | None:
    candidates = [Path(checkpoint_dir)]
    if model == "vital_act":
        candidates.insert(0, Path(checkpoint_dir) / "vendor_vital_act_run")
    for candidate in candidates:
        if (candidate / "policy_best.ckpt").is_file() or (candidate / "best.ckpt").is_file():
            return candidate
    if missing_ok:
        return None
    raise FileNotFoundError(f"Missing {model} checkpoint artifacts in {checkpoint_dir}")


def _inputs_to_univtac_act_obs(rgb: np.ndarray, gelsight: np.ndarray, qpos: np.ndarray) -> dict[str, Any]:
    return {
        "qpos": qpos,
        "cam_wrist": _act_camera_tensor(rgb),
        "tac_left": _act_tactile_tensor(gelsight),
        "tac_right": _act_tactile_tensor(gelsight),
    }


def _inputs_to_vital_act_obs(model: Any, rgb: np.ndarray, gelsight: np.ndarray, qpos: np.ndarray) -> dict[str, Any]:
    camera = _act_camera_tensor(rgb)
    tactile = _vital_tactile_tensor(gelsight, stats=getattr(model, "stats", None))
    obs: dict[str, Any] = {"qpos": qpos}
    for name in getattr(model, "camera_names", ["cam_wrist", "cam_left_tactile", "cam_right_tactile"]):
        if name in {"cam_wrist", "cam_high"}:
            obs[name] = camera
        elif name in {"cam_left_tactile", "cam_right_tactile"}:
            obs[name] = tactile
        else:
            raise KeyError(f"Unsupported ViTAL ACT camera name: {name}")
    return obs


def _act_camera_tensor(image: np.ndarray):
    import torch

    chw = _unit_hwc_to_chw_array(image)
    chw = (chw - _ACT_CAMERA_MEAN) / _ACT_CAMERA_STD
    return torch.from_numpy(chw.copy()).float()


def _act_tactile_tensor(image: np.ndarray):
    import torch

    return torch.from_numpy(_unit_hwc_to_chw_array(image).copy()).float()


def _vital_tactile_tensor(image: np.ndarray, *, stats: Any):
    import torch

    mean, std = _vital_tactile_stats(stats)
    chw = _unit_hwc_to_chw_array(image)
    chw = (chw - mean.reshape(3, 1, 1)) / std.reshape(3, 1, 1)
    return torch.from_numpy(chw.copy()).float()


def _vital_tactile_stats(stats: Any) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(stats, dict):
        if "gelsight_mean" in stats and "gelsight_std" in stats:
            mean = np.asarray(stats["gelsight_mean"], dtype=np.float32)
            std = np.asarray(stats["gelsight_std"], dtype=np.float32)
            return mean, np.clip(std, 1e-2, np.inf)
        if all(key in stats for key in ("left_tac_mean", "left_tac_std", "right_tac_mean", "right_tac_std")):
            mean = (
                np.asarray(stats["left_tac_mean"], dtype=np.float32)
                + np.asarray(stats["right_tac_mean"], dtype=np.float32)
            ) / 2.0
            std = (
                np.asarray(stats["left_tac_std"], dtype=np.float32)
                + np.asarray(stats["right_tac_std"], dtype=np.float32)
            ) / 2.0
            return mean, np.clip(std, 1e-2, np.inf)
    return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)


def _fill_vital_act_runtime_args(args: dict[str, Any]) -> None:
    if "camera_names" not in args:
        camera = list(args.get("camera", ["cam_wrist"]))
        tactile = list(args.get("tactile", ["cam_left_tactile", "cam_right_tactile"]))
        args["camera_names"] = camera + tactile
    if "state_dim" not in args:
        args["state_dim"] = 8
    if "cam_backbone_mapping" not in args:
        mapping = {name: 0 for name in args["camera_names"]}
        for name in args["camera_names"]:
            if name.endswith("tactile"):
                mapping[name] = 1
        args["cam_backbone_mapping"] = mapping


def _postprocess_action_chunk(model: Any, normalized_actions: Any) -> np.ndarray:
    if hasattr(normalized_actions, "detach"):
        normalized_actions = normalized_actions.detach().cpu().numpy()
    actions = np.asarray(normalized_actions, dtype=np.float64)
    if actions.ndim == 3:
        actions = actions[0]
    if hasattr(model, "post_process"):
        actions = np.asarray(model.post_process(actions), dtype=np.float64)
    return actions


def _unit_hwc_to_chw_array(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    if image.max(initial=0.0) > 1.0:
        image = image / 255.0
    return np.transpose(image, (2, 0, 1))


def _resolve_device_name(device_name: str, torch_module: Any) -> str:
    value = str(device_name).strip().lower()
    if value == "auto":
        return "cuda:0" if torch_module.cuda.is_available() else "cpu"
    if value == "cuda":
        return "cuda:0"
    return str(device_name)


def _normalize_pose7_gripper_actions(actions: np.ndarray, qpos: np.ndarray) -> np.ndarray:
    normalized = np.asarray(actions, dtype=np.float64).copy()
    ref_quat = _unit_quat(np.asarray(qpos, dtype=np.float64)[3:7])
    if ref_quat is None:
        ref_quat = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    for row in normalized:
        quat = _unit_quat(row[3:7])
        if quat is None:
            quat = ref_quat
        elif float(np.dot(quat, ref_quat)) < 0.0:
            quat = -quat
        row[3:7] = quat
        row[7] = float(np.clip(row[7], 0.0, 1.0))
    return normalized


def _unit_quat(quat: np.ndarray) -> np.ndarray | None:
    values = np.asarray(quat, dtype=np.float64).reshape(-1)
    if values.shape != (4,) or not np.isfinite(values).all():
        return None
    norm = float(np.linalg.norm(values))
    if norm <= 1e-8:
        return None
    return values / norm
