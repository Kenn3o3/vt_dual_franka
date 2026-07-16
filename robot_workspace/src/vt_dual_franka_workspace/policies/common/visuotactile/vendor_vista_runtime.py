from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

from ...mpd.math import tcp_state_to_pose7d_and_gripper
from .config import get_model_spec
from .runtime import RuntimeManifests, _prepare_healpy_imports, load_runtime_manifests


POLICIES_ROOT = Path(__file__).resolve().parents[2]
VISTA_ROOT = POLICIES_ROOT / "VISTA"


class VendorVISTACheckpointBackend:
    """Run real VT_Franka inference from VISTA checkpoint files."""

    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        device: str = "auto",
        manifests: RuntimeManifests | None = None,
        checkpoint_file: str | Path | None = None,
        temporal_agg: bool = False,
        temporal_agg_k: float = 0.01,
        sampling_scheduler: str = "checkpoint",
        num_inference_steps: int | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.manifests = manifests or load_runtime_manifests(self.checkpoint_dir)
        self.model_spec = get_model_spec(self.manifests.policy["model"])
        if self.model_spec.name not in {"vista_so2", "vista_so3"}:
            raise ValueError(f"VendorVISTACheckpointBackend does not support {self.model_spec.name!r}")
        self.obs_horizon = int(self.manifests.policy.get("obs_horizon", self.model_spec.obs_horizon))
        self.action_horizon = int(self.manifests.policy.get("action_horizon", self.model_spec.action_horizon))
        self.action_dim = int(self.manifests.policy.get("action_dim", self.model_spec.action_dim))
        self._device_name = device
        self.temporal_agg = bool(temporal_agg)
        self.temporal_agg_k = float(temporal_agg_k)
        self.sampling_scheduler = str(sampling_scheduler or "checkpoint").strip().lower()
        self.num_inference_steps = None if num_inference_steps is None else int(num_inference_steps)
        self._temporal_action_history: dict[int, np.ndarray] = {}
        self._temporal_step_index = 0
        self._model = None
        self._torch = None
        self._image_shapes: dict[str, tuple[int, int, int]] = {}
        self._lowdim_keys: list[str] = []
        self._ckpt_path = _resolve_vista_checkpoint_path(self.checkpoint_dir, checkpoint_file=checkpoint_file)

    def predict_action_chunk(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        self._ensure_loaded()
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        obs = self._inputs_to_vendor_obs(inputs)
        with torch.inference_mode():
            output = self._model.predict_action(obs)
        action = output.get("action") if isinstance(output, dict) else output
        if action is None:
            raise RuntimeError("VISTA checkpoint returned no action output")
        action_np = action.detach().cpu().numpy()
        if action_np.ndim == 3:
            action_np = action_np[0]
        if action_np.ndim != 2 or action_np.shape[1] != self.action_dim:
            raise ValueError(f"Expected action chunk [T,{self.action_dim}], got {action_np.shape}")
        action_np = action_np[: self.action_horizon].astype(np.float64)
        if self.temporal_agg:
            return self._aggregate_temporal_action(action_np)[None, :]
        return action_np

    def close(self) -> None:
        self._model = None

    def ensure_loaded(self) -> None:
        self._ensure_loaded()

    def reset(self) -> None:
        self._temporal_action_history.clear()
        self._temporal_step_index = 0

    def _aggregate_temporal_action(self, predicted_actions: np.ndarray) -> np.ndarray:
        step_index = self._temporal_step_index
        self._temporal_step_index += 1
        predicted_actions = np.asarray(predicted_actions, dtype=np.float64)
        self._temporal_action_history[step_index] = predicted_actions

        candidates: list[np.ndarray] = []
        for source_step in sorted(self._temporal_action_history):
            chunk = self._temporal_action_history[source_step]
            offset = step_index - source_step
            if 0 <= offset < int(chunk.shape[0]):
                candidates.append(np.asarray(chunk[offset], dtype=np.float64))
        if not candidates:
            aggregated = predicted_actions[0]
        else:
            stacked = np.stack(candidates, axis=0)
            weights = np.exp(-self.temporal_agg_k * np.arange(len(candidates), dtype=np.float64))
            weights = weights / np.sum(weights)
            aggregated = (stacked * weights[:, None]).sum(axis=0)

        expired_steps = [
            source_step
            for source_step, chunk in self._temporal_action_history.items()
            if source_step + int(chunk.shape[0]) <= step_index + 1
        ]
        for source_step in expired_steps:
            self._temporal_action_history.pop(source_step, None)

        return aggregated.astype(np.float64)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        _prepare_vendor_imports()
        _patch_scipy_todense()
        try:
            import dill
            import hydra
            import torch
            from omegaconf import OmegaConf
        except ImportError as exc:
            raise RuntimeError(
                "Running VISTA visuotactile checkpoints requires the UniVTAC/VISTA "
                "training environment dependencies. Use the isp-derived real robot "
                "conda env, not the lightweight vt-dual-franka-workspace env."
            ) from exc

        _prepare_healpy_imports()
        payload = torch.load(open(self._ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
        cfg = payload["cfg"]
        model = hydra.utils.instantiate(cfg.policy)
        state_dicts = payload.get("state_dicts", {})
        if "ema_model" in state_dicts:
            model.load_state_dict(state_dicts["ema_model"])
        elif "model" in state_dicts:
            model.load_state_dict(state_dicts["model"])
        else:
            raise KeyError(f"Checkpoint has no model weights under state_dicts: {self._ckpt_path}")

        self._device = _resolve_device(self._device_name, torch)
        _configure_vista_sampling(
            model,
            scheduler_name=self.sampling_scheduler,
            num_inference_steps=self.num_inference_steps,
        )
        model.eval()
        model.to(self._device)

        shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
        if not isinstance(shape_meta, dict) or "obs" not in shape_meta:
            raise ValueError("VISTA checkpoint cfg.shape_meta is missing obs metadata")
        self._image_shapes = {}
        self._lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type in {"rgb", "tactile_rgb"}:
                self._image_shapes[key] = tuple(int(v) for v in attr["shape"])
            elif obs_type == "low_dim":
                self._lowdim_keys.append(key)
            else:
                raise ValueError(f"Unsupported VISTA observation type {obs_type!r} for key {key!r}")
        self._torch = torch
        self._model = model

    def _inputs_to_vendor_obs(self, inputs: dict[str, np.ndarray]) -> dict[str, Any]:
        torch = self._torch
        assert torch is not None
        rgb = np.asarray(inputs["rgb_wrist"], dtype=np.float32)
        gelsight = np.asarray(inputs["gelsight"], dtype=np.float32)
        qpos = np.asarray(inputs["qpos"], dtype=np.float32)
        if rgb.ndim != 4 or rgb.shape[-1] != 3:
            raise ValueError(f"rgb_wrist must be [T,H,W,3], got {rgb.shape}")
        if gelsight.ndim != 4 or gelsight.shape[-1] != 3:
            raise ValueError(f"gelsight must be [T,H,W,3], got {gelsight.shape}")
        if qpos.ndim != 2 or qpos.shape[1] != 10:
            raise ValueError(f"qpos must be [T,10] for VISTA checkpoints, got {qpos.shape}")

        pose7 = []
        gripper = []
        for row in qpos:
            this_pose7, closedness = tcp_state_to_pose7d_and_gripper(row)
            pose7.append(this_pose7)
            gripper.append(closedness)
        pose7_array = np.stack(pose7, axis=0).astype(np.float32)
        gripper_array = np.repeat(np.asarray(gripper, dtype=np.float32)[:, None], 2, axis=1)

        obs: dict[str, Any] = {}
        for key, shape in self._image_shapes.items():
            source = rgb if key == "robot0_eye_in_hand_image" else gelsight
            obs[key] = self._image_tensor(source, expected_shape=shape)
        for key in self._lowdim_keys:
            if key == "robot0_eef_pos":
                value = pose7_array[:, :3]
            elif key == "robot0_eef_quat":
                value = pose7_array[:, 3:7]
            elif key == "robot0_gripper_qpos":
                value = gripper_array
            else:
                raise KeyError(f"Unsupported VISTA lowdim observation key: {key}")
            obs[key] = torch.from_numpy(value).unsqueeze(0).to(self._device)
        return obs

    def _image_tensor(self, images: np.ndarray, *, expected_shape: tuple[int, int, int]) -> Any:
        torch = self._torch
        assert torch is not None
        c, h, w = expected_shape
        if c != 3:
            raise ValueError(f"Only 3-channel images are supported, got shape {expected_shape}")
        tensor = torch.from_numpy(np.transpose(images, (0, 3, 1, 2))).unsqueeze(0).to(self._device)
        tensor = tensor.to(dtype=torch.float32)
        if tuple(tensor.shape[-2:]) != (h, w):
            import torch.nn.functional as F

            flat = tensor.reshape(-1, c, *tensor.shape[-2:])
            flat = F.interpolate(flat, size=(h, w), mode="bilinear", align_corners=False)
            tensor = flat.reshape(1, images.shape[0], c, h, w)
        return tensor


def can_load_vendor_vista_checkpoint(
    checkpoint_dir: Path,
    manifests: RuntimeManifests | None = None,
    checkpoint_file: str | Path | None = None,
) -> bool:
    checkpoint_dir = Path(checkpoint_dir)
    if not _resolve_vista_checkpoint_path(checkpoint_dir, checkpoint_file=checkpoint_file, missing_ok=True):
        return False
    if manifests is None:
        try:
            manifests = load_runtime_manifests(checkpoint_dir)
        except FileNotFoundError:
            return False
    try:
        spec = get_model_spec(manifests.policy["model"])
    except Exception:
        return False
    return spec.name in {"vista_so2", "vista_so3"}


def _resolve_vista_checkpoint_path(
    checkpoint_dir: Path,
    *,
    checkpoint_file: str | Path | None = None,
    missing_ok: bool = False,
) -> Path | None:
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_file is not None:
        path = Path(checkpoint_file).expanduser()
        candidates = [path if path.is_absolute() else checkpoint_dir / path]
        if not path.is_absolute() and path.parent == Path("."):
            candidates.append(checkpoint_dir / "checkpoints" / path)
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        if missing_ok:
            return None
        raise FileNotFoundError(
            f"Missing VISTA checkpoint_file={checkpoint_file!s}. Expected one of: "
            f"{', '.join(str(candidate) for candidate in candidates)}"
        )
    candidates = [
        checkpoint_dir / "checkpoints" / "best.ckpt",
        checkpoint_dir / "best.ckpt",
        checkpoint_dir / "checkpoints" / "latest.ckpt",
    ]
    candidates.extend(_sorted_epoch_checkpoints(checkpoint_dir / "checkpoints"))
    for path in candidates:
        if path.is_file():
            return path
    if missing_ok:
        return None
    raise FileNotFoundError(
        f"Missing VISTA checkpoint. Expected one of: {', '.join(str(path) for path in candidates)}"
    )


def _sorted_epoch_checkpoints(checkpoints_dir: Path) -> list[Path]:
    if not checkpoints_dir.is_dir():
        return []

    def epoch_key(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem.split("=", 1)[1]), path.name
        except (IndexError, ValueError):
            return -1, path.name

    return sorted(checkpoints_dir.glob("epoch=*.ckpt"), key=epoch_key, reverse=True)


def _prepare_vendor_imports() -> None:
    for path in (VISTA_ROOT, POLICIES_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    loaded_vista = sys.modules.get("vista")
    if loaded_vista is not None:
        loaded_vista_file = Path(getattr(loaded_vista, "__file__", "") or "")
        if loaded_vista_file and VISTA_ROOT not in loaded_vista_file.resolve().parents:
            raise ImportError(
                f"Visuotactile VISTA checkpoint requires the vendored vista package, "
                f"but vista is already loaded from {loaded_vista_file}."
            )


def _patch_scipy_todense() -> None:
    try:
        import scipy.sparse
    except ImportError:
        return
    if getattr(scipy.sparse.spmatrix.todense, "_vt_franka_patched", False):
        return
    original = scipy.sparse.spmatrix.todense

    def patched(self, order=None, out=None):
        return np.asarray(original(self, order=order, out=out))

    patched._vt_franka_patched = True  # type: ignore[attr-defined]
    scipy.sparse.spmatrix.todense = patched


def _configure_vista_sampling(
    model: Any,
    *,
    scheduler_name: str,
    num_inference_steps: int | None,
) -> None:
    scheduler_name = str(scheduler_name or "checkpoint").strip().lower()
    if scheduler_name not in {"checkpoint", "ddpm", "ddim"}:
        raise ValueError(f"Unsupported VISTA sampling_scheduler={scheduler_name!r}; expected checkpoint, ddpm, or ddim")
    if num_inference_steps is not None and num_inference_steps <= 0:
        raise ValueError("VISTA num_inference_steps must be positive")

    if scheduler_name == "ddpm":
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

        model.noise_scheduler = DDPMScheduler.from_config(model.noise_scheduler.config)
    elif scheduler_name == "ddim":
        from diffusers.schedulers.scheduling_ddim import DDIMScheduler

        model.noise_scheduler = DDIMScheduler.from_config(model.noise_scheduler.config)

    if num_inference_steps is not None:
        model.num_inference_steps = int(num_inference_steps)


def _resolve_device(configured: str, torch_module: Any) -> str:
    if configured == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("Visuotactile policy configured for CUDA, but torch.cuda.is_available() is false")
    return configured
