from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from scipy.spatial.transform import Rotation

from ...mpd.math import (
    gripper_width_to_closedness,
    pose7d_and_gripper_to_tcp_state,
    tcp_state_to_pose7d_and_gripper,
)
from ....sensors.standardization import assert_standardized_rgb_frame
from vt_dual_franka_shared.pose_math import xyzw_to_wxyz
from .config import VisuotactileModelSpec, get_model_spec
from .image_preprocess import CropSpec, ImagePreprocessSpec, bgr_to_rgb, preprocess_image_rgb, rgb_to_bgr


class VisuotactileBackend(Protocol):
    model_spec: VisuotactileModelSpec
    obs_horizon: int
    action_horizon: int
    action_dim: int

    def predict_action_chunk(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        ...

    def ensure_loaded(self) -> None:
        ...

    def close(self) -> None:
        ...


@dataclass(frozen=True)
class RuntimeManifests:
    policy: dict[str, Any]
    preprocess1: dict[str, Any]
    preprocess2: dict[str, Any]
    normalizer_stats: dict[str, Any]


class TorchScriptVisuotactileBackend:
    def __init__(
        self,
        checkpoint_dir: Path,
        *,
        device: str = "auto",
        manifests: RuntimeManifests | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.manifests = manifests or load_runtime_manifests(self.checkpoint_dir)
        self.model_spec = get_model_spec(self.manifests.policy["model"])
        self.obs_horizon = int(self.manifests.policy.get("obs_horizon", self.model_spec.obs_horizon))
        self.action_horizon = int(self.manifests.policy.get("action_horizon", self.model_spec.action_horizon))
        self.action_dim = int(self.manifests.policy.get("action_dim", self.model_spec.action_dim))
        self._device_name = device
        self._model = None
        self._torch = None

    def predict_action_chunk(self, inputs: dict[str, np.ndarray]) -> np.ndarray:
        self._ensure_loaded()
        torch = self._torch
        assert torch is not None
        assert self._model is not None
        tensor_inputs = {
            key: torch.from_numpy(np.asarray(value, dtype=np.float32)).unsqueeze(0).to(self._device)
            for key, value in inputs.items()
        }
        with torch.no_grad():
            output = self._model(tensor_inputs)
        if isinstance(output, dict):
            output = output.get("action", output.get("actions"))
        if output is None:
            raise RuntimeError("TorchScript visuotactile model returned no action output")
        action = output.detach().cpu().numpy()
        if action.ndim == 3:
            action = action[0]
        if action.ndim != 2 or action.shape[1] != self.action_dim:
            raise ValueError(f"Expected action chunk [T,{self.action_dim}], got {action.shape}")
        return action.astype(np.float64)

    def close(self) -> None:
        self._model = None

    def ensure_loaded(self) -> None:
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Torch is required to run visuotactile policies. Install the training/runtime environment first."
            ) from exc
        artifact = self.checkpoint_dir / "model_torchscript.pt"
        if not artifact.exists():
            raise FileNotFoundError(
                f"Missing visuotactile runtime artifact: {artifact}. "
                "Train/export the selected model first. DP/VISTA checkpoints can be loaded directly from "
                "checkpoints/best.ckpt; TorchScript models should provide model_torchscript.pt."
            )
        self._torch = torch
        self._device = _resolve_device(self._device_name, torch)
        self._model = torch.jit.load(str(artifact), map_location=self._device)
        self._model.eval()


class RuntimePreprocessor:
    def __init__(
        self,
        manifests: RuntimeManifests,
        *,
        gripper_open_width_m: float,
        force_gripper_closedness: bool = False,
    ) -> None:
        self.manifests = manifests
        self._validate_manifest_alignment(manifests)
        self.gripper_open_width_m = float(gripper_open_width_m)
        self.force_gripper_closedness = bool(force_gripper_closedness)
        self.model_spec = get_model_spec(manifests.policy["model"])
        self._use_vendor_jpeg_compat = self.model_spec.name in {
            "dp_manifeel",
            "dp_equidiff_tact",
            "vista_so2",
            "vista_so3",
        }
        preprocess2 = manifests.preprocess2.get("preprocess2", manifests.preprocess2)
        self.preprocess2_specs = {
            key: _image_spec_from_manifest(value)
            for key, value in preprocess2.items()
            if key in {"rgb_wrist", "gelsight"} and isinstance(value, dict)
        }

    def observation_window_to_model_inputs(
        self,
        observation_window: list[dict[str, Any]],
        *,
        model_spec: VisuotactileModelSpec,
    ) -> dict[str, np.ndarray]:
        if not observation_window:
            raise ValueError("Visuotactile policy requires a non-empty observation window")
        states7: list[np.ndarray] = []
        states10: list[np.ndarray] = []
        rgb: list[np.ndarray] = []
        gelsight: list[np.ndarray] = []
        for observation in observation_window:
            rgb.append(self._preprocess_stream(observation, stream="rgb_wrist"))
            gelsight.append(self._preprocess_stream(observation, stream="gelsight"))
            pose7, closedness = self._state_from_observation(observation)
            if self.force_gripper_closedness:
                closedness = 1.0
            states7.append(np.concatenate([pose7, np.asarray([closedness], dtype=np.float64)]))
            states10.append(pose7d_and_gripper_to_tcp_state(pose7, closedness))
        qpos = np.stack(states10 if model_spec.action_representation == "pose10_rot6d_gripper" else states7, axis=0)
        return {
            "rgb_wrist": np.stack(rgb, axis=0).astype(np.float32) / 255.0,
            "gelsight": np.stack(gelsight, axis=0).astype(np.float32) / 255.0,
            "qpos": qpos.astype(np.float32),
        }

    def _preprocess_stream(self, observation: dict[str, Any], *, stream: str) -> np.ndarray:
        item = _image_item_from_observation(observation, stream=stream)
        image_rgb = assert_standardized_rgb_frame(np.asarray(item["image"]), item.get("metadata"))
        image_rgb = _standardized_dataset_jpeg_roundtrip_rgb(image_rgb)
        preprocessed = preprocess_image_rgb(image_rgb, self.preprocess2_specs[stream])
        if self._use_vendor_jpeg_compat:
            preprocessed = _vendor_backend_jpeg_roundtrip_rgb(preprocessed)
        return preprocessed

    def _state_from_observation(self, observation: dict[str, Any]) -> tuple[np.ndarray, float]:
        controller_state = observation.get("proprioception", {}).get("controller_state")
        if not isinstance(controller_state, dict):
            raise ValueError("Visuotactile policy requires observation['proprioception']['controller_state']")
        tcp_pose = controller_state.get("tcp_pose")
        if tcp_pose is None:
            raise ValueError("Visuotactile policy requires controller_state.tcp_pose")
        closedness = gripper_width_to_closedness(
            float(controller_state.get("gripper_width", self.gripper_open_width_m)),
            open_width_m=self.gripper_open_width_m,
        )
        return np.asarray(tcp_pose, dtype=np.float64), closedness

    @staticmethod
    def _validate_manifest_alignment(manifests: RuntimeManifests) -> None:
        preprocess2 = manifests.preprocess2.get("preprocess2", manifests.preprocess2)
        for stream in ("rgb_wrist", "gelsight"):
            if stream not in preprocess2:
                raise ValueError(f"Checkpoint preprocess2 manifest is missing stream {stream!r}")
        policy_model = manifests.policy.get("model")
        if not policy_model:
            raise ValueError("Checkpoint policy manifest is missing model")


def load_runtime_manifests(checkpoint_dir: Path) -> RuntimeManifests:
    checkpoint_dir = resolve_runtime_checkpoint_dir(Path(checkpoint_dir))
    policy_path = checkpoint_dir / "policy_manifest.json"
    if not policy_path.exists():
        raise FileNotFoundError(f"Missing visuotactile checkpoint manifest: {policy_path}")
    policy = _read_json(policy_path)
    return RuntimeManifests(
        policy=policy,
        preprocess1=_read_json(checkpoint_dir / "preprocess1_manifest.json"),
        preprocess2=_read_json(checkpoint_dir / "preprocess2_manifest.json"),
        normalizer_stats=_read_json(checkpoint_dir / "normalizer_stats.json"),
    )


def resolve_runtime_checkpoint_dir(checkpoint_dir: Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    if (checkpoint_dir / "policy_manifest.json").is_file():
        return checkpoint_dir
    candidates = sorted(path for path in checkpoint_dir.glob("*/policy_manifest.json") if path.is_file())
    if len(candidates) == 1:
        return candidates[0].parent
    return checkpoint_dir


def write_runtime_manifests(
    checkpoint_dir: str | Path,
    manifests: RuntimeManifests,
    *,
    overwrite: bool = True,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("policy_manifest.json", manifests.policy),
        ("preprocess1_manifest.json", manifests.preprocess1),
        ("preprocess2_manifest.json", manifests.preprocess2),
        ("normalizer_stats.json", manifests.normalizer_stats),
    ):
        path = checkpoint_dir / name
        if path.exists() and not overwrite:
            raise FileExistsError(path)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def action_row_to_vt_action(
    row: np.ndarray,
    *,
    model_spec: VisuotactileModelSpec,
    target_duration_sec: float,
    gripper_open_width_m: float,
    gripper_close_threshold: float,
) -> dict[str, Any]:
    values = np.asarray(row, dtype=np.float64)
    if model_spec.action_representation == "pose10_rot6d_gripper":
        pose7, gripper_scalar = _pytorch3d_row_rot6d_state_to_pose7d_and_gripper(values)
    else:
        if values.shape != (8,):
            raise ValueError(f"Expected 8D pose7+gripper action, got {values.shape}")
        pose7 = values[:7]
        gripper_scalar = float(np.clip(values[7], 0.0, 1.0))
    action: dict[str, Any] = {
        "target_tcp": pose7.astype(float).tolist(),
        "target_duration_sec": float(target_duration_sec),
        "metadata": {
            "visuotactile_model": model_spec.name,
            "visuotactile_action_representation": model_spec.action_representation,
            "visuotactile_action_row": values.astype(float).tolist(),
        },
    }
    if model_spec.action_representation == "pose10_rot6d_gripper":
        action["metadata"]["visuotactile_rot6d_convention"] = "pytorch3d_first_two_rows"
    if gripper_scalar >= gripper_close_threshold:
        action["gripper_closed"] = True
    else:
        action["gripper_width"] = float(gripper_open_width_m)
    return action


def _pytorch3d_row_rot6d_state_to_pose7d_and_gripper(state: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(state, dtype=np.float64)
    if values.shape != (10,):
        raise ValueError(f"Expected 10D pose10_rot6d_gripper action, got {values.shape}")
    matrix = _pytorch3d_row_rot6d_to_matrix(values[3:9])
    quat_xyzw = Rotation.from_matrix(matrix).as_quat()
    pose7 = np.concatenate([values[:3], xyzw_to_wxyz(quat_xyzw)]).astype(np.float64)
    gripper_scalar = float(np.clip(values[9], 0.0, 1.0))
    return pose7, gripper_scalar


def _pytorch3d_row_rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"Expected 6D rotation, got {values.shape}")
    first_row = _normalize_vector(values[:3])
    second_raw = values[3:]
    second_row = second_raw - float(np.dot(first_row, second_raw)) * first_row
    second_row = _normalize_vector(second_row)
    third_row = np.cross(first_row, second_row)
    return np.stack([first_row, second_row, third_row], axis=0)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero rotation vector")
    return vector / norm


def _image_item_from_observation(observation: dict[str, Any], *, stream: str) -> dict[str, Any]:
    if stream == "rgb_wrist":
        item = observation.get("images", {}).get("wrist")
    elif stream == "gelsight":
        tactile = observation.get("tactile", {})
        item = tactile.get("tactile_left", tactile.get("gelsight_frame"))
    else:
        raise ValueError(f"Unsupported stream: {stream}")
    if not isinstance(item, dict) or "image" not in item:
        raise ValueError(f"Observation is missing required stream {stream}")
    return item


def _image_spec_from_manifest(payload: dict[str, Any]) -> ImagePreprocessSpec:
    crop_payload = payload.get("crop", {})
    crop = CropSpec(
        mode=crop_payload.get("mode", "none"),
        box_xyxy=None if crop_payload.get("box_xyxy") is None else tuple(int(v) for v in crop_payload["box_xyxy"]),
        margin_fraction=float(crop_payload.get("margin_fraction", 0.0)),
    )
    output_size = payload.get("output_size", [224, 224])
    return ImagePreprocessSpec(
        output_size=(int(output_size[0]), int(output_size[1])),
        crop=crop,
        interpolation=payload.get("interpolation", "area"),
    )


def _jpeg_roundtrip_bgr(image_bgr: np.ndarray, *, quality: int | None = None) -> np.ndarray:
    cv2 = _require_cv2()
    image = np.asarray(image_bgr, dtype=np.uint8)
    params = [] if quality is None else [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)]
    ok, payload = cv2.imencode(".jpg", image, params)
    if not ok:
        raise RuntimeError("OpenCV failed to JPEG-encode runtime image")
    decoded = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("OpenCV failed to JPEG-decode runtime image")
    return np.ascontiguousarray(decoded)


def _vendor_backend_jpeg_roundtrip_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image_bgr = rgb_to_bgr(image_rgb)
    image_bgr = _jpeg_roundtrip_bgr(image_bgr, quality=95)
    return bgr_to_rgb(image_bgr)


def _standardized_dataset_jpeg_roundtrip_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image_bgr = rgb_to_bgr(image_rgb)
    image_bgr = _jpeg_roundtrip_bgr(image_bgr, quality=90)
    return bgr_to_rgb(image_bgr)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("OpenCV is required for visuotactile runtime preprocessing") from exc
    return cv2


def _prepare_healpy_imports() -> None:
    """Preload healpy before VISTA/EquiDiff model imports.

    Some conda builds segfault when healpy pulls in astropy.coordinates before
    its ERFA time formats are fully initialized. Importing astropy.coordinates
    (and healpy itself) up front avoids that import-order crash.
    """
    if sys.modules.get("healpy") is not None:
        return
    try:
        import astropy.coordinates  # noqa: F401
    except ImportError:
        pass
    try:
        import healpy  # noqa: F401
    except ImportError:
        pass


def _resolve_device(configured: str, torch_module: Any) -> str:
    if configured == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if configured == "cuda" and not torch_module.cuda.is_available():
        raise RuntimeError("Visuotactile policy configured for CUDA, but torch.cuda.is_available() is false")
    return configured
