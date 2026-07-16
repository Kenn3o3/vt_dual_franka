from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from collections import deque

REPO_ROOT = Path(__file__).resolve().parents[2]
_DP_ROOT = str(Path(__file__).resolve().parent)
if _DP_ROOT not in sys.path:
    sys.path.append(_DP_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))
loaded_dp = sys.modules.get("dp")
if loaded_dp is not None:
    loaded_dp_file = Path(getattr(loaded_dp, "__file__", "") or "")
    if loaded_dp_file and Path(_DP_ROOT) not in loaded_dp_file.resolve().parents:
        raise ImportError(
            f"policy.DP requires its local dp package, but dp is already loaded from {loaded_dp_file}."
        )

from policy._base_policy import BasePolicy
from .debug_artifacts import RolloutDebugWriter

import numpy as np
import scipy.sparse
import cv2
import json

import torch
import dill
import hydra
from omegaconf import OmegaConf
from torchvision import transforms
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_quaternion
from dp.common.univtac_util import canonicalize_gripper_qpos

# Monkey-patch scipy.sparse.todense to return np.ndarray instead of np.matrix.
# Needed because escnn calls .todense() which returns np.matrix, but sklearn >=1.4
# rejects np.matrix in check_array(). This only affects the escnn basis computation
# during model construction.
_orig_todense = scipy.sparse.spmatrix.todense

def _patched_todense(self, order=None, out=None):
    return np.asarray(_orig_todense(self, order=order, out=out))

scipy.sparse.spmatrix.todense = _patched_todense

# Map from DP obs key → simulator observation path
# Each value is (top_level_key, sensor_name, data_key)
_OBS_SOURCE = {
    "robot0_eye_in_hand_image": [("observation", "wrist", "rgb")],
    "robot0_tactile_left_image": [
        ("tactile", "left_tactile", "rgb_marker"),
        ("tactile", "left_gsmini", "rgb_marker"),
    ],
    "robot0_tactile_right_image": [
        ("tactile", "right_tactile", "rgb_marker"),
        ("tactile", "right_gsmini", "rgb_marker"),
    ],
}


def _resolve_observation_source(observation, dp_key):
    candidates = _OBS_SOURCE.get(dp_key)
    if candidates is None:
        raise KeyError(f"No simulator source mapping for DP obs key '{dp_key}'")
    for top, sensor, data_key in candidates:
        sensor_obs = observation.get(top, {}).get(sensor)
        if sensor_obs is not None and data_key in sensor_obs:
            return top, sensor, data_key
    raise KeyError(
        f"Missing simulator source for DP obs key '{dp_key}'. Tried: {candidates}"
    )


def _quat_angle_deg_wxyz(q1, q2) -> float:
    q1 = torch.as_tensor(q1, dtype=torch.float32).reshape(-1)
    q2 = torch.as_tensor(q2, dtype=torch.float32).reshape(-1)
    q1 = q1 / torch.clamp(torch.linalg.norm(q1), min=1e-12)
    q2 = q2 / torch.clamp(torch.linalg.norm(q2), min=1e-12)
    dot = torch.clamp(torch.abs(torch.dot(q1, q2)), min=-1.0, max=1.0)
    angle_rad = 2.0 * torch.arccos(dot)
    return float(torch.rad2deg(angle_rad).cpu())


def _center_square_crop_image(image_rgb: np.ndarray) -> np.ndarray:
    """Center-crop an HWC RGB image to the largest square region."""
    height, width = image_rgb.shape[:2]
    size = min(height, width)
    x0 = (width - size) // 2
    y0 = (height - size) // 2
    return image_rgb[y0 : y0 + size, x0 : x0 + size]


def _local_tactile_encoder_ckpt() -> Path:
    override = os.environ.get("UNIVTAC_TACTILE_ENCODER_CKPT")
    if override:
        return Path(override).expanduser()
    return REPO_ROOT / "checkpoints" / "UniVTAC_encoder" / "best.pth"


def _repair_tactile_encoder_ckpt_path(cfg) -> None:
    """Patch downloaded remote checkpoint configs before Hydra instantiation."""
    local_ckpt = _local_tactile_encoder_ckpt()
    config_paths = (
        "policy.tactile_ckpt",
        "policy.obs_encoder.tactile_encoder.ckpt_path",
    )

    for config_path in config_paths:
        ckpt_value = OmegaConf.select(cfg, config_path)
        if ckpt_value in (None, "none", "None", "null", "Null"):
            continue

        ckpt_path = Path(str(ckpt_value)).expanduser()
        if ckpt_path.is_file():
            continue

        if not local_ckpt.is_file():
            raise FileNotFoundError(
                "Downloaded DP checkpoint references a missing UniVTAC tactile "
                f"encoder checkpoint at {ckpt_path}, and the local fallback was "
                f"not found at {local_ckpt}. Copy the pretrained encoder to the "
                "fallback path or set UNIVTAC_TACTILE_ENCODER_CKPT."
            )

        OmegaConf.update(cfg, config_path, str(local_ckpt), merge=False)
        print(
            "Repaired UniVTAC tactile encoder checkpoint path: "
            f"{config_path} {ckpt_path} -> {local_ckpt}"
        )


class Policy(BasePolicy):
    def __init__(self, args):
        """Initialize DP policy for deployment (SO3 / SO2 / SO2-tact)."""
        task_name = args["task_name"]
        ep_num = os.environ.get("EP_NUM", "100")
        ckpt_path = os.environ.get("DP_CKPT_PATH")
        if not ckpt_path:
            raise ValueError(
                "DP deployment requires DP_CKPT_PATH from the clean orchestration layer."
            )

        # Load checkpoint
        payload = torch.load(
            open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu"
        )
        cfg = payload["cfg"]
        _repair_tactile_encoder_ckpt_path(cfg)
        # Instantiate model and load weights (includes normalizer + ws_center)
        # Prefer EMA model if available (used during training eval)
        self.model = hydra.utils.instantiate(cfg.policy)
        if "ema_model" in payload["state_dicts"]:
            self.model.load_state_dict(payload["state_dicts"]["ema_model"])
            print("Loaded EMA model weights")
        else:
            self.model.load_state_dict(payload["state_dicts"]["model"])
            print("Loaded model weights (no EMA found)")

        self.device = torch.device(
            os.environ.get(
                "DP_DEVICE",
                "cuda" if torch.cuda.is_available() else "cpu",
            )
        )
        self.model.eval()
        self.model.to(self.device)

        # Detect required observation keys from checkpoint config
        shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
        self.obs_rgb_keys = []
        self.obs_tactile_rgb_keys = []
        self.obs_image_keys = []
        self.obs_rgb_shapes = {}
        self.obs_lowdim_keys = []
        for key, attr in shape_meta["obs"].items():
            obs_type = attr.get("type", "low_dim")
            if obs_type in {"rgb", "tactile_rgb"}:
                self.obs_image_keys.append(key)
                self.obs_rgb_keys.append(key)
                shape = tuple(attr["shape"])
                self.obs_rgb_shapes[key] = (shape[1], shape[2])
                if obs_type == "tactile_rgb":
                    self.obs_tactile_rgb_keys.append(key)
            else:
                self.obs_lowdim_keys.append(key)

        # Config
        self.n_obs_steps = self.model.n_obs_steps
        self.n_action_steps = self.model.n_action_steps
        self.execution_horizon = int(
            os.environ.get("DP_EXEC_HORIZON", "1")
        )
        self.execution_horizon = max(1, min(self.execution_horizon, self.n_action_steps))
        self.postprocessing = os.environ.get(
            "DP_POSTPROCESSING",
            "normal",
        ).lower()
        if self.postprocessing not in {"normal", "temporal_agg"}:
            raise ValueError(
                f"Unsupported DP_POSTPROCESSING={self.postprocessing}. "
                "Expected one of: normal, temporal_agg."
            )
        self.temporal_agg_k = float(
            os.environ.get("DP_TEMPORAL_AGG_K", "0.01")
        )
        if self.postprocessing == "temporal_agg":
            self.execution_horizon = 1
        self.obs_clamp_mode = os.environ.get("DP_OBS_CLAMP_MODE", "off").lower()
        self.max_inferences = int(
            os.environ.get("DP_MAX_INFERENCES", "20")
        )
        self.action_type = os.environ.get(
            "DP_ACTION_TYPE",
            "ee_servo",
        ).lower()
        if self.action_type not in {"ee", "ee_servo"}:
            raise ValueError(f"Unsupported DP_ACTION_TYPE={self.action_type}. Expected one of: ee, ee_servo")

        # State
        self.obs_history = deque(maxlen=self.n_obs_steps)
        self.action_buffer = None
        self.action_idx = 0
        self.temporal_action_history = {}
        self.policy_step = 0

        # NOTE: historical DP deploy resized every RGB observation to 84x84 here.
        # Image transform: resize to 84x84, CHW float [0,1]
        # self.img_transform = transforms.Resize((84, 84))

        self._obs_clamp = {}
        if self.obs_clamp_mode == "hard":
            for key in self.obs_lowdim_keys:
                if "quat" in key:
                    continue
                try:
                    stats = self.model.normalizer.params_dict[key]["input_stats"]
                    self._obs_clamp[key] = (stats["min"], stats["max"])
                except (KeyError, AttributeError):
                    pass

        variant = cfg.get("name", "unknown")
        self._inference_count = 0
        self._debug_enabled = os.environ.get("UNIVTAC_DEBUG_DUMP", "0") == "1"
        self.policy_family = os.environ.get("UNIVTAC_DEPLOY_POLICY_FAMILY", "DP")
        self._debug_writer = RolloutDebugWriter(
            enabled=self._debug_enabled,
            root_dir=os.environ.get("UNIVTAC_DEBUG_DIR"),
            policy_name=self.policy_family,
            variant=variant,
        )
        self._debug_seed = None
        self._task_name = task_name
        self._ep_num = ep_num
        self._variant = variant
        self.ws_center = getattr(self.model, "ws_center", None)

        print(
            f"{self.policy_family} policy ({variant}) loaded from {ckpt_path}\n"
            f"  n_obs_steps={self.n_obs_steps}, n_action_steps={self.n_action_steps}, "
            f"execution_horizon={self.execution_horizon}, postprocessing={self.postprocessing}, "
            f"obs_clamp_mode={self.obs_clamp_mode}, "
            f"max_inferences={self.max_inferences}, action_type={self.action_type}\n"
            f"  rgb_keys={self.obs_rgb_keys}, tactile_rgb_keys={self.obs_tactile_rgb_keys}, "
            f"lowdim_keys={self.obs_lowdim_keys}"
        )
        if self.ws_center is not None:
            print(f"  ws_center={self.ws_center.detach().cpu().tolist()}")
        if self.postprocessing == "temporal_agg":
            print(f"  temporal_agg_k={self.temporal_agg_k}")
        if self.execution_horizon > 1:
            print("  [WARN] DP is running open-loop chunk execution; horizon=1 is recommended.")
        for key, (lo, hi) in self._obs_clamp.items():
            print(f"  normalizer range [{key}]: min={lo.tolist()}, max={hi.tolist()}")

    def _ensure_debug_rollout(self, task) -> None:
        if not self._debug_enabled:
            return
        seed = int(getattr(task.cfg, "seed", -1))
        if self._debug_seed == seed:
            return
        self._debug_writer.start_rollout(
            seed=seed,
            metadata={
                "task_name": self._task_name,
                "n_demo": self._ep_num,
                "save_root": str(getattr(task, "save_root", "")),
            },
        )
        self._debug_seed = seed

    @staticmethod
    def _jsonable(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu()
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(k): Policy._jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [Policy._jsonable(v) for v in value]
        return value

    def _write_json(self, path: Path, payload: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._jsonable(payload), f, indent=2)

    def _get_image(self, observation, dp_key):
        """Fetch an RGB image and apply the same wrist preprocessing used in training.

        Replicates the training data pipeline: camera RGB → cv2.imencode (treats as BGR)
        → cv2.imdecode (BGR) → cvtColor(BGR2RGB). This JPEG roundtrip with the BGR/RGB
        mismatch creates a specific color transform that the model was trained on.
        """
        top, sensor, data_key = _resolve_observation_source(observation, dp_key)
        img = observation[top][sensor][data_key]  # (H, W, 3) uint8 RGB tensor
        # Replicate training JPEG roundtrip: RGB tensor → numpy → imencode → imdecode → BGR2RGB
        img_np = img.cpu().numpy().astype(np.uint8)
        _, jpeg_buf = cv2.imencode(".jpg", img_np)
        img_bgr = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        if dp_key == "robot0_eye_in_hand_image":
            # NOTE: the BIGFOV wrist camera is 480x270; we remove the two side bands
            # before resizing so train/eval share the same square crop.
            img_rgb = _center_square_crop_image(img_rgb)
        img_t = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0  # (3, H, W)
        target_hw = self.obs_rgb_shapes[dp_key]
        # NOTE: historical DP deploy used the single fixed transform below.
        # return self.img_transform(img_t)  # (3, 84, 84)
        if tuple(img_t.shape[-2:]) != target_hw:
            img_t = transforms.Resize(target_hw)(img_t)
        return img_t

    def _get_raw_image(self, observation, dp_key):
        top, sensor, data_key = _resolve_observation_source(observation, dp_key)
        return observation[top][sensor][data_key].cpu().numpy().astype(np.uint8)

    def _dump_tactile_debug(self, task, observation, inf_dir: Path):
        tactile = observation.get("tactile", {})
        sensor_map = {
            "robot0_tactile_left_image": "left_tactile",
            "robot0_tactile_right_image": "right_tactile",
        }
        for prefix, sensor_name in sensor_map.items():
            sensor_obs = tactile.get(sensor_name)
            if sensor_obs is None:
                continue

            marker_rgb = sensor_obs.get("rgb_marker")
            if marker_rgb is not None:
                marker_rgb_np = marker_rgb.cpu().numpy().astype(np.uint8)
                cv2.imwrite(
                    str(inf_dir / f"{prefix}_marker.png"),
                    cv2.cvtColor(marker_rgb_np, cv2.COLOR_RGB2BGR),
                )

            marker = sensor_obs.get("marker")
            if marker is None:
                continue

            marker_cpu = marker.detach().cpu()
            payload = {
                "shape": list(marker_cpu.shape),
                "marker": marker_cpu.tolist(),
            }
            if marker_cpu.ndim == 3 and marker_cpu.shape[0] >= 2 and marker_cpu.shape[-1] == 2:
                curr = marker_cpu[1]
                valid = (
                    (curr[:, 0] > 0)
                    & (curr[:, 0] < 320)
                    & (curr[:, 1] > 0)
                    & (curr[:, 1] < 240)
                )
                payload["stats"] = {
                    "num_markers": int(curr.shape[0]),
                    "num_nonzero_curr": int((curr.abs().sum(dim=-1) > 0).sum().item()),
                    "num_valid_curr": int(valid.sum().item()),
                    "curr_uv_min": curr.min(dim=0).values.tolist(),
                    "curr_uv_max": curr.max(dim=0).values.tolist(),
                }
            self._write_json(inf_dir / f"{prefix}_marker.json", payload)

            tactile_manager = getattr(task, "_tactile_manager", None)
            tactile_sensor = getattr(tactile_manager, "tactiles", {}).get(sensor_name) if tactile_manager is not None else None
            if tactile_sensor is not None and hasattr(tactile_sensor, "get_marker_projection_debug"):
                try:
                    projection_debug = tactile_sensor.get_marker_projection_debug()
                except Exception as exc:
                    projection_debug = {"error": str(exc)}
                if projection_debug is not None:
                    self._write_json(inf_dir / f"{prefix}_projection_debug.json", projection_debug)

    def _eef_pose_from_observation(self, observation):
        ee = observation["embodiment"]["ee"]
        grip = canonicalize_gripper_qpos(observation["embodiment"]["joint"][-2:])[0:1]
        return torch.cat([ee[:7].float(), grip.float()], dim=0)

    def _action10d_to_ee_action(self, action_10d: torch.Tensor) -> torch.Tensor:
        pos = action_10d[:3]
        rot6d = action_10d[3:9]
        rot_mat = rotation_6d_to_matrix(rot6d.unsqueeze(0))
        quat_wxyz = matrix_to_quaternion(rot_mat).squeeze(0)
        gripper = action_10d[9:10]
        return torch.cat([pos, quat_wxyz, gripper]).cpu()

    def _update_task_overlay(self, task, current_pose=None, target_pose=None, inferencing=None, executing=None):
        if not hasattr(task, "set_video_overlay_state"):
            return
        payload = {
            "inference_count": self._inference_count,
            "max_inferences": None if self.max_inferences <= 0 else self.max_inferences,
        }
        if current_pose is not None:
            payload["current_eef_pose"] = current_pose.detach().cpu().tolist()
        if target_pose is not None:
            payload["target_eef_pose"] = target_pose.detach().cpu().tolist()
        if inferencing is not None:
            payload["inferencing"] = bool(inferencing)
        if executing is not None:
            payload["executing"] = bool(executing)
        task.set_video_overlay_state(**payload)

    def encode_obs(self, observation):
        """Convert simulator observation to a single-frame DP obs dict."""
        obs = {}

        # RGB-like images (wrist camera and marker-RGB tactile)
        for key in self.obs_rgb_keys:
            obs[key] = self._get_image(observation, key)

        # Low-dim observations
        for key in self.obs_lowdim_keys:
            if key == "robot0_eef_pos":
                obs[key] = observation["embodiment"]["ee"][:3]
            elif key == "robot0_eef_quat":
                # Pose.totensor() is wxyz; the DP normalizer is trained on wxyz as-is.
                obs[key] = observation["embodiment"]["ee"][3:7]
            elif key == "robot0_gripper_qpos":
                obs[key] = canonicalize_gripper_qpos(
                    observation["embodiment"]["joint"][-2:]
                ).float()
            else:
                raise KeyError(f"Unknown DP lowdim obs key: '{key}'")

        for key, (lo, hi) in self._obs_clamp.items():
            if key in obs:
                obs[key] = torch.clamp(obs[key].float(), min=lo.to(obs[key].device), max=hi.to(obs[key].device))

        return obs

    def _save_inference_inputs(self, *, inference_index: int, obs_dict: dict[str, torch.Tensor]) -> None:
        if not self._debug_writer.should_save_inference_montage(inference_index=inference_index):
            return
        tiles = {}
        for key in self.obs_rgb_keys:
            horizon = int(obs_dict[key].shape[1])
            for t in range(horizon):
                tiles[f"{key} t{t}"] = obs_dict[key][0, t].detach().cpu()
        self._debug_writer.save_inference_montage(inference_index=inference_index, tiles=tiles)

    def _horizon_action_payload(self, action_buffer: torch.Tensor, current_pose: torch.Tensor) -> list[dict]:
        payload = []
        obs_pose_cpu = current_pose.detach().cpu()
        obs_pos = obs_pose_cpu[:3]
        obs_quat = obs_pose_cpu[3:7]
        obs_grip = obs_pose_cpu[7]
        for idx in range(action_buffer.shape[0]):
            ee_action = self._action10d_to_ee_action(action_buffer[idx].detach().cpu())
            payload.append(
                {
                    "index": idx,
                    "pos_xyz": ee_action[:3].tolist(),
                    "quat_wxyz": ee_action[3:7].tolist(),
                    "gripper": float(ee_action[7].item()),
                    "delta_pos_mm_from_obs": float(torch.linalg.norm(ee_action[:3] - obs_pos).cpu() * 1000.0),
                    "delta_rot_deg_from_obs": _quat_angle_deg_wxyz(obs_quat, ee_action[3:7]),
                    "delta_grip_from_obs": float(torch.abs(ee_action[7] - obs_grip).cpu()),
                }
            )
        return payload

    def _stack_obs_history(self) -> dict[str, torch.Tensor]:
        obs_dict = {}
        for key in self.obs_history[0]:
            stacked = torch.stack([frame[key] for frame in self.obs_history], dim=0)
            obs_dict[key] = stacked.unsqueeze(0).to(self.device)
        return obs_dict

    def _aggregate_temporal_action(
        self,
        *,
        step_index: int,
        predicted_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, int]:
        predicted_actions_cpu = predicted_actions.detach().cpu()
        self.temporal_action_history[step_index] = predicted_actions_cpu

        candidates = []
        for source_step in sorted(self.temporal_action_history):
            chunk = self.temporal_action_history[source_step]
            offset = step_index - source_step
            if 0 <= offset < int(chunk.shape[0]):
                candidates.append(chunk[offset])
        if not candidates:
            aggregated = predicted_actions_cpu[0]
            num_candidates = 1
        else:
            stacked = torch.stack(candidates, dim=0)
            weights = np.exp(-self.temporal_agg_k * np.arange(len(candidates), dtype=np.float32))
            weights = weights / np.sum(weights)
            weight_tensor = torch.from_numpy(weights).to(dtype=stacked.dtype).unsqueeze(1)
            aggregated = (stacked * weight_tensor).sum(dim=0)
            num_candidates = len(candidates)

        expired_steps = [
            source_step
            for source_step, chunk in self.temporal_action_history.items()
            if source_step + int(chunk.shape[0]) <= step_index + 1
        ]
        for source_step in expired_steps:
            self.temporal_action_history.pop(source_step, None)

        return aggregated, num_candidates

    @torch.no_grad()
    def eval(self, task, observation):
        """Run one step of DP policy evaluation."""
        self._ensure_debug_rollout(task)
        # DEBUG: Send a fixed ee action to test if the robot follows commands
        _TEST_FIXED_ACTION = os.environ.get("DP_TEST_FIXED_ACTION")
        if _TEST_FIXED_ACTION:
            ee_action = torch.tensor(
                [float(x) for x in _TEST_FIXED_ACTION.split(",")],
                dtype=torch.float32,
            )
            current_eef_pose = self._eef_pose_from_observation(observation).detach().cpu()
            self._debug_writer.record_step(
                {
                    "sim_step": int(getattr(task, "step_count", 0)) + 1,
                    "did_inference": False,
                    "inference_index": None,
                    "action_buffer_index": 0,
                    "execution_horizon": 1,
                    "policy_latency_ms": None,
                    "proprio": {
                        "ee_pos_xyz": current_eef_pose[:3].tolist(),
                        "ee_quat_wxyz": current_eef_pose[3:7].tolist(),
                        "gripper": float(current_eef_pose[7].item()),
                    },
                    "final_action": {
                        "pos_xyz": ee_action[:3].tolist(),
                        "quat_wxyz": ee_action[3:7].tolist(),
                        "gripper": float(ee_action[7].item()),
                    },
                    "inference_actions": None,
                }
            )
            if hasattr(task, 'set_rollout_trace_context'):
                task.set_rollout_trace_context(
                    policy_name=self.policy_family,
                    mode='fixed_action',
                    inference_count=int(self._inference_count),
                    action_buffer_index=0,
                    execution_horizon=1,
                    did_inference=False,
                )
            task.take_action(ee_action, action_type=self.action_type)
            return

        current_eef_pose = self._eef_pose_from_observation(observation)
        self._update_task_overlay(task, current_pose=current_eef_pose, inferencing=False, executing=False)

        obs_frame = self.encode_obs(observation)
        self.obs_history.append(obs_frame)

        # Need enough observation frames
        if len(self.obs_history) < self.n_obs_steps:
            return

        did_inference = False
        inference_index = self._inference_count - 1 if self._inference_count > 0 else None
        horizon_actions = None
        policy_latency_ms = None
        aggregation_source_count = None

        if self.postprocessing == "temporal_agg":
            if self.max_inferences > 0 and self._inference_count >= self.max_inferences:
                self._update_task_overlay(task, current_pose=current_eef_pose, inferencing=False, executing=False)
                if hasattr(task, 'request_stop'):
                    task.request_stop(f'DP reached max inference count: {self.max_inferences}')
                return

            obs_dict = self._stack_obs_history()
            inference_index = self._inference_count
            self._save_inference_inputs(inference_index=inference_index, obs_dict=obs_dict)
            latency_start = time.perf_counter()
            result = self.model.predict_action(obs_dict)
            policy_latency_ms = (time.perf_counter() - latency_start) * 1000.0
            self.action_buffer = result["action"][0]
            self.action_idx = 0
            horizon_actions = self._horizon_action_payload(self.action_buffer.detach().cpu(), current_eef_pose)
            action_10d, aggregation_source_count = self._aggregate_temporal_action(
                step_index=self.policy_step,
                predicted_actions=self.action_buffer,
            )
            self._inference_count += 1
            did_inference = True
            action_buffer_index = 0
        else:
            # Re-inference when buffer depleted
            if self.action_buffer is None or self.action_idx >= self.execution_horizon:
                if self.max_inferences > 0 and self._inference_count >= self.max_inferences:
                    self._update_task_overlay(task, current_pose=current_eef_pose, inferencing=False, executing=False)
                    if hasattr(task, 'request_stop'):
                        task.request_stop(f'DP reached max inference count: {self.max_inferences}')
                    return

                obs_dict = self._stack_obs_history()
                inference_index = self._inference_count
                self._save_inference_inputs(inference_index=inference_index, obs_dict=obs_dict)
                latency_start = time.perf_counter()
                result = self.model.predict_action(obs_dict)
                policy_latency_ms = (time.perf_counter() - latency_start) * 1000.0
                self.action_buffer = result["action"][0]
                self.action_idx = 0
                horizon_actions = self._horizon_action_payload(self.action_buffer.detach().cpu(), current_eef_pose)
                self._inference_count += 1
                did_inference = True

            action_10d = self.action_buffer[self.action_idx]
            action_buffer_index = int(self.action_idx)
            self.action_idx += 1
        ee_action = self._action10d_to_ee_action(action_10d)

        self._update_task_overlay(
            task,
            current_pose=current_eef_pose,
            target_pose=ee_action,
            inferencing=did_inference,
            executing=False,
        )
        self._update_task_overlay(
            task,
            current_pose=current_eef_pose,
            target_pose=ee_action,
            inferencing=False,
            executing=True,
        )
        if hasattr(task, 'set_rollout_trace_context'):
            task.set_rollout_trace_context(
                policy_name=self.policy_family,
                mode='policy',
                inference_count=int(self._inference_count if did_inference else max(self._inference_count - 1, 0)),
                action_buffer_index=action_buffer_index,
                execution_horizon=int(self.execution_horizon),
                did_inference=bool(did_inference),
                n_obs_steps=int(self.n_obs_steps),
                n_action_steps=int(self.n_action_steps),
            )
        self._debug_writer.record_step(
            {
                "sim_step": int(getattr(task, "step_count", 0)) + 1,
                "did_inference": bool(did_inference),
                "inference_index": inference_index,
                "action_buffer_index": action_buffer_index,
                "execution_horizon": int(self.execution_horizon),
                "postprocessing": self.postprocessing,
                "aggregation_source_count": aggregation_source_count,
                "policy_latency_ms": policy_latency_ms,
                "proprio": {
                    "ee_pos_xyz": current_eef_pose[:3].tolist(),
                    "ee_quat_wxyz": current_eef_pose[3:7].tolist(),
                    "gripper": float(current_eef_pose[7].item()),
                },
                "final_action": {
                    "pos_xyz": ee_action[:3].tolist(),
                    "quat_wxyz": ee_action[3:7].tolist(),
                    "gripper": float(ee_action[7].item()),
                },
                "inference_actions": horizon_actions,
            }
        )
        task.take_action(ee_action, action_type=self.action_type)
        self._update_task_overlay(
            task,
            current_pose=ee_action,
            target_pose=ee_action,
            inferencing=False,
            executing=False,
        )
        self.policy_step += 1
        
    def reset(self):
        """Reset observation history and action buffer."""
        self.obs_history.clear()
        self.action_buffer = None
        self.action_idx = 0
        self.temporal_action_history.clear()
        self.policy_step = 0
        self._inference_count = 0
        self._debug_seed = None

    def on_rollout_end(self, task, result: str | None = None):
        self._debug_writer.finalize_rollout(result=result)
        self._debug_seed = None

    def close(self):
        self._debug_writer.close()
        super().close()
