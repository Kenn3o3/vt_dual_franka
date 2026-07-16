from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_VITAL_ROOT = Path(__file__).resolve().parent
_VENDOR_ROOT = _VITAL_ROOT.parent
if str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))
if str(_VITAL_ROOT) not in sys.path:
    sys.path.insert(0, str(_VITAL_ROOT))

import cv2
import numpy as np
import torch
from torchvision import transforms
from univtac.eval.debug_artifacts import RolloutDebugWriter

policy_root = _VITAL_ROOT.parent
if str(policy_root) not in sys.path:
    sys.path.append(str(policy_root))

from .._base_policy import BasePolicy
from .policy import ACT
from univtac.common.gripper import canonicalize_gripper_qpos


def _quat_angle_deg_wxyz(q1, q2) -> float:
    q1 = torch.as_tensor(q1, dtype=torch.float32).reshape(-1)
    q2 = torch.as_tensor(q2, dtype=torch.float32).reshape(-1)
    q1 = q1 / torch.clamp(torch.linalg.norm(q1), min=1e-12)
    q2 = q2 / torch.clamp(torch.linalg.norm(q2), min=1e-12)
    dot = torch.clamp(torch.abs(torch.dot(q1, q2)), min=-1.0, max=1.0)
    angle_rad = 2.0 * torch.arccos(dot)
    return float(torch.rad2deg(angle_rad).cpu())


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_rgb_tensor(image) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        tensor = image.detach().cpu()
    else:
        tensor = torch.as_tensor(image)
    if tensor.dtype != torch.float32:
        tensor = tensor.float()
    return tensor


def _training_rgb_roundtrip(image) -> np.ndarray:
    """Replicate the RGB/BGR JPEG roundtrip used when raw HDF5 was prepared."""
    img_np = _as_rgb_tensor(image).numpy().astype(np.uint8)
    ok, jpeg_buf = cv2.imencode(".jpg", img_np)
    if not ok:
        raise ValueError("Failed to encode ViTAL input image for training-compatible preprocessing.")
    img_bgr = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Failed to decode ViTAL input image for training-compatible preprocessing.")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


_CAMERA_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_CAMERA_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


class Policy(BasePolicy):
    def __init__(self, args):
        super().__init__(args)
        ckpt_path = os.environ.get("VITAL_CKPT_PATH")
        if not ckpt_path:
            raise ValueError("ViTAL clean deploy requires VITAL_CKPT_PATH.")

        self.ckpt_path = Path(ckpt_path).expanduser().resolve()
        self.ckpt_dir = self.ckpt_path.parent
        self.task_name = args["task_name"]
        self.device = torch.device(os.environ.get("VITAL_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
        self.execution_horizon = max(1, int(os.environ.get("VITAL_EXEC_HORIZON", "1")))
        self.max_inferences = int(os.environ.get("VITAL_MAX_INFERENCES", "20"))
        self.action_type = os.environ.get("VITAL_ACTION_TYPE", "ee_servo").lower()
        if self.action_type not in {"ee_servo"}:
            raise ValueError("ViTAL clean deploy currently supports ee_servo actions only.")

        args_path = self.ckpt_dir / "args.json"
        if not args_path.is_file():
            raise FileNotFoundError(f"ViTAL args.json not found next to checkpoint: {args_path}")
        model_args = _load_json(args_path)
        model_args["ckpt_dir"] = str(self.ckpt_dir)
        model_args["device"] = str(self.device)

        stats_path = self.ckpt_dir / "dataset_stats.pkl"
        norm_stats_path = self.ckpt_dir / "norm_stats.json"
        if not stats_path.is_file() and not norm_stats_path.is_file():
            raise FileNotFoundError(f"ViTAL stats not found in {self.ckpt_dir}")
        if "state_dim" not in model_args:
            stats_payload = _load_json(norm_stats_path) if norm_stats_path.is_file() else None
            if isinstance(stats_payload, dict) and "qpos_mean" in stats_payload:
                model_args["state_dim"] = len(stats_payload["qpos_mean"])
            elif isinstance(stats_payload, dict) and "action_mean" in stats_payload:
                model_args["state_dim"] = len(stats_payload["action_mean"])
            else:
                model_args["state_dim"] = 8

        if "camera_names" not in model_args:
            camera = model_args.get("camera", ["cam_high"])
            tactile = model_args.get("tactile", ["cam_left_tactile", "cam_right_tactile"])
            model_args["camera_names"] = list(camera) + list(tactile)
        model_args["cam_backbone_mapping"] = {name: 0 for name in model_args["camera_names"]}
        model_args["cam_backbone_mapping"]["cam_left_tactile"] = 1
        model_args["cam_backbone_mapping"]["cam_right_tactile"] = 1

        from .clip_pretraining import modified_resnet18

        backbones = [modified_resnet18(), modified_resnet18()]
        self.model = ACT(model_args, backbones)
        self.model.policy.load_state_dict(torch.load(self.ckpt_path, map_location=self.device))
        self.model.policy.eval()
        self.model.policy.to(self.device)
        if not getattr(self.model, "temporal_agg", False):
            self.model.query_frequency = max(
                1,
                min(int(self.execution_horizon), int(getattr(self.model, "num_queries", self.execution_horizon))),
            )
        self.camera_names = list(self.model.camera_names)
        self.inference_count = 0
        self.query_every_step = os.environ.get("VITAL_ACT_QUERY_EVERY_STEP", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.model.query_every_step = self.query_every_step
        self.camera_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._debug_enabled = os.environ.get("UNIVTAC_DEBUG_DUMP", "0") == "1"
        self._debug_writer = RolloutDebugWriter(
            enabled=self._debug_enabled,
            root_dir=os.environ.get("UNIVTAC_DEBUG_DIR"),
            policy_name="ViTAL",
            variant=self.ckpt_dir.name,
        )
        self._debug_seed = None
        print(
            f"ViTAL policy loaded from {self.ckpt_path}\n"
            f"  camera_names={self.camera_names}, execution_horizon={self.execution_horizon}, "
            f"max_inferences={self.max_inferences}, action_type={self.action_type}, "
            f"query_every_step={self.query_every_step}"
        )

    def _normalize_quat_in_action(self, action: torch.Tensor, fallback_quat: torch.Tensor) -> torch.Tensor:
        quat = action[3:7]
        if not torch.isfinite(quat).all():
            action[3:7] = fallback_quat
            return action
        quat_norm = torch.linalg.norm(quat)
        if quat_norm <= 1e-6:
            action[3:7] = fallback_quat
            return action
        action[3:7] = quat / quat_norm
        return action

    def _tactile_stats(self) -> tuple[np.ndarray, np.ndarray]:
        stats = self.model.stats
        if stats is None:
            return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32)
        if "gelsight_mean" in stats and "gelsight_std" in stats:
            mean = np.asarray(stats["gelsight_mean"], dtype=np.float32)
            std = np.asarray(stats["gelsight_std"], dtype=np.float32)
            return mean, np.clip(std, 1e-2, np.inf)
        left_mean = np.asarray(stats["left_tac_mean"], dtype=np.float32)
        right_mean = np.asarray(stats["right_tac_mean"], dtype=np.float32)
        left_std = np.asarray(stats["left_tac_std"], dtype=np.float32)
        right_std = np.asarray(stats["right_tac_std"], dtype=np.float32)
        return (left_mean + right_mean) / 2.0, np.clip((left_std + right_std) / 2.0, 1e-2, np.inf)

    def _camera_transform(self, image) -> torch.Tensor:
        img = cv2.resize(_training_rgb_roundtrip(image), (256, 256), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        return self.camera_normalize(img)

    def _tactile_transform(self, image) -> torch.Tensor:
        mean, std = self._tactile_stats()
        img = cv2.resize(_training_rgb_roundtrip(image), (256, 256), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        img = transforms.Normalize(mean=mean.tolist(), std=std.tolist())(img)
        return img

    def _input_for_debug(self, cam_name: str, image: torch.Tensor) -> torch.Tensor:
        image = image.detach().cpu()
        if cam_name.startswith("cam_") and not cam_name.endswith("tactile"):
            return torch.clamp(image * _CAMERA_STD + _CAMERA_MEAN, 0.0, 1.0)
        mean, std = self._tactile_stats()
        mean_t = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        return torch.clamp(image * std_t + mean_t, 0.0, 1.0)

    def _save_inference_inputs(self, *, inference_index: int, encoded_obs: dict) -> None:
        if not self._debug_writer.should_save_inference_montage(inference_index=inference_index):
            return
        tiles = {}
        for cam_name in self.camera_names:
            if cam_name in encoded_obs:
                tiles[f"input {cam_name}"] = self._input_for_debug(cam_name, encoded_obs[cam_name])
        self._debug_writer.save_inference_montage(inference_index=inference_index, tiles=tiles)

    def encode_obs(self, observation):
        encoded = {}
        for cam_name in self.camera_names:
            if cam_name == "cam_high":
                encoded[cam_name] = self._camera_transform(observation["observation"]["head"]["rgb"])
            elif cam_name == "cam_wrist":
                encoded[cam_name] = self._camera_transform(observation["observation"]["wrist"]["rgb"])
            elif cam_name == "cam_left_tactile":
                encoded[cam_name] = self._tactile_transform(observation["tactile"]["left_tactile"]["rgb_marker"])
            elif cam_name == "cam_right_tactile":
                encoded[cam_name] = self._tactile_transform(observation["tactile"]["right_tactile"]["rgb_marker"])
            else:
                raise KeyError(f"Unsupported ViTAL camera name: {cam_name}")
        ee = observation["embodiment"]["ee"][:7].detach().cpu().float()
        grip_pair = canonicalize_gripper_qpos(observation["embodiment"]["joint"][-2:]).detach().cpu().float()
        grip = grip_pair[:1]
        encoded["qpos"] = torch.cat([ee, grip], dim=0).numpy()
        return encoded

    def _eef_pose_from_observation(self, observation: dict) -> torch.Tensor:
        ee = observation["embodiment"]["ee"][:7].detach().cpu().float()
        grip = canonicalize_gripper_qpos(observation["embodiment"]["joint"][-2:]).detach().cpu().float()[:1]
        return torch.cat([ee, grip], dim=0)

    def _ensure_debug_rollout(self, task) -> None:
        if not self._debug_enabled:
            return
        seed = int(getattr(task.cfg, "seed", -1))
        if self._debug_seed == seed:
            return
        self._debug_writer.start_rollout(
            seed=seed,
            metadata={
                "task_name": self.task_name,
                "ckpt_path": str(self.ckpt_path),
                "camera_names": self.camera_names,
            },
        )
        self._debug_seed = seed

    def _query_action_payload(self, query_actions: np.ndarray, current_pose: torch.Tensor) -> list[dict]:
        payload = []
        current_pose = current_pose.detach().cpu()
        for idx, query_action in enumerate(query_actions):
            query_action_t = torch.from_numpy(query_action).float()
            payload.append(
                {
                    "index": idx,
                    "pos_xyz": query_action_t[:3].tolist(),
                    "quat_wxyz": query_action_t[3:7].tolist(),
                    "gripper": float(query_action_t[7].item()),
                    "delta_pos_mm_from_obs": float(
                        torch.linalg.norm(query_action_t[:3] - current_pose[:3]).item() * 1000.0
                    ),
                    "delta_rot_deg_from_obs": _quat_angle_deg_wxyz(current_pose[3:7], query_action_t[3:7]),
                    "delta_grip_from_obs": float(torch.abs(query_action_t[7] - current_pose[7]).item()),
                }
            )
        return payload

    def _record_debug_step(
        self,
        *,
        task,
        current_eef_pose: torch.Tensor,
        action: torch.Tensor,
        did_inference: bool,
        inference_index: int | None,
        action_buffer_index: int,
        query_actions: np.ndarray | None,
    ) -> None:
        if not self._debug_enabled:
            return
        action_cpu = action.detach().cpu()
        step_payload = {
            "sim_step": int(getattr(task, "step_count", 0)) + 1,
            "did_inference": bool(did_inference),
            "inference_index": inference_index,
            "action_buffer_index": int(action_buffer_index),
            "execution_horizon": int(self.execution_horizon if self.query_every_step else getattr(self.model, "query_frequency", self.execution_horizon)),
            "policy_latency_ms": None,
            "proprio": {
                "ee_pos_xyz": current_eef_pose[:3].tolist(),
                "ee_quat_wxyz": current_eef_pose[3:7].tolist(),
                "gripper": float(current_eef_pose[7].item()),
            },
            "final_action": {
                "pos_xyz": action_cpu[:3].tolist(),
                "quat_wxyz": action_cpu[3:7].tolist(),
                "gripper": float(action_cpu[7].item()),
            },
            "inference_actions": None
            if query_actions is None
            else self._query_action_payload(query_actions, current_eef_pose),
        }
        if hasattr(task, "consume_physx_joint_writes"):
            step_payload["physx_joint_writes"] = task.consume_physx_joint_writes()
        self._debug_writer.record_step(step_payload)

    @torch.no_grad()
    def eval(self, task, observation):
        self._ensure_debug_rollout(task)
        current_eef_pose = self._eef_pose_from_observation(observation)
        model_t_before = int(getattr(self.model, "t", 0))
        model_query_frequency = max(1, int(getattr(self.model, "query_frequency", 1)))
        query_frequency = 1 if self.query_every_step else model_query_frequency
        did_inference = self.query_every_step or (model_t_before % model_query_frequency) == 0
        inference_index = self.inference_count if did_inference else (
            self.inference_count - 1 if self.inference_count > 0 else None
        )

        if did_inference and self.max_inferences > 0 and self.inference_count >= self.max_inferences:
            if hasattr(task, "request_stop"):
                task.request_stop(f"ViTAL reached max inference count: {self.max_inferences}")
            return

        obs = self.encode_obs(observation)
        if did_inference:
            self._save_inference_inputs(inference_index=self.inference_count, encoded_obs=obs)
        model_action = self.model.get_action(obs)

        action_buffer_index = 0 if self.query_every_step else int(model_t_before % model_query_frequency)
        query_actions = None
        if did_inference and hasattr(self.model, "all_actions"):
            query_actions = self.model.post_process(self.model.all_actions.detach().cpu().numpy())[0]
            self.inference_count += 1
        action = torch.from_numpy(model_action.reshape(-1)).float().to(task.device)
        action = self._normalize_quat_in_action(
            action,
            observation["embodiment"]["ee"][3:7].float().to(task.device),
        )
        if hasattr(task, "set_rollout_trace_context"):
            task.set_rollout_trace_context(
                policy_name="ViTAL",
                mode="policy",
                inference_count=int(self.inference_count),
                action_buffer_index=action_buffer_index,
                execution_horizon=int(query_frequency),
                did_inference=bool(did_inference),
                n_action_steps=int(getattr(self.model, "num_queries", 1)),
            )
        task.take_action(action, action_type=self.action_type)
        self._record_debug_step(
            task=task,
            current_eef_pose=current_eef_pose,
            action=action,
            did_inference=did_inference,
            inference_index=inference_index,
            action_buffer_index=action_buffer_index,
            query_actions=query_actions,
        )

    def reset(self):
        if hasattr(self.model, "reset"):
            self.model.reset()
        self.inference_count = 0
        self._debug_seed = None

    def on_rollout_end(self, task, result: str | None = None):
        self._debug_writer.finalize_rollout(result=result)
        self._debug_seed = None

    def close(self):
        self._debug_writer.close()
        super().close()
