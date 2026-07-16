from __future__ import annotations

import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ACT_ROOT = Path(__file__).resolve().parent
if str(ACT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACT_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policy._base_policy import BasePolicy

import os
import cv2
import yaml
import numpy as np
import torch
from .act_policy import ACT
from torchvision import transforms
from .debug_artifacts import RolloutDebugWriter
from .univtac_util import canonicalize_gripper_qpos


_CAMERA_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_CAMERA_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def _quat_angle_deg_wxyz(q1, q2) -> float:
    q1 = torch.as_tensor(q1, dtype=torch.float32).reshape(-1)
    q2 = torch.as_tensor(q2, dtype=torch.float32).reshape(-1)
    q1 = q1 / torch.clamp(torch.linalg.norm(q1), min=1e-12)
    q2 = q2 / torch.clamp(torch.linalg.norm(q2), min=1e-12)
    dot = torch.clamp(torch.abs(torch.dot(q1, q2)), min=-1.0, max=1.0)
    angle_rad = 2.0 * torch.arccos(dot)
    return float(torch.rad2deg(angle_rad).cpu())


class Policy(BasePolicy):
    def __init__(self, args):
        explicit_ckpt_path = (
            args.get("ckpt_path")
            or os.environ.get("ACT_CKPT_PATH")
        )
        explicit_ckpt_dir = (
            args.get("ckpt_dir")
            or os.environ.get("ACT_CKPT_DIR")
        )

        self.train_config_name = os.environ.get(
            "ACT_TRAIN_CONFIG", os.environ.get("TRAIN_CONFIG", "train_config_ee")
        )
        self.ep_num = os.environ.get("EP_NUM", "50")
        if explicit_ckpt_path:
            ckpt_path = Path(explicit_ckpt_path).expanduser().resolve()
            ckpt_dir = ckpt_path.parent
        elif explicit_ckpt_dir:
            ckpt_dir = Path(explicit_ckpt_dir).expanduser().resolve()
            ckpt_path = None
        else:
            raise ValueError(
                "ACT deployment requires ACT_CKPT_PATH or ACT_CKPT_DIR."
            )

        self.task_name = args["task_name"]
        with open(REPO_ROOT / "policy" / "task_settings.json", "r") as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get("camera_type", "head")
        print(f"Using camera type '{self.camera_type}' for task '{self.task_name}'")

        args_path = ckpt_dir / "args.json"
        if args_path.exists():
            with open(args_path, "r", encoding="utf-8") as f:
                saved_args = json.load(f)
            if isinstance(saved_args.get("policy_config"), dict):
                train_config = saved_args["policy_config"]
            else:
                train_config = saved_args
        else:
            with open(Path(__file__).parent / f"{self.train_config_name}.yml", "r", encoding="utf-8") as f:
                train_config = yaml.load(f, Loader=yaml.FullLoader)

        self.camera_names = list(train_config.get("camera_names", ["cam_high"]))
        self.tactile_names = list(train_config.get("tactile_names", ["tac_left", "tac_right"]))
        print(f"Configured ACT cameras: {self.camera_names}, tactile: {self.tactile_names}")

        train_config.update(
            {
                "task_name": args["task_name"],
                "task_config": args["task_config"],
                "ckpt_dir": str(ckpt_dir),
                "ckpt_path": str(ckpt_path) if ckpt_path is not None else None,
                "seed": args.get("seed", 0),
                "num_epochs": 1,
                "temporal_agg": False,
            }
        )

        self.model = ACT(train_config)
        self._debug_enabled = os.environ.get("UNIVTAC_DEBUG_DUMP", "0") == "1"
        self._debug_writer = RolloutDebugWriter(
            enabled=self._debug_enabled,
            root_dir=os.environ.get("UNIVTAC_DEBUG_DIR"),
            policy_name="ACT",
            variant=self.train_config_name,
        )
        self._debug_seed = None
        self._inference_count = 0
        print(f"ACT policy loaded from {ckpt_dir}")

    def _update_task_overlay(
        self,
        task,
        *,
        inference_count: int,
        current_pose=None,
        target_pose=None,
        inferencing=None,
        executing=None,
    ) -> None:
        if not hasattr(task, "set_video_overlay_state"):
            return
        payload = {
            "inference_count": int(inference_count),
        }
        if current_pose is not None:
            payload["current_eef_pose"] = torch.as_tensor(current_pose).detach().cpu().tolist()
        if target_pose is not None:
            payload["target_eef_pose"] = torch.as_tensor(target_pose).detach().cpu().tolist()
        if inferencing is not None:
            payload["inferencing"] = bool(inferencing)
        if executing is not None:
            payload["executing"] = bool(executing)
        task.set_video_overlay_state(**payload)

    def _camera_transform(self, img: torch.Tensor):
        img = transforms.Resize((256, 256))(img.permute(2, 0, 1))
        img = img / 255.0
        img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img

    def _tactile_transform(self, img: torch.Tensor):
        img = transforms.Resize((256, 256))(img.permute(2, 0, 1))
        img = img / 255.0
        return img

    def _eef_pose_from_observation(self, observation: dict) -> torch.Tensor:
        ee = observation["embodiment"]["ee"][:7].float()
        grip = canonicalize_gripper_qpos(observation["embodiment"]["joint"][-2:])[0:1].float()
        return torch.cat([ee, grip], dim=0)

    def encode_obs(self, observation):
        ret = {
            "qpos": self._eef_pose_from_observation(observation).cpu().numpy()
        }

        camera_sources = {
            "cam_high": ("head", "rgb"),
            "cam_wrist": ("wrist", "rgb"),
        }
        for cam_name in self.camera_names:
            if cam_name not in camera_sources:
                raise KeyError(f"Unsupported ACT camera name in deploy: {cam_name}")
            obs_group, obs_key = camera_sources[cam_name]
            ret[cam_name] = self._camera_transform(observation["observation"][obs_group][obs_key])

        tactile_sources = observation.get("tactile", {})
        left_tactile_source = tactile_sources.get("left_tactile") or tactile_sources.get("left_gsmini")
        right_tactile_source = tactile_sources.get("right_tactile") or tactile_sources.get("right_gsmini")
        tactile_name_to_source = {
            "tac_left": left_tactile_source,
            "tac_right": right_tactile_source,
        }
        for tactile_name in self.tactile_names:
            tactile_obs = tactile_name_to_source.get(tactile_name)
            if tactile_obs is None:
                available = sorted(tactile_sources.keys())
                raise KeyError(
                    f"Missing tactile source for {tactile_name}. Available tactile keys: {available}"
                )
            ret[tactile_name] = self._tactile_transform(tactile_obs["rgb_marker"])

        return ret

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
                "n_demo": self.ep_num,
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

    def _save_rgb_tensor(self, path: Path, chw: torch.Tensor):
        image = chw.detach().cpu()
        image = torch.clamp(image, 0.0, 1.0)
        image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def _camera_input_for_debug(self, chw: torch.Tensor) -> torch.Tensor:
        return torch.clamp(chw.detach().cpu() * _CAMERA_STD + _CAMERA_MEAN, 0.0, 1.0)

    def _save_inference_inputs(self, *, inference_index: int, encoded_obs) -> None:
        if not self._debug_writer.should_save_inference_montage(inference_index=inference_index):
            return

        tiles = {}
        for cam_name in self.camera_names:
            if cam_name in encoded_obs:
                tiles[f"input {cam_name}"] = self._camera_input_for_debug(encoded_obs[cam_name])
        for tactile_name in self.tactile_names:
            if tactile_name in encoded_obs:
                tiles[f"input {tactile_name}"] = encoded_obs[tactile_name]
        self._debug_writer.save_inference_montage(inference_index=inference_index, tiles=tiles)

    def _query_action_payload(self, query_actions: np.ndarray, current_pose: torch.Tensor) -> list[dict]:
        payload = []
        current_pose = current_pose.detach().cpu()
        for idx, query_action in enumerate(query_actions):
            query_action_t = torch.from_numpy(query_action).float()
            payload.append(
                {
                    "index": idx,
                    "action_8d": query_action.tolist(),
                    "delta_pos_mm_from_obs": float(
                        torch.linalg.norm(query_action_t[:3] - current_pose[:3]).item() * 1000.0
                    ),
                    "delta_rot_deg_from_obs": _quat_angle_deg_wxyz(current_pose[3:7], query_action_t[3:7]),
                    "delta_grip_from_obs": float(torch.abs(query_action_t[7] - current_pose[7]).item()),
                }
            )
        return payload

    def eval(self, task, observation):
        self._ensure_debug_rollout(task)
        obs = self.encode_obs(observation)
        if self._debug_enabled and self.model.t % 10 == 0 and self._debug_writer.rollout_dir is not None:
            self.save(task.get_frame_shot(observation), task.take_action_cnt, self._debug_writer.rollout_dir)

        current_eef_pose = self._eef_pose_from_observation(observation).detach().cpu()
        model_t_before = int(self.model.t)
        query_frequency = max(1, int(getattr(self.model, "query_frequency", 1)))
        did_inference = (model_t_before % query_frequency) == 0
        inference_index = self._inference_count if did_inference else (self._inference_count - 1 if self._inference_count > 0 else None)
        latency_start = time.perf_counter()
        action = self.model.get_action(obs).reshape(-1)
        policy_latency_ms = (time.perf_counter() - latency_start) * 1000.0
        query_actions = None
        if did_inference and hasattr(self.model, "all_actions"):
            query_actions = self.model.post_process(self.model.all_actions.detach().cpu().numpy())[0]

        action = torch.from_numpy(action).to(task.device).float()
        action = self._normalize_quat_in_action(
            action, observation["embodiment"]["ee"][3:7].float().to(task.device)
        )
        if did_inference:
            self._save_inference_inputs(inference_index=self._inference_count, encoded_obs=obs)

        action_cpu = action.detach().cpu()
        display_inference_count = int(self._inference_count + (1 if did_inference else 0))
        self._update_task_overlay(
            task,
            inference_count=display_inference_count,
            current_pose=current_eef_pose,
            target_pose=action_cpu,
            inferencing=did_inference,
            executing=False,
        )
        self._update_task_overlay(
            task,
            inference_count=display_inference_count,
            current_pose=current_eef_pose,
            target_pose=action_cpu,
            inferencing=False,
            executing=True,
        )
        if hasattr(task, "set_rollout_trace_context"):
            task.set_rollout_trace_context(
                policy_name="ACT",
                mode="policy",
                inference_count=display_inference_count,
                action_buffer_index=int(model_t_before % query_frequency),
                execution_horizon=int(query_frequency),
                did_inference=bool(did_inference),
                n_action_steps=int(getattr(self.model, "num_queries", 1)),
            )
        step_payload = {
            "sim_step": int(getattr(task, "step_count", 0)) + 1,
            "did_inference": bool(did_inference),
            "inference_index": inference_index,
            "action_buffer_index": int(model_t_before % query_frequency),
            "execution_horizon": int(query_frequency),
            "policy_latency_ms": policy_latency_ms if did_inference else None,
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
            "inference_actions": None if query_actions is None else self._query_action_payload(query_actions, current_eef_pose),
        }

        if did_inference:
            self._inference_count += 1
        task.take_action(action, action_type="ee_servo")
        if hasattr(task, "consume_physx_joint_writes"):
            step_payload["physx_joint_writes"] = task.consume_physx_joint_writes()
        self._debug_writer.record_step(step_payload)

    def reset(self):
        if hasattr(self.model, "reset"):
            self.model.reset()
        self._debug_seed = None
        self._inference_count = 0

    def on_rollout_end(self, task, result: str | None = None):
        self._debug_writer.finalize_rollout(result=result)
        self._debug_seed = None

    def close(self):
        self._debug_writer.close()
        super().close()

    def save(self, img, t, dump_dir: Path | None = None):
        from PIL import Image
        from PIL import ImageDraw, ImageFont

        obs = Image.fromarray(img.cpu().numpy())

        draw = ImageDraw.Draw(obs)
        font = ImageFont.load_default()

        draw.text((obs.width - 120, obs.height - 60), f"{t:03d}", fill=(255, 0, 0), font=font)
        if dump_dir is None:
            output_path = Path(f"ACT_{self.task_name}_{self.train_config_name}.png")
        else:
            output_path = dump_dir / "rollout_overview_latest.png"
        obs.save(output_path)
