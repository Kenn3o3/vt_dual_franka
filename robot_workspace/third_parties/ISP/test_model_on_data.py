#!/usr/bin/env python3
"""Test ISP model checkpoint on actual training data.

Loads an HDF5 episode and feeds the real training observations through
the model to see if predictions match the ground truth actions.
This isolates the model from the simulator to determine if the model
itself is the problem.

Usage:
    python policy/ISP/test_model_on_data.py \
        --ckpt /home/zhenya/kenny/visuotact/UniVTAC/policy/ISP/data/outputs/tasks/insert_HDMI_demo100/isp_so2/checkpoints/best.ckpt \
        --data data/insert_HDMI/clean/0.hdf5 \
        --start_step 0
"""

import sys
from pathlib import Path

_ISP_ROOT = str(Path(__file__).parent)
if _ISP_ROOT not in sys.path:
    sys.path.insert(0, _ISP_ROOT)

import argparse
import json

import cv2
import dill
import h5py
import hydra
import numpy as np
import scipy.sparse
import torch
from omegaconf import OmegaConf

from isp.common.univtac_util import canonicalize_gripper_qpos, gripper_scalar_from_qpos
from isp.dataset.univtac_replay_image_dataset import _decode_jpeg_to_rgb
from isp.model.common.rotation_transformer import RotationTransformer

# Monkey-patch for escnn/sklearn compatibility
_orig_todense = scipy.sparse.spmatrix.todense


def _patched_todense(self, order=None, out=None):
    return np.asarray(_orig_todense(self, order=order, out=out))


scipy.sparse.spmatrix.todense = _patched_todense


_H5_RGB_SOURCE = {
    "agentview_image": ["observation/head/rgb"],
    "robot0_eye_in_hand_image": ["observation/wrist/rgb", "observation/head/rgb"],
    "robot0_tactile_left_image": ["tactile/left_gsmini/rgb", "tactile/left_tactile/rgb"],
    "robot0_tactile_right_image": ["tactile/right_gsmini/rgb", "tactile/right_tactile/rgb"],
}


def load_model(ckpt_path, device="cpu"):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill, map_location="cpu")
    cfg = payload["cfg"]

    model = hydra.utils.instantiate(cfg.policy)
    if "ema_model" in payload["state_dicts"]:
        model.load_state_dict(payload["state_dicts"]["ema_model"])
        print("Loaded EMA model weights")
    else:
        model.load_state_dict(payload["state_dicts"]["model"])
        print("Loaded model weights (no EMA found)")

    model.eval()
    model.to(device)

    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    rgb_keys = []
    lowdim_keys = []
    rgb_shapes = {}
    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") == "rgb":
            rgb_keys.append(key)
            rgb_shapes[key] = tuple(attr["shape"])
        else:
            lowdim_keys.append(key)

    print(f"Model: n_obs_steps={model.n_obs_steps}, n_action_steps={model.n_action_steps}")
    print(f"  rgb_keys={rgb_keys}, lowdim_keys={lowdim_keys}")
    if hasattr(model, "ws_center"):
        print(f"  ws_center={model.ws_center.tolist()}")

    for key in lowdim_keys:
        try:
            stats = model.normalizer.params_dict[key]["input_stats"]
            print(f"  normalizer [{key}]: min={stats['min'].tolist()}, max={stats['max'].tolist()}")
        except Exception:
            pass
    try:
        stats = model.normalizer.params_dict["action"]["input_stats"]
        print(f"  normalizer [action]: min={stats['min'].tolist()}, max={stats['max'].tolist()}")
    except Exception:
        pass

    return model, cfg, rgb_keys, lowdim_keys, rgb_shapes


def _load_rgb_frame(h5_file, key, t, rgb_shapes):
    source_candidates = _H5_RGB_SOURCE.get(key)
    if source_candidates is None:
        raise ValueError(f"Unsupported RGB key in checkpoint shape_meta: {key}")

    source_path = next((path for path in source_candidates if path in h5_file), None)
    if source_path is None:
        raise KeyError(
            f"Missing source dataset for key '{key}'. Tried: {source_candidates}"
        )

    c, h, w = rgb_shapes[key]
    if c != 3:
        raise ValueError(f"Only RGB(3 channels) is supported, got {c} for {key}")

    img_rgb = _decode_jpeg_to_rgb(h5_file[source_path][t])
    if img_rgb.shape[0] != h or img_rgb.shape[1] != w:
        img_rgb = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0


def load_episode(hdf5_path, rgb_keys, lowdim_keys, rgb_shapes):
    """Load an episode from HDF5, matching ISP dataset preprocessing."""
    rot_tf = RotationTransformer("quaternion", "rotation_6d")

    with h5py.File(hdf5_path, "r") as f:
        ee_raw = f["embodiment/ee"][:]
        joint_raw = f["embodiment/joint"][:]
        T_raw = ee_raw.shape[0]

        obs_ee = ee_raw[:-1]
        act_ee = ee_raw[1:]
        obs_joint = joint_raw[:-1]
        act_joint = joint_raw[1:]
        T = obs_ee.shape[0]

        print(f"\nEpisode: {hdf5_path}")
        print(f"  T_raw={T_raw}, T_obs={T}")
        print(f"  obs_ee[0]  = {obs_ee[0].tolist()}")
        print(f"  obs_ee[-1] = {obs_ee[-1].tolist()}")
        print(f"  obs_joint[0, 7:9] = {obs_joint[0, 7:9].tolist()}")

        obs_frames = []
        for t in range(T):
            obs = {}
            for key in lowdim_keys:
                if key == "robot0_eef_pos":
                    obs[key] = torch.from_numpy(obs_ee[t, :3].astype(np.float32))
                elif key == "robot0_eef_quat":
                    obs[key] = torch.from_numpy(obs_ee[t, 3:7].astype(np.float32))
                elif key == "robot0_gripper_qpos":
                    gripper_qpos = canonicalize_gripper_qpos(
                        obs_joint[t, 7:9].astype(np.float32)
                    ).astype(np.float32)
                    obs[key] = torch.from_numpy(gripper_qpos)
                else:
                    raise ValueError(f"Unsupported lowdim key in checkpoint shape_meta: {key}")

            for key in rgb_keys:
                obs[key] = _load_rgb_frame(f, key, t, rgb_shapes)

            obs_frames.append(obs)

        action_pos = act_ee[:, :3].astype(np.float32)
        action_quat_wxyz = act_ee[:, 3:7].astype(np.float32)
        action_rot6d = rot_tf.forward(torch.from_numpy(action_quat_wxyz)).numpy()
        action_gripper = gripper_scalar_from_qpos(act_joint[:, 7:9]).astype(np.float32)
        actions_10d = np.concatenate(
            [action_pos, action_rot6d, action_gripper], axis=-1
        ).astype(np.float32)

        print(f"  actions_10d[0] pos={actions_10d[0, :3].tolist()}, grip={actions_10d[0, 9]:.6f}")
        print(f"  actions_10d[-1] pos={actions_10d[-1, :3].tolist()}, grip={actions_10d[-1, 9]:.6f}")

    return obs_frames, torch.from_numpy(actions_10d)


@torch.no_grad()
def test_model(model, obs_frames, gt_actions, rgb_keys, lowdim_keys, device, start_step=0):
    n_obs = model.n_obs_steps
    T = len(obs_frames)

    dump_dir = Path("train_data")
    dump_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 80}")
    print(f"Testing model predictions starting from step {start_step}")
    print(f"Dumping training data to {dump_dir.resolve()}")
    print(f"{'=' * 80}")

    for inf_idx, inf_start in enumerate(range(start_step, min(start_step + 3, T - n_obs))):
        obs_dict = {}
        frames = obs_frames[inf_start : inf_start + n_obs]
        for key in frames[0]:
            stacked = torch.stack([f[key] for f in frames], dim=0)
            obs_dict[key] = stacked.unsqueeze(0).to(device)

        inf_dir = dump_dir / f"inf_{inf_idx:03d}"
        inf_dir.mkdir(parents=True, exist_ok=True)
        lowdim_dump = {}
        for t in range(n_obs):
            for key in rgb_keys:
                img_chw = obs_dict[key][0, t].cpu()
                img_hwc = (img_chw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                cv2.imwrite(
                    str(inf_dir / f"{key}_t{t}.png"),
                    cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR),
                )
            for key in lowdim_keys:
                lowdim_dump[f"{key}_t{t}"] = obs_dict[key][0, t].cpu().tolist()
        with open(inf_dir / "lowdim.json", "w") as f_out:
            json.dump(lowdim_dump, f_out, indent=2)

        print(
            f"\n--- Inference at step {inf_start} "
            f"(using obs steps {inf_start}-{inf_start + n_obs - 1}) ---"
        )
        for key in lowdim_keys:
            val = obs_dict[key][0, -1].cpu().tolist()
            print(f"  obs {key} = {val}")

        result = model.predict_action(obs_dict)
        pred_actions = result["action"][0].cpu()

        actions_dump = {}
        for i in range(pred_actions.shape[0]):
            a = pred_actions[i].tolist()
            actions_dump[f"action_{i}"] = {"pos": a[:3], "rot6d": a[3:9], "grip": a[9]}
        with open(inf_dir / "actions.json", "w") as f_out:
            json.dump(actions_dump, f_out, indent=2)

        gt_start = inf_start + n_obs - 1
        gt_dump = {}
        for i in range(min(pred_actions.shape[0], T - gt_start)):
            gt_idx = gt_start + i
            if gt_idx >= T:
                break
            gt = gt_actions[gt_idx].tolist()
            gt_dump[f"gt_action_{i}"] = {"pos": gt[:3], "rot6d": gt[3:9], "grip": gt[9]}
        with open(inf_dir / "gt_actions.json", "w") as f_out:
            json.dump(gt_dump, f_out, indent=2)

        print(
            f"\n  {'':>4s} | {'Predicted pos':^40s} | {'Ground truth pos':^40s} | {'Pos error':^12s}"
        )
        print(f"  {'':>4s} | {'Predicted Z':>12s} | {'GT Z':>12s}")
        for i in range(min(pred_actions.shape[0], T - gt_start)):
            gt_idx = gt_start + i
            if gt_idx >= T:
                break
            pred = pred_actions[i].tolist()
            gt = gt_actions[gt_idx].tolist()
            pos_err = np.linalg.norm(np.array(pred[:3]) - np.array(gt[:3]))
            print(f"  [{i:2d}] | pred_pos={pred[:3]} | gt_pos={gt[:3]} | err={pos_err:.6f}")

            pred_r = pred[3:9]
            gt_r = gt[3:9]
            rot_err = np.linalg.norm(np.array(pred_r) - np.array(gt_r))
            print(f"       | pred_rot6d={[f'{v:.4f}' for v in pred_r]}")
            print(f"       | gt_rot6d  ={[f'{v:.4f}' for v in gt_r]}  rot_err={rot_err:.6f}")
            print(f"       | pred_grip={pred[9]:.6f} gt_grip={gt[9]:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument("--data", required=True, help="HDF5 episode path")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--start_step", type=int, default=0)
    args = parser.parse_args()

    model, cfg, rgb_keys, lowdim_keys, rgb_shapes = load_model(args.ckpt, args.device)
    obs_frames, gt_actions = load_episode(args.data, rgb_keys, lowdim_keys, rgb_shapes)
    test_model(
        model,
        obs_frames,
        gt_actions,
        rgb_keys,
        lowdim_keys,
        args.device,
        args.start_step,
    )


if __name__ == "__main__":
    main()
