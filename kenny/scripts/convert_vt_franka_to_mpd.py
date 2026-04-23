"""Convert vt_franka aligned episode data to movement-primitive-diffusion format.

Usage:
    python kenny/scripts/convert_vt_franka_to_mpd.py \
        --run-dir robot_workspace/data/runs/put_cup_on_plate_20260422_171246 \
        --output-dir robot_workspace/data/mpd/put_cup_on_plate \
        --val-episodes 2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

IMAGE_KEYS = ("rgb_wrist", "rgb_third_person")
TARGET_IMAGE_SIZE = (320, 240)  # (W, H) for cv2.resize


def load_episode(episode_dir: Path) -> dict[str, np.ndarray]:
    npz_path = episode_dir / "aligned_episode.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"No aligned_episode.npz in {episode_dir}")
    return dict(np.load(npz_path, allow_pickle=True))


def resolve_image_paths(
    episode_dir: Path, frame_paths: np.ndarray
) -> list[Path]:
    resolved = []
    for rel in frame_paths:
        rel = str(rel)
        if not rel:
            raise ValueError(f"Empty frame path in {episode_dir}")
        full = episode_dir / rel
        if not full.exists():
            raise FileNotFoundError(f"Image not found: {full}")
        resolved.append(full)
    return resolved


def convert_episode(
    episode_dir: Path,
    output_dir: Path,
    demo_name: str,
) -> dict[str, int]:
    data = load_episode(episode_dir)
    T = data["timestamps"].shape[0]

    # Find timesteps where all image paths are present
    valid_mask = np.ones(T, dtype=bool)
    for cam_key in IMAGE_KEYS:
        path_key = f"{cam_key}_frame_paths"
        if path_key not in data:
            continue
        for i, rel in enumerate(data[path_key]):
            rel = str(rel)
            if not rel or not (episode_dir / rel).exists():
                valid_mask[i] = False

    valid_indices = np.where(valid_mask)[0]
    n_dropped = T - len(valid_indices)
    if n_dropped > 0:
        print(f"(dropped {n_dropped}/{T} steps with missing images) ", end="", flush=True)
    T_valid = len(valid_indices)
    if T_valid == 0:
        raise RuntimeError(f"No valid timesteps in {episode_dir}")

    demo_dir = output_dir / demo_name
    if demo_dir.exists():
        shutil.rmtree(demo_dir)
    demo_dir.mkdir(parents=True)

    tcp_pose = data["robot_tcp_pose"][valid_indices].astype(np.float64)  # (T_valid, 7)
    gripper_width = data["gripper_width"][valid_indices].astype(np.float64).reshape(T_valid, 1)
    teleop_tcp = data["teleop_target_tcp"][valid_indices].astype(np.float64)
    gripper_closed = data["teleop_gripper_closed"][valid_indices].astype(np.float64).reshape(T_valid, 1)
    actions = np.concatenate([teleop_tcp, gripper_closed], axis=1)  # (T_valid, 8)

    np.savez_compressed(demo_dir / "tcp_pose.npz", tcp_pose)
    np.savez_compressed(demo_dir / "gripper_width.npz", gripper_width)
    np.savez_compressed(demo_dir / "actions.npz", actions)

    stats = {"timesteps": T_valid, "dropped": n_dropped}

    for cam_key in IMAGE_KEYS:
        path_key = f"{cam_key}_frame_paths"
        if path_key not in data:
            continue
        frame_paths = data[path_key][valid_indices]
        cam_dir = demo_dir / cam_key
        cam_dir.mkdir()
        for i, rel in enumerate(frame_paths):
            src = episode_dir / str(rel)
            img = cv2.imread(str(src))
            if img is None:
                raise RuntimeError(f"Failed to read image: {src}")
            img = cv2.resize(img, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(cam_dir / f"{i:04d}.png"), img)
        stats[f"{cam_key}_images"] = len(frame_paths)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Convert vt_franka data to MPD format")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to vt_franka run directory")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for MPD data")
    parser.add_argument("--val-episodes", type=int, default=2, help="Number of episodes for validation split")
    args = parser.parse_args()

    episodes_dir = args.run_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"No episodes directory in {args.run_dir}")

    episode_dirs = sorted(
        d for d in episodes_dir.iterdir()
        if d.is_dir() and (d / "aligned_episode.npz").exists()
    )
    if not episode_dirs:
        raise FileNotFoundError(f"No episodes with aligned data in {episodes_dir}")

    n_total = len(episode_dirs)
    n_val = min(args.val_episodes, n_total - 1)
    n_train = n_total - n_val
    print(f"Found {n_total} episodes: {n_train} train, {n_val} val")

    if args.output_dir.exists():
        print(f"Removing existing output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)

    train_dir = args.output_dir / "train"
    val_dir = args.output_dir / "val"
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)

    train_episodes = episode_dirs[:n_train]
    val_episodes = episode_dirs[n_train:]

    all_stats = []
    for i, ep_dir in enumerate(train_episodes):
        demo_name = f"demo_{i:03d}"
        print(f"  [train] {ep_dir.name} -> {demo_name} ...", end=" ", flush=True)
        stats = convert_episode(ep_dir, train_dir, demo_name)
        print(f"T={stats['timesteps']}")
        all_stats.append({"split": "train", "source": ep_dir.name, "demo": demo_name, **stats})

    for i, ep_dir in enumerate(val_episodes):
        demo_name = f"demo_{i:03d}"
        print(f"  [val]   {ep_dir.name} -> {demo_name} ...", end=" ", flush=True)
        stats = convert_episode(ep_dir, val_dir, demo_name)
        print(f"T={stats['timesteps']}")
        all_stats.append({"split": "val", "source": ep_dir.name, "demo": demo_name, **stats})

    manifest = {
        "source_run": str(args.run_dir),
        "target_image_size": list(TARGET_IMAGE_SIZE),
        "image_keys": list(IMAGE_KEYS),
        "state_keys": ["tcp_pose", "gripper_width"],
        "action_keys": ["actions"],
        "action_description": "teleop_target_tcp (7D) + gripper_closed (1D)",
        "fps": 10,
        "episodes": all_stats,
    }
    manifest_path = args.output_dir / "conversion_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _compute_and_save_scalers(args.output_dir)

    total_steps = sum(s["timesteps"] for s in all_stats)
    print(f"\nDone. {n_total} episodes, {total_steps} total steps.")
    print(f"Output: {args.output_dir}")
    print(f"Manifest: {manifest_path}")


def _compute_and_save_scalers(output_dir: Path) -> None:
    """Pre-compute normalization scalers from all demos (train + val).

    Saves as both .npz (numpy, no torch dependency) and prints values.
    The inference wrapper converts these to torch tensors at load time.
    """
    scalar_keys = ["tcp_pose", "gripper_width", "actions"]
    all_data: dict[str, list[np.ndarray]] = {k: [] for k in scalar_keys}

    for split in ("train", "val"):
        split_dir = output_dir / split
        if not split_dir.exists():
            continue
        for demo_dir in sorted(split_dir.iterdir()):
            if not demo_dir.is_dir():
                continue
            for key in scalar_keys:
                npz_path = demo_dir / f"{key}.npz"
                if npz_path.exists():
                    arr = np.load(npz_path)["arr_0"]
                    all_data[key].append(arr)

    scaler_dict: dict[str, np.ndarray] = {}
    for key in scalar_keys:
        if not all_data[key]:
            continue
        combined = np.concatenate(all_data[key], axis=0).astype(np.float32)
        scaler_dict[f"{key}_min"] = combined.min(axis=0)
        scaler_dict[f"{key}_max"] = combined.max(axis=0)
        scaler_dict[f"{key}_mean"] = combined.mean(axis=0)
        scaler_dict[f"{key}_std"] = combined.std(axis=0)
        print(f"  scaler[{key}]: min={scaler_dict[f'{key}_min'].tolist()}, max={scaler_dict[f'{key}_max'].tolist()}")

    scaler_path = output_dir / "scaler_values.npz"
    np.savez(scaler_path, **scaler_dict)
    print(f"Scaler values saved to {scaler_path}")


if __name__ == "__main__":
    main()
