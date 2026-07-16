import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
ACT_ROOT = SCRIPT_DIR
TASK_SETTINGS_PATH = REPO_ROOT / "policy" / "task_settings.json"

sys.path.insert(0, str(ACT_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import os
import h5py
import numpy as np
import argparse
import json
from tqdm import tqdm
from univtac_util import gripper_scalar_from_qpos
import cv2

RAW_CAMERA_TO_OUTPUT = {
    "head": "cam_high",
    "wrist": "cam_wrist",
}


def detect_available_modalities(dataset_path, preferred_camera_type):
    raw_camera_names = []
    with h5py.File(str(dataset_path), "r") as f:
        for raw_camera_name in RAW_CAMERA_TO_OUTPUT:
            if f"observation/{raw_camera_name}/rgb" in f:
                raw_camera_names.append(raw_camera_name)

        if not raw_camera_names and preferred_camera_type != "all":
            fallback_path = f"observation/{preferred_camera_type}/rgb"
            if fallback_path in f:
                raw_camera_names.append(preferred_camera_type)

        if "tactile/left_tactile/rgb_marker" in f and "tactile/right_tactile/rgb_marker" in f:
            tactile_paths = (
                "tactile/left_tactile/rgb_marker",
                "tactile/right_tactile/rgb_marker",
            )
        elif "tactile/left_gsmini/rgb_marker" in f and "tactile/right_gsmini/rgb_marker" in f:
            tactile_paths = (
                "tactile/left_gsmini/rgb_marker",
                "tactile/right_gsmini/rgb_marker",
            )
        else:
            raise KeyError(f"Unsupported tactile streams in {dataset_path}")

    if not raw_camera_names:
        raise KeyError(
            f"No supported camera streams found in {dataset_path}. "
            f"Expected one of: {list(RAW_CAMERA_TO_OUTPUT)}"
        )

    return raw_camera_names, tactile_paths


def decode_jpeg_sequence(dataset) -> np.ndarray:
    images = []
    for item in dataset:
        if isinstance(item, (bytes, bytearray, np.bytes_)):
            buffer = np.frombuffer(bytes(item), dtype=np.uint8)
        elif isinstance(item, np.ndarray) and item.dtype == np.uint8:
            buffer = item
        else:
            raise TypeError(f"Unsupported encoded image buffer type: {type(item)}")
        image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imdecode returned None while preparing ACT data.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    return np.asarray(images, dtype=np.uint8)


def load_hdf5(dataset_paths, raw_camera_names, tactile_paths, downsample_factor):
    episode_start = 0
    episode_ends = []
    per_episode = []
    for dataset_path in dataset_paths:
        with h5py.File(dataset_path, "r") as f:
            joint = f["embodiment/joint"][:]
            ee = f["embodiment/ee"][:]
            downsample_idx = np.arange(0, len(joint) - 1, downsample_factor)
            episode = {
                "embodiment/joint_state": joint[:-1][downsample_idx],
                "embodiment/joint_action": joint[1:][downsample_idx],
                "embodiment/ee_state": ee[:-1][downsample_idx],
                "embodiment/ee_action": ee[1:][downsample_idx],
            }
            for raw_camera_name in raw_camera_names:
                key = f"observation/{raw_camera_name}/rgb"
                episode[key] = decode_jpeg_sequence(f[key])[ :-1][downsample_idx]
            for tactile_path in tactile_paths:
                episode[tactile_path] = decode_jpeg_sequence(f[tactile_path])[:-1][downsample_idx]
            episode_start += len(downsample_idx)
            episode_ends.append(episode_start)
            per_episode.append(episode)

    keys = list(per_episode[0].keys())
    data = {key: np.concatenate([episode[key] for episode in per_episode], axis=0) for key in keys}
    data["episode_ends"] = np.asarray(episode_ends, dtype=np.int64)
    return data


def data_transform(path, episode_num, save_path):
    hdf5_dir = Path(path) / "hdf5"
    if not hdf5_dir.exists():
        hdf5_dir = Path(path)
        if len(list(hdf5_dir.glob("*.hdf5"))) == 0:
            print(f"HDF5 directory does not exist at \n{hdf5_dir}\n")
            raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")

    hdf5_files = sorted(hdf5_dir.glob("*.hdf5"), key=lambda x: int(x.stem))
    if episode_num <= 0:
        episode_num = len(hdf5_files)
    assert episode_num <= len(hdf5_files), f"data num not enough: requested {episode_num}, found {len(hdf5_files)}"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    global task_name
    with open(TASK_SETTINGS_PATH, "r") as f:
        task_settings = json.load(f)
    assert task_name in task_settings, f"Task '{task_name}' not found in task_settings.json"
    preferred_camera_type = task_settings[task_name].get("camera_type", "head")
    downsample_factor = task_settings[task_name].get("downsample", 1)

    dataset_paths = [str(hdf5_files[i]) for i in range(episode_num)]
    raw_camera_names, tactile_paths = detect_available_modalities(
        hdf5_files[0], preferred_camera_type
    )
    print(
        f"Loading {episode_num} EE episodes with cameras {raw_camera_names}, "
        f"downsample factor {downsample_factor}."
    )
    data = load_hdf5(dataset_paths[:episode_num], raw_camera_names, tactile_paths, downsample_factor)

    ee_state_all = data["embodiment/ee_state"][:, :7].astype(np.float32)
    ee_action_all = data["embodiment/ee_action"][:, :7].astype(np.float32)
    joint_state_all = data["embodiment/joint_state"].astype(np.float32)
    joint_action_all = data["embodiment/joint_action"].astype(np.float32)
    # Match the ISP EE convention: raw embodiment/ee plus a single gripper scalar from the last two joints.
    qpos_all = np.concatenate(
        [
            ee_state_all,
            gripper_scalar_from_qpos(joint_state_all[:, -2:]).astype(np.float32),
        ],
        axis=-1,
    )
    action_all = np.concatenate(
        [
            ee_action_all,
            gripper_scalar_from_qpos(joint_action_all[:, -2:]).astype(np.float32),
        ],
        axis=-1,
    )
    camera_data_all = {
        raw_camera_name: data[f"observation/{raw_camera_name}/rgb"]
        for raw_camera_name in raw_camera_names
    }
    left_tac_all = data[tactile_paths[0]]
    right_tac_all = data[tactile_paths[1]]
    episode_ends = data["episode_ends"]

    start_idx = 0
    for i in tqdm(range(episode_num), desc="Writing episodes"):
        end_idx = episode_ends[i]

        qpos = qpos_all[start_idx:end_idx]
        action = action_all[start_idx:end_idx]
        camera_data = {
            raw_camera_name: camera_data_all[raw_camera_name][start_idx:end_idx]
            for raw_camera_name in raw_camera_names
        }
        left_tac = left_tac_all[start_idx:end_idx]
        right_tac = right_tac_all[start_idx:end_idx]

        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")
        with h5py.File(hdf5path, "w") as f:
            f.create_dataset("action", data=np.asarray(action))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.asarray(qpos))
            image = obs.create_group("images")
            for raw_camera_name in raw_camera_names:
                image.create_dataset(
                    RAW_CAMERA_TO_OUTPUT[raw_camera_name],
                    data=np.asarray(camera_data[raw_camera_name]),
                    dtype=np.uint8,
                )
            image.create_dataset("tac_left", data=np.asarray(left_tac), dtype=np.uint8)
            image.create_dataset("tac_right", data=np.asarray(right_tac), dtype=np.uint8)
        start_idx = end_idx

    camera_output_names = [RAW_CAMERA_TO_OUTPUT[raw_camera_name] for raw_camera_name in raw_camera_names]
    return episode_num, camera_output_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TacArena episodes for ACT EE training.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., insert_hole)",
    )
    parser.add_argument("task_config", type=str, help="Task config (e.g., demo)")
    parser.add_argument("expert_data_num", nargs="?", default=-1, type=int, help="Number of episodes to process")
    parser.add_argument("--input-dir", type=str, default=None, help="Explicit raw task directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Explicit processed dataset directory")
    parser.add_argument("--sim-task-configs-path", type=str, default=str(SCRIPT_DIR / "SIM_TASK_CONFIGS.json"))
    parser.add_argument("--task-key", type=str, default=None, help="Task key to register in SIM_TASK_CONFIGS")
    parser.add_argument("--skip-sim-task-configs", action="store_true")

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num

    input_path = args.input_dir or str(REPO_ROOT / "data" / task_name / task_config)
    output_path = args.output_dir or str(SCRIPT_DIR / "data" / f"sim-{task_name}-ee" / f"{task_config}-{expert_data_num}")

    begin, camera_output_names = data_transform(input_path, expert_data_num, output_path)

    if not args.skip_sim_task_configs:
        sim_task_configs_path = Path(args.sim_task_configs_path)
        try:
            with open(sim_task_configs_path, "r", encoding="utf-8") as f:
                sim_task_configs = json.load(f)
        except Exception:
            sim_task_configs = {}

        task_key = args.task_key or f"sim-{task_name}-ee-{task_config}-{begin}"
        sim_task_configs[task_key] = {
            "dataset_dir": str(Path(output_path)),
            "num_episodes": begin,
            "episode_len": 1000,
            "camera_names": camera_output_names + ["tac_left", "tac_right"],
        }

        with open(sim_task_configs_path, "w", encoding="utf-8") as f:
            json.dump(sim_task_configs, f, indent=4)
