from typing import Dict, List, Tuple
import copy
import hashlib
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import zarr
from filelock import FileLock
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from dp.common.normalize_util import (
    abs_action_only_normalizer_from_stat,
    array_to_stats,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from dp.common.pytorch_util import dict_apply
from dp.common.replay_buffer import ReplayBuffer
from dp.common.sampler import SequenceSampler, get_val_mask
from dp.common.univtac_util import (
    canonicalize_gripper_qpos,
    compute_ws_center,
    gripper_scalar_from_qpos,
)
from dp.dataset.base_dataset import BaseImageDataset
from dp.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from dp.model.common.rotation_transformer import RotationTransformer


def _sorted_episode_files(task_dir: Path) -> List[Path]:
    files = list(task_dir.glob("*.hdf5"))
    files = sorted(files, key=lambda p: int(p.stem))
    return files


def _decode_jpeg_to_rgb(jpeg_buffer) -> np.ndarray:
    if isinstance(jpeg_buffer, np.ndarray):
        if jpeg_buffer.dtype == np.uint8:
            arr = jpeg_buffer
        else:
            arr = np.frombuffer(bytes(jpeg_buffer), dtype=np.uint8)
    elif isinstance(jpeg_buffer, (bytes, bytearray, np.bytes_)):
        arr = np.frombuffer(bytes(jpeg_buffer), dtype=np.uint8)
    else:
        raise TypeError(f"Unsupported JPEG buffer type: {type(jpeg_buffer)}")

    image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("cv2.imdecode returned None for JPEG buffer")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def _center_square_crop(image: np.ndarray) -> np.ndarray:
    """Center-crop a rectangular RGB image to the largest square."""
    height, width = image.shape[:2]
    size = min(height, width)
    x0 = (width - size) // 2
    y0 = (height - size) // 2
    return image[y0 : y0 + size, x0 : x0 + size]


def _decode_jpeg_sequence(
    jpeg_dataset,
    target_hw: Tuple[int, int],
    center_crop_square: bool = False,
) -> np.ndarray:
    target_h, target_w = target_hw
    n_frames = int(jpeg_dataset.shape[0])
    output = np.empty((n_frames, target_h, target_w, 3), dtype=np.uint8)
    for i in range(n_frames):
        image = _decode_jpeg_to_rgb(jpeg_dataset[i])
        # NOTE: historical ISP preprocessing directly resized decoded wrist frames
        # from 480x270 to the target size, which squashed the wide-FOV image.
        # if image.shape[0] != target_h or image.shape[1] != target_w:
        #     image = cv2.resize(
        #         image, (target_w, target_h), interpolation=cv2.INTER_AREA
        #     )
        if center_crop_square:
            # NOTE: for BIGFOV wrist RGB we first remove the two side bands so the
            # network sees a square crop centered on the contact-relevant region.
            image = _center_square_crop(image)
        if image.shape[0] != target_h or image.shape[1] != target_w:
            image = cv2.resize(
                image, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
        output[i] = image
    return output


def _shape_meta_signature(shape_meta: dict) -> str:
    payload = json.dumps(shape_meta, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


def _task_root_from_dataset_dir(dataset_dir: Path, split: str) -> Path:
    if dataset_dir.name != "hdf5":
        raise ValueError(f"Expected dataset_dir to end with 'hdf5', got {dataset_dir}")
    if dataset_dir.parent.name == split:
        return dataset_dir.parent.parent
    return dataset_dir.parent


def _dataset_manifest_signature(dataset_dir: Path, episode_files: List[Path], split: str) -> str:
    task_root = _task_root_from_dataset_dir(dataset_dir, split)
    manifest = {
        "task_root": str(task_root),
        "dataset_dir": str(dataset_dir),
        "episodes": [],
    }
    metadata_path = task_root / "metadata.json"
    if metadata_path.is_file():
        manifest["metadata"] = metadata_path.read_text(encoding="utf-8")
    for episode_path in episode_files:
        stat = episode_path.stat()
        manifest["episodes"].append(
            {
                "name": episode_path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    payload = json.dumps(manifest, sort_keys=True)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:10]


IMAGE_OBS_TYPES = {"rgb", "tactile_rgb"}


def _resolve_h5_source_path(h5_file: h5py.File, key: str, obs_type: str) -> str:
    key_to_candidates = {
        "agentview_image": ["observation/head/rgb"],
        "robot0_eye_in_hand_image": ["observation/wrist/rgb", "observation/head/rgb"],
    }
    tactile_key_to_candidates = {
        "robot0_tactile_left_image": [
            "tactile/left_tactile/rgb_marker",
            "tactile/left_gsmini/rgb_marker",
        ],
        "robot0_tactile_right_image": [
            "tactile/right_tactile/rgb_marker",
            "tactile/right_gsmini/rgb_marker",
        ],
    }

    if key in tactile_key_to_candidates:
        candidates = tactile_key_to_candidates[key]
    elif obs_type == "tactile_rgb":
        candidates = tactile_key_to_candidates.get(key)
    elif obs_type == "rgb":
        candidates = key_to_candidates.get(key)
    else:
        raise ValueError(f"Unsupported image obs type '{obs_type}' for key '{key}'")

    if candidates is None:
        raise ValueError(f"Unsupported {obs_type} key in shape_meta: {key}")

    for source_path in candidates:
        if source_path in h5_file:
            return source_path

    raise ValueError(
        f"Missing source dataset for '{key}'. Tried: {candidates}"
    )


def _episode_to_arrays(
    episode_path: Path,
    image_keys: List[str],
    lowdim_keys: List[str],
    image_shapes: Dict[str, Tuple[int, int, int]],
    image_types: Dict[str, str],
    action_dim: int,
    rotation_transformer: RotationTransformer,
) -> Dict[str, np.ndarray]:
    with h5py.File(str(episode_path), "r") as f:
        if "action" not in f:
            raise ValueError(
                f"Episode {episode_path} is missing root 'action' dataset. "
                "Regenerate the backend export so commanded actions are available."
            )
        commanded_action = f["action"][:]  # (T, 8), pos + quat(wxyz) + gripper
        ee = f["embodiment/ee"][:]  # (T, 7), observed pose
        joint = f["embodiment/joint"][:]  # (T, 9), observed gripper state
        if commanded_action.shape[0] < 1:
            raise ValueError(f"Episode {episode_path} has no frames")
        if ee.shape[0] != commanded_action.shape[0] or joint.shape[0] != commanded_action.shape[0]:
            raise ValueError(
                f"Temporal length mismatch in {episode_path}: "
                f"action={commanded_action.shape[0]}, ee={ee.shape[0]}, joint={joint.shape[0]}"
            )

        steps = commanded_action.shape[0]

        action_pos = commanded_action[:, :3].astype(np.float32)
        action_quat_wxyz = commanded_action[:, 3:7].astype(np.float32)
        action_rot6d = rotation_transformer.forward(action_quat_wxyz).astype(np.float32)
        action_gripper = commanded_action[:, 7:8].astype(np.float32)
        action = np.concatenate([action_pos, action_rot6d, action_gripper], axis=-1).astype(np.float32)
        if action.shape[-1] != action_dim:
            raise ValueError(
                f"Action dim mismatch. Expected {action_dim}, got {action.shape[-1]} "
                f"for episode {episode_path}"
            )

        arrays = {"action": action}
        for key in lowdim_keys:
            if key == "robot0_eef_pos":
                arrays[key] = ee[:, :3].astype(np.float32)
            elif key == "robot0_eef_quat":
                arrays[key] = ee[:, 3:7].astype(np.float32)
            elif key == "robot0_gripper_qpos":
                arrays[key] = canonicalize_gripper_qpos(joint[:, 7:9]).astype(np.float32) # normailzed
            else:
                raise ValueError(f"Unsupported lowdim key in shape_meta: {key}")

        decoded_cache = dict()
        for key in image_keys:
            c, h, w = image_shapes[key]
            if c != 3:
                raise ValueError(f"Only RGB(3 channels) is supported, got {c} for {key}")
            source_path = _resolve_h5_source_path(f, key, image_types[key])
            center_crop_square = (
                key == "robot0_eye_in_hand_image"
                and source_path == "observation/wrist/rgb"
            )
            cache_key = (source_path, h, w, center_crop_square)

            if cache_key not in decoded_cache:
                decoded_cache[cache_key] = _decode_jpeg_sequence(
                    f[source_path],
                    target_hw=(h, w),
                    center_crop_square=center_crop_square,
                )
            arrays[key] = decoded_cache[cache_key]

        for key, value in arrays.items():
            if value.shape[0] != steps:
                raise ValueError(
                    f"Temporal length mismatch for {key} in {episode_path}: "
                    f"{value.shape[0]} vs {steps}"
                )

    return arrays


class UniVTACReplayImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_root: str,
        task_name: str,
        split: str = "clean",
        n_demo: int = 100,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps=None,
        abs_action: bool = True,
        use_legacy_normalizer: bool = False,
        normalization_mode: str = "default",
        use_cache: bool = True,
        seed: int = 42,
        val_ratio: float = 0.0,
        image_identity_normalizer: bool = False,
        cache_dir: str = None,
        **kwargs,
    ):
        if not isinstance(shape_meta, dict):
            shape_meta = OmegaConf.to_container(shape_meta, resolve=True)

        self.n_demo = n_demo
        self.abs_action = abs_action
        self.shape_meta = shape_meta
        self.task_name = task_name
        self.split = split
        self.dataset_root = dataset_root
        mode = str(normalization_mode).strip().lower()
        if mode in {"", "none"}:
            mode = "legacy" if use_legacy_normalizer else "default"
        mode_aliases = {
            "default": "default",
            "on": "default",
            "legacy": "legacy",
            "off": "off",
            "identity": "off",
        }
        if mode not in mode_aliases:
            raise ValueError(
                f"Unsupported normalization_mode={normalization_mode!r}. "
                "Expected one of: default/on, legacy, off/identity."
            )
        self.normalization_mode = mode_aliases[mode]
        self.use_legacy_normalizer = self.normalization_mode == "legacy"
        self.image_identity_normalizer = image_identity_normalizer or self.normalization_mode == "off"

        dataset_root_path = Path(dataset_root).expanduser()
        flat_dataset_dir = dataset_root_path.joinpath(task_name, "hdf5")
        if flat_dataset_dir.is_dir():
            dataset_dir = flat_dataset_dir
        else:
            raise FileNotFoundError(
                f"Dataset directory not found. Tried flat={flat_dataset_dir}"
            )
        episode_files = _sorted_episode_files(dataset_dir)
        if len(episode_files) == 0:
            raise RuntimeError(f"No episode files found under {dataset_dir}")
        if n_demo > len(episode_files):
            raise ValueError(
                f"Requested n_demo={n_demo}, but only found {len(episode_files)} episodes "
                f"under {dataset_dir}"
            )
        episode_files = episode_files[:n_demo]
        print(
            f"[UniVTACDataset] task={task_name} split={split} episodes={len(episode_files)}"
        )

        image_keys = list()
        rgb_keys = list()
        tactile_rgb_keys = list()
        lowdim_keys = list()
        image_shapes = dict()
        image_types = dict()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type in IMAGE_OBS_TYPES:
                image_keys.append(key)
                image_shapes[key] = tuple(attr["shape"])
                image_types[key] = obs_type
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "tactile_rgb":
                tactile_rgb_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)
            else:
                raise ValueError(f"Unsupported obs type '{obs_type}' for key '{key}'")

        action_shape = tuple(shape_meta["action"]["shape"])
        if len(action_shape) != 1:
            raise ValueError(f"Expected 1D action shape, got {action_shape}")
        action_dim = action_shape[0]

        rotation_transformer = RotationTransformer(
            from_rep="quaternion", to_rep="rotation_6d"
        )

        replay_buffer = None
        if use_cache:
            cache_split = split if split else "default"
            if cache_dir is None:
                cache_dir = str(dataset_root_path.joinpath(task_name, ".cache"))
            cache_dir = Path(cache_dir).expanduser()
            cache_dir.mkdir(parents=True, exist_ok=True)
            manifest_sig = _dataset_manifest_signature(dataset_dir, episode_files, split)
            cache_name = (
                f"univtac_commanded_action_v1_{task_name}_{cache_split}_demo{n_demo}_"
                f"{_shape_meta_signature(shape_meta)}_{manifest_sig}.zarr.zip"
            )
            cache_path = cache_dir.joinpath(cache_name)
            lock_path = str(cache_path) + ".lock"

            with FileLock(lock_path):
                if not cache_path.exists():
                    print(f"[UniVTACDataset] Building cache: {cache_path}")
                    replay_buffer = ReplayBuffer.create_empty_numpy()
                    for episode_path in tqdm(
                        episode_files, desc="[UniVTACDataset] decode", leave=False
                    ):
                        arrays = _episode_to_arrays(
                            episode_path=episode_path,
                            image_keys=image_keys,
                            lowdim_keys=lowdim_keys,
                            image_shapes=image_shapes,
                            image_types=image_types,
                            action_dim=action_dim,
                            rotation_transformer=rotation_transformer,
                        )
                        replay_buffer.add_episode(arrays)

                    with zarr.ZipStore(str(cache_path), mode="w") as zip_store:
                        replay_buffer.save_to_store(zip_store)
                    print("[UniVTACDataset] Cache build complete")
                else:
                    print(f"[UniVTACDataset] Loading cache: {cache_path}")
                    with zarr.ZipStore(str(cache_path), mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("[UniVTACDataset] Cache loaded")
        else:
            print("[UniVTACDataset] Cache disabled, decoding episodes directly")
            replay_buffer = ReplayBuffer.create_empty_numpy()
            for episode_path in tqdm(
                episode_files, desc="[UniVTACDataset] decode", leave=False
            ):
                arrays = _episode_to_arrays(
                    episode_path=episode_path,
                    image_keys=image_keys,
                    lowdim_keys=lowdim_keys,
                    image_shapes=image_shapes,
                    image_types=image_types,
                    action_dim=action_dim,
                    rotation_transformer=rotation_transformer,
                )
                replay_buffer.add_episode(arrays)

        key_first_k = dict()
        if n_obs_steps is not None:
            print(
                f"[UniVTACDataset] Using first {n_obs_steps} obs steps for "
                f"image_keys={image_keys}, lowdim_keys={lowdim_keys}"
            )
            for key in image_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.image_keys = image_keys
        self.rgb_keys = rgb_keys
        self.tactile_rgb_keys = tactile_rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.ws_center = compute_ws_center(self.replay_buffer["robot0_eef_pos"])
        print(f"[UniVTACDataset] auto ws_center={self.ws_center.tolist()}")

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        stat = array_to_stats(self.replay_buffer["action"])
        if self.abs_action:
            if self.normalization_mode == "off":
                action_normalizer = get_identity_normalizer_from_stat(stat)
            elif self.use_legacy_normalizer:
                action_normalizer = normalizer_from_stat(stat)
            else:
                action_normalizer = abs_action_only_normalizer_from_stat(stat)
        else:
            action_normalizer = get_identity_normalizer_from_stat(stat)
        normalizer["action"] = action_normalizer

        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            if self.normalization_mode == "off":
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("pos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("quat"):
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("qpos"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"Unsupported lowdim key '{key}'")
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            if self.normalization_mode == "off" or self.image_identity_normalizer:
                normalizer[key] = get_image_identity_normalizer()
            else:
                normalizer[key] = get_image_range_normalizer()
        for key in self.tactile_rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not hasattr(self, "_threadpool_limited"):
            threadpool_limits(1)
            self._threadpool_limited = True
        data = self.sampler.sample_sequence(idx)
        t_slice = slice(self.n_obs_steps)

        obs_dict = dict()
        for key in self.image_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][t_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]
        for key in self.lowdim_keys:
            obs_dict[key] = data[key][t_slice].astype(np.float32)
            del data[key]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
        return torch_data


def normalizer_from_stat(stat):
    max_abs = np.maximum(stat["max"].max(), np.abs(stat["min"]).max())
    scale = np.full_like(stat["max"], fill_value=1 / max_abs)
    offset = np.zeros_like(stat["max"])
    return SingleFieldLinearNormalizer.create_manual(
        scale=scale, offset=offset, input_stats_dict=stat
    )
