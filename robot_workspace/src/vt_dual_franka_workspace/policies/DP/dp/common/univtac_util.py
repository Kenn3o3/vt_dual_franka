from __future__ import annotations

from typing import Union

import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]


def gripper_scalar_from_qpos(gripper_qpos: ArrayLike) -> ArrayLike:
    if isinstance(gripper_qpos, torch.Tensor):
        return gripper_qpos.mean(dim=-1, keepdim=True)
    arr = np.asarray(gripper_qpos)
    return arr.mean(axis=-1, keepdims=True)


def canonicalize_gripper_qpos(gripper_qpos: ArrayLike) -> ArrayLike:
    scalar = gripper_scalar_from_qpos(gripper_qpos)
    if isinstance(scalar, torch.Tensor):
        return scalar.repeat_interleave(2, dim=-1)
    return np.repeat(scalar, repeats=2, axis=-1)


def compute_ws_center(eef_pos: np.ndarray) -> np.ndarray:
    pos_min = np.min(eef_pos, axis=0)
    print("pos_min: ", pos_min)
    pos_max = np.max(eef_pos, axis=0)
    print("pos_max: ", pos_max)
    return ((pos_min + pos_max) / 2.0).astype(np.float32)
