from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn as torch_nn
from torchvision import models as vision_models
import torchvision
from escnn import gspaces, nn
from escnn.group import CyclicGroup
from einops import rearrange

from dp.model.common.module_attr_mixin import ModuleAttrMixin
from dp.common.pytorch_util import replace_submodules
from dp.model.vision.spatial_softmax import SpatialSoftmax
from dp.model.equi.manifeel_equi_encoder import EquivariantResEncoder230Cyclic


class Identity(torch_nn.Module):
    def forward(self, x):
        return x


def quaternion_wxyz_to_sixd(quat_wxyz: torch.Tensor) -> torch.Tensor:
    quat_wxyz = F.normalize(quat_wxyz, p=2, dim=-1)
    w, x, y, z = quat_wxyz.unbind(-1)
    two_s = 2.0 / torch.clamp((quat_wxyz * quat_wxyz).sum(-1), min=1e-12)
    matrix = torch.stack(
        (
            1 - two_s * (y * y + z * z),
            two_s * (x * y - z * w),
            two_s * (x * z + y * w),
            two_s * (x * y + z * w),
            1 - two_s * (x * x + z * z),
            two_s * (y * z - x * w),
            two_s * (x * z - y * w),
            two_s * (y * z + x * w),
            1 - two_s * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(quat_wxyz.shape[:-1] + (3, 3))
    return matrix[..., :2, :].reshape(quat_wxyz.shape[:-1] + (6,))


class TacRGBEncoder(torch_nn.Module):
    def __init__(self, out_size: int):
        super().__init__()
        net = vision_models.resnet18(weights=None)
        net = replace_submodules(
            root_module=net,
            predicate=lambda x: isinstance(x, torch_nn.BatchNorm2d),
            func=lambda x: torch_nn.GroupNorm(
                num_groups=max(1, x.num_features // 16),
                num_channels=x.num_features,
            ),
        )
        self.resnet = torch_nn.Sequential(*(list(net.children())[:-2]))
        self.spatial_softmax = SpatialSoftmax([512, 3, 3], num_kp=out_size // 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        batch_size = image.shape[0]
        return self.spatial_softmax(self.resnet(image)).reshape(batch_size, -1)


class EquivariantObsEnc(ModuleAttrMixin):
    """ManiFeel EquiDiff observation encoder adapted to UniVTAC keys.

    The source ManiFeel encoder assumes keys such as ``wrist`` and ``state``.
    This local version consumes the clean UniVTAC dataset keys directly:
    wrist RGB, optional left/right tactile RGB, and proprioception split into
    eef position, eef quaternion in wxyz order, and gripper qpos.
    """

    def __init__(
        self,
        obs_shape_meta,
        obs_shape=(3, 230, 230),
        crop_shape=(224, 224),
        n_hidden=128,
        N=8,
        initialize=True,
    ):
        super().__init__()
        obs_channel = obs_shape[0]
        self.n_hidden = n_hidden
        self.N = N
        self.group = gspaces.no_base_space(CyclicGroup(self.N))
        self.token_type = nn.FieldType(self.group, self.n_hidden * [self.group.regular_repr])

        self.rgb_keys = [
            key
            for key, value in obs_shape_meta.items()
            if value.get("type", "low_dim") in {"rgb", "tactile_rgb"}
        ]
        self.lowdim_keys = [
            key
            for key, value in obs_shape_meta.items()
            if value.get("type", "low_dim") == "low_dim"
        ]
        self.wrist_keys = [key for key in self.rgb_keys if key == "robot0_eye_in_hand_image"]
        self.tactile_keys = [key for key in self.rgb_keys if key.startswith("robot0_tactile_")]

        input_type_list = []
        if self.wrist_keys:
            self.enc_obs = EquivariantResEncoder230Cyclic(
                obs_channel,
                self.n_hidden,
                initialize,
                N=self.N,
            )
            input_type_list += self.n_hidden * [self.group.regular_repr]

        self.tactile_encoders = torch_nn.ModuleDict()
        for key in self.tactile_keys:
            self.tactile_encoders[key] = TacRGBEncoder(self.n_hidden)
            input_type_list += self.n_hidden * [self.group.trivial_repr]

        input_type_list += 4 * [self.group.irrep(1)]
        input_type_list += 2 * [self.group.trivial_repr]

        self.enc_out = nn.Linear(nn.FieldType(self.group, input_type_list), self.token_type)
        self.crop_randomizer = torchvision.transforms.CenterCrop(tuple(crop_shape))
        self.normalizer = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        self.resize_shape = tuple(obs_shape)

    def _encode_wrist(self, image: torch.Tensor) -> torch.Tensor:
        batch_time = image.shape[0]
        image = F.interpolate(
            image,
            size=(self.resize_shape[1], self.resize_shape[2]),
            mode="bilinear",
            align_corners=False,
        )
        image = self.crop_randomizer(image)
        image = self.normalizer(image)
        return self.enc_obs(image).tensor.reshape(batch_time, -1)

    def _encode_tactile(self, key: str, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, size=(84, 84), mode="bilinear", align_corners=False)
        return self.tactile_encoders[key](image)

    def forward(self, nobs):
        if "robot0_eef_pos" not in nobs or "robot0_eef_quat" not in nobs:
            raise KeyError("EquiDiff tactile baseline requires robot0_eef_pos and robot0_eef_quat.")

        ee_pos = nobs["robot0_eef_pos"]
        ee_quat = nobs["robot0_eef_quat"]
        batch_size, horizon = ee_pos.shape[:2]
        flat_pos = rearrange(ee_pos, "b t d -> (b t) d")
        flat_quat = rearrange(ee_quat, "b t d -> (b t) d")
        ee_rot = quaternion_wxyz_to_sixd(flat_quat)

        features = []
        for key in self.wrist_keys:
            image = rearrange(nobs[key], "b t c h w -> (b t) c h w")
            features.append(self._encode_wrist(image))
        for key in self.tactile_keys:
            image = rearrange(nobs[key], "b t c h w -> (b t) c h w")
            features.append(self._encode_tactile(key, image))

        pos_xy = flat_pos[:, 0:2]
        pos_z = flat_pos[:, 2:3]
        gripper = nobs.get("robot0_gripper_qpos")
        if gripper is None:
            gripper_feature = torch.zeros_like(pos_z)
        else:
            gripper_feature = rearrange(gripper, "b t d -> (b t) d").mean(dim=-1, keepdim=True)

        features.extend(
            [
                pos_xy,
                ee_rot[:, 0:1],
                ee_rot[:, 3:4],
                ee_rot[:, 1:2],
                ee_rot[:, 4:5],
                ee_rot[:, 2:3],
                ee_rot[:, 5:6],
                pos_z,
                gripper_feature,
            ]
        )

        packed = torch.cat(features, dim=1)
        packed = nn.GeometricTensor(packed, self.enc_out.in_type)
        out = self.enc_out(packed).tensor
        return rearrange(out, "(b t) d -> b t d", b=batch_size, t=horizon)
