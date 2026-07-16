from __future__ import annotations

from pathlib import Path
from typing import Dict
import sys

import torch
from torch import nn
import torchvision

from dp.model.common.module_attr_mixin import ModuleAttrMixin

POLICIES_ROOT = Path(__file__).resolve().parents[4]
if str(POLICIES_ROOT) not in sys.path:
    sys.path.insert(0, str(POLICIES_ROOT))

from ViTAL.clip_pretraining import modified_resnet18


class ViTALObsEncoder(ModuleAttrMixin):
    """ViTAL-style wrist/tactile encoder for Diffusion Policy.

    The encoder loads the CLIP-pretrained ViTAL vision and tactile ResNet18
    backbones, pools their feature maps, concatenates them with proprioception,
    and leaves the diffusion UNet unchanged. This makes `ViTAL/dp` a true
    ViTAL-encoder baseline instead of a plain DP alias.
    """

    def __init__(
        self,
        shape_meta: dict,
        vision_backbone_path: str,
        gelsight_backbone_path: str,
        freeze_backbones: bool = True,
        vision_imagenet_norm: bool = True,
        tactile_imagenet_norm: bool = False,
    ):
        super().__init__()
        self.shape_meta = shape_meta
        obs_meta = shape_meta["obs"]
        self.wrist_key = "robot0_eye_in_hand_image"
        self.tactile_keys = [
            key
            for key, attr in obs_meta.items()
            if attr.get("type", "low_dim") == "tactile_rgb"
        ]
        self.low_dim_keys = sorted(
            key
            for key, attr in obs_meta.items()
            if attr.get("type", "low_dim") == "low_dim"
        )
        if self.wrist_key not in obs_meta:
            raise ValueError("ViTALObsEncoder requires robot0_eye_in_hand_image.")
        if len(self.tactile_keys) == 0:
            raise ValueError("ViTALObsEncoder requires at least one tactile_rgb key.")

        self.vision_encoder = self._load_backbone(vision_backbone_path)
        self.gelsight_encoder = self._load_backbone(gelsight_backbone_path)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.vision_norm = (
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if vision_imagenet_norm
            else nn.Identity()
        )
        self.tactile_norm = (
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            if tactile_imagenet_norm
            else nn.Identity()
        )
        if freeze_backbones:
            self.vision_encoder.requires_grad_(False)
            self.gelsight_encoder.requires_grad_(False)

        self._output_dim = 512 * (1 + len(self.tactile_keys))
        for key in self.low_dim_keys:
            self._output_dim += int(obs_meta[key]["shape"][0])

    @staticmethod
    def _resolve_path(path_value: str) -> Path:
        path = Path(path_value).expanduser()
        if not path.is_absolute():
            path = POLICIES_ROOT / path
        return path

    def _load_backbone(self, path_value: str) -> nn.Module:
        path = self._resolve_path(path_value)
        if not path.is_file():
            raise FileNotFoundError(f"ViTAL encoder checkpoint not found: {path}")
        model = modified_resnet18()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model

    def _encode_image(self, encoder: nn.Module, image: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        image = norm(image)
        return self.pool(encoder(image))

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        features = [self._encode_image(self.vision_encoder, obs_dict[self.wrist_key], self.vision_norm)]
        for key in self.tactile_keys:
            features.append(self._encode_image(self.gelsight_encoder, obs_dict[key], self.tactile_norm))
        for key in self.low_dim_keys:
            features.append(obs_dict[key])
        return torch.cat(features, dim=-1)

    @torch.no_grad()
    def output_shape(self):
        return (self._output_dim,)
