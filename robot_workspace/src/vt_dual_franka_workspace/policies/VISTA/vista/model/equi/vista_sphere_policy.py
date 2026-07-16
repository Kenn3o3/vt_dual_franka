from __future__ import annotations

import math

import e3nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from vista.model.equi.i2s_policy import ImageEncoder
from vista.model.equi.spherical_attention import SphericalCrossAttention
from vista.model.equi.vista_image_to_sphere import VistaImageToSphere
from vista.model.equi.vista_tactile_to_sphere import VistaTactileToSphere
from vista.model.equi.vista_util import S2Conv, SO3Conv, s2_healpix_grid, so3_near_identity_grid


class VistaSpherePolicy(nn.Module):
    def __init__(
        self,
        encoder: str = "equiresnet50",
        lmax: int = 6,
        visual_dim: int = 1024,
        tactile_dim: int = 1024,
        fused_dim: int = 1024,
        so3_dim: int = 64,
        N: int = 8,
        initialize: bool = True,
        f_out: int = 128,
        tactile_shape=(3, 84, 84),
        rec_level: int = 3,
        max_beta: float = math.pi / 2,
        attention_heads: int = 8,
        attention_head_dim: int = 64,
        tactile_mode: str = "raw",
    ):
        super().__init__()
        self.lmax = int(lmax)
        if str(tactile_mode) != "raw":
            raise ValueError(
                "VISTA now supports only tactile_mode='raw' "
                "(tactile RGB image with markers)."
            )

        self.image_encoder = ImageEncoder(
            encoder=encoder,
            out_fdim=visual_dim,
            out_shape=(7, 7),
            N=N,
            initialize=initialize,
        )
        self.image_to_sphere = VistaImageToSphere(
            fmap_shape=self.image_encoder.output_shape,
            lmax=lmax,
            rec_level=rec_level,
            max_beta=max_beta,
        )
        self.tactile_to_sphere = VistaTactileToSphere(
            tactile_shape=tactile_shape,
            encoder=encoder,
            feature_dim=visual_dim,
            token_dim=tactile_dim,
            sphere_dirs=self.image_to_sphere.sphere_dirs,
            harmonic_Y=self.image_to_sphere.Y,
            harmonic_omega=self.image_to_sphere.omega,
            N=N,
            initialize=initialize,
        )
        self.cross_attention = SphericalCrossAttention(
            visual_dim=visual_dim,
            tactile_dim=tactile_dim,
            out_dim=fused_dim,
            num_heads=attention_heads,
            head_dim=attention_head_dim,
            num_tactile_sources=1,
            use_source_embeddings=True,
        )
        s2_kernel_grid = s2_healpix_grid(max_beta=math.inf, rec_level=1)
        so3_kernel_grid = so3_near_identity_grid()
        self.irreps = o3.Irreps([(1, (l, 1)) for l in range(lmax + 1)])
        self.s2_conv = S2Conv(fused_dim, so3_dim, lmax, s2_kernel_grid)
        self.s2_act = e3nn.nn.SO3Activation(lmax, lmax, act=torch.relu_, resolution=8)
        self.so3_conv_1 = SO3Conv(so3_dim, 64, lmax, so3_kernel_grid, lmax)
        self.so3_act_1 = e3nn.nn.SO3Activation(lmax, 6, act=torch.relu_, resolution=8)
        self.so3_conv_2 = SO3Conv(64, f_out, 6, so3_kernel_grid, 6)
        self.last_debug = None

    @staticmethod
    def _as_tensor(fmap):
        return fmap if isinstance(fmap, torch.Tensor) else fmap.tensor

    def forward(
        self,
        image: torch.Tensor,
        left_tactile: torch.Tensor,
        eef_quat: torch.Tensor,
    ) -> torch.Tensor:
        fmap = self._as_tensor(self.image_encoder(image))
        if fmap.shape[-2:] != (7, 7):
            fmap = F.adaptive_avg_pool2d(fmap, (7, 7))
        visual_tokens = self.image_to_sphere.project_tokens(fmap)

        tactile_tokens = self.tactile_to_sphere(left_tactile)
        tactile_tokens_stacked = tactile_tokens.unsqueeze(1)
        fused_tokens, attention = self.cross_attention(
            visual_tokens,
            tactile_tokens_stacked,
        )
        coeff = self.image_to_sphere.harmonic_project(fused_tokens)
        coeff = torch.einsum("nij,ncj->nci", self.irreps.D_from_quaternion(eef_quat), coeff)
        x = self.s2_conv(coeff)
        x = self.s2_act(x)
        x = self.so3_conv_1(x)
        x = self.so3_act_1(x)
        out = self.so3_conv_2(x)
        self.last_debug = {
            "visual_tokens": visual_tokens.detach(),
            "tactile_tokens": tactile_tokens.detach(),
            "tactile_tokens_stacked": tactile_tokens_stacked.detach(),
            "fused_tokens": fused_tokens.detach(),
            "attention": attention.detach(),
        }
        return out
