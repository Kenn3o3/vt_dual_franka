import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from e3nn import o3

from vista.model.equi.vista_util import s2_healpix_grid


class VistaImageToSphere(nn.Module):
    """Lift a ResNet-style feature map to ordered sphere tokens and harmonics."""

    def __init__(
        self,
        fmap_shape,
        lmax: int,
        coverage: float = 1.0,
        sigma: float = 0.2,
        max_beta: float = math.pi / 2,
        taper_beta: float = math.pi / 2,
        rec_level: int = 3,
        harmonic_norm: str = "sqrt_n",
    ):
        super().__init__()
        self.lmax = int(lmax)
        _, h, w = tuple(fmap_shape)
        kernel_grid = s2_healpix_grid(rec_level=rec_level, max_beta=max_beta)
        alpha, beta = kernel_grid
        sphere_dirs = o3.angles_to_xyz(alpha, beta).float()
        self.register_buffer("sphere_dirs", F.normalize(sphere_dirs, dim=-1))

        max_radius = torch.linalg.norm(sphere_dirs[:, [0, 2]], dim=1).max().clamp_min(1e-6)
        sample_x = coverage * sphere_dirs[:, 2] / max_radius
        sample_y = coverage * sphere_dirs[:, 0] / max_radius

        gridx, gridy = torch.meshgrid(
            torch.linspace(-1, 1, h),
            torch.linspace(-1, 1, w),
            indexing="ij",
        )
        weight = torch.exp(
            -(
                (gridx.unsqueeze(-1) - sample_x).pow(2)
                + (gridy.unsqueeze(-1) - sample_y).pow(2)
            )
            / (2.0 * sigma**2)
        )
        weight = weight / weight.sum((0, 1), keepdim=True).clamp_min(1e-6)

        if taper_beta < max_beta:
            mask = ((beta - max_beta) / (taper_beta - max_beta)).clamp(max=1)
            weight = weight * mask.view(1, 1, -1)
        self.weight = nn.Parameter(weight.float(), requires_grad=True)

        Y = o3.spherical_harmonics_alpha_beta(
            range(self.lmax + 1),
            alpha,
            beta,
            normalization="component",
        ).float()
        self.register_buffer("Y", Y)

        p = Y.shape[0]
        if harmonic_norm == "sqrt_n":
            omega = torch.ones(p) / math.sqrt(p)
        elif harmonic_norm == "area":
            omega = torch.ones(p) * (4.0 * math.pi / p)
        else:
            raise ValueError(f"Unsupported harmonic_norm={harmonic_norm}")
        self.register_buffer("omega", omega.float())

    def project_tokens(self, fmap: torch.Tensor) -> torch.Tensor:
        if not isinstance(fmap, torch.Tensor):
            fmap = fmap.tensor
        if fmap.ndim != 4:
            raise ValueError(f"Expected feature map [N,C,H,W], got {tuple(fmap.shape)}")
        tokens = torch.einsum("nchw,hwp->npc", fmap, self.weight.to(fmap.dtype))
        return F.relu(tokens)

    def harmonic_project(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens [N,P,C], got {tuple(tokens.shape)}")
        if tokens.shape[1] != self.Y.shape[0]:
            raise ValueError(f"Sphere token count mismatch: {tuple(tokens.shape)} vs {tuple(self.Y.shape)}")
        return torch.einsum(
            "npc,pk,p->nck",
            tokens,
            self.Y.to(tokens.dtype),
            self.omega.to(tokens.dtype),
        )

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return self.harmonic_project(self.project_tokens(fmap))
