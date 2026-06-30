from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from vista.model.equi.i2s_policy import ImageEncoder


class VistaTactileToSphere(nn.Module):
    """Left tactile branch with its own 2D-to-sphere projection.

    The tactile and visual branches share the ordered sphere grid / harmonic
    basis, but not the spatial projection weights. This lets tactile pixels
    learn a contact-local interpretation while still producing tokens indexed
    by the same directions as the visual branch.
    """

    def __init__(
        self,
        tactile_shape=(3, 84, 84),
        encoder: str = "equiresnet50",
        feature_dim: int = 1024,
        token_dim: int = 1024,
        sphere_dirs: torch.Tensor | None = None,
        harmonic_Y: torch.Tensor | None = None,
        harmonic_omega: torch.Tensor | None = None,
        N: int = 8,
        initialize: bool = True,
    ):
        super().__init__()
        if sphere_dirs is None:
            raise ValueError("sphere_dirs must be shared with the visual branch")
        self.encoder = ImageEncoder(
            encoder=encoder,
            out_fdim=feature_dim,
            out_shape=(7, 7),
            N=N,
            obs_channel=int(tactile_shape[0]),
            initialize=initialize,
        )
        self.token_proj = nn.Linear(feature_dim, token_dim) if feature_dim != token_dim else nn.Identity()
        p = int(sphere_dirs.shape[0])
        fmap_h, fmap_w = self.encoder.output_shape[-2:]
        self.proj_logits = nn.Parameter(torch.empty(fmap_h, fmap_w, p))
        nn.init.normal_(self.proj_logits, mean=0.0, std=0.02)
        self.register_buffer("sphere_dirs", F.normalize(sphere_dirs.float(), dim=-1))
        if harmonic_Y is not None:
            self.register_buffer("Y", harmonic_Y.float())
        else:
            self.Y = None
        if harmonic_omega is not None:
            self.register_buffer("omega", harmonic_omega.float())
        else:
            self.omega = None

    @staticmethod
    def _as_tensor(fmap):
        return fmap if isinstance(fmap, torch.Tensor) else fmap.tensor

    def forward(self, tactile_image: torch.Tensor) -> torch.Tensor:
        fmap = self._as_tensor(self.encoder(tactile_image))
        if fmap.shape[-2:] != (7, 7):
            fmap = F.adaptive_avg_pool2d(fmap, (7, 7))
        weight = torch.softmax(self.proj_logits.reshape(-1, self.proj_logits.shape[-1]), dim=0)
        weight = weight.reshape_as(self.proj_logits).to(dtype=fmap.dtype)
        tokens = torch.einsum("nchw,hwp->npc", fmap, weight)
        tokens = F.relu(tokens)
        return self.token_proj(tokens)
