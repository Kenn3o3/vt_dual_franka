from __future__ import annotations

import math

import torch
import torch.nn as nn


class SphericalCrossAttention(nn.Module):
    """Cross-attention from visual sphere tokens to one or more tactile sources."""

    def __init__(
        self,
        visual_dim: int,
        tactile_dim: int,
        out_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        num_tactile_sources: int = 1,
        use_source_embeddings: bool = True,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_tactile_sources = int(num_tactile_sources)
        if self.num_tactile_sources < 1:
            raise ValueError("num_tactile_sources must be >= 1")
        inner_dim = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(visual_dim, inner_dim)
        self.k_proj = nn.Linear(tactile_dim, inner_dim)
        self.v_proj = nn.Linear(tactile_dim, inner_dim)
        self.out_proj = nn.Linear(inner_dim, out_dim)
        self.visual_residual = nn.Identity() if visual_dim == out_dim else nn.Linear(visual_dim, out_dim)
        self.fuse = nn.Sequential(
            nn.Linear(visual_dim + out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

        if use_source_embeddings and self.num_tactile_sources > 1:
            self.source_embedding = nn.Parameter(
                torch.zeros(self.num_tactile_sources, tactile_dim)
            )
            nn.init.normal_(self.source_embedding, mean=0.0, std=0.02)
        else:
            self.register_parameter("source_embedding", None)

    def _prepare_tactile_tokens(
        self,
        visual_tokens: torch.Tensor,
        tactile_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, int, int]:
        if visual_tokens.ndim != 3:
            raise ValueError(
                f"Expected visual_tokens [N,P,C], got {tuple(visual_tokens.shape)}"
            )
        n, p, _ = visual_tokens.shape

        if tactile_tokens.ndim == 3:
            if tactile_tokens.shape[:2] != visual_tokens.shape[:2]:
                raise ValueError(
                    "Visual/tactile token grids differ: "
                    f"{tuple(visual_tokens.shape)} vs {tuple(tactile_tokens.shape)}"
                )
            return tactile_tokens, 1, p

        if tactile_tokens.ndim == 4:
            n2, s, p2, c = tactile_tokens.shape
            if n2 != n or p2 != p:
                raise ValueError(
                    "Visual/tactile token grids differ: "
                    f"{tuple(visual_tokens.shape)} vs {tuple(tactile_tokens.shape)}"
                )
            if self.source_embedding is not None:
                if s > self.source_embedding.shape[0]:
                    raise ValueError(
                        f"Got {s} tactile sources, but only "
                        f"{self.source_embedding.shape[0]} source embeddings exist"
                    )
                tactile_tokens = tactile_tokens + self.source_embedding[:s].view(
                    1, s, 1, c
                ).to(dtype=tactile_tokens.dtype)
            return tactile_tokens.reshape(n, s * p2, c), s, p2

        raise ValueError(
            "Expected tactile_tokens [N,P,C] or [N,S,P,C], "
            f"got {tuple(tactile_tokens.shape)}"
        )

    @staticmethod
    def _prepare_mask(
        tactile_mask: torch.Tensor | None,
        n: int,
        s: int,
        p: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if tactile_mask is None:
            return None
        mask = tactile_mask.to(device=device, dtype=torch.bool)
        if mask.ndim == 2:
            if mask.shape == (n, s):
                return mask[:, :, None].expand(n, s, p).reshape(n, s * p)
            if mask.shape == (n, s * p):
                return mask
        elif mask.ndim == 3 and mask.shape == (n, s, p):
            return mask.reshape(n, s * p)
        raise ValueError(
            f"Bad tactile_mask shape {tuple(mask.shape)}; expected "
            f"[N,S], [N,S,P], or [N,S*P]"
        )

    def forward(
        self,
        visual_tokens: torch.Tensor,
        tactile_tokens: torch.Tensor,
        tactile_mask: torch.Tensor | None = None,
    ):
        n, p, _ = visual_tokens.shape
        h = self.num_heads
        d = self.head_dim
        tactile_tokens, source_count, tactile_p = self._prepare_tactile_tokens(
            visual_tokens, tactile_tokens
        )
        pt = tactile_tokens.shape[1]
        q = self.q_proj(visual_tokens).view(n, p, h, d).transpose(1, 2)
        k = self.k_proj(tactile_tokens).view(n, pt, h, d).transpose(1, 2)
        v = self.v_proj(tactile_tokens).view(n, pt, h, d).transpose(1, 2)
        scores = torch.einsum("nhid,nhjd->nhij", q, k) / math.sqrt(d)

        flat_mask = self._prepare_mask(
            tactile_mask,
            n=n,
            s=source_count,
            p=tactile_p,
            device=scores.device,
        )
        if flat_mask is not None:
            if not torch.all(flat_mask.any(dim=-1)):
                raise ValueError("Every batch item needs at least one tactile source.")
            scores = scores.masked_fill(
                ~flat_mask[:, None, None, :],
                torch.finfo(scores.dtype).min,
            )

        attn = self.dropout(torch.softmax(scores, dim=-1))
        msg = torch.einsum("nhij,nhjd->nhid", attn, v)
        msg = self.out_proj(msg.transpose(1, 2).reshape(n, p, h * d))
        delta = self.fuse(torch.cat([visual_tokens, msg], dim=-1))
        fused = self.norm(self.visual_residual(visual_tokens) + delta)
        if source_count > 1:
            attn = attn.view(n, h, p, source_count, tactile_p)
        return fused, attn
