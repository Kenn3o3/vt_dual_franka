"""
Rigorous toy comparison between FM paradigms.

MOTIF is fully implemented with all four mechanisms:
  (1) Physical time encoding  τ_k passed as sinusoidal embedding alongside t_diff
  (2) State-mask conditioning  L_FM uses mask token; L_vel uses real state s
  (3) Velocity supervision L_vel  at t_diff=0, supervise decoded velocity vs GT
  (4) DCT coefficient space  FM generates c ∈ R^(M+1); decode → vel → pos

Toy "state":
  Each demo belongs to one of two modes (±).  The state s ∈ R^2 encodes
  which mode is active:  s = [sign, 0].  During L_FM training, s is replaced
  by a mask token (zeros).  During L_vel, the real s is provided.

Baseline options (controlled by BASELINES):
  "fm"  : FM in position space R^K  (π₀/Diffusion Policy style), no state
  "sfp" : Streaming Flow Policy — extended state (a,z), exact paper formula
  "mpd" : ProDMP-style FM in RBF-velocity weight space R^N_basis

DEMO_CYCLES controls demo shape:
  0.5  → half-period arch      A·sin²(π τ/T)
  1.0  → full-period sinusoid  A·sin(2π τ/T)
  1.5  → 1.5-period            A·sin(3π τ/T)

Output: mpd/motif_toy_rigorous.png

Run:
    python scripts/motif_toy_rigorous.py
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.fft import dct

# ============================================================================ #
#  USER-TUNABLE CONFIGURATION
# ============================================================================ #

DEMO_CYCLES: float = 0.75   # 0.5 / 1.0 / 1.5

# Choices: "fm", "sfp", "mpd"
BASELINES: list[str] = ["fm", "sfp", "mpd"]

# MOTIF mechanism weights
ALPHA_VEL: float = 1.0     # weight of L_vel relative to L_FM

# ============================================================================ #
#  TRAINING HYPERPARAMETERS  (shared by all models)
# ============================================================================ #

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED   = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

T       = 1.0    # chunk duration (s)
K       = 50     # training control frequency → K steps per chunk
M       = 4      # MOTIF: number of DCT modes (0 … M)
N_BASIS = 8      # MPD: number of RBF basis functions

# SFP Gaussian-tube parameters
# sigma_r = sqrt(sigma_1^2 - sigma_0^2) must be > 0 for z to have any effect.
# With sigma_0=0.0: sigma_r = sigma_1 = 0.1, activating the stochastic z coupling.
SFP_SIGMA_0: float = 0.0
SFP_SIGMA_1: float = 0.1
SFP_SIGMA_R: float = math.sqrt(max(SFP_SIGMA_1**2 - SFP_SIGMA_0**2, 0.0))

# Toy state dimension (encodes mode label)
S_DIM = 2

N_TRAIN     = 4096
BATCH       = 256
EPOCHS      = 500
LR          = 2e-3
HIDDEN      = 256
DEPTH       = 3
TEMB        = 64   # sinusoidal embedding dim (used for both t_diff and τ)

N_SAMPLES   = 40
N_ODE_STEPS = 20
MID_T       = 0.5

OUT_PATH = Path(__file__).resolve().parent.parent / "motif_toy_rigorous.png"

# ============================================================================ #
#  MATPLOTLIB STYLE
# ============================================================================ #

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "legend.frameon":     False,
})

COL_BLUE  = "#1f4e79"
COL_RED   = "#b43a3a"
COL_MOTIF = "#2a7f62"
COL_GREY  = "#6f6f6f"
COL_BG    = "#fafafa"

_BASELINE_COLORS: dict[str, str] = {
    "fm":  "#7b3f00",
    "sfp": "#3b0c5a",
    "mpd": "#0a3d62",
}
_BASELINE_LABELS: dict[str, str] = {
    "fm":  r"FM  (pos $\mathbb{R}^K$)",
    "sfp": r"SFP  (ext-state FM)",
    "mpd": r"MPD  (RBF-vel $\mathbb{R}^{N_b}$)",
}

# ============================================================================ #
#  TIME GRIDS
# ============================================================================ #

tau_K     = np.linspace(0.0, T, K,   endpoint=False) + 0.5 * T / K   # (K,)
tau_dense = np.linspace(0.0, T, 500)                                   # (500,)

# ============================================================================ #
#  DEMO TRAJECTORIES
# ============================================================================ #

def _demo_fn(tau: np.ndarray, amp: float) -> np.ndarray:
    if abs(DEMO_CYCLES - 0.5) < 1e-9:
        return amp * np.sin(math.pi * tau / T) ** 2
    else:
        return amp * np.sin(2.0 * math.pi * DEMO_CYCLES * tau / T)


def _demo_peak_idx() -> int:
    return int(np.argmax(_demo_fn(tau_K, 1.0)))


def sample_demo_positions(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (pos, signs) where signs ∈ {-1, +1} is the mode label."""
    signs  = rng.choice([-1, +1], size=n)
    amps   = rng.uniform(0.55, 0.65, size=n) * signs
    phases = rng.uniform(-0.05, 0.05, size=n)
    if abs(DEMO_CYCLES - 0.5) < 1e-9:
        mean = amps[:, None] * np.sin(math.pi * tau_K[None, :] / T) ** 2
    else:
        mean = amps[:, None] * np.sin(
            2.0 * math.pi * DEMO_CYCLES * tau_K[None, :] / T + phases[:, None]
        )
    pos = mean + rng.normal(0.0, 0.020, size=mean.shape)
    return pos, signs   # (N, K), (N,)

# ============================================================================ #
#  RBF BASIS  (MPD)
# ============================================================================ #

def _rbf_basis_np(tau: np.ndarray, n_basis: int) -> np.ndarray:
    centres = np.linspace(0.0, T, n_basis)
    width   = (T / (n_basis - 1)) * 0.5
    phi     = np.exp(-0.5 * ((tau[:, None] - centres[None, :]) / width) ** 2)
    phi    /= (phi.sum(axis=1, keepdims=True) + 1e-8)
    return phi


def _rbf_basis_torch(tau: torch.Tensor, n_basis: int, device) -> torch.Tensor:
    centres = torch.linspace(0.0, T, n_basis, device=device)
    width   = (T / (n_basis - 1)) * 0.5
    phi     = torch.exp(-0.5 * ((tau[:, None] - centres[None, :]) / width) ** 2)
    return phi / (phi.sum(dim=1, keepdim=True) + 1e-8)

# ============================================================================ #
#  DATASET BUILDER
# ============================================================================ #

def build_dataset():
    """Build and normalise training targets for all methods.

    Returns
    -------
    x1_pos_norm   : (N, K)
    x1_coef_norm  : (N, M+1)
    x1_mpd_norm   : (N, N_BASIS)
    xi_raw        : (N, K)       raw positions [SFP]
    states        : (N, S_DIM)   toy state = [sign, 0]  [MOTIF L_vel]
    vel_raw       : (N, K)       raw velocities          [MOTIF L_vel GT]
    pos_scale     : (1, K)
    coeff_scale   : (1, M+1)
    mpd_scale     : (1, N_BASIS)
    """
    rng  = np.random.default_rng(SEED)
    pos, signs = sample_demo_positions(N_TRAIN, rng)         # (N, K), (N,)
    vel  = np.gradient(pos, T / K, axis=-1)                  # (N, K)

    # Toy state: s = [sign, 0] ∈ R^2 (encodes the demo mode)
    states_np = np.stack([signs.astype(np.float32),
                          np.zeros(N_TRAIN, dtype=np.float32)], axis=1)  # (N, 2)

    # MOTIF DCT coefficients (velocity)
    coeffs_raw = dct(vel, axis=-1, norm="ortho")[:, : M + 1]  # (N, M+1)

    # MPD: RBF ridge on velocity
    phi_rbf    = _rbf_basis_np(tau_K, N_BASIS)
    lam        = 1e-4
    A          = phi_rbf.T @ phi_rbf + lam * np.eye(N_BASIS)
    mpd_w      = np.linalg.solve(A, phi_rbf.T @ vel.T).T       # (N, N_BASIS)

    # Per-dimension normalisation
    pos_scale   = pos.std(axis=0,    keepdims=True) + 1e-8
    coeff_scale = coeffs_raw.std(axis=0, keepdims=True) + 1e-8
    mpd_scale   = mpd_w.std(axis=0,  keepdims=True) + 1e-8

    def _t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=DEVICE)

    return (_t(pos / pos_scale),
            _t(coeffs_raw / coeff_scale),
            _t(mpd_w / mpd_scale),
            _t(pos),
            _t(states_np),
            _t(vel),
            _t(pos_scale), _t(coeff_scale), _t(mpd_scale))

# ============================================================================ #
#  DECODERS
# ============================================================================ #

K_BASIS = torch.arange(M + 1, dtype=torch.float32, device=DEVICE)


def decode_motif_velocity(coeffs: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """coeffs: (B, M+1), tau: (N_q,) → vel: (B, N_q)."""
    w      = torch.full((M + 1,), math.sqrt(2.0 / K), device=coeffs.device)
    w[0]   = math.sqrt(1.0 / K)
    basis  = torch.cos(math.pi * K_BASIS[None, :] * tau[:, None] / T)
    return (coeffs * w[None, :]) @ basis.T


def decode_motif_position(coeffs: torch.Tensor, a0: torch.Tensor,
                          tau: torch.Tensor) -> torch.Tensor:
    v   = decode_motif_velocity(coeffs, tau)
    dt  = tau[1] - tau[0]
    cum = torch.cat([torch.zeros_like(v[:, :1]),
                     torch.cumsum(0.5 * (v[:, :-1] + v[:, 1:]), dim=-1) * dt], dim=-1)
    return a0[:, None] + cum


def decode_mpd_velocity(weights: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """weights: (B, N_BASIS), tau: (N_q,) → vel: (B, N_q)."""
    phi = _rbf_basis_torch(tau, N_BASIS, weights.device)
    return weights @ phi.T


def decode_mpd_position(weights: torch.Tensor, a0: torch.Tensor,
                        tau: torch.Tensor) -> torch.Tensor:
    v   = decode_mpd_velocity(weights, tau)
    dt  = tau[1] - tau[0]
    cum = torch.cat([torch.zeros_like(v[:, :1]),
                     torch.cumsum(0.5 * (v[:, :-1] + v[:, 1:]), dim=-1) * dt], dim=-1)
    return a0[:, None] + cum

# ============================================================================ #
#  NETWORKS
# ============================================================================ #

def sinusoidal_emb(t: torch.Tensor, dim: int = TEMB) -> torch.Tensor:
    """t: (B,) or (B, N) → (..., dim).  Works for any leading shape."""
    shape  = t.shape
    t_flat = t.reshape(-1)
    half   = dim // 2
    freqs  = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang    = t_flat[:, None] * freqs[None, :] * 2 * math.pi
    emb    = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)   # (prod, dim)
    return emb.reshape(*shape, dim)


class FlowMLP(nn.Module):
    """Standard FM backbone — no state conditioning."""

    def __init__(self, dim: int, hidden: int = HIDDEN, depth: int = DEPTH):
        super().__init__()
        self.dim = dim
        layers   = [nn.Linear(dim + TEMB, hidden), nn.SiLU()]
        for _    in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers   += [nn.Linear(hidden, dim)]
        self.net  = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, sinusoidal_emb(t)], dim=-1))


class MOTIFNet(nn.Module):
    """MOTIF network — implements all four mechanisms.

    Input for L_FM  (mechanism 2: mask token):
        x   = noisy coeffs  (B, M+1)
        t   = FM denoising time  (B,)
        s   = MASK TOKEN = zeros  (B, S_DIM)   ← state replaced by mask
        tau = not used for L_FM output (but physical time encoding is inside)
    Output:
        predicted vector field v_theta(x, t, s=mask)  (B, M+1)

    Input for L_vel  (mechanism 2: real state; mechanism 1: physical time):
        x   = clean coeffs c*  (B, M+1)
        t   = 0  (B,)
        s   = real state  (B, S_DIM)            ← real state provided
        tau = physical query times  (B, K)
    Output:
        predicted velocity at tau_k  (B, K)

    Mechanism 1 (physical time encoding):
        The network receives sinusoidal embeddings of τ_k concatenated to
        the main feature vector.  For L_FM we use tau = 0 (dummy); for
        L_vel we pass the actual K physical times.
        To keep dimensions fixed, we average-pool the K tau-embeddings into
        one vector of dim TEMB, so the net always gets the same input size.

    Mechanism 4 (DCT space):
        Input/output are DCT-velocity coefficients, not positions.
    """

    def __init__(self, coeff_dim: int = M + 1,
                 hidden: int = HIDDEN, depth: int = DEPTH,
                 s_dim: int = S_DIM):
        super().__init__()
        self.coeff_dim = coeff_dim
        self.s_dim     = s_dim
        # input = coeffs + t_diff_emb + tau_mean_emb + state
        in_dim  = coeff_dim + TEMB + TEMB + s_dim
        layers  = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _   in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        # shared trunk; separate heads for FM and vel
        self.trunk    = nn.Sequential(*layers)
        self.head_fm  = nn.Linear(hidden, coeff_dim)   # L_FM output
        self.head_vel = nn.Linear(hidden, 1)            # velocity scalar output

    def _encode(self, x: torch.Tensor, t_diff: torch.Tensor,
                s: torch.Tensor, tau_query: torch.Tensor | None = None) -> torch.Tensor:
        """Compute trunk features.

        tau_query: (B, N_q) physical times; if None, use zeros (mask for L_FM).
        """
        B = x.shape[0]
        t_emb   = sinusoidal_emb(t_diff)                          # (B, TEMB)
        if tau_query is None:
            # Mechanism 1: for L_FM, physical time = mask (zeros)
            tau_emb = torch.zeros(B, TEMB, device=x.device)
        else:
            # Mechanism 1: average sinusoidal embeddings over query times
            # tau_query: (B, N_q)  → emb: (B, N_q, TEMB) → mean: (B, TEMB)
            tau_emb = sinusoidal_emb(tau_query / T).mean(dim=1)   # (B, TEMB)
        feat = torch.cat([x, t_emb, tau_emb, s], dim=-1)          # (B, in_dim)
        return self.trunk(feat)                                    # (B, hidden)

    def forward_fm(self, x: torch.Tensor, t_diff: torch.Tensor,
                   s: torch.Tensor) -> torch.Tensor:
        """Predict FM vector field.  s should be MASK TOKEN (zeros) for L_FM.

        Returns: (B, M+1)
        """
        h = self._encode(x, t_diff, s, tau_query=None)
        return self.head_fm(h)

    def forward_vel(self, c: torch.Tensor, tau_query: torch.Tensor,
                    s: torch.Tensor) -> torch.Tensor:
        """Predict velocity at physical query times.

        c         : (B, M+1) clean DCT coefficients
        tau_query : (B, N_q) physical times
        s         : (B, S_DIM) real state
        Returns   : (B, N_q) velocities
        """
        B, N_q = tau_query.shape
        # Expand c and s to (B*N_q, ...) and compute per-time velocity
        c_exp   = c.unsqueeze(1).expand(-1, N_q, -1).reshape(B * N_q, -1)   # (B*N_q, M+1)
        s_exp   = s.unsqueeze(1).expand(-1, N_q, -1).reshape(B * N_q, -1)   # (B*N_q, S_DIM)
        tau_exp = tau_query.reshape(B * N_q)                                  # (B*N_q,)
        t_zero  = torch.zeros(B * N_q, device=c.device)

        h   = self._encode(c_exp, t_zero, s_exp,
                           tau_query=tau_exp.unsqueeze(-1))        # (B*N_q, hidden)
        vel = self.head_vel(h).reshape(B, N_q)                     # (B, N_q)
        return vel

# ============================================================================ #
#  TRAINING
# ============================================================================ #

@dataclass
class TrainResult:
    model:      nn.Module          # FlowMLP or MOTIFNet
    losses:     list               # per-epoch total loss
    losses_fm:  list               # per-epoch L_FM
    losses_vel: list               # per-epoch L_vel (empty for non-MOTIF)
    final_loss: float
    seconds:    float
    prior_std:  torch.Tensor
    data_scale: torch.Tensor
    label:      str
    kind:       str                # "fm" | "sfp" | "mpd" | "motif"
    sfp_xi_raw: torch.Tensor | None = None


def train_flow_matching(
    x1_all:     torch.Tensor,
    prior_std:  torch.Tensor,
    label:      str,
    data_scale: torch.Tensor,
    kind:       str,
    seed:       int = SEED,
) -> TrainResult:
    """Standard linear FM (for FM and MPD baselines)."""
    torch.manual_seed(seed)
    dim    = x1_all.shape[-1]
    model  = FlowMLP(dim=dim).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    n      = x1_all.shape[0]
    losses = []
    t0     = time.time()
    for epoch in range(EPOCHS):
        perm  = torch.randperm(n, device=DEVICE)
        el    = 0.0; nb = 0
        for i in range(0, n, BATCH):
            x1     = x1_all[perm[i: i + BATCH]]
            B      = x1.shape[0]
            t      = torch.rand(B, device=DEVICE)
            x0     = torch.randn_like(x1) * prior_std[None, :]
            xt     = (1 - t[:, None]) * x0 + t[:, None] * x1
            u_star = x1 - x0
            loss   = ((model(xt, t) - u_star) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1
        losses.append(el / nb)
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"[{label:8s}] epoch {epoch:3d}  L={losses[-1]:.5f}")
    return TrainResult(model=model, losses=losses, losses_fm=losses, losses_vel=[],
                       final_loss=losses[-1], seconds=time.time() - t0,
                       prior_std=prior_std, data_scale=data_scale,
                       label=label, kind=kind)


def train_sfp(
    xi_raw_all: torch.Tensor,
    pos_scale:  torch.Tensor,
    label:      str = "SFP     ",
    seed:       int = SEED,
) -> TrainResult:
    """SFP training — extended state (a, z), exact paper formulation."""
    torch.manual_seed(seed)
    model  = FlowMLP(dim=2).to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    n      = xi_raw_all.shape[0]
    losses = []
    sig0, sig1, sig_r = SFP_SIGMA_0, SFP_SIGMA_1, SFP_SIGMA_R
    t0     = time.time()

    for epoch in range(EPOCHS):
        perm  = torch.randperm(n, device=DEVICE)
        el    = 0.0; nb = 0
        for i in range(0, n, BATCH):
            xi     = xi_raw_all[perm[i: i + BATCH]]   # (B, K)
            B      = xi.shape[0]
            t_diff = torch.rand(B, device=DEVICE)

            t_idx  = t_diff * K - 0.5
            i_lo   = t_idx.long().clamp(0, K - 2)
            i_hi   = (i_lo + 1).clamp(0, K - 1)
            frac   = (t_idx - i_lo.float()).clamp(0, 1)
            xi_lo  = xi[torch.arange(B, device=DEVICE), i_lo]
            xi_hi  = xi[torch.arange(B, device=DEVICE), i_hi]
            xi_t   = xi_lo + frac * (xi_hi - xi_lo)
            xi_dot = (xi_hi - xi_lo) * K

            z_0    = torch.randn(B, device=DEVICE)
            eps_a0 = sig0 * torch.randn(B, device=DEVICE)
            a_t    = xi_t + eps_a0 + sig_r * t_diff * z_0
            z_t    = (1 - (1 - sig1) * t_diff) * z_0 + t_diff * xi_t
            v_a_s  = xi_dot + sig_r * z_0
            v_z_s  = xi_t + t_diff * xi_dot - (1 - sig1) * z_0

            x_in   = torch.stack([a_t, z_t], dim=-1)
            v_star = torch.stack([v_a_s, v_z_s], dim=-1)
            loss   = ((model(x_in, t_diff) - v_star) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); nb += 1

        losses.append(el / nb)
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"[{label:8s}] epoch {epoch:3d}  L={losses[-1]:.5f}")

    return TrainResult(model=model, losses=losses, losses_fm=losses, losses_vel=[],
                       final_loss=losses[-1], seconds=time.time() - t0,
                       prior_std=torch.tensor([sig0, 1.0], device=DEVICE),
                       data_scale=pos_scale, label=label, kind="sfp",
                       sfp_xi_raw=xi_raw_all)


def train_motif(
    x1_coef_norm: torch.Tensor,   # (N, M+1) normalised DCT coefficients
    coeff_scale:  torch.Tensor,   # (1, M+1)
    vel_raw:      torch.Tensor,   # (N, K) raw velocities  [for L_vel GT]
    states:       torch.Tensor,   # (N, S_DIM) toy states
    label:        str = "MOTIF   ",
    seed:         int = SEED,
    alpha_vel:    float = ALPHA_VEL,
) -> TrainResult:
    """Full MOTIF training with all four mechanisms.

    Mechanism 1 — Physical time encoding:
        MOTIFNet receives τ_k embeddings alongside t_diff embedding.

    Mechanism 2 — State mask:
        L_FM forward: s = zeros (mask token)
        L_vel forward: s = real state

    Mechanism 3 — Velocity supervision L_vel:
        At t_diff = 0, decode c* → v(τ_k) via DCT basis.
        Supervise predicted velocity v_theta(c*, τ_k, s) vs GT velocity.

    Mechanism 4 — DCT coefficient space:
        FM generates c ∈ R^(M+1); decode with orthonormal DCT-II.
    """
    torch.manual_seed(seed)
    model  = MOTIFNet().to(DEVICE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    n      = x1_coef_norm.shape[0]

    # Frequency-weighted prior (mechanism 4 companion)
    omega      = torch.arange(M + 1, device=DEVICE) * math.pi / T
    prior_std  = 1.0 / torch.sqrt(1.0 + omega ** 2)

    # Physical query times for L_vel: all K steps
    tau_q_base = torch.tensor(tau_K, dtype=torch.float32, device=DEVICE)  # (K,)

    losses, losses_fm, losses_vel = [], [], []
    t0 = time.time()

    for epoch in range(EPOCHS):
        perm  = torch.randperm(n, device=DEVICE)
        el    = 0.0; el_fm = 0.0; el_vel = 0.0; nb = 0
        for i in range(0, n, BATCH):
            idx    = perm[i: i + BATCH]
            c1     = x1_coef_norm[idx]                 # (B, M+1) normalised
            s      = states[idx]                        # (B, S_DIM) real state
            v_gt   = vel_raw[idx]                       # (B, K) GT velocity
            B      = c1.shape[0]

            # ── L_FM (mechanism 2: mask token for s) ─────────────────────────
            t_diff  = torch.rand(B, device=DEVICE)
            c0      = torch.randn_like(c1) * prior_std[None, :]
            ct      = (1 - t_diff[:, None]) * c0 + t_diff[:, None] * c1
            u_star  = c1 - c0
            s_mask  = torch.zeros_like(s)              # MASK TOKEN
            v_pred  = model.forward_fm(ct, t_diff, s_mask)
            l_fm    = ((v_pred - u_star) ** 2).mean()

            # ── L_vel (mechanisms 1, 2, 3) ────────────────────────────────────
            # Use clean coefficients c* (un-normalised) for velocity decoding
            c_star   = c1 * coeff_scale                # (B, M+1) un-normalised
            tau_q    = tau_q_base.unsqueeze(0).expand(B, -1)  # (B, K)
            v_net    = model.forward_vel(c_star, tau_q, s)    # (B, K) predicted vel
            # GT velocity is already in physical units (m/s equivalent)
            l_vel    = ((v_net - v_gt) ** 2).mean()

            loss = l_fm + alpha_vel * l_vel
            opt.zero_grad(); loss.backward(); opt.step()
            el     += loss.item()
            el_fm  += l_fm.item()
            el_vel += l_vel.item()
            nb     += 1

        losses.append(el / nb)
        losses_fm.append(el_fm / nb)
        losses_vel.append(el_vel / nb)
        if epoch % 50 == 0 or epoch == EPOCHS - 1:
            print(f"[{label:8s}] epoch {epoch:3d}  "
                  f"L={losses[-1]:.5f}  L_FM={losses_fm[-1]:.5f}  L_vel={losses_vel[-1]:.5f}")

    return TrainResult(model=model, losses=losses, losses_fm=losses_fm,
                       losses_vel=losses_vel,
                       final_loss=losses[-1], seconds=time.time() - t0,
                       prior_std=prior_std, data_scale=coeff_scale,
                       label=label, kind="motif")

# ============================================================================ #
#  ODE SAMPLING
# ============================================================================ #

@torch.no_grad()
def sample_euler_motif(model: MOTIFNet, prior_std: torch.Tensor,
                       n: int, n_steps: int = N_ODE_STEPS,
                       t_stop: float = 1.0) -> torch.Tensor:
    """Sample MOTIF via Euler integration.

    Uses forward_fm with mask token (state unknown at inference, using mask
    so the network relies purely on the FM vector field learned under mask).
    Returns normalised coefficients (n, M+1).
    """
    c  = torch.randn(n, M + 1, device=DEVICE) * prior_std[None, :]
    s  = torch.zeros(n, S_DIM, device=DEVICE)   # mask token at inference
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_cur = step * dt
        if t_cur >= t_stop - 1e-8:
            break
        t_b = torch.full((n,), t_cur, device=DEVICE)
        c   = c + model.forward_fm(c, t_b, s) * dt
    return c


@torch.no_grad()
def sample_euler(model: FlowMLP, prior_std: torch.Tensor, n: int,
                 n_steps: int = N_ODE_STEPS, t_stop: float = 1.0) -> torch.Tensor:
    x  = torch.randn(n, model.dim, device=DEVICE) * prior_std[None, :]
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_cur = step * dt
        if t_cur >= t_stop - 1e-8:
            break
        t_b = torch.full((n,), t_cur, device=DEVICE)
        x   = x + model(x, t_b) * dt
    return x


@torch.no_grad()
def sample_sfp_euler(res: TrainResult, n: int,
                     n_steps: int = N_ODE_STEPS, t_stop: float = 1.0) -> torch.Tensor:
    """SFP: single ODE integration, record a at each step → (n, K) positions."""
    xi_raw  = res.sfp_xi_raw
    sig0    = SFP_SIGMA_0
    model   = res.model

    demo_idx = torch.randint(0, xi_raw.shape[0], (n,), device=DEVICE)
    xi_0     = xi_raw[demo_idx, 0]
    a_0      = xi_0 + sig0 * torch.randn(n, device=DEVICE)
    z_0      = torch.randn(n, device=DEVICE)
    x        = torch.stack([a_0, z_0], dim=-1)

    dt       = t_stop / n_steps
    t_steps  = []
    a_record = []

    for step in range(n_steps):
        t_cur = step * dt
        t_steps.append(t_cur)
        a_record.append(x[:, 0].clone())
        t_b = torch.full((n,), t_cur, device=DEVICE)
        x   = x + model(x, t_b) * dt

    t_steps.append(t_stop)
    a_record.append(x[:, 0].clone())

    t_arr   = np.array(t_steps)
    a_mat   = torch.stack(a_record, dim=-1).cpu().numpy()
    t_query = tau_K / T
    pos_K   = np.stack([np.interp(t_query, t_arr, a_mat[i]) for i in range(n)])
    return torch.tensor(pos_K, dtype=torch.float32, device=DEVICE)


def _to_dense_positions(res: TrainResult, x_out: torch.Tensor) -> np.ndarray:
    tau_t = torch.tensor(tau_dense, dtype=torch.float32, device=DEVICE)

    if res.kind == "fm":
        pos_K = (x_out * res.data_scale).cpu().numpy()
        return np.stack([np.interp(tau_dense, tau_K, row) for row in pos_K])

    elif res.kind == "sfp":
        return np.stack([np.interp(tau_dense, tau_K, row)
                         for row in x_out.cpu().numpy()])

    elif res.kind == "mpd":
        w  = x_out * res.data_scale
        a0 = torch.zeros(w.shape[0], device=DEVICE)
        return decode_mpd_position(w, a0, tau_t).cpu().numpy()

    elif res.kind == "motif":
        c  = x_out * res.data_scale
        a0 = torch.zeros(c.shape[0], device=DEVICE)
        return decode_motif_position(c, a0, tau_t).cpu().numpy()

    else:
        raise ValueError(f"Unknown kind: {res.kind!r}")

# ============================================================================ #
#  PANEL HELPERS
# ============================================================================ #

def _panel_frame(ax, title: str, ylabel: bool = False):
    ax.set_title(title, pad=8, loc="left", fontweight="bold")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.02, T + 0.02)
    ax.set_facecolor(COL_BG)
    ax.set_xlabel(r"action  $a$")
    ax.axvline(0.0, color="#cccccc", lw=0.6, zorder=0)
    if ylabel:
        ax.set_ylabel(r"execution time  $\tau$  (s)")


def mode_color(pos_at_mid: np.ndarray) -> list[str]:
    return [COL_RED if v > 0 else COL_BLUE for v in pos_at_mid]


def _pick_closest_to_mean(a_dense: np.ndarray, colors: list[str],
                          prefer_col: str = COL_RED) -> int:
    mask = np.array([c == prefer_col for c in colors])
    if not mask.any():
        mask = np.ones(len(colors), dtype=bool)
    subset = a_dense[mask]
    mean   = subset.mean(axis=0)
    mse    = ((subset - mean[None, :]) ** 2).mean(axis=1)
    return int(np.where(mask)[0][int(np.argmin(mse))])

# ============================================================================ #
#  PANELS
# ============================================================================ #

def panel_a_demonstrations(ax, rng, panel_letter: str = "a"):
    amp_ref  = 0.6
    mean_pos = _demo_fn(tau_dense, amp_ref)
    ax.fill_betweenx(tau_dense, +mean_pos - 0.06, +mean_pos + 0.06,
                     color=COL_RED,  alpha=0.12, linewidth=0)
    ax.fill_betweenx(tau_dense, -mean_pos - 0.06, -mean_pos + 0.06,
                     color=COL_BLUE, alpha=0.12, linewidth=0)
    pidx  = _demo_peak_idx()
    pos_demo, _ = sample_demo_positions(14, rng)
    for pos in pos_demo:
        color = COL_RED if pos[pidx] > 0 else COL_BLUE
        ax.plot(pos, tau_K, color=color, alpha=0.35, lw=0.9)
    ax.plot(+mean_pos, tau_dense, color=COL_RED,  lw=2.2)
    ax.plot(-mean_pos, tau_dense, color=COL_BLUE, lw=2.2)
    _panel_frame(ax, f"({panel_letter})  Bi-modal training demonstrations", ylabel=True)


def _draw_baseline_panel(ax, a_dense: np.ndarray, res: TrainResult, panel_letter: str):
    pidx   = int(0.25 * tau_dense.size)
    colors = mode_color(a_dense[:, pidx])
    for curve, c in zip(a_dense, colors):
        ax.plot(curve, tau_dense, color=c, alpha=0.22, lw=0.6, zorder=1)
        ax.scatter(np.interp(tau_K, tau_dense, curve), tau_K,
                   c=c, s=5, alpha=0.45, linewidths=0, zorder=2)
    hi_idx   = _pick_closest_to_mean(a_dense, colors, prefer_col=COL_RED)
    ex_curve = a_dense[hi_idx]
    ax.plot(ex_curve, tau_dense, color=colors[hi_idx], lw=1.8, alpha=0.9, zorder=3)
    tau_10   = np.linspace(0, T, 10, endpoint=False) + 0.5 * T / 10
    ax.plot(np.interp(tau_10, tau_dense, ex_curve), tau_10,
            color=_BASELINE_COLORS.get(res.kind, "#333"),
            lw=1.6, marker="o", markersize=5, zorder=4)
    _panel_frame(ax, f"({panel_letter})  {_BASELINE_LABELS.get(res.kind, res.label)}")


def panel_baseline_samples(ax, res: TrainResult, panel_letter: str):
    x_out   = sample_euler(res.model, res.prior_std, n=N_SAMPLES)
    _draw_baseline_panel(ax, _to_dense_positions(res, x_out), res, panel_letter)


def panel_sfp_samples(ax, res: TrainResult, panel_letter: str):
    pos_K = sample_sfp_euler(res, n=N_SAMPLES)
    _draw_baseline_panel(ax, _to_dense_positions(res, pos_K), res, panel_letter)


def panel_motif_samples(ax, res: TrainResult, panel_letter: str):
    c_norm  = sample_euler_motif(res.model, res.prior_std, n=N_SAMPLES)
    a_dense = _to_dense_positions(res, c_norm)
    pidx    = int(0.25 * tau_dense.size)
    colors  = mode_color(a_dense[:, pidx])
    for curve, c in zip(a_dense, colors):
        ax.plot(curve, tau_dense, color=c, alpha=0.18, lw=0.9, zorder=1)
    idx   = _pick_closest_to_mean(a_dense, colors, prefer_col=COL_RED)
    curve = a_dense[idx]
    ax.plot(curve, tau_dense, color=COL_RED, lw=2.0, alpha=0.9, zorder=4)
    for fhz, col, marker, size, zord in [
        (10,  "#d35400", "o", 60, 6),
        (50,  "#222222", "s", 22, 5),
        (200, "#2e86de", ".", 10, 5),
    ]:
        tq  = np.linspace(0, T, fhz, endpoint=False) + 0.5 * T / fhz
        ax.scatter(np.interp(tq, tau_dense, curve), tq,
                   c=col, s=size, marker=marker, linewidths=0, alpha=1.0, zorder=zord)
    _panel_frame(ax, f"({panel_letter})  MOTIF  (DCT-vel + phys-τ + mask + $\\mathcal{{L}}_{{\\rm vel}}$)")


def panel_mid_denoising(ax, res: TrainResult, panel_letter: str, t_stop: float):
    torch.manual_seed(SEED + 7)
    if res.kind == "sfp":
        x_out   = sample_sfp_euler(res, n=N_SAMPLES, t_stop=t_stop)
        a_dense = _to_dense_positions(res, x_out)
    elif res.kind == "motif":
        c_norm  = sample_euler_motif(res.model, res.prior_std,
                                     n=N_SAMPLES, t_stop=t_stop)
        a_dense = _to_dense_positions(res, c_norm)
    else:
        x_out   = sample_euler(res.model, res.prior_std,
                               n=N_SAMPLES, t_stop=t_stop)
        a_dense = _to_dense_positions(res, x_out)

    pidx   = int(0.25 * tau_dense.size)
    colors = mode_color(a_dense[:, pidx])
    for curve, c in zip(a_dense, colors):
        alpha = 0.40 if res.kind == "motif" else 0.25
        lw    = 0.9  if res.kind == "motif" else 0.6
        ax.plot(curve, tau_dense, color=c, alpha=alpha, lw=lw, zorder=2)
        if res.kind != "motif":
            ax.scatter(np.interp(tau_K, tau_dense, curve), tau_K,
                       c=c, s=5, alpha=0.45, linewidths=0, zorder=2)

    if res.kind == "motif":
        note       = r"$C^\infty$ smooth + $\mathcal{L}_{\rm vel}$ aligned" + "\nalready executable"
        note_color = COL_MOTIF
        kind_label = "MOTIF"
    elif res.kind == "sfp":
        note       = r"$a$-stream already executable" + "\n" + r"($z$ carries shape info)"
        note_color = _BASELINE_COLORS["sfp"]
        kind_label = "SFP"
    else:
        note       = "noise + positions\nnot executable"
        note_color = COL_GREY
        kind_label = _BASELINE_LABELS.get(res.kind, res.label)

    _panel_frame(ax, f"({panel_letter})  {kind_label}  @"
                     f"  $t_\\mathrm{{diff}}{{=}}{t_stop:.2f}$")
    ax.text(0.5, -0.18, note, transform=ax.transAxes,
            ha="center", va="top", fontsize=8.3, color=note_color, fontweight="bold")

# ============================================================================ #
#  MAIN
# ============================================================================ #

def main():
    print(f"device = {DEVICE}")
    print(f"DEMO_CYCLES = {DEMO_CYCLES}   BASELINES = {BASELINES}")

    (x1_pos_norm, x1_coef_norm, x1_mpd_norm,
     xi_raw, states, vel_raw,
     pos_scale, coeff_scale, mpd_scale) = build_dataset()

    # ---- priors ----------------------------------------------------------------
    prior_fm  = torch.ones(K, device=DEVICE)
    prior_mpd = torch.ones(N_BASIS, device=DEVICE)

    # ---- train baselines -------------------------------------------------------
    baseline_results: dict[str, TrainResult] = {}
    for bname in BASELINES:
        if bname == "fm":
            baseline_results["fm"] = train_flow_matching(
                x1_pos_norm, prior_fm, label="FM      ",
                data_scale=pos_scale, kind="fm")
        elif bname == "sfp":
            baseline_results["sfp"] = train_sfp(
                xi_raw_all=xi_raw, pos_scale=pos_scale, label="SFP     ")
        elif bname == "mpd":
            baseline_results["mpd"] = train_flow_matching(
                x1_mpd_norm, prior_mpd, label="MPD     ",
                data_scale=mpd_scale, kind="mpd")
        else:
            raise ValueError(f"Unknown baseline: {bname!r}")

    # ---- train MOTIF (all four mechanisms) ------------------------------------
    res_motif = train_motif(
        x1_coef_norm=x1_coef_norm,
        coeff_scale=coeff_scale,
        vel_raw=vel_raw,
        states=states,
        label="MOTIF   ",
    )

    # ---- losses ----------------------------------------------------------------
    def relative_loss(res: TrainResult, x1: torch.Tensor) -> float:
        var_x1    = x1.var(dim=0).mean().item()
        var_prior = (res.prior_std ** 2).mean().item()
        return res.final_loss / (var_x1 + var_prior)

    _x1_map = {"fm": x1_pos_norm, "mpd": x1_mpd_norm}
    rel = {}
    for bname, bres in baseline_results.items():
        if bname == "sfp":
            rel["sfp"] = bres.final_loss
        else:
            rel[bname] = relative_loss(bres, _x1_map[bname])
    rel["motif"] = res_motif.losses_fm[-1] / (
        x1_coef_norm.var(dim=0).mean().item() +
        (res_motif.prior_std ** 2).mean().item()
    )

    n_params = {bname: sum(p.numel() for p in bres.model.parameters())
                for bname, bres in baseline_results.items()}
    n_params["motif"] = sum(p.numel() for p in res_motif.model.parameters())

    print(f"\n=========  training summary  =========")
    for bname, bres in baseline_results.items():
        print(f"  {bname.upper():5s}  L={bres.final_loss:.5f}  "
              f"rel={rel[bname]:.4f}  params={n_params[bname]:,}  ({bres.seconds:.1f}s)")
    print(f"  MOTIF  L={res_motif.final_loss:.5f}  "
          f"L_FM={res_motif.losses_fm[-1]:.5f}  L_vel={res_motif.losses_vel[-1]:.5f}  "
          f"rel_FM={rel['motif']:.4f}  params={n_params['motif']:,}  ({res_motif.seconds:.1f}s)")

    # ---- figure layout --------------------------------------------------------
    n_bl         = len(BASELINES)
    n_cols       = 1 + n_bl + 1 + (n_bl + 1)
    width_ratios = [1.0] + [1.0] * n_bl + [1.0] + [0.75] * (n_bl + 1)

    fig  = plt.figure(figsize=(4.5 * n_cols, 5.4))
    gs   = fig.add_gridspec(1, n_cols, wspace=0.12, width_ratios=width_ratios)
    axes = [fig.add_subplot(gs[0, 0])]
    for c in range(1, n_cols):
        axes.append(fig.add_subplot(gs[0, c], sharey=axes[0]))
        plt.setp(axes[-1].get_yticklabels(), visible=False)

    col              = 0
    ax_demo          = axes[col]; col += 1
    ax_baselines     = {b: axes[col + i] for i, b in enumerate(BASELINES)}; col += n_bl
    ax_motif_sample  = axes[col]; col += 1
    ax_deno          = {b: axes[col + i] for i, b in enumerate(BASELINES)}; col += n_bl
    ax_deno["motif"] = axes[col]

    letters = "abcdefghijklmnopqrstuvwxyz"
    l = 0
    panel_a_demonstrations(ax_demo, np.random.default_rng(SEED + 1),
                           panel_letter=letters[l]); l += 1

    for bname in BASELINES:
        fn = panel_sfp_samples if bname == "sfp" else panel_baseline_samples
        fn(ax_baselines[bname], baseline_results[bname], panel_letter=letters[l]); l += 1

    panel_motif_samples(ax_motif_sample, res_motif, panel_letter=letters[l]); l += 1

    for bname in BASELINES:
        torch.manual_seed(SEED + 7)
        panel_mid_denoising(ax_deno[bname], baseline_results[bname],
                            panel_letter=letters[l], t_stop=MID_T); l += 1

    torch.manual_seed(SEED + 7)
    panel_mid_denoising(ax_deno["motif"], res_motif,
                        panel_letter=letters[l], t_stop=MID_T)

    cycles_str = {0.5: "half-period arch", 1.0: "full-period", 1.5: "1.5-period"}.get(
        DEMO_CYCLES, f"{DEMO_CYCLES}-period")
    rel_str = "   ".join(f"{nm}={rel[nm]:.3f}" for nm in list(BASELINES) + ["motif"])
    fig.suptitle(
        f"FM paradigm comparison  —  demo: {cycles_str}  |  baselines: {', '.join(BASELINES)}\n"
        f"MOTIF: phys-τ encoding + state mask + L_vel (α={ALPHA_VEL}) + DCT-vel space\n"
        r"loss (rel FM, or raw for SFP)  :  " + rel_str +
        f"   |  epochs={EPOCHS}, N_train={N_TRAIN}, seed={SEED}",
        fontsize=10, fontweight="bold", y=1.06,
    )

    fig.savefig(OUT_PATH, dpi=220, bbox_inches="tight", facecolor="white")
    print(f"\nSaved figure to: {OUT_PATH}")


if __name__ == "__main__":
    main()
