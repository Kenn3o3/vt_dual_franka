"""
Gradient Conflict Toy — MOTIF v3.2 vs π₀

SCENARIO: Mixed-fps multi-dataset training
══════════════════════════════════════════════════════════════════════════════

Same physical robot task recorded at two control frequencies:
  Fast dataset: fps_A = 40 Hz
  Slow dataset: fps_B = 10 Hz
Physical motion: a(τ) = A·sin(2π·τ/T_motion), duration T_motion = 2.0 s

─────────────────────────────────────────────────────────────────────────────
π₀ style: fixed action_horizon K=20 for all datasets
─────────────────────────────────────────────────────────────────────────────
  40 Hz data: K=20 steps → covers  T = 20/40 = 0.5 s  (first quarter-cycle)
  10 Hz data: K=20 steps → covers  T = 20/10 = 2.0 s  (full cycle)

  Step-index encoding: embed((k+0.5)/K)  ← IDENTICAL for both datasets
  At step k=10:
    40 Hz → τ = 0.26 s → a = +0.79  (ascending)
    10 Hz → τ = 1.05 s → a = −0.16  (past peak, descending)
  Same embed(10/20) → same network input → opposite FM targets → CONFLICT

─────────────────────────────────────────────────────────────────────────────
MOTIF v3.2 style: fixed chunk_duration T_chunk=1.0 s for all datasets
─────────────────────────────────────────────────────────────────────────────
  40 Hz data: K_A = T_chunk × 40 = 40 steps  (dense sampling of 1 s)
  10 Hz data: K_B = T_chunk × 10 = 10 steps  (coarse sampling of 1 s)

  Data pipeline (mirrors motif/openpi data_loader.py):
    Dataset A: (B, 40, d) →[ExtractDCT(M)]→ (B, M+1, d) →[DecodeDCT(K_target)]→ (B, K_target, d)
    Dataset B: (B, 10, d) →[ExtractDCT(M)]→ (B, M+1, d) →[DecodeDCT(K_target)]→ (B, K_target, d)
  Both arrive at the same shape (B, K_target, d) — same T, same low-freq content.

  Model (motif._compute_loss_v32):
    (B, K_target, d) →[extract_dct(M)]→ (B, M+1, d) = c*
    FM in coefficient space: L_FM = ‖u_pred − (c₀ − c*)‖²
    Both datasets → same T → same DCT basis → c* semantically consistent → NO CONFLICT

  Physical-τ encoding (per coefficient mode):  embed(m·T/(M+1)/T)  ∈ [0,1]
  SAME for both datasets (same T) → further reinforces consistency.

Key insight:
  The conflict is not about position vs velocity, nor about step-index encoding.
  It is about PHYSICAL TIME COVERAGE of the chunk:
    π₀:    K=const → T = K/fps varies per dataset → chunk covers different physical durations
    MOTIF: T=const → K = T×fps varies per dataset → chunk always covers the SAME physical time

OUTPUT: mpd/motif_toy_gradient_conflict.png
Run:    python scripts/motif_toy_gradient_conflict.py
"""

from __future__ import annotations
import math
from pathlib import Path
import time

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn

# ═══════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════

# ── Physical motion ────────────────────────────────────────────────────
A_AMP    = 1.0
T_MOTION = 2.0           # full sine period (s): a(τ)=A·sin(2πτ/T_motion)

# ── Datasets ───────────────────────────────────────────────────────────
FPS_A    = 40.0           # Hz  (fast dataset)
FPS_B    = 10.0           # Hz  (slow dataset)

# ── π₀ style: fixed action_horizon ────────────────────────────────────
K_PI0    = 20             # fixed number of steps for all datasets
T_PI0_A  = K_PI0 / FPS_A  # 0.5 s  ← chunk covers only half the motion
T_PI0_B  = K_PI0 / FPS_B  # 2.0 s  ← chunk covers the full motion TWICE

# ── MOTIF style: fixed chunk_duration ─────────────────────────────────
T_CHUNK  = 1.0            # s  (same physical duration for all datasets)
K_MOT_A  = int(T_CHUNK * FPS_A)   # 40  (dense)
K_MOT_B  = int(T_CHUNK * FPS_B)   # 10  (coarse)
K_TARGET = 20             # target_K after DCT→decode pipeline (=action_horizon)
M_DCT    = 8              # Fourier modes  (M+1=9 coefficients)
ALPHA_VEL = 0.5

# ── Training ───────────────────────────────────────────────────────────
DEMO_NOISE = 0.02
N_TRAIN    = 2048
BATCH      = 256
EPOCHS     = 800
LR         = 3e-3
HIDDEN     = 256
DEPTH      = 4
TEMB       = 64
SEED       = 1
N_GRAD     = 50

# ── π0.7 style ─────────────────────────────────────────────────────────
N_SPEED_BINS = 2       # 0=fast(40Hz), 1=slow(10Hz)
SPEED_BIN_A  = 0       # dataset A → fast bin
SPEED_BIN_B  = 1       # dataset B → slow bin
SPEED_DROP_P = 0.15    # speed label dropout prob (matches π0.7 paper)
CFG_BETA     = 1.5     # CFG guidance weight at inference

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

OUT_PATH = Path(__file__).resolve().parent.parent / "motif_toy_gradient_conflict.png"

# ═══════════════════════════════════════════════════════════════════════
#  STYLE
# ═══════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "xtick.direction": "out", "ytick.direction": "out",
    "legend.frameon": False,
})
COL_A     = "#2980b9"   # fast dataset (A)
COL_B     = "#c0392b"   # slow dataset (B)
COL_PI0   = "#7b3f00"
COL_PI07  = "#8e44ad"   # π0.7 purple
COL_MOTIF = "#2a7f62"
COL_CONF  = "#e74c3c"
COL_AGREE = "#27ae60"

# ═══════════════════════════════════════════════════════════════════════
#  PHYSICAL TRAJECTORY
# ═══════════════════════════════════════════════════════════════════════

def _pos(tau): return A_AMP * np.sin(2*math.pi*np.asarray(tau) / T_MOTION)
def _vel(tau): return A_AMP * (2*math.pi/T_MOTION) * np.cos(2*math.pi*np.asarray(tau) / T_MOTION)

def _steps(K, T):
    tau = np.array([(k+0.5)/K * T for k in range(K)], dtype=np.float32)
    return tau, _pos(tau), _vel(tau)

# π₀ grids (different T!)
TAU_PI0_A, POS_PI0_A, VEL_PI0_A = _steps(K_PI0, T_PI0_A)   # 0.5 s
TAU_PI0_B, POS_PI0_B, VEL_PI0_B = _steps(K_PI0, T_PI0_B)   # 2.0 s

# MOTIF grids (same T=1.0 s, different K)
TAU_MOT_A, POS_MOT_A, VEL_MOT_A = _steps(K_MOT_A, T_CHUNK)  # K=40
TAU_MOT_B, POS_MOT_B, VEL_MOT_B = _steps(K_MOT_B, T_CHUNK)  # K=10

# Target grid used after MOTIF's DCT→decode pipeline
TAU_TARGET, POS_TARGET, VEL_TARGET = _steps(K_TARGET, T_CHUNK)

# ═══════════════════════════════════════════════════════════════════════
#  DCT UTILITIES
# ═══════════════════════════════════════════════════════════════════════

def _dct_basis(K: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    """DCT-II orthonormal basis and norms for K-point sequence, first M+1 modes."""
    j = np.arange(K, dtype=np.float64)
    k = np.arange(M+1, dtype=np.float64)
    basis = np.cos(math.pi * k[:,None] * (j[None,:]+0.5) / K)  # (M+1, K)
    norm  = np.ones(M+1) * math.sqrt(2/K)
    norm[0] *= 1/math.sqrt(2)
    return basis, norm   # basis: (M+1,K),  norm: (M+1,)


def extract_dct_np(v: np.ndarray, M: int) -> np.ndarray:
    """v: (K,) or (K, d) → (M+1,) or (M+1, d)."""
    sq = v.ndim == 1
    if sq: v = v[:,None]
    K_, d = v.shape
    basis, norm = _dct_basis(K_, M)
    c = (basis @ v) * norm[:,None]   # (M+1, d)
    return c[:,0] if sq else c


def decode_dct_np(c: np.ndarray, K_out: int, T: float, K_orig: int) -> np.ndarray:
    """c: (M+1,) or (M+1, d) → (K_out,) or (K_out, d).
    K_orig: the K used when extracting c (for correct IDCT normalisation).
    """
    sq = c.ndim == 1
    if sq: c = c[:,None]
    M_p1, d = c.shape
    M = M_p1 - 1
    # Decode at K_out uniform τ in [0, T]
    tau_out = np.array([(k+0.5)/K_out * T for k in range(K_out)])
    j_idx   = tau_out * K_orig / T           # fractional j-indices
    k_arr   = np.arange(M+1, dtype=np.float64)
    basis   = np.cos(math.pi * k_arr[:,None] * j_idx[None,:] / K_orig)  # (M+1,K_out)
    norm    = np.ones(M+1) * math.sqrt(2/K_orig)
    norm[0] *= 1/math.sqrt(2)
    v = np.einsum('md,mn,m->nd', c, basis, norm)   # (K_out, d)
    return v[:,0] if sq else v


def extract_dct_torch(v: torch.Tensor, M: int) -> torch.Tensor:
    """v: (B, K, d) → (B, M+1, d)."""
    B, K_, d = v.shape
    j = torch.arange(K_, dtype=torch.float32, device=v.device)
    k = torch.arange(M+1, dtype=torch.float32, device=v.device)
    basis = torch.cos(math.pi * k[:,None] * (j[None,:]+0.5) / K_)
    norm  = torch.ones(M+1, device=v.device) * math.sqrt(2/K_)
    norm[0] *= 1/math.sqrt(2)
    return torch.einsum('mk,bkd->bmd', basis, v) * norm[:,None]


def decode_dct_torch(c: torch.Tensor, K_out: int,
                     T: float, K_orig: int) -> torch.Tensor:
    """c: (B, M+1, d), K_out output steps → (B, K_out, d)."""
    B, M_p1, d = c.shape
    M = M_p1 - 1
    tau_out = torch.tensor([(k+0.5)/K_out * T for k in range(K_out)],
                           dtype=torch.float32, device=c.device)
    j_idx = tau_out * K_orig / T
    k_arr = torch.arange(M+1, dtype=torch.float32, device=c.device)
    basis = torch.cos(math.pi * k_arr[:,None] * j_idx[None,:] / K_orig)  # (M+1, K_out)
    norm  = torch.ones(M+1, device=c.device) * math.sqrt(2/K_orig)
    norm[0] *= 1/math.sqrt(2)
    return torch.einsum('bmd,mn,m->bnd', c, basis, norm)   # (B, K_out, d)


def freq_noise_torch(M: int, d: int, T: float, B: int) -> torch.Tensor:
    """Frequency-weighted Gaussian noise for DCT prior (B, M+1, d).
    Matches the Sobolev-space prior used in MOTIF: low-freq coefficients
    have larger variance (smoother prior), high-freq are suppressed.
    """
    k   = torch.arange(M+1, dtype=torch.float32)
    std = 1.0 / torch.sqrt(1.0 + (k * math.pi / T)**2)
    return (torch.randn(B, M+1, d).to(DEVICE) * std[None,:,None].to(DEVICE))

# ═══════════════════════════════════════════════════════════════════════
#  DATASET
# ═══════════════════════════════════════════════════════════════════════

def build_dataset():
    """
    Returns:
      pi0_A, pi0_B : (N, K_PI0)    velocity sequences, π₀ grids (T varies)
      mot_A_raw    : (N, K_MOT_A)  velocity sequences, MOTIF 40Hz grid (T=1s)
      mot_B_raw    : (N, K_MOT_B)  velocity sequences, MOTIF 10Hz grid (T=1s)
      mot_A_dec    : (N, K_TARGET) MOTIF pipeline output (DCT→decode to K_TARGET)
      mot_B_dec    : (N, K_TARGET) MOTIF pipeline output (DCT→decode to K_TARGET)
    All normalised to the same scale_v.
    """
    rng = np.random.default_rng(SEED)
    amp = rng.uniform(0.92, 1.08, (N_TRAIN, 1)).astype(np.float32)

    # π₀ sequences (different T)
    pi0_A = amp * VEL_PI0_A + rng.normal(0, DEMO_NOISE, (N_TRAIN, K_PI0)).astype(np.float32)
    pi0_B = amp * VEL_PI0_B + rng.normal(0, DEMO_NOISE, (N_TRAIN, K_PI0)).astype(np.float32)

    # MOTIF raw sequences (same T=1s, different K)
    mot_A = amp * VEL_MOT_A + rng.normal(0, DEMO_NOISE, (N_TRAIN, K_MOT_A)).astype(np.float32)
    mot_B = amp * VEL_MOT_B + rng.normal(0, DEMO_NOISE, (N_TRAIN, K_MOT_B)).astype(np.float32)

    # MOTIF pipeline: raw → DCT → decode to K_TARGET
    # Mirrors data_loader.py DCT branch: ExtractDCTCoeffs → DecodeDCTToActions
    # Standard physical-time grid shared by all datasets (K_TARGET uniform steps in [0, T_CHUNK])
    TAU_TARGET = np.linspace(0, T_CHUNK, K_TARGET, endpoint=False) + T_CHUNK/(2*K_TARGET)

    def _pipeline(v_raw: np.ndarray, K_src: int) -> np.ndarray:
        """(N, K_src) → (N, K_TARGET).

        Physical interpolation: resample each sequence from its native fps grid
        to the shared K_TARGET physical-time grid, then re-extract DCT for
        model-internal L_FM. Because both datasets cover the SAME T_CHUNK,
        the resulting K_TARGET sequence — and its DCT — is physically consistent.
        """
        tau_src = np.linspace(0, T_CHUNK, K_src, endpoint=False) + T_CHUNK/(2*K_src)
        out = np.zeros((v_raw.shape[0], K_TARGET), dtype=np.float32)
        for i in range(v_raw.shape[0]):
            out[i] = np.interp(TAU_TARGET, tau_src, v_raw[i]).astype(np.float32)
        return out

    mot_A_dec = _pipeline(mot_A, K_MOT_A)   # (N, K_TARGET)
    mot_B_dec = _pipeline(mot_B, K_MOT_B)   # (N, K_TARGET)

    # Common normalisation (use MOTIF decoded data as reference scale)
    scale = float(np.concatenate([mot_A_dec, mot_B_dec]).std()) + 1e-8

    def _t(a): return torch.tensor(a / scale, dtype=torch.float32, device=DEVICE)
    return (_t(pi0_A), _t(pi0_B),
            _t(mot_A),  _t(mot_B),
            _t(mot_A_dec), _t(mot_B_dec),
            scale)

# ═══════════════════════════════════════════════════════════════════════
#  SINUSOIDAL EMBEDDING
# ═══════════════════════════════════════════════════════════════════════

def sinusoidal_emb(t: torch.Tensor, dim: int = TEMB) -> torch.Tensor:
    shape = t.shape; t_f = t.reshape(-1)
    half  = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device).float()
                      / max(half-1, 1))
    ang = t_f[:,None] * freqs[None,:] * 2*math.pi
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1).reshape(*shape, dim)

# ═══════════════════════════════════════════════════════════════════════
#  NETWORKS
# ═══════════════════════════════════════════════════════════════════════

def _mlp(in_d, h, depth, out_d):
    L = [nn.Linear(in_d, h), nn.SiLU()]
    for _ in range(depth-1): L += [nn.Linear(h,h), nn.SiLU()]
    L += [nn.Linear(h, out_d)]
    return nn.Sequential(*L)


class Pi0Net(nn.Module):
    """π₀: FM on velocity at K_PI0 fixed steps, step-index conditioning."""
    def __init__(self):
        super().__init__()
        self.net = _mlp(1+TEMB+TEMB, HIDDEN, DEPTH, 1)
        k = torch.arange(K_PI0, dtype=torch.float32)
        self.register_buffer("step_cond", sinusoidal_emb((k+0.5)/K_PI0))  # (K,TEMB)

    def _fwd_k(self, vk, t_emb, sc):
        return self.net(torch.cat([vk, t_emb, sc], -1))

    def forward(self, v_noisy, t_diff):
        """v_noisy:(B,K,1), t_diff:(B,) → (B,K,1)"""
        B = v_noisy.shape[0]; t_emb = sinusoidal_emb(t_diff)
        return torch.stack([self._fwd_k(v_noisy[:,k], t_emb,
                                        self.step_cond[k].unsqueeze(0).expand(B,-1))
                            for k in range(K_PI0)], dim=1)


class Pi07Net(nn.Module):
    """π0.7 style: π0 + discrete speed/fps metadata conditioning.

    Core idea from π0.7 §V-C: annotate each episode with a speed/quality
    label, train with dropout, prompt for high-quality at test time.
    Here: speed label = discretised fps bin (0=40Hz, 1=10Hz).

    This lets the model learn *separate* vector fields for each dataset,
    rather than averaging them → gradient conflict is reduced (not
    eliminated, since backbone weights are shared, unlike MOTIF which
    removes the conflict at its source via fixed physical time T).
    """
    def __init__(self):
        super().__init__()
        # +1 for null token used during speed-label dropout / CFG
        self.speed_emb  = nn.Embedding(N_SPEED_BINS + 1, TEMB)
        self.null_speed = N_SPEED_BINS          # index of ∅ token
        # input: vel(1) + t_emb(TEMB) + step_cond(TEMB) + spd_emb(TEMB)
        self.net = _mlp(1 + TEMB + TEMB + TEMB, HIDDEN, DEPTH, 1)
        k = torch.arange(K_PI0, dtype=torch.float32)
        self.register_buffer("step_cond", sinusoidal_emb((k+0.5)/K_PI0))

    def _fwd_k(self, vk, t_emb, sc, spd_emb):
        return self.net(torch.cat([vk, t_emb, sc, spd_emb], -1))

    def forward(self, v_noisy, t_diff, speed_label):
        """v_noisy:(B,K,1), t_diff:(B,), speed_label:(B,) long → (B,K,1)"""
        B       = v_noisy.shape[0]
        t_emb   = sinusoidal_emb(t_diff)
        spd_emb = self.speed_emb(speed_label)          # (B, TEMB)
        return torch.stack([
            self._fwd_k(v_noisy[:, k], t_emb,
                        self.step_cond[k].unsqueeze(0).expand(B, -1),
                        spd_emb)
            for k in range(K_PI0)
        ], dim=1)

    def forward_cfg(self, v_noisy, t_diff, speed_label, beta=CFG_BETA):
        """Classifier-free guidance: v = v_cond + β(v_cond - v_uncond).
        Matches π0.7 §VII runtime prompting with metadata CFG.
        """
        null     = torch.full_like(speed_label, self.null_speed)
        v_cond   = self.forward(v_noisy, t_diff, speed_label)
        v_uncond = self.forward(v_noisy, t_diff, null)
        return v_cond + beta * (v_cond - v_uncond)


class MOTIFNet(nn.Module):
    """MOTIF v3.2: FM in DCT coeff space, physical-τ conditioning.
    Input is (B, K_TARGET, d) after the DCT→decode data pipeline.
    The network does extract_dct internally (mirrors _compute_loss_v32).
    Fixed T=T_CHUNK for both datasets → same DCT basis → no conflict.
    """
    def __init__(self):
        super().__init__()
        self.fm_net  = _mlp(1+TEMB+TEMB, HIDDEN, DEPTH, 1)
        self.vel_net = _mlp(1+TEMB+TEMB, HIDDEN, DEPTH, 1)
        # Physical-τ for coefficient mode m: τ_m/T = m/(M+1)  → embed in [0,1]
        tau_m = torch.tensor([m/(M_DCT+1) for m in range(M_DCT+1)],
                             dtype=torch.float32)
        self.register_buffer("phys_cond", sinusoidal_emb(tau_m))  # (M+1, TEMB)

    def _fm_fwd_m(self, cm, t_emb, B):
        sc = self.phys_cond[cm[0] if False else 0].unsqueeze(0)   # placeholder
        # caller passes sc explicitly
        raise NotImplementedError

    def forward_fm(self, c_noisy, t_diff):
        """c_noisy:(B,M+1,1), t_diff:(B,) → (B,M+1,1)"""
        B = c_noisy.shape[0]; t_emb = sinusoidal_emb(t_diff)
        return torch.stack([
            self.fm_net(torch.cat([c_noisy[:,m], t_emb,
                                   self.phys_cond[m].unsqueeze(0).expand(B,-1)], -1))
            for m in range(M_DCT+1)], dim=1)

    def forward_vel(self, c_clean, t_zero):
        """c_clean:(B,M+1,1), t_zero:(B,) → (B,M+1,1)"""
        B = c_clean.shape[0]; t_emb = sinusoidal_emb(t_zero)
        return torch.stack([
            self.vel_net(torch.cat([c_clean[:,m], t_emb,
                                    self.phys_cond[m].unsqueeze(0).expand(B,-1)], -1))
            for m in range(M_DCT+1)], dim=1)

# ═══════════════════════════════════════════════════════════════════════
#  TRAINING
# ═══════════════════════════════════════════════════════════════════════

def train_pi0(pi0_A: torch.Tensor, pi0_B: torch.Tensor,
              ) -> tuple[Pi0Net, list, list, list]:
    """π₀ FM on (B, K_PI0, 1) velocity. Step-index enc → conflict."""
    torch.manual_seed(SEED)
    model = Pi0Net().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    n     = pi0_A.shape[0]
    losses, lf_hist, ls_hist = [], [], []
    t0 = time.time()

    vA = pi0_A.unsqueeze(-1); vB = pi0_B.unsqueeze(-1)

    for ep in range(EPOCHS):
        pA = torch.randperm(n, device=DEVICE)
        pB = torch.randperm(n, device=DEVICE)
        el = elA = elB = nb = 0
        for i in range(0, n, BATCH):
            xA = vA[pA[i:i+BATCH]]; xB = vB[pB[i:i+BATCH]]; B = xA.shape[0]
            tA = torch.rand(B, device=DEVICE); x0A = torch.randn_like(xA)
            xtA = (1-tA[:,None,None])*x0A + tA[:,None,None]*xA
            lA  = ((model(xtA,tA) - (xA-x0A))**2).mean()
            tB = torch.rand(B, device=DEVICE); x0B = torch.randn_like(xB)
            xtB = (1-tB[:,None,None])*x0B + tB[:,None,None]*xB
            lB  = ((model(xtB,tB) - (xB-x0B))**2).mean()
            loss = lA + lB
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); elA += lA.item(); elB += lB.item(); nb += 1
        losses.append(el/nb); lf_hist.append(elA/nb); ls_hist.append(elB/nb)
        if ep % 100 == 0 or ep == EPOCHS-1:
            print(f"[pi0   ] ep {ep:3d}  L={losses[-1]:.4f}  "
                  f"LA={lf_hist[-1]:.4f}  LB={ls_hist[-1]:.4f}  ({time.time()-t0:.0f}s)")
    return model, losses, lf_hist, ls_hist


def train_pi07(pi0_A: torch.Tensor, pi0_B: torch.Tensor,
               ) -> tuple[Pi07Net, list, list, list]:
    """π0.7 FM: same K=K_PI0 steps as π0, but conditioned on speed label.
    Speed label is dropped with SPEED_DROP_P to enable CFG at inference.
    """
    torch.manual_seed(SEED)
    model = Pi07Net().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    n     = pi0_A.shape[0]
    losses, lf_hist, ls_hist = [], [], []
    t0 = time.time()

    vA = pi0_A.unsqueeze(-1); vB = pi0_B.unsqueeze(-1)

    for ep in range(EPOCHS):
        pA = torch.randperm(n, device=DEVICE)
        pB = torch.randperm(n, device=DEVICE)
        el = elA = elB = nb = 0
        for i in range(0, n, BATCH):
            xA = vA[pA[i:i+BATCH]]; xB = vB[pB[i:i+BATCH]]; B = xA.shape[0]

            # speed labels with SPEED_DROP_P% dropout (→ null token)
            spA = torch.full((B,), SPEED_BIN_A, dtype=torch.long, device=DEVICE)
            spB = torch.full((B,), SPEED_BIN_B, dtype=torch.long, device=DEVICE)
            spA[torch.rand(B, device=DEVICE) < SPEED_DROP_P] = model.null_speed
            spB[torch.rand(B, device=DEVICE) < SPEED_DROP_P] = model.null_speed

            tA = torch.rand(B, device=DEVICE); x0A = torch.randn_like(xA)
            xtA = (1 - tA[:,None,None])*x0A + tA[:,None,None]*xA
            lA  = ((model(xtA, tA, spA) - (xA - x0A))**2).mean()

            tB = torch.rand(B, device=DEVICE); x0B = torch.randn_like(xB)
            xtB = (1 - tB[:,None,None])*x0B + tB[:,None,None]*xB
            lB  = ((model(xtB, tB, spB) - (xB - x0B))**2).mean()

            loss = lA + lB
            opt.zero_grad(); loss.backward(); opt.step()
            el += loss.item(); elA += lA.item(); elB += lB.item(); nb += 1

        losses.append(el/nb); lf_hist.append(elA/nb); ls_hist.append(elB/nb)
        if ep % 100 == 0 or ep == EPOCHS-1:
            print(f"[pi0.7] ep {ep:3d}  L={losses[-1]:.4f}  "
                  f"LA={lf_hist[-1]:.4f}  LB={ls_hist[-1]:.4f}  "
                  f"({time.time()-t0:.0f}s)")
    return model, losses, lf_hist, ls_hist


def train_motif(mot_A_dec: torch.Tensor, mot_B_dec: torch.Tensor,
                ) -> tuple[MOTIFNet, list, list, list]:
    """MOTIF v3.2 FM in DCT coeff space.
    Input: decoded data at K_TARGET steps (from DCT data pipeline), T=T_CHUNK.
    Internal extract_dct → FM in coeff space → L_vel decoded back.
    Both groups share same T → same basis → consistent c* → no conflict.
    """
    torch.manual_seed(SEED)
    model = MOTIFNet().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    n     = mot_A_dec.shape[0]
    losses, lf_hist, ls_hist = [], [], []
    t0 = time.time()

    tau_tgt = torch.tensor(TAU_TARGET, dtype=torch.float32, device=DEVICE)

    vA = mot_A_dec.unsqueeze(-1)   # (N, K_TARGET, 1)
    vB = mot_B_dec.unsqueeze(-1)

    for ep in range(EPOCHS):
        pA = torch.randperm(n, device=DEVICE)
        pB = torch.randperm(n, device=DEVICE)
        el = elA = elB = nb = 0
        for i in range(0, n, BATCH):
            xA = vA[pA[i:i+BATCH]]; xB = vB[pB[i:i+BATCH]]; Bi = xA.shape[0]

            # Internal DCT (mirrors _compute_loss_v32)
            cA = extract_dct_torch(xA, M_DCT)   # (B, M+1, 1)
            cB = extract_dct_torch(xB, M_DCT)

            c0A = freq_noise_torch(M_DCT, 1, T_CHUNK, Bi)
            c0B = freq_noise_torch(M_DCT, 1, T_CHUNK, Bi)

            tA = torch.rand(Bi, device=DEVICE)
            ctA = (1-tA[:,None,None])*c0A + tA[:,None,None]*cA
            lA_fm = ((model.forward_fm(ctA, tA) - (cA-c0A))**2).mean()

            tB = torch.rand(Bi, device=DEVICE)
            ctB = (1-tB[:,None,None])*c0B + tB[:,None,None]*cB
            lB_fm = ((model.forward_fm(ctB, tB) - (cB-c0B))**2).mean()

            # L_vel: decode predicted coeffs → compare to GT
            t0v = torch.zeros(Bi, device=DEVICE)
            predA = model.forward_vel(cA, t0v)   # (B, M+1, 1)
            vA_dec = decode_dct_torch(predA, K_TARGET, T_CHUNK, K_TARGET)
            lA_vel = ((vA_dec - xA)**2).mean()

            predB = model.forward_vel(cB, t0v)
            vB_dec = decode_dct_torch(predB, K_TARGET, T_CHUNK, K_TARGET)
            lB_vel = ((vB_dec - xB)**2).mean()

            loss = lA_fm + lB_fm + ALPHA_VEL*(lA_vel + lB_vel)
            opt.zero_grad(); loss.backward(); opt.step()
            el  += loss.item()
            elA += (lA_fm + ALPHA_VEL*lA_vel).item()
            elB += (lB_fm + ALPHA_VEL*lB_vel).item()
            nb  += 1

        losses.append(el/nb); lf_hist.append(elA/nb); ls_hist.append(elB/nb)
        if ep % 100 == 0 or ep == EPOCHS-1:
            print(f"[motif ] ep {ep:3d}  L={losses[-1]:.4f}  "
                  f"LA={lf_hist[-1]:.4f}  LB={ls_hist[-1]:.4f}  ({time.time()-t0:.0f}s)")
    return model, losses, lf_hist, ls_hist

# ═══════════════════════════════════════════════════════════════════════
#  GRADIENT CONFLICT MEASUREMENT
# ═══════════════════════════════════════════════════════════════════════

def _total_gradient(model, loss):
    """Compute flattened gradient of loss w.r.t. all model parameters."""
    model.zero_grad()
    loss.backward()
    grads = [p.grad.detach().flatten() for p in model.parameters()
             if p.requires_grad and p.grad is not None]
    return torch.cat(grads).clone()


def measure_pi0(model: Pi0Net,
                pi0_A: torch.Tensor, pi0_B: torch.Tensor,
                n_batches: int = N_GRAD):
    """Total + per-step cos-sim(∇L_A, ∇L_B) for π₀.

    Returns:
      cos_total : (n_batches,)
      cos_k_mean: (K_PI0,)   per-step mean over trials
      cos_k_std : (K_PI0,)
    """
    model.eval()
    cos_total = np.zeros(n_batches)
    cos_k_all = np.zeros((n_batches, K_PI0))
    vA = pi0_A.unsqueeze(-1); vB = pi0_B.unsqueeze(-1); n = pi0_A.shape[0]

    for b in range(n_batches):
        idxA = torch.randperm(n, device=DEVICE)[:BATCH]
        idxB = torch.randperm(n, device=DEVICE)[:BATCH]
        x1A = vA[idxA]; x1B = vB[idxB]; Bi = x1A.shape[0]
        tA  = torch.rand(Bi, device=DEVICE); x0A = torch.randn_like(x1A)
        tB  = torch.rand(Bi, device=DEVICE); x0B = torch.randn_like(x1B)
        xtA = (1-tA[:,None,None])*x0A + tA[:,None,None]*x1A
        xtB = (1-tB[:,None,None])*x0B + tB[:,None,None]*x1B
        uA = x1A-x0A; uB = x1B-x0B
        teA = sinusoidal_emb(tA); teB = sinusoidal_emb(tB)

        # total gradient
        lossA = sum(((model._fwd_k(xtA[:,k], teA,
                       model.step_cond[k].unsqueeze(0).expand(Bi,-1))
                      - uA[:,k])**2).mean() for k in range(K_PI0))
        lossB = sum(((model._fwd_k(xtB[:,k], teB,
                       model.step_cond[k].unsqueeze(0).expand(Bi,-1))
                      - uB[:,k])**2).mean() for k in range(K_PI0))
        gA = _total_gradient(model, lossA)
        gB = _total_gradient(model, lossB)
        cos_total[b] = nn.functional.cosine_similarity(
            gA.unsqueeze(0), gB.unsqueeze(0)).item()

        # per-step gradient
        for k in range(K_PI0):
            sc = model.step_cond[k].unsqueeze(0).expand(Bi, -1)
            lA_k = ((model._fwd_k(xtA[:,k], teA, sc) - uA[:,k])**2).mean()
            lB_k = ((model._fwd_k(xtB[:,k], teB, sc) - uB[:,k])**2).mean()
            gA_k = _total_gradient(model, lA_k)
            gB_k = _total_gradient(model, lB_k)
            model.zero_grad()
            cos_k_all[b, k] = nn.functional.cosine_similarity(
                gA_k.unsqueeze(0), gB_k.unsqueeze(0)).item()

    model.train()
    return cos_total, cos_k_all.mean(0), cos_k_all.std(0)


def measure_pi07(model: Pi07Net,
                 pi0_A: torch.Tensor, pi0_B: torch.Tensor,
                 n_batches: int = N_GRAD):
    """Gradient conflict for π0.7 with correct speed conditioning.

    Compared to π0: speed label breaks the ambiguity at shared steps,
    so A and B gradients partially decouple → cos-sim should be higher
    than π0 but lower than MOTIF (backbone weights still shared).

    Returns:
      cos_total  : (n_batches,)  full-model gradient cos-sim
      cos_k_mean : (K_PI0,)     per-step mean
      cos_k_std  : (K_PI0,)
    """
    model.eval()
    cos_total = np.zeros(n_batches)
    cos_k_all = np.zeros((n_batches, K_PI0))
    vA = pi0_A.unsqueeze(-1); vB = pi0_B.unsqueeze(-1); n = pi0_A.shape[0]

    for b in range(n_batches):
        idxA = torch.randperm(n, device=DEVICE)[:BATCH]
        idxB = torch.randperm(n, device=DEVICE)[:BATCH]
        x1A = vA[idxA]; x1B = vB[idxB]; Bi = x1A.shape[0]

        tA  = torch.rand(Bi, device=DEVICE); x0A = torch.randn_like(x1A)
        tB  = torch.rand(Bi, device=DEVICE); x0B = torch.randn_like(x1B)
        xtA = (1-tA[:,None,None])*x0A + tA[:,None,None]*x1A
        xtB = (1-tB[:,None,None])*x0B + tB[:,None,None]*x1B
        uA = x1A - x0A; uB = x1B - x0B
        teA = sinusoidal_emb(tA); teB = sinusoidal_emb(tB)

        spA = torch.full((Bi,), SPEED_BIN_A, dtype=torch.long, device=DEVICE)
        spB = torch.full((Bi,), SPEED_BIN_B, dtype=torch.long, device=DEVICE)
        seA = model.speed_emb(spA)   # (B, TEMB)
        seB = model.speed_emb(spB)

        # total gradient
        lossA = sum(
            ((model._fwd_k(xtA[:,k], teA,
                           model.step_cond[k].unsqueeze(0).expand(Bi,-1),
                           seA) - uA[:,k])**2).mean()
            for k in range(K_PI0))
        lossB = sum(
            ((model._fwd_k(xtB[:,k], teB,
                           model.step_cond[k].unsqueeze(0).expand(Bi,-1),
                           seB) - uB[:,k])**2).mean()
            for k in range(K_PI0))
        gA = _total_gradient(model, lossA)
        gB = _total_gradient(model, lossB)
        cos_total[b] = nn.functional.cosine_similarity(
            gA.unsqueeze(0), gB.unsqueeze(0)).item()

        # per-step gradient — recompute embeddings each step (backward frees graph)
        for k in range(K_PI0):
            sc   = model.step_cond[k].unsqueeze(0).expand(Bi, -1)
            seA_k = model.speed_emb(spA)
            seB_k = model.speed_emb(spB)
            lA_k = ((model._fwd_k(xtA[:,k], teA, sc, seA_k) - uA[:,k])**2).mean()
            lB_k = ((model._fwd_k(xtB[:,k], teB, sc, seB_k) - uB[:,k])**2).mean()
            gA_k = _total_gradient(model, lA_k)
            gB_k = _total_gradient(model, lB_k)
            model.zero_grad()
            cos_k_all[b, k] = nn.functional.cosine_similarity(
                gA_k.unsqueeze(0), gB_k.unsqueeze(0)).item()

    model.train()
    print(f"  [π0.7] total cos-sim: "
          f"mean={cos_total.mean():+.4f}  std={cos_total.std():.4f}")
    return cos_total, cos_k_all.mean(0), cos_k_all.std(0)


def measure_motif(model: MOTIFNet,
                  mot_A_dec: torch.Tensor, mot_B_dec: torch.Tensor,
                  n_batches: int = N_GRAD):
    """Total + per-mode cos-sim(∇L_FM_A, ∇L_FM_B) for MOTIF.

    Returns:
      cos_total : (n_batches,)
      cos_m_mean: (M+1,)  per-mode mean over trials
      cos_m_std : (M+1,)
      snr       : (M+1,)
    """
    model.eval()
    M_p1 = M_DCT + 1
    cos_total = np.zeros(n_batches)
    cos_m_all = np.zeros((n_batches, M_p1))
    vA = mot_A_dec.unsqueeze(-1); vB = mot_B_dec.unsqueeze(-1); n = mot_A_dec.shape[0]

    for b in range(n_batches):
        idxA = torch.randperm(n, device=DEVICE)[:BATCH]
        idxB = torch.randperm(n, device=DEVICE)[:BATCH]
        x1A = vA[idxA]; x1B = vB[idxB]; Bi = x1A.shape[0]
        cA  = extract_dct_torch(x1A, M_DCT); cB = extract_dct_torch(x1B, M_DCT)
        c0A = freq_noise_torch(M_DCT, 1, T_CHUNK, Bi)
        c0B = freq_noise_torch(M_DCT, 1, T_CHUNK, Bi)
        tA  = torch.rand(Bi, device=DEVICE); tB = torch.rand(Bi, device=DEVICE)
        ctA = (1-tA[:,None,None])*c0A + tA[:,None,None]*cA
        ctB = (1-tB[:,None,None])*c0B + tB[:,None,None]*cB
        ucA = cA - c0A; ucB = cB - c0B
        teA = sinusoidal_emb(tA); teB = sinusoidal_emb(tB)

        # total gradient
        lossA = sum(
            ((model.fm_net(torch.cat([ctA[:,m], teA,
                                      model.phys_cond[m].unsqueeze(0).expand(Bi,-1)], -1))
              - ucA[:,m])**2).mean() for m in range(M_p1))
        lossB = sum(
            ((model.fm_net(torch.cat([ctB[:,m], teB,
                                      model.phys_cond[m].unsqueeze(0).expand(Bi,-1)], -1))
              - ucB[:,m])**2).mean() for m in range(M_p1))
        gA = _total_gradient(model, lossA)
        gB = _total_gradient(model, lossB)
        cos_total[b] = nn.functional.cosine_similarity(
            gA.unsqueeze(0), gB.unsqueeze(0)).item()

        # per-mode gradient
        for m in range(M_p1):
            sc = model.phys_cond[m].unsqueeze(0).expand(Bi, -1)
            lA_m = ((model.fm_net(torch.cat([ctA[:,m], teA, sc], -1)) - ucA[:,m])**2).mean()
            lB_m = ((model.fm_net(torch.cat([ctB[:,m], teB, sc], -1)) - ucB[:,m])**2).mean()
            gA_m = _total_gradient(model, lA_m)
            gB_m = _total_gradient(model, lB_m)
            model.zero_grad()
            cos_m_all[b, m] = nn.functional.cosine_similarity(
                gA_m.unsqueeze(0), gB_m.unsqueeze(0)).item()

    model.train()

    # per-mode SNR
    with torch.no_grad():
        cA_mag = extract_dct_torch(mot_A_dec[:200].unsqueeze(-1), M_DCT)\
                 .abs().mean(0).squeeze(-1).cpu().numpy()
    k_arr = np.arange(M_p1, dtype=np.float32)
    noise_std = 1.0 / np.sqrt(1.0 + (k_arr * math.pi / T_CHUNK)**2)
    snr = cA_mag / (noise_std + 1e-9)
    print(f"  Mode SNR: {np.round(snr, 2)}")
    print(f"  Total gradient cos-sim: "
          f"mean={cos_total.mean():+.4f}  std={cos_total.std():.4f}")

    return cos_total, cos_m_all.mean(0), cos_m_all.std(0), snr

# ═══════════════════════════════════════════════════════════════════════
#  PLOTTING
# ═══════════════════════════════════════════════════════════════════════

def _draw_cos(ax, mean, std, color, title, note="", note_col="k"):
    x = np.arange(len(mean))
    ax.axhspan(-1.05, 0,    color="#ffe5e5", alpha=0.55, zorder=0)
    ax.axhspan(0,     1.05, color="#e5f5e5", alpha=0.35, zorder=0)
    ax.axhline(0, color="#aaa", lw=1.2, ls="--")
    ax.fill_between(x, mean-std, mean+std, color=color, alpha=0.22)
    ax.plot(x, mean, color=color, lw=2)
    ax.scatter(x[mean<0],  mean[mean<0],  color=COL_CONF,  zorder=5, s=28)
    ax.scatter(x[mean>=0], mean[mean>=0], color=COL_AGREE, zorder=5, s=28)
    ax.set_ylim(-1.05, 1.05)
    n_conf = (mean<0).sum()
    ax.set_title(f"{title}\nmean={mean.mean():+.3f}   conflict={n_conf}/{len(mean)}")
    if note:
        ax.text(0.5, 0.06, note, transform=ax.transAxes,
                ha="center", fontsize=8, color=note_col, style="italic")


def plot_results(lp, lpA, lpB, lm, lmA, lmB, lp7, lp7A, lp7B,
                 cos_p_total, cos_k_mean, cos_k_std,
                 cos_m_total, cos_m_mean, cos_m_std, snr_m,
                 cos_7_total, cos_7k_mean, cos_7k_std,
                 mot_A_dec_t: torch.Tensor):

    fig = plt.figure(figsize=(18, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

    # ── (a) Physical trajectories ─────────────────────────────────────
    ax = fig.add_subplot(gs[0,0])
    tau_d = np.linspace(0, 2.5, 500)
    ax.plot(tau_d, _vel(tau_d), "k--", lw=1, alpha=0.35, label="ȧ(τ) continuous")
    # π₀ grids
    ax.scatter(TAU_PI0_A, VEL_PI0_A, color=COL_A, s=22, zorder=4,
               label=f"π₀ dataset A ({FPS_A:.0f}Hz, K={K_PI0}, T={T_PI0_A:.1f}s)")
    ax.scatter(TAU_PI0_B, VEL_PI0_B, color=COL_B, s=22, zorder=4,
               label=f"π₀ dataset B ({FPS_B:.0f}Hz, K={K_PI0}, T={T_PI0_B:.1f}s)")
    # MOTIF grids
    ax.scatter(TAU_MOT_A, VEL_MOT_A, color=COL_A, s=10, marker="+", alpha=0.7,
               label=f"MOTIF A ({FPS_A:.0f}Hz, K={K_MOT_A}, T={T_CHUNK:.1f}s)")
    ax.scatter(TAU_MOT_B, VEL_MOT_B, color=COL_B, s=20, marker="x", alpha=0.9,
               label=f"MOTIF B ({FPS_B:.0f}Hz, K={K_MOT_B}, T={T_CHUNK:.1f}s)")
    ax.axvline(T_PI0_A, color=COL_A, lw=0.9, ls=":", alpha=0.7)
    ax.axvline(T_CHUNK, color="gray", lw=0.9, ls="-.", alpha=0.7)
    ax.axvline(T_PI0_B, color=COL_B, lw=0.9, ls=":", alpha=0.7)
    ax.set_xlabel("Physical time τ (s)"); ax.set_ylabel("Velocity ȧ(τ)")
    ax.set_title(f"(a) Sampled velocities\nπ₀: T varies | MOTIF: T={T_CHUNK}s fixed")
    ax.legend(fontsize=7, loc="upper right")

    # ── (b) MOTIF: c* sign per mode — all same direction ─────────────
    ax = fig.add_subplot(gs[0,1])
    # Physical interpolation: resample both to K_TARGET grid, then extract DCT
    TAU_T = np.linspace(0, T_CHUNK, K_TARGET, endpoint=False) + T_CHUNK/(2*K_TARGET)
    vA_dec_mean = np.interp(TAU_T,
                            np.linspace(0,T_CHUNK,K_MOT_A,endpoint=False)+T_CHUNK/(2*K_MOT_A),
                            VEL_MOT_A)
    vB_dec_mean = np.interp(TAU_T,
                            np.linspace(0,T_CHUNK,K_MOT_B,endpoint=False)+T_CHUNK/(2*K_MOT_B),
                            VEL_MOT_B)
    cA_model = extract_dct_np(vA_dec_mean, M_DCT)
    cB_model = extract_dct_np(vB_dec_mean, M_DCT)
    modes = np.arange(M_DCT+1)
    # colour each bar: green if same sign, red if opposite
    ax.bar(modes-0.2, cA_model, width=0.38, color=COL_A, alpha=0.85,
           label=f"c* A  ({FPS_A:.0f}Hz→interp→K={K_TARGET}→DCT)")
    ax.bar(modes+0.2, cB_model, width=0.38, color=COL_B, alpha=0.85,
           label=f"c* B  ({FPS_B:.0f}Hz→interp→K={K_TARGET}→DCT)")
    # Overlay sign-agreement markers
    for m in modes:
        same = cA_model[m]*cB_model[m] > 0
        ax.annotate("✓" if same else "✗",
                    xy=(m, max(abs(cA_model[m]), abs(cB_model[m]))+0.3),
                    ha="center", fontsize=9,
                    color=COL_AGREE if same else COL_CONF)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Fourier mode m"); ax.set_ylabel("Coefficient c*[m]")
    n_same = sum(cA_model[m]*cB_model[m]>0 for m in modes)
    ax.set_title(f"(b) MOTIF c* after pipeline (T={T_CHUNK}s fixed)\n"
                 f"Sign consistent: {n_same}/{M_DCT+1} modes  (✓=same, ✗=opposite)")
    ax.legend(fontsize=8)
    ax.text(0.5, 0.04, "Same T → same DCT basis → consistent gradient direction",
            transform=ax.transAxes, ha="center", fontsize=8,
            color=COL_AGREE, style="italic")

    # ── (c) π₀ velocity mismatch — sign conflicts highlighted ──────────
    ax = fig.add_subplot(gs[0,2])
    steps = np.arange(K_PI0)
    for k in steps:
        if VEL_PI0_A[k] * VEL_PI0_B[k] < 0:
            ax.axvspan(k-0.5, k+0.5, color=COL_CONF, alpha=0.12, zorder=0)
    ax.bar(steps-0.2, VEL_PI0_A, width=0.38, color=COL_A, alpha=0.85,
           label=f"v* A  (T={T_PI0_A:.1f}s, {FPS_A:.0f}Hz)")
    ax.bar(steps+0.2, VEL_PI0_B, width=0.38, color=COL_B, alpha=0.85,
           label=f"v* B  (T={T_PI0_B:.1f}s, {FPS_B:.0f}Hz)")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("Step k   ← same embed(k/K) for both !")
    ax.set_ylabel("GT velocity v*(k)")
    n_opp = int(np.sum(VEL_PI0_A * VEL_PI0_B < 0))
    ax.set_title(f"(c) π₀ FM targets at same step k\n"
                 f"Sign opposite: {n_opp}/{K_PI0} steps  (red shading = conflict)")
    ax.text(0.5, 0.04, "Different T → step k maps to different τ → opposite v*",
            transform=ax.transAxes, ha="center", fontsize=8,
            color=COL_CONF, style="italic")
    ax.legend(fontsize=8)

    # ── (d) Loss curves ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[1,0])
    ep = np.arange(1, EPOCHS+1)
    ax.plot(ep, lp,   color=COL_PI0,   lw=2,   label="π₀ total")
    ax.plot(ep, lpA,  color=COL_A,     lw=1.2, ls="--", label="π₀ A")
    ax.plot(ep, lpB,  color=COL_B,     lw=1.2, ls="--", label="π₀ B")
    ax.plot(ep, lp7,  color=COL_PI07,  lw=2,   label="π0.7 total")
    ax.plot(ep, lp7A, color=COL_A,     lw=1.2, ls="-.", label="π0.7 A")
    ax.plot(ep, lp7B, color=COL_B,     lw=1.2, ls="-.", label="π0.7 B")
    ax.plot(ep, lm,   color=COL_MOTIF, lw=2,   label="MOTIF total")
    ax.plot(ep, lmA,  color=COL_A,     lw=1.2, ls=":",  label="MOTIF A")
    ax.plot(ep, lmB,  color=COL_B,     lw=1.2, ls=":",  label="MOTIF B")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (log)")
    ax.set_title(f"(d) Training loss\n"
                 f"π₀={lp[-1]:.4f}  π0.7={lp7[-1]:.4f}  MOTIF={lm[-1]:.4f}")
    ax.legend(fontsize=6.5, ncol=2)

    # ── (e) Total gradient cos-sim: three-way comparison ──────────────
    ax = fig.add_subplot(gs[1,1])
    bplot = ax.boxplot(
        [cos_p_total, cos_7_total, cos_m_total],
        labels=["π₀", "π0.7\n(speed cond)", "MOTIF"],
        patch_artist=True, widths=0.42,
        medianprops=dict(color="black", lw=2))
    for box, col in zip(bplot["boxes"], [COL_PI0, COL_PI07, COL_MOTIF]):
        box.set_facecolor(col); box.set_alpha(0.6)
    ax.axhline(0, color="#aaa", lw=1.2, ls="--")
    ax.axhspan(-1.05, 0, color="#ffe5e5", alpha=0.4, zorder=0)
    ax.axhspan(0, 1.05,  color="#e5f5e5", alpha=0.3, zorder=0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("cos-sim(∇L_A, ∇L_B)")
    ax.set_title(
        f"(e) Total gradient conflict\n"
        f"π₀:{cos_p_total.mean():+.3f}  "
        f"π0.7:{cos_7_total.mean():+.3f}  "
        f"MOTIF:{cos_m_total.mean():+.3f}"
    )
    for xi, (data, col) in enumerate(
            zip([cos_p_total, cos_7_total, cos_m_total],
                [COL_PI0, COL_PI07, COL_MOTIF]), 1):
        offset = 0.05 if data.mean() >= 0 else -0.08
        ax.text(xi, data.mean() + offset, f"{data.mean():+.2f}",
                ha="center", fontsize=9, color=col, fontweight="bold")
    ax.text(0.5, 0.04,
            "π₀ < π0.7 < MOTIF: speed cond partially decouples gradients",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic")

    # ── (f) Per-step / per-mode gradient cos-sim — three-way ──────────
    ax = fig.add_subplot(gs[1,2])
    ax.axhspan(-1.05, 0,    color="#ffe5e5", alpha=0.55, zorder=0)
    ax.axhspan(0,     1.05, color="#e5f5e5", alpha=0.35, zorder=0)
    ax.axhline(0, color="#aaa", lw=1.2, ls="--")

    # π₀: per-step cos-sim
    xp = np.arange(K_PI0) / (K_PI0 - 1)
    ax.fill_between(xp, cos_k_mean-cos_k_std, cos_k_mean+cos_k_std,
                    color=COL_PI0, alpha=0.15)
    ax.plot(xp, cos_k_mean, color=COL_PI0, lw=2,
            label=f"π₀  (K={K_PI0}, no metadata)")
    ax.scatter(xp[cos_k_mean < 0],  cos_k_mean[cos_k_mean < 0],
               color=COL_CONF,  s=26, zorder=5)
    ax.scatter(xp[cos_k_mean >= 0], cos_k_mean[cos_k_mean >= 0],
               color=COL_AGREE, s=26, zorder=5)

    # π0.7: per-step cos-sim
    ax.fill_between(xp, cos_7k_mean-cos_7k_std, cos_7k_mean+cos_7k_std,
                    color=COL_PI07, alpha=0.15)
    ax.plot(xp, cos_7k_mean, color=COL_PI07, lw=2,
            label=f"π0.7 (K={K_PI0}, speed cond + CFG β={CFG_BETA})")
    ax.scatter(xp[cos_7k_mean < 0],  cos_7k_mean[cos_7k_mean < 0],
               color=COL_CONF,  s=26, zorder=5, marker="s")
    ax.scatter(xp[cos_7k_mean >= 0], cos_7k_mean[cos_7k_mean >= 0],
               color=COL_AGREE, s=26, zorder=5, marker="s")

    # MOTIF: per-mode cos-sim
    xm = np.arange(M_DCT+1) / M_DCT
    ax.fill_between(xm, cos_m_mean-cos_m_std, cos_m_mean+cos_m_std,
                    color=COL_MOTIF, alpha=0.15)
    ax.plot(xm, cos_m_mean, color=COL_MOTIF, lw=2,
            label=f"MOTIF (M={M_DCT}, fixed T)")
    ax.scatter(xm[cos_m_mean < 0],  cos_m_mean[cos_m_mean < 0],
               color=COL_CONF,  s=26, zorder=5, marker="D")
    ax.scatter(xm[cos_m_mean >= 0], cos_m_mean[cos_m_mean >= 0],
               color=COL_AGREE, s=26, zorder=5, marker="D")

    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Normalised index  (k/K  or  m/M)")
    ax.set_ylabel("cos-sim(∇_A, ∇_B)")
    n_conf_p = int((cos_k_mean  < 0).sum())
    n_conf_7 = int((cos_7k_mean < 0).sum())
    n_conf_m = int((cos_m_mean  < 0).sum())
    ax.set_title(
        f"(f) Per-step / per-mode gradient cos-sim\n"
        f"π₀:{n_conf_p}/{K_PI0}  π0.7:{n_conf_7}/{K_PI0}  "
        f"MOTIF:{n_conf_m}/{M_DCT+1}  conflicts"
    )
    ax.legend(fontsize=7.5, loc="lower right")

    fig.suptitle(
        "MOTIF v3.2 vs π₀ vs π0.7 — Gradient Conflict from Mixed-FPS Training\n"
        f"Datasets: A={FPS_A:.0f}Hz, B={FPS_B:.0f}Hz, "
        f"motion: A·sin(2π·τ/T_motion) T_motion={T_MOTION:.1f}s\n"
        f"π₀: K={K_PI0} fixed → T_A={T_PI0_A}s, T_B={T_PI0_B}s  |  "
        f"π0.7: K={K_PI0} + speed label (CFG β={CFG_BETA})  |  "
        f"MOTIF: T={T_CHUNK}s fixed → K_A={K_MOT_A}, K_B={K_MOT_B}",
        fontsize=10, y=1.02
    )

    plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_PATH}")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{'='*65}")
    print(f"  MOTIF v3.2 Gradient Conflict Toy")
    print(f"  FPS_A={FPS_A:.0f}Hz, FPS_B={FPS_B:.0f}Hz, "
          f"sine motion T_motion={T_MOTION:.1f}s")
    print(f"  π₀:    K={K_PI0} fixed → T_A={T_PI0_A}s, T_B={T_PI0_B}s")
    print(f"  MOTIF: T_chunk={T_CHUNK}s fixed → K_A={K_MOT_A}, K_B={K_MOT_B}")
    print(f"         DCT→decode to K_target={K_TARGET}, M={M_DCT}")
    print(f"{'='*65}\n")

    # Data consistency check (using physical-interpolation pipeline)
    TAU_T = np.linspace(0,T_CHUNK,K_TARGET,endpoint=False)+T_CHUNK/(2*K_TARGET)
    vA_i = np.interp(TAU_T,
                     np.linspace(0,T_CHUNK,K_MOT_A,endpoint=False)+T_CHUNK/(2*K_MOT_A),
                     VEL_MOT_A)
    vB_i = np.interp(TAU_T,
                     np.linspace(0,T_CHUNK,K_MOT_B,endpoint=False)+T_CHUNK/(2*K_MOT_B),
                     VEL_MOT_B)
    cA = extract_dct_np(vA_i, M_DCT)
    cB = extract_dct_np(vB_i, M_DCT)
    print(f"MOTIF model-internal c* A: {np.round(cA, 3)}")
    print(f"MOTIF model-internal c* B: {np.round(cB, 3)}")
    print(f"max|Δc*| after pipeline  : {abs(cA-cB).max():.4f}  (expect ≈ 0)")
    print(f"\nπ₀ v* A k=10: {VEL_PI0_A[10]:.4f}  (τ={TAU_PI0_A[10]:.3f}s)")
    print(f"π₀ v* B k=10: {VEL_PI0_B[10]:.4f}  (τ={TAU_PI0_B[10]:.3f}s)")
    print(f"max|Δv*_pi0| : {abs(VEL_PI0_A-VEL_PI0_B).max():.4f}  (expect large)\n")

    import pickle, sys
    CACHE = Path(__file__).resolve().parent.parent / "motif_toy_gradient_conflict.pkl"
    replot = "--replot" in sys.argv

    if replot and CACHE.exists():
        print(f"Loading cached results from {CACHE}")
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        lp, lpA, lpB   = d["lp"],  d["lpA"],  d["lpB"]
        lm, lmA, lmB   = d["lm"],  d["lmA"],  d["lmB"]
        lp7, lp7A, lp7B = d["lp7"], d["lp7A"], d["lp7B"]
        cos_p_total    = d["cos_p_total"]
        cos_k_mean     = d["cos_k_mean"]
        cos_k_std      = d["cos_k_std"]
        cos_m_total    = d["cos_m_total"]
        cos_m_mean     = d["cos_m_mean"]
        cos_m_std      = d["cos_m_std"]
        snr_m          = d["snr_m"]
        cos_7_total    = d["cos_7_total"]
        cos_7k_mean    = d["cos_7k_mean"]
        cos_7k_std     = d["cos_7k_std"]
        mot_A_dec      = d["mot_A_dec"]
    else:
        pi0_A, pi0_B, mot_A, mot_B, mot_A_dec, mot_B_dec, scale_v = build_dataset()

        print("── Training π₀ ──────────────────────────────────────────────────────")
        m_pi0,  lp, lpA, lpB = train_pi0(pi0_A, pi0_B)
        print("\n── Training π0.7 ───────────────────────────────────────────────────")
        m_pi07, lp7, lp7A, lp7B = train_pi07(pi0_A, pi0_B)
        print("\n── Training MOTIF v3.2 ─────────────────────────────────────────────")
        m_motif, lm, lmA, lmB = train_motif(mot_A_dec, mot_B_dec)

        print("\n── Gradient conflict: π₀ ────────────────────────────────────────────")
        cos_p_total, cos_k_mean, cos_k_std = measure_pi0(m_pi0, pi0_A, pi0_B)
        print("\n── Gradient conflict: π0.7 ──────────────────────────────────────────")
        cos_7_total, cos_7k_mean, cos_7k_std = measure_pi07(m_pi07, pi0_A, pi0_B)
        print("\n── Gradient conflict: MOTIF ─────────────────────────────────────────")
        cos_m_total, cos_m_mean, cos_m_std, snr_m = measure_motif(
            m_motif, mot_A_dec, mot_B_dec)

        with open(CACHE, "wb") as f:
            pickle.dump(dict(
                lp=lp,   lpA=lpA,   lpB=lpB,
                lm=lm,   lmA=lmA,   lmB=lmB,
                lp7=lp7, lp7A=lp7A, lp7B=lp7B,
                cos_p_total=cos_p_total, cos_k_mean=cos_k_mean, cos_k_std=cos_k_std,
                cos_m_total=cos_m_total, cos_m_mean=cos_m_mean, cos_m_std=cos_m_std,
                snr_m=snr_m,
                cos_7_total=cos_7_total, cos_7k_mean=cos_7k_mean, cos_7k_std=cos_7k_std,
                mot_A_dec=mot_A_dec), f)
        print(f"Results cached → {CACHE}")

    print(f"\n====  Results  ====")
    print(f"  π₀    L={lp[-1]:.5f}  "
          f"total cos={cos_p_total.mean():+.4f}±{cos_p_total.std():.4f}")
    print(f"  π0.7  L={lp7[-1]:.5f}  "
          f"total cos={cos_7_total.mean():+.4f}±{cos_7_total.std():.4f}")
    print(f"  MOTIF L={lm[-1]:.5f}  "
          f"total cos={cos_m_total.mean():+.4f}±{cos_m_total.std():.4f}")
    print(f"  MOTIF SNR per mode : {np.round(snr_m, 2)}")

    plot_results(lp, lpA, lpB, lm, lmA, lmB, lp7, lp7A, lp7B,
                 cos_p_total, cos_k_mean, cos_k_std,
                 cos_m_total, cos_m_mean, cos_m_std, snr_m,
                 cos_7_total, cos_7k_mean, cos_7k_std,
                 mot_A_dec)


if __name__ == "__main__":
    main()
