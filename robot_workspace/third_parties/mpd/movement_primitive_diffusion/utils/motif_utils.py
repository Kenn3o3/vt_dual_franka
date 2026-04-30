"""
MOTIF Utilities: DCT encoding/decoding, frequency-weighted noise, and energy-based M estimation.

This module implements the core mathematical operations for MOTIF v3.2:
- DCT-II coefficient extraction from velocity sequences
- Fourier decoding from coefficients to velocity at arbitrary physical times
- Frequency-weighted Gaussian noise prior
- Energy-based criterion for determining the number of Fourier modes M
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
from scipy.fft import dct, idct


class MOTIFHandler:
    """
    Handler for MOTIF trajectory encoding/decoding using DCT-II basis.
    
    This replaces ProDMPHandler for MOTIF, providing:
    - Encoding: velocity sequence -> DCT coefficients
    - Decoding: DCT coefficients -> velocity at arbitrary times
    - Structural C^∞ continuity by construction

    Ablation flag:
    - use_dct=False: identity encode/decode, num_modes is overridden to traj_steps-1
      so that the model operates on K raw velocity frames instead of M+1 DCT tokens.
    """
    
    def __init__(
        self,
        num_dof: int,
        dt: float,
        traj_steps: int,
        num_modes: int,
        chunk_duration: float = 1.0,
        device: Union[str, torch.device] = "cpu",
        use_dct: bool = True,
    ):
        """
        Initialize MOTIF handler.
        
        Args:
            num_dof: Number of degrees of freedom (action dimension)
            dt: Time step of the trajectory (seconds)
            traj_steps: Number of steps in the trajectory (K)
            num_modes: Number of Fourier modes to retain (M); ignored when use_dct=False
            chunk_duration: Duration of action chunk in seconds (T)
            device: Device for computation
            use_dct: If False, skip DCT and operate on raw velocity frames (M3 ablation)
        """
        self.num_dof = num_dof
        self.dt = dt
        self.traj_steps = traj_steps
        self.use_dct = use_dct

        # When DCT is disabled use K tokens (traj_steps) instead of M+1
        self.num_modes = num_modes if use_dct else traj_steps - 1
        self.chunk_duration = chunk_duration
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Precompute frequency values for noise weighting
        self.omega_k = torch.arange(0, self.num_modes + 1, dtype=torch.float32, device=device) * np.pi / chunk_duration
        
    @property
    def encoding_size(self) -> int:
        """Return the size of the coefficient vector: (M+1) * num_dof"""
        return (self.num_modes + 1) * self.num_dof
    
    def encode(
        self,
        velocities: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode velocity sequences to DCT-II coefficients.

        When use_dct=False (M3 ablation), returns the raw velocity frames
        unchanged as [batch_size, K, num_dof] (identity operation).
        
        Args:
            velocities: Velocity sequences [batch_size, traj_steps, num_dof]
            
        Returns:
            coeffs: DCT coefficients [batch_size, num_modes+1, num_dof],
                    or raw velocities [batch_size, K, num_dof] when use_dct=False
        """
        batch_size, K, d = velocities.shape
        assert K == self.traj_steps, f"Expected {self.traj_steps} steps, got {K}"
        assert d == self.num_dof, f"Expected {self.num_dof} DOF, got {d}"

        if not self.use_dct:
            # Identity: return raw velocity frames (num_modes = K-1, so K tokens)
            return velocities.to(self.device).float()
        
        # Move to CPU for scipy DCT, then back to device
        velocities_np = velocities.detach().cpu().numpy()
        
        # Apply DCT-II along time axis (axis=1) with orthonormal normalization
        coeffs_full = dct(velocities_np, axis=1, norm='ortho')  # [B, K, d]
        
        # Retain only first M+1 modes
        # Note: If K < M+1, we take all K modes
        actual_modes = min(K, self.num_modes + 1)
        coeffs = coeffs_full[:, :actual_modes, :]  # [B, min(K, M+1), d]
        
        # Debug: check if we have fewer modes than expected
        if actual_modes < self.num_modes + 1:
            print(f"[WARNING] MOTIFHandler.encode: K={K} < M+1={self.num_modes + 1}")
            print(f"  Only {actual_modes} modes available, expected {self.num_modes + 1}")
            print(f"  This will cause shape mismatches!")
        
        return torch.from_numpy(coeffs).to(self.device).float()
    
    def decode(
        self,
        coeffs: torch.Tensor,
        times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode DCT coefficients to velocity at specified physical times.
        
        Implements: v(τ) = Σ_{k=0}^M c_k · φ_k(τ), where φ_k(τ) = cos(kπτ/T)

        When use_dct=False (M3 ablation), returns the token values directly
        (the "coeffs" are already raw velocity frames, so this is identity).
        
        Args:
            coeffs: DCT coefficients [batch_size, num_modes+1, num_dof]
                    or raw velocities [batch_size, K, num_dof] when use_dct=False
            times: Physical times in seconds [batch_size, N_query] or None
                   If None, uses uniform grid [0, dt, 2*dt, ..., (K-1)*dt]
            
        Returns:
            velocities: Decoded velocities [batch_size, N_query, num_dof]
        """
        batch_size, Mp1, d = coeffs.shape
        assert Mp1 == self.num_modes + 1, f"Expected {self.num_modes + 1} modes, got {Mp1}"
        assert d == self.num_dof, f"Expected {self.num_dof} DOF, got {d}"
        device = coeffs.device

        if not self.use_dct:
            # Identity: coeffs are raw velocity frames, return as-is
            return coeffs
        
        if times is None:
            # Default: uniform grid matching training data
            times = torch.arange(0, self.traj_steps, dtype=torch.float32, device=device) * self.dt
            times = times.unsqueeze(0).expand(batch_size, -1)  # [B, K]
        else:
            times = times.to(device)
        
        N_query = times.shape[1]
        
        # Compute DCT-II basis functions: φ_k(τ) = cos(kπτ/T)
        k_indices = torch.arange(0, self.num_modes + 1, dtype=torch.float32, device=device)  # [M+1]
        
        # Broadcast: [B, N_query, 1] * [1, 1, M+1] -> [B, N_query, M+1]
        # Note: DCT-II uses cos(k*pi*(j+0.5)/N) for encoding, but for decoding at arbitrary times
        # we use cos(k*pi*tau/T) where tau is physical time
        basis = torch.cos(
            k_indices[None, None, :] * np.pi * times[:, :, None] / self.chunk_duration
        )  # [B, N_query, M+1]
        
        # Apply orthonormal normalization (DCT-II convention)
        # The coefficients from scipy.fft.dct with norm='ortho' are already normalized
        # For decoding, we need to apply the inverse normalization
        norm_weights = torch.ones(self.num_modes + 1, dtype=torch.float32, device=device)
        norm_weights[0] = 1.0 / np.sqrt(2.0)
        norm_weights = norm_weights * np.sqrt(2.0 / self.traj_steps)  # [M+1]
        
        # Apply normalization to basis instead of coefficients
        basis_normalized = basis * norm_weights[None, None, :]  # [B, N_query, M+1]
        
        # Decode: [B, N_query, M+1] @ [B, M+1, d] -> [B, N_query, d]
        velocities = torch.einsum('bnm,bmd->bnd', basis_normalized, coeffs)
        
        return velocities
    
    def decode_to_position(
        self,
        coeffs: torch.Tensor,
        initial_position: torch.Tensor,
        times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode coefficients to position trajectory via integration.
        
        Args:
            coeffs: DCT coefficients [batch_size, num_modes+1, num_dof]
            initial_position: Initial position [batch_size, num_dof]
            times: Physical times [batch_size, N_query] or None
            
        Returns:
            positions: Position trajectory [batch_size, N_query, num_dof]
        """
        velocities = self.decode(coeffs, times)  # [B, N, d]
        initial_position = initial_position.to(velocities.device)
        
        # Cumulative integration: a(t) = a_0 + ∫_0^t v(s) ds
        # Approximate with cumulative sum: a[k] = a_0 + Σ_{i=0}^{k-1} v[i] * dt
        positions = initial_position[:, None, :] + torch.cumsum(velocities * self.dt, dim=1)
        
        return positions


def extract_dct_coeffs(
    velocities: np.ndarray,
    num_modes: int,
) -> np.ndarray:
    """
    Extract DCT-II coefficients from velocity sequences (offline preprocessing).
    
    This function is used during dataset preprocessing to cache DCT coefficients.
    
    Args:
        velocities: Velocity sequence [K, d] or [B, K, d]
        num_modes: Number of Fourier modes to retain (M)
        
    Returns:
        coeffs: DCT coefficients [M+1, d] or [B, M+1, d]
    """
    if velocities.ndim == 2:
        # Single trajectory: [K, d]
        coeffs_full = dct(velocities, axis=0, norm='ortho')
        return coeffs_full[:num_modes + 1, :]
    elif velocities.ndim == 3:
        # Batch of trajectories: [B, K, d]
        coeffs_full = dct(velocities, axis=1, norm='ortho')
        return coeffs_full[:, :num_modes + 1, :]
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {velocities.shape}")


def decode_coeffs_to_velocity(
    coeffs: np.ndarray,
    times: np.ndarray,
    chunk_duration: float,
    K_norm: int,
) -> np.ndarray:
    """
    Decode DCT coefficients to velocity at arbitrary physical times.
    
    Args:
        coeffs: DCT coefficients [M+1, d] or [B, M+1, d]
        times: Physical times in seconds [N_query,] or [B, N_query]
        chunk_duration: Duration T in seconds
        K_norm: Original sequence length for orthonormal normalization
        
    Returns:
        velocities: Decoded velocities [N_query, d] or [B, N_query, d]
    """
    if coeffs.ndim == 2:
        # Single trajectory
        M = coeffs.shape[0] - 1
        k = np.arange(0, M + 1, dtype=np.float32)
        
        # Orthonormal weights
        norm = np.ones(M + 1)
        norm[0] = 1.0 / np.sqrt(2)
        norm = norm * np.sqrt(2.0 / K_norm)
        
        # Basis: [M+1, N_query]
        basis = np.cos(np.pi * k[:, None] * times[None, :] / chunk_duration)
        
        # Decode: [M+1, d] * [M+1, 1] -> [M+1, d], then [M+1, d].T @ [M+1, N_query] -> [d, N_query] -> [N_query, d]
        return ((coeffs * norm[:, None]).T @ basis).T
    elif coeffs.ndim == 3:
        # Batch of trajectories
        B, Mp1, d = coeffs.shape
        M = Mp1 - 1
        k = np.arange(0, M + 1, dtype=np.float32)
        
        norm = np.ones(M + 1)
        norm[0] = 1.0 / np.sqrt(2)
        norm = norm * np.sqrt(2.0 / K_norm)
        
        # Basis: [M+1, N_query]
        if times.ndim == 1:
            times = np.broadcast_to(times[None, :], (B, len(times)))
        
        basis = np.cos(np.pi * k[None, :, None] * times[:, None, :] / chunk_duration)  # [B, M+1, N_query]
        
        # Decode: [B, M+1, d] * [M+1] -> [B, M+1, d], then einsum
        coeffs_normalized = coeffs * norm[None, :, None]
        return np.einsum('bmd,bmn->bnd', coeffs_normalized, basis)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {coeffs.shape}")


def sample_freq_weighted_noise(
    num_modes: int,
    num_dof: int,
    chunk_duration: float,
    batch_size: int,
    sigma: float = 1.0,
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """
    Sample frequency-weighted Gaussian noise for FM prior.
    
    Implements Eq. 3.4 from paper: ξ_k ~ N(0, σ²/(1+ω_k²))
    where ω_k = kπ/T
    
    This assigns smaller variance to high-frequency modes, consistent with
    the H^1 structure of smooth velocity trajectories.
    
    Args:
        num_modes: Number of Fourier modes M
        num_dof: Action dimension d
        chunk_duration: Duration T in seconds
        batch_size: Batch size B
        sigma: Base noise scale
        device: Device for computation
        
    Returns:
        noise: Frequency-weighted noise [B, M+1, d]
    """
    # Frequency values: ω_k = kπ/T
    k_indices = torch.arange(0, num_modes + 1, dtype=torch.float32, device=device)
    omega_k = k_indices * np.pi / chunk_duration  # [M+1]
    
    # Standard deviation for each mode: σ_k = σ / sqrt(1 + ω_k²)
    std_k = sigma / torch.sqrt(1.0 + omega_k ** 2)  # [M+1]
    
    # Sample standard Gaussian and scale by mode-specific std
    xi = torch.randn(batch_size, num_modes + 1, num_dof, device=device)
    return xi * std_k[None, :, None]


def estimate_num_modes(
    velocities: np.ndarray,
    energy_threshold: float = 0.95,
) -> int:
    """
    Estimate the number of Fourier modes M using energy-based criterion.
    
    Implements: M = min{m : Σ_{k=0}^m ||c_k||² / Σ_{k=0}^{K/2} ||c_k||² ≥ ε}
    
    This should be run once on the training dataset to determine M before training.
    
    Args:
        velocities: Velocity sequences [N_trajectories, K, d]
        energy_threshold: Energy threshold ε (default 0.95 = 95%)
        
    Returns:
        M: Number of modes to retain
    """
    N, K, d = velocities.shape
    
    # Compute DCT for all trajectories
    coeffs_full = dct(velocities, axis=1, norm='ortho')  # [N, K, d]
    
    # Compute energy per mode: ||c_k||² summed over DOF
    energy_per_mode = (coeffs_full ** 2).sum(axis=2)  # [N, K]
    
    # Average energy across trajectories
    mean_energy = energy_per_mode.mean(axis=0)  # [K]
    
    # Cumulative energy
    cumulative_energy = np.cumsum(mean_energy)
    total_energy = cumulative_energy[-1]
    
    # Find minimum M such that cumulative energy >= threshold * total
    M = int(np.argmax(cumulative_energy >= energy_threshold * total_energy))
    
    return M


def compute_velocity_from_position(
    positions: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    Compute velocity from position using finite difference.
    
    Uses forward difference: v[k] = (pos[k+1] - pos[k]) / dt
    For the last timestep, uses backward difference.
    
    Args:
        positions: Position sequence [batch_size, K, d]
        dt: Time step in seconds
        
    Returns:
        velocities: Velocity sequence [batch_size, K, d]
    """
    batch_size, K, d = positions.shape
    
    # Forward difference for all but last timestep
    velocities = torch.zeros_like(positions)
    velocities[:, :-1, :] = (positions[:, 1:, :] - positions[:, :-1, :]) / dt
    
    # Backward difference for last timestep
    velocities[:, -1, :] = velocities[:, -2, :]
    
    return velocities


def compute_velocity_from_position_np(
    positions: np.ndarray,
    dt: float,
) -> np.ndarray:
    """
    Compute velocity from position using finite difference (NumPy version).
    
    Args:
        positions: Position sequence [K, d] or [B, K, d]
        dt: Time step in seconds
        
    Returns:
        velocities: Velocity sequence [K, d] or [B, K, d]
    """
    if positions.ndim == 2:
        K, d = positions.shape
        velocities = np.zeros_like(positions)
        velocities[:-1, :] = (positions[1:, :] - positions[:-1, :]) / dt
        velocities[-1, :] = velocities[-2, :]
    elif positions.ndim == 3:
        B, K, d = positions.shape
        velocities = np.zeros_like(positions)
        velocities[:, :-1, :] = (positions[:, 1:, :] - positions[:, :-1, :]) / dt
        velocities[:, -1, :] = velocities[:, -2, :]
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {positions.shape}")
    
    return velocities
