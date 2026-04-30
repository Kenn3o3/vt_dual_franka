"""
MOTIF Diffusion Model

Implements the dual loss structure for MOTIF:
L = L_FM + α·L_vel

where:
- L_FM: Standard Flow Matching loss in coefficient space
- L_vel: Velocity supervision loss via Fourier decoding
"""

import hydra
import torch
from omegaconf import DictConfig
from typing import Dict, Union, Tuple

from movement_primitive_diffusion.models.base_inner_model import BaseInnerModel
from movement_primitive_diffusion.models.base_model import BaseModel
from movement_primitive_diffusion.models.scaling import Scaling


class MOTIFDiffusionModel(BaseModel):
    """
    MOTIF Diffusion Model with dual loss structure.
    
    Key differences from standard DiffusionModel:
    1. Operates in coefficient space (not position space)
    2. Adds velocity supervision loss L_vel
    3. Uses frequency-weighted noise prior
    """
    
    def __init__(
        self,
        inner_model_config: DictConfig,
        scaling_config: DictConfig,
        alpha_vel: float = 1.0,
        algorithm: str = 'diffusion',  # 'diffusion' or 'flow_matching'
        use_ot_sampler: bool = False,  # For Flow Matching
    ):
        """
        Initialize MOTIF Diffusion Model.
        
        Args:
            inner_model_config: Config for inner model (MOTIFTransformerInnerModel)
            scaling_config: Config for noise scaling
            alpha_vel: Weight for velocity supervision loss (default 1.0)
            algorithm: 'diffusion' or 'flow_matching'
            use_ot_sampler: Whether to use OT sampler for flow matching
        """
        super().__init__()
        self.inner_model: BaseInnerModel = hydra.utils.instantiate(inner_model_config)
        self.scaling: Scaling = hydra.utils.instantiate(scaling_config)
        self.alpha_vel = alpha_vel
        self.algorithm = algorithm
        self.use_ot_sampler = use_ot_sampler
        
    def loss(
        self,
        state: torch.Tensor,
        coeffs: torch.Tensor,
        sigma: torch.Tensor,
        extra_inputs: Dict,
        return_denoised: bool = False,
        return_loss_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict]]:
        """
        Compute MOTIF dual loss: L = L_main + α·L_vel
        where L_main is either diffusion or flow matching loss.
        
        Args:
            state: State observation [batch_size, t_obs, state_size]
            coeffs: Clean DCT coefficients [batch_size, num_modes+1, num_dof]
            sigma: Noise level [batch_size, 1] (or time t for flow matching)
            extra_inputs: Dictionary containing:
                - gt_velocities: Ground truth velocities [batch_size, K, num_dof]
                - physical_times: Physical times [batch_size, K]
                - gt_states: Demonstration states [batch_size, K, num_dof]
            return_denoised: Whether to return denoised/predicted coefficients
            return_loss_components: Whether to return individual loss components
            
        Returns:
            If return_denoised:
                (total_loss, coeffs_prediction)
            elif return_loss_components:
                (total_loss, loss_dict)
            else:
                total_loss
        """
        if self.algorithm == 'diffusion':
            return self._diffusion_loss(state, coeffs, sigma, extra_inputs, return_denoised, return_loss_components)
        elif self.algorithm == 'flow_matching':
            return self._flow_matching_loss(state, coeffs, sigma, extra_inputs, return_denoised, return_loss_components)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _diffusion_loss(
        self,
        state: torch.Tensor,
        coeffs: torch.Tensor,
        sigma: torch.Tensor,
        extra_inputs: Dict,
        return_denoised: bool = False,
        return_loss_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict]]:
        """Original diffusion loss with velocity supervision"""
        # Expand sigma to match coefficient dimensions
        for _ in range(coeffs.ndim - sigma.ndim):
            sigma = sigma.unsqueeze(-1)
        assert torch.all(sigma >= 0), "Sigma must be positive"
        
        # Sample noise (frequency-weighted for MOTIF)
        # Apply frequency weighting to noise BEFORE multiplying by sigma
        # ω_k = kπ/T, σ_k = σ / sqrt(1 + ω_k²)
        if hasattr(self.inner_model, 'motif_handler'):
            num_modes = self.inner_model.motif_handler.num_modes
            chunk_duration = self.inner_model.motif_handler.chunk_duration
            
            # Get actual number of modes from coeffs shape
            actual_num_modes = coeffs.shape[1] - 1  # coeffs is [B, M+1, d]
            
            k_indices = torch.arange(0, actual_num_modes + 1, dtype=torch.float32, device=coeffs.device)
            omega_k = k_indices * torch.pi / chunk_duration
            freq_weight = 1.0 / torch.sqrt(1.0 + omega_k ** 2)  # [M+1]
            
            # Sample standard Gaussian noise
            noise = torch.randn_like(coeffs)
            
            # Apply frequency weighting: [B, M+1, d] * [1, M+1, 1] * [B, 1, 1]
            noise = noise * freq_weight[None, :, None] * sigma
        else:
            # Fallback: standard noise
            noise = torch.randn_like(coeffs) * sigma
        
        # Forward process: add noise to clean coefficients
        noised_coeffs = coeffs + noise
        
        # Add t_diffusion to extra_inputs for state masking
        # During training, we sample various t values, so we use sigma as a proxy
        # At t=0 (sigma=0), state should be unmasked; at t>0, state should be masked
        extra_inputs['t_diffusion'] = (sigma.squeeze() > 1e-6).float()
        
        # Predict denoised coefficients
        denoised_coeffs = self.forward(state, noised_coeffs, sigma, extra_inputs)
        
        # ===== L_FM: Flow Matching loss in coefficient space =====
        # Standard L2 denoising loss
        loss_fm = (coeffs - denoised_coeffs).pow(2).flatten().mean()
        
        # ===== L_vel: Velocity supervision loss =====
        # Decode predicted coefficients to velocities at demonstration times
        gt_velocities = extra_inputs['gt_velocities']  # [B, K, d]
        physical_times = extra_inputs['physical_times']  # [B, K]
        
        # Decode at t=0 (clean coefficients) with demonstration states
        # Create a copy of extra_inputs with t_diffusion=0 for velocity supervision
        vel_extra_inputs = {
            'physical_times': physical_times,
            'gt_states': extra_inputs.get('gt_states', None),
            't_diffusion': torch.zeros(coeffs.shape[0], device=coeffs.device),  # t=0 for execution
        }
        
        # Get clean prediction at t=0
        # We need to query the model at t=0 with clean coefficients as input
        # This simulates the execution phase where t_diffusion=0
        with torch.no_grad():
            # Use clean coefficients as input (no noise)
            sigma_zero = torch.zeros_like(sigma)
            for _ in range(coeffs.ndim - sigma_zero.ndim):
                sigma_zero = sigma_zero.unsqueeze(-1)
            
        # Actually, we should use the denoised_coeffs directly for velocity supervision
        # Decode denoised coefficients to velocities
        if hasattr(self.inner_model, 'decode_to_velocity'):
            pred_velocities = self.inner_model.decode_to_velocity(denoised_coeffs, physical_times)
        else:
            # Fallback: use handler directly
            pred_velocities = self.inner_model.motif_handler.decode(denoised_coeffs, physical_times)
        
        # Velocity supervision loss: L2 in velocity space
        loss_vel = (pred_velocities - gt_velocities).pow(2).flatten().mean()
        
        # ===== Total loss =====
        total_loss = loss_fm + self.alpha_vel * loss_vel
        
        # Return based on flags
        if return_loss_components:
            loss_dict = {
                'loss_fm': loss_fm.item(),
                'loss_vel': loss_vel.item(),
                'total_loss': total_loss.item(),
            }
            if return_denoised:
                return total_loss, denoised_coeffs, loss_dict
            else:
                return total_loss, loss_dict
        elif return_denoised:
            return total_loss, denoised_coeffs
        else:
            return total_loss
    
    def forward(
        self,
        state: torch.Tensor,
        noised_coeffs: torch.Tensor,
        sigma: torch.Tensor,
        extra_inputs: Dict,
    ) -> torch.Tensor:
        """
        Forward pass: predict denoised coefficients.
        
        Args:
            state: State observation [batch_size, t_obs, state_size]
            noised_coeffs: Noised coefficients [batch_size, num_modes+1, num_dof]
            sigma: Noise level [batch_size, 1]
            extra_inputs: Extra inputs dictionary
            
        Returns:
            denoised_coeffs: Predicted clean coefficients [batch_size, num_modes+1, num_dof]
        """
        # Get scaling factors
        c_skip, self.c_out, c_in, c_noise = self.scaling(sigma)
        
        # Expand scaling factors to match coefficient dimensions [B, M+1, d]
        # c_skip, c_out, c_in are [B, 1], need to be [B, 1, 1] for broadcasting
        for _ in range(noised_coeffs.ndim - c_skip.ndim):
            c_skip = c_skip.unsqueeze(-1)
            self.c_out = self.c_out.unsqueeze(-1)
            c_in = c_in.unsqueeze(-1)
        
        # Compute denoised coefficients with skip connection
        inner_model_output = self.inner_model(state, c_in * noised_coeffs, c_noise, extra_inputs)
        
        # Debug: print shapes if mismatch detected
        if inner_model_output.shape != noised_coeffs.shape:
            print(f"[DEBUG] Shape mismatch in forward:")
            print(f"  noised_coeffs: {noised_coeffs.shape}")
            print(f"  inner_model_output: {inner_model_output.shape}")
            print(f"  c_skip: {c_skip.shape}")
            print(f"  c_out: {self.c_out.shape}")
            raise RuntimeError(f"Shape mismatch: inner_model_output {inner_model_output.shape} != noised_coeffs {noised_coeffs.shape}")
        
        denoised_coeffs = c_skip * noised_coeffs + self.c_out * inner_model_output
        
        return denoised_coeffs
    
    def _flow_matching_loss(
        self,
        state: torch.Tensor,
        c_1: torch.Tensor,
        t: torch.Tensor,
        extra_inputs: Dict,
        return_prediction: bool = False,
        return_loss_components: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, Dict]]:
        """
        Flow Matching loss with velocity supervision for MOTIF.
        
        Args:
            state: State observation
            c_1: Target coefficients (clean)
            t: Time parameter [batch_size, 1], in [0, 1]
            extra_inputs: Extra inputs including gt_velocities, physical_times
            return_prediction: Whether to return predicted c_1
            return_loss_components: Whether to return loss components
            
        Returns:
            loss or (loss, prediction) or (loss, loss_dict)
        """
        # Sample c_0 from frequency-weighted prior
        if hasattr(self.inner_model, 'motif_handler'):
            num_modes = self.inner_model.motif_handler.num_modes
            chunk_duration = self.inner_model.motif_handler.chunk_duration
            
            actual_num_modes = c_1.shape[1] - 1
            k_indices = torch.arange(0, actual_num_modes + 1, dtype=torch.float32, device=c_1.device)
            omega_k = k_indices * torch.pi / chunk_duration
            freq_weight = 1.0 / torch.sqrt(1.0 + omega_k ** 2)
            
            # Sample standard Gaussian and apply frequency weighting
            c_0 = torch.randn_like(c_1) * freq_weight[None, :, None]
        else:
            c_0 = torch.randn_like(c_1)
        
        # Optional: Use OT sampler
        if self.use_ot_sampler and c_1.shape[0] > 1:
            perm = torch.randperm(c_1.shape[0], device=c_1.device)
            c_0 = c_0[perm]
        
        # Expand t to match dimensions
        for _ in range(c_1.ndim - t.ndim):
            t = t.unsqueeze(-1)
        
        # Linear interpolation path: c_t = t*c_1 + (1-t)*c_0
        c_t = t * c_1 + (1 - t) * c_0
        
        # Target vector field: u_t = c_1 - c_0
        u_t = c_1 - c_0
        
        # Add t_diffusion to extra_inputs
        extra_inputs['t_diffusion'] = (t.squeeze() > 1e-6).float()
        
        # Predict vector field v_t
        v_t = self.forward(state, c_t, t, extra_inputs)
        
        # Flow matching loss: ||u_t - v_t||^2
        loss_fm = (u_t - v_t).pow(2).flatten().mean()
        
        # ===== L_vel: Velocity supervision loss =====
        # Reconstruct c_1 from v_t: c_1 = c_t + (1-t)*v_t
        predicted_c_1 = c_t + (1 - t) * v_t
        
        gt_velocities = extra_inputs['gt_velocities']
        physical_times = extra_inputs['physical_times']
        
        # Decode predicted coefficients to velocities
        if hasattr(self.inner_model, 'decode_to_velocity'):
            pred_velocities = self.inner_model.decode_to_velocity(predicted_c_1, physical_times)
        else:
            pred_velocities = self.inner_model.motif_handler.decode(predicted_c_1, physical_times)
        
        loss_vel = (pred_velocities - gt_velocities).pow(2).flatten().mean()
        
        # Total loss
        total_loss = loss_fm + self.alpha_vel * loss_vel
        
        # Return based on flags
        if return_loss_components:
            loss_dict = {
                'loss_fm': loss_fm.item(),
                'loss_vel': loss_vel.item(),
                'total_loss': total_loss.item(),
            }
            if return_prediction:
                return total_loss, predicted_c_1, loss_dict
            else:
                return total_loss, loss_dict
        elif return_prediction:
            return total_loss, predicted_c_1
        else:
            return total_loss
