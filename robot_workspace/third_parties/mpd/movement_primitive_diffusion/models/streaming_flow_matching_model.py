"""
Streaming Flow Matching Model

Implements Streaming Flow Policy with extended state space (a(t), z(t)).
Based on "Streaming Flow Policy" - extends action space with latent noise variable.

Key features:
- Extended state space: x = (a, z) where a is action and z is latent variable
- z starts from N(0, 1) and drifts towards trajectory information
- Conditional flow with Gaussian tubes: σ₀ at t=0, σ₁ at t=1
- Flow equations:
  * a(t) = ξ(t) + (a₀ - ξ(0))exp(-kt) + σᵣtz₀
  * z(t) = (1 - (1-σ₁)t)z₀ + tξ(t)

References:
- Streaming Flow Policy paper
- Flow Matching for Generative Modeling (Lipman et al., 2023)
"""

import hydra
import torch
import numpy as np

from omegaconf import DictConfig
from typing import Dict, Union, Tuple

from movement_primitive_diffusion.models.base_inner_model import BaseInnerModel
from movement_primitive_diffusion.models.base_model import BaseModel


class StreamingFlowMatchingModel(BaseModel):
    """
    Streaming Flow Matching Model with extended state space.
    
    The model learns a vector field v(a, z, t) for the extended state (a, z):
    - a: action trajectory
    - z: latent noise variable that drifts towards trajectory information
    
    Training:
    - Sample t uniformly from [0, 1]
    - Compute flow paths for both a(t) and z(t)
    - Learn vector field to match target velocities (va, vz)
    
    Inference:
    - Start from a₀ ~ N(ξ(0), σ₀²) and z₀ ~ N(0, 1)
    - Integrate ODE for extended state
    - Extract action trajectory a(t)
    """
    
    def __init__(
        self,
        inner_model_config: DictConfig,
        sigma_0: float = 0.1,
        sigma_1: float = 0.1,
        k: float = 0.0,
    ):
        """
        Initialize Streaming Flow Matching Model.
        
        Args:
            inner_model_config: Config for inner model (e.g., CausalTransformer)
            sigma_0: Standard deviation of Gaussian tube at t=0
            sigma_1: Standard deviation of Gaussian tube at t=1  
            k: Decay rate for a(t) (default 0.0 for no decay)
        """
        super().__init__()
        
        # Update action_size in inner_model_config to account for extended state
        # The model will process (a, z) concatenated, so double the action dimension
        if hasattr(inner_model_config, "action_size"):
            # Store original action size
            self.action_dim = inner_model_config.action_size
            # Double it for (a, z)
            inner_model_config.action_size = inner_model_config.action_size * 2
        else:
            raise ValueError("inner_model_config must have action_size attribute")
            
        self.inner_model: BaseInnerModel = hydra.utils.instantiate(inner_model_config)
        
        # Flow parameters
        self.sigma_0 = sigma_0
        self.sigma_1 = sigma_1
        self.k = k
        
        # Residual standard deviation: σᵣ = √(σ₁² - σ₀²exp(-2k))
        assert 0 <= sigma_0 * np.exp(-k) <= sigma_1, f"σ₁ ({sigma_1}) is too small relative to σ₀ ({sigma_0}) with k={k}"
        self.sigma_r = np.sqrt(sigma_1**2 - sigma_0**2 * np.exp(-2 * k))
        
    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        t: torch.Tensor,
        extra_inputs: Dict,
        return_prediction: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute Streaming Flow Matching loss.
        
        Following the original sfps.py implementation where:
        - We sample ONE time point t for the entire trajectory
        - The trajectory ξ is treated as a piecewise linear interpolation
        - We compute ξ(t) and ξ̇(t) by interpolating the action sequence
        
        Args:
            state: State observation [batch_size, t_obs, state_dim]
            action: Clean action trajectory (demonstration) [batch_size, t_pred, action_dim]
            t: Time parameter in [0, 1] [batch_size, 1] or [batch_size]
            extra_inputs: Extra inputs dictionary
            return_prediction: Whether to return predicted vector field
            
        Returns:
            If return_prediction:
                (loss, predicted_vector_field)
            else:
                loss
        """
        # Ensure t is proper shape
        if t.ndim == 1:
            t = t.unsqueeze(-1)  # [batch_size, 1]
                
        assert torch.all(t >= 0) and torch.all(t <= 1), "t must be in [0, 1]"
        
        batch_size = action.shape[0]
        t_pred = action.shape[1]
        action_dim = action.shape[2]
        device = action.device
        
        # Interpolate trajectory at time t
        # action is shape [batch_size, t_pred, action_dim]
        # We need to get ξ(t) by linear interpolation
        # t=0 corresponds to action[:, 0, :], t=1 corresponds to action[:, -1, :]
        
        t_float = t.squeeze(-1)  # [batch_size]
        # Map t from [0,1] to indices in [0, t_pred-1]
        t_indices = t_float * (t_pred - 1)  # [batch_size]
        
        # Get integer parts and fractional parts
        t_floor = torch.floor(t_indices).long()  # [batch_size]
        t_ceil = torch.ceil(t_indices).long()  # [batch_size]
        t_frac = t_indices - t_floor.float()  # [batch_size]
        
        # Clamp indices
        t_floor = torch.clamp(t_floor, 0, t_pred - 1)
        t_ceil = torch.clamp(t_ceil, 0, t_pred - 1)
        
        # Linear interpolation for ξ(t)
        action_floor = action[torch.arange(batch_size), t_floor, :]  # [batch_size, action_dim]
        action_ceil = action[torch.arange(batch_size), t_ceil, :]  # [batch_size, action_dim]
        xi_t = action_floor + t_frac.unsqueeze(-1) * (action_ceil - action_floor)  # [batch_size, action_dim]
        
        # Compute ξ̇(t) - derivative using finite difference
        # For a piecewise linear trajectory, the derivative is constant between points
        if t_pred > 1:
            xi_dot_t = action_ceil - action_floor  # [batch_size, action_dim]
            # Normalize by time step (each segment spans 1/(t_pred-1) of the time)
            xi_dot_t = xi_dot_t * (t_pred - 1)
        else:
            xi_dot_t = torch.zeros_like(xi_t)
        
        # Sample z₀ ~ N(0, 1)
        z_0 = torch.randn(batch_size, action_dim, device=device)  # [batch_size, action_dim]
        
        # Sample ε for a₀ ~ N(ξ(0), σ₀²)  
        # Note: Original code uses ε_a0 ~ N(0, σ₀²)
        epsilon_a0 = self.sigma_0 * torch.randn(batch_size, action_dim, device=device)  # [batch_size, action_dim]
        
        # Compute flow state at time t
        # Simplified formula (k=0): a(t) = ξ(t) + ε_a0 + σᵣt·z₀
        t_scalar = t.squeeze(-1).unsqueeze(-1)  # [batch_size, 1]
        a_t = xi_t + epsilon_a0 + self.sigma_r * t_scalar * z_0  # [batch_size, action_dim]
        
        # z(t) = (1 - (1-σ₁)t)z₀ + tξ(t)
        z_t = (1 - (1 - self.sigma_1) * t_scalar) * z_0 + t_scalar * xi_t  # [batch_size, action_dim]
        
        # Compute target velocity field
        # va = ξ̇(t) + σᵣz₀ (simplified for k=0)
        va_target = xi_dot_t + self.sigma_r * z_0  # [batch_size, action_dim]
        
        # vz = ξ(t) + tξ̇(t) - (1-σ₁)z₀
        vz_target = xi_t + t_scalar * xi_dot_t - (1 - self.sigma_1) * z_0  # [batch_size, action_dim]
        
        # Reshape for model: add time dimension back
        # The model expects [batch_size, t_pred, action_dim]
        # But we're only predicting for ONE time point, so we unsqueeze
        a_t = a_t.unsqueeze(1)  # [batch_size, 1, action_dim]
        z_t = z_t.unsqueeze(1)  # [batch_size, 1, action_dim]
        va_target = va_target.unsqueeze(1)  # [batch_size, 1, action_dim]
        vz_target = vz_target.unsqueeze(1)  # [batch_size, 1, action_dim]
        
        # Concatenate (a, z) for model input
        x_t = torch.cat([a_t, z_t], dim=-1)  # [batch_size, 1, 2*action_dim]
        v_target = torch.cat([va_target, vz_target], dim=-1)  # [batch_size, 1, 2*action_dim]
        
        # Predict vector field v(a, z, t)
        v_pred = self.forward(state, x_t, t, extra_inputs)
        
        # Flow matching loss: MSE between target and predicted vector fields
        loss = (v_target - v_pred).pow(2).mean()
        
        if return_prediction:
            return loss, v_pred
        else:
            return loss
    
    def forward(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        extra_inputs: Dict,
    ) -> torch.Tensor:
        """
        Forward pass: predict vector field v(a, z, t).
        
        Args:
            state: State observation [batch_size, t_obs, state_dim]
            x_t: Current extended flow state (a, z) [batch_size, t_pred, 2*action_dim]
            t: Time parameter in [0, 1] [batch_size, 1]
            extra_inputs: Extra inputs dictionary
            
        Returns:
            v_t: Predicted vector field (va, vz) [batch_size, t_pred, 2*action_dim]
        """
        # The inner model takes (state, x_t, t, extra_inputs)
        # x_t contains concatenated (a, z)
        v_t = self.inner_model(state, x_t, t, extra_inputs)
        
        return v_t
    
    def sample_ode_step(
        self,
        state: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        extra_inputs: Dict,
    ) -> torch.Tensor:
        """
        Single ODE integration step using Euler method.
        
        Args:
            state: State observation
            x_t: Current extended flow state (a, z)
            t: Current time
            dt: Time step size
            extra_inputs: Extra inputs
            
        Returns:
            x_{t+dt}: Next extended flow state (a, z)
        """
        v_t = self.forward(state, x_t, t, extra_inputs)
        x_next = x_t + dt * v_t
        return x_next
    
    def extract_action(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Extract action trajectory from extended state.
        
        Args:
            x_t: Extended state (a, z) [batch_size, 1, 2*action_dim] or [batch_size, 2*action_dim]
            
        Returns:
            a: Action trajectory [batch_size, 1, action_dim] or [batch_size, action_dim]
        """
        # First half is action, second half is latent
        # x_t shape: [..., 2*action_dim], we want [..., action_dim]
        return x_t[..., :self.action_dim]
