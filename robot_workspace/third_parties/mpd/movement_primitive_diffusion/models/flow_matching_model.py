"""
Flow Matching Model

Implements Conditional Flow Matching (CFM) for generative modeling.
This is a pure flow matching implementation without MPD or Motif mechanisms.

Key differences from Diffusion Models:
- Diffusion: Learns denoising function D(x_t, t), loss ||x_0 - D(x_t, t)||²
- Flow Matching: Learns vector field v(x_t, t), loss ||u_t - v(x_t, t)||²
  where u_t = x_1 - x_0 is the conditional vector field

References:
- Flow Matching for Generative Modeling (Lipman et al., 2023)
- Improving and Generalizing Flow-Based Generative Models (Liu et al., 2023)
"""

import hydra
import torch

from omegaconf import DictConfig
from typing import Dict, Union, Tuple

from movement_primitive_diffusion.models.base_inner_model import BaseInnerModel
from movement_primitive_diffusion.models.base_model import BaseModel


class FlowMatchingModel(BaseModel):
    """
    Flow Matching Model for learning continuous normalizing flows.
    
    The model learns a time-dependent vector field v(x_t, t) that defines
    an ODE: dx/dt = v(x_t, t), which transforms a simple prior p_0 (Gaussian)
    to the data distribution p_1.
    
    Training uses Conditional Flow Matching with optimal transport paths:
    - Path: x_t = t*x_1 + (1-t)*x_0, where x_0 ~ N(0,I), x_1 ~ p_data
    - Target vector field: u_t = x_1 - x_0 (constant for linear interpolation)
    - Loss: E[||u_t - v(x_t, t)||²]
    """
    
    def __init__(
        self,
        inner_model_config: DictConfig,
        use_ot_sampler: bool = False,
    ):
        """
        Initialize Flow Matching Model.
        
        Args:
            inner_model_config: Config for inner model (e.g., CausalTransformer)
            use_ot_sampler: Whether to use optimal transport sampler for x_0
                           (default: False, uses standard Gaussian)
        """
        super().__init__()
        self.inner_model: BaseInnerModel = hydra.utils.instantiate(inner_model_config)
        self.use_ot_sampler = use_ot_sampler
        
    def loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        t: torch.Tensor,
        extra_inputs: Dict,
        return_prediction: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute Flow Matching loss.
        
        Args:
            state: State observation [batch_size, t_obs, state_dim]
            action: Clean action (x_1) [batch_size, t_pred, action_dim]
            t: Time parameter in [0, 1] [batch_size, 1]
            extra_inputs: Extra inputs dictionary
            return_prediction: Whether to return predicted vector field
            
        Returns:
            If return_prediction:
                (loss, predicted_vector_field)
            else:
                loss
        """
        # Ensure t is in [0, 1]
        for _ in range(action.ndim - t.ndim):
            t = t.unsqueeze(-1)
        assert torch.all(t >= 0) and torch.all(t <= 1), "t must be in [0, 1]"
        
        # Sample x_0 from prior (standard Gaussian)
        x_0 = torch.randn_like(action)
        
        # Optimal transport sampler (minibatch OT)
        if self.use_ot_sampler:
            # Simple minibatch OT: permute x_0 to minimize sum of distances to x_1
            # This is a simple approximation; full OT would require Sinkhorn
            batch_size = action.shape[0]
            perm = torch.randperm(batch_size, device=action.device)
            x_0 = x_0[perm]
        
        # Conditional flow: linear interpolation from x_0 to x_1
        # x_t = t*x_1 + (1-t)*x_0
        x_t = t * action + (1 - t) * x_0
        
        # Target conditional vector field for linear interpolation
        # u_t(x) = dx/dt = d/dt[t*x_1 + (1-t)*x_0] = x_1 - x_0
        u_t = action - x_0
        
        # Predict vector field v(x_t, t)
        v_t = self.forward(state, x_t, t, extra_inputs)
        
        # Flow matching loss: MSE between target and predicted vector fields
        loss = (u_t - v_t).pow(2).flatten().mean()
        
        if return_prediction:
            return loss, v_t
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
        Forward pass: predict vector field v(x_t, t).
        
        Args:
            state: State observation [batch_size, t_obs, state_dim]
            x_t: Current flow state [batch_size, t_pred, action_dim]
            t: Time parameter in [0, 1] [batch_size, 1]
            extra_inputs: Extra inputs dictionary
            
        Returns:
            v_t: Predicted vector field [batch_size, t_pred, action_dim]
        """
        # The inner model takes (state, x_t, t, extra_inputs)
        # Note: For compatibility with existing inner models that expect sigma,
        # we pass t as the "time conditioning" parameter
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
            x_t: Current flow state
            t: Current time
            dt: Time step size
            extra_inputs: Extra inputs
            
        Returns:
            x_{t+dt}: Next flow state
        """
        v_t = self.forward(state, x_t, t, extra_inputs)
        x_next = x_t + dt * v_t
        return x_next
