import hydra
import torch

from omegaconf import DictConfig
from typing import Dict, Union, Tuple

from movement_primitive_diffusion.models.base_inner_model import BaseInnerModel
from movement_primitive_diffusion.models.base_model import BaseModel
from movement_primitive_diffusion.models.scaling import Scaling


class DiffusionModel(BaseModel):
    def __init__(
        self,
        inner_model_config: DictConfig,
        scaling_config: DictConfig,
        algorithm: str = 'diffusion',  # 'diffusion' or 'flow_matching'
        use_ot_sampler: bool = False,  # For Flow Matching: use optimal transport sampler
    ):
        super().__init__()
        self.inner_model: BaseInnerModel = hydra.utils.instantiate(inner_model_config)
        self.scaling: Scaling = hydra.utils.instantiate(scaling_config)
        self.algorithm = algorithm
        self.use_ot_sampler = use_ot_sampler

    def loss(self, state: torch.Tensor, action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict, return_denoised: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Computes the loss of the model with the current mini-batch.
        Supports both diffusion and flow matching algorithms.
        
        Args:
            state: state tensor [batch_size, obs_dim]
            action: Action tensor [batch_size, action_dim] (x_1 for flow matching)
            sigma: Noise level tensor [batch_size, 1] (or time t for flow matching)
            extra_inputs: Extra inputs dictionary

        Returns:
            loss or (loss, denoised_action/predicted_x1)
        """
        if self.algorithm == 'diffusion':
            return self._diffusion_loss(state, action, sigma, extra_inputs, return_denoised)
        elif self.algorithm == 'flow_matching':
            return self._flow_matching_loss(state, action, sigma, extra_inputs, return_denoised)
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def _diffusion_loss(self, state: torch.Tensor, action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict, return_denoised: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Original diffusion loss"""
        # Noise is sampled from a normal distribution with mean 0 and std sigma
        for _ in range(action.ndim - sigma.ndim):
            sigma = sigma.unsqueeze(-1)
        assert torch.all(sigma >= 0), "Sigma must be positive"

        # Noise is first drawn from a normal distribution with mean 0 and std 1, then scaled by sigma (the desired std)
        noise = torch.randn_like(action) * sigma

        # Forward process of diffusion probabilistic model
        noised_action = action + noise

        # Predict the denoised action
        denoised_action = self.forward(state, noised_action, sigma, extra_inputs)

        # Compute the L2 denoising error
        loss = (action - denoised_action).pow(2).flatten().mean()

        if return_denoised:
            return loss, denoised_action
        else:
            return loss
    
    def _flow_matching_loss(self, state: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor, extra_inputs: Dict, return_prediction: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Flow Matching loss using Conditional Flow Matching (CFM).
        
        Args:
            state: State observation
            x_1: Target data (clean action/params)
            t: Time parameter [batch_size, 1], should be in [0, 1]
            extra_inputs: Extra inputs
            return_prediction: Whether to return predicted x_1
            
        Returns:
            loss or (loss, predicted_x_1)
        """
        # Sample x_0 from prior N(0, I)
        x_0 = torch.randn_like(x_1)
        
        # Optional: Use OT sampler (simple minibatch permutation)
        if self.use_ot_sampler and x_1.shape[0] > 1:
            perm = torch.randperm(x_1.shape[0], device=x_1.device)
            x_0 = x_0[perm]
        
        # Expand t to match dimensions
        for _ in range(x_1.ndim - t.ndim):
            t = t.unsqueeze(-1)
        
        # Linear interpolation path: x_t = t*x_1 + (1-t)*x_0
        x_t = t * x_1 + (1 - t) * x_0
        
        # Target vector field: u_t = x_1 - x_0
        u_t = x_1 - x_0
        
        # Predict vector field v_t
        # For flow matching, we treat sigma/t as time parameter
        v_t = self.forward(state, x_t, t, extra_inputs)
        
        # Flow matching loss: ||u_t - v_t||^2
        loss = (u_t - v_t).pow(2).flatten().mean()
        
        if return_prediction:
            # Reconstruct x_1 from v_t: x_1 = x_t + (1-t)*v_t
            predicted_x_1 = x_t + (1 - t) * v_t
            return loss, predicted_x_1
        else:
            return loss

    def forward(self, state: torch.Tensor, noised_action: torch.Tensor, sigma: torch.Tensor, extra_inputs: Dict) -> torch.Tensor:
        """
        Forward pass of the model, applying the noise levels to the input
        Then passing it through the inner model
        Args:
            state: state tensor [batch_size, obs_dim]
            noised_action: Action tensor [batch_size, action_dim]
            sigma: Noise level tensor [batch_size, 1]
            extra_inputs: Extra inputs dictionary

        Returns:
            denoised_action: Denoised action tensor [batch_size, action_dim]

        """
        # Get scaling factors and store c_out to correctly scale the loss
        c_skip, self.c_out, c_in, c_noise = self.scaling(sigma)

        # Compute the denoised action by first passing the (scaled) noised action through the inner model and then applying the scaling factors (including skip connection)
        inner_model_output = self.inner_model(state, c_in * noised_action, c_noise, extra_inputs)
        denoised_action = c_skip * noised_action + self.c_out * inner_model_output

        return denoised_action
