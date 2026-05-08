"""
Minimal wrapper for official Freqpolicy code
"""
import torch
import torch.nn as nn
from movement_primitive_diffusion.models.freqpolicy_official.Freqpolicy import Freqpolicy


class FreqpolicyWrapper(nn.Module):
    """Wrapper that adapts official Freqpolicy to MPD framework."""
    
    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        t_pred: int,
        n_obs_steps: int,
        encoder_embed_dim: int = 128,
        encoder_depth: int = 3,
        decoder_embed_dim: int = 128,
        decoder_depth: int = 3,
        num_iter: int = 5,
        num_sampling_steps: int = 10,
        diffloss_d: int = 6,
        diffloss_w: int = 144,
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__()
        
        self.device = torch.device(device)
        
        # Create official Freqpolicy
        # Note: condition_dim is the dimension per observation step, NOT total
        self.freqpolicy = Freqpolicy(
            trajectory_dim=action_dim,
            horizon=t_pred,
            n_obs_steps=n_obs_steps,
            encoder_embed_dim=encoder_embed_dim,
            encoder_depth=encoder_depth,
            encoder_num_heads=max(1, encoder_embed_dim // 16),
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=max(1, decoder_embed_dim // 16),
            mask=True,
            condition_dim=state_dim,  # Dimension per observation step
            diffloss_d=diffloss_d,
            diffloss_w=diffloss_w,
            num_iter=num_iter,
            num_sampling_steps=str(num_sampling_steps),
            diffusion_batch_mul=1,
        ).to(self.device)
        
        # Store total condition dimension for input checking
        self.total_condition_dim = state_dim * n_obs_steps
    
    def forward(self, trajectory, observation, **kwargs):
        """Training forward."""
        B = trajectory.shape[0]
        # Ensure everything is on the same device
        trajectory = trajectory.to(self.device)
        observation = observation.to(self.device)
        # Ensure observation is 2D (B, state_dim * n_obs_steps)
        if observation.dim() == 3:  # (B, n_obs_steps, state_dim)
            observation = observation.reshape(B, -1)
        elif observation.dim() == 2:  # (B, state_dim) or (B, state_dim * n_obs_steps)
            if observation.shape[1] != self.total_condition_dim:
                # Need to expand: repeat the single observation for n_obs_steps
                observation = observation.unsqueeze(1).repeat(1, self.freqpolicy.n_obs_steps, 1).reshape(B, -1)
        return self.freqpolicy.forward(trajectory, observation, loss_weight=False)
    
    def sample(self, observation, num_samples=1):
        """Inference sampling."""
        B = observation.shape[0]
        # Ensure everything is on the same device
        observation = observation.to(self.device)
        # Ensure observation is 2D (B, state_dim * n_obs_steps)
        if observation.dim() == 3:  # (B, n_obs_steps, state_dim)
            observation = observation.reshape(B, -1)
        elif observation.dim() == 2:  # (B, state_dim) or (B, state_dim * n_obs_steps)
            if observation.shape[1] != self.total_condition_dim:
                # Need to expand
                observation = observation.unsqueeze(1).repeat(1, self.freqpolicy.n_obs_steps, 1).reshape(B, -1)
        if num_samples > 1:
            observation = observation.repeat_interleave(num_samples, dim=0)
        # Use sample_tokens_mask method from official Freqpolicy
        return self.freqpolicy.sample_tokens_mask(bsz=observation.shape[0], conditions=observation)
    
    def configure_optimizers(self, weight_decay=1e-3, learning_rate=1e-4, betas=(0.9, 0.95)):
        """Configure optimizer."""
        if hasattr(self.freqpolicy, 'configure_optimizers'):
            return self.freqpolicy.configure_optimizers(
                weight_decay=weight_decay, learning_rate=learning_rate, betas=betas
            )
        return torch.optim.AdamW(self.parameters(), lr=learning_rate, 
                                weight_decay=weight_decay, betas=betas)
