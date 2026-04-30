"""
MOTIF Agent

Implements Flow Matching in coefficient space with:
- Frequency-weighted noise sampling
- Coefficient-to-action decoding
- Dual loss training (L_FM + L_vel)
"""

import torch
from typing import Dict, Tuple

from movement_primitive_diffusion.agents.diffusion_agent import DiffusionAgent


class MOTIFAgent(DiffusionAgent):
    """
    MOTIF Agent for training and inference.
    
    Key differences from ParameterSpaceDiffusionAgent:
    - Operates in DCT coefficient space
    - Uses frequency-weighted noise prior
    - Supports dual loss training
    - Decodes coefficients to velocities/positions
    """
    
    @torch.no_grad()
    def predict(self, observation: Dict, extra_inputs: Dict) -> torch.Tensor:
        """
        Predict action sequence from observation.
        
        Args:
            observation: Observation dictionary
            extra_inputs: Extra inputs including initial_position, physical_times
            
        Returns:
            action: Predicted position sequence [batch_size, traj_steps, num_dof]
        """
        # Load EMA weights if using EMA
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_encoder.store(self.encoder.parameters())
            self.ema_model.copy_to(self.model.parameters())
            self.ema_encoder.copy_to(self.encoder.parameters())
        
        # Set models to eval mode
        self.model.eval()
        self.encoder.eval()
        
        batch_size = observation[list(observation.keys())[0]].shape[0]
        
        # Encode observation
        state = self.encoder(observation)
        
        # Get noise schedule
        sigmas = self.noise_scheduler.get_sigmas(self.diffusion_steps).to(self.device)
        
        # Sample initial noisy coefficients with frequency weighting
        # Get coefficient shape from process_batch
        coeff_shape = self.process_batch.coefficient_shape  # (M+1, d)
        
        # Sample frequency-weighted noise
        if hasattr(self.process_batch, 'motif_handler'):
            from movement_primitive_diffusion.utils.motif_utils import sample_freq_weighted_noise
            
            num_modes = self.process_batch.motif_handler.num_modes
            num_dof = self.process_batch.motif_handler.num_dof
            chunk_duration = self.process_batch.motif_handler.chunk_duration
            
            noised_coeffs = sample_freq_weighted_noise(
                num_modes=num_modes,
                num_dof=num_dof,
                chunk_duration=chunk_duration,
                batch_size=batch_size,
                sigma=self.sigma_max,
                device=self.device,
            )
        else:
            # Fallback to standard Gaussian
            noised_coeffs = torch.randn((batch_size, *coeff_shape), device=self.device) * self.sigma_max
        
        # Add current state for state-conditioned query (at inference, t=0)
        if 'initial_position' in extra_inputs:
            extra_inputs['current_state'] = extra_inputs['initial_position']
        
        # FM denoising in coefficient space
        denoised_coeffs = self.sampler.sample(
            model=self.model,
            state=state,
            action=noised_coeffs,
            sigmas=sigmas,
            extra_inputs=extra_inputs,
        )
        
        # Decode coefficients to action (position) sequence
        action = self.process_batch.coefficients_to_action(denoised_coeffs, extra_inputs)
        
        # Restore original weights if using EMA
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())
        
        return action
    
    @torch.no_grad()
    def evaluate(self, batch: Dict) -> Tuple[float, float, float]:
        """
        Evaluate model on a batch.
        
        Args:
            batch: Mini batch dictionary
            
        Returns:
            Tuple of (loss, start_point_deviation, end_point_deviation)
        """
        # Load EMA weights if using EMA
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_encoder.store(self.encoder.parameters())
            self.ema_model.copy_to(self.model.parameters())
            self.ema_encoder.copy_to(self.encoder.parameters())
        
        # Set models to eval mode
        self.model.eval()
        self.encoder.eval()
        
        # Process batch to get coefficients, observation, and extra inputs
        coeffs, observation, extra_inputs = self.process_batch(batch)
        
        # Sample noise level
        sigma = self.sigma_distribution.sample(shape=coeffs.shape[0]).to(self.device)
        
        with torch.no_grad():
            # Encode observation
            state = self.encoder(observation)
            
            # Compute loss and get denoised coefficients
            if hasattr(self.model, 'loss'):
                # MOTIFDiffusionModel returns dual loss
                eval_loss, denoised_coeffs = self.model.loss(
                    state, coeffs, sigma, extra_inputs, return_denoised=True
                )
            else:
                # Fallback to standard loss
                eval_loss = self.model.loss(state, coeffs, sigma, extra_inputs)
                # Get denoised coefficients via forward pass
                denoised_coeffs = self.model.forward(state, coeffs, sigma, extra_inputs)
            
            # Decode to action sequences
            denoised_action = self.process_batch.coefficients_to_action(denoised_coeffs, extra_inputs)
            action = extra_inputs["action"]
            
            # Calculate L2 error at start and end points
            start_point_deviation = torch.linalg.norm(
                action[:, 0, :] - denoised_action[:, 0, :], dim=-1
            ).mean()
            end_point_deviation = torch.linalg.norm(
                action[:, -1, :] - denoised_action[:, -1, :], dim=-1
            ).mean()
        
        # Restore original weights if using EMA
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())
        
        return eval_loss.item(), start_point_deviation.item(), end_point_deviation.item()
    
    def train_step(self, batch: Dict) -> float:
        """
        Perform one training step.
        
        Args:
            batch: Mini batch dictionary
            
        Returns:
            loss: Training loss value
        """
        # Set models to train mode
        self.model.train()
        self.encoder.train()
        
        # Process batch
        coeffs, observation, extra_inputs = self.process_batch(batch)
        
        # Sample noise level
        sigma = self.sigma_distribution.sample(shape=coeffs.shape[0]).to(self.device)
        
        # Encode observation
        state = self.encoder(observation)
        
        # Compute loss (dual loss for MOTIF)
        if hasattr(self.model, 'loss'):
            loss = self.model.loss(state, coeffs, sigma, extra_inputs)
        else:
            # Fallback
            loss = self.model.forward(state, coeffs, sigma, extra_inputs)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping if configured
        if hasattr(self, 'max_grad_norm') and self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + list(self.encoder.parameters()),
                self.max_grad_norm
            )
        
        self.optimizer.step()
        
        # Update EMA if using (EMA is handled by BaseAgent)
        if self.use_ema and hasattr(self, 'ema_model') and self.ema_model is not None:
            self.ema_model.step(self.model.parameters())
            if hasattr(self, 'ema_encoder') and self.ema_encoder is not None:
                self.ema_encoder.step(self.encoder.parameters())
        
        # Update learning rate scheduler
        self.lr_scheduler.step()
        
        return loss.item()
