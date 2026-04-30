"""
Flow Matching Agent

Implements training and inference for Flow Matching models.
This agent handles the ODE integration for sampling and the training loop
for learning the vector field.

Key differences from Diffusion Agent:
- Training: samples t ~ Uniform(0, 1) instead of sigma from a distribution
- Inference: ODE integration from t=0 to t=1 instead of denoising from high to low sigma
- Loss: Flow matching loss ||u_t - v(x_t, t)||² instead of denoising loss
"""

from pathlib import Path
import hydra
import torch

from omegaconf import DictConfig
from typing import Dict, Tuple, Union, Optional

from movement_primitive_diffusion.agents.base_agent import BaseAgent


class FlowMatchingAgent(BaseAgent):
    """
    Agent for training and inference with Flow Matching models.
    
    Training:
    - Sample t uniformly from [0, 1]
    - Compute flow path: x_t = t*x_1 + (1-t)*x_0
    - Learn vector field v(x_t, t) to match u_t = x_1 - x_0
    
    Inference:
    - Start from x_0 ~ N(0, I)
    - Integrate ODE: dx/dt = v(x_t, t) from t=0 to t=1
    - Use Euler or higher-order ODE solvers
    """
    
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        lr_scheduler_config: DictConfig,
        encoder_config: DictConfig,
        process_batch_config: DictConfig,
        t_obs: int,
        predict_past: bool,
        num_integration_steps: int = 50,
        solver: str = "euler",
        ema_config: DictConfig = None,
        use_ema: bool = False,
        device: Union[str, torch.device] = "cpu",
        special_optimizer_function: bool = False,
        special_optimizer_config: Optional[DictConfig] = None,
    ):
        """
        Initialize Flow Matching Agent.
        
        Args:
            model_config: Configuration for the flow matching model
            optimizer_config: Configuration for optimizer
            lr_scheduler_config: Configuration for learning rate scheduler
            encoder_config: Configuration for state encoder
            process_batch_config: Configuration for batch processing
            t_obs: Number of observation timesteps
            predict_past: Whether to predict past actions
            num_integration_steps: Number of ODE integration steps for inference
            solver: ODE solver type ("euler", "heun", "rk4")
            ema_config: Configuration for EMA
            use_ema: Whether to use exponential moving average
            device: Device to run on
            special_optimizer_function: Whether to use special optimizer setup
            special_optimizer_config: Configuration for special optimizer
        """
        super().__init__(
            process_batch_config=process_batch_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            encoder_config=encoder_config,
            ema_config=ema_config,
            use_ema=use_ema,
            device=device,
            special_optimizer_function=special_optimizer_function,
            special_optimizer_config=special_optimizer_config,
        )
        
        self.t_obs = t_obs
        self.predict_past = predict_past
        self.num_integration_steps = num_integration_steps
        self.solver = solver
        
    def train_step(self, batch: Dict) -> float:
        """
        Execute a single training step.
        
        Args:
            batch: Mini batch dictionary
            
        Returns:
            Training loss value
        """
        # Process batch to get observation, action and extra inputs
        action, observation, extra_inputs = self.process_batch(batch)
        
        # Set model to train mode
        self.model.train()
        self.encoder.train()
        
        # Encode observation
        state = self.encoder(observation)
        
        # Sample time t uniformly from [0, 1]
        # Shape: (batch_size, 1)
        t = torch.rand(action.shape[0], 1, device=self.device)
        
        # Compute flow matching loss
        loss = self.model.loss(state, action, t, extra_inputs)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        
        # Update EMA weights
        if self.use_ema:
            self.ema_model.step(self.model.parameters())
            self.ema_encoder.step(self.encoder.parameters())
        
        return loss.item()
    
    @torch.no_grad()
    def predict(self, observation: Dict, extra_inputs: Dict) -> torch.Tensor:
        """
        Generate predictions by integrating the learned ODE.
        
        Args:
            observation: Observation dictionary
            extra_inputs: Extra inputs dictionary
            
        Returns:
            Generated actions
        """
        # Load EMA weights if using EMA
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_encoder.store(self.encoder.parameters())
            self.ema_model.copy_to(self.model.parameters())
            self.ema_encoder.copy_to(self.encoder.parameters())
        
        # Set model to eval mode
        self.model.eval()
        self.encoder.eval()
        
        batch_size = observation[list(observation.keys())[0]].shape[0]
        
        # Encode state
        state = self.encoder(observation)
        
        # Initialize from prior: x_0 ~ N(0, I)
        x_t = torch.randn((batch_size, *self.process_batch.action_shape), device=self.device)
        
        # Integrate ODE from t=0 to t=1
        dt = 1.0 / self.num_integration_steps
        
        if self.solver == "euler":
            # Euler method: x_{t+dt} = x_t + dt * v(x_t, t)
            for step in range(self.num_integration_steps):
                t = torch.full((batch_size, 1), step * dt, device=self.device)
                v_t = self.model.forward(state, x_t, t, extra_inputs)
                x_t = x_t + dt * v_t
                
        elif self.solver == "heun":
            # Heun's method (2nd order Runge-Kutta)
            for step in range(self.num_integration_steps):
                t = torch.full((batch_size, 1), step * dt, device=self.device)
                t_next = torch.full((batch_size, 1), (step + 1) * dt, device=self.device)
                
                # Predictor step
                v_t = self.model.forward(state, x_t, t, extra_inputs)
                x_pred = x_t + dt * v_t
                
                # Corrector step
                v_next = self.model.forward(state, x_pred, t_next, extra_inputs)
                x_t = x_t + dt * (v_t + v_next) / 2
                
        elif self.solver == "rk4":
            # 4th order Runge-Kutta
            for step in range(self.num_integration_steps):
                t = torch.full((batch_size, 1), step * dt, device=self.device)
                t_mid = torch.full((batch_size, 1), (step + 0.5) * dt, device=self.device)
                t_next = torch.full((batch_size, 1), (step + 1) * dt, device=self.device)
                
                k1 = self.model.forward(state, x_t, t, extra_inputs)
                k2 = self.model.forward(state, x_t + dt * k1 / 2, t_mid, extra_inputs)
                k3 = self.model.forward(state, x_t + dt * k2 / 2, t_mid, extra_inputs)
                k4 = self.model.forward(state, x_t + dt * k3, t_next, extra_inputs)
                
                x_t = x_t + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        # Restore original weights if using EMA
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())
        
        # Handle predict_past option
        if self.predict_past:
            return x_t[:, self.t_obs-1:, :]
        
        return x_t
    
    @torch.no_grad()
    def evaluate(self, batch: Dict) -> Tuple[float, float, float]:
        """
        Evaluate the model on a batch of data.
        
        Args:
            batch: Mini batch dictionary
            
        Returns:
            Tuple of (eval_loss, start_point_deviation, end_point_deviation)
        """
        # Load EMA weights if using EMA
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_encoder.store(self.encoder.parameters())
            self.ema_model.copy_to(self.model.parameters())
            self.ema_encoder.copy_to(self.encoder.parameters())
        
        # Set model to eval mode
        self.model.eval()
        self.encoder.eval()
        
        # Process batch
        action, observation, extra_inputs = self.process_batch(batch)
        
        # Encode observation
        state = self.encoder(observation)
        
        # Sample time t uniformly
        t = torch.rand(action.shape[0], 1, device=self.device)
        
        # Compute loss and get prediction
        eval_loss, v_pred = self.model.loss(state, action, t, extra_inputs, return_prediction=True)
        
        # Reconstruct x_t and compute denoised action
        # x_t = t*x_1 + (1-t)*x_0, where x_0 ~ N(0,I)
        # For evaluation, we approximate by using the predicted vector field
        # This is not perfect but gives a sense of reconstruction quality
        for _ in range(action.ndim - t.ndim):
            t = t.unsqueeze(-1)
        x_0 = torch.randn_like(action)
        x_t = t * action + (1 - t) * x_0
        
        # Approximate denoised action: x_1 ≈ x_t + (1-t) * v_pred
        # This is based on v = (x_1 - x_0) and x_t = t*x_1 + (1-t)*x_0
        denoised_action = x_t + (1 - t) * v_pred
        
        # Calculate L2 error of first and last action
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
    
    def load_pretrained(self, path: Union[str, Path]):
        """
        Load a pretrained model.
        
        Args:
            path: Path to the pretrained model checkpoint
        """
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict["model"])
        self.encoder.load_state_dict(state_dict["encoder"])
        if self.use_ema:
            self.ema_model.load_state_dict(state_dict["ema_model"])
            self.ema_encoder.load_state_dict(state_dict["ema_encoder"])
        
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "lr_scheduler" in state_dict:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
    
    def save_model(
        self,
        path: Union[str, Path],
        save_optimizer: bool = False,
        save_lr_scheduler: bool = False
    ):
        """
        Save the current model.
        
        Args:
            path: Path to save the model
            save_optimizer: Whether to save optimizer state
            save_lr_scheduler: Whether to save lr_scheduler state
        """
        state_dict = {
            "model": self.model.state_dict(),
            "encoder": self.encoder.state_dict()
        }
        
        if self.use_ema:
            state_dict["ema_model"] = self.ema_model.state_dict()
            state_dict["ema_encoder"] = self.ema_encoder.state_dict()
        
        if save_optimizer:
            state_dict["optimizer"] = self.optimizer.state_dict()
        if save_lr_scheduler:
            state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()
        
        torch.save(state_dict, path)
