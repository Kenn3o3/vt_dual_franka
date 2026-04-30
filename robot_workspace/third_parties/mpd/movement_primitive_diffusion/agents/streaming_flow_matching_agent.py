"""
Streaming Flow Matching Agent

Implements training and inference for Streaming Flow Matching models.
This agent handles the ODE integration for extended state space (a, z).

Key differences from standard Flow Matching:
- Extended state space: (a, z) where z is latent variable
- z starts from N(0, 1) and drifts towards trajectory
- Initial state: a₀ ~ N(ξ(0), σ₀²), z₀ ~ N(0, 1)
- Returns only action trajectory a(t), discarding z(t)
"""

from pathlib import Path
import hydra
import torch

from omegaconf import DictConfig
from typing import Dict, Tuple, Union, Optional

from movement_primitive_diffusion.agents.base_agent import BaseAgent


class StreamingFlowMatchingAgent(BaseAgent):
    """
    Agent for training and inference with Streaming Flow Matching models.
    
    Training:
    - Sample t uniformly from [0, 1]
    - Compute flow paths for extended state (a(t), z(t))
    - Learn vector field v(a, z, t) to match target velocities
    
    Inference:
    - Start from a₀ ~ N(ξ(0), σ₀²) and z₀ ~ N(0, 1)
    - Integrate ODE for extended state (a, z)
    - Extract and return action trajectory a(t)
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
        Initialize Streaming Flow Matching Agent.
        
        Args:
            model_config: Configuration for the streaming flow matching model
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
        
        # Compute streaming flow matching loss
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
        Generate predictions by integrating the learned ODE for extended state.
        
        Following the original sfps.py implementation:
        - Start from a₀ (current action) and z₀ ~ N(0,1)
        - Integrate ODE with concatenated state (a, z)
        - The model input should be shape [batch, 1, 2*action_dim] for single point
        
        Args:
            observation: Observation dictionary
            extra_inputs: Extra inputs dictionary
            
        Returns:
            Generated actions (only a(t), not z(t))
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
        
        # Get action shape from process_batch
        action_shape = self.process_batch.action_shape
        action_dim = action_shape[-1] if len(action_shape) > 0 else action_shape[0]
        
        # Initialize a_0 = xi(0): the current robot state in action space.
        # For streaming SFP, a_0 must approximate the trajectory start point xi(0) so that
        # the ODE integrates along the physical trajectory rather than from an arbitrary origin.
        #
        # Strategy 1: For tasks following the "_action" suffix convention (e.g. SOFA), each
        #   action key "xxx_action" has a matching state key "xxx" in the observation dict.
        #   Concatenate the last-timestep values of all matched state keys.
        a_0 = None
        a_0_parts = []
        all_matched = True
        for action_key in self.process_batch.action_keys:
            state_key = action_key.replace("_action", "")
            # Only accept the match if the suffix was actually removed (avoid "action"->"action")
            if state_key != action_key and state_key in observation:
                a_0_parts.append(observation[state_key][:, -1, :])
            else:
                all_matched = False
                break
        if all_matched and len(a_0_parts) > 0:
            candidate = torch.cat(a_0_parts, dim=-1)
            if candidate.shape[-1] == action_dim:
                a_0 = candidate

        if a_0 is None:
            # Strategy 2: Find any single observation key whose last-timestep feature covers
            #   the full action dimension (e.g. obstacle_avoidance: "agent_pos" dim == action_dim).
            for obs_tensor in observation.values():
                if obs_tensor.shape[-1] >= action_dim:
                    a_0 = obs_tensor[:, -1, :action_dim]
                    break

        if a_0 is None:
            # Strategy 3: Fall back to zeros. The ODE will still converge but from a
            #   suboptimal starting point.
            a_0 = torch.zeros((batch_size, action_dim), device=self.device)
        
        # z₀ ~ N(0, I) - this is the stochastic component
        z_0 = torch.randn((batch_size, action_dim), device=self.device)
        
        # Stack to form initial extended state
        x_t = torch.stack([a_0, z_0], dim=1)  # [batch_size, 2, action_dim]
        
        # Integrate ODE from t=0 to t=1
        # Following original code: integrate with multiple steps and save trajectory
        t_pred = action_shape[0] if len(action_shape) > 1 else 16
        
        # Calculate integration steps: we want to generate t_pred points
        # Original uses: integration_steps_per_action * num_actions
        # For simplicity, we use fixed num_integration_steps total
        dt = 1.0 / self.num_integration_steps
        
        # Store trajectory at specific time points
        # We want to output t_pred action points
        output_indices = torch.linspace(0, self.num_integration_steps, t_pred, device=self.device).long()
        output_indices = torch.clamp(output_indices, 0, self.num_integration_steps)
        trajectory_actions = []
        
        if self.solver == "euler":
            # Euler method: x_{t+dt} = x_t + dt * v(x_t, t)
            # Save initial state
            if 0 in output_indices:
                trajectory_actions.append(x_t[:, 0, :].clone())
            
            for step in range(self.num_integration_steps):
                t = torch.full((batch_size, 1), step * dt, device=self.device)
                
                # Reshape x_t for model: [batch_size, 2, action_dim] -> [batch_size, 1, 2*action_dim]
                x_t_flat = x_t.reshape(batch_size, 1, 2 * action_dim)
                
                # Get velocity
                v_t = self.model.forward(state, x_t_flat, t, extra_inputs)  # [batch_size, t_pred, 2*action_dim]
                
                # The model outputs full sequence, but we only need the first timestep
                # since we're doing single-step integration
                v_t = v_t[:, 0:1, :]  # [batch_size, 1, 2*action_dim]
                
                # Reshape back to [batch_size, 2, action_dim]
                v_t = v_t.reshape(batch_size, 2, action_dim)
                
                # Update state
                x_t = x_t + dt * v_t
                
                # Save if this is an output time point
                if (step + 1) in output_indices:
                    trajectory_actions.append(x_t[:, 0, :].clone())
                
        elif self.solver == "heun":
            # Heun's method (2nd order Runge-Kutta)
            # Save initial state
            if 0 in output_indices:
                trajectory_actions.append(x_t[:, 0, :].clone())
            
            for step in range(self.num_integration_steps):
                t = torch.full((batch_size, 1), step * dt, device=self.device)
                t_next = torch.full((batch_size, 1), (step + 1) * dt, device=self.device)
                
                # Predictor step
                x_t_flat = x_t.reshape(batch_size, 1, 2 * action_dim)
                v_t = self.model.forward(state, x_t_flat, t, extra_inputs)
                v_t = v_t[:, 0:1, :]  # Take first timestep only
                v_t = v_t.reshape(batch_size, 2, action_dim)
                x_pred = x_t + dt * v_t
                
                # Corrector step
                x_pred_flat = x_pred.reshape(batch_size, 1, 2 * action_dim)
                v_next = self.model.forward(state, x_pred_flat, t_next, extra_inputs)
                v_next = v_next[:, 0:1, :]  # Take first timestep only
                v_next = v_next.reshape(batch_size, 2, action_dim)
                x_t = x_t + dt * (v_t + v_next) / 2
                
                # Save if this is an output time point
                if (step + 1) in output_indices:
                    trajectory_actions.append(x_t[:, 0, :].clone())
                
        elif self.solver == "rk4":
            # 4th order Runge-Kutta
            # Save initial state
            if 0 in output_indices:
                trajectory_actions.append(x_t[:, 0, :].clone())
            
            for step in range(self.num_integration_steps):
                t = torch.full((batch_size, 1), step * dt, device=self.device)
                t_mid = torch.full((batch_size, 1), (step + 0.5) * dt, device=self.device)
                t_next = torch.full((batch_size, 1), (step + 1) * dt, device=self.device)
                
                def get_v(x, time):
                    x_flat = x.reshape(batch_size, 1, 2 * action_dim)
                    v = self.model.forward(state, x_flat, time, extra_inputs)
                    v = v[:, 0:1, :]  # Take first timestep only
                    return v.reshape(batch_size, 2, action_dim)
                
                k1 = get_v(x_t, t)
                k2 = get_v(x_t + dt * k1 / 2, t_mid)
                k3 = get_v(x_t + dt * k2 / 2, t_mid)
                k4 = get_v(x_t + dt * k3, t_next)
                
                x_t = x_t + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                
                # Save if this is an output time point
                if (step + 1) in output_indices:
                    trajectory_actions.append(x_t[:, 0, :].clone())
        else:
            raise ValueError(f"Unknown solver: {self.solver}")
        
        # Stack trajectory actions
        # trajectory_actions is a list of [batch_size, action_dim] tensors
        if len(trajectory_actions) > 0:
            action_pred = torch.stack(trajectory_actions, dim=1)  # [batch_size, t_pred, action_dim]
        else:
            # Fallback: use final state
            action_pred = x_t[:, 0, :].unsqueeze(1).expand(-1, t_pred, -1)
        
        # Restore original weights if using EMA
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())
        
        # Handle predict_past option
        if self.predict_past:
            return action_pred[:, self.t_obs-1:, :]
        
        return action_pred
    
    @torch.no_grad()
    def evaluate(self, batch: Dict) -> Tuple[float, float, float]:
        """
        Evaluate the model on a batch of data.

        Uses single-step prediction (t=1→0) to compute start/end point deviation,
        avoiding the 50-step ODE integration that would make val 51x slower than train.
        Full ODE rollout is reserved for test_agent() in the workspace.

        Args:
            batch: Mini batch dictionary

        Returns:
            Tuple of (eval_loss, start_point_deviation, end_point_deviation)
        """
        # Set model to eval mode
        self.model.eval()
        self.encoder.eval()

        # Load EMA weights if using EMA
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_encoder.store(self.encoder.parameters())
            self.ema_model.copy_to(self.model.parameters())
            self.ema_encoder.copy_to(self.encoder.parameters())

        # Process batch
        action, observation, extra_inputs = self.process_batch(batch)

        # Encode observation
        state = self.encoder(observation)

        with torch.no_grad():
            # Sample time t uniformly for loss computation
            t = torch.rand(action.shape[0], 1, device=self.device)
            eval_loss, _ = self.model.loss(state, action, t, extra_inputs, return_prediction=True)

            # Single Euler step from t=0 to t=1 for start/end deviation estimate.
            # SFP works on (a, z) pairs: x = [a, z] with shape (B, t_pred, 2*action_dim).
            # Construct x_0 as concatenation of a_0 and z_0, then flatten for model input.
            batch_size, t_pred, action_dim = action.shape
            a_0 = torch.zeros(batch_size, t_pred, action_dim, device=self.device)
            z_0 = torch.randn(batch_size, t_pred, action_dim, device=self.device)
            x_0 = torch.cat([a_0, z_0], dim=-1)  # (B, t_pred, 2*action_dim)
            t_zero = torch.zeros(action.shape[0], 1, device=self.device)
            v_0 = self.model.forward(state, x_0, t_zero, extra_inputs)
            # v_0 predicts velocity for (a, z); extract only the action component
            denoised_action = (x_0 + v_0)[:, :, :action_dim]

        # Restore original weights if using EMA
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())

        # Calculate L2 error of first and last action
        start_point_deviation = torch.linalg.norm(
            action[:, 0, :] - denoised_action[:, 0, :], dim=-1
        ).mean()
        end_point_deviation = torch.linalg.norm(
            action[:, -1, :] - denoised_action[:, -1, :], dim=-1
        ).mean()

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
