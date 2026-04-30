"""
ProcessBatchMOTIF: Data preprocessing for MOTIF training.

This module handles:
- Computing velocities from position data
- Extracting DCT coefficients as training targets
- Generating physical time values (in seconds)
- Providing initial states for state-conditioned queries
"""

import torch
import hydra
from typing import Dict, List, Tuple, Union
from omegaconf import DictConfig

from movement_primitive_diffusion.datasets.process_batch import ProcessBatch
from movement_primitive_diffusion.utils.matrix import unsqueeze_and_expand
from movement_primitive_diffusion.utils.motif_utils import MOTIFHandler, compute_velocity_from_position


class ProcessBatchMOTIF(ProcessBatch):
    """
    Process batch for MOTIF training.
    
    Key differences from ProcessBatchProDMP:
    - Computes velocities from positions using finite difference
    - Extracts DCT coefficients as targets (instead of ProDMP parameters)
    - Provides physical times τ in seconds (instead of step indices)
    - Includes current state for state-conditioned velocity queries
    """
    
    def __init__(
        self,
        t_obs: int,
        t_pred: int,
        action_keys: DictConfig,
        observation_keys: Union[str, List[str]],
        initial_position_keys: Union[str, List[str]],
        initial_velocity_keys: Union[str, List[str]],
        motif_handler_config: DictConfig,
        initial_values_come_from_action_data: bool = False,
        relative_action_values: bool = False,
        predict_past: bool = False,
    ):
        """
        Initialize ProcessBatchMOTIF.
        
        Args:
            t_obs: Number of observation timesteps
            t_pred: Number of prediction timesteps
            action_keys: Configuration for action keys
            observation_keys: Keys for observation data
            initial_position_keys: Keys for initial position
            initial_velocity_keys: Keys for initial velocity
            motif_handler_config: Configuration for MOTIF handler
            initial_values_come_from_action_data: Whether initial values are from action data
            relative_action_values: Whether to use relative action values
            predict_past: Whether to predict past actions (not implemented for MOTIF)
        """
        if predict_past:
            raise NotImplementedError("predict_past is not yet implemented for MOTIF.")
        
        super().__init__(
            t_obs=t_obs,
            t_pred=t_pred,
            action_keys=action_keys,
            observation_keys=observation_keys,
            relative_action_values=relative_action_values,
            predict_past=predict_past,
        )
        
        if isinstance(initial_position_keys, str):
            initial_position_keys = [initial_position_keys]
        self.initial_position_keys = initial_position_keys
        
        if isinstance(initial_velocity_keys, str):
            initial_velocity_keys = [initial_velocity_keys]
        self.initial_velocity_keys = initial_velocity_keys
        
        self.initial_values_come_from_action_data = initial_values_come_from_action_data
        self.initial_value_index = self.t_obs - 2 if self.initial_values_come_from_action_data else self.t_obs - 1
        
        if self.initial_values_come_from_action_data:
            assert self.t_obs > 1, "If initial values come from action data, at least two observations are required."
        
        # Initialize MOTIF handler
        motif_handler_config.num_dof = sum(self.action_sizes)
        self.motif_handler: MOTIFHandler = hydra.utils.instantiate(motif_handler_config)
        
    def __call__(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process a batch for MOTIF training.
        
        Returns:
            coeffs: DCT coefficients [batch_size, num_modes+1, num_dof]
            observation: Observation dictionary
            extra_inputs: Dictionary containing:
                - action: Original action (position) sequence for validation
                - gt_velocities: Ground truth velocities for L_vel loss
                - physical_times: Physical times τ in seconds
                - initial_position: Initial position
                - initial_velocity: Initial velocity
                - gt_states: Demonstration states for state-conditioned training
        """
        # Get action (position) sequence
        action, observation, extra_inputs = super().__call__(batch)
        
        # Get initial values
        initial_position = torch.cat(
            [batch[key][:, self.initial_value_index] for key in self.initial_position_keys],
            dim=-1
        )
        initial_velocity = torch.cat(
            [batch[key][:, self.initial_value_index] for key in self.initial_velocity_keys],
            dim=-1
        )
        
        if self.relative_action_values:
            initial_position = torch.zeros_like(initial_position)
        
        # Compute velocities from positions using finite difference
        velocities = compute_velocity_from_position(action, self.motif_handler.dt)
        
        # Extract DCT coefficients from velocities
        coeffs = self.motif_handler.encode(velocities)
        
        # Debug: verify shapes
        if coeffs.shape[1] != self.motif_handler.num_modes + 1:
            print(f"[DEBUG ProcessBatchMOTIF] Shape mismatch:")
            print(f"  action: {action.shape}")
            print(f"  velocities: {velocities.shape}")
            print(f"  coeffs: {coeffs.shape}")
            print(f"  expected coeffs: [{action.shape[0]}, {self.motif_handler.num_modes + 1}, {self.motif_handler.num_dof}]")
            raise RuntimeError(f"Coefficient shape mismatch: got {coeffs.shape}, expected [*, {self.motif_handler.num_modes + 1}, {self.motif_handler.num_dof}]")
        
        # Generate physical times τ in seconds (not step indices!)
        # Times: [0, dt, 2*dt, ..., (K-1)*dt]
        times = torch.arange(
            0,
            self.motif_handler.traj_steps,
            dtype=torch.float32,
            device=action.device
        ) * self.motif_handler.dt
        times = times.unsqueeze(0).expand(action.size(0), -1)  # [B, K]
        
        # Get demonstration states for state-conditioned training
        # These are the actual states at each timestep during the demonstration
        gt_states = action  # Position states [B, K, d]
        
        # Package extra inputs
        extra_inputs = {
            "action": action,  # Original position sequence for validation
            "gt_velocities": velocities,  # For L_vel loss
            "physical_times": times,  # Physical times in seconds
            "initial_position": initial_position,
            "initial_velocity": initial_velocity,
            "gt_states": gt_states,  # Demonstration states for training
            **extra_inputs,
        }
        
        return coeffs, observation, extra_inputs
    
    def coefficients_to_action(
        self,
        coeffs: torch.Tensor,
        extra_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Decode DCT coefficients to action (position) sequence.
        
        Args:
            coeffs: DCT coefficients [batch_size, num_modes+1, num_dof]
            extra_inputs: Dictionary containing initial_position
            
        Returns:
            action: Position sequence [batch_size, traj_steps, num_dof]
        """
        # Decode to velocities
        velocities = self.motif_handler.decode(coeffs)
        
        # Integrate to positions
        positions = self.motif_handler.decode_to_position(
            coeffs,
            extra_inputs["initial_position"]
        )
        
        return positions
    
    def coefficients_to_velocity(
        self,
        coeffs: torch.Tensor,
        times: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Decode DCT coefficients to velocity sequence.
        
        Args:
            coeffs: DCT coefficients [batch_size, num_modes+1, num_dof]
            times: Physical times [batch_size, N_query] or None
            
        Returns:
            velocities: Velocity sequence [batch_size, N_query, num_dof]
        """
        return self.motif_handler.decode(coeffs, times)
    
    def process_env_observation(
        self,
        observation: Dict[str, torch.Tensor],
        skip_initial_values: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Process observation from environment for inference.
        
        Args:
            observation: Observation dictionary from environment
            skip_initial_values: Whether to skip extracting initial values
            
        Returns:
            processed_observation: Filtered observation dictionary
            extra_inputs: Dictionary with initial values and physical times
        """
        processed_observation, extra_inputs = super().process_env_observation(observation)
        
        if not skip_initial_values:
            # Extract initial values from last timestep
            extra_inputs = {
                "initial_position": torch.cat(
                    [observation[key][:, -1] for key in self.initial_position_keys],
                    dim=-1
                ),
                "initial_velocity": torch.cat(
                    [observation[key][:, -1] for key in self.initial_velocity_keys],
                    dim=-1
                ),
            }
            
            if self.relative_action_values:
                extra_inputs["initial_position"] = torch.zeros_like(extra_inputs["initial_position"])
        
        # Add physical times for inference
        batch_size = list(processed_observation.values())[0].shape[0]
        times = torch.arange(
            0,
            self.motif_handler.traj_steps,
            dtype=torch.float32,
            device=list(processed_observation.values())[0].device
        ) * self.motif_handler.dt
        times = times.unsqueeze(0).expand(batch_size, -1)
        extra_inputs["physical_times"] = times
        
        return processed_observation, extra_inputs
    
    @property
    def action_size(self) -> int:
        """
        Return the size of each coefficient vector (num_dof).
        
        Note: For MOTIF, action_size is the dimension of each coefficient vector (d),
        not the total encoding size (M+1)*d. This is because the Transformer processes
        coefficients as a sequence of (M+1) tokens, each with dimension d.
        """
        return self.motif_handler.num_dof
    
    @property
    def coefficient_shape(self) -> Tuple[int, int]:
        """Return the shape of the coefficient tensor: (num_modes+1, num_dof)"""
        return (self.motif_handler.num_modes + 1, self.motif_handler.num_dof)
    
    @property
    def parameter_shape(self) -> Tuple[int, int]:
        """Alias for coefficient_shape for compatibility with agent code."""
        return self.coefficient_shape
