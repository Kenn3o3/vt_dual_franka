"""
MOTIF Transformer Inner Model

Implements the 4 core changes from MOTIF v3.2:
1. Physical time encoding (seconds, not step indices)
2. Fourier coefficient space output
3. State-conditioned velocity query
4. Fourier decoder for velocity supervision
"""

import torch
import hydra
import numpy as np
from omegaconf import DictConfig
from typing import Optional, Dict, Tuple, List

from movement_primitive_diffusion.models.inner_models.transformers.causal_transformer_inner_model import CausalTransformer
from movement_primitive_diffusion.utils.motif_utils import MOTIFHandler


class MotifTimeEmbedding(torch.nn.Module):
    """
    Physical time embedding for MOTIF.
    
    Implements sinusoidal embedding of physical time τ (in seconds),
    replacing the ordinal step index k used in π0.
    
    Formula: TimeEncode(τ) = [sin(ω₁τ), cos(ω₁τ), ..., sin(ωₘτ), cos(ωₘτ)]
    where ω_i = exp(i · log(10⁴) / m)
    """
    
    def __init__(self, time_embed_dim: int = 64, embedding_size: int = 256):
        """
        Args:
            time_embed_dim: Dimension of sinusoidal time encoding (m in paper)
            embedding_size: Output embedding size
        """
        super().__init__()
        self.time_embed_dim = time_embed_dim
        self.embedding_size = embedding_size
        
        # Compute frequency values: ω_i = exp(i · log(10⁴) / m)
        i = torch.arange(0, time_embed_dim, dtype=torch.float32)
        omega = torch.exp(i * np.log(10000.0) / time_embed_dim)
        self.register_buffer('omega', omega)
        
        # Linear projection from sinusoidal encoding to embedding_size
        self.projection = torch.nn.Linear(2 * time_embed_dim, embedding_size)
        
    def forward(self, times: torch.Tensor) -> torch.Tensor:
        """
        Encode physical times.
        
        Args:
            times: Physical times in seconds [batch_size, num_tokens]
            
        Returns:
            embeddings: Time embeddings [batch_size, num_tokens, embedding_size]
        """
        # times: [B, N]
        # omega: [time_embed_dim]
        # Broadcast: [B, N, 1] * [1, 1, time_embed_dim] -> [B, N, time_embed_dim]
        angles = times.unsqueeze(-1) * self.omega[None, None, :]
        
        # Sinusoidal encoding: [B, N, 2*time_embed_dim]
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        
        # Project to embedding_size: [B, N, embedding_size]
        embeddings = self.projection(encoding)
        
        return embeddings


class MOTIFTransformerInnerModel(CausalTransformer):
    """
    MOTIF Transformer implementing all 4 core changes.
    
    Key modifications from CausalTransformer:
    1. Physical time encoding instead of learnable position embedding
    2. Output head produces (M+1)×d coefficients instead of K×d positions
    3. State query mechanism with t-conditioned masking
    4. Fourier decoder for velocity output
    """
    
    def __init__(
        self,
        action_size: int,
        state_size: int,
        sigma_embedding_config: DictConfig,
        motif_handler_config: DictConfig,
        t_pred: int,
        t_obs: int,
        time_embed_dim: int = 64,
        n_layers: int = 8,
        n_heads: int = 4,
        embedding_size: int = 256,
        dropout_probability_embedding: float = 0.0,
        dropout_probability_attention: float = 0.01,
        n_cond_layers: int = 0,
        use_physical_time_encoding: bool = True,
    ) -> None:
        """
        Initialize MOTIF Transformer.
        
        Args:
            action_size: Action dimension (num_dof)
            state_size: State observation dimension
            sigma_embedding_config: Config for sigma (noise level) embedding
            motif_handler_config: Config for MOTIF handler
            t_pred: Number of prediction timesteps
            t_obs: Number of observation timesteps
            time_embed_dim: Dimension of sinusoidal time encoding
            n_layers: Number of decoder layers
            n_heads: Number of attention heads
            embedding_size: Embedding dimension
            dropout_probability_embedding: Dropout for embeddings
            dropout_probability_attention: Dropout for attention
            n_cond_layers: Number of encoder layers
            use_physical_time_encoding: If False, use learnable position embedding
                instead of sinusoidal physical-time encoding (M1 ablation)
        """
        # Initialize parent class (but we'll override some components)
        super().__init__(
            action_size=action_size,
            state_size=state_size,
            sigma_embedding_config=sigma_embedding_config,
            t_pred=t_pred,
            t_obs=t_obs,
            predict_past=False,  # MOTIF doesn't support predict_past
            n_layers=n_layers,
            n_heads=n_heads,
            embedding_size=embedding_size,
            dropout_probability_embedding=dropout_probability_embedding,
            dropout_probability_attention=dropout_probability_attention,
            n_cond_layers=n_cond_layers,
        )
        
        # Initialize MOTIF handler
        motif_handler_config.num_dof = action_size
        self.motif_handler: MOTIFHandler = hydra.utils.instantiate(motif_handler_config)
        
        # Store latest coefficients for debugging/visualization
        self.latest_coeffs: Optional[torch.Tensor] = None

        self.use_physical_time_encoding = use_physical_time_encoding
        
        # CHANGE 1: Physical time encoding (M1)
        # Full MOTIF: replace learnable position embedding with physical time encoder
        # M1 ablation (use_physical_time_encoding=False): keep learnable position embedding
        coeff_seq_len_for_pe = self.motif_handler.num_modes + 1
        if use_physical_time_encoding:
            self.time_embedding = MotifTimeEmbedding(time_embed_dim, embedding_size)
            delattr(self, 'position_embedding')
            self.position_embedding = None
        else:
            # Learnable position embedding over coefficient sequence length
            delattr(self, 'position_embedding')
            self.position_embedding = torch.nn.Parameter(
                torch.zeros(1, coeff_seq_len_for_pe, embedding_size)
            )
            torch.nn.init.normal_(self.position_embedding, mean=0.0, std=0.02)
            self.time_embedding = None
        
        # CHANGE 2: Coefficient space output head
        # Output (M+1) × d coefficients instead of K × d positions
        self.head = torch.nn.Linear(embedding_size, self.motif_handler.encoding_size, bias=False)
        
        # Override action_embedding to handle coefficient dimensions
        # Input: [B, M+1, d] -> need to embed each coefficient vector
        self.action_embedding = torch.nn.Linear(action_size, embedding_size)
        
        # CHANGE 3: State-conditioned query mechanism
        # State projection and mask token
        self.state_proj = torch.nn.Linear(action_size, embedding_size)
        self.state_mask = torch.nn.Parameter(torch.zeros(embedding_size))
        
        # Re-initialize the new components
        torch.nn.init.normal_(self.state_proj.weight, mean=0.0, std=0.02)
        if self.state_proj.bias is not None:
            torch.nn.init.zeros_(self.state_proj.bias)
        torch.nn.init.zeros_(self.state_mask)
        
        # Override mask for coefficient sequence length (M+1 instead of t_pred)
        coeff_seq_len = self.motif_handler.num_modes + 1
        print(f"[DEBUG MOTIFTransformerInnerModel.__init__] Creating mask:")
        print(f"  num_modes: {self.motif_handler.num_modes}")
        print(f"  coeff_seq_len (M+1): {coeff_seq_len}")
        print(f"  t_pred: {t_pred}")
        mask = (torch.triu(torch.ones(coeff_seq_len, coeff_seq_len)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        self.register_buffer("mask", mask, persistent=False)
        print(f"  mask shape: {mask.shape}")
    
    def get_optim_groups(self, weight_decay: float = 1e-3) -> List[dict]:
        """
        Override parent method to handle MOTIF-specific parameters.
        
        This is necessary because MOTIF adds new parameters (state_mask, state_proj)
        that need to be properly categorized for weight decay.
        """
        # Separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.MultiheadAttention)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for module_name, module in self.named_modules():
            for parameter_name, _ in module.named_parameters():
                fpn = "%s.%s" % (module_name, parameter_name) if module_name else parameter_name
                
                if parameter_name.endswith("bias"):
                    no_decay.add(fpn)
                elif parameter_name.startswith("bias"):
                    no_decay.add(fpn)
                elif parameter_name.endswith("weight") and isinstance(module, whitelist_weight_modules):
                    decay.add(fpn)
                elif parameter_name.endswith("weight") and isinstance(module, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        # Special case: parameters that should not have weight decay
        no_decay.add("condition_position_embedding")
        no_decay.add("state_mask")  # MOTIF-specific: state mask token
        
        # Validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}

        # M1 ablation: learnable position_embedding is an nn.Parameter, not a module
        # weight, so it won't be caught by the loop above; add it here after param_dict
        # is available.
        if not self.use_physical_time_encoding and "position_embedding" in param_dict:
            no_decay.add("position_embedding")
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" % (str(param_dict.keys() - union_params),)
        
        # Create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        return optim_groups
        
    def forward(
        self,
        state: torch.Tensor,
        noised_coeffs: torch.Tensor,
        sigma: torch.Tensor,
        extra_inputs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass of MOTIF model.
        
        Args:
            state: State observation [batch_size, t_obs, state_size]
            noised_coeffs: Noised DCT coefficients [batch_size, num_modes+1, num_dof]
            sigma: Noise level [batch_size, 1]
            extra_inputs: Dictionary containing:
                - physical_times: Physical times τ in seconds [batch_size, num_modes+1]
                - gt_states: Demonstration states [batch_size, K, num_dof] (optional)
                - t_diffusion: Current diffusion time (for state masking)
                
        Returns:
            coeffs: Predicted DCT coefficients [batch_size, num_modes+1, num_dof]
        """
        batch_size = noised_coeffs.shape[0]
        num_modes_plus_1 = noised_coeffs.shape[1]
        
        # Extract extra inputs
        if extra_inputs is None:
            extra_inputs = {}
        
        # Get physical times (in seconds, not step indices!)
        # For coefficient tokens, use uniform grid: τ_k = k * T / (M+1), k=0,...,M
        # This ensures each token has a physical time encoding
        times = torch.arange(0, num_modes_plus_1, dtype=torch.float32, device=noised_coeffs.device)
        times = times * self.motif_handler.chunk_duration / num_modes_plus_1  # [M+1]
        times = times.unsqueeze(0).expand(batch_size, -1)  # [B, M+1]
        
        # Get t_diffusion for state masking (critical correctness condition!)
        t_diffusion = extra_inputs.get('t_diffusion', torch.ones(batch_size, device=noised_coeffs.device))
        
        # CHANGE 1: Physical time encoding (M1)
        # Embed noised coefficients
        action_embedding = self.action_embedding(noised_coeffs)  # [B, M+1, embedding_size]

        if self.use_physical_time_encoding:
            # Full MOTIF: sinusoidal physical-time encoding
            time_emb = self.time_embedding(times)  # [B, M+1, embedding_size]
            action_embedding = action_embedding + time_emb
        else:
            # M1 ablation: learnable position embedding (step-index style)
            action_embedding = action_embedding + self.position_embedding[:, :num_modes_plus_1, :]
        
        # CHANGE 3: State-conditioned query
        # Apply state mask based on t_diffusion (NOT train/test flag!)
        if 'gt_states' in extra_inputs:
            # During training: use demonstration states
            # Take the first state as representative (could also use mean or specific timestep)
            current_state = extra_inputs['gt_states'][:, 0, :]  # [B, num_dof]
        else:
            # During inference: should be provided in extra_inputs as 'current_state'
            current_state = extra_inputs.get('current_state', torch.zeros(batch_size, self.motif_handler.num_dof, device=noised_coeffs.device))
        
        # State masking: mask when t > 0, use real state when t = 0
        # is_execution: [B, 1]
        is_execution = (t_diffusion == 0).float().unsqueeze(-1)
        
        # State embedding: [B, embedding_size]
        state_emb = self.state_proj(current_state)
        
        # Masked state: use real state at t=0, mask token at t>0
        masked_state = is_execution * state_emb + (1 - is_execution) * self.state_mask[None, :]
        
        # Add state to all action tokens (broadcast)
        action_embedding = action_embedding + masked_state.unsqueeze(1)  # [B, M+1, embedding_size]
        
        # Embed sigma
        sigma_embedding = self.sigma_embedding(sigma.view(batch_size, -1)).unsqueeze(1)  # [B, 1, embedding_size]
        
        # Encoder: process state observations
        condition_embedding = self.condition_embedding(state)  # [B, t_obs, embedding_size]
        condition_embedding = torch.cat([sigma_embedding, condition_embedding], dim=1)  # [B, 1+t_obs, embedding_size]
        
        x = self.drop(condition_embedding + self.condition_position_embedding)
        x = self.encoder(x)
        memory = x  # [B, 1+t_obs, embedding_size]
        
        # Decoder: process action tokens with causal mask
        x = self.drop(action_embedding)
        x = self.decoder(tgt=x, memory=memory, tgt_mask=self.mask[:num_modes_plus_1, :num_modes_plus_1], memory_mask=self.memory_mask)
        
        # Output head: produce coefficients
        x = self.ln_f(x[:, -1, :])  # Take last token [B, embedding_size]
        coeffs = self.head(x)  # [B, (M+1)*num_dof]
        
        # Reshape to [B, M+1, num_dof]
        coeffs = coeffs.view(batch_size, self.motif_handler.num_modes + 1, self.motif_handler.num_dof)
        
        # Store for debugging
        self.latest_coeffs = coeffs.detach()
        
        return coeffs
    
    def decode_to_velocity(
        self,
        coeffs: torch.Tensor,
        times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        CHANGE 4: Fourier decoder for velocity supervision.
        
        Decode coefficients to velocity at specified times.
        
        Args:
            coeffs: DCT coefficients [batch_size, num_modes+1, num_dof]
            times: Physical times [batch_size, N_query] or None
            
        Returns:
            velocities: Decoded velocities [batch_size, N_query, num_dof]
        """
        return self.motif_handler.decode(coeffs, times)
    
    def decode_to_position(
        self,
        coeffs: torch.Tensor,
        initial_position: torch.Tensor,
        times: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Decode coefficients to position trajectory.
        
        Args:
            coeffs: DCT coefficients [batch_size, num_modes+1, num_dof]
            initial_position: Initial position [batch_size, num_dof]
            times: Physical times [batch_size, N_query] or None
            
        Returns:
            positions: Position trajectory [batch_size, N_query, num_dof]
        """
        return self.motif_handler.decode_to_position(coeffs, initial_position, times)
