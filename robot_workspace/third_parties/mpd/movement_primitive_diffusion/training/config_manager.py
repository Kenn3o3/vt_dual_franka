import os
import torch

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing import Tuple, Union

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace
from movement_primitive_diffusion.workspaces.base_workspace import BaseWorkspace
from .data_builder import get_dataloaders, get_dataloaders_for_fixed_split
from .agent_builder import setup_agent
from .workspace_builder import setup_workspace


def setup_train(cfg: DictConfig) -> Tuple[DataLoader, DataLoader, BaseAgent, Union[BaseWorkspace, BaseVectorWorkspace]]:
    """Main setup function for training: instantiates dataloaders, agent, and workspace.
    
    Args:
        cfg: Hydra config
        
    Returns:
        Tuple of (train_dataloader, val_dataloader, agent, workspace)
    """
    # Deactivate tqdm if configured
    if cfg.get("deactivate_tqdm", False):
        os.environ["TQDM_DISABLE"] = "1"

    # Figure out which device to use
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        assert isinstance(cfg.device, str), f"Expected device to be a str, got {type(cfg.device)=}."
        assert cfg.device in ["cuda", "cpu"], f"Please set device to either cpu or cuda. Got {cfg.device=}."
        device = cfg.device
    cfg.device = device
    cfg.agent_config.device = device

    # Load data: either fixed split or single directory with split ratio
    if "fixed_split" in cfg and cfg.fixed_split:
        train_dataloader, val_dataloader = get_dataloaders_for_fixed_split(cfg)
    else:
        train_dataloader, val_dataloader = get_dataloaders(cfg)

    # Setup agent with data-dependent configuration
    agent = setup_agent(cfg, train_dataloader)

    # Setup workspace
    workspace = setup_workspace(cfg)

    return train_dataloader, val_dataloader, agent, workspace


def setup_agent_and_workspace(cfg: DictConfig) -> Tuple[BaseAgent, BaseWorkspace]:
    """Simplified setup function for agent and workspace without data loading.
    
    Used for inference/testing scenarios where dataloaders are not needed.
    
    Args:
        cfg: Hydra config
        
    Returns:
        Tuple of (agent, workspace)
    """
    # Figure out which device to use
    if cfg.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        assert isinstance(cfg.device, str), f"Expected device to be a str, got {type(cfg.device)=}."
        assert cfg.device in ["cuda", "cpu"], f"Please set device to either cpu or cuda. Got {cfg.device=}."
        device = cfg.device
    cfg.device = device
    cfg.agent_config.device = device

    # Instantiate the agent
    agent: BaseAgent = hydra.utils.instantiate(cfg.agent_config)

    # Make sure sigma_data is set if the scaling needs it
    if scaling := getattr(agent.model, "scaling", False):
        if getattr(scaling, "sigma_data", False) is None:
            raise ValueError("Please set sigma_data in the scaling module of the model.")

    # Instantiate the workspace
    workspace = setup_workspace(cfg)

    return agent, workspace
