import hydra

from omegaconf import DictConfig
from typing import Union

from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace
from movement_primitive_diffusion.workspaces.base_workspace import BaseWorkspace


def setup_workspace(cfg: DictConfig) -> Union[BaseWorkspace, BaseVectorWorkspace]:
    """Build workspace from config.
    
    Args:
        cfg: Hydra config
        
    Returns:
        Workspace instance
    """
    workspace: BaseWorkspace = hydra.utils.instantiate(cfg.workspace_config)
    return workspace
