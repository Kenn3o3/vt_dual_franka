import git
import hydra
import logging

from copy import deepcopy
from omegaconf import DictConfig
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Tuple

from movement_primitive_diffusion.datasets.trajectory_dataset import SubsequenceTrajectoryDataset
from movement_primitive_diffusion.utils.helper import tensor_to_list

log = logging.getLogger(__name__)


def look_for_trajectory_dir(search_dir: str) -> Path:
    """Find trajectory directory in either absolute path or relative to git root.
    
    Args:
        search_dir: Directory name or path to search for
        
    Returns:
        Path to the trajectory directory
    """
    # Get the root directory of the repository
    git_repo = git.Repo(".", search_parent_directories=True)
    git_root = git_repo.working_tree_dir

    relative_trajectory_dir = Path(f"{git_root}/data/{search_dir}/")
    absolute_trajectory_dir = Path(search_dir)

    if absolute_trajectory_dir.is_dir() and relative_trajectory_dir.is_dir():
        raise ValueError(f"Found two directories for trajectories: {relative_trajectory_dir=} and {absolute_trajectory_dir=}.")
    elif not absolute_trajectory_dir.is_dir() and not relative_trajectory_dir.is_dir():
        raise ValueError(f"Could not find trajectory directory. Looked in {relative_trajectory_dir=} and {absolute_trajectory_dir=}.")
    elif absolute_trajectory_dir.is_dir():
        trajectory_dir = absolute_trajectory_dir
    else:
        trajectory_dir = relative_trajectory_dir

    return trajectory_dir


def get_dataloaders_for_fixed_split(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders from separate directories.
    
    Args:
        cfg: Hydra config containing train_trajectory_dir and val_trajectory_dir
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Look up all available trajectory paths
    assert "train_trajectory_dir" in cfg and cfg.train_trajectory_dir is not None, "train_trajectory_dir must be set if fixed_split is True"
    assert "val_trajectory_dir" in cfg and cfg.val_trajectory_dir is not None, "val_trajectory_dir must be set if fixed_split is True"
    train_trajectory_dir = look_for_trajectory_dir(cfg.train_trajectory_dir)
    val_trajectory_dir = look_for_trajectory_dir(cfg.val_trajectory_dir)
    train_trajectories = [path for path in train_trajectory_dir.iterdir() if path.is_dir()]
    val_trajectories = [path for path in val_trajectory_dir.iterdir() if path.is_dir()]

    # Load all available trajectories to compute correct scaler values
    combined_trajectories = train_trajectories + val_trajectories
    combined_dataset_config = deepcopy(cfg.dataset_config)
    combined_dataset_config.trajectory_dirs = combined_trajectories
    combined_dataset = hydra.utils.instantiate(combined_dataset_config, _convert_="all")
    scaler_values = tensor_to_list(combined_dataset.scaler_values)

    # Delete the combined dataset to free up memory
    del combined_dataset

    # Set the scaler values in the dataset config
    cfg.dataset_config.scaler_values = scaler_values

    # Set the scaler values in the workspace config (skip for DummyWorkspace which has no env_config)
    if "env_config" in cfg.workspace_config:
        cfg.workspace_config.env_config.scaler_config.scaler_values = scaler_values

    # Instantiate train and validation datasets
    # If num_trajectories is set, only use the first num_trajectories for training
    train_dataset_config = deepcopy(cfg.dataset_config)
    val_dataset_config = deepcopy(cfg.dataset_config)
    train_dataset_config.trajectory_dirs = train_trajectories
    if cfg.get("num_trajectories", False):
        train_dataset_config.trajectory_dirs = train_dataset_config.trajectory_dirs[: cfg.num_trajectories]
    val_dataset_config.trajectory_dirs = val_trajectories

    # Training and validation data come from their own directories
    train_dataset = hydra.utils.instantiate(train_dataset_config, _convert_="all")
    val_dataset = hydra.utils.instantiate(val_dataset_config, _convert_="all")

    # Move the dataset to the correct device
    if cfg.dataset_fully_on_gpu:
        train_dataset.to(cfg.device)
        val_dataset.to(cfg.device)

    # If the batch size is -1, set it to the length of the dataset
    if cfg.data_loader_config.batch_size == -1:
        cfg.data_loader_config.batch_size = max(len(train_dataset), len(val_dataset))
        log.log(logging.INFO, f"Set batch size to {cfg.data_loader_config.batch_size} to fit all data in one batch.")

    train_dataloader = DataLoader(train_dataset, **cfg.data_loader_config)
    val_dataloader = DataLoader(val_dataset, **cfg.data_loader_config)

    return train_dataloader, val_dataloader


def get_dataloaders(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders by splitting a single dataset.
    
    Args:
        cfg: Hydra config containing trajectory_dir and train_split
        
    Returns:
        Tuple of (train_dataloader, val_dataloader)
    """
    # Look for data
    trajectory_dir = look_for_trajectory_dir(cfg.trajectory_dir)
    cfg.dataset_config.trajectory_dirs = [path for path in trajectory_dir.iterdir() if path.is_dir()]

    # If num_trajectories is set, only use the first num_trajectories
    if cfg.get("num_trajectories", False):
        cfg.dataset_config.trajectory_dirs = cfg.dataset_config.trajectory_dirs[: cfg.num_trajectories]

    # Create the dataset, move it to the correct device and split it into train and val data
    dataset: SubsequenceTrajectoryDataset = hydra.utils.instantiate(cfg.dataset_config, _convert_="all")
    if cfg.dataset_fully_on_gpu:
        dataset.to(cfg.device)
    (train_dataset, val_dataset), _ = dataset.split([cfg.train_split, 1 - cfg.train_split])

    # Set the scaler values in the workspace config (skip for DummyWorkspace which has no env_config)
    if "env_config" in cfg.workspace_config:
        cfg.workspace_config.env_config.scaler_config.scaler_values = tensor_to_list(dataset.scaler_values)

    # Delete the original dataset to free up memory immediately
    del dataset

    # If the batch size is -1, set it to the length of the dataset
    if cfg.data_loader_config.batch_size == -1:
        cfg.data_loader_config.batch_size = max(len(train_dataset), len(val_dataset))
        log.log(logging.INFO, f"Set batch size to {cfg.data_loader_config.batch_size} to fit all data in one batch.")

    train_dataloader = DataLoader(train_dataset, **cfg.data_loader_config)
    val_dataloader = DataLoader(val_dataset, **cfg.data_loader_config)

    return train_dataloader, val_dataloader
