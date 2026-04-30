import hydra
import torch
import logging
import re
import swanlab
import git

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
import numpy as np
import random

from movement_primitive_diffusion.utils.setup_helper import get_group_from_override, setup_agent_and_workspace, parse_swanlab_to_hydra_config, setup_swanlab_test_metrics

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base=None, config_path="../conf", config_name="test_agent_checkpoints_in_env")
def main(cfg: DictConfig) -> None:
    # Figure out where the files are
    config_file_path = Path(cfg.config)
    if not config_file_path.is_file():
        if config_file_path.is_absolute():
            raise FileNotFoundError(f"Could not find config file at {config_file_path}.")
        else:
            git_repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
            git_root = git_repo.working_tree_dir
            config_file_path = Path(git_root) / config_file_path
            if not config_file_path.is_file():
                raise FileNotFoundError(f"Could not find config file at {config_file_path}.")

    checkpoint_dir_path = Path(cfg.checkpoint_dir)
    if not checkpoint_dir_path.is_dir():
        if checkpoint_dir_path.is_absolute():
            raise FileNotFoundError(f"Could not find checkpoint dir at {checkpoint_dir_path}.")
        else:
            git_repo = git.Repo(Path(__file__).parent, search_parent_directories=True)
            git_root = git_repo.working_tree_dir
            checkpoint_dir_path = Path(git_root) / checkpoint_dir_path
            if not checkpoint_dir_path.is_dir():
                raise FileNotFoundError(f"Could not find checkpoint dir at {checkpoint_dir_path}.")

    # Load config that was saved in swanlab
    swanlab_config = OmegaConf.load(config_file_path)

    # Get all weight files in checkpoint dir that match the regex
    weight_file_regex = re.compile(cfg.weight_file_regex)
    files_in_checkpoint_dir = [file for file in checkpoint_dir_path.iterdir() if file.is_file()]
    weight_files = [file for file in files_in_checkpoint_dir if weight_file_regex.search(file.name) is not None]
    if len(weight_files) == 0:
        raise FileNotFoundError(f"Could not find any weight files in {checkpoint_dir_path} that match {cfg.weight_file_regex}.")
    weight_files.sort(key=lambda file: int(weight_file_regex.search(file.name).groups()[0]))
    epoch_numbers = [int(weight_file_regex.search(file.name).groups()[0]) for file in weight_files]

    # Parse swanlab config to hydra config
    hydra_config = parse_swanlab_to_hydra_config(swanlab_config)

    # Seeds
    if "seed" in hydra_config:
        torch.manual_seed(hydra_config.seed)
        np.random.seed(hydra_config.seed)
        random.seed(hydra_config.seed)

    # Update config with new values
    hydra_config = OmegaConf.merge(hydra_config, cfg.to_change)

    # Create a config that holds both cfg and hydra_config for logging
    log_conf = OmegaConf.create({"cfg": cfg, "hydra_config": hydra_config})
    swanlab_config_dict = OmegaConf.to_container(log_conf, resolve=True, throw_on_missing=True)

    # Set group name
    if "group_from_overrides" in cfg and cfg.group_from_overrides:
        cfg.swanlab.group = get_group_from_override()

    # Set swanlab kwargs
    swanlab_kwargs = {
        "project": cfg.swanlab.project,
        "workspace": cfg.swanlab.get("entity", None),
        "experiment_name": cfg.swanlab.get("group", None),
        "mode": cfg.swanlab.mode,
        "config": swanlab_config_dict,
    }

    # Optionally set run name
    if "run_name" in cfg.swanlab:
        swanlab_kwargs["experiment_name"] = cfg.swanlab.run_name

    # Initialize swanlab run
    swanlab.init(**swanlab_kwargs)

    # Setup agent, and workspace
    agent, workspace = setup_agent_and_workspace(hydra_config)

    # Setup swanlab metrics to log metrics over epochs
    workspace_result_keys = workspace.get_result_dict_keys()
    setup_swanlab_test_metrics(workspace_result_keys)

    # Test each checkpoint
    for weight_file, epoch_number in zip(weight_files, epoch_numbers):
        # Load the weights
        agent.load_pretrained(weight_file)

        # Test the agent in the environment
        test_results = workspace.test_agent(agent, cfg.num_trajectories)

        # Set epoch number to the checkpoint epoch number and log the test results
        swanlab.log({**test_results, "epoch": epoch_number})


if __name__ == "__main__":
    main()
