import hydra
import torch

from omegaconf import DictConfig
from torch.utils.data import DataLoader

from movement_primitive_diffusion.agents.base_agent import BaseAgent


def setup_agent(cfg: DictConfig, train_dataloader: DataLoader) -> BaseAgent:
    """Build and configure agent from config.
    
    Args:
        cfg: Hydra config
        train_dataloader: Training dataloader (used for config validation and sigma_data calculation)
        
    Returns:
        Configured agent instance
    """
    # Get a batch of data to determine the observation sizes and validate the observation keys
    data = next(iter(train_dataloader))

    # Set the process_batch observation keys based on the encoder config
    encoder_observation_keys = []
    for network_config in cfg.agent_config.encoder_config.network_configs:
        encoder_observation_keys.append(network_config.observation_key)
    cfg.agent_config.process_batch_config.observation_keys = encoder_observation_keys

    # VALIDATION: Check that these keys are present in the data
    for key in encoder_observation_keys:
        assert key in data.keys(), f"Key {key} not present in data"

    # Set the observation sizes in the encoder config
    for network_config in cfg.agent_config.encoder_config.network_configs:
        network_config.feature_size = list(data[network_config.observation_key].shape[2:])
        if hasattr(inner_config := network_config.network_config, "feature_size"):
            inner_config.feature_size = network_config.feature_size

    # Set the sizes in the process_batch config
    for info in cfg.agent_config.process_batch_config.action_keys:
        info.feature_size = list(data[info.key].shape[2:])

    # NOTE: movement_primitive_diffusion.utils.lr_scheduler.get_scheduler expects the number of training steps as argument.
    # to not break compatibility with directly instantiating other schedulers, we check for the existence of the
    # num_training_steps attribute.
    if hasattr(cfg.agent_config.lr_scheduler_config, "num_training_steps"):
        # Figure out the number of training steps for the LR scheduler
        if cfg.epochs is None:
            raise ValueError("If you want to use an lr scheduler wit num_training_steps, you need to specify the number of epochs.")
        cfg.agent_config.lr_scheduler_config.num_training_steps = len(train_dataloader) * cfg.epochs

    # Instantiate the agent
    agent: BaseAgent = hydra.utils.instantiate(cfg.agent_config)

    # If necessary (there is a sigma_data, and its value is None), calculate and set sigma_data for scaling
    if scaling := getattr(agent.model, "scaling", False):
        if getattr(scaling, "sigma_data", False) is None:
            scaling.set_sigma_data(scaling.calculate_sigma_data_of_action(agent, train_dataloader))

    return agent
