import hydra

from typing import List, Optional


def setup_swanlab_metrics(workspace_result_keys: List[str], performance_metric: str) -> None:
    """Set up metrics for swanlab logging.
    
    Args:
        workspace_result_keys: Keys of workspace results from workspace.test_agent(agent)
        performance_metric: Metric used to determine if model is better than previous best
        
    Note:
        SwanLab automatically handles step metrics, no need to explicitly define them
    """
    pass


def setup_swanlab_test_metrics(workspace_result_keys: List[str]) -> None:
    """Set up metrics for swanlab in test scripts.
    
    Args:
        workspace_result_keys: Keys of workspace results from workspace.test_agent(agent)
        
    Note:
        SwanLab automatically handles step metrics, no need to explicitly define them
    """
    pass


def get_group_from_override(length: int = 2, ignore_keys: Optional[List[str]] = None) -> str:
    """Get group name from hydra overrides.
    
    Takes the last part of the override, splits by '_', and takes the first letter(s) of each part.
    Example: agent_config.model_config.train_btm_image_prodmp_residual_mlp -> t_b_i_p_r_m (length=1)
    
    Args:
        length: Number of characters to take from each component
        ignore_keys: Keys to ignore in the override name
        
    Returns:
        Group name string
    """
    overrides = hydra.utils.HydraConfig.get()["overrides"]["task"]
    overrides_shortened = []
    ignore_keys = ignore_keys or []
    for override in overrides:
        if "seed" in override:
            continue
        override_value = override.split("=")[-1]
        override_key = override.split("=")[-2]
        key_components = override_key.split(".")  # overrides from setting a param.value: val
        if len(key_components) == 1:
            key_components = key_components[0].split("/")  # overrides from overriding a config/value: config_name

        if key_components[-1] in ignore_keys:
            continue
        else:
            override_shortened = "_".join([o[:length] for o in key_components[-1].split("_")])
            overrides_shortened.append(f"{override_shortened}={override_value}")

    return ",".join(overrides_shortened)
