import hydra
import torch
import logging
import json
import operator
import os
import shutil

from omegaconf import DictConfig, OmegaConf
from pathlib import Path
from tqdm import tqdm
import numpy as np
import random

from movement_primitive_diffusion.utils.helper import dictionary_to_device, format_loss
from movement_primitive_diffusion.training import setup_train, setup_swanlab_metrics, get_group_from_override
from movement_primitive_diffusion.training.logger import setup_training_logger
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace

try:
    import swanlab
except ImportError:
    class _SwanLabRunStub:
        def finish(self):
            return None


    class _SwanLabStub:
        @staticmethod
        def login(*args, **kwargs):
            del args, kwargs

        @staticmethod
        def init(*args, **kwargs):
            del args, kwargs
            return _SwanLabRunStub()

        @staticmethod
        def log(*args, **kwargs):
            del args, kwargs

        @staticmethod
        def get_run():
            return None

        @staticmethod
        def define_metric(*args, **kwargs):
            del args, kwargs

    swanlab = _SwanLabStub()

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("eval", eval)

CONFIG = "experiments/bimanual_tissue_manipulation/train_prodmp_transformer.yaml"


def _save_vt_franka_run_artifacts(cfg: DictConfig, logging_path: Path) -> None:
    OmegaConf.save(config=cfg, f=logging_path / "resolved_config.yaml", resolve=True)

    train_dir = Path(str(cfg.get("train_trajectory_dir", "")))
    prepared_dir = train_dir.parent if train_dir.name == "train" else train_dir
    manifest_path = prepared_dir / "dataset_manifest.json"
    if manifest_path.exists():
        output_manifest_path = logging_path / "dataset_manifest.json"
        shutil.copy2(manifest_path, output_manifest_path)
        _stamp_vt_franka_state_anchored_manifest(output_manifest_path)

    prepared_scaler_path = prepared_dir / "scaler_values.npz"
    if prepared_scaler_path.exists():
        shutil.copy2(prepared_scaler_path, logging_path / "scaler_values.npz")
        return

    scaler_values = OmegaConf.to_container(cfg.dataset_config.scaler_values, resolve=True)
    flattened = {}
    for key, stats in scaler_values.items():
        for stat_name, value in stats.items():
            flattened[f"{key}_{stat_name}"] = np.asarray(value, dtype=np.float32)
    if flattened:
        np.savez_compressed(logging_path / "scaler_values.npz", **flattened)


def _stamp_vt_franka_state_anchored_manifest(path: Path) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest.setdefault("action_alignment", "causal_future_command")
    manifest.setdefault("velocity_convention", "finite_difference_after_normalization")
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


@hydra.main(version_base=None, config_path="../conf", config_name=CONFIG)
def main(cfg: DictConfig) -> float:
    # SwanLab login from environment variable or config
    swanlab_api_key = os.getenv("SWANLAB_API_KEY")
    if swanlab_api_key:
        swanlab.login(api_key=swanlab_api_key)

    # ── GPU 性能优化（默认开启，可通过配置关闭）──────────────────────────────
    # TF32: H100/A100 矩阵乘法加速 3-8x，精度损失可忽略
    use_tf32 = cfg.get("use_tf32", True)
    torch.backends.cuda.matmul.allow_tf32 = use_tf32
    torch.backends.cudnn.allow_tf32 = use_tf32
    # cuDNN benchmark: 自动为固定输入尺寸选最优 kernel（第一个 epoch 略慢）
    use_cudnn_benchmark = cfg.get("use_cudnn_benchmark", True)
    torch.backends.cudnn.benchmark = use_cudnn_benchmark

    # Seeds:
    if "seed" in cfg:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

    # Set the performance comparison operator
    performance_comparison_operator = operator.ge if cfg.performance_direction == "max" else operator.le
    performance_metric = cfg.performance_metric

    # Setup data, agent, and workspace
    train_dataloader, val_dataloader, agent, workspace = setup_train(cfg)

    # torch.compile: 融合算子，生成优化 CUDA kernel（PyTorch >= 2.0）
    # mode="default" 兼容所有模型；"reduce-overhead" 使用 CUDAGraphs 但有些模型会冲突
    compile_mode = cfg.get("compile_mode", "default")
    if cfg.get("compile_model", True) and hasattr(torch, "compile"):
        try:
            agent.model = torch.compile(agent.model, mode=compile_mode)
            agent.encoder = torch.compile(agent.encoder, mode=compile_mode)
            log.info(f"torch.compile enabled (mode={compile_mode})")
        except Exception as e:
            log.warning(f"torch.compile failed, skipping: {e}")

    # Initialize swanlab
    # Init swanlab stored config with entire hydra config used in this experiment
    swanlab_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    if "group_from_overrides" in cfg and cfg.group_from_overrides:
        cfg.swanlab.group = get_group_from_override()

    # Generate experiment name: {method_name}-{task_name}
    method_name = cfg.get("method_name", cfg.swanlab.get("group", "mpd"))
    # Prefer explicit task_name; fall back to trajectory_dir / train_trajectory_dir
    task_name = cfg.get("task_name")
    if not task_name:
        traj_dir = cfg.get("trajectory_dir") or cfg.get("train_trajectory_dir", "unknown")
        if traj_dir is None:
            traj_dir = "unknown"
        task_name = traj_dir.removesuffix("_train").removesuffix("_val")
    experiment_name = f"{method_name}-{task_name}"
    
    swanlab_kwargs = {
        "project": cfg.swanlab.project,
        "workspace": cfg.swanlab.get("entity", None),
        "experiment_name": experiment_name,
        "mode": cfg.swanlab.mode,
        "config": swanlab_config,
    }

    # if "run_name" in cfg.swanlab:
    #     swanlab_kwargs["experiment_name"] = cfg.swanlab.run_name
    # elif "name_from_overrides" in cfg and cfg.name_from_overrides:
    #     swanlab_kwargs["experiment_name"] = get_group_from_override(ignore_keys=cfg.get("ignore_in_name", []))

    # init swanlab logger and config from hydra path
    swanlab_run = swanlab.init(**swanlab_kwargs)

    # Setup logging
    logging_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    setup_training_logger(logging_path)
    _save_vt_franka_run_artifacts(cfg, logging_path)

    # Setup swanlab metrics
    workspace_result_keys = workspace.get_result_dict_keys()
    setup_swanlab_metrics(workspace_result_keys, performance_metric)

    best_performance_value = -torch.inf if cfg.performance_direction == "max" else torch.inf
    best_epoch = 0
    current_early_stopping_patience = 0
    done = False

    if not cfg.early_stopping and cfg.epochs is None:
        raise ValueError("Either early stopping or epochs must be set otherwise training will never stop.")

    epoch_magnitude = len(str(cfg.epochs))
    with tqdm(range(cfg.epochs)) as pbar_epochs:
        for current_epoch in pbar_epochs:
            train_losses = []
            val_losses = []
            start_point_deviations = []
            end_point_deviations = []

            epoch_string = str(current_epoch).zfill(epoch_magnitude)

            with tqdm(train_dataloader, leave=False) as pbar_train:
                # Train for one epoch
                for batch in pbar_train:
                    if not cfg.dataset_fully_on_gpu:
                        batch = dictionary_to_device(batch, cfg.device)
                    loss_value = agent.train_step(batch)
                    train_losses.append(loss_value)

                    pbar_train.set_description(f"Train epoch {epoch_string}/{cfg.epochs}")
                    pbar_train.set_postfix(loss=format_loss(loss_value))

            # Validate for one epoch
            # Switch EMA weights once before the loop instead of per-batch,
            # reducing EMA weight copy ops from 6*N_batches to just 2.
            ema_was_active = agent.use_ema
            if ema_was_active:
                agent.use_ema_weights()
            with tqdm(val_dataloader, leave=False) as pbar_val:
                for batch in pbar_val:
                    if not cfg.dataset_fully_on_gpu:
                        batch = dictionary_to_device(batch, cfg.device)
                    val_loss_value, start_point_deviation, end_point_deviation = agent.evaluate(batch)
                    val_losses.append(val_loss_value)
                    start_point_deviations.append(start_point_deviation)
                    end_point_deviations.append(end_point_deviation)

                    pbar_val.set_description(f"Valid epoch {epoch_string}/{cfg.epochs}")
                    pbar_val.set_postfix(val_loss=format_loss(val_loss_value))
            if ema_was_active:
                agent.restore_model_weights()

            # Log the epoch info
            mean_train_loss = sum(train_losses) / len(train_losses)
            mean_val_loss = sum(val_losses) / len(val_losses)
            mean_start_point_deviation = sum(start_point_deviations) / len(start_point_deviations)
            mean_end_point_deviation = sum(end_point_deviations) / len(end_point_deviations)
            epoch_info = {
                "epoch": current_epoch,
                "loss": mean_train_loss,
                "val_loss": mean_val_loss,
                "start_point_deviation": mean_start_point_deviation,
                "end_point_deviation": mean_end_point_deviation,
                "lr": agent.optimizer.param_groups[0]["lr"],
            }
            swanlab.log(epoch_info)

            # Test agent in workspace every eval_in_env_after_epochs epochs
            is_last_epoch = (current_epoch == cfg.epochs - 1)
            is_eval_epoch = cfg.eval_in_env_after_epochs > 0 and current_epoch > 0 and current_epoch % cfg.eval_in_env_after_epochs == 0
            if is_last_epoch or is_eval_epoch:
                # Set current epoch in workspace for proper directory organization
                workspace.current_epoch = current_epoch
                test_results = workspace.test_agent(agent, cfg.num_trajectories_in_env)
                swanlab.log(test_results)
                pbar_epochs.set_description(f"Epoch {epoch_string}/{cfg.epochs}")
                pbar_epochs.set_postfix(**test_results)
            else:
                test_results = {}

            # Check if the current model is better than the previous best
            combined_epoch_info = {**epoch_info, **test_results}
            # NOTE: We do this check, because the performance_metric could be either in epoch_info or test_results.
            # If there are no test results in this epoch, we do not want to trigger early stopping based on the epoch_info.
            
            if performance_metric in combined_epoch_info:
                if performance_comparison_operator(combined_epoch_info[performance_metric], best_performance_value):
                    best_performance_value = combined_epoch_info[performance_metric]
                    best_epoch = current_epoch
                    current_early_stopping_patience = 0
                    # Write info about the best model to a text file
                    with open(logging_path / "best_model_info.txt", "w") as f:
                        f.write(f"epoch={epoch_string}, {performance_metric=}, {best_performance_value=}, {mean_train_loss=}, {mean_val_loss=}\n")
                    # Overwrite the best model
                    agent.save_model(logging_path / "best_model.pth")
                else:
                    early_stopping_warmup_epochs = cfg.get("early_stopping_warmup_epochs", None)
                    if early_stopping_warmup_epochs is not None and current_epoch >= early_stopping_warmup_epochs:
                        current_early_stopping_patience += 1
                        # Check if training should be stopped due to early stopping
                        early_stopping = cfg.get("early_stopping", False)
                        if early_stopping and current_early_stopping_patience >= cfg.early_stopping_patience:
                            done = True
                            log.log(logging.INFO, f"Early stopping after {current_epoch} epochs with best {performance_metric} of {best_performance_value} at epoch {best_epoch}.")

            # Save intermediate checkpoints of the model
            if cfg.save_distance is not None and current_epoch % cfg.save_distance == 0:
                agent.save_model(logging_path / f"model_epoch_{epoch_string}.pth")

            # Log the current best model metrics
            swanlab.log({f"best_{performance_metric}": best_performance_value})
            swanlab.log({"best_epoch": best_epoch})

            # Check if training should be stopped due to reaching the maximum number of epochs
            if cfg.epochs is not None and current_epoch >= cfg.epochs or done:
                log.log(logging.INFO, f"Finished after {current_epoch} epochs with best {performance_metric} of {best_performance_value} at epoch {best_epoch}.")
                break

    # Save the final model
    # NOTE: This ist not the best model, but the model of the final epoch
    model_name = "model_last_epoch.pth"
    agent.save_model(logging_path / model_name)

    # Close the workspace and it's environments.
    # If that takes longer than 60 seconds, terminate the subproccesses of the vectorized environment.
    if isinstance(workspace, BaseVectorWorkspace):
        workspace.close(timeout=60)
    else:
        workspace.close()

    # Finish the swanlab run
    swanlab_run.finish()

    return best_performance_value


if __name__ == "__main__":
    main()
