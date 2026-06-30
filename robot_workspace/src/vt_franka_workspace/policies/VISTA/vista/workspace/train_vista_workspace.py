if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import json
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from typing import List, Optional, Tuple
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from vista.workspace.base_workspace import BaseWorkspace
from vista.policy.base_image_policy import BaseImagePolicy
from vista.dataset.base_dataset import BaseImageDataset
from vista.common.checkpoint_util import TopKCheckpointManager
from vista.common.json_logger import JsonLogger
from vista.common.pytorch_util import dict_apply, optimizer_to
from vista.model.diffusion.ema_model import EMAModel
from vista.model.common.lr_scheduler import get_scheduler

OmegaConf.register_new_resolver("eval", eval, replace=True)


def _init_wandb_with_fallback(cfg: OmegaConf, output_dir):
    logging_cfg = OmegaConf.to_container(cfg.logging, resolve=True)

    try:
        return wandb.init(
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **logging_cfg,
        )
    except Exception as exc:
        error_text = str(exc).lower()
        requested_mode = str(logging_cfg.get("mode", "online")).lower()
        should_fallback = requested_mode == "online" and (
            "permission denied" in error_text or "403" in error_text
        )
        if not should_fallback:
            raise

        print(
            "[wandb] online init failed with a permission error; "
            "falling back to offline mode."
        )
        try:
            wandb.finish(quiet=True)
        except Exception:
            pass

        offline_logging_cfg = dict(logging_cfg)
        offline_logging_cfg["mode"] = "offline"
        return wandb.init(
            dir=str(output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **offline_logging_cfg,
        )


class TrainVISTAWorkspace(BaseWorkspace):
    include_keys = ["global_step", "epoch"]

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: BaseImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: BaseImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        total_params = 0
        for param_group in self.optimizer.param_groups:
            for param in param_group["params"]:
                total_params += param.numel()
        assert total_params == sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        if cfg.training.get("allow_tf32", True):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if cfg.training.get("cudnn_benchmark", True):
            torch.backends.cudnn.benchmark = True # “你看到同样形状的卷积输入时，花一点时间先 benchmark 一下几种算法，然后挑最快的那个。”

        # resume training
        if cfg.training.resume:
            resume_ckpt_path = _resolve_resume_checkpoint(self.output_dir)
            if resume_ckpt_path is not None:
                print(f"Resuming from checkpoint {resume_ckpt_path}")
                self.load_checkpoint(path=resume_ckpt_path)
                self.epoch += 1
                self.global_step += 1

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()
        
        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        self.model.set_normalizer(normalizer)
        self.model.set_ws_center(torch.from_numpy(dataset.ws_center).float())
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)
            self.ema_model.set_ws_center(torch.from_numpy(dataset.ws_center).float())

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs)
            // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step - 1,
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # configure logging
        wandb_run = _init_wandb_with_fallback(cfg, self.output_dir)
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )

        # configure checkpoint
        checkpoint_mode = str(cfg.training.get("checkpoint_mode", "topk_val_loss"))
        use_milestone_checkpoints = checkpoint_mode == "milestone_train_loss"
        topk_manager = None
        if not use_milestone_checkpoints:
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk
            )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        log_every_steps = max(1, int(cfg.training.get("log_every_steps", 10)))

        # training loop
        log_path = os.path.join(self.output_dir, "logs.json.txt")
        with JsonLogger(log_path) as json_logger:
            while self.epoch < cfg.training.num_epochs:
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(
                    train_dataloader,
                    desc=f"Training epoch {self.epoch}",
                    leave=False,
                    mininterval=cfg.training.tqdm_interval_sec,
                ) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):

                        # del batch['obs']['agentview_image']
                        # device transfer
                        batch = dict_apply(
                            batch, lambda x: x.to(device, non_blocking=True)
                        )

                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        raw_loss = self.model.compute_loss(batch)
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        loss.backward()

                        # step optimizer
                        if (
                            self.global_step % cfg.training.gradient_accumulate_every
                            == 0
                        ):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            "train_loss": raw_loss_cpu,
                            "global_step": self.global_step,
                            "epoch": self.epoch,
                            "lr": lr_scheduler.get_last_lr()[0],
                        }

                        is_last_batch = batch_idx == (len(train_dataloader) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            if (self.global_step % log_every_steps) == 0:
                                wandb_run.log(step_log, step=self.global_step)
                                json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) and batch_idx >= (
                            cfg.training.max_train_steps - 1
                        ):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log["train_loss"] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if (not use_milestone_checkpoints) and (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(
                            val_dataloader,
                            desc=f"Validation epoch {self.epoch}",
                            leave=False,
                            mininterval=cfg.training.tqdm_interval_sec,
                        ) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                # del batch['obs']['agentview_image']
                                batch = dict_apply(
                                    batch, lambda x: x.to(device, non_blocking=True)
                                )
                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (
                                    cfg.training.max_val_steps is not None
                                ) and batch_idx >= (cfg.training.max_val_steps - 1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log["val_loss"] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        if "agentview_image" in train_sampling_batch["obs"]:
                            del train_sampling_batch["obs"]["agentview_image"]
                        batch = dict_apply(
                            train_sampling_batch,
                            lambda x: x.to(device, non_blocking=True),
                        )

                        obs_dict = batch["obs"]
                        gt_action = batch["action"]

                        result = policy.predict_action(obs_dict)
                        pred_action = result["action_pred"]
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log["train_action_mse_error"] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                if use_milestone_checkpoints:
                    _save_milestone_checkpoint_if_needed(self, step_log, cfg)
                else:
                    if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if (self.epoch % cfg.training.checkpoint_every) == 0 and cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace("/", "_")
                        metric_dict[new_key] = value

                    # Track the top-k best checkpoint on every validation epoch.
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)
                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path, use_thread=False)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1


def _resolve_resume_checkpoint(output_dir) -> Optional[pathlib.Path]:
    ckpt_dir = pathlib.Path(output_dir) / "checkpoints"
    candidates = [
        ckpt_dir / "latest.ckpt",
    ]
    candidates.extend(_sorted_epoch_checkpoints(ckpt_dir))
    candidates.extend([ckpt_dir / "best.ckpt", pathlib.Path(output_dir) / "best.ckpt"])
    for path in candidates:
        if path.is_file():
            return path
    return None


def _sorted_epoch_checkpoints(ckpt_dir: pathlib.Path) -> List[pathlib.Path]:
    def epoch_key(path: pathlib.Path) -> Tuple[int, str]:
        try:
            return int(path.stem.split("=", 1)[1]), path.name
        except (IndexError, ValueError):
            return -1, path.name

    return sorted(ckpt_dir.glob("epoch=*.ckpt"), key=epoch_key, reverse=True) if ckpt_dir.is_dir() else []


def _save_milestone_checkpoint_if_needed(workspace, step_log: dict, cfg: OmegaConf) -> None:
    checkpoint_every = int(cfg.training.checkpoint_every)
    if checkpoint_every <= 0:
        return
    epoch = int(workspace.epoch)
    if ((epoch + 1) % checkpoint_every) != 0:
        return
    ckpt_dir = pathlib.Path(workspace.output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch={epoch:03d}.ckpt"
    workspace.save_checkpoint(path=ckpt_path, use_thread=False)
    if cfg.checkpoint.save_last_ckpt:
        workspace.save_checkpoint(path=ckpt_dir / "latest.ckpt", use_thread=False)
    _write_checkpoint_info(ckpt_path, step_log, monitor_key="train_loss", mode="milestone_train_loss")


def _write_checkpoint_info(ckpt_path: pathlib.Path, step_log: dict, *, monitor_key: str, mode: str) -> None:
    payload = {
        "checkpoint": ckpt_path.name,
        "monitor_key": monitor_key,
        "mode": mode,
    }
    if monitor_key in step_log:
        payload["monitor_value"] = float(step_log[monitor_key])
    for key in ("epoch", "global_step", "train_loss", "val_loss", "lr", "train_action_mse_error"):
        if key in step_log:
            value = step_log[key]
            try:
                payload[key] = float(value)
            except (TypeError, ValueError):
                payload[key] = value
    info_path = ckpt_path.with_suffix(".info.json")
    tmp_path = info_path.with_name(f".tmp.{info_path.name}")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, info_path)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem,
)
def main(cfg):
    workspace = TrainVISTAWorkspace(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
