"""Agent for official Freqpolicy"""
from pathlib import Path
import torch
from omegaconf import DictConfig
from typing import Dict, Tuple, Union, Optional
from movement_primitive_diffusion.agents.base_agent import BaseAgent


class FreqpolicyAgent(BaseAgent):
    """Agent using official Freqpolicy implementation."""
    
    def __init__(
        self,
        model_config: DictConfig,
        optimizer_config: DictConfig,
        lr_scheduler_config: DictConfig,
        encoder_config: DictConfig,
        process_batch_config: DictConfig,
        t_obs: int,
        predict_past: bool,
        ema_config: DictConfig = None,
        use_ema: bool = False,
        device: Union[str, torch.device] = "cpu",
        special_optimizer_function: bool = False,
        special_optimizer_config: Optional[DictConfig] = None,
    ):
        super().__init__(
            process_batch_config=process_batch_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            lr_scheduler_config=lr_scheduler_config,
            encoder_config=encoder_config,
            ema_config=ema_config,
            use_ema=use_ema,
            device=device,
            special_optimizer_function=special_optimizer_function,
            special_optimizer_config=special_optimizer_config,
        )
        self.t_obs = t_obs
        self.predict_past = predict_past
    
    def train_step(self, batch: Dict) -> float:
        action, observation, extra_inputs = self.process_batch(batch)
        self.model.train()
        self.encoder.train()
        state = self.encoder(observation)
        loss = self.model(action, state)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.lr_scheduler.step()
        if self.use_ema:
            self.ema_model.step(self.model.parameters())
            self.ema_encoder.step(self.encoder.parameters())
        return loss.item()
    
    @torch.no_grad()
    def predict(self, observation: Dict, extra_inputs: Dict) -> torch.Tensor:
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_encoder.store(self.encoder.parameters())
            self.ema_model.copy_to(self.model.parameters())
            self.ema_encoder.copy_to(self.encoder.parameters())
        self.model.eval()
        self.encoder.eval()
        state = self.encoder(observation)
        action = self.model.sample(state, num_samples=1)
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())
        if self.predict_past:
            return action[:, self.t_obs-1:, :]
        return action
    
    @torch.no_grad()
    def evaluate(self, batch: Dict) -> Tuple[float, float, float]:
        if self.use_ema:
            self.ema_model.store(self.model.parameters())
            self.ema_encoder.store(self.encoder.parameters())
            self.ema_model.copy_to(self.model.parameters())
            self.ema_encoder.copy_to(self.encoder.parameters())
        self.model.eval()
        self.encoder.eval()
        action, observation, extra_inputs = self.process_batch(batch)
        state = self.encoder(observation)
        eval_loss = self.model(action, state)
        predicted_action = self.model.sample(state, num_samples=1)
        start_point_deviation = torch.linalg.norm(
            action[:, 0, :] - predicted_action[:, 0, :], dim=-1
        ).mean()
        end_point_deviation = torch.linalg.norm(
            action[:, -1, :] - predicted_action[:, -1, :], dim=-1
        ).mean()
        if self.use_ema:
            self.ema_model.restore(self.model.parameters())
            self.ema_encoder.restore(self.encoder.parameters())
        return eval_loss.item(), start_point_deviation.item(), end_point_deviation.item()
    
    def load_pretrained(self, path: Union[str, Path]):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict["model"])
        self.encoder.load_state_dict(state_dict["encoder"])
        if self.use_ema:
            self.ema_model.load_state_dict(state_dict["ema_model"])
            self.ema_encoder.load_state_dict(state_dict["ema_encoder"])
        if "optimizer" in state_dict:
            self.optimizer.load_state_dict(state_dict["optimizer"])
        if "lr_scheduler" in state_dict:
            self.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
    
    def save_model(self, path: Union[str, Path], save_optimizer: bool = False, save_lr_scheduler: bool = False):
        state_dict = {
            "model": self.model.state_dict(),
            "encoder": self.encoder.state_dict()
        }
        if self.use_ema:
            state_dict["ema_model"] = self.ema_model.state_dict()
            state_dict["ema_encoder"] = self.ema_encoder.state_dict()
        if save_optimizer:
            state_dict["optimizer"] = self.optimizer.state_dict()
        if save_lr_scheduler:
            state_dict["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(state_dict, path)
