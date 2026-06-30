from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from dp.model.common.normalizer import LinearNormalizer
from dp.model.diffusion.mask_generator import LowdimMaskGenerator
from dp.model.equi.manifeel_equi_conditional_unet1d_vel import EquiDiffusionUNetVel
from dp.model.equi.manifeel_equi_obs_encoder import EquivariantObsEnc
from dp.policy.base_image_policy import BaseImagePolicy


class DiffusionEquiUNetCNNEncRelPolicy(BaseImagePolicy):
    """ManiFeel EquiDiff tactile baseline adapted to UniVTAC's 10D action format."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        resize_shape=(3, 230, 230),
        crop_shape=(224, 224),
        N=8,
        enc_n_hidden=128,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        **kwargs,
    ):
        super().__init__()

        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) != 1 or int(action_shape[0]) != 10:
            raise ValueError(f"EquiDiff-Tact expects UniVTAC action shape [10], got {action_shape}.")

        self.action_dim = 10
        self.equi_action_dim = 13
        self.enc = EquivariantObsEnc(
            obs_shape_meta=shape_meta["obs"],
            obs_shape=resize_shape,
            crop_shape=crop_shape,
            n_hidden=enc_n_hidden,
            N=N,
        )
        obs_feature_dim = enc_n_hidden
        global_cond_dim = obs_feature_dim * n_obs_steps
        self.diff = EquiDiffusionUNetVel(
            action_dim=self.equi_action_dim,
            act_emb_dim=64,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            N=N,
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.equi_action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_feature_dim = obs_feature_dim
        self.noise_scheduler = noise_scheduler
        self.kwargs = kwargs
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def get_optimizer(
        self,
        weight_decay: float,
        learning_rate: float | None = None,
        lr: float | None = None,
        betas: Tuple[float, float] = (0.95, 0.999),
        eps: float = 1.0e-8,
    ) -> torch.optim.Optimizer:
        learning_rate = lr if learning_rate is None else learning_rate
        return torch.optim.AdamW(
            self.parameters(),
            weight_decay=weight_decay,
            lr=learning_rate,
            betas=betas,
            eps=eps,
        )

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _to_equi_action(self, action_10d: torch.Tensor) -> torch.Tensor:
        pos = action_10d[..., :3]
        rot6d = action_10d[..., 3:9]
        grip = action_10d[..., 9:10]
        matrix = rotation_6d_to_matrix(rot6d).reshape(action_10d.shape[:-1] + (9,))
        return torch.cat([pos, matrix, grip], dim=-1)

    def _from_equi_action(self, action_13d: torch.Tensor) -> torch.Tensor:
        pos = action_13d[..., :3]
        matrix = action_13d[..., 3:12].reshape(action_13d.shape[:-1] + (3, 3))
        rot6d = matrix_to_rotation_6d(matrix)
        grip = action_13d[..., 12:13]
        return torch.cat([pos, rot6d, grip], dim=-1)

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.diff(
                trajectory,
                timestep,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = self.noise_scheduler.step(
                model_output,
                timestep,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _global_condition(self, nobs: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        nobs_features = self.enc(nobs)
        return nobs_features[:, : self.n_obs_steps].reshape(batch_size, -1)

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]

        global_cond = self._global_condition(nobs, batch_size)
        cond_data = torch.zeros(
            size=(batch_size, self.horizon, self.equi_action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        nsample_equi = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )
        nsample = self._from_equi_action(nsample_equi)
        action_pred = self.normalizer["action"].unnormalize(nsample)
        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        global_cond = self._global_condition(nobs, batch_size)
        trajectory = self._to_equi_action(nactions)
        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        pred = self.diff(
            noisy_trajectory,
            timesteps,
            local_cond=None,
            global_cond=global_cond,
        )
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "sample":
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        loss = loss * (~condition_mask).type(loss.dtype)
        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean()
