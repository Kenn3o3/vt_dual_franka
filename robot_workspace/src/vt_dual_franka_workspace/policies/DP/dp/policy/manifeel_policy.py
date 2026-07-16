from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from einops import reduce

from dp.common.pytorch_util import dict_apply
from dp.model.common.normalizer import LinearNormalizer
from dp.model.diffusion.conditional_unet1d import ConditionalUnet1D
from dp.model.diffusion.mask_generator import LowdimMaskGenerator
from dp.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from dp.policy.base_image_policy import BaseImagePolicy


class ManiFeelPolicy(BaseImagePolicy):
    """ManiFeel-style visual-tactile Diffusion Policy.

    This baseline follows ManiFeel's plain DP fusion path: wrist RGB, tactile
    RGB, and proprioception are encoded by independent ResNet branches through
    MultiImageObsEncoder, concatenated, and used as the global condition for a
    1D diffusion policy. It intentionally does not use UniVTAC encoders, ISP
    encoders, or tactile force-field inputs.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        obs_encoder: MultiImageObsEncoder,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        num_inference_steps: int | None = None,
        diffusion_step_embed_dim: int = 256,
        down_dims=(256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
        **kwargs,
    ):
        super().__init__()
        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = int(action_shape[0])

        obs_feature_dim = obs_encoder.output_shape()[0]
        global_cond_dim = obs_feature_dim * n_obs_steps
        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.normalizer = LinearNormalizer()
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        condition_mask: torch.Tensor,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ) -> torch.Tensor:
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for t in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                trajectory,
                t,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = self.noise_scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
                **kwargs,
            ).prev_sample
        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    def _global_condition(self, nobs: Dict[str, torch.Tensor], batch_size: int) -> torch.Tensor:
        this_nobs = dict_apply(
            nobs,
            lambda x: x[:, : self.n_obs_steps, ...].reshape(-1, *x.shape[2:]),
        )
        nobs_features = self.obs_encoder(this_nobs)
        return nobs_features.reshape(batch_size, -1)

    @torch.no_grad()
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        assert "past_action" not in obs_dict
        nobs = self.normalizer.normalize(obs_dict)
        value = next(iter(nobs.values()))
        batch_size = value.shape[0]

        global_cond = self._global_condition(nobs, batch_size)
        cond_data = torch.zeros(
            size=(batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        nsample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            **self.kwargs,
        )
        naction_pred = nsample[..., : self.action_dim]
        action_pred = self.normalizer["action"].unnormalize(naction_pred)

        start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        return {
            "action": action_pred[:, start:end],
            "action_pred": action_pred,
        }

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def set_ws_center(self, ws_center: torch.Tensor):
        del ws_center

    def compute_loss(self, batch):
        assert "valid_mask" not in batch
        nobs = self.normalizer.normalize(batch["obs"])
        nactions = self.normalizer["action"].normalize(batch["action"])
        batch_size = nactions.shape[0]

        global_cond = self._global_condition(nobs, batch_size)
        trajectory = nactions
        cond_data = trajectory
        condition_mask = self.mask_generator(trajectory.shape)

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, noise, timesteps)
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        pred = self.model(
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
