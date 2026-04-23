import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from isp.model.diffusion.mask_generator import LowdimMaskGenerator
from isp.model.common.normalizer import LinearNormalizer
from isp.model.common.rotation_transformer import RotationTransformer
from isp.model.equi.equi_conditional_unet1d_c8 import EquiDiffusionUNet
from isp.model.equi.equi_group_sampling import EquiGroupSamplingC8
from isp.model.equi.equi_obs_encoder_so2_tact import ISPObsEncTact
from isp.model.vision.rot_randomizer import RotRandomizer, RotRandomizerForPrediction
from isp.policy.base_image_policy import BaseImagePolicy
from isp.policy.isp_so2_policy import ISPSO2Policy


class ISPSO2TactPolicy(ISPSO2Policy):
    """ISPSO2 policy variant that fuses tactile observations."""

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        crop_shape=(76, 76),
        N=8,
        enc_n_hidden=128,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        cond_predict_scale=True,
        rot_aug=False,
        lmax=6,
        s2_fdim=1024,
        so3_fdim=48,
        initialize=True,
        encoder="equiresnet50",
        tactile_shape=None,
        tactile_embed_dim=32,
        **kwargs,
    ):
        # Skip ISPSO2Policy.__init__ to avoid constructing the non-tactile encoder.
        BaseImagePolicy.__init__(self)

        action_shape = shape_meta["action"]["shape"]
        assert len(action_shape) == 1
        action_dim = action_shape[0]
        obs_shape_meta = shape_meta["obs"]

        self.enc = ISPObsEncTact(
            obs_shape=obs_shape_meta["robot0_eye_in_hand_image"]["shape"],
            crop_shape=crop_shape,
            n_hidden=enc_n_hidden,
            N=8,
            initialize=initialize,
            lmax=lmax,
            s2_fdim=s2_fdim,
            so3_fdim=so3_fdim,
            encoder=encoder,
            tactile_shape=tactile_shape,
            tactile_embed_dim=tactile_embed_dim,
        )

        obs_feature_dim = enc_n_hidden
        global_cond_dim = obs_feature_dim * n_obs_steps
        self.equi_sampler = EquiGroupSamplingC8(lmax=lmax, f_out=obs_feature_dim)

        self.diff = EquiDiffusionUNet(
            act_emb_dim=64,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            N=N,
            lmax=lmax,
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()
        self.rot_randomizer = RotRandomizer()
        self.rot_randomizer2 = RotRandomizerForPrediction()

        self.horizon = horizon
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.crop_shape = crop_shape
        self.obs_feature_dim = obs_feature_dim
        self.rot_aug = rot_aug
        self.kwargs = kwargs

        self.noise_scheduler = noise_scheduler
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.register_buffer("ws_center", torch.tensor([0.0, 0.0, 0.8], dtype=torch.float32))
        self.sixd2mat = RotationTransformer("rotation_6d", "matrix")
