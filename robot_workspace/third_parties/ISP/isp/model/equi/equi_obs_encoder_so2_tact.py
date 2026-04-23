import torch
import e3nn
import e3nn.nn
from e3nn import o3
from einops import rearrange

import isp.model.vision.crop_randomizer as dmvc
from isp.model.common.module_attr_mixin import ModuleAttrMixin
from isp.model.common.rotation_transformer import RotationTransformer
from isp.model.equi.i2s_policy import I2SPolicy
from isp.model.equi.isp_util import SO3Conv, so3_irreps, so3_near_identity_grid


class ISPObsEncTact(ModuleAttrMixin):
    """SO(2) ISP observation encoder with tactile scalar injection."""

    def __init__(
        self,
        encoder="equiresnet50",
        obs_shape=(3, 84, 84),
        crop_shape=(76, 76),
        n_hidden=128,
        N=8,
        initialize=True,
        lmax=6,
        s2_fdim=512,
        so3_fdim=24,
        tactile_shape=None,
        tactile_embed_dim=32,
    ):
        super().__init__()
        self.lmax = lmax
        self.enc_obs = I2SPolicy(
            encoder=encoder,
            lmax=lmax,
            s2_fdim=s2_fdim,
            so3_fdim=so3_fdim,
            N=N,
            initialize=initialize,
            f_out=n_hidden,
        )

        so3_kernel_grid = so3_near_identity_grid(n_alpha=4)
        self.enc_out = SO3Conv(n_hidden + 32, n_hidden, lmax, so3_kernel_grid, lmax)

        self.quaternion_to_sixd = RotationTransformer("quaternion", "rotation_6d")
        self.quaternion_to_matrix = RotationTransformer("quaternion", "matrix")

        # NOTE: historical ISP always inserted a CropRandomizer here to map
        # 84x84 inputs down to 76x76. We keep the original lines commented out
        # as a reference and only enable the crop when crop_shape is requested.
        # self.crop_randomizer = dmvc.CropRandomizer(
        #     input_shape=obs_shape,
        #     crop_height=crop_shape[0],
        #     crop_width=crop_shape[1],
        # )
        self.crop_randomizer = None
        if crop_shape is not None:
            self.crop_randomizer = dmvc.CropRandomizer(
                input_shape=obs_shape,
                crop_height=crop_shape[0],
                crop_width=crop_shape[1],
            )

        self.tactile_embed_dim = 0
        self.tactile_encoder = None
        if tactile_shape is not None:
            self.tactile_embed_dim = tactile_embed_dim
            c_in = tactile_shape[0]
            self.tactile_encoder = torch.nn.Sequential(
                torch.nn.Conv2d(c_in, 32, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 64, 3, stride=2, padding=1),
                torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
                torch.nn.Flatten(),
                torch.nn.Linear(64, tactile_embed_dim),
            )

        n_scalars = 2 + self.tactile_embed_dim
        self.proj_1 = o3.Linear(f"{n_scalars}x0e+3x1e", so3_irreps(1), f_in=1, f_out=32)
        self.proj_2 = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), f_in=32, f_out=32)
        self.so3_act = e3nn.nn.SO3Activation(1, lmax, act=torch.relu, resolution=8)

    def get_matrix_rotation(self, quat):
        # data is wxyz, which rotation transformer expects
        return self.quaternion_to_matrix.forward(quat)

    def get_sixd_rotation(self, quat):
        # data is wxyz, which rotation transformer expects
        return self.quaternion_to_sixd.forward(quat)

    def forward(self, nobs):
        ee_pos = nobs["robot0_eef_pos"]
        ee_quat = nobs["robot0_eef_quat"]
        ee_q = nobs["robot0_gripper_qpos"]
        ih = nobs["robot0_eye_in_hand_image"]

        batch_size = ih.shape[0]
        ih = rearrange(ih, "b t c h w -> (b t) c h w")
        ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
        ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
        ee_q = rearrange(ee_q, "b t d -> (b t) d")
        # NOTE: historical ISP always applied the extra 76x76 crop here.
        # ih = self.crop_randomizer(ih)
        if self.crop_randomizer is not None:
            ih = self.crop_randomizer(ih)
        ee_rot_xy = self.get_matrix_rotation(ee_quat).transpose(2, 1)[:, :2, :]

        enc_out = self.enc_obs(ih, ee_quat)
        ee_rot_xy = ee_rot_xy.reshape(enc_out.shape[0], 2 * 3)

        state_dim = 11 + self.tactile_embed_dim
        robot_state_embed = torch.zeros(
            [enc_out.shape[0], 1, state_dim], device=enc_out.device
        )
        robot_state_embed[:, 0, :2] = ee_q
        robot_state_embed[:, 0, 2:5] = ee_pos
        robot_state_embed[:, 0, 5:11] = ee_rot_xy

        if self.tactile_encoder is not None:
            tac_left = nobs["robot0_tactile_left_image"]
            tac_left = rearrange(tac_left, "b t c h w -> (b t) c h w")
            tactile_feat = self.tactile_encoder(tac_left)
            robot_state_embed[:, 0, 11:] = tactile_feat

        robot_state_embed = self.proj_1(robot_state_embed)
        robot_state_embed = self.so3_act(robot_state_embed)
        robot_state_embed = self.proj_2(robot_state_embed)
        features = torch.cat([enc_out, robot_state_embed], dim=1)
        out = self.enc_out(features)
        return rearrange(out, "(b t) c f -> b t c f", b=batch_size)
