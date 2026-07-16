import e3nn
import torch
from e3nn import o3
from einops import rearrange

import vista.model.vision.crop_randomizer as dmvc
from vista.model.common.module_attr_mixin import ModuleAttrMixin
from vista.model.common.rotation_transformer import RotationTransformer
from vista.model.equi.vista_sphere_policy import VistaSpherePolicy
from vista.model.equi.vista_util import SO3Conv, so3_irreps, so3_near_identity_grid


class VISTAObsEncoder(ModuleAttrMixin):
    def __init__(
        self,
        obs_shape=(3, 84, 84),
        crop_shape=None,
        n_hidden=128,
        N=8,
        initialize=True,
        lmax=6,
        visual_dim=1024,
        tactile_dim=1024,
        fused_dim=1024,
        so3_dim=64,
        encoder="equiresnet50",
        tactile_shape=(3, 84, 84),
        rec_level=3,
        max_beta=1.5707963267948966,
        attention_heads=8,
        attention_head_dim=64,
        tactile_mode="raw",
        **kwargs,
    ):
        super().__init__()
        self.lmax = lmax
        if str(tactile_mode) != "raw":
            raise ValueError(
                "VISTA now supports only tactile_mode='raw' "
                "(tactile RGB image with markers)."
            )
        self.enc_obs = VistaSpherePolicy(
            encoder=encoder,
            lmax=lmax,
            visual_dim=visual_dim,
            tactile_dim=tactile_dim,
            fused_dim=fused_dim,
            so3_dim=so3_dim,
            N=N,
            initialize=initialize,
            f_out=n_hidden,
            tactile_shape=tactile_shape,
            rec_level=rec_level,
            max_beta=max_beta,
            attention_heads=attention_heads,
            attention_head_dim=attention_head_dim,
            tactile_mode=tactile_mode,
        )
        so3_kernel_grid = so3_near_identity_grid(n_alpha=4)
        self.enc_out = SO3Conv(n_hidden + 32, n_hidden, lmax, so3_kernel_grid, lmax)
        self.quaternion_to_matrix = RotationTransformer("quaternion", "matrix")
        self.crop_randomizer = None
        if crop_shape is not None:
            self.crop_randomizer = dmvc.CropRandomizer(
                input_shape=obs_shape,
                crop_height=crop_shape[0],
                crop_width=crop_shape[1],
            )
        self.proj_1 = o3.Linear("2x0e+3x1e", so3_irreps(1), f_in=1, f_out=32)
        self.proj_2 = o3.Linear(so3_irreps(lmax), so3_irreps(lmax), f_in=32, f_out=32)
        self.so3_act = e3nn.nn.SO3Activation(1, lmax, act=torch.relu, resolution=8)

    def getMatrixRotation(self, quat):
        return self.quaternion_to_matrix.forward(quat)

    def forward(self, nobs):
        required = [
            "robot0_eye_in_hand_image",
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        ]
        required.append("robot0_tactile_left_image")
        missing = [key for key in required if key not in nobs]
        if missing:
            raise KeyError(f"Missing VISTA observation keys: {missing}")

        ih = nobs["robot0_eye_in_hand_image"]
        tl = nobs["robot0_tactile_left_image"]
        ee_pos = nobs["robot0_eef_pos"]
        ee_quat = nobs["robot0_eef_quat"]
        ee_q = nobs["robot0_gripper_qpos"]
        batch_size = ih.shape[0]

        ih = rearrange(ih, "b t c h w -> (b t) c h w")
        tl = rearrange(tl, "b t c h w -> (b t) c h w")
        ee_pos = rearrange(ee_pos, "b t d -> (b t) d")
        ee_quat = rearrange(ee_quat, "b t d -> (b t) d")
        ee_q = rearrange(ee_q, "b t d -> (b t) d")

        if self.crop_randomizer is not None:
            ih = self.crop_randomizer(ih)
        ee_rot_xy = self.getMatrixRotation(ee_quat).transpose(2, 1)[:, :2, :]
        enc_128_out = self.enc_obs(
            ih,
            tl,
            ee_quat,
        )

        ee_rot_xy = ee_rot_xy.reshape(enc_128_out.shape[0], 2 * 3)
        robot_state_embed = torch.zeros(
            [enc_128_out.shape[0], 1, 11],
            device=enc_128_out.device,
            dtype=enc_128_out.dtype,
        )
        robot_state_embed[:, 0, :2] = ee_q
        robot_state_embed[:, 0, 2:5] = ee_pos
        robot_state_embed[:, 0, 5:11] = ee_rot_xy

        robot_state_embed = self.proj_1(robot_state_embed)
        robot_state_embed = self.so3_act(robot_state_embed)
        robot_state_embed = self.proj_2(robot_state_embed)
        features = torch.cat([enc_128_out, robot_state_embed], dim=1)
        out = self.enc_out(features)
        return rearrange(out, "(b t) c f -> b t c f", b=batch_size)
