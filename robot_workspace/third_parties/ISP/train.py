"""
Usage:
Training:
EGL_DEVICE_ID=0 MUJOCO_EGL_DEVICE_ID=0 python train.py --config-name=train_isp_so2 task_name=stack_three_d1 n_demo=100 training.device=0 training.seed=1

if cuda out of memory, try:
MUJOCO_GL=osmesa PYOPENGL_PLATTFORM=osmesa python train.py --config-name=train_isp_so2 task_name=stack_three_d1 n_demo=100 training.device=0 training.seed=1
"""

import sys

# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from isp.workspace.base_workspace import BaseWorkspace

# UniVTAC tasks
VALID_TASKS = {
    "grasp_classify",
    "insert_HDMI",
    "insert_HDMI_D1",
    "insert_HDMI_D2",
    "insert_hole",
    "insert_tube",
    "insert_tube_D1",
    "insert_tube_D2",
    "lift_bottle",
    "lift_can",
    "pull_out_key",
    "put_bottle_in_shelf",
    "put_bottle_in_shelf_D1",
    "put_bottle_in_shelf_D2",
}


def get_ws_x_center(task_name):
    return 0.0


def get_ws_y_center(task_name):
    return 0.0


def get_ws_z_center(task_name):
    return 0.8


OmegaConf.register_new_resolver("get_ws_x_center", get_ws_x_center, replace=True)
OmegaConf.register_new_resolver("get_ws_y_center", get_ws_y_center, replace=True)
OmegaConf.register_new_resolver("get_ws_z_center", get_ws_z_center, replace=True)

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath("isp", "config")),
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)
    if "task_name" in cfg and cfg.task_name not in VALID_TASKS:
        raise ValueError(f"Unknown UniVTAC task_name: {cfg.task_name}")

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()


if __name__ == "__main__":
    main()
