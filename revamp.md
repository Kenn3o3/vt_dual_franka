我现在正在做vt_franka的项目你可以参考一下以下这些文档作为navigator:
- /home/zhenya/kenny/visuotact/vt_franka/docs/data_collection_pipeline.md
- /home/zhenya/kenny/visuotact/vt_franka/docs/mpd.md
我们这个vt_franka里面有两个部分:data collection 和 policy inference的pipeline.
现在默认robot_controller的代码在一台robot controller pc 上面跑, robot_workspace的代码在一台workspace pc上面跑.

现在的数据采集和推理的流程是这样的:

Robot Controller PC:

Terminal C1:

```bash
conda activate polymetis-local
cd /home/medair/vt_franka/fairo/polymetis
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.0.2
```

Terminal C2:

```bash
conda activate polymetis-local
cd /home/medair/vt_franka/fairo/polymetis
launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=172.16.0.2
```

Terminal C3:

```bash
conda activate polymetis-local
cd /home/medair/vt_franka/robot_controller
vt-franka-controller run --config /home/medair/vt_franka/robot_controller/config/controller.yaml
```

Workspace PC:

```bash
conda activate vt-franka-workspace
cd /home/zhenya/kenny/visuotact/vt_franka/robot_workspace
vt-franka-workspace collect \
  --config /home/zhenya/kenny/visuotact/vt_franka/robot_workspace/config/workspace.yaml \
  --run task_name
```

这里采集完数据

训练和推理评估mdp模型：

请参考 /home/zhenya/kenny/visuotact/vt_franka/docs/mpd.md

之后会有更多其他模型加入：

请参考：/home/zhenya/kenny/visuotact/vt_franka/robot_workspace/third_parties/mpd （这个repo现在可能有点乱）

数据转换

```bash
cd /home/zhenya/kenny/visuotact/vt_franka

python kenny/scripts/convert_vt_franka_to_mpd.py \
    --run-dir robot_workspace/data/runs/put_cup_on_plate_20260422_171246 \
    --output-dir robot_workspace/data/mpd/put_cup_on_plate \
    --val-episodes 2
```
启动训练

```bash
conda activate mpd
cd /home/zhenya/kenny/visuotact/vt_franka/robot_workspace/third_parties/movement-primitive-diffusion

python scripts/train.py \
    --config-name dummy \
    +experiments/put_cup_on_plate=train \
    train_trajectory_dir=/home/zhenya/kenny/visuotact/vt_franka/robot_workspace/data/mpd/put_cup_on_plate/train \
    val_trajectory_dir=/home/zhenya/kenny/visuotact/vt_franka/robot_workspace/data/mpd/put_cup_on_plate/val \
    wandb.project=vt-franka-diffusion \
    wandb.entity=roccendalanda-cuhk
```

cd /home/zhenya/kenny/visuotact/vt_franka/robot_workspace
conda activate mpd
PYTHONPATH=/home/zhenya/kenny/visuotact/vt_franka:$PYTHONPATH \
    vt-franka-workspace rollout --run put_cup_on_plate_dp

请你专注看vt_franka这个repo的训练和推理pipeline。帮我看看现在的代码算不算工整（clean and modular），还是写的很乱。

我之后的代码还需要扩展到更多模型包括/home/zhenya/kenny/visuotact/vt_franka/robot_workspace/third_parties/mpd 里面的模型