# VT Dual Franka commands

下面是 dual Franka 的当前推荐链路。和单臂 `vt_franka` 不同，controller PC 侧需要先启动 **两套 Polymetis robot/gripper server**，然后再启动两个 `vt-dual-franka-controller` HTTP API。

## Controller PC

每个 Controller PC terminal 都先进入 controller 代码目录并激活 Polymetis 环境：

```bash
source /home/zhenya/miniforge3/etc/profile.d/conda.sh
conda activate polymetis-local
cd /home/zhenya/kenny/visuotact/vt_dual_franka/robot_controller
```

### Terminal C1: left Polymetis robot server

```bash
launch_robot.py \
  robot_client=franka_hardware \
  robot_client.executable_cfg.robot_ip=172.16.0.2
```

### Terminal C2: left Polymetis gripper server

```bash
launch_gripper.py \
  gripper=franka_hand \
  gripper.executable_cfg.robot_ip=172.16.0.2
```

### Terminal C3: right Polymetis robot server

The right arm must use a different local Polymetis robot server port from the left arm. `controller_right.yaml` currently expects `127.0.0.1:50061`.

```bash
launch_robot.py \
  robot_client=franka_hardware \
  robot_client.executable_cfg.robot_ip=172.16.1.2 \
  port=50061 \
  robot_client.executable_cfg.server_address=localhost:50061
```

### Terminal C4: right Polymetis gripper server

The right gripper must use a different local Polymetis gripper server port from the left gripper. `controller_right.yaml` currently expects `127.0.0.1:50062`.

```bash
launch_gripper.py \
  gripper=franka_hand \
  gripper.executable_cfg.robot_ip=172.16.1.2 \
  port=50062
```

The important invariant is:

```text
left  robot server   -> 127.0.0.1:50051
left  gripper server -> 127.0.0.1:50052
right robot server   -> 127.0.0.1:50061
right gripper server -> 127.0.0.1:50062
```

For `launch_robot.py`, both `port=50061` and `robot_client.executable_cfg.server_address=localhost:50061` are needed: the first changes the server bind port, and the second makes the Franka hardware client connect to that same local server.

### Terminal C5: dual Controller API

```bash
source /home/zhenya/miniforge3/etc/profile.d/conda.sh
conda activate polymetis-local
cd /home/zhenya/kenny/visuotact/vt_dual_franka/robot_controller

export PYTHONPATH=../shared/src:src:${PYTHONPATH:-}
python scripts/preflight_dual_network.py
scripts/run_dual_controllers.sh
```

Expected Controller API endpoints:

```bash
curl http://127.0.0.1:8092/api/v1/health
curl http://127.0.0.1:8093/api/v1/health
```

They should report `arm_id: left` and `arm_id: right`, respectively.

## Workspace PC / workstation

启动环境：
```bash
conda activate vt-dual-franka-workspace
cd /home/zhenya/kenny/visuotact/vt_dual_franka
```

用 USB 连接 Meta Quest 到电脑

```bash
adb devices
adb reverse tcp:8082 tcp:8082
adb shell settings put global stay_on_while_plugged_in 3
adb shell am broadcast -a com.oculus.vrpowermanager.prox_close --ei timeout 0
adb shell setprop debug.oculus.guardian_pause 1
```

In the Quest TactAR app, set the workstation IP to:

```text
127.0.0.1
```

用这个command采集数据：
```bash
vt-dual-franka-workspace collect \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task bimanual_demo
```

之后对齐数据用这个command，生成双臂 20D baseline 可用的 commanded-action dataset：

```bash
vt-dual-franka-workspace make-bimanual-dataset \
  robot_workspace/data/collect/bimanual_demo \
  --name real_bimanual_demo \
  --target-hz 10 \
  --overwrite
```

里面的action label需要是真的commanded action同步得出的而不是从observed next ee得出的。

之后把所有代码相关的文件 rsync 到 remote PC 上面（建议使用独立 remote 目录，例如 `/mnt/pfs_cuhk/kenny/vt_dual_franka`，exclude 所有大文件包括 dataset）

```bash
rsync -avh --info=progress2 \
  -e "ssh -i ~/.ssh/zlkenny -p 538 -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  --exclude='.git/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='robot_workspace/data' \
  /home/zhenya/kenny/visuotact/vt_dual_franka/ \
  zlkenny@120.48.58.215:/mnt/pfs_cuhk/kenny/vt_dual_franka/
```

之后把统一好的可训练数据 e.g. `real_bimanual_demo` rsync 到 remote PC 上面

```bash
rsync -avh --info=progress2 \
  -e "ssh -i ~/.ssh/zlkenny_yzy673_ed25519 -p 538 -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  /mnt/kenny_ssd/vt_dual_franka/data/datasets/bimanual_demo/real_bimanual_demo/ \
  zlkenny@120.48.58.215:/mnt/pfs_cuhk/kenny/vt_dual_franka/robot_workspace/data/datasets/bimanual_demo/real_bimanual_demo/
```

之后不同的模型可以在remote直接通过比如以下的命令进行训练：

```bash
tmux new -s bimanual_demo_dp

cd /mnt/pfs_cuhk/kenny/vt_dual_franka
export PYTHONPATH=robot_workspace/src:shared/src:${PYTHONPATH:-}
conda activate isp

CUDA_VISIBLE_DEVICES=7 python -m vt_dual_franka_workspace.cli train-visuotactile \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name bimanual_demo \
  --model dp_bimanual \
  --dataset-name real_bimanual_demo \
  --dataset-dir robot_workspace/data/datasets/bimanual_demo/real_bimanual_demo \
  --checkpoint-dir robot_workspace/data/checkpoints/bimanual_demo/dp_bimanual \
  --device cuda:0 \
  --extra-arg training.checkpoint_every=30
```

这里训练会自动生成模型specific的dataset cache，应该用的action也是commanded action而不是observed next ee。

训练结束后

用比如以下的指令下载需要模型的checkpoint到local pc：

```bash
mkdir -p /mnt/kenny_ssd/vt_dual_franka/data/checkpoints/bimanual_demo/dp_bimanual/checkpoints

rsync -avh --info=progress2 \
  -e "ssh -i ~/.ssh/zlkenny_yzy673_ed25519 -p 538 -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  --exclude='backend_dataset/' \
  --exclude='checkpoints/' \
  zlkenny@120.48.58.215:/mnt/pfs_cuhk/kenny/vt_dual_franka/robot_workspace/data/checkpoints/bimanual_demo/dp_bimanual/ \
  /mnt/kenny_ssd/vt_dual_franka/data/checkpoints/bimanual_demo/dp_bimanual/

rsync -avh --info=progress2 \
  -e "ssh -i ~/.ssh/zlkenny_yzy673_ed25519 -p 538 -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  zlkenny@120.48.58.215:/mnt/pfs_cuhk/kenny/vt_dual_franka/robot_workspace/data/checkpoints/bimanual_demo/dp_bimanual/checkpoints/epoch=209.ckpt \
  zlkenny@120.48.58.215:/mnt/pfs_cuhk/kenny/vt_dual_franka/robot_workspace/data/checkpoints/bimanual_demo/dp_bimanual/checkpoints/epoch=209.info.json \
  /mnt/kenny_ssd/vt_dual_franka/data/checkpoints/bimanual_demo/dp_bimanual/checkpoints/
```

然后直接在local跑推理：

```bash
source /home/zhenya/miniforge3/etc/profile.d/conda.sh
conda activate vt-dual-franka-workspace
export PYTHONPATH=$PWD/robot_workspace/src:$PWD/shared/src:$PYTHONPATH

python -m vt_dual_franka_workspace.cli run-policy \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task bimanual_demo \
  --inference-config robot_workspace/config/inference/bimanual_demo_dp.yaml \
  --policy-config robot_workspace/config/policies/dp_bimanual_demo.yaml
```
