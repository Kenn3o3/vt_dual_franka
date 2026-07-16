# VT Dual Franka commands

纯双臂链路：controller PC 先起 **两套 Polymetis robot/gripper**，再起两个 HTTP controller；workspace 侧只用 `arms.left/right`（无单臂 `controller` 别名）。

## Controller PC

```bash
cd /home/medair/vt_dual_franka/robot_controller
conda activate polymetis-local
```

### Terminal C1: left Polymetis robot server

```bash
conda activate polymetis-local
taskset -c 1,5 launch_robot.py \
  robot_client=franka_hardware \
  robot_client.executable_cfg.robot_ip=172.16.0.2
```

### Terminal C2: left Polymetis gripper server

```bash
conda activate polymetis-local
launch_gripper.py \
  gripper=franka_hand \
  gripper.executable_cfg.robot_ip=172.16.0.2
```

### Terminal C3: right Polymetis robot server

右臂必须用与左臂不同的本地 Polymetis robot port；`controller_right.yaml` 期望 `127.0.0.1:50061`，且 **`robot_ip` 必须是 `172.16.1.2`**（错写成 `172.16.0.2` 会出现 “right controller 控制 left Franka”）。

```bash
conda activate polymetis-local
taskset -c 2,6 launch_robot.py \
  robot_client=franka_hardware \
  robot_client.executable_cfg.robot_ip=172.16.1.2 \
  port=50061
```

### Terminal C4: right Polymetis gripper server

```bash
conda activate polymetis-local
launch_gripper.py \
  gripper=franka_hand \
  gripper.executable_cfg.robot_ip=172.16.1.2 \
  port=50062
```

端口不变量：

```text
left  robot server   -> 127.0.0.1:50051  + Franka 172.16.0.2
left  gripper server -> 127.0.0.1:50052
right robot server   -> 127.0.0.1:50061  + Franka 172.16.1.2
right gripper server -> 127.0.0.1:50062
```

对 `launch_robot.py robot_client=franka_hardware`，覆盖顶层 `port=50061` 会同时更新本地 bind 与 Franka control port。

### Terminal C5: dual Controller API

两个终端分别启动（或用 `scripts/run_dual_controllers.sh`）：

```bash
conda activate polymetis-local
cd /home/medair/vt_dual_franka/robot_controller
export PYTHONPATH=../shared/src:src:${PYTHONPATH:-}
taskset -c 0,4 python -m vt_dual_franka_controller.cli run \
  --config config/controller_left.yaml
```

```bash
conda activate polymetis-local
cd /home/medair/vt_dual_franka/robot_controller
export PYTHONPATH=../shared/src:src:${PYTHONPATH:-}
taskset -c 3,7 python -m vt_dual_franka_controller.cli run \
  --config config/controller_right.yaml
```

健康检查（应分别报 `arm_id: left` / `right`，并带 `expected_physical_robot_ip`）：

```bash
curl -s http://127.0.0.1:8092/api/v1/health
curl -s http://127.0.0.1:8093/api/v1/health
```

## Workspace PC / workstation

```bash
conda activate vt-dual-franka-workspace
cd /home/zhenya/kenny/visuotact/vt_dual_franka
```

Quest USB：

```bash
adb devices
adb reverse tcp:8082 tcp:8082
adb shell settings put global stay_on_while_plugged_in 3
adb shell am broadcast -a com.oculus.vrpowermanager.prox_close --ei timeout 0
adb shell setprop debug.oculus.guardian_pause 1
```

Quest TactAR 工作站 IP 设为 `127.0.0.1`。

仅双臂遥操作（不录数据）：

```bash
vt-dual-franka-workspace teleop \
  --workspace-config robot_workspace/config/workspace.yaml
```

采集（`BimanualDataCollector` + 左右初始姿 `task.initial_poses`）：

```bash
vt-dual-franka-workspace collect \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task bimanual_demo
```

对齐为双臂 20D commanded-action dataset：

```bash
vt-dual-franka-workspace make-dataset \
  robot_workspace/data/collect/bimanual_demo \
  --name real_bimanual_demo \
  --target-hz 10 \
  --overwrite
```

action label 须来自 commanded action 同步，而不是 observed next EE。

代码 rsync 到 remote（建议独立目录，exclude 大数据）：

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

数据集 rsync：

```bash
rsync -avh --info=progress2 \
  -e "ssh -i ~/.ssh/zlkenny_yzy673_ed25519 -p 538 -o IdentitiesOnly=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6" \
  /mnt/kenny_ssd/vt_dual_franka/data/datasets/bimanual_demo/real_bimanual_demo/ \
  zlkenny@120.48.58.215:/mnt/pfs_cuhk/kenny/vt_dual_franka/robot_workspace/data/datasets/bimanual_demo/real_bimanual_demo/
```

Remote 训练（`dp_bimanual`）：

```bash
tmux new -s bimanual_demo_dp

cd /mnt/pfs_cuhk/kenny/vt_dual_franka
export PYTHONPATH=robot_workspace/src:shared/src:${PYTHONPATH:-}
conda activate isp

CUDA_VISIBLE_DEVICES=7 python -m vt_dual_franka_workspace.cli train \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name bimanual_demo \
  --dataset-name real_bimanual_demo \
  --dataset-dir robot_workspace/data/datasets/bimanual_demo/real_bimanual_demo \
  --checkpoint-dir robot_workspace/data/checkpoints/bimanual_demo/dp_bimanual \
  --device cuda:0 \
  --extra-arg training.checkpoint_every=30
```

下载 checkpoint 到本机：

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

本机双臂推理：

```bash
conda activate vt-dual-franka-workspace
cd /home/zhenya/kenny/visuotact/vt_dual_franka

vt-dual-franka-workspace run-policy \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task bimanual_demo \
  --inference-config robot_workspace/config/inference/bimanual_demo_dp.yaml \
  --policy-config robot_workspace/config/policies/dp_bimanual_demo.yaml
```
