# Movement-Primitive-Diffusion: 从数据采集到部署的完整流程

## 概览

```
数据采集 (vt_franka teleop, 10 Hz, impedance control)
    ↓
数据转换 (vt_franka aligned NPZ → MPD NPZ + PNG)
    ↓
训练 (movement-primitive-diffusion, UNet1D diffusion policy)
    ↓
部署 (vt_franka RolloutSupervisor, impedance control)
    ↓
[待解决] temporal ensemble 平滑 action chunk 边界
```

---

## 0. 环境准备 (只需做一次)

```bash
cd /home/zhenya/kenny/visuotact/vt_franka
bash kenny/scripts/setup_mpd_env.sh
```

还需要在 mpd 环境中安装 vt_franka 依赖和 Orbbec SDK:
```bash
conda activate mpd
pip install -e shared/
pip install -e robot_workspace/
pip install /home/zhenya/kenny/visuotact/vt_franka/third_party/pyorbbecsdk2-2.0.18-cp310-cp310-linux_x86_64.whl
```

如果 PyTorch CUDA 版本不匹配 (driver 12.9 vs torch cu130), 降级:
```bash
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

验证:
```bash
conda activate mpd
python -c "import movement_primitive_diffusion; print('OK')"
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

MPD repo 中有一个 import 需要修复 (`diffusers` 版本兼容), 已改:
```
# movement_primitive_diffusion/utils/lr_scheduler.py
# 原: from diffusers.optimization import Union, ...
# 改: from typing import Union, Optional; from torch.optim import Optimizer
```

---

## 1. 数据采集

```bash
cd /home/zhenya/kenny/visuotact/vt_franka/robot_workspace
vt-franka-workspace collect --run put_cup_on_plate
```

数据保存在: `robot_workspace/data/runs/put_cup_on_plate_YYYYMMDD_HHMMSS/`

当前数据: `put_cup_on_plate_20260422_171246` (44 episodes)

---

## 2. 数据转换

```bash
cd /home/zhenya/kenny/visuotact/vt_franka

python kenny/scripts/convert_vt_franka_to_mpd.py \
    --run-dir robot_workspace/data/runs/put_cup_on_plate_20260422_171246 \
    --output-dir robot_workspace/data/mpd/put_cup_on_plate \
    --val-episodes 2
```

输出:
```
robot_workspace/data/mpd/put_cup_on_plate/
├── train/demo_000/ ...
├── val/demo_000/ ...
├── scaler_values.npz          # 归一化参数 (推理时需要)
└── conversion_manifest.json
```

每个 demo: `tcp_pose.npz` (T,7), `gripper_width.npz` (T,1), `actions.npz` (T,8), `rgb_wrist/` 和 `rgb_third_person/` (320x240 PNG)

注意: 偶尔第一帧图像缺失 (相机启动延迟), 转换脚本自动跳过。

---

## 3. 训练

### 3.1 复制配置到 MPD repo

```bash
cd /home/zhenya/kenny/visuotact/vt_franka
cp -r kenny/configs/mpd/put_cup_on_plate \
    robot_workspace/third_parties/movement-primitive-diffusion/conf/experiments/put_cup_on_plate
```

注意: `kenny/configs/` 是源, mpd repo 里的 copy 是训练实际读取的。修改后需重新 cp。

### 3.2 启动训练

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

### 3.3 训练参数 (`train_defaults.yaml`)

- `t_obs=3`, `t_pred=10`, `predict_past=True` (action tensor 长度 12, 可被 4 整除)
- `batch_size=128`, `epochs=300`, `save_distance=40`
- 图像: 320x240 -> crop 288x216, DependentTimeStepsResNet, embedding_size=128
- 模型: UNet1D diffusion, 10 inference steps, EMA, cosine LR

### 3.4 训练产物

```
wandb/latest-run/files/
├── best_model.pth, model_epoch_040.pth, ...

outputs/<date>/<time>/.hydra/config.yaml   # 推理时需要
```

---

## 4. 部署 (推理)

### 4.1 配置

rollout 配置在 `robot_workspace/config/workspace.yaml` 的 `rollout` 部分:

```yaml
rollout:
  control_hz: 10.0
  max_duration_sec: 30.0
  policy:
    entrypoint: "kenny.policies.mpd_diffusion_policy:create_policy"
    kwargs:
      checkpoint_path: "/path/to/best_model.pth"
      config_path: "/path/to/outputs/.hydra/config.yaml"
      scaler_path: "/path/to/scaler_values.npz"
      device: "cuda"
      confirm_before_execute: false
    inputs:
      controller_state: true
      rgb_cameras: [wrist, third_person]
```

### 4.2 启动 rollout

```bash
cd /home/zhenya/kenny/visuotact/vt_franka/robot_workspace
conda activate mpd
PYTHONPATH=/home/zhenya/kenny/visuotact/vt_franka:$PYTHONPATH \
    vt-franka-workspace rollout --run put_cup_on_plate_dp
```

操作: `H`=reset+开夹爪, `R`=开始, `E`=保存, `D`=丢弃, `Q`=退出

### 4.3 逐步确认模式

`confirm_before_execute: true` 时每步打印 proprioception + proposed action, 按 ENTER 执行, `s` 跳过, `q` 终止。