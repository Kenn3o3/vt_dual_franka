# vt_franka Pipeline 完整分析

## 1. 双 PC 架构概览

系统由两台 PC 协同工作:

- **Controller PC** (robot_controller): 运行低层实时控制, 直连 Franka 机械臂
- **Workspace PC** (robot_workspace): 运行传感器采集、遥操作、数据记录、策略部署

通信方式: Workspace PC 通过 HTTP 调用 Controller PC 的 FastAPI 服务 (`192.168.217.180:8092`)

---

## 2. Data Collection 时各 PC 的数据流

### 2.1 Controller PC 发送/接收什么

| 方向 | 内容 | 频率 | 协议 |
|------|------|------|------|
| 接收 (来自 Workspace) | TCP 目标位姿命令 `TcpTargetCommand` | 60 Hz | HTTP POST `/api/v1/commands/tcp` |
| 接收 (来自 Workspace) | 夹爪命令 | 按需 | HTTP POST `/api/v1/commands/gripper/*` |
| 发送 (给 Workspace) | `ControllerState` (机器人全状态) | 60 Hz (被轮询) | HTTP GET `/api/v1/state` |
| 内部 | 笛卡尔阻抗控制循环 | **300 Hz** | 本地 Polymetis gRPC |

**Controller PC 不储存任何数据**, 它只做实时控制和状态缓存。

### 2.2 Workspace PC 发送/接收什么

| 方向 | 内容 | 频率 | 目标 |
|------|------|------|------|
| 轮询 Controller | 机器人状态 | 60 Hz | HTTP GET -> controller_state.jsonl |
| 接收 Quest | 遥操作手部位姿 | ~60 Hz | HTTP POST from Quest headset |
| 发送 Controller | TCP 目标位姿 | 60 Hz | HTTP POST -> Controller PC |
| 发送 Quest | 机器人状态反馈 | 60 Hz | UDP -> Quest (192.168.217.221:10001) |
| 发送 Quest | 触觉可视化 | ~12 Hz | UDP -> Quest (port 10002) |
| 本地采集 | Orbbec RGB (wrist) | 30 FPS 采集, **15 Hz 记录** | 本地存储 JPG |
| 本地采集 | Orbbec RGB (third_person) | 30 FPS 采集, **15 Hz 记录** | 本地存储 JPG |
| 本地采集 | GelSight 触觉 | **15 Hz** | 本地存储 JSONL |

---

## 3. 频率控制机制

所有频率都在 YAML 配置文件中定义, 可以直接修改:

**Controller 端** (`robot_controller/config/controller.yaml`):
```yaml
control:
  control_frequency_hz: 300.0   # 底层阻抗控制循环
  teleop_command_hz: 60.0       # 遥操作命令默认持续时间 = 1/60 秒
  state_cache_hz: 60.0          # 状态缓存刷新频率
```

**Workspace 端** (`robot_workspace/config/workspace.yaml`):
```yaml
teleop:
  loop_hz: 60.0                 # 遥操作处理循环
quest_feedback:
  state_publish_hz: 60.0        # 状态发布到 Quest
gelsight:
  fps: 15                       # GelSight 采集帧率
  record_hz: 0.0                # 0 = 不限制记录频率, 跟随采集
rgb_cameras:
  wrist:
    color_fps: 30               # 相机硬件采集帧率
    record_hz: 15.0             # 实际记录频率 (降采样)
  third_person:
    color_fps: 30
    record_hz: 15.0
recording:
  postprocess_target_hz: 10.0   # 后处理对齐目标频率
collect:
  sync_hz: 10.0                 # 采集同步循环频率
  controller_state_poll_hz: 60.0
rollout:
  control_hz: 12.0              # 策略推理控制频率
```

**频率限制实现方式**: `JsonlStreamRecorder._should_record()` 通过时间戳差值判断是否应该记录当前帧, 实现基于时间的降采样。各传感器循环使用 `precise_sleep(period)` 控制循环频率。

---

## 4. Data Collection 全流程

### 4.1 启动流程

1. **Controller PC** 启动三个进程:
   - Polymetis robot server (gRPC 50051) - Franka 机械臂驱动
   - Polymetis gripper server (gRPC 50052) - 夹爪驱动
   - `vt-franka-controller run` (FastAPI 8092) - 300Hz 控制循环 + HTTP API

2. **Workspace PC** 启动一个命令:
   ```bash
   vt-franka-workspace collect --config workspace.yaml --run fold_cloth
   ```
   这会启动: 状态桥接、遥操作服务器、传感器采集、Operator UI (8083)

### 4.2 Episode 生命周期

1. 操作员按 `H` 将机器人移到 ready pose
2. 按 `R` 开始新 episode (有 2 秒倒计时)
3. 操作员通过 Quest 遥操作控制机器人
4. 按 `E` 结束并保存 episode
5. 自动后处理生成 `aligned_episode.npz`
6. 自动 QC 检查
7. 按 `D` 可丢弃最近保存的 episode

### 4.3 数据流 (采集期间)

```
Quest Headset ---(HTTP 60Hz)---> Workspace PC (teleop server)
                                      |
                                      | queue_tcp (HTTP POST 60Hz)
                                      v
                              Controller PC (300Hz 控制循环)
                                      |
                                      | Polymetis gRPC
                                      v
                                 Franka 机械臂
                                      |
                                      | 状态反馈
                                      v
                              Controller PC (状态缓存 60Hz)
                                      |
                                      | HTTP GET 60Hz
                                      v
                              Workspace PC (state bridge)
                                      |
                          +-----------+-----------+
                          |           |           |
                     记录 JSONL   UDP 反馈    Quest 可视化
                                  到 Quest
```

---

## 5. 储存格式与目录结构

### 5.1 目录结构

```
data/runs/
└── fold_cloth_20260415_185010/          # run 目录
    ├── run_manifest.json                 # run 元数据
    ├── operator_events.jsonl             # 操作员事件日志
    ├── latest_status.json                # 最新状态快照
    └── episodes/
        └── episode_0000/                 # 单个 episode
            ├── episode_manifest.json     # episode 元数据
            ├── aligned_episode.npz       # 后处理对齐数据 (训练用)
            ├── aligned_episode_manifest.json
            └── streams/                  # 原始数据流
                ├── controller_state.jsonl
                ├── teleop_commands.jsonl
                ├── quest_messages.jsonl   # (可选)
                ├── gelsight_markers.jsonl # (可选)
                ├── rgb_wrist.jsonl        # 图像元数据
                ├── rgb_wrist/             # 图像文件
                │   ├── 000000.jpg
                │   ├── 000001.jpg
                │   └── ...
                ├── rgb_third_person.jsonl
                └── rgb_third_person/
                    ├── 000000.jpg
                    └── ...
```

### 5.2 各 JSONL 流的内容

**controller_state.jsonl** (每条记录):
```json
{
  "source_wall_time": 1713200000.123,
  "source_monotonic_time": 12345.678,
  "received_wall_time": 1713200000.125,
  "state": {
    "tcp_pose": [x, y, z, qw, qx, qy, qz],
    "tcp_velocity": [vx, vy, vz, wx, wy, wz],
    "tcp_wrench": [fx, fy, fz, tx, ty, tz],
    "joint_positions": [q1, q2, q3, q4, q5, q6, q7],
    "joint_velocities": [dq1, dq2, dq3, dq4, dq5, dq6, dq7],
    "gripper_width": 0.04,
    "gripper_force": 2.5,
    "wall_time": 1713200000.123,
    "monotonic_time": 12345.678,
    "control_frequency_hz": 300.0
  }
}
```

**teleop_commands.jsonl** (每条记录):
```json
{
  "source_wall_time": 1713200000.130,
  "target_tcp": [x, y, z, qw, qx, qy, qz],
  "gripper_closed": false
}
```

**rgb_wrist.jsonl** (每条记录):
```json
{
  "camera_name": "wrist_orbbec_rgb",
  "serial_number": "CP0E753000KA",
  "captured_wall_time": 1713200000.140,
  "device_timestamp_us": 123456789,
  "frame_path": "rgb_wrist/000042.jpg",
  "frame_width": 640,
  "frame_height": 480,
  "color_format": "RGB"
}
```

**gelsight_markers.jsonl** (每条记录):
```json
{
  "captured_wall_time": 1713200000.150,
  "marker_locations": [[nx1, ny1], [nx2, ny2], ...],
  "marker_offsets": [[dx1, dy1], [dx2, dy2], ...]
}
```

### 5.3 各模态的记录频率

| 模态 | 采集频率 | 记录频率 | 配置位置 |
|------|----------|----------|----------|
| controller_state | 60 Hz (轮询) | 60 Hz | `collect.controller_state_poll_hz` |
| teleop_commands | 60 Hz | 60 Hz (或由 `command_record_hz` 限制) | `teleop.loop_hz` |
| rgb_wrist | 30 FPS | **15 Hz** | `rgb_cameras.wrist.record_hz` |
| rgb_third_person | 30 FPS | **15 Hz** | `rgb_cameras.third_person.record_hz` |
| gelsight_markers | 15 FPS | **15 Hz** | `gelsight.fps` / `gelsight.record_hz` |
| quest_messages | ~60 Hz | 默认关闭 (0 Hz) | `teleop.quest_message_record_hz` |

### 5.4 时间戳

每条记录都有时间戳, 使用 `time.time()` (wall clock):
- Controller state: `source_wall_time` (Controller PC 上的采集时间)
- Teleop commands: `source_wall_time` (命令发出时间)
- RGB cameras: `captured_wall_time` (帧捕获时间)
- GelSight: `captured_wall_time` (帧捕获时间)

**注意**: 两台 PC 之间没有硬件时钟同步, 依赖 NTP 或系统时钟。

---

## 6. 数据后处理 (Postprocessing)

后处理由 `align_episode()` 函数完成 (`robot_workspace/src/vt_franka_workspace/recording/postprocess.py`), 在 episode 结束后自动运行 (如果 `auto_postprocess: true`)。

### 6.1 对齐算法

1. **读取原始流**: controller_state.jsonl, teleop_commands.jsonl, gelsight_markers.jsonl, rgb_*.jsonl

2. **创建时间网格**:
   - 起点: 第一条 controller_state 的时间戳
   - 终点: 最后一条 controller_state 的时间戳
   - 步长: `1.0 / target_hz` (默认 10 Hz, 由 `postprocess_target_hz` 配置)
   - `grid = np.arange(start_time, end_time + step * 0.5, step)`

3. **因果对齐策略** (`alignment_mode: "causal_observation_future_action"`):
   - **Observation (因果)**: 取网格时间点**之前或等于**的最新 controller_state
   - **Action (未来)**: 取网格时间点**之后**的第一条 teleop_command
   - **GelSight**: 取网格时间点之前或等于的最新触觉数据
   - **RGB**: 取网格时间点之前或等于的最新图像帧

4. **丢弃规则**:
   - 没有未来 action 的步 -> 丢弃
   - action_lead_sec <= 0 -> 丢弃
   - action_lead_sec > action_horizon (默认 = step) -> 丢弃

### 6.2 输出: aligned_episode.npz

```python
{
    # 时间戳
    "timestamps": (N,),                          # float64, 对齐网格时间

    # 观察空间 - 本体感知
    "robot_tcp_pose": (N, 7),                    # float64, [x, y, z, qw, qx, qy, qz]
    "robot_tcp_velocity": (N, 6),                # float64, [vx, vy, vz, wx, wy, wz]
    "robot_tcp_wrench": (N, 6),                  # float64, [fx, fy, fz, tx, ty, tz]
    "robot_joint_positions": (N, 7),             # float64, 7 个关节角度 (rad)
    "robot_joint_velocities": (N, 7),            # float64, 7 个关节角速度
    "gripper_width": (N,),                       # float64, 夹爪宽度 (m)
    "gripper_force": (N,),                       # float64, 夹爪力 (N)
    "gripper_state": (N, 2),                     # float64, [width, force]

    # 动作空间
    "teleop_target_tcp": (N, 7),                 # float64, [x, y, z, qw, qx, qy, qz]
    "teleop_gripper_closed": (N,),               # bool, 夹爪是否闭合

    # 触觉 (可选)
    "gelsight_marker_locations": (N,),           # object array, 每个元素是 marker 2D 位置列表
    "gelsight_marker_offsets": (N,),             # object array, 每个元素是 marker 2D 偏移列表

    # 图像路径 (可选)
    "rgb_wrist_frame_paths": (N,),               # object array, 相对路径字符串
    "rgb_third_person_frame_paths": (N,),        # object array

    # 审计/调试字段
    "controller_state_valid": (N,),              # bool
    "controller_state_age_sec": (N,),            # float64, observation 的延迟
    "controller_state_source_timestamps": (N,),  # float64
    "teleop_command_source_timestamps": (N,),    # float64
    "teleop_action_lead_sec": (N,),              # float64, action 的提前量
    "gelsight_capture_timestamps": (N,),         # float64
    "rgb_wrist_capture_timestamps": (N,),        # float64
    "rgb_third_person_capture_timestamps": (N,), # float64
}
```

---

## 7. 观察空间 (Observation Space)

### 7.1 本体感知 (Proprioception)

来源: Controller PC 的 `ControllerState`, 通过 HTTP 轮询获取。

| 字段 | 维度 | 单位 | 说明 |
|------|------|------|------|
| `tcp_pose` | 7 | m + 四元数 | 末端执行器位姿 [x, y, z, qw, qx, qy, qz] |
| `tcp_velocity` | 6 | m/s + rad/s | 末端速度 [vx, vy, vz, wx, wy, wz] |
| `tcp_wrench` | 6 | N + Nm | 末端力/力矩 [fx, fy, fz, tx, ty, tz] |
| `joint_positions` | 7 | rad | Franka 7 自由度关节角度 |
| `joint_velocities` | 7 | rad/s | 关节角速度 |
| `gripper_width` | 1 | m | 夹爪开合宽度 (0 ~ 0.078m) |
| `gripper_force` | 1 | N | 夹爪力 |

**总维度**: 7 + 6 + 6 + 7 + 7 + 1 + 1 = **35 维**

### 7.2 视觉 (RGB Cameras)

| 相机 | 分辨率 | 格式 | 采集/记录频率 |
|------|--------|------|---------------|
| wrist (手腕 Orbbec) | 640 x 480 | RGB JPG | 30 FPS / 15 Hz |
| third_person (第三人称 Orbbec) | 640 x 480 | RGB JPG | 30 FPS / 15 Hz |

图像维度: `(480, 640, 3)` uint8

### 7.3 触觉 (GelSight)

| 字段 | 维度 | 说明 |
|------|------|------|
| `marker_locations` | (M, 2) | 归一化 2D marker 位置 |
| `marker_offsets` | (M, 2) | 归一化 2D marker 偏移量 |

M = marker 数量 (约 9 个, 其中 marker 8 被 mask 掉)。GelSight 分辨率 640x480, 15 FPS。

### 7.4 Rollout 时的 Observation 字典

策略在 rollout 时收到的 observation 字典结构:

```python
observation = {
    "controller_state": {
        "tcp_pose": [x, y, z, qw, qx, qy, qz],      # 7D
        "tcp_velocity": [vx, vy, vz, wx, wy, wz],     # 6D
        "tcp_wrench": [fx, fy, fz, tx, ty, tz],        # 6D
        "joint_positions": [q1, ..., q7],               # 7D
        "joint_velocities": [dq1, ..., dq7],            # 7D
        "gripper_width": 0.04,                          # 1D
        "gripper_force": 2.5,                           # 1D
        "wall_time": ...,
        "monotonic_time": ...,
        "control_frequency_hz": 300.0,
    },
    # 以下为可选, 取决于 rollout.policy.inputs 配置
    "wrist": {                                          # rgb_cameras 中配置的 role
        "image": np.ndarray,                            # (480, 640, 3) uint8
        "metadata": {...},
        "captured_wall_time": ...,
    },
    "third_person": {
        "image": np.ndarray,
        "metadata": {...},
        "captured_wall_time": ...,
    },
    "gelsight_markers": {
        "marker_locations": [[nx, ny], ...],
        "marker_offsets": [[dx, dy], ...],
        "metadata": {...},
        "captured_wall_time": ...,
    },
    "gelsight_frame": {
        "image": np.ndarray,                            # GelSight 原始图像
        "metadata": {...},
        "captured_wall_time": ...,
    },
}
```

---

## 8. 动作空间 (Action Space)

### 8.1 训练数据中的动作

在 `aligned_episode.npz` 中:

| 字段 | 维度 | 说明 |
|------|------|------|
| `teleop_target_tcp` | (N, 7) | 目标末端位姿 [x, y, z, qw, qx, qy, qz] |
| `teleop_gripper_closed` | (N,) | 夹爪闭合布尔值 |

**动作总维度**: 7 (TCP 位姿) + 1 (夹爪) = **8 维**

动作来源: 遥操作期间, Quest 手柄的位姿经过坐标变换后转为机器人目标位姿, 夹爪状态由 trigger 按钮控制。

### 8.2 策略输出的动作格式

策略函数需要返回:

```python
action = {
    "target_tcp": [x, y, z, qw, qx, qy, qz],  # 7D, 必须
    "gripper_closed": True/False,                 # 布尔, 二选一
    # 或者
    "gripper_width": 0.05,                        # float (m), 二选一
    # 可选
    "gripper_velocity": 0.1,                      # 默认 0.1
    "gripper_force_limit": 5.0,                   # 默认 5.0
    "terminate": True,                            # 可选, 终止 episode
}
```

### 8.3 动作如何被执行

策略输出 action 后, 执行链路:

```
Policy 输出 action dict
    |
    v
RolloutSupervisor._execute_action()
    |
    +-- controller.queue_tcp(target_tcp, source="rollout")
    |       |
    |       v
    |   HTTP POST /api/v1/commands/tcp -> Controller PC
    |       |
    |       v
    |   ControllerService.queue_tcp_command()
    |       |
    |       v
    |   command_queue.append({target_pose, target_time})
    |       |
    |       v
    |   300Hz 控制循环: PoseTrajectoryInterpolator 插值
    |       |
    |       v
    |   backend.update_desired_tcp(interpolated_pose)
    |       |
    |       v
    |   Polymetis 笛卡尔阻抗控制 -> Franka 机械臂执行
    |
    +-- controller.grasp_gripper() 或 controller.move_gripper()
            |
            v
        HTTP POST /api/v1/commands/gripper/* -> Controller PC
            |
            v
        Polymetis 夹爪控制 -> Franka Hand 执行
```

关键细节:
- `target_duration_sec` 默认为 `1/teleop_command_hz` = 1/60 秒
- 300Hz 控制循环使用 `PoseTrajectoryInterpolator` 在 waypoint 之间平滑插值
- 笛卡尔阻抗控制参数: stiffness=[750, 750, 750, 15, 15, 15], damping=[37, 37, 37, 2, 2, 2]
- 夹爪命令在独立线程中执行, 不阻塞主控制循环

---

## 9. 模型部署 (Policy Deployment)

### 9.1 当前部署架构

策略通过 `module:function` 的 Python callable 入口点加载:

```bash
vt-franka-workspace rollout \
  --config workspace.yaml \
  --run fold_cloth_policy \
  --policy local_policies.my_policy:rollout_policy
```

Rollout 循环 (默认 12 Hz):
1. `ObservationAssembler.assemble()` 收集当前 observation
2. `policy(observation)` 调用策略得到 action
3. `_execute_action(action)` 发送到 Controller PC
4. `precise_sleep()` 等待到下一个控制周期
5. 每步记录到 `policy_steps.jsonl`

策略可以通过属性自定义频率:
```python
def my_policy(observation):
    ...
my_policy.__vt_franka_control_hz__ = 10.0
my_policy.__vt_franka_max_duration_sec__ = 60.0
```

### 9.2 当前架构的合理性分析

**合理的方面**:
- 控制与感知分离 (300Hz 实时控制 vs 12Hz 策略推理) 是标准做法
- HTTP API 解耦了两台 PC, 方便独立开发和调试
- `module:function` 入口点灵活, 不限制模型框架
- 笛卡尔阻抗控制 + 位姿插值提供了平滑的运动执行
- Rollout 记录了完整的 policy step 数据, 方便调试

**潜在问题**:
- HTTP 通信引入 ~1-5ms 延迟, 对于高频控制可能不够
- 12 Hz 策略频率对于需要快速反应的任务可能偏低
- 没有 action chunking 支持 (每步只输出一个 action)
- 两台 PC 之间没有硬件时钟同步
- 当前没有内置的 observation history buffer (需要策略自己维护)

### 9.3 如何部署 pi0.5 等模型

要部署 pi0.5 或类似的 VLA (Vision-Language-Action) 模型, 需要编写一个 wrapper:

```python
# local_policies/pi0_wrapper.py
import torch
import numpy as np

class Pi0Policy:
    def __init__(self):
        self.model = None
        self.obs_history = []

    def _load_model(self):
        # 加载 pi0.5 checkpoint
        # 这里需要根据 pi0.5 的具体实现来写
        from pi0 import Pi0Model
        self.model = Pi0Model.from_pretrained("path/to/checkpoint")
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

    def reset(self):
        """每个 episode 开始时调用"""
        self.obs_history = []

    def __call__(self, observation: dict) -> dict:
        if self.model is None:
            self._load_model()

        # 1. 提取观察
        state = observation["controller_state"]
        tcp_pose = np.array(state["tcp_pose"])       # (7,)
        gripper = state["gripper_width"]              # scalar

        # 2. 处理图像 (如果有)
        image = None
        if "wrist" in observation:
            image = observation["wrist"]["image"]     # (480, 640, 3) uint8

        # 3. 构建模型输入
        # 根据 pi0.5 的输入格式转换
        proprio = np.concatenate([tcp_pose, [gripper]])  # (8,)

        # 4. 推理
        with torch.no_grad():
            action = self.model.predict(
                image=image,
                proprio=proprio,
                # language_instruction="fold the cloth",
            )

        # 5. 转换为 vt_franka 动作格式
        return {
            "target_tcp": action[:7].tolist(),        # [x, y, z, qw, qx, qy, qz]
            "gripper_closed": bool(action[7] > 0.5),  # 或 "gripper_width"
        }

# 入口点函数
_policy = Pi0Policy()

def rollout_policy(observation: dict) -> dict:
    return _policy(observation)

rollout_policy.__vt_franka_control_hz__ = 10.0
rollout_policy.__vt_franka_max_duration_sec__ = 60.0
rollout_policy.reset = _policy.reset
```

运行:
```bash
export PYTHONPATH=/home/zhenya/kenny/visuotact/vt_franka:$PYTHONPATH
vt-franka-workspace rollout \
  --config workspace.yaml \
  --run pi0_fold_cloth \
  --policy local_policies.pi0_wrapper:rollout_policy
```

**部署不同模型的关键步骤**:
1. 在 `workspace.yaml` 的 `rollout.policy.inputs` 中启用模型需要的输入模态
2. 编写 wrapper 将 observation dict 转为模型输入格式
3. 编写 wrapper 将模型输出转为 `{target_tcp, gripper_closed/gripper_width}` 格式
4. 如果模型需要 action chunking, 在 wrapper 中维护 action buffer
5. 如果模型需要 observation history, 在 wrapper 中维护 history buffer
6. 如果模型有不同的依赖, 可以创建独立的 conda 环境

**需要注意的坐标系问题**:
- `target_tcp` 的四元数格式是 `[x, y, z, qw, qx, qy, qz]` (position + quaternion)
- 训练数据中的 action 是绝对位姿, 不是相对位移
- 如果模型输出相对动作 (delta), 需要在 wrapper 中加上当前位姿

---

## 10. 总结: 数据维度速查表

| 类别 | 字段 | 维度 | 频率 |
|------|------|------|------|
| **观察 - 本体感知** | tcp_pose | 7 | 60 Hz |
| | tcp_velocity | 6 | 60 Hz |
| | tcp_wrench | 6 | 60 Hz |
| | joint_positions | 7 | 60 Hz |
| | joint_velocities | 7 | 60 Hz |
| | gripper_width | 1 | 60 Hz |
| | gripper_force | 1 | 60 Hz |
| **观察 - 视觉** | wrist RGB | (480, 640, 3) | 15 Hz |
| | third_person RGB | (480, 640, 3) | 15 Hz |
| **观察 - 触觉** | marker_locations | (M, 2) | 15 Hz |
| | marker_offsets | (M, 2) | 15 Hz |
| **动作** | target_tcp | 7 | 60 Hz (teleop) / 12 Hz (rollout) |
| | gripper_closed | 1 (bool) | 同上 |
| **对齐数据** | aligned_episode.npz | N steps | 10 Hz (postprocess_target_hz) |






