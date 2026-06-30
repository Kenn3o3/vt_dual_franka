# VT Franka system gap analysis

## 结论
你的目标链路整体是合理的。

需要保留的是：
- `collect` 记录原始 teleop / controller state / image streams。
- `make-dataset` 负责把 raw collect 对齐成统一 common dataset。
- `controller_state` 仍然应该作为 observation / proprioception 输入。

需要改掉的是：
- 训练 label 不能再从 `embodiment/ee[1:]` 这种“next observed ee”重建。
- `run-policy` 不能继续依赖 policy-config 作为主入口。
- eval 输出目录不能再按 `policy_family/policy_name` 组织。
- checkpoint / runtime discovery 里不能继续保留隐式 fallback 搜索。

## 需求是否合理

合理，但有一个边界要保留：
- 你要替换的是 **action label source**，不是 observation source。
- 也就是说，训练时的 action 应该来自 commanded action / teleop command target。
- 但 qpos / proprioception 仍然必须来自实际观察到的 controller state。

## 当前系统 vs 目标系统

| Area | Current behavior | Desired behavior | Keep / Change / Delete |
|---|---|---|---|
| `collect` | 记录 `teleop_commands`、`controller_state`、image streams | 继续保留 | Keep |
| `make-dataset` | 用 `teleop_commands.target_tcp` 生成 common dataset 的 `action`，并保留 `controller_state.tcp_pose` | 这一层基本正确 | Keep, document clearly |
| `prepare-visuotactile` | common branch 把 `controller_state.tcp_pose` 写成 `qpos`，把 `action.target_tcp` 写成模型 label | 基本符合你的目标 | Keep |
| `export_backend` | 写出 `action`、`embodiment/ee_action`、`embodiment/ee` | 需要保证 downstream 只消费 command action 作为 label | Keep storage, change consumers |
| VISTA/DP dataset loader | 直接用 `embodiment/ee[1:]` 重建 action label | 必须改成读取 `action` 或 `embodiment/ee_action` | Change, delete next-ee reconstruction |
| ACT / ViTAL unified training path | 统一训练入口已经走 backend HDF5 / `action` 字段 | 大体可接受，但要确认所有入口都不再绕回 legacy preprocess scripts | Keep unified path, delete legacy defaults |
| `run-policy` CLI | 仍然要求 `--policy-config` + `--inference-config` | checkpoint-first，policy-config 退场 | Change |
| eval 目录 | `data/eval/<policy_family>/<policy_name>/datetime` | `data/eval/<task_name>/<model_name>/datetime` | Change |
| checkpoint resolution | 允许隐式目录搜索、best/latest/epoch fallback | 只能显式 resolve 到指定 checkpoint / checkpoint dir | Delete fallback search |
| remote sync | 现在是工具化，但不是统一“代码 + dataset + checkpoint”流程 | 固定成可重复的 remote/local pipeline | Change |

## 需要改的代码逻辑

| File | Why it matters | What to change |
|---|---|---|
| `robot_workspace/src/vt_franka_workspace/datasets/common.py` | common dataset 的 action 对齐定义在这里 | 保持 `teleop_commands.target_tcp` 作为 label source，补充明确的 schema / manifest 语义 |
| `robot_workspace/src/vt_franka_workspace/policies/common/visuotactile/prepare.py` | 这里把 common dataset 转成模型可读格式 | 保证写入的 `action_pose*` 都来自 commanded action，不再引入 observed-next-ee 语义 |
| `robot_workspace/src/vt_franka_workspace/policies/common/visuotactile/export_backend.py` | backend HDF5 的 `action` / `embodiment/ee_action` 在这里生成 | 保持 command action 写入，但确保下游默认读取 `action` 或 `ee_action` |
| `robot_workspace/src/vt_franka_workspace/policies/VISTA/vista/dataset/univtac_replay_image_dataset.py` | 当前 VISTA 训练 label 明确来自 `ee[1:]` | 改为读 backend `action` / `embodiment/ee_action` |
| `robot_workspace/src/vt_franka_workspace/policies/DP/dp/dataset/univtac_replay_image_dataset.py` | DP 训练 label 同样来自 `ee[1:]` | 同上 |
| `robot_workspace/src/vt_franka_workspace/policies/common/visuotactile/runtime.py` | 负责 runtime decode / rot6d convention / input preprocessing | 保持唯一一套 convention，去掉隐式兼容分支和模糊兜底 |
| `robot_workspace/src/vt_franka_workspace/policies/common/visuotactile/policy.py` | backend 选择仍然是 probe-based fallback | 改成显式 checkpoint type / manifest 驱动 |
| `robot_workspace/src/vt_franka_workspace/inference/policy_runner.py` | 目前 eval group 仍按 policy family/name 分组 | 改成 task/model 分组，并记录 checkpoint 语义 |
| `robot_workspace/src/vt_franka_workspace/cli.py` | `run-policy` 仍以 policy-config 为主 | 改成 checkpoint-first CLI，policy-config 变成可选或删除 |
| `robot_workspace/src/vt_franka_workspace/policies/common/visuotactile/remote.py` | 远程同步/下载还不是你要的固定 pipeline | 统一成明确的 code sync / dataset sync / checkpoint sync / local eval 入口 |

## 需要删掉或停止默认使用的旧逻辑

- `VISTA` / `DP` loader 里从 `embodiment/ee[1:]` 直接造 action label 的逻辑。
- `resolve_runtime_checkpoint_dir()` 这种“目录里找一个唯一 manifest 就算成功”的隐式探测。
- `vendor_*_runtime` 里按 `best.ckpt`、`latest.ckpt`、`epoch=*.ckpt` 自动兜底的默认搜索。
- `run-policy` 仍然依赖 policy YAML 作为主入口的 legacy 路径。
- `policy_family/policy_name` 作为 eval 主目录的默认命名。
- `ACT/process_data*.py`、`ViTAL/process_data*.py` 这类旧 preprocessing 脚本作为默认训练入口。

## 建议的最终形态

1. `collect` 只记录原始数据。
2. `make-dataset` 只生成 command-driven common dataset。
3. `prepare/export_backend` 只负责格式转换，不改 action 语义。
4. 各模型训练统一消费 `action` / `embodiment/ee_action`，不再自己重建 label。
5. `run-policy` 直接吃 checkpoint。
6. eval 输出目录以 `task/model/datetime` 固定化。

