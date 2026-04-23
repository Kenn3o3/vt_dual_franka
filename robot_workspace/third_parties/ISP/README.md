# ISP Backend

`policy/ISP/` 只保留 ISP backend 本体。

用户侧不要直接走这里的旧脚本，而是统一使用 repo 根目录的新入口：

## Training

```bash
bash scripts/train_isp.sh <EXPERIMENT_NAME> <config_name> <n_demo> all
```

例如：

```bash
bash scripts/train_isp.sh BIGFOV_EXPERIMENT so3 100 insert_HDMI
```

对应编排入口：

```bash
python -m univtac.train.isp <task_name> --experiment-name <EXPERIMENT_NAME> --config-name <config_name> --n-demo <n_demo>
```

## Parallel Training

```bash
bash scripts/train_isp_parallel.sh <EXPERIMENT_NAME> "<config1,config2,...>" <n_demo> <gpu_ids> all
```

默认 remote conda env 使用 `isp`。如需覆盖，可在 `--` 后追加 `--conda-env <env_name>`。

## Evaluation

```bash
bash scripts/eval_ckpt.sh <checkpoint_path> <inference_config> <total_num>
```

对应编排入口：

```bash
python -m univtac.eval.runner --ckpt <checkpoint_path> --inference-config <inference_config> --total-num <total_num>
```

## Current Layout Assumption

Raw demos:

```text
data/<EXPERIMENT_NAME>/<task_name>/hdf5/*.hdf5
```

ISP cache:

```text
prepared_data/<EXPERIMENT_NAME>/<task_name>/ISP/<config_name_demoN>/cache/*
```

Checkpoints:

```text
checkpoints/<EXPERIMENT_NAME>/<task_name>/ISP/<config_name_demoN>/<run_id>/*
```

更多整体流程见 [RUN.md](/home/zhenya/kenny/visuotact/UniVTAC/RUN.md)。
