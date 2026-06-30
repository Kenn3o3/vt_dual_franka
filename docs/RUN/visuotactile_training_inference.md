# Visuotactile Training and Inference

This document is the runbook for real VT_Franka visuotactile policies:

- `dp_manifeel`
- `dp_equidiff_tact`
- `act_univtac`
- `vital_act`
- `vital_dp`
- `vista_so2`
- `vista_so3`

The commands below assume:

```bash
cd /home/zhenya/kenny/visuotact/vt_franka
```

## Storage Layout

Use the normal workspace paths:

```text
robot_workspace/data/collect
robot_workspace/data/preprocess1
robot_workspace/data/prepared
robot_workspace/data/checkpoints
```

On this machine, these `data/*` paths are backed by the SSD:

```text
/mnt/kenny_ssd/vt_franka/data/preprocess1
/mnt/kenny_ssd/vt_franka/data/prepared
/mnt/kenny_ssd/vt_franka/data/checkpoints
```

Keep using the `robot_workspace/data/...` paths in commands. Remote training also preserves workspace-relative paths such as `data/prepared/...` and `data/checkpoints/...`.

The data stages are:

```text
raw collect episodes
  -> aligned_episode.npz
  -> preprocess1 canonical images
  -> prepared model dataset
  -> backend HDF5/export inside each checkpoint run
  -> model checkpoint/runtime manifests
```

`preprocess2` is model-specific and lives inside the prepared dataset. There is no separate long-term `preprocess2` cache.

## Environment

Use `UniVTAC_isp` for training:

```bash
conda activate UniVTAC_isp
pip install pydantic
export PYTHONPATH=$PWD/robot_workspace/src:$PWD/shared/src:$PYTHONPATH
```

`pydantic` is not a model dependency. VT_Franka uses it to load and validate `workspace.yaml`, policy configs, and inference configs. The visuotactile train module imports `vt_franka_workspace.config`, so the training environment needs `pydantic`.

Quick check:

```bash
python -m vt_franka_workspace.policies.visuotactile.train --help
```

## Prepare Data

First make sure the raw episodes are aligned:

```bash
python tools/align_episode.py robot_workspace/data/collect/usb_insertion --hz 10 --overwrite
```

Prepare one model dataset explicitly:

```bash
python -m vt_franka_workspace.policies.visuotactile.prepare \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name usb_insertion \
  --model vista_so2 \
  --raw-run-dir robot_workspace/data/collect/usb_insertion \
  --overwrite
```

Prepared datasets are written to:

```text
robot_workspace/data/prepared/usb_insertion/visuotactile/real_canonical_v1/<model>/
```

Preprocess1 is written once per task/profile:

```text
robot_workspace/data/preprocess1/usb_insertion/real_canonical_v1/episodes/episode_XXXX/
```

Each episode contains canonical image chunks plus `canonical_episode.npz`, which stores the action/proprio arrays needed for remote dataset preparation. The task-level file
`robot_workspace/data/preprocess1/<task>/<profile>/dataset_manifest.json` makes the bundle portable; remote training should sync this bundle instead of raw `collect/`.

Prepare directly from an existing portable `preprocess1` bundle:

```bash
python -m vt_franka_workspace.policies.visuotactile.prepare \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name usb_insertion \
  --model vista_so2 \
  --source preprocess1 \
  --source-root robot_workspace/data/preprocess1/usb_insertion/real_canonical_v1 \
  --overwrite
```

Default image sizes:

| model | wrist | tactile | action |
| --- | ---: | ---: | --- |
| `dp_manifeel` | 224 | 224 | 10D xyz + rot6d + gripper |
| `dp_equidiff_tact` | 224 | 224 | 10D xyz + rot6d + gripper |
| `act_univtac` | 256 | 256 | 8D xyz + quat + gripper |
| `vital_act` | 256 | 256 | 8D xyz + quat + gripper |
| `vital_dp` | 256 | 256 | 10D xyz + rot6d + gripper |
| `vista_so2` | 224 | 224 | 10D xyz + rot6d + gripper |
| `vista_so3` | 224 | 224 | 10D xyz + rot6d + gripper |

The single real GelSight stream is duplicated as left/right tactile when exporting vendor HDF5 datasets.

## Local Training

For a first smoke test, run one epoch:

```bash
python -m vt_franka_workspace.policies.visuotactile.train \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name usb_insertion \
  --model vista_so2 \
  --raw-run-dir robot_workspace/data/collect/usb_insertion \
  --batch-size 64 \
  --epochs 1 \
  --overwrite
```

This command automatically prepares the dataset if it is missing, exports the backend HDF5 view, writes runtime manifests, and launches the vendor trainer.

Full training uses the same command without `--epochs 1`:

```bash
python -m vt_franka_workspace.policies.visuotactile.train \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name usb_insertion \
  --model vista_so2 \
  --raw-run-dir robot_workspace/data/collect/usb_insertion \
  --batch-size 32 \
  --overwrite
```

Change only `--model` for other policies:

```bash
for model in dp_manifeel dp_equidiff_tact act_univtac vital_act vital_dp vista_so2 vista_so3; do
  python -m vt_franka_workspace.policies.visuotactile.train \
    --workspace-config robot_workspace/config/workspace.yaml \
    --task-name usb_insertion \
    --model "$model" \
    --raw-run-dir robot_workspace/data/collect/usb_insertion \
    --batch-size 8 \
    --epochs 1 \
    --overwrite
done
```

Checkpoint runs are written to:

```text
robot_workspace/data/checkpoints/usb_insertion/<model>/
```

Important files:

```text
policy_manifest.json
preprocess1_manifest.json
preprocess2_manifest.json
normalizer_stats.json
train_command.sh
train_config.json
backend_dataset/
```

`act_univtac` uses `UniVTAC_encoder/best.pth`. `vital_act` and `vital_dp` use `VITAL_encoder/best_vision_encoder.pth` and `VITAL_encoder/best_gelsight_encoder.pth` from the standalone VT_Franka vendor checkpoint directory.

## Remote Training

Remote training v2 keeps raw data local. The remote machine receives code and `preprocess1`, then prepares model-specific datasets and trains there.

Default remote repo root:

```text
/mnt/pfs_cuhk/kenny/vt_franka
```

Sync code:

```bash
REMOTE=user@remote-host \
REMOTE_ROOT=/mnt/pfs_cuhk/kenny/vt_franka \
remote_pc/sync_repo_to_remote.sh
```

Sync only portable `preprocess1`:

```bash
REMOTE=user@remote-host \
TASK_NAME=usb_insertion \
PROFILE_NAME=real_canonical_v1 \
REMOTE_ROOT=/mnt/pfs_cuhk/kenny/vt_franka \
remote_pc/sync_preprocess1_to_remote.sh
```

On the remote host:

```bash
cd /mnt/pfs_cuhk/kenny/vt_franka
conda activate UniVTAC_isp
export PYTHONPATH=$PWD/robot_workspace/src:$PWD/shared/src:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python -m vt_franka_workspace.policies.visuotactile.prepare \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name usb_insertion \
  --model dp_manifeel \
  --output-dir robot_workspace/data/prepared/usb_insertion/visuotactile/real_canonical_v1/dp_manifeel \
  --source preprocess1 \
  --source-root robot_workspace/data/preprocess1/usb_insertion/real_canonical_v1 \
  --overwrite && \
CUDA_VISIBLE_DEVICES=0 python -m vt_franka_workspace.policies.visuotactile.train \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name usb_insertion \
  --model dp_manifeel \
  --dataset-dir robot_workspace/data/prepared/usb_insertion/visuotactile/real_canonical_v1/dp_manifeel \
  --checkpoint-dir robot_workspace/data/checkpoints/usb_insertion/dp_manifeel \
  --backend-dataset-root robot_workspace/data/checkpoints/usb_insertion/dp_manifeel/backend_dataset \
  --seed 0 \
  --batch-size 32 \
  --epochs 100 \
  --device cuda:0 \
  --no-prepare \
  --overwrite \
  --extra-arg logging.mode=offline
```

Run each model in its own `tmux` session. Checkpoint outputs are fixed model directories:

```text
robot_workspace/data/checkpoints/<task>/visuotactile/<model>/
```

Download checkpoints back:

```bash
REMOTE=user@remote-host \
TASK_NAME=usb_insertion \
REMOTE_ROOT=/mnt/pfs_cuhk/kenny/vt_franka \
remote_pc/download_checkpoints_from_remote.sh
```

Remote sync rules:

- sync code excludes `robot_workspace/data/collect`, `preprocess1`, `prepared`, and `checkpoints`
- sync data uploads only `robot_workspace/data/preprocess1/<task>/<profile>`
- remote generated `prepared/` is disposable
- checkpoints are the only remote training artifact downloaded back

## Runtime Inference

The VT_Franka online visuotactile policy loads the backend best checkpoint and runtime manifests from the checkpoint directory:

```text
checkpoints/best.ckpt
policy_manifest.json
preprocess1_manifest.json
preprocess2_manifest.json
normalizer_stats.json
```

Example policy config:

```yaml
type: visuotactile
checkpoint_path: /home/zhenya/kenny/visuotact/vt_franka/robot_workspace/data/checkpoints/usb_insertion/vista_so2
config:
  model: vista_so2
  task_name: usb_insertion
  device: cuda
  gripper_close_threshold: 0.5
```

Example inference config requirements:

```yaml
obs_horizon: 2
exe_horizon: 8
control_hz: 10.0
task_name: usb_insertion

modality:
  proprioception: true
  rgb_cameras: [wrist]
  gelsight_frame: true
  controller_state_max_age_sec: 2.0
  rgb_camera_max_age_sec: 2.0
  gelsight_max_age_sec: 2.0

gelsight:
  enabled: true

rgb_cameras:
  wrist:
    enabled: true
    backend: orbbec
    stream_name: rgb_wrist
    camera_name: wrist_orbbec_rgb
    serial_number: CP0E753000KA
    color_width: 640
    color_height: 0
    color_format: RGB
    color_fps: 30
    frame_timeout_ms: 200
    save_frames: true
    record_hz: 15.0
```

Run policy:

```bash
conda activate vt-franka-workspace
cd /home/zhenya/kenny/visuotact/vt_franka

vt-franka-workspace run-policy \
  --workspace-config robot_workspace/config/workspace.yaml \
  --policy-config robot_workspace/config/policies/<visuotactile_policy>.yaml \
  --inference-config robot_workspace/config/inference/<visuotactile_inference>.yaml
```

Runtime preprocessing is aligned with training:

```text
live raw frame -> preprocess1 from checkpoint manifest -> model preprocess2 from checkpoint manifest -> policy input
```

Do not change live crop/resize independently of the checkpoint manifests.

## Recommended Bringup Order

1. Run `vista_so2` with `--epochs 1`.
2. Check that `backend_dataset_manifest.json`, `policy_manifest.json`, and vendor checkpoint files exist.
3. Run `dp_manifeel` with `--epochs 1`.
4. Run `act_univtac` with `--epochs 1` to verify UniVTAC encoder loading.
5. Run `vital_dp` and `vital_act` with `--epochs 1` to verify VITAL encoder loading.
6. Start full training only after all smoke tests pass.
