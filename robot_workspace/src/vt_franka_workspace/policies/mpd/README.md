# MPD Policies

This folder contains the VT Franka adapter for the MPD-family baselines from `UNTOUCHED_REFERENCE/mpd/baseline_train.sh`.

Supported algorithms:

- `dp`
- `fm`
- `sfp`
- `mpd`
- `motif`

Unsupported by design:

- `prodmp_fm`
- `motif_fm`

## Prepare Dataset

```bash
conda activate mpd
cd /home/zhenya/kenny/visuotact/vt_franka

python -m vt_franka_workspace.policies.mpd.prepare \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name put_cup_on_plate \
  --overwrite
```

Output:

```text
robot_workspace/data/prepared/mpd/put_cup_on_plate/vt_franka_mpd_v1/
  train/demo_XXX/{agent_pos,agent_vel,action,action_vel,timestamps}.npz
  val/demo_XXX/{agent_pos,agent_vel,action,action_vel,timestamps}.npz
  dataset_manifest.json
  scaler_values.npz
```

The prepared 10D vector is:

```text
xyz + rot6d + gripper_closedness
```

Actions are selected causally from the first teleop command strictly after each observation timestamp.

## Smooth Gripper Label Dataset

For MOTIF-style models, a binary gripper action can create a sharp velocity spike in the action label. To train with a continuous gripper transition, create a separate prepared dataset variant:

```bash
python -m vt_franka_workspace.policies.mpd.smooth_gripper_dataset \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name put_cup_on_plate \
  --source-dataset-name vt_franka_mpd_v1 \
  --output-dataset-name vt_franka_mpd_v1_gripper_open_smooth_t050 \
  --switch-threshold 0.5 \
  --pre-switch-ramp-steps 4 \
  --post-switch-ramp-steps 4 \
  --overwrite
```

This keeps the same 10D vector layout, but changes the gripper scalar convention:

```text
xyz + rot6d + gripper_open_fraction
```

where:

```text
1.0 = fully open
0.0 = fully closed
0.5 = switch point for open/close control
```

The generated dataset is:

```text
robot_workspace/data/prepared/mpd/put_cup_on_plate/vt_franka_mpd_v1_gripper_open_smooth_t050/
```

The first train demo includes a quick sanity-check plot:

```text
train/demo_000/gripper_open_fraction_transition_ramp.png
```

When a checkpoint is trained from this dataset, the train script copies the dataset manifest into the checkpoint directory. Real-robot inference reads that manifest and automatically switches the MPD adapter to `gripper_open_fraction`, so the existing policy runner can be used without changing its interface.

## Train

DP:

```bash
python -m vt_franka_workspace.policies.mpd.train \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name put_cup_on_plate \
  --algorithm dp \
  --device cuda \
  --epochs 500

python -m vt_franka_workspace.policies.mpd.train \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name put_cup_on_plate \
  --algorithm fm \
  --device cuda \
  --epochs 500

python -m vt_franka_workspace.policies.mpd.train \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name put_cup_on_plate \
  --algorithm sfp \
  --device cuda \
  --epochs 500

python -m vt_franka_workspace.policies.mpd.train \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name put_cup_on_plate \
  --algorithm mpd \
  --device cuda \
  --epochs 500

python -m vt_franka_workspace.policies.mpd.train \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task-name put_cup_on_plate \
  --algorithm motif \
  --prepared-dataset-dir robot_workspace/data/prepared/mpd/put_cup_on_plate/vt_franka_mpd_v1_gripper_open_smooth_t050 \
  --checkpoint-dir robot_workspace/data/checkpoints/put_cup_on_plate/mpd/motif/motif_state_gripper_open_smooth_t050 \
  --device cuda \
  --epochs 500 \
  --swanlab-group motif_state_gripper_open_smooth_t050

```

Other algorithms use the same command with `--algorithm dp`, `--algorithm fm`, `--algorithm sfp`, `--algorithm mpd`, or `--algorithm motif`.

Checkpoints are written to:

```text
robot_workspace/data/checkpoints/<task_name>/mpd/<algorithm>/<policy_name>/
```

Example:

```text
robot_workspace/data/checkpoints/put_cup_on_plate/mpd/dp/dp_state/best_model.pth
```

The train launcher disables simulator evaluation by default and uses the upstream training script only for train/validation loss.

## Inference

```bash
conda activate mpd
cd /home/zhenya/kenny/visuotact/vt_franka

vt-franka-workspace run-policy \
  --workspace-config robot_workspace/config/workspace.yaml \
  --policy-config robot_workspace/config/policies/mpd_motif_state.yaml \
  --inference-config robot_workspace/config/inference/mpd_state.yaml
```

The real-robot adapter mirrors the simulator observation contract:

- `dp`, `fm`, `sfp` consume `agent_pos`.
- `mpd`, `motif` consume `agent_pos` plus action history for `action/action_vel` initial conditions.

At episode start, action history is initialized explicitly from the padded initial observation, matching the simulator reset behavior where initial `action` equals current `agent_pos`. After each executed chunk, `PolicyRunner` feeds the actually executed actions back into the policy before the next inference call. There is no hidden postprocessing or temporal aggregation.
