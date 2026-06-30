# Policy Inference

Replay policy:

```bash
conda activate vt-franka-workspace
conda activate /mnt/kenny_ssd/conda_envs/isp_real
export PYTHONPATH=$PWD/robot_workspace/src:$PWD/shared/src:$PYTHONPATH
cd /home/zhenya/kenny/visuotact/vt_franka

python -m vt_franka_workspace.cli run-policy \
  --workspace-config robot_workspace/config/workspace.yaml \
  --policy-config robot_workspace/config/policies/visuotactile_usb_insertion_vista_so3_epoch179.yaml \
  --inference-config robot_workspace/config/inference/usb_insertion_visuotactile.yaml
```

MPD-family policy:

```bash
conda activate mpd
cd /home/zhenya/kenny/visuotact/vt_franka

vt-franka-workspace run-policy \
  --workspace-config robot_workspace/config/workspace.yaml \
  --policy-config robot_workspace/config/policies/mpd_dp_state.yaml \
  --inference-config robot_workspace/config/inference/mpd_state.yaml
```

The policy runner uses one inference loop for every policy:

- Build initial observation history by padding the first live observation.
- Wait during policy inference; no robot action is sent while the policy is computing.
- Execute the first `exe_horizon` actions returned by the policy.
- Feed executed actions back into the policy lifecycle.
- Append the next live observation and repeat.

Inference configs support two optional initial-state controls:

- `gripper_forever_closed: true` keeps the gripper closed for replay and learned-policy rollout. After every `H`, press `C` to close it before `R`; policy/replay open commands are suppressed and executed-action logs are recorded as closed.
- `rand_init_pose: [x, y, z]` adds a uniform random xyz offset in `[-range, +range]` meters to the configured initial EEF pose for each `H`.

For replay, the current replay policy still emits absolute recorded TCP targets. Randomizing the initial pose changes where the robot starts, but the replayed trajectory itself is not automatically translated.

Policy run output is written under:

```text
robot_workspace/data/eval/<policy_family>/<policy>/<date_time>/episodes/episode_XXXX/
```

By default eval recording is enabled with the third-person camera at 10 Hz. Each saved episode writes:

```text
rollout.mp4
```

Use `inference.eval.enabled` and `inference.eval.cameras` to turn eval recording off or select `third`, `wrist`, or `wrist+third`.

GelSight marker observations are no longer supported. Use `modality.gelsight_frame: true` for tactile images; when `gelsight.buffered_recording: true`, policy observation records keep GelSight frame metadata and avoid synchronous GelSight image writes in the policy loop.
