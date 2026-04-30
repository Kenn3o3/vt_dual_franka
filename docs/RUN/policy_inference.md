# Policy Inference

Replay policy:

```bash
conda activate vt-franka-workspace
cd /home/zhenya/kenny/visuotact/vt_franka

vt-franka-workspace run-policy \
  --workspace-config robot_workspace/config/workspace.yaml \
  --policy-config robot_workspace/config/policies/replay.yaml \
  --inference-config robot_workspace/config/inference/replay.yaml
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

Policy run output is written under:

```text
robot_workspace/data/eval/<run_name>/episodes/episode_XXXX/
```
