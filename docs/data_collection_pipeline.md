# Data Collection Pipeline 

### 3.1 Controller machine

The controller machine still runs the low-level robot stack as a long-lived service:

- Polymetis robot server
- Polymetis gripper server
- `vt-franka-controller run`

Run the following commands in three terminals:

Terminal C1:

```bash
conda activate polymetis-local
cd /home/medair/vt_franka/fairo/polymetis
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.0.2
```

Terminal C2:

```bash
conda activate polymetis-local
cd /home/medair/vt_franka/fairo/polymetis
launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=172.16.0.2
```

Terminal C3:

```bash
conda activate polymetis-local
cd /home/medair/vt_franka/robot_controller
vt-franka-controller run --config /home/medair/vt_franka/robot_controller/config/controller.yaml
```

### 3.2 Workspace machine

Normal collection should use one command:

```bash
conda activate vt-franka-workspace
cd /home/zhenya/kenny/visuotact/vt_franka/robot_workspace
vt-franka-workspace collect \
  --config /home/zhenya/kenny/visuotact/vt_franka/robot_workspace/config/workspace.yaml \
  --run put_cup_on_plate
```

Visualize:

```bash
conda activate vt-franka-workspace
cd /home/zhenya/kenny/visuotact/vt_franka/robot_workspace
vt-franka-workspace visualize \
  --config /home/zhenya/kenny/visuotact/vt_franka/robot_workspace/config/workspace.yaml \
  --episode-dir /home/zhenya/kenny/visuotact/vt_franka/robot_workspace/data/runs/put_cup_on_plate_20260422_171246/episodes/episode_0012
```

Replay:

```bash
conda activate vt-franka-workspace
cd /home/zhenya/kenny/visuotact/vt_franka/robot_workspace
vt-franka-workspace rollout-once \
  --config /home/zhenya/kenny/visuotact/vt_franka/robot_workspace/config/workspace.yaml \
  --policy vt_franka_workspace.rollout.replay_policy:build_replay_policy \
  --episode-dir /home/zhenya/kenny/visuotact/vt_franka/robot_workspace/data/runs/put_bowl_on_plate_20260421_212945/episodes/episode_0000 \
  --go-ready
```

reset joints (on controller):
```bash
conda activate polymetis-local
cd /home/medair/vt_franka/robot_controller
python scripts/joint_reset.py --config /home/medair/vt_franka/robot_controller/config/controller.yaml
```

### 3.3 Upload A Folder To ModelScope

Use the repo helper script:

```bash
cd /home/zhenya/kenny/visuotact/vt_franka
```

Set the upload parameters:

```bash
export MODELSCOPE_TOKEN='ms-f6ff6aaf-8cd2-4aed-b57d-c0be1e7799aa' # It is ok to expose this token here
export MODELSCOPE_REPO_ID='kenn3o3/put_cup_on_plate'
export MODELSCOPE_VISIBILITY='private'
export MODELSCOPE_BATCH_SIZE=64
export MODELSCOPE_MAX_WORKERS=4
export MODELSCOPE_RESUME=1
```

Dry run first:

```bash
export MODELSCOPE_DRY_RUN=1

bash kenny/scripts/upload_data_to_modelscope.sh \
  robot_workspace/data/mpd/put_cup_on_plate \
  /
```

Real upload:

```bash
unset MODELSCOPE_DRY_RUN

bash kenny/scripts/upload_data_to_modelscope.sh \
  robot_workspace/data/mpd/put_cup_on_plate \
  /
```

Notes:

- First argument is the local folder to upload.
- Second argument is the destination path inside the ModelScope repo.
- Use `/` to upload the folder contents to the repo root.
- Use `put_cup_on_plate` instead of `/` if you want the remote repo to contain a nested `put_cup_on_plate/` directory.
- `MODELSCOPE_BATCH_SIZE` controls how many files are uploaded per commit batch.
- `MODELSCOPE_MAX_WORKERS` controls concurrent uploads inside one batch. `2` to `4` is a reasonable range.
- `MODELSCOPE_RESUME=1` makes the script list remote files first and skip files that are already committed.
- If ModelScope returns a transient `502` or timeout, rerun the same command. The script is designed for resume/retry.
- The script path is `kenny/scripts/upload_data_to_modelscope.sh`.

This enters an operator mode with:

- health checks
- worker startup
- browser operator UI on `http://127.0.0.1:8083/operator`
- hotkeys
- episode lifecycle management
- automatic postprocess and QC

The browser UI is the primary operator surface:

- structured readiness and worker state
- recent logs in a dedicated panel instead of terminal spam
- buttons for reset/start/stop/discard/quit
- a frozen Orbbec pre-episode view when the next episode is allowed to start

Terminal hotkeys still work as a fallback.

Recommended hotkeys:

- `R`: start next episode
- `E`: end and save current episode
- `D`: discard current episode
- `H`: move robot to configured `ready` pose
- `Q`: quit operator mode

Optional:

- `N`: annotate the current episode with a short note
- `P`: pause teleop command forwarding without shutting workers down

Postprocessing semantics for `aligned_episode.npz`:

- `timestamps` are observation times on the aligned grid.
- Proprioception and camera fields are selected causally from the latest sample at or before each aligned timestamp.
- Teleop action fields are selected from the first teleop command strictly after each aligned timestamp.
- Steps without a valid near-future teleop action are dropped from the aligned training episode instead of being paired with stale commands.
- The aligned file also stores per-step source timestamps so the observation/action pairing can be audited directly.
