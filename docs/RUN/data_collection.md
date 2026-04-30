# Data Collection

Connect Meta Quest on USB

```bash
adb devices
adb reverse tcp:8082 tcp:8082
adb shell settings put global stay_on_while_plugged_in 3
adb shell am broadcast -a com.oculus.vrpowermanager.prox_close --ei timeout 0
adb shell setprop debug.oculus.guardian_pause 1
```

In the Quest app, set the workstation IP to:

```text
127.0.0.1
```

## Run Collection

### Run on the robot controller machine:

#### Terminal C1: Polymetis robot server

```bash
launch_robot.py robot_client=franka_hardware robot_client.executable_cfg.robot_ip=172.16.0.2
```

#### Terminal C2: Polymetis gripper server

```bash
launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=172.16.0.2
```

#### Terminal C3: Controller API

```bash
vt-franka-controller run --config /home/zhenya/kenny/visuotact/vt_franka/robot_controller/config/controller.yaml
```

### Run on the workspace machine:

```bash
conda activate vt-franka-workspace
cd /home/zhenya/kenny/visuotact/vt_franka

vt-franka-workspace collect \
  --workspace-config robot_workspace/config/workspace.yaml \
  --task put_cup_on_plate
```

Operator controls:

- `H`: move to the task initial EEF pose.
- `R`: start recording an episode.
- `E`: stop and save the current episode.
- `D`: discard the latest saved episode.
- `Q`: quit.

Output:

```text
robot_workspace/data/collect/put_cup_on_plate/episodes/episode_XXXX/
```

Each episode records raw streams such as `controller_state.jsonl`, `teleop_commands.jsonl`, configured RGB camera streams, and optional tactile streams. There is no postprocessing step in the clean pipeline.

Optional commands:

output aligned dataset:

```bash
python tools/align_episode.py robot_workspace/data/collect/put_cup_on_plate --hz 10 --overwrite
```

visualize aligned dataset

```bash
python tools/visualize_aligned_episode.py robot_workspace/data/collect/put_cup_on_plate --overwrite

# or

python tools/visualize_aligned_episode.py robot_workspace/data/collect/put_cup_on_plate/episodes/episode_0000 --overwrite
```

