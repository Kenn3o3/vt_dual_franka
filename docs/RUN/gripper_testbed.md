# Gripper Testbed

This testbed is for Franka Hand experiments without integrating the ROS gripper path into the main arm data-collection controller.

## Control Model

The workspace UI sends edge-triggered gripper commands:

- `+1`: open to `max_gripper_width` with `open_velocity`.
- `-1`: close with `close_velocity` and `force_threshold`.
- `0`: stop immediately. The UI sends this when the operator presses `B`.

On session start, the UI sends `+1` so the gripper starts fully open. Pressing the selected Quest trigger sends `-1`. Pressing `B` sends `0`. Releasing the trigger sends `+1`.

## Franka ROS Frequency Note

The official `franka_gripper` interface is action-based, not a high-frequency streaming width servo. `MoveAction`, `GraspAction`, and `StopAction` are appropriate for sparse state transitions such as open, close, and stop.

Use the UI velocities to tune physical speed. Do not send continuously changing width targets at 20-60 Hz through the ROS gripper action server. Polling `/api/v1/state` at UI rates is fine, but command issuance should remain edge-triggered.

## Run With Polymetis Gripper

Terminal C1 on the robot controller machine:

```bash
launch_gripper.py gripper=franka_hand gripper.executable_cfg.robot_ip=172.16.0.2
```

Terminal C2 on the robot controller machine:

```bash
cd /home/zhenya/kenny/visuotact/vt_franka
vt-franka-controller gripper-testbed \
  --config robot_controller/config/gripper_testbed_controller.yaml
```

Terminal W1 on the workspace machine:

```bash
cd /home/zhenya/kenny/visuotact/vt_franka
vt-franka-workspace gripper-testbed \
  --config robot_workspace/config/gripper_testbed.yaml
```

Open the UI at `http://<workspace-pc-ip>:8084`.

## Run With Franka ROS Gripper

Do not run `launch_gripper.py` in this mode. Only the ROS `franka_gripper` node should own the Franka Hand.

Terminal C1 on the robot controller machine:

```bash
roscore
```

Terminal C2 on the robot controller machine:

```bash
source /opt/ros/noetic/setup.bash
source /path/to/franka_ros_ws/devel/setup.bash
roslaunch franka_gripper franka_gripper.launch robot_ip:=172.16.0.2
```

If your launch file uses a different action namespace or joint state topic, update `robot_controller/config/franka_ros_gripper_testbed.yaml`.

Terminal C3 on the robot controller machine:

```bash
cd /home/zhenya/kenny/visuotact/vt_franka
source /opt/ros/noetic/setup.bash
source /path/to/franka_ros_ws/devel/setup.bash
vt-franka-controller ros-gripper-testbed \
  --config robot_controller/config/franka_ros_gripper_testbed.yaml
```

Terminal W1 on the workspace machine:

```bash
cd /home/zhenya/kenny/visuotact/vt_franka
vt-franka-workspace gripper-testbed \
  --config robot_workspace/config/gripper_testbed_franka_ros.yaml
```

Open the UI at `http://<workspace-pc-ip>:8084`.

## UI Procedure

1. Confirm `controller_host` and `controller_port` point to the active controller. For ROS gripper mode, the default port is `8094`.
2. Click `Start Live Session`. The gripper should open to max width.
3. Adjust `Force threshold`, `Close velocity`, `Open velocity`, and `Control hand` in the UI.
4. Press the selected Quest trigger to close slowly.
5. Press keyboard `B` when the desired contact point is reached. The UI sends the stop command.
6. Release the Quest trigger. The UI sends open at `open_velocity`.
7. Check the command/state readout and recorded JSONL streams under `robot_workspace/data/gripper_testbed`.
