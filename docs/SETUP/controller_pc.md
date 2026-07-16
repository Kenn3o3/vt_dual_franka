# Controller PC Setup

The controller PC runs the Franka-side controller API. It should be directly connected to the Franka controller over Ethernet.

Start the robot and gripper services with the local Polymetis setup, then run:

```bash
conda activate vt-dual-franka-controller
cd /home/medair/vt_franka/robot_controller

vt-dual-franka-controller run \
  --config /home/medair/vt_franka/robot_controller/config/controller.yaml
```

The workspace PC expects the controller API at the address configured in:

```text
robot_workspace/config/workspace.yaml
```

The current default is:

```yaml
controller:
  host: 10.0.0.1
  port: 8092
```

Before collecting or running policies, verify from the workspace PC:

```bash
curl http://10.0.0.1:8092/api/v1/health
```
