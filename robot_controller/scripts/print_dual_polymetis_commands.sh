#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
# Controller PC common setup for each terminal:
source /home/zhenya/miniforge3/etc/profile.d/conda.sh
conda activate polymetis-local
cd /home/zhenya/kenny/visuotact/vt_dual_franka/robot_controller

# Terminal P1: left Polymetis robot server
launch_robot.py \
  robot_client=franka_hardware \
  robot_client.executable_cfg.robot_ip=172.16.0.2

# Terminal P2: left Polymetis gripper server
launch_gripper.py \
  gripper=franka_hand \
  gripper.executable_cfg.robot_ip=172.16.0.2

# Terminal P3: right Polymetis robot server
# Replace <POLYMETIS_ROBOT_SERVER_PORT_OVERRIDE_FOR_50061> with the
# installed Polymetis Hydra override that binds the local robot gRPC server
# to 127.0.0.1:50061.
launch_robot.py \
  robot_client=franka_hardware \
  robot_client.executable_cfg.robot_ip=172.16.1.2 \
  <POLYMETIS_ROBOT_SERVER_PORT_OVERRIDE_FOR_50061>

# Terminal P4: right Polymetis gripper server
# Replace <POLYMETIS_GRIPPER_SERVER_PORT_OVERRIDE_FOR_50062> with the
# installed Polymetis Hydra override that binds the local gripper gRPC server
# to 127.0.0.1:50062.
launch_gripper.py \
  gripper=franka_hand \
  gripper.executable_cfg.robot_ip=172.16.1.2 \
  <POLYMETIS_GRIPPER_SERVER_PORT_OVERRIDE_FOR_50062>

# Required invariant for robot_controller/config/controller_left.yaml and
# robot_controller/config/controller_right.yaml:
#   left  robot server   -> 127.0.0.1:50051
#   left  gripper server -> 127.0.0.1:50052
#   right robot server   -> 127.0.0.1:50061
#   right gripper server -> 127.0.0.1:50062
EOF
