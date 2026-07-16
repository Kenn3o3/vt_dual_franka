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
launch_robot.py \
  robot_client=franka_hardware \
  robot_client.executable_cfg.robot_ip=172.16.1.2 \
  port=50061 \
  robot_client.executable_cfg.server_address=localhost:50061

# Terminal P4: right Polymetis gripper server
launch_gripper.py \
  gripper=franka_hand \
  gripper.executable_cfg.robot_ip=172.16.1.2 \
  port=50062

# Required invariant for robot_controller/config/controller_left.yaml and
# robot_controller/config/controller_right.yaml:
#   left  robot server   -> 127.0.0.1:50051
#   left  gripper server -> 127.0.0.1:50052
#   right robot server   -> 127.0.0.1:50061
#   right gripper server -> 127.0.0.1:50062
EOF
