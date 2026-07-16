#!/usr/bin/env python3

from __future__ import annotations

import time

import numpy as np
import torch
from polymetis import RobotInterface
from scipy.spatial.transform import Rotation


# Edit these values directly for your test.
ROBOT_IP = "127.0.0.1"
ROBOT_PORT = 50051

BASE_POSITION_M = [0, 0.3, 0.4]
BASE_RPY_DEG = [-180, 0, 45]
RPY_SEQUENCE = "xyz"
POSE_MOVE_TIME_S = 4.0
POST_MOVE_SETTLE_S = 1.0

# Action commands are interpreted as rotations around the tool/TCP axes.
# With BASE_RPY_DEG=[-180, 0, 45], editing absolute Euler angles can look
# coupled in photos even when the final Euler display prints the requested
# values. These incremental commands make the intended axis explicit.
ACTION_ROTATION_FRAME = "tool"  # "tool" (local/TCP axes) or "world" (base axes)
ACTION_COMMANDS_DEG = [
    ("roll", 45.0),
    ("pitch", 45.0),
    ("yaw", 45.0),
]
RETURN_TO_BASE_BEFORE_EACH_ACTION = True
RUN_ACTION_TESTS = True

# Target joint positions in radians.
TARGET_JOINTS_RAD = [-1.5857, -0.7485, 0.0049, -2.8095, 0.0293, 2.0184, 0.9069]
# TARGET_JOINTS_RAD_2 = [1.3922, -0.7209, 0.1799, -2.8098, 0.1171, 2.0371, 0.7]
JOINT_MOVE_TIME_S = 5.0
RUN_JOINT_MOVE = False

# Keep this on so the robot does not move until you explicitly confirm.
ASK_FOR_CONFIRMATION = True

AXIS_VECTORS = {
    "roll": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
    "pitch": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
    "yaw": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
}


def tensor_to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def format_array(values) -> str:
    return np.array2string(np.asarray(values, dtype=np.float64), precision=4, suppress_small=True)


def quaternion_xyzw_to_rpy_deg(quaternion_xyzw: np.ndarray) -> np.ndarray:
    return Rotation.from_quat(quaternion_xyzw).as_euler(RPY_SEQUENCE, degrees=True)


def rpy_deg_to_quaternion_xyzw(rpy_deg: np.ndarray) -> np.ndarray:
    return Rotation.from_euler(RPY_SEQUENCE, rpy_deg, degrees=True).as_quat().astype(np.float32)


def action_target_rotation(base_rotation: Rotation, axis_name: str, degrees: float) -> Rotation:
    if axis_name not in AXIS_VECTORS:
        raise ValueError(f"Unknown axis {axis_name!r}; expected one of {sorted(AXIS_VECTORS)}")
    delta_rotation = Rotation.from_rotvec(np.deg2rad(float(degrees)) * AXIS_VECTORS[axis_name])
    if ACTION_ROTATION_FRAME == "tool":
        return base_rotation * delta_rotation
    if ACTION_ROTATION_FRAME == "world":
        return delta_rotation * base_rotation
    raise ValueError("ACTION_ROTATION_FRAME must be 'tool' or 'world'")


def relative_rotvec_deg(reference_quaternion_xyzw: np.ndarray, current_quaternion_xyzw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference_rotation = Rotation.from_quat(reference_quaternion_xyzw)
    current_rotation = Rotation.from_quat(current_quaternion_xyzw)
    # For a target built as base * delta, the tool-frame rotvec should show the
    # commanded single axis most directly.
    tool_frame_delta = reference_rotation.inv() * current_rotation
    world_frame_delta = current_rotation * reference_rotation.inv()
    return np.rad2deg(tool_frame_delta.as_rotvec()), np.rad2deg(world_frame_delta.as_rotvec())


def rotation_error_deg(target_quaternion_xyzw: np.ndarray, current_quaternion_xyzw: np.ndarray) -> float:
    target_rotation = Rotation.from_quat(target_quaternion_xyzw)
    current_rotation = Rotation.from_quat(current_quaternion_xyzw)
    return float(np.rad2deg((target_rotation.inv() * current_rotation).magnitude()))


def get_pose_and_joints(robot: RobotInterface) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_positions = tensor_to_numpy(robot.get_joint_positions())
    ee_position, ee_quaternion_xyzw = robot.get_ee_pose()
    ee_position = tensor_to_numpy(ee_position)
    ee_quaternion_xyzw = tensor_to_numpy(ee_quaternion_xyzw)
    return ee_position, ee_quaternion_xyzw, joint_positions


def print_pose_and_joints(
    robot: RobotInterface,
    label: str,
    *,
    reference_quaternion_xyzw: np.ndarray | None = None,
    target_position_m: np.ndarray | None = None,
    target_quaternion_xyzw: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ee_position, ee_quaternion_xyzw, joint_positions = get_pose_and_joints(robot)
    ee_rpy_deg = quaternion_xyzw_to_rpy_deg(ee_quaternion_xyzw)

    print(f"{label} position xyz (m): {format_array(ee_position)}")
    print(f"{label} orientation quaternion xyzw: {format_array(ee_quaternion_xyzw)}")
    print(f"{label} orientation {RPY_SEQUENCE} euler (deg): {format_array(ee_rpy_deg)}")
    if reference_quaternion_xyzw is not None:
        rel_tool_deg, rel_world_deg = relative_rotvec_deg(reference_quaternion_xyzw, ee_quaternion_xyzw)
        print(f"{label} relative rotation from base, tool-frame rotvec deg xyz: {format_array(rel_tool_deg)}")
        print(f"{label} relative rotation from base, world-frame rotvec deg xyz: {format_array(rel_world_deg)}")
    if target_position_m is not None:
        position_error = float(np.linalg.norm(ee_position - np.asarray(target_position_m, dtype=np.float64)))
        print(f"{label} position error to target (m): {position_error:.6f}")
    if target_quaternion_xyzw is not None:
        print(f"{label} rotation error to target (deg): {rotation_error_deg(target_quaternion_xyzw, ee_quaternion_xyzw):.6f}")
    print(f"{label} joints (rad): {format_array(joint_positions)}")
    print()
    return ee_position, ee_quaternion_xyzw, joint_positions


def confirm_or_abort(message: str) -> None:
    if not ASK_FOR_CONFIRMATION:
        return
    response = input(f"{message} Type 'yes' to continue: ").strip().lower()
    if response != "yes":
        raise SystemExit("Aborted.")


def wait_after_move() -> None:
    if POST_MOVE_SETTLE_S > 0.0:
        time.sleep(POST_MOVE_SETTLE_S)


def print_target(label: str, position_m: np.ndarray, quaternion_xyzw: np.ndarray, reference_quaternion_xyzw: np.ndarray | None = None) -> None:
    print(f"{label} command:")
    print(f"  position xyz (m): {format_array(position_m)}")
    print(f"  orientation quaternion xyzw: {format_array(quaternion_xyzw)}")
    print(f"  orientation {RPY_SEQUENCE} euler (deg): {format_array(quaternion_xyzw_to_rpy_deg(quaternion_xyzw))}")
    if reference_quaternion_xyzw is not None:
        rel_tool_deg, rel_world_deg = relative_rotvec_deg(reference_quaternion_xyzw, quaternion_xyzw)
        print(f"  relative rotation from base, tool-frame rotvec deg xyz: {format_array(rel_tool_deg)}")
        print(f"  relative rotation from base, world-frame rotvec deg xyz: {format_array(rel_world_deg)}")
    print()


def move_to_pose(robot: RobotInterface, label: str, position_m: np.ndarray, quaternion_xyzw: np.ndarray) -> None:
    print_target(label, position_m, quaternion_xyzw)
    confirm_or_abort(f"Move robot to {label}?")
    robot.move_to_ee_pose(
        position=torch.tensor(position_m),
        orientation=torch.tensor(quaternion_xyzw),
        time_to_go=POSE_MOVE_TIME_S,
    )
    wait_after_move()


def main() -> None:
    base_position = np.asarray(BASE_POSITION_M, dtype=np.float32)
    base_rpy_deg = np.asarray(BASE_RPY_DEG, dtype=np.float64)
    base_rotation = Rotation.from_euler(RPY_SEQUENCE, base_rpy_deg, degrees=True)
    base_quaternion_xyzw = base_rotation.as_quat().astype(np.float32)
    target_joints = np.asarray(TARGET_JOINTS_RAD, dtype=np.float32)

    robot = RobotInterface(ip_address=ROBOT_IP, port=ROBOT_PORT)
    try:
        print_pose_and_joints(robot, "Current")

        move_to_pose(robot, "base pose", base_position, base_quaternion_xyzw)
        print()
        print_pose_and_joints(
            robot,
            "After base pose move",
            reference_quaternion_xyzw=base_quaternion_xyzw,
            target_position_m=base_position,
            target_quaternion_xyzw=base_quaternion_xyzw,
        )

        if RUN_ACTION_TESTS:
            print(f"Action rotation frame: {ACTION_ROTATION_FRAME}")
            print(f"Return to base before each action: {RETURN_TO_BASE_BEFORE_EACH_ACTION}")
            print()
            for index, (axis_name, degrees) in enumerate(ACTION_COMMANDS_DEG):
                if RETURN_TO_BASE_BEFORE_EACH_ACTION and index > 0:
                    move_to_pose(robot, "base pose reset", base_position, base_quaternion_xyzw)
                    print()
                    print_pose_and_joints(
                        robot,
                        "After base pose reset",
                        reference_quaternion_xyzw=base_quaternion_xyzw,
                        target_position_m=base_position,
                        target_quaternion_xyzw=base_quaternion_xyzw,
                    )

                target_rotation = action_target_rotation(base_rotation, axis_name, degrees)
                target_quaternion_xyzw = target_rotation.as_quat().astype(np.float32)
                label = f"action +{degrees:g} deg {axis_name}"
                print_target(label, base_position, target_quaternion_xyzw, reference_quaternion_xyzw=base_quaternion_xyzw)
                confirm_or_abort(f"Move robot to {label}?")
                robot.move_to_ee_pose(
                    position=torch.tensor(base_position),
                    orientation=torch.tensor(target_quaternion_xyzw),
                    time_to_go=POSE_MOVE_TIME_S,
                )
                wait_after_move()
                print()
                print_pose_and_joints(
                    robot,
                    f"After {label}",
                    reference_quaternion_xyzw=base_quaternion_xyzw,
                    target_position_m=base_position,
                    target_quaternion_xyzw=target_quaternion_xyzw,
                )

        if RUN_JOINT_MOVE:
            print("Target joint command:")
            print(f"  joints (rad): {format_array(target_joints)}")
            print()
            confirm_or_abort("Move robot to the target joints?")
            robot.move_to_joint_positions(
                positions=torch.tensor(target_joints),
                time_to_go=JOINT_MOVE_TIME_S,
            )
            wait_after_move()
            print()
            print_pose_and_joints(robot, "After joint move")
    finally:
        try:
            robot.terminate_current_policy()
        except Exception:
            pass


if __name__ == "__main__":
    main()
