from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from vt_dual_franka_shared.models import ControllerState

from ..settings import RosGripperActionSettings
from .base import FrankaBackend

LOGGER = logging.getLogger(__name__)


@dataclass
class RosGripperState:
    width: float
    force: float = 0.0
    wall_time: float = 0.0


class RosGripperDriver(Protocol):
    def move(self, *, width: float, speed: float) -> None:
        ...

    def grasp(self, *, width: float, speed: float, force: float, epsilon_inner: float, epsilon_outer: float) -> None:
        ...

    def stop(self) -> None:
        ...

    def get_state(self) -> RosGripperState:
        ...

    def home(self) -> None:
        ...

    def shutdown(self) -> None:
        ...


class FrankaRosGripperOnlyBackend(FrankaBackend):
    name = "franka-ros-gripper-only"

    def __init__(self, settings: RosGripperActionSettings, driver: RosGripperDriver | None = None) -> None:
        self.settings = settings
        self._driver = driver if driver is not None else FrankaRosGripperActionDriver(settings)
        self._last_width = float(settings.max_gripper_width)
        self._last_force = 0.0
        if settings.home_on_start:
            self._driver.home()

    def get_tcp_pose(self) -> np.ndarray:
        return np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def get_controller_state(self, control_frequency_hz: float) -> ControllerState:
        state = self._driver.get_state()
        self._last_width = float(state.width)
        self._last_force = float(state.force)
        return ControllerState(
            tcp_pose=self.get_tcp_pose().tolist(),
            tcp_velocity=[0.0] * 6,
            tcp_wrench=[0.0] * 6,
            joint_positions=[0.0] * 7,
            joint_velocities=[0.0] * 7,
            gripper_width=self._last_width,
            gripper_force=self._last_force,
            control_frequency_hz=control_frequency_hz,
            backend=self.name,
        )

    def start_cartesian_impedance(self, stiffness: Sequence[float], damping: Sequence[float]) -> None:
        raise RuntimeError("Franka ROS gripper-only backend does not support arm impedance control")

    def update_desired_tcp(self, target_pose6d: np.ndarray) -> None:
        raise RuntimeError("Franka ROS gripper-only backend does not support arm motion")

    def move_gripper(self, width: float, velocity: float, force_limit: float) -> None:
        width = min(max(float(width), 0.0), float(self.settings.max_gripper_width))
        velocity = float(velocity)
        force_limit = float(force_limit)
        if width <= float(self.settings.close_width_threshold):
            self._driver.grasp(
                width=width,
                speed=velocity,
                force=force_limit,
                epsilon_inner=float(self.settings.grasp_epsilon_inner),
                epsilon_outer=float(self.settings.grasp_epsilon_outer),
            )
        else:
            self._driver.move(width=width, speed=velocity)
        self._last_width = width
        self._last_force = force_limit

    def grasp(self, velocity: float, force_limit: float) -> None:
        self._driver.grasp(
            width=0.0,
            speed=float(velocity),
            force=float(force_limit),
            epsilon_inner=float(self.settings.grasp_epsilon_inner),
            epsilon_outer=float(self.settings.grasp_epsilon_outer),
        )
        self._last_width = 0.0
        self._last_force = float(force_limit)

    def stop_gripper(self) -> None:
        self._driver.stop()

    def go_home(self, ee_pose: Sequence[float], duration_sec: float) -> None:
        raise RuntimeError("Franka ROS gripper-only backend does not support arm motion")

    def move_to_joints(self, joint_positions: Sequence[float], duration_sec: float | None = None) -> None:
        raise RuntimeError("Franka ROS gripper-only backend does not support arm motion")

    def shutdown(self) -> None:
        self._driver.shutdown()


class FrankaRosGripperActionDriver:
    """ROS1 actionlib driver for the official franka_gripper action server."""

    def __init__(self, settings: RosGripperActionSettings) -> None:
        self.settings = settings
        try:
            import actionlib
            import rospy
            from franka_gripper.msg import GraspAction, GraspGoal, HomingAction, HomingGoal, MoveAction, MoveGoal, StopAction, StopGoal
            from sensor_msgs.msg import JointState
        except ImportError as exc:  # pragma: no cover - depends on ROS installation
            raise RuntimeError(
                "Franka ROS gripper backend requires a sourced ROS1 environment with "
                "rospy, actionlib, franka_gripper, and sensor_msgs available."
            ) from exc

        self._actionlib = actionlib
        self._rospy = rospy
        self._GraspGoal = GraspGoal
        self._HomingGoal = HomingGoal
        self._MoveGoal = MoveGoal
        self._StopGoal = StopGoal
        self._goal_lock = threading.RLock()
        self._state_lock = threading.Lock()
        self._stop_requested = threading.Event()
        self._latest_state = RosGripperState(width=float(settings.max_gripper_width), wall_time=0.0)

        if not rospy.core.is_initialized():
            rospy.init_node("vt_franka_ros_gripper_testbed", anonymous=True, disable_signals=True)

        self._move_client = actionlib.SimpleActionClient(self._action_name("move"), MoveAction)
        self._grasp_client = actionlib.SimpleActionClient(self._action_name("grasp"), GraspAction)
        self._stop_client = actionlib.SimpleActionClient(self._action_name("stop"), StopAction)
        self._homing_client = actionlib.SimpleActionClient(self._action_name("homing"), HomingAction)
        self._wait_for_server(self._move_client, "move")
        self._wait_for_server(self._grasp_client, "grasp")
        self._wait_for_server(self._stop_client, "stop")
        self._joint_sub = rospy.Subscriber(settings.joint_states_topic, JointState, self._joint_state_callback, queue_size=1)

    def move(self, *, width: float, speed: float) -> None:
        goal = self._MoveGoal()
        goal.width = float(width)
        goal.speed = float(speed)
        with self._goal_lock:
            self._stop_requested.clear()
            self._grasp_client.cancel_all_goals()
            self._move_client.send_goal(goal)
        self._wait_for_result(self._move_client, "move")

    def grasp(self, *, width: float, speed: float, force: float, epsilon_inner: float, epsilon_outer: float) -> None:
        goal = self._GraspGoal()
        goal.width = float(width)
        goal.speed = float(speed)
        goal.force = float(force)
        goal.epsilon.inner = float(epsilon_inner)
        goal.epsilon.outer = float(epsilon_outer)
        with self._goal_lock:
            self._stop_requested.clear()
            self._move_client.cancel_all_goals()
            self._grasp_client.send_goal(goal)
        self._wait_for_result(self._grasp_client, "grasp")

    def stop(self) -> None:
        goal = self._StopGoal()
        with self._goal_lock:
            self._stop_requested.set()
            self._move_client.cancel_all_goals()
            self._grasp_client.cancel_all_goals()
            self._stop_client.send_goal(goal)
        self._wait_for_result(self._stop_client, "stop", allow_stop_request=True)

    def home(self) -> None:
        self._wait_for_server(self._homing_client, "homing")
        with self._goal_lock:
            self._stop_requested.clear()
            self._homing_client.send_goal(self._HomingGoal())
        self._wait_for_result(self._homing_client, "homing")

    def get_state(self) -> RosGripperState:
        with self._state_lock:
            state = RosGripperState(
                width=self._latest_state.width,
                force=self._latest_state.force,
                wall_time=self._latest_state.wall_time,
            )
        age = time.time() - state.wall_time if state.wall_time > 0.0 else float("inf")
        if age > float(self.settings.state_stale_after_sec):
            LOGGER.debug("Franka ROS gripper joint state is stale: age=%.3fs", age)
        return state

    def shutdown(self) -> None:
        try:
            self._joint_sub.unregister()
        except Exception:  # pragma: no cover - ROS shutdown path
            LOGGER.exception("Failed to unregister Franka ROS gripper joint state subscriber")

    def _joint_state_callback(self, msg) -> None:
        if not msg.position:
            return
        width = float(msg.position[0]) * 2.0
        force = 0.0
        if msg.effort:
            force = max(abs(float(value)) for value in msg.effort)
        with self._state_lock:
            self._latest_state = RosGripperState(width=width, force=force, wall_time=time.time())

    def _action_name(self, suffix: str) -> str:
        namespace = self.settings.action_namespace.strip("/")
        return f"/{namespace}/{suffix}" if namespace else f"/{suffix}"

    def _wait_for_server(self, client, label: str) -> None:
        timeout = self._rospy.Duration(float(self.settings.action_server_timeout_sec))
        if not client.wait_for_server(timeout):
            raise RuntimeError(f"Timed out waiting for franka_gripper {label} action server")

    def _wait_for_result(self, client, label: str, *, allow_stop_request: bool = False) -> None:
        timeout = self._rospy.Duration(float(self.settings.action_result_timeout_sec))
        if not client.wait_for_result(timeout):
            if not allow_stop_request:
                self.stop()
            raise RuntimeError(f"Timed out waiting for franka_gripper {label} action result")
        result = client.get_result()
        if self._stop_requested.is_set() and label in {"move", "grasp"}:
            return
        success = True if result is None or not hasattr(result, "success") else bool(result.success)
        if not success:
            error = "" if result is None or not hasattr(result, "error") else str(result.error)
            raise RuntimeError(f"franka_gripper {label} action failed: {error}")
