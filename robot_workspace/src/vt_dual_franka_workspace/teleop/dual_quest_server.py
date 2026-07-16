from __future__ import annotations

import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from threading import Event, Lock, Thread

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from scipy.spatial.transform import Rotation

from vt_dual_franka_shared.models import ArmId, ControllerState, QuestHandState, UnityTeleopMessage, parse_unity_teleop_message
from vt_dual_franka_shared.timing import precise_sleep
from vt_dual_franka_shared.transforms import BimanualCalibration

from ..config import TeleopSettings
from ..recording.raw_recorder import JsonlStreamRecorder
from ..runtime.dual_arm import ARM_ORDER, DualArmCoordinator

LOGGER = logging.getLogger(__name__)


@dataclass
class _ArmTeleopState:
    tracking: bool = False
    start_real_tcp: np.ndarray | None = None
    start_hand_tcp: np.ndarray | None = None
    gripper_closed: bool = False
    gripper_force: float = 0.0
    width_history: list[float] = field(default_factory=list)


class DualQuestTeleopService:
    """Natural Quest mapping: left hand controls left arm, right hand controls right arm."""

    def __init__(
        self,
        settings: TeleopSettings,
        coordinator: DualArmCoordinator,
        calibration: BimanualCalibration,
        *,
        command_recorder: JsonlStreamRecorder | None = None,
        quest_message_recorder: JsonlStreamRecorder | None = None,
    ) -> None:
        self.settings = settings
        self.coordinator = coordinator
        self.calibration = calibration
        self.command_recorder = command_recorder
        self.quest_message_recorder = quest_message_recorder
        self._states: dict[ArmId, _ArmTeleopState] = {arm: _ArmTeleopState() for arm in ARM_ORDER}
        self._latest_message: UnityTeleopMessage | None = None
        self._last_message_wall_time: float | None = None
        self._message_lock = Lock()
        self._running = Event()
        self._teleop_enabled = False
        self._loop_thread: Thread | None = None
        self._operator_yaw_rot: Rotation | None = None
        if self.settings.operator_yaw_offset_deg != 0.0:
            self._operator_yaw_rot = Rotation.from_euler("z", self.settings.operator_yaw_offset_deg, degrees=True)

    def submit_message(self, message: UnityTeleopMessage) -> None:
        source_wall_time = time.time()
        with self._message_lock:
            self._latest_message = message
            self._last_message_wall_time = source_wall_time
        if self.quest_message_recorder is not None:
            self.quest_message_recorder.record_event(
                {"quest_timestamp": message.timestamp, "source_wall_time": source_wall_time, "message": message.model_dump(mode="json")},
                event_time=source_wall_time,
            )

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self._loop_thread = Thread(target=self._control_loop, name="dual-quest-teleop-loop", daemon=True)
        self._loop_thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)

    def set_teleop_enabled(self, enabled: bool) -> None:
        self._teleop_enabled = bool(enabled)
        if not enabled:
            for state in self._states.values():
                state.tracking = False

    def has_recent_message(self, timeout_sec: float) -> bool:
        with self._message_lock:
            last = self._last_message_wall_time
        return last is not None and time.time() - last <= timeout_sec

    def get_gripper_status(self) -> dict[str, bool]:
        payload: dict[str, bool] = {}
        for arm_id in ARM_ORDER:
            state = self._states[arm_id]
            stable_closed, stable_open = self._gripper_stability(state)
            payload[f"{arm_id}_gripper_stable_closed"] = stable_closed
            payload[f"{arm_id}_gripper_stable_open"] = stable_open
        return payload

    def _control_loop(self) -> None:
        period = 1.0 / self.settings.loop_hz
        while self._running.is_set():
            try:
                message = self._latest_message_copy()
                if message is None or not self._teleop_enabled:
                    precise_sleep(period)
                    continue
                dual_state = self.coordinator.get_state(max_age_sec=2.0)
                self._update_gripper_states({"left": dual_state.left, "right": dual_state.right})
                targets: dict[ArmId, list[float]] = {}
                for arm_id in ARM_ORDER:
                    hand = message.leftHand if arm_id == "left" else message.rightHand
                    state = dual_state.left if arm_id == "left" else dual_state.right
                    target = self._target_for_arm(arm_id, hand, state)
                    if target is not None:
                        targets[arm_id] = target
                        self._handle_gripper(arm_id, hand)
                    elif self._states[arm_id].tracking:
                        targets[arm_id] = list(state.tcp_pose)
                if len(targets) == 2:
                    command_id = self.coordinator.queue_tcp_pair(targets, source="dual_teleop", target_duration_sec=period)
                    self._record_command(command_id, targets)
            except Exception as exc:
                LOGGER.warning("Dual teleop iteration failed: %s", exc)
            precise_sleep(period)

    def _target_for_arm(self, arm_id: ArmId, hand: QuestHandState, state: ControllerState) -> list[float] | None:
        hand_pose = self.calibration.arm(arm_id).unity_to_robot_pose(np.asarray(hand.wristPos + hand.wristQuat, dtype=np.float64))
        if self._operator_yaw_rot is not None:
            hand_pose = _rotate_pose(hand_pose, self._operator_yaw_rot)
        arm_state = self._states[arm_id]
        tracking_pressed = _button_pressed(hand, self.settings.tracking_button_index)
        if tracking_pressed and not arm_state.tracking:
            arm_state.tracking = True
            arm_state.start_real_tcp = np.asarray(state.tcp_pose, dtype=np.float64)
            arm_state.start_hand_tcp = hand_pose
        elif not tracking_pressed and arm_state.tracking:
            arm_state.tracking = False
            self.coordinator.arms[arm_id].controller.stop_gripper()
        if not arm_state.tracking:
            return None
        if arm_state.start_real_tcp is None or arm_state.start_hand_tcp is None:
            return None
        target = _calculate_relative_target(self.settings, hand_pose, arm_state.start_hand_tcp, arm_state.start_real_tcp)
        if np.linalg.norm(target[:3] - np.asarray(state.tcp_pose[:3], dtype=np.float64)) > self.settings.max_tracking_position_error_m:
            arm_state.tracking = False
            return None
        return target.tolist()

    def _handle_gripper(self, arm_id: ArmId, hand: QuestHandState) -> None:
        state = self._states[arm_id]
        wants_closed = hand.triggerState > self.settings.trigger_close_threshold or _button_pressed(hand, 3)
        controller = self.coordinator.arms[arm_id].controller
        if wants_closed and not state.gripper_closed:
            if self.settings.use_force_control_for_gripper:
                controller.grasp_gripper(self.settings.gripper_velocity, self.settings.grasp_force, source="dual_teleop")
            else:
                controller.move_gripper(self.settings.min_gripper_width, self.settings.gripper_velocity, self.settings.grasp_force, source="dual_teleop")
            state.gripper_closed = True
        elif not wants_closed and state.gripper_closed:
            controller.move_gripper(self.settings.max_gripper_width, self.settings.gripper_velocity, self.settings.grasp_force, source="dual_teleop")
            state.gripper_closed = False

    def _update_gripper_states(self, states: dict[ArmId, ControllerState]) -> None:
        for arm_id, controller_state in states.items():
            teleop_state = self._states[arm_id]
            teleop_state.gripper_force = float(controller_state.gripper_force)
            teleop_state.width_history.append(float(controller_state.gripper_width))
            teleop_state.width_history = teleop_state.width_history[-self.settings.gripper_stability_window :]

    def _gripper_stability(self, state: _ArmTeleopState) -> tuple[bool, bool]:
        if len(state.width_history) < self.settings.gripper_stability_window:
            return False, False
        variation = max(state.width_history) - min(state.width_history)
        stable = variation < self.settings.gripper_width_vis_precision
        return state.gripper_force >= self.settings.gripper_force_close_threshold and stable, state.gripper_force < self.settings.gripper_force_open_threshold and stable

    def _record_command(self, command_id: str, targets: dict[ArmId, list[float]]) -> None:
        if self.command_recorder is None:
            return
        self.command_recorder.record_event(
            {
                "schema_version": "vt_dual_franka_commanded_action_v1",
                "source_wall_time": time.time(),
                "command_id": command_id,
                "target_tcp": {arm_id: targets[arm_id] for arm_id in ARM_ORDER},
                "arm_order": list(ARM_ORDER),
            },
            event_time=time.time(),
        )

    def _latest_message_copy(self) -> UnityTeleopMessage | None:
        with self._message_lock:
            return self._latest_message.model_copy(deep=True) if self._latest_message is not None else None


def create_dual_teleop_app(service: DualQuestTeleopService) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service.start()
        try:
            yield
        finally:
            service.stop()

    app = FastAPI(title="VT Dual Franka Teleop", version="0.1.0", lifespan=lifespan)

    @app.post("/unity")
    async def unity(request: Request):
        try:
            payload = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="Invalid Quest teleop JSON") from exc
        try:
            message = parse_unity_teleop_message(payload)
        except Exception as exc:
            return JSONResponse(status_code=422, content={"detail": str(exc)})
        service.submit_message(message)
        return {"status": "ok"}

    @app.get("/get_current_gripper_state")
    def get_current_gripper_state():
        return service.get_gripper_status()

    return app


def _calculate_relative_target(settings: TeleopSettings, current_hand: np.ndarray, start_hand: np.ndarray, start_real: np.ndarray) -> np.ndarray:
    target = np.zeros(7, dtype=np.float64)
    raw_delta = current_hand[:3] - start_hand[:3]
    translation_delta = np.array([raw_delta[1], -raw_delta[0], raw_delta[2]], dtype=np.float64)
    target[:3] = settings.relative_translation_scale * translation_delta + start_real[:3]
    current_rotation = Rotation.from_quat(_wxyz_to_xyzw(current_hand[3:]))
    start_rotation = Rotation.from_quat(_wxyz_to_xyzw(start_hand[3:]))
    robot_start_rotation = Rotation.from_quat(_wxyz_to_xyzw(start_real[3:]))
    raw_rotvec = (current_rotation * start_rotation.inv()).as_rotvec()
    rotvec = np.array([raw_rotvec[1], -raw_rotvec[0], raw_rotvec[2]], dtype=np.float64)
    relative_rotation = Rotation.from_rotvec(rotvec * settings.relative_rotation_scale)
    quat_xyzw = (relative_rotation * robot_start_rotation).as_quat()
    target[3:] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return target


def _button_pressed(hand: QuestHandState, index: int) -> bool:
    return index < len(hand.buttonState) and bool(hand.buttonState[index])


def _rotate_pose(pose7: np.ndarray, rotation: Rotation) -> np.ndarray:
    rotated = pose7.copy()
    rotated[:3] = rotation.apply(pose7[:3])
    original_rot = Rotation.from_quat(_wxyz_to_xyzw(pose7[3:]))
    quat_xyzw = (rotation * original_rot).as_quat()
    rotated[3:] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
    return rotated


def _wxyz_to_xyzw(quaternion_wxyz: np.ndarray) -> np.ndarray:
    return np.array([quaternion_wxyz[1], quaternion_wxyz[2], quaternion_wxyz[3], quaternion_wxyz[0]], dtype=np.float64)
