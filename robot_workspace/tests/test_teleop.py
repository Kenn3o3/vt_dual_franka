from fastapi.testclient import TestClient
import numpy as np
from scipy.spatial.transform import Rotation

from vt_dual_franka_shared.models import ControllerState
from vt_dual_franka_shared.transforms import SingleArmCalibration
from vt_dual_franka_workspace.operator import OperatorLogBuffer
from vt_dual_franka_workspace.settings import TeleopSettings
from vt_dual_franka_workspace.teleop.quest_server import QuestTeleopService, create_teleop_app


class FakeController:
    def __init__(self):
        self.tcp_targets = []
        self.gripper_moves = []
        self.gripper_grasps = []

    def get_state(self):
        return ControllerState(tcp_pose=[0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])

    def queue_tcp(self, target_tcp, source="test", target_duration_sec=None):
        del source, target_duration_sec
        self.tcp_targets.append(target_tcp)

    def move_gripper(self, width, velocity, force_limit, source="test", blocking=False):
        self.gripper_moves.append((width, velocity, force_limit, source, blocking))
        return None

    def grasp_gripper(self, velocity, force_limit, source="test", blocking=False):
        self.gripper_grasps.append((velocity, force_limit, source, blocking))
        return None

    def stop_gripper(self):
        return None


def test_relative_target_swaps_xy_translation_and_flips_forward_axis():
    calibration = SingleArmCalibration.from_dir(
        "/home/zhenya/kenny/visuotact/vt_dual_franka/robot_workspace/config/calibration/v6"
    )
    controller = FakeController()
    service = QuestTeleopService(TeleopSettings(relative_translation_scale=1.0), controller, calibration)
    service._start_real_tcp = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
    service._start_hand_tcp = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    target_from_x = service._calculate_relative_target(np.array([0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(target_from_x[:3], [0.4, -0.1, 0.3])

    target_from_y = service._calculate_relative_target(np.array([0.0, 0.1, 0.0, 1.0, 0.0, 0.0, 0.0]))
    assert np.allclose(target_from_y[:3], [0.5, 0.0, 0.3])


def test_relative_rotation_preserves_latest_axis_mapping_and_flips_roll():
    calibration = SingleArmCalibration.from_dir(
        "/home/zhenya/kenny/visuotact/vt_dual_franka/robot_workspace/config/calibration/v6"
    )
    controller = FakeController()
    service = QuestTeleopService(TeleopSettings(relative_rotation_scale=1.0), controller, calibration)
    service._start_real_tcp = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
    service._start_hand_tcp = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    target_from_raw_x = service._calculate_relative_target(
        np.array([0.0, 0.0, 0.0, *Rotation.from_rotvec([0.2, 0.0, 0.0]).as_quat()[[3, 0, 1, 2]]])
    )
    rotation_from_raw_x = Rotation.from_quat(
        [target_from_raw_x[4], target_from_raw_x[5], target_from_raw_x[6], target_from_raw_x[3]]
    )
    assert np.allclose(rotation_from_raw_x.as_rotvec(), [0.0, -0.2, 0.0], atol=1e-6)

    target_from_raw_y = service._calculate_relative_target(
        np.array([0.0, 0.0, 0.0, *Rotation.from_rotvec([0.0, 0.2, 0.0]).as_quat()[[3, 0, 1, 2]]])
    )
    rotation_from_raw_y = Rotation.from_quat(
        [target_from_raw_y[4], target_from_raw_y[5], target_from_raw_y[6], target_from_raw_y[3]]
    )
    assert np.allclose(rotation_from_raw_y.as_rotvec(), [0.2, 0.0, 0.0], atol=1e-6)

    target_from_raw_z = service._calculate_relative_target(
        np.array([0.0, 0.0, 0.0, *Rotation.from_rotvec([0.0, 0.0, 0.2]).as_quat()[[3, 0, 1, 2]]])
    )
    rotation_from_raw_z = Rotation.from_quat(
        [target_from_raw_z[4], target_from_raw_z[5], target_from_raw_z[6], target_from_raw_z[3]]
    )
    assert np.allclose(rotation_from_raw_z.as_rotvec(), [0.0, 0.0, 0.2], atol=1e-6)


def test_teleop_endpoint_accepts_flattened_tactar_payload():
    calibration = SingleArmCalibration.from_dir(
        "/home/zhenya/kenny/visuotact/vt_dual_franka/robot_workspace/config/calibration/v6"
    )
    controller = FakeController()
    service = QuestTeleopService(TeleopSettings(relative_translation_scale=1.0), controller, calibration)
    app = create_teleop_app(service)

    payload = {
        "timestamp": 123.0,
        "leftHandPose": [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        "leftGripperState": 0.75,
        "buttonStates": {"button_4": True},
    }

    with TestClient(app) as client:
        response = client.post("/unity", json=payload)

    assert response.status_code == 200
    assert service._latest_message is not None
    assert service._latest_message.leftHand.wristPos == [0.1, 0.2, 0.3]
    assert service._latest_message.leftHand.triggerState == 0.75
    assert service._latest_message.leftHand.buttonState == [False, False, False, False, True]


def test_teleop_app_can_mount_operator_routes():
    calibration = SingleArmCalibration.from_dir(
        "/home/zhenya/kenny/visuotact/vt_dual_franka/robot_workspace/config/calibration/v6"
    )
    controller = FakeController()
    service = QuestTeleopService(TeleopSettings(relative_translation_scale=1.0), controller, calibration)

    class FakeOperator:
        def get_operator_status(self):
            return {
                "mode": "collect",
                "ready": False,
                "reasons": ["blocked"],
                "allowed_actions": {
                    "reset": True,
                    "confirm_gripper_closed": False,
                    "open_gripper": False,
                    "start": False,
                    "stop": False,
                    "mark_success": False,
                    "mark_fail": False,
                    "discard": False,
                    "quit": True,
                },
                "controller_state": {"age_sec": 0.0},
                "workers": {},
                "snapshots": {"third_person": {"available": False}},
            }

        def get_operator_snapshot(self, name: str):
            del name
            return None

        def operator_reset_ready_pose(self):
            return None

        def operator_confirm_gripper_closed(self):
            return None

        def operator_open_gripper(self):
            return None

        def operator_start_episode(self):
            return None

        def operator_stop_episode(self):
            return None

        def operator_mark_episode_success(self):
            return None

        def operator_mark_episode_fail(self):
            return None

        def operator_discard_latest_episode(self):
            return None

        def operator_quit(self):
            return None

    app = create_teleop_app(service, operator_controller=FakeOperator(), operator_log_buffer=OperatorLogBuffer())
    with TestClient(app) as client:
        response = client.get("/operator/api/status")

    assert response.status_code == 200
    assert response.json()["mode"] == "collect"


def test_teleop_forever_closed_records_closed_and_suppresses_open():
    calibration = SingleArmCalibration.from_dir(
        "/home/zhenya/kenny/visuotact/vt_dual_franka/robot_workspace/config/calibration/v6"
    )
    controller = FakeController()
    records = []

    class FakeCommandRecorder:
        def record_event(self, payload, event_time=None):
            del event_time
            records.append(payload)

    service = QuestTeleopService(
        TeleopSettings(relative_translation_scale=1.0),
        controller,
        calibration,
        command_recorder=FakeCommandRecorder(),
        state_provider=lambda: ControllerState(tcp_pose=[0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]),
        gripper_forever_closed=True,
    )
    service.set_teleop_enabled(True)
    service._tracking = True
    service._start_real_tcp = np.array([0.4, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0])
    service._start_hand_tcp = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    message = type(
        "Message",
        (),
        {
            "leftHand": type(
                "Hand",
                (),
                {
                    "wristPos": [0.0, 0.0, 0.0],
                    "wristQuat": [1.0, 0.0, 0.0, 0.0],
                    "triggerState": 0.0,
                    "buttonState": [False, False, False, False, True],
                },
            )()
        },
    )()

    service._handle_gripper_toggle(message)
    target_pose = service._calculate_relative_target(np.array(message.leftHand.wristPos + message.leftHand.wristQuat))
    controller.queue_tcp(target_pose.tolist(), source="teleop")
    service.command_recorder.record_event(
        {"source_wall_time": 1.0, "target_tcp": target_pose.tolist(), "gripper_closed": service._gripper_closed}
    )

    assert service._gripper_closed is True
    assert controller.gripper_moves == []
    assert len(controller.gripper_grasps) == 1
    assert records[-1]["gripper_closed"] is True
