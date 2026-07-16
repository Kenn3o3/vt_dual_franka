from pathlib import Path

import numpy as np

from vt_dual_franka_shared.transforms import BimanualCalibration


def test_bimanual_calibration_loads_both_v6_assets():
    repo_root = Path(__file__).resolve().parents[2]
    calibration = BimanualCalibration.from_dir(repo_root / "robot_workspace/config/calibration/v6")
    assert calibration.left.world_to_robot_base.shape == (4, 4)
    assert calibration.right.world_to_robot_base.shape == (4, 4)
    unity_pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert calibration.left.unity_to_robot_pose(unity_pose).shape == (7,)
    assert calibration.right.unity_to_robot_pose(unity_pose).shape == (7,)
