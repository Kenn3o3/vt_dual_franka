from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

from vt_franka_workspace import cli
from vt_franka_workspace.config import load_inference_config, load_policy_config, load_task_config, load_workspace_config


def test_workspace_config_contains_only_global_runtime_sections():
    workspace = load_workspace_config("robot_workspace/config/workspace.yaml")

    assert workspace.controller.host == "10.0.0.1"
    assert workspace.recording.image_format == "jpg"
    assert workspace.recording.checkpoints_root.name == "checkpoints"
    assert not hasattr(workspace, "rollout")
    assert not hasattr(workspace, "collect")


def test_task_and_inference_configs_load_clean_modalities():
    task = load_task_config("robot_workspace/config/tasks/put_cup_on_plate.yaml")
    inference = load_inference_config("robot_workspace/config/inference/replay.yaml")
    policy = load_policy_config("robot_workspace/config/policies/replay.yaml")

    assert task.task_name == "put_cup_on_plate"
    assert task.modality.rgb_cameras == ["wrist", "third_person"]
    assert inference.obs_horizon == 2
    assert inference.exe_horizon == 8
    assert inference.eval.enabled is True
    assert inference.eval.cameras == ["third_person"]
    assert inference.eval.video_hz == 10.0
    assert inference.modality.rgb_cameras == []
    assert "third_person" in inference.rgb_cameras
    assert policy.type == "replay"


def test_cli_exposes_only_collect_and_run_policy(capsys):
    parser = argparse.ArgumentParser(description="VT Franka workspace CLI")
    del parser
    with pytest.raises(SystemExit):
        old_argv = sys.argv
        try:
            sys.argv = ["vt-franka-workspace", "--help"]
            cli.main()
        finally:
            sys.argv = old_argv
    help_text = capsys.readouterr().out
    assert "collect" in help_text
    assert "run-policy" in help_text
    assert "postprocess" not in help_text
    assert "auto-collect" not in help_text
    assert "rollout" not in help_text


def test_cli_collect_accepts_task_name(monkeypatch, tmp_path: Path):
    calls = {}
    config_dir = tmp_path / "config"
    (config_dir / "tasks").mkdir(parents=True)
    workspace_path = config_dir / "workspace.yaml"
    workspace_path.write_text(
        """
controller:
  host: 127.0.0.1
  port: 8092
  request_timeout_sec: 0.1
recording:
  collect_root: {root}
  eval_root: {eval_root}
operator_ui:
  enabled: false
calibration:
  calibration_dir: robot_workspace/config/calibration/v6
""".format(root=tmp_path / "collect", eval_root=tmp_path / "eval"),
        encoding="utf-8",
    )
    (config_dir / "tasks" / "demo.yaml").write_text(
        """
task_name: demo
initial_eef_pose_xyz_rpy_deg: [0, 0.4, 0.5, 180, 0, 0]
""",
        encoding="utf-8",
    )

    class FakeController:
        def __init__(self, host, port, request_timeout_sec):
            calls["controller"] = (host, port, request_timeout_sec)

    class FakeCollector:
        def __init__(self, workspace, task, controller, calibration, log_buffer=None):
            calls["task"] = task

        def run(self):
            calls["run"] = True

    monkeypatch.setattr(cli, "ControllerClient", FakeController)
    monkeypatch.setattr(cli, "SingleArmCalibration", type("FakeCalibration", (), {"from_dir": staticmethod(lambda path: "calib")}))
    monkeypatch.setattr(cli, "DataCollector", FakeCollector)
    old_argv = sys.argv
    try:
        sys.argv = [
            "vt-franka-workspace",
            "collect",
            "--workspace-config",
            str(workspace_path),
            "--task",
            "demo",
        ]
        cli.main()
    finally:
        sys.argv = old_argv

    assert calls["task"].task_name == "demo"
    assert calls["run"] is True


def test_cli_collect_constructs_data_collector(monkeypatch, tmp_path: Path):
    calls = {}
    workspace_path = tmp_path / "workspace.yaml"
    workspace_path.write_text(
        """
controller:
  host: 127.0.0.1
  port: 8092
  request_timeout_sec: 0.1
recording:
  collect_root: {root}
  eval_root: {eval_root}
operator_ui:
  enabled: false
calibration:
  calibration_dir: robot_workspace/config/calibration/v6
""".format(root=tmp_path / "collect", eval_root=tmp_path / "eval"),
        encoding="utf-8",
    )
    task_path = tmp_path / "task.yaml"
    task_path.write_text(
        """
task_name: demo
initial_eef_pose_xyz_rpy_deg: [0, 0.4, 0.5, 180, 0, 0]
modality:
  proprioception: true
""",
        encoding="utf-8",
    )

    class FakeController:
        def __init__(self, host, port, request_timeout_sec):
            calls["controller"] = (host, port, request_timeout_sec)

    class FakeCollector:
        def __init__(self, workspace, task, controller, calibration, log_buffer=None):
            calls["collector"] = (workspace, task, controller, calibration, log_buffer)

        def run(self):
            calls["run"] = True

    monkeypatch.setattr(cli, "ControllerClient", FakeController)
    monkeypatch.setattr(cli, "SingleArmCalibration", type("FakeCalibration", (), {"from_dir": staticmethod(lambda path: "calib")}))
    monkeypatch.setattr(cli, "DataCollector", FakeCollector)
    old_argv = sys.argv
    try:
        sys.argv = [
            "vt-franka-workspace",
            "collect",
            "--workspace-config",
            str(workspace_path),
            "--task-config",
            str(task_path),
            "--task-name",
            "override",
        ]
        cli.main()
    finally:
        sys.argv = old_argv

    assert calls["controller"] == ("127.0.0.1", 8092, 0.1)
    assert calls["collector"][1].task_name == "override"
    assert calls["run"] is True
