from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from vt_franka_workspace import cli
from vt_franka_workspace.config import EvalRuntimeSettings, load_inference_config, load_policy_config, load_task_config, load_workspace_config


def test_workspace_config_contains_only_global_runtime_sections():
    workspace = load_workspace_config("robot_workspace/config/workspace.yaml")

    assert workspace.controller.host == "10.0.0.1"
    assert workspace.recording.image_format == "jpg"
    assert workspace.recording.checkpoints_root.name == "checkpoints"
    assert workspace.operator_ui.preview_camera_role == "wrist"
    assert workspace.operator_ui.preview_refresh_hz == 5.0
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
    assert inference.eval.cameras == []
    assert inference.eval.stream_cameras == ["third_person"]
    assert inference.eval.video_hz == 10.0
    assert inference.modality.rgb_cameras == []
    assert "third_person" in inference.rgb_cameras
    assert policy.type == "replay"


def test_eval_recording_camera_aliases_include_gelsight_and_dedupe():
    settings = EvalRuntimeSettings(cameras=["wrist", "gelsight", "wrist"], stream_cameras=["third", "third_person"])

    assert settings.cameras == ["wrist", "gelsight"]
    assert settings.stream_cameras == ["third_person"]


def test_eval_action_step_cameras_reject_third_person():
    with pytest.raises(ValueError, match="stream_cameras"):
        EvalRuntimeSettings(cameras=["third"])


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
    assert "train-visuotactile" in help_text
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


def test_cli_train_visuotactile_preserves_subcommand_when_command_override(monkeypatch, tmp_path: Path):
    calls = {}
    workspace_path = tmp_path / "workspace.yaml"
    workspace_path.write_text(
        """
controller:
  host: 127.0.0.1
  port: 8092
  request_timeout_sec: 0.1
recording:
  collect_root: {collect_root}
  prepared_root: {prepared_root}
  checkpoints_root: {checkpoints_root}
operator_ui:
  enabled: false
calibration:
  calibration_dir: robot_workspace/config/calibration/v6
""".format(
            collect_root=tmp_path / "collect",
            prepared_root=tmp_path / "prepared",
            checkpoints_root=tmp_path / "checkpoints",
        ),
        encoding="utf-8",
    )

    class FakeResult:
        checkpoint_dir = tmp_path / "ckpt"
        dataset_dir = tmp_path / "dataset"
        backend_dataset_root = tmp_path / "backend"
        manifest_path = tmp_path / "ckpt" / "policy_manifest.json"

    def fake_train(config):
        calls["config"] = config
        return FakeResult()

    monkeypatch.setattr("vt_franka_workspace.policies.visuotactile.train.train_visuotactile", fake_train)
    old_argv = sys.argv
    try:
        sys.argv = [
            "vt-franka-workspace",
            "train-visuotactile",
            "--workspace-config",
            str(workspace_path),
            "--task-name",
            "usb_insertion",
            "--model",
            "dp_manifeel",
            "--dry-run",
            "--command",
            "echo",
            "ok",
        ]
        cli.main()
    finally:
        sys.argv = old_argv

    assert calls["config"].task_name == "usb_insertion"
    assert calls["config"].command_override == ["echo", "ok"]


def test_cli_run_policy_resolves_checkpoint_first(monkeypatch, tmp_path: Path):
    calls = {}
    workspace_path = tmp_path / "workspace.yaml"
    workspace_path.write_text(
        """
controller:
  host: 127.0.0.1
  port: 8092
  request_timeout_sec: 0.1
recording:
  collect_root: {collect_root}
  eval_root: {eval_root}
  checkpoints_root: {checkpoints_root}
operator_ui:
  enabled: false
calibration:
  calibration_dir: robot_workspace/config/calibration/v6
""".format(
            collect_root=tmp_path / "collect",
            eval_root=tmp_path / "eval",
            checkpoints_root=tmp_path / "checkpoints",
        ),
        encoding="utf-8",
    )
    inference_path = tmp_path / "inference.yaml"
    inference_path.write_text(
        """
task_name: policy_run
obs_horizon: 2
exe_horizon: 1
eval:
  enabled: false
operator_ui:
  enabled: false
""",
        encoding="utf-8",
    )
    checkpoint_root = tmp_path / "checkpoints" / "pencil_insertion" / "vista_so3"
    checkpoint_file = checkpoint_root / "checkpoints" / "epoch=209.ckpt"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_bytes(b"fake")
    for name, payload in {
        "policy_manifest.json": {"model": "vista_so3", "task_name": "pencil_insertion"},
        "preprocess1_manifest.json": {},
        "preprocess2_manifest.json": {},
        "normalizer_stats.json": {},
    }.items():
        (checkpoint_root / name).write_text(json.dumps(payload), encoding="utf-8")

    class FakeController:
        def __init__(self, host, port, request_timeout_sec):
            calls["controller"] = (host, port, request_timeout_sec)

    class FakeRunner:
        def __init__(self, workspace, inference, controller, calibration, policy, run_name=None, log_buffer=None, resume_run=True):
            calls["runner"] = {
                "workspace": workspace,
                "inference": inference,
                "controller": controller,
                "calibration": calibration,
                "policy": policy,
                "run_name": run_name,
                "resume_run": resume_run,
            }

        def run(self):
            calls["run"] = True

    def fake_resolve_policy(policy_config, inference_config, workspace):
        calls["policy_config"] = policy_config
        calls["inference_config"] = inference_config
        calls["policy_workspace"] = workspace
        return "policy"

    monkeypatch.setattr(cli, "ControllerClient", FakeController)
    monkeypatch.setattr(cli, "SingleArmCalibration", type("FakeCalibration", (), {"from_dir": staticmethod(lambda path: "calib")}))
    monkeypatch.setattr(cli, "resolve_policy", fake_resolve_policy)
    monkeypatch.setattr(cli, "PolicyRunner", FakeRunner)
    old_argv = sys.argv
    try:
        sys.argv = [
            "vt-franka-workspace",
            "run-policy",
            "--workspace-config",
            str(workspace_path),
            "--checkpoint",
            str(checkpoint_file),
            "--inference-config",
            str(inference_path),
            "--no-resume",
        ]
        cli.main()
    finally:
        sys.argv = old_argv

    policy_config = calls["policy_config"]
    assert policy_config.type == "visuotactile"
    assert policy_config.checkpoint_path == checkpoint_root
    assert policy_config.config["model"] == "vista_so3"
    assert policy_config.config["task_name"] == "pencil_insertion"
    assert policy_config.config["checkpoint_file"] == "checkpoints/epoch=209.ckpt"
    assert calls["inference_config"].task_name == "pencil_insertion"
    assert calls["runner"]["resume_run"] is False
    assert calls["run"] is True


def test_cli_run_policy_resume_uses_checkpoint_run_name(monkeypatch, tmp_path: Path):
    calls = {}
    workspace_path = tmp_path / "workspace.yaml"
    workspace_path.write_text(
        """
controller:
  host: 127.0.0.1
  port: 8092
  request_timeout_sec: 0.1
recording:
  collect_root: {collect_root}
  eval_root: {eval_root}
  checkpoints_root: {checkpoints_root}
operator_ui:
  enabled: false
calibration:
  calibration_dir: robot_workspace/config/calibration/v6
""".format(
            collect_root=tmp_path / "collect",
            eval_root=tmp_path / "eval",
            checkpoints_root=tmp_path / "checkpoints",
        ),
        encoding="utf-8",
    )
    inference_path = tmp_path / "inference.yaml"
    inference_path.write_text(
        """
task_name: policy_run
obs_horizon: 2
exe_horizon: 1
eval:
  enabled: false
""",
        encoding="utf-8",
    )
    checkpoint_root = tmp_path / "checkpoints" / "pencil_insertion" / "vista_so3"
    checkpoint_file = checkpoint_root / "checkpoints" / "epoch=209.ckpt"
    checkpoint_file.parent.mkdir(parents=True)
    checkpoint_file.write_bytes(b"fake")
    for name, payload in {
        "policy_manifest.json": {"model": "vista_so3", "task_name": "pencil_insertion"},
        "preprocess1_manifest.json": {},
        "preprocess2_manifest.json": {},
        "normalizer_stats.json": {},
    }.items():
        (checkpoint_root / name).write_text(json.dumps(payload), encoding="utf-8")

    class FakeController:
        def __init__(self, host, port, request_timeout_sec):
            del host, port, request_timeout_sec

    class FakeRunner:
        def __init__(self, workspace, inference, controller, calibration, policy, run_name=None, log_buffer=None, resume_run=True):
            del workspace, inference, controller, calibration, policy, log_buffer
            calls["run_name"] = run_name
            calls["resume_run"] = resume_run

        def run(self):
            calls["run"] = True

    monkeypatch.setattr(cli, "ControllerClient", FakeController)
    monkeypatch.setattr(cli, "SingleArmCalibration", type("FakeCalibration", (), {"from_dir": staticmethod(lambda path: "calib")}))
    monkeypatch.setattr(cli, "resolve_policy", lambda policy_config, inference_config, workspace: "policy")
    monkeypatch.setattr(cli, "PolicyRunner", FakeRunner)
    old_argv = sys.argv
    try:
        sys.argv = [
            "vt-franka-workspace",
            "run-policy",
            "--workspace-config",
            str(workspace_path),
            "--checkpoint",
            str(checkpoint_file),
            "--inference-config",
            str(inference_path),
            "--resume",
        ]
        cli.main()
    finally:
        sys.argv = old_argv

    assert calls["run_name"] == "epoch_209"
    assert calls["resume_run"] is True
    assert calls["run"] is True
