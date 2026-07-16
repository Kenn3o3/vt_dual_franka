from __future__ import annotations

import json
import sys
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vt_dual_franka_workspace.config import InferenceRuntimeSettings, ModalitySettings, PolicyConfig, WorkspaceSettings
from vt_dual_franka_workspace.policies.registry import resolve_policy
from vt_dual_franka_workspace.policies.visuotactile.canonical import (
    CanonicalPreprocessConfig,
    build_preprocess1_from_collection_streams,
    preprocess_aligned_episode_images,
    write_preprocess1_dataset_manifest,
)
from vt_dual_franka_workspace.policies.visuotactile.export_backend import export_prepared_dataset_for_backend
from vt_dual_franka_workspace.policies.visuotactile.image_preprocess import CropSpec, ImagePreprocessSpec, preprocess_image_rgb
from vt_dual_franka_workspace.policies.visuotactile.prepare import build_prepare_config_from_workspace, prepare_visuotactile_dataset
from vt_dual_franka_workspace.policies.visuotactile.remote import RemoteTrainConfig, remote_train_visuotactile
from vt_dual_franka_workspace.policies.visuotactile.train import TrainVisuotactileConfig, train_visuotactile
from vt_dual_franka_workspace.policies.visuotactile.runtime import RuntimeManifests, RuntimePreprocessor, action_row_to_vt_action
from vt_dual_franka_workspace.policies.visuotactile.config import VisuotactilePolicySettings, get_model_spec
from vt_dual_franka_workspace.recording import align_episode, default_canonical_stream_specs
from vt_dual_franka_workspace.recording.canonical_preprocess1 import CanonicalPreprocess1StreamRecorder
from vt_dual_franka_workspace.sensors.standardization import standardize_camera_frame


def test_preprocess_image_rgb_center_square_resize() -> None:
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    image[:, 2:6] = [10, 20, 30]

    out = preprocess_image_rgb(
        image,
        ImagePreprocessSpec(output_size=(2, 2), crop=CropSpec(mode="center_square")),
    )

    assert out.shape == (2, 2, 3)
    assert np.all(out == [10, 20, 30])


def test_preprocess_and_prepare_visuotactile_dataset(tmp_path: Path) -> None:
    run_dir = tmp_path / "collect" / "usb_insertion"
    _write_fake_episode(run_dir / "episodes" / "episode_0000", offset=0.0)
    _write_fake_episode(run_dir / "episodes" / "episode_0001", offset=10.0)
    workspace = WorkspaceSettings(
        recording={
            "collect_root": tmp_path / "collect",
            "preprocess1_root": tmp_path / "preprocess1",
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )

    pre = preprocess_aligned_episode_images(
        run_dir / "episodes" / "episode_0000",
        CanonicalPreprocessConfig(canonical_size=16, chunk_frames=2, overwrite=True),
    )
    assert pre.kept_steps == 4
    assert (pre.output_dir / "preprocess1_manifest.json").exists()
    assert (pre.output_dir / "canonical_episode.npz").exists()
    assert (pre.output_dir / "chunks" / "chunk_000000.npz").exists()

    config = build_prepare_config_from_workspace(
        workspace,
        task_name="usb_insertion",
        model="dp_manifeel",
        raw_run_dir=run_dir,
        output_dir=tmp_path / "prepared_dataset",
        image_size=8,
        canonical_size=16,
        val_episodes=1,
        overwrite=True,
    )
    result = prepare_visuotactile_dataset(config)

    assert result.train_episodes == 1
    assert result.val_episodes == 1
    assert result.total_steps == 8
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["model"] == "dp_manifeel"
    assert manifest["preprocess2"]["model_image_size"] == 8
    assert manifest["preprocess1_storage"] == "centralized"
    assert manifest["preprocess1_root"] == str(tmp_path / "preprocess1" / "usb_insertion" / "real_canonical_v1")
    with np.load(result.output_dir / "train" / "episode_0000.npz") as data:
        assert data["rgb_wrist"].shape == (4, 8, 8, 3)
        assert data["gelsight"].shape == (4, 8, 8, 3)
        assert data["qpos_pose10_rot6d_gripper"].shape == (4, 10)


def test_prepare_visuotactile_dataset_from_preprocess1_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "collect" / "usb_insertion"
    _write_fake_episode(run_dir / "episodes" / "episode_0000", offset=0.0)
    _write_fake_episode(run_dir / "episodes" / "episode_0001", offset=10.0)
    workspace = WorkspaceSettings(
        recording={
            "collect_root": tmp_path / "collect",
            "preprocess1_root": tmp_path / "preprocess1",
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )
    raw_prepared = prepare_visuotactile_dataset(
        build_prepare_config_from_workspace(
            workspace,
            task_name="usb_insertion",
            model="dp_manifeel",
            raw_run_dir=run_dir,
            output_dir=tmp_path / "prepared_raw",
            image_size=8,
            canonical_size=16,
            val_episodes=1,
            overwrite=True,
        )
    )
    preprocess1_root = tmp_path / "preprocess1" / "usb_insertion" / "real_canonical_v1"
    write_preprocess1_dataset_manifest(
        preprocess1_root,
        task_name="usb_insertion",
        profile_name="real_canonical_v1",
        raw_run_dir=run_dir,
    )
    p1_prepared = prepare_visuotactile_dataset(
        build_prepare_config_from_workspace(
            workspace,
            task_name="usb_insertion",
            model="dp_manifeel",
            output_dir=tmp_path / "prepared_p1",
            image_size=8,
            val_episodes=1,
            overwrite=True,
            source="preprocess1",
            source_root=preprocess1_root,
        )
    )

    assert p1_prepared.total_steps == raw_prepared.total_steps
    manifest = json.loads(p1_prepared.manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"] == "preprocess1"
    with np.load(raw_prepared.output_dir / "train" / "episode_0000.npz") as raw, np.load(
        p1_prepared.output_dir / "train" / "episode_0000.npz"
    ) as p1:
        np.testing.assert_allclose(raw["qpos_pose10_rot6d_gripper"], p1["qpos_pose10_rot6d_gripper"])
        np.testing.assert_allclose(raw["action_pose10_rot6d_gripper"], p1["action_pose10_rot6d_gripper"])
        np.testing.assert_array_equal(raw["rgb_wrist"], p1["rgb_wrist"])


def test_collection_time_preprocess1_bundle_matches_offline_preprocess1(tmp_path: Path) -> None:
    run_dir = tmp_path / "collect" / "usb_insertion"
    episode_dir = run_dir / "episodes" / "episode_0000"
    _write_fake_raw_stream_episode(episode_dir, textured_images=True)
    offline = preprocess_aligned_episode_images(
        episode_dir,
        CanonicalPreprocessConfig(canonical_size=8, chunk_frames=2, overwrite=True),
    )
    _write_fake_raw_stream_episode(episode_dir, textured_images=True)
    specs = default_canonical_stream_specs(
        canonical_size=8,
        gelsight_crop_box=None,
        gelsight_margin_fraction=0.0,
        wrist_raw_jpeg_compat=True,
    )
    sessions = _StaticEpisodeSession(episode_dir)
    rgb_recorder = CanonicalPreprocess1StreamRecorder(sessions, specs["rgb_wrist"], queue_size=4, chunk_frames=2)
    gel_recorder = CanonicalPreprocess1StreamRecorder(sessions, specs["gelsight"], queue_size=4, chunk_frames=2)
    for i in range(4):
        rgb_recorder.record_frame(_fake_wrist_frame_bgr(i, textured_images=True), captured_wall_time=i * 0.1, sequence_id=i)
        gel_recorder.record_frame(_fake_gelsight_frame_bgr(i, textured_images=True), captured_wall_time=i * 0.1, sequence_id=i)
    rgb_recorder.flush_episode(episode_dir)
    gel_recorder.flush_episode(episode_dir)
    rgb_recorder.close()
    gel_recorder.close()
    align_episode(episode_dir, overwrite=True)
    collection = build_preprocess1_from_collection_streams(
        episode_dir,
        CanonicalPreprocessConfig(
            canonical_size=8,
            chunk_frames=2,
            overwrite=True,
            output_root=tmp_path / "collection_preprocess1",
        ),
    )

    with np.load(offline.output_dir / "chunks" / "chunk_000000.npz") as offline_chunk, np.load(
        collection.output_dir / "chunks" / "chunk_000000.npz"
    ) as collection_chunk:
        np.testing.assert_array_equal(collection_chunk["rgb_wrist"], offline_chunk["rgb_wrist"])
        np.testing.assert_array_equal(collection_chunk["gelsight"], offline_chunk["gelsight"])


def test_visuotactile_registry_accepts_fake_backend(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt"
    checkpoint.mkdir()
    manifests = RuntimeManifests(
        policy={
            "schema_version": "test",
            "model": "dp_manifeel",
            "obs_horizon": 2,
            "action_horizon": 1,
            "action_dim": 10,
        },
        preprocess1={
            "schema_version": "test",
            "streams": {
                "rgb_wrist": {"preprocess": ImagePreprocessSpec(output_size=(4, 4), crop=CropSpec(mode="none")).to_json()},
                "gelsight": {"preprocess": ImagePreprocessSpec(output_size=(4, 4), crop=CropSpec(mode="none")).to_json()},
            },
        },
        preprocess2={
            "schema_version": "test",
            "preprocess2": {
                "rgb_wrist": ImagePreprocessSpec(output_size=(4, 4), crop=CropSpec(mode="none")).to_json(),
                "gelsight": ImagePreprocessSpec(output_size=(4, 4), crop=CropSpec(mode="none")).to_json(),
            },
        },
        normalizer_stats={},
    )
    for name, payload in {
        "policy_manifest.json": manifests.policy,
        "preprocess1_manifest.json": manifests.preprocess1,
        "preprocess2_manifest.json": manifests.preprocess2,
        "normalizer_stats.json": manifests.normalizer_stats,
    }.items():
        (checkpoint / name).write_text(json.dumps(payload), encoding="utf-8")

    policy = resolve_policy(
        PolicyConfig(type="visuotactile", checkpoint_path=checkpoint, config={"model": "manifeel"}),
        InferenceRuntimeSettings(
            task_name="usb_insertion",
            control_hz=10.0,
            modality=ModalitySettings(proprioception=True, rgb_cameras=["wrist"], gelsight_frame=True),
        ),
        WorkspaceSettings(recording={"checkpoints_root": tmp_path / "checkpoints"}),
    )

    assert policy.__class__.__name__ == "VisuotactilePolicy"


def test_visuotactile_policy_settings_accept_sampling_override() -> None:
    settings = VisuotactilePolicySettings.model_validate(
        {
            "model": "vista_so3",
            "sampling_scheduler": "DDIM",
            "num_inference_steps": 16,
        }
    )

    assert settings.sampling_scheduler == "ddim"
    assert settings.num_inference_steps == 16

    with pytest.raises(ValueError):
        VisuotactilePolicySettings.model_validate({"model": "vista_so3", "num_inference_steps": 0})


def test_visuotactile_pose10_action_row_uses_pytorch3d_row_convention() -> None:
    rot = Rotation.from_euler("xyz", [20.0, 30.0, 40.0], degrees=True)
    row6 = rot.as_matrix()[:2, :].reshape(6)
    row = np.asarray([0.1, 0.2, 0.3, *row6, 0.75], dtype=np.float64)

    action = action_row_to_vt_action(
        row,
        model_spec=get_model_spec("vista_so3"),
        target_duration_sec=0.1,
        gripper_open_width_m=0.078,
        gripper_close_threshold=0.5,
    )

    assert action["metadata"]["visuotactile_rot6d_convention"] == "pytorch3d_first_two_rows"
    target_tcp = np.asarray(action["target_tcp"], dtype=np.float64)
    decoded = Rotation.from_quat([target_tcp[4], target_tcp[5], target_tcp[6], target_tcp[3]])
    np.testing.assert_allclose(decoded.as_matrix(), rot.as_matrix(), atol=1e-6)
    np.testing.assert_allclose(target_tcp[:3], row[:3], atol=1e-8)


def test_train_visuotactile_dry_run_uses_existing_dataset(tmp_path: Path) -> None:
    run_dir = tmp_path / "collect" / "usb_insertion"
    _write_fake_episode(run_dir / "episodes" / "episode_0000", offset=0.0)
    _write_fake_episode(run_dir / "episodes" / "episode_0001", offset=10.0)
    workspace = WorkspaceSettings(
        recording={
            "collect_root": tmp_path / "collect",
            "preprocess1_root": tmp_path / "preprocess1",
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )
    prepared = prepare_visuotactile_dataset(
        build_prepare_config_from_workspace(
            workspace,
            task_name="usb_insertion",
            model="vista_so2",
            raw_run_dir=run_dir,
            output_dir=tmp_path / "prepared_dataset",
            image_size=8,
            canonical_size=16,
            val_episodes=1,
            overwrite=True,
        )
    )

    result = train_visuotactile(
        TrainVisuotactileConfig(
            workspace=workspace,
            task_name="usb_insertion",
            model="vista_so2",
            dataset_dir=prepared.output_dir,
            checkpoint_dir=tmp_path / "ckpt",
            dry_run=True,
        )
    )

    assert result.dataset_dir == prepared.output_dir
    assert result.checkpoint_dir == tmp_path / "ckpt"
    assert result.backend_dataset_root == tmp_path / "ckpt" / "backend_dataset"
    assert "--config-name=train_vista_so2" in result.command
    assert f"dataset_root={tmp_path / 'ckpt' / 'backend_dataset'}" in result.command
    assert "task=univtac_vista_lr" in result.command
    assert "shape_meta.obs.robot0_eye_in_hand_image.shape=[3,224,224]" not in result.command
    assert "task.shape_meta.obs.robot0_eye_in_hand_image.shape=[3,224,224]" in result.command
    assert "policy.tactile_shape=[3,224,224]" in result.command
    assert "n_demo=1" in result.command


def test_runtime_preprocess_applies_vista_jpeg_compat_after_prepared_frame(tmp_path: Path) -> None:
    run_dir = tmp_path / "collect" / "usb_insertion"
    episode_dir = run_dir / "episodes" / "episode_0000"
    _write_fake_episode(episode_dir, offset=0.0, textured_images=True)
    _write_fake_episode(run_dir / "episodes" / "episode_0001", offset=10.0, textured_images=True)
    workspace = WorkspaceSettings(
        recording={
            "collect_root": tmp_path / "collect",
            "preprocess1_root": tmp_path / "preprocess1",
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )
    prepared = prepare_visuotactile_dataset(
        build_prepare_config_from_workspace(
            workspace,
            task_name="usb_insertion",
            model="vista_so3",
            raw_run_dir=run_dir,
            output_dir=tmp_path / "prepared_dataset",
            image_size=8,
            canonical_size=16,
            val_episodes=1,
            overwrite=True,
        )
    )
    manifest = json.loads((prepared.output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    p1_manifest = json.loads(Path(manifest["episodes"][0]["preprocess1_manifest"]).read_text(encoding="utf-8"))
    manifests = RuntimeManifests(
        policy={
            "schema_version": "test",
            "model": "vista_so3",
            "obs_horizon": 2,
            "action_horizon": 8,
            "action_dim": 10,
        },
        preprocess1=p1_manifest,
        preprocess2={
            "schema_version": "test",
            "preprocess2": manifest["preprocess2"],
        },
        normalizer_stats={},
    )
    preprocessor = RuntimePreprocessor(manifests, gripper_open_width_m=0.078)

    rgb_std = standardize_camera_frame(
        _fake_wrist_frame_bgr(0, textured_images=True),
        stream_name="rgb_wrist",
        camera_name="test_wrist",
        source_color="BGR",
    )
    gelsight_std = standardize_camera_frame(
        _fake_gelsight_frame_bgr(0, textured_images=True),
        stream_name="tactile_left",
        camera_name="test_gelsight",
        source_color="BGR",
    )
    observation = {
        "images": {"wrist": {"image": rgb_std.image_rgb, "metadata": rgb_std.metadata}},
        "tactile": {"tactile_left": {"image": gelsight_std.image_rgb, "metadata": gelsight_std.metadata}},
        "proprioception": {"controller_state": {"tcp_pose": [0.3, 0.1, 0.2, 1, 0, 0, 0], "gripper_width": 0.078}},
    }
    runtime_inputs = preprocessor.observation_window_to_model_inputs(
        [observation],
        model_spec=get_model_spec("vista_so3"),
    )
    with np.load(prepared.output_dir / "train" / "episode_0000.npz") as data:
        assert not np.array_equal(np.rint(runtime_inputs["rgb_wrist"][0] * 255).astype(np.uint8), data["rgb_wrist"][0])
        assert not np.array_equal(np.rint(runtime_inputs["gelsight"][0] * 255).astype(np.uint8), data["gelsight"][0])


def test_runtime_preprocess_forces_gripper_closedness_when_requested(tmp_path: Path) -> None:
    manifests = RuntimeManifests(
        policy={
            "schema_version": "test",
            "model": "vista_so3",
            "obs_horizon": 1,
            "action_horizon": 8,
            "action_dim": 10,
        },
        preprocess1={
            "schema_version": "test",
            "streams": {
                "rgb_wrist": {"preprocess": ImagePreprocessSpec(output_size=(4, 4), crop=CropSpec(mode="none")).to_json()},
                "gelsight": {"preprocess": ImagePreprocessSpec(output_size=(4, 4), crop=CropSpec(mode="none")).to_json()},
            },
        },
        preprocess2={
            "schema_version": "test",
            "preprocess2": {
                "rgb_wrist": ImagePreprocessSpec(output_size=(4, 4), crop=CropSpec(mode="none")).to_json(),
                "gelsight": ImagePreprocessSpec(output_size=(4, 4), crop=CropSpec(mode="none")).to_json(),
            },
        },
        normalizer_stats={},
    )
    preprocessor = RuntimePreprocessor(manifests, gripper_open_width_m=0.078, force_gripper_closedness=True)

    observation = {
        "images": {"wrist": {"image": np.zeros((480, 640, 3), dtype=np.uint8), "metadata": {}}},
        "tactile": {"tactile_left": {"image": np.zeros((480, 640, 3), dtype=np.uint8), "metadata": {}}},
        "proprioception": {"controller_state": {"tcp_pose": [0.3, 0.1, 0.2, 1, 0, 0, 0], "gripper_width": 0.0}},
    }

    model_inputs = preprocessor.observation_window_to_model_inputs([observation], model_spec=get_model_spec("vista_so3"))

    assert model_inputs["qpos"].shape == (1, 10)
    assert model_inputs["qpos"][0, -1] == 1.0


def test_runtime_preprocess_matches_vista_backend_hdf5_first_frame(tmp_path: Path) -> None:
    h5py = _require_h5py()
    cv2 = _require_cv2()
    run_dir = tmp_path / "collect" / "usb_insertion"
    episode_dir = run_dir / "episodes" / "episode_0000"
    _write_fake_episode(episode_dir, offset=0.0, textured_images=True)
    _write_fake_episode(run_dir / "episodes" / "episode_0001", offset=10.0, textured_images=True)
    workspace = WorkspaceSettings(
        recording={
            "collect_root": tmp_path / "collect",
            "preprocess1_root": tmp_path / "preprocess1",
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )
    prepared = prepare_visuotactile_dataset(
        build_prepare_config_from_workspace(
            workspace,
            task_name="usb_insertion",
            model="vista_so3",
            raw_run_dir=run_dir,
            output_dir=tmp_path / "prepared_dataset",
            image_size=8,
            canonical_size=16,
            val_episodes=1,
            overwrite=True,
        )
    )
    export = export_prepared_dataset_for_backend(
        prepared.output_dir,
        tmp_path / "backend",
        model="vista_so3",
        task_name="usb_insertion",
        overwrite=True,
    )

    manifest = json.loads((prepared.output_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    p1_manifest = json.loads(Path(manifest["episodes"][0]["preprocess1_manifest"]).read_text(encoding="utf-8"))
    manifests = RuntimeManifests(
        policy={
            "schema_version": "test",
            "model": "vista_so3",
            "obs_horizon": 2,
            "action_horizon": 8,
            "action_dim": 10,
        },
        preprocess1=p1_manifest,
        preprocess2={
            "schema_version": "test",
            "preprocess2": manifest["preprocess2"],
        },
        normalizer_stats={},
    )
    preprocessor = RuntimePreprocessor(manifests, gripper_open_width_m=0.078)

    rgb_std = standardize_camera_frame(
        _fake_wrist_frame_bgr(0, textured_images=True),
        stream_name="rgb_wrist",
        camera_name="test_wrist",
        source_color="BGR",
    )
    gelsight_std = standardize_camera_frame(
        _fake_gelsight_frame_bgr(0, textured_images=True),
        stream_name="tactile_left",
        camera_name="test_gelsight",
        source_color="BGR",
    )
    observation = {
        "images": {"wrist": {"image": rgb_std.image_rgb, "metadata": rgb_std.metadata}},
        "tactile": {"tactile_left": {"image": gelsight_std.image_rgb, "metadata": gelsight_std.metadata}},
        "proprioception": {"controller_state": {"tcp_pose": [0.3, 0.1, 0.2, 1, 0, 0, 0], "gripper_width": 0.078}},
    }
    runtime_inputs = preprocessor.observation_window_to_model_inputs(
        [observation],
        model_spec=get_model_spec("vista_so3"),
    )

    with h5py.File(export.hdf5_dir / "0.hdf5", "r") as h5_file:
        expected_rgb = _decode_jpeg_to_rgb(h5_file["observation/wrist/rgb"][0])
        expected_gelsight = _decode_jpeg_to_rgb(h5_file["tactile/left_tactile/rgb_marker"][0])

    actual_rgb = np.rint(runtime_inputs["rgb_wrist"][0] * 255).astype(np.uint8)
    actual_gelsight = np.rint(runtime_inputs["gelsight"][0] * 255).astype(np.uint8)
    assert actual_rgb.shape == expected_rgb.shape
    assert actual_gelsight.shape == expected_gelsight.shape


def test_runtime_preprocess_requires_standardized_rgb_input() -> None:
    spec = ImagePreprocessSpec(output_size=(6, 6), crop=CropSpec(mode="none"))
    manifests = RuntimeManifests(
        policy={
            "schema_version": "test",
            "model": "vista_so3",
            "obs_horizon": 1,
            "action_horizon": 1,
            "action_dim": 10,
        },
        preprocess1={
            "schema_version": "test",
            "streams": {
                "rgb_wrist": {
                    "preprocess": spec.to_json(),
                    "compatibility_transforms": [{"name": "bgr_to_rgb"}, {"name": "preprocess1"}],
                },
                "gelsight": {
                    "preprocess": spec.to_json(),
                    "compatibility_transforms": [{"name": "bgr_to_rgb"}, {"name": "preprocess1"}],
                },
            },
        },
        preprocess2={
            "schema_version": "test",
            "preprocess2": {"rgb_wrist": spec.to_json(), "gelsight": spec.to_json()},
        },
        normalizer_stats={},
    )
    preprocessor = RuntimePreprocessor(manifests, gripper_open_width_m=0.078)

    observation = {
        "images": {"wrist": {"image": np.zeros((12, 20, 3), dtype=np.uint8)}},
        "tactile": {"tactile_left": {"image": np.zeros((12, 20, 3), dtype=np.uint8)}},
        "proprioception": {"controller_state": {"tcp_pose": [0.3, 0.1, 0.2, 1, 0, 0, 0], "gripper_width": 0.078}},
    }
    with pytest.raises(ValueError, match="Expected standardized camera frame shape"):
        preprocessor.observation_window_to_model_inputs([observation], model_spec=get_model_spec("vista_so3"))


def test_backend_export_writes_vendor_hdf5_views(tmp_path: Path) -> None:
    h5py = _require_h5py()
    cv2 = _require_cv2()
    run_dir = tmp_path / "collect" / "usb_insertion"
    _write_fake_episode(run_dir / "episodes" / "episode_0000", offset=0.0)
    _write_fake_episode(run_dir / "episodes" / "episode_0001", offset=10.0)
    workspace = WorkspaceSettings(
        recording={
            "collect_root": tmp_path / "collect",
            "preprocess1_root": tmp_path / "preprocess1",
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )
    prepared = prepare_visuotactile_dataset(
        build_prepare_config_from_workspace(
            workspace,
            task_name="usb_insertion",
            model="dp_manifeel",
            raw_run_dir=run_dir,
            output_dir=tmp_path / "prepared_dataset",
            image_size=8,
            canonical_size=16,
            val_episodes=1,
            overwrite=True,
        )
    )

    result = export_prepared_dataset_for_backend(
        prepared.output_dir,
        tmp_path / "backend",
        model="dp_manifeel",
        task_name="usb_insertion",
    )

    assert result.num_episodes == 2
    assert (result.act_hdf5_dir / "episode_0.hdf5").exists()
    assert (result.act_hdf5_dir / "norm_stats.json").exists()
    with h5py.File(result.hdf5_dir / "0.hdf5", "r") as root:
        jpeg_payload = root["observation/wrist/rgb"][0]
        decoded = cv2.imdecode(jpeg_payload, cv2.IMREAD_COLOR)
        assert decoded is not None
        assert root["observations/images/cam_wrist"].shape == (4, 8, 8, 3)
        assert root["observations/images/tac_left"].shape == (4, 8, 8, 3)
        assert root["action"].shape == (4, 8)
    metadata = json.loads((result.act_hdf5_dir / "norm_stats.json").read_text(encoding="utf-8"))
    assert metadata["camera"] == ["cam_wrist"]
    assert metadata["tactile"] == ["cam_left_tactile", "cam_right_tactile"]
    backend_manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert backend_manifest["format"]["duplicate_single_gelsight_as_lr"] is True


def test_vista_backend_loader_uses_commanded_action_not_next_observed_ee(tmp_path: Path) -> None:
    h5py = _require_h5py()
    cv2 = _require_cv2()
    pytest.importorskip("torch")
    pytest.importorskip("zarr")
    vista_root = Path(__file__).resolve().parents[1] / "src" / "vt_dual_franka_workspace" / "policies" / "VISTA"
    if str(vista_root) not in sys.path:
        sys.path.insert(0, str(vista_root))
    dataset_module = pytest.importorskip("vista.dataset.univtac_replay_image_dataset")
    rotation_module = pytest.importorskip("vista.model.common.rotation_transformer")

    path = tmp_path / "0.hdf5"
    rgb = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    gelsight = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    observed_ee = np.asarray(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [9.0, 9.0, 9.0, 1.0, 0.0, 0.0, 0.0],
            [8.0, 8.0, 8.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    commanded = np.asarray(
        [
            [0.10, 0.20, 0.30, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.11, 0.21, 0.31, 1.0, 0.0, 0.0, 0.0, 1.0],
            [0.12, 0.22, 0.32, 1.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    joint = np.zeros((3, 9), dtype=np.float32)
    joint[:, 7:9] = 0.0

    with h5py.File(path, "w") as root:
        root.create_dataset("action", data=commanded)
        embodiment = root.create_group("embodiment")
        embodiment.create_dataset("ee", data=observed_ee)
        embodiment.create_dataset("joint", data=joint)
        obs = root.create_group("observation")
        wrist = obs.create_group("wrist")
        _write_test_jpeg_vlen(h5py, cv2, wrist, "rgb", rgb)
        tactile = root.create_group("tactile")
        left = tactile.create_group("left_tactile")
        right = tactile.create_group("right_tactile")
        _write_test_jpeg_vlen(h5py, cv2, left, "rgb_marker", gelsight)
        _write_test_jpeg_vlen(h5py, cv2, right, "rgb_marker", gelsight)

    arrays = dataset_module._episode_to_arrays(
        episode_path=path,
        rgb_keys=["robot0_eye_in_hand_image"],
        image_types={"robot0_eye_in_hand_image": "rgb"},
        lowdim_keys=["robot0_eef_pos"],
        rgb_shapes={"robot0_eye_in_hand_image": (3, 8, 8)},
        action_dim=10,
        rotation_transformer=rotation_module.RotationTransformer(from_rep="quaternion", to_rep="rotation_6d"),
    )

    assert arrays["action"].shape == (3, 10)
    np.testing.assert_allclose(arrays["action"][:, :3], commanded[:, :3])
    assert not np.allclose(arrays["action"][0, :3], observed_ee[1, :3])
    np.testing.assert_allclose(arrays["robot0_eef_pos"], observed_ee[:, :3])


def test_train_visuotactile_act_dry_run_builds_config_command(tmp_path: Path) -> None:
    run_dir = tmp_path / "collect" / "usb_insertion"
    _write_fake_episode(run_dir / "episodes" / "episode_0000", offset=0.0)
    _write_fake_episode(run_dir / "episodes" / "episode_0001", offset=10.0)
    workspace = WorkspaceSettings(
        recording={
            "collect_root": tmp_path / "collect",
            "preprocess1_root": tmp_path / "preprocess1",
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )
    prepared = prepare_visuotactile_dataset(
        build_prepare_config_from_workspace(
            workspace,
            task_name="usb_insertion",
            model="act_univtac",
            raw_run_dir=run_dir,
            output_dir=tmp_path / "prepared_dataset",
            image_size=8,
            canonical_size=16,
            val_episodes=1,
            overwrite=True,
        )
    )

    result = train_visuotactile(
        TrainVisuotactileConfig(
            workspace=workspace,
            task_name="usb_insertion",
            model="act_univtac",
            dataset_dir=prepared.output_dir,
            checkpoint_dir=tmp_path / "ckpt",
            dry_run=True,
            epochs=123,
        )
    )

    assert "-m" in result.command
    assert "vt_dual_franka_workspace.policies.ACT.imitate_episodes" in result.command
    assert "--config_path" in result.command
    assert str(tmp_path / "ckpt" / "backend_dataset" / "usb_insertion" / "act_hdf5") in result.command


def test_train_visuotactile_vital_dp_dry_run_uses_vital_encoder_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "collect" / "usb_insertion"
    _write_fake_episode(run_dir / "episodes" / "episode_0000", offset=0.0)
    _write_fake_episode(run_dir / "episodes" / "episode_0001", offset=10.0)
    workspace = WorkspaceSettings(
        recording={
            "collect_root": tmp_path / "collect",
            "preprocess1_root": tmp_path / "preprocess1",
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )
    prepared = prepare_visuotactile_dataset(
        build_prepare_config_from_workspace(
            workspace,
            task_name="usb_insertion",
            model="vital_dp",
            raw_run_dir=run_dir,
            output_dir=tmp_path / "prepared_dataset",
            image_size=8,
            canonical_size=16,
            val_episodes=1,
            overwrite=True,
        )
    )

    result = train_visuotactile(
        TrainVisuotactileConfig(
            workspace=workspace,
            task_name="usb_insertion",
            model="vital_dp",
            dataset_dir=prepared.output_dir,
            checkpoint_dir=tmp_path / "ckpt",
            dry_run=True,
        )
    )

    assert "policy.obs_encoder.vision_backbone_path=" in " ".join(result.command)
    assert "best_vision_encoder.pth" in " ".join(result.command)
    assert "best_gelsight_encoder.pth" in " ".join(result.command)
    assert "shape_meta.obs.robot0_eye_in_hand_image.shape=[3,256,256]" in result.command


def test_remote_train_visuotactile_dry_run_commands(tmp_path: Path) -> None:
    workspace = WorkspaceSettings(
        recording={
            "collect_root": tmp_path / "collect",
            "preprocess1_root": tmp_path / "preprocess1",
            "prepared_root": tmp_path / "prepared",
            "checkpoints_root": tmp_path / "checkpoints",
        }
    )
    dataset_dir = tmp_path / "prepared" / "usb_insertion" / "visuotactile" / "real_canonical_v1" / "vista_so2"
    checkpoint_dir = tmp_path / "checkpoints" / "run"

    result = remote_train_visuotactile(
        RemoteTrainConfig(
            local_train=TrainVisuotactileConfig(
                workspace=workspace,
                task_name="usb_insertion",
                model="vista_so2",
                dataset_dir=dataset_dir,
                checkpoint_dir=checkpoint_dir,
                dry_run=True,
            ),
            remote="trainer@example",
            remote_root="/remote/robot_workspace",
            dry_run=True,
        )
    )

    rendered = [" ".join(command) for command in result.commands]
    assert any("rsync" in command and "src/" not in command for command in rendered)
    assert any("python -m vt_dual_franka_workspace.policies.visuotactile.train" in command for command in rendered)
    assert any("--backend-dataset-root" in command for command in rendered)
    assert any("--no-prepare" in command for command in rendered)
    assert result.remote_checkpoint_dir.endswith("/external/run")


def test_remote_train_visuotactile_defaults_to_common_dataset_and_remote_prepare(tmp_path: Path) -> None:
    workspace_root = tmp_path / "vt_franka"
    data_root = workspace_root / "robot_workspace" / "data"
    workspace_root.mkdir(parents=True)
    workspace = WorkspaceSettings(
        recording={
            "collect_root": data_root / "collect",
            "preprocess1_root": data_root / "preprocess1",
            "prepared_root": data_root / "prepared",
            "checkpoints_root": data_root / "checkpoints",
        }
    )

    from vt_dual_franka_workspace.policies.visuotactile import remote as remote_module

    previous_workspace_root = remote_module.WORKSPACE_ROOT
    remote_module.WORKSPACE_ROOT = workspace_root
    try:
        result = remote_train_visuotactile(
            RemoteTrainConfig(
                local_train=TrainVisuotactileConfig(
                    workspace=workspace,
                    task_name="erasing",
                    model="vista_so3",
                    dataset_name="real_erasing",
                    dry_run=True,
                ),
                remote="trainer@example",
                remote_root="/remote/vt_franka",
                dry_run=True,
            )
        )
    finally:
        remote_module.WORKSPACE_ROOT = previous_workspace_root

    rendered = "\n".join(" ".join(command) for command in result.commands)
    assert "/remote/vt_franka/robot_workspace/data/datasets/erasing/real_erasing" in rendered
    assert "--dataset-dir /remote/vt_franka/robot_workspace/data/datasets/erasing/real_erasing" in rendered
    assert "--no-prepare" not in rendered
    assert "PYTHONPATH=robot_workspace/src:shared/src:${PYTHONPATH:-}" in rendered
    assert "robot_workspace/config/workspace.yaml" in rendered


def test_remote_train_visuotactile_keeps_workspace_relative_symlink_paths(tmp_path: Path, monkeypatch: Any) -> None:
    workspace_root = tmp_path / "robot_workspace"
    backing_root = tmp_path / "ssd" / "vt_franka" / "data"
    workspace_root.mkdir()
    backing_root.mkdir(parents=True)
    data_link = workspace_root / "data"
    data_link.symlink_to(backing_root, target_is_directory=True)
    monkeypatch.setattr(
        "vt_dual_franka_workspace.policies.visuotactile.remote.WORKSPACE_ROOT",
        workspace_root,
    )
    workspace = WorkspaceSettings(
        recording={
            "collect_root": data_link / "collect",
            "preprocess1_root": data_link / "preprocess1",
            "prepared_root": data_link / "prepared",
            "checkpoints_root": data_link / "checkpoints",
        }
    )
    dataset_dir = data_link / "prepared" / "usb_insertion" / "visuotactile" / "real_canonical_v1" / "vista_so2"
    checkpoint_dir = data_link / "checkpoints" / "usb_insertion" / "visuotactile" / "vista_so2" / "run"

    result = remote_train_visuotactile(
        RemoteTrainConfig(
            local_train=TrainVisuotactileConfig(
                workspace=workspace,
                task_name="usb_insertion",
                model="vista_so2",
                dataset_dir=dataset_dir,
                checkpoint_dir=checkpoint_dir,
                dry_run=True,
            ),
            remote="trainer@example",
            remote_root="/remote/robot_workspace",
            dry_run=True,
        )
    )

    rendered = "\n".join(" ".join(command) for command in result.commands)
    assert "/remote/robot_workspace/data/prepared/usb_insertion/visuotactile/real_canonical_v1/vista_so2" in rendered
    assert "/remote/robot_workspace/data/checkpoints/usb_insertion/visuotactile/vista_so2/run" in rendered
    assert "/remote/robot_workspace/external" not in rendered
    assert result.remote_checkpoint_dir == "/remote/robot_workspace/data/checkpoints/usb_insertion/visuotactile/vista_so2/run"


def test_remote_scheduler_dry_run_expands_plan(tmp_path: Path) -> None:
    plan = tmp_path / "plan.txt"
    plan.write_text("usb_insertion\ndp_manifeel vista_so2\n", encoding="utf-8")
    output = subprocess.check_output(
        [
            "python",
            "remote_pc/train_scheduler.py",
            "--plan",
            str(plan),
            "--remote-root",
            "/remote/robot_workspace",
            "--gpus",
            "0,1",
            "--batch-size",
            "8",
            "--epochs",
            "1",
            "--dry-run",
        ],
        text=True,
    )
    payload = json.loads(output)
    assert payload["remote_root"] == "/remote/robot_workspace"
    assert len(payload["jobs"]) == 2
    assert payload["jobs"][0]["run_name"] == "usb_insertion_dp_manifeel"


def _write_fake_episode(episode_dir: Path, *, offset: float, textured_images: bool = False) -> None:
    cv2 = _require_cv2()
    streams = episode_dir / "streams"
    rgb_dir = streams / "rgb_wrist"
    gel_dir = streams / "gelsight_frames"
    rgb_dir.mkdir(parents=True)
    gel_dir.mkdir(parents=True)
    timestamps = np.asarray([offset + i * 0.1 for i in range(4)], dtype=np.float64)

    rgb_paths = []
    gel_paths = []
    gel_indices = []
    gel_frames = []
    for i, timestamp in enumerate(timestamps):
        rgb = _fake_wrist_frame_bgr(i, textured_images=textured_images)
        rgb_path = rgb_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(rgb_path), rgb)
        rgb_paths.append(rgb_path.relative_to(episode_dir).as_posix())
        gel_frame = _fake_gelsight_frame_bgr(i, textured_images=textured_images)
        gel_frames.append(gel_frame)
        gel_paths.append("streams/gelsight_frames/chunk_000000.npz")
        gel_indices.append(i)
    np.savez(gel_dir / "chunk_000000.npz", frames=np.stack(gel_frames, axis=0))

    robot_pose = np.asarray([[0.3 + i * 0.01, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0] for i in range(4)], dtype=np.float64)
    action_pose = np.asarray([[0.31 + i * 0.01, 0.1, 0.2, 1.0, 0.0, 0.0, 0.0] for i in range(4)], dtype=np.float64)
    np.savez_compressed(
        episode_dir / "aligned_episode.npz",
        timestamps=timestamps,
        robot_tcp_pose=robot_pose,
        teleop_target_tcp=action_pose,
        gripper_width=np.asarray([0.078, 0.04, 0.02, 0.0], dtype=np.float64),
        teleop_gripper_closed=np.asarray([False, False, True, True], dtype=bool),
        rgb_wrist_frame_paths=np.asarray(rgb_paths, dtype=object),
        gelsight_frame_paths=np.asarray(gel_paths, dtype=object),
        gelsight_frame_indices=np.asarray(gel_indices, dtype=np.int64),
    )
    (episode_dir / "episode_manifest.json").write_text(
        json.dumps({"episode_id": episode_dir.name, "outcome": "saved", "started_at_wall_time": float(offset)}),
        encoding="utf-8",
    )


def _write_fake_raw_stream_episode(episode_dir: Path, *, textured_images: bool = False) -> None:
    cv2 = _require_cv2()
    streams = episode_dir / "streams"
    rgb_dir = streams / "rgb_wrist"
    gel_dir = streams / "gelsight_frames"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    gel_dir.mkdir(parents=True, exist_ok=True)
    rgb_lines = []
    gel_lines = []
    gel_frames = []
    for i in range(4):
        timestamp = float(i * 0.1)
        rgb = _fake_wrist_frame_bgr(i, textured_images=textured_images)
        rgb_path = rgb_dir / f"frame_{i:06d}.jpg"
        cv2.imwrite(str(rgb_path), rgb)
        rgb_lines.append(
            json.dumps(
                {
                    "frame_path": rgb_path.relative_to(episode_dir).as_posix(),
                    "captured_wall_time": timestamp,
                    "sequence_id": i,
                }
            )
        )
        gel_frames.append(_fake_gelsight_frame_bgr(i, textured_images=textured_images))
        gel_lines.append(
            json.dumps(
                {
                    "frame_path": "streams/gelsight_frames/chunk_000000.npz",
                    "chunk_path": "streams/gelsight_frames/chunk_000000.npz",
                    "index_in_chunk": i,
                    "captured_wall_time": timestamp,
                    "sequence_id": i,
                }
            )
        )
    np.savez(gel_dir / "chunk_000000.npz", frames=np.stack(gel_frames, axis=0))
    (streams / "rgb_wrist.jsonl").write_text("\n".join(rgb_lines) + "\n", encoding="utf-8")
    (streams / "gelsight_frames.jsonl").write_text("\n".join(gel_lines) + "\n", encoding="utf-8")
    controller_lines = []
    teleop_lines = []
    for i in range(6):
        timestamp = float(i * 0.1)
        controller_lines.append(
            json.dumps(
                {
                    "received_wall_time": timestamp,
                    "state": {
                        "tcp_pose": [0.3 + i * 0.01, 0.1, 0.2, 1, 0, 0, 0],
                        "tcp_velocity": [0.0] * 6,
                        "tcp_wrench": [0.0] * 6,
                        "joint_positions": [0.0] * 7,
                        "joint_velocities": [0.0] * 7,
                        "gripper_width": 0.078,
                        "gripper_force": 0.0,
                    },
                }
            )
        )
        teleop_lines.append(
            json.dumps(
                {
                    "source_wall_time": timestamp + 0.05,
                    "target_tcp": [0.31 + i * 0.01, 0.1, 0.2, 1, 0, 0, 0],
                    "gripper_closed": False,
                }
            )
        )
    (streams / "controller_state.jsonl").write_text("\n".join(controller_lines) + "\n", encoding="utf-8")
    (streams / "teleop_commands.jsonl").write_text("\n".join(teleop_lines) + "\n", encoding="utf-8")
    (episode_dir / "episode_manifest.json").write_text(
        json.dumps(
            {
                "episode_id": episode_dir.name,
                "outcome": "saved",
                "started_at_wall_time": 0.0,
                "stopped_at_wall_time": 0.5,
            }
        ),
        encoding="utf-8",
    )
    align_episode(episode_dir, overwrite=True)


class _StaticEpisodeSession:
    def __init__(self, episode_dir: Path) -> None:
        self.episode_dir = episode_dir

    def get_active_episode_dir(self) -> Path:
        return self.episode_dir


def _fake_wrist_frame_bgr(index: int, *, textured_images: bool) -> np.ndarray:
    if not textured_images:
        return np.full((12, 20, 3), [10 + index, 20, 30], dtype=np.uint8)
    yy, xx = np.indices((12, 20))
    return np.stack(
        [
            (xx * 11 + yy * 3 + index * 7) % 256,
            (xx * 5 + yy * 17 + 20 + index * 9) % 256,
            (xx * 19 + yy * 2 + 30 + index * 13) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def _fake_gelsight_frame_bgr(index: int, *, textured_images: bool) -> np.ndarray:
    if not textured_images:
        return np.full((18, 26, 3), [40, 50 + index, 60], dtype=np.uint8)
    yy, xx = np.indices((18, 26))
    return np.stack(
        [
            (xx * 13 + yy * 4 + 40 + index * 5) % 256,
            (xx * 7 + yy * 9 + 50 + index * 11) % 256,
            (xx * 3 + yy * 23 + 60 + index * 17) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)


def _decode_jpeg_to_rgb(jpeg_buffer) -> np.ndarray:
    cv2 = _require_cv2()
    image_bgr = cv2.imdecode(np.asarray(jpeg_buffer, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("cv2.imdecode returned None")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _write_test_jpeg_vlen(h5py, cv2, group, name: str, images: np.ndarray) -> None:
    dtype = h5py.vlen_dtype(np.dtype("uint8"))
    dataset = group.create_dataset(name, shape=(len(images),), dtype=dtype)
    for index, image in enumerate(np.asarray(images, dtype=np.uint8)):
        ok, payload = cv2.imencode(".jpg", image[:, :, ::-1])
        if not ok:
            raise RuntimeError("failed to encode test JPEG")
        dataset[index] = np.asarray(payload, dtype=np.uint8)


def _require_cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("OpenCV is required for this test") from exc
    return cv2


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("h5py is required for this test") from exc
    return h5py
