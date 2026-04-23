from __future__ import annotations

from vt_franka_shared.models import ResetCommand

from .settings import ResetProfileSettings, WorkspaceSettings


def build_reset_command(
    settings: WorkspaceSettings,
    *,
    source: str,
    profile_name: str | None = None,
) -> ResetCommand:
    resolved_profile_name = profile_name or settings.reset.default_profile
    profile = settings.reset.profiles.get(resolved_profile_name)
    if profile is None:
        raise KeyError(f"Reset profile is not configured: {resolved_profile_name}")
    return _command_from_profile(resolved_profile_name, profile, settings=settings, source=source)


def _command_from_profile(
    profile_name: str,
    profile: ResetProfileSettings,
    *,
    settings: WorkspaceSettings,
    source: str,
) -> ResetCommand:
    gripper_width = profile.gripper_width
    if gripper_width is None and profile.gripper_target == "open":
        gripper_width = settings.teleop.max_gripper_width
    gripper_velocity = profile.gripper_velocity
    if gripper_velocity is None and profile.gripper_target != "unchanged":
        gripper_velocity = settings.teleop.gripper_velocity
    gripper_force_limit = profile.gripper_force_limit
    if gripper_force_limit is None and profile.gripper_target != "unchanged":
        gripper_force_limit = settings.teleop.grasp_force

    return ResetCommand(
        profile=profile_name,
        joint_positions=None if profile.joint_positions is None else list(profile.joint_positions),
        joint_duration_sec=profile.joint_duration_sec,
        eef_pose_xyz_rpy_deg=None if profile.eef_pose_xyz_rpy_deg is None else list(profile.eef_pose_xyz_rpy_deg),
        eef_duration_sec=profile.eef_duration_sec,
        gripper_target=profile.gripper_target,
        gripper_width=gripper_width,
        gripper_velocity=gripper_velocity,
        gripper_force_limit=gripper_force_limit,
        source=source,
    )
