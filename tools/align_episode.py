#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from episode_alignment import align_episode, collect_episode_dirs
except ModuleNotFoundError:
    from tools.episode_alignment import align_episode, collect_episode_dirs

try:
    from vt_dual_franka_workspace.policies.visuotactile.canonical import CanonicalPreprocessConfig, preprocess_aligned_episode_images
    from vt_dual_franka_workspace.policies.visuotactile.config import DEFAULT_PREPROCESS1_PROFILE
    from vt_dual_franka_workspace.policies.visuotactile.image_preprocess import parse_crop_box
except ModuleNotFoundError:
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root / "robot_workspace" / "src"))
    sys.path.insert(0, str(repo_root / "shared" / "src"))
    from vt_dual_franka_workspace.policies.visuotactile.canonical import CanonicalPreprocessConfig, preprocess_aligned_episode_images
    from vt_dual_franka_workspace.policies.visuotactile.config import DEFAULT_PREPROCESS1_PROFILE
    from vt_dual_franka_workspace.policies.visuotactile.image_preprocess import parse_crop_box


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate aligned_episode.npz from raw VT Dual Franka episode streams.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Episode directories, a run directory containing episodes/, or an episodes/ directory.",
    )
    parser.add_argument("--hz", type=float, default=10.0, help="Aligned grid frequency in Hz.")
    parser.add_argument(
        "--max-action-lead-sec",
        type=float,
        default=None,
        help="Maximum allowed future teleop command lead. Defaults to one aligned step.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing aligned_episode.npz files.")
    parser.add_argument("--preprocess-images", action="store_true", help="Also build preprocess1 canonical image chunks.")
    parser.add_argument("--preprocess-profile", default=DEFAULT_PREPROCESS1_PROFILE)
    parser.add_argument("--canonical-size", type=int, default=480)
    parser.add_argument("--gelsight-crop-box", default=None, help="Optional x0,y0,x1,y1 GelSight crop before canonical resize.")
    parser.add_argument("--preprocess-output-root", type=Path, default=None)
    parser.add_argument("--task-name", default=None, help="Task name to record in centralized preprocess1 manifests.")
    parser.add_argument("--gelsight-margin-fraction", type=float, default=0.0)
    args = parser.parse_args()

    episode_dirs = collect_episode_dirs([Path(path) for path in args.paths])
    if not episode_dirs:
        raise SystemExit("No episode directories found.")

    failures: list[tuple[Path, Exception]] = []
    for episode_dir in episode_dirs:
        try:
            output_path = align_episode(
                episode_dir,
                target_hz=args.hz,
                max_action_lead_sec=args.max_action_lead_sec,
                overwrite=args.overwrite,
            )
            message = f"[ok] {episode_dir} -> {output_path}"
            if args.preprocess_images:
                preprocess_result = preprocess_aligned_episode_images(
                    episode_dir,
                    CanonicalPreprocessConfig(
                        profile_name=args.preprocess_profile,
                        canonical_size=args.canonical_size,
                        overwrite=args.overwrite,
                        output_root=args.preprocess_output_root,
                        task_name=args.task_name,
                        gelsight_crop_box=parse_crop_box(args.gelsight_crop_box),
                        gelsight_margin_fraction=args.gelsight_margin_fraction,
                    ),
                )
                message += (
                    f" | preprocess1={preprocess_result.output_dir} "
                    f"kept={preprocess_result.kept_steps} dropped={preprocess_result.dropped_steps}"
                )
            print(message)
        except Exception as exc:
            failures.append((episode_dir, exc))
            print(f"[failed] {episode_dir}: {exc}")

    if failures:
        raise SystemExit(f"Failed to align {len(failures)} / {len(episode_dirs)} episodes.")


if __name__ == "__main__":
    main()
