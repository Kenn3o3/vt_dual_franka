#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from episode_alignment import align_episode, collect_episode_dirs
except ModuleNotFoundError:
    from tools.episode_alignment import align_episode, collect_episode_dirs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate aligned_episode.npz from raw VT Franka episode streams.")
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
            print(f"[ok] {episode_dir} -> {output_path}")
        except Exception as exc:
            failures.append((episode_dir, exc))
            print(f"[failed] {episode_dir}: {exc}")

    if failures:
        raise SystemExit(f"Failed to align {len(failures)} / {len(episode_dirs)} episodes.")


if __name__ == "__main__":
    main()
