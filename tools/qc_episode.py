#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
ROBOT_SRC = REPO_ROOT / "robot_workspace" / "src"
if str(ROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(ROBOT_SRC))

try:
    from episode_alignment import collect_episode_dirs
except ModuleNotFoundError:
    from tools.episode_alignment import collect_episode_dirs

from vt_franka_workspace.recording import analyze_episode_quality, format_episode_qc_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute QC metrics for VT Franka raw episode streams.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Episode directories, a run directory containing episodes/, or an episodes/ directory.",
    )
    parser.add_argument("--no-write", action="store_true", help="Print QC without writing episode_qc.json.")
    args = parser.parse_args()

    episode_dirs = collect_episode_dirs([Path(path) for path in args.paths])
    if not episode_dirs:
        raise SystemExit("No episode directories found.")

    failures: list[tuple[Path, Exception]] = []
    for episode_dir in episode_dirs:
        try:
            report = analyze_episode_quality(episode_dir, write=not args.no_write)
            print(f"[ok] {episode_dir}: {format_episode_qc_summary(report)}")
        except Exception as exc:
            failures.append((episode_dir, exc))
            print(f"[failed] {episode_dir}: {exc}")

    if failures:
        raise SystemExit(f"Failed to QC {len(failures)} / {len(episode_dirs)} episodes.")


if __name__ == "__main__":
    main()
