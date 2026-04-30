from __future__ import annotations

import argparse
from pathlib import Path

from ...config import load_workspace_config
from .config import DEFAULT_DATASET_NAME
from .data import build_prepare_config_from_workspace, prepare_mpd_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare VT Franka raw episodes for MPD-family training")
    parser.add_argument("--workspace-config", default="robot_workspace/config/workspace.yaml")
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--raw-run-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-episodes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    workspace = load_workspace_config(args.workspace_config)
    config = build_prepare_config_from_workspace(
        workspace,
        task_name=args.task_name,
        raw_run_dir=args.raw_run_dir,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        target_hz=args.target_hz,
        val_ratio=args.val_ratio,
        val_episodes=args.val_episodes,
        overwrite=args.overwrite,
    )
    result = prepare_mpd_dataset(config)
    print(f"Prepared MPD dataset: {result.output_dir}")
    print(f"Train episodes: {result.train_episodes}")
    print(f"Val episodes: {result.val_episodes}")
    print(f"Total steps: {result.total_steps}")
    print(f"Manifest: {result.manifest_path}")


if __name__ == "__main__":
    main()
