#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
from pathlib import Path

from modelscope.hub.snapshot_download import dataset_snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VT_Franka inference checkpoint bundles from ModelScope.")
    parser.add_argument("--repo-id", default=os.environ.get("MODELSCOPE_REPO_ID"), required=os.environ.get("MODELSCOPE_REPO_ID") is None)
    parser.add_argument("--token", default=os.environ.get("MODELSCOPE_TOKEN"))
    parser.add_argument("--revision", default=os.environ.get("MODELSCOPE_REVISION", "master"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--bundle-id", default=os.environ.get("MODELSCOPE_BUNDLE_ID"))
    parser.add_argument("--path-in-repo", default="checkpoints")
    parser.add_argument("--dst-root", type=Path, default=Path("robot_workspace/data/checkpoints"))
    parser.add_argument("--local-dir", type=Path, default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    base = args.path_in_repo.strip("/")
    if args.bundle_id:
        base = f"{base}/{args.bundle_id.strip('/')}"
    allow_patterns = [f"{base}/{args.task}/{model}/**" for model in args.models]
    download_root = args.local_dir.expanduser().resolve() if args.local_dir else None

    with tempfile.TemporaryDirectory(prefix="vt_franka_ms_dl_") as tmp:
        local_dir = str(download_root or Path(tmp))
        snapshot = Path(
            dataset_snapshot_download(
                dataset_id=args.repo_id,
                revision=args.revision,
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                max_workers=args.max_workers,
                token=args.token,
            )
        )
        src_task = snapshot / base / args.task
        dst_task = args.dst_root.expanduser().resolve() / args.task
        if not src_task.is_dir():
            raise FileNotFoundError(f"downloaded snapshot is missing {src_task}")
        for model in args.models:
            src_model = src_task / model
            dst_model = dst_task / model
            if not src_model.is_dir():
                raise FileNotFoundError(f"downloaded snapshot is missing model {model}: {src_model}")
            if dst_model.exists():
                shutil.rmtree(dst_model)
            dst_model.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_model, dst_model)
            if not _checkpoint_files(dst_model):
                raise FileNotFoundError(f"downloaded model has no checkpoint artifacts: {dst_model}")
            print(f"downloaded {model} -> {dst_model}")
    print(f"downloaded checkpoint bundles from {args.repo_id}:{base}/{args.task}")


def _checkpoint_files(model_dir: Path) -> list[Path]:
    candidates = [
        model_dir / "checkpoints" / "best.ckpt",
        model_dir / "best.ckpt",
        model_dir / "policy_best.ckpt",
        model_dir / "checkpoints" / "latest.ckpt",
    ]
    checkpoints_dir = model_dir / "checkpoints"
    if checkpoints_dir.is_dir():
        candidates.extend(sorted(checkpoints_dir.glob("epoch=*.ckpt")))
    return [path for path in candidates if path.is_file()]


if __name__ == "__main__":
    main()
