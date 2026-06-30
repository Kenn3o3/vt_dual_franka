#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from modelscope.hub.api import HubApi


EXCLUDE_NAMES = {
    "backend_dataset",
    "wandb",
    "__pycache__",
    ".hydra",
}
EXCLUDE_SUFFIXES = (
    ".hdf5",
    ".zarr",
    ".zarr.zip",
    ".mp4",
    ".pyc",
)


def _stage_model(src_root: Path, stage_root: Path, task: str, model: str) -> None:
    run_dir = _find_model_checkpoint_dir(src_root, task, model)
    if not run_dir.is_dir():
        raise FileNotFoundError(f"checkpoint run directory not found: {run_dir}")

    dst_dir = stage_root / task / model
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    shutil.copytree(run_dir, dst_dir, ignore=_ignore_inference_bundle_files)

    if not _checkpoint_files(dst_dir):
        raise FileNotFoundError(f"no checkpoint artifacts found under {run_dir}")


def _ignore_inference_bundle_files(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        path = Path(directory) / name
        if name in EXCLUDE_NAMES or name.startswith(".tmp."):
            ignored.add(name)
        elif path.is_file() and (name.endswith(EXCLUDE_SUFFIXES) or name.endswith(".lock")):
            ignored.add(name)
    return ignored


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


def _find_model_checkpoint_dir(src_root: Path, task: str, model: str) -> Path:
    candidates = [
        src_root / task / model,
        src_root / task / "visuotactile" / model,
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload VT_Franka inference checkpoint bundles to ModelScope.")
    parser.add_argument("--repo-id", default=os.environ.get("MODELSCOPE_REPO_ID"), required=os.environ.get("MODELSCOPE_REPO_ID") is None)
    parser.add_argument("--token", default=os.environ.get("MODELSCOPE_TOKEN"))
    parser.add_argument("--revision", default=os.environ.get("MODELSCOPE_REVISION", "master"))
    parser.add_argument("--repo-type", default=os.environ.get("MODELSCOPE_REPO_TYPE", "dataset"))
    parser.add_argument("--src-root", type=Path, default=Path("robot_workspace/data/checkpoints"))
    parser.add_argument("--task", required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--bundle-id", default=os.environ.get("MODELSCOPE_BUNDLE_ID"))
    parser.add_argument("--path-in-repo", default="checkpoints")
    parser.add_argument("--visibility", default="private")
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()

    src_root = args.src_root.expanduser().resolve()
    with tempfile.TemporaryDirectory(prefix="vt_franka_ckpts_") as tmp:
        stage_root = Path(tmp) / "checkpoints"
        for model in args.models:
            _stage_model(src_root, stage_root, args.task, model)
        if args.bundle_id:
            bundle_manifest = {
            "schema_version": "vt_franka_modelscope_checkpoint_bundle_v2",
                "bundle_id": args.bundle_id,
                "task": args.task,
                "models": args.models,
                "src_root": str(src_root),
                "created_at_wall_time": time.time(),
            }
            manifest_path = stage_root / args.task / "bundle_manifest.json"
            manifest_path.write_text(json.dumps(bundle_manifest, indent=2) + "\n", encoding="utf-8")

        api = HubApi()
        if args.token:
            api.login(args.token)
        if not api.repo_exists(args.repo_id, repo_type=args.repo_type, token=args.token):
            api.create_repo(
                args.repo_id,
                repo_type=args.repo_type,
                token=args.token,
                visibility=args.visibility,
                exist_ok=True,
            )
        path_in_repo = args.path_in_repo.strip("/")
        if args.bundle_id:
            path_in_repo = f"{path_in_repo}/{args.bundle_id.strip('/')}"
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            folder_path=stage_root,
            path_in_repo=path_in_repo,
            token=args.token,
            revision=args.revision,
            max_workers=args.max_workers,
            commit_message=f"upload checkpoint bundles for {args.task}: {','.join(args.models)}",
        )
    print(f"uploaded checkpoint bundles to {args.repo_id}:{path_in_repo}/{args.task}")


if __name__ == "__main__":
    main()
