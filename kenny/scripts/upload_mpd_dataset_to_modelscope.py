"""Resumable ModelScope uploader for MPD-style datasets.

Usage:
    export MODELSCOPE_TOKEN="your-token"
    python kenny/scripts/upload_mpd_dataset_to_modelscope.py \
        --repo-id kenn3o3/put_cup_on_plate \
        --local-dir robot_workspace/data/mpd/put_cup_on_plate
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from modelscope.hub.api import HubApi


DEFAULT_PROGRESS_FILE = ".modelscope_upload_progress.json"


@dataclass(frozen=True)
class UploadChunk:
    kind: str
    local_path: Path
    path_in_repo: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an MPD dataset to ModelScope with resumable chunks."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target ModelScope repo id, e.g. kenn3o3/put_cup_on_plate",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        required=True,
        help="Local MPD dataset directory",
    )
    parser.add_argument(
        "--visibility",
        default="private",
        choices=["public", "private", "internal"],
        help="Visibility used only when the dataset repo is created",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("MODELSCOPE_TOKEN"),
        help="ModelScope token. Defaults to MODELSCOPE_TOKEN",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=None,
        help=(
            "Path to the local resume file. Defaults to "
            "<local-dir>/.modelscope_upload_progress.json"
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Concurrent uploads inside each chunk. Lower values are more stable.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=6,
        help="Retries per chunk before failing",
    )
    parser.add_argument(
        "--backoff-seconds",
        type=float,
        default=5.0,
        help="Initial retry backoff in seconds",
    )
    parser.add_argument(
        "--socket-timeout",
        type=float,
        default=300.0,
        help="Default network socket timeout in seconds. Use 0 to disable.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned chunks without uploading",
    )
    return parser.parse_args()


def build_chunks(local_dir: Path) -> list[UploadChunk]:
    if not local_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {local_dir}")

    chunks: list[UploadChunk] = []

    for path in sorted(local_dir.iterdir()):
        if path.is_file() and path.name != DEFAULT_PROGRESS_FILE:
            chunks.append(
                UploadChunk(
                    kind="file",
                    local_path=path,
                    path_in_repo=path.name,
                )
            )

    for split in ("train", "val"):
        split_dir = local_dir / split
        if not split_dir.exists():
            continue
        for demo_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            chunks.append(
                UploadChunk(
                    kind="folder",
                    local_path=demo_dir,
                    path_in_repo=f"{split}/{demo_dir.name}",
                )
            )

    if not chunks:
        raise ValueError(f"No uploadable files found in {local_dir}")

    return chunks


def load_progress(progress_file: Path) -> set[str]:
    if not progress_file.exists():
        return set()
    payload = json.loads(progress_file.read_text())
    completed = payload.get("completed", [])
    if not isinstance(completed, list):
        raise ValueError(f"Invalid progress file format: {progress_file}")
    return {str(item) for item in completed}


def save_progress(
    progress_file: Path,
    repo_id: str,
    local_dir: Path,
    completed: Iterable[str],
) -> None:
    payload = {
        "repo_id": repo_id,
        "local_dir": str(local_dir),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "completed": sorted(set(completed)),
    }
    progress_file.write_text(json.dumps(payload, indent=2) + "\n")


def upload_chunk(
    api: HubApi,
    chunk: UploadChunk,
    repo_id: str,
    token: str,
    max_workers: int,
) -> None:
    if chunk.kind == "file":
        api.upload_file(
            repo_id=repo_id,
            path_or_fileobj=str(chunk.local_path),
            path_in_repo=chunk.path_in_repo,
            repo_type="dataset",
            commit_message=f"Upload {chunk.path_in_repo}",
            token=token,
        )
        return

    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(chunk.local_path),
        path_in_repo=chunk.path_in_repo,
        repo_type="dataset",
        commit_message=f"Upload {chunk.path_in_repo}",
        commit_description="Resumable MPD dataset upload",
        token=token,
        max_workers=max_workers,
    )


def main() -> None:
    args = parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    if not args.token:
        raise ValueError("ModelScope token missing. Set MODELSCOPE_TOKEN or pass --token.")

    if args.socket_timeout > 0:
        socket.setdefaulttimeout(args.socket_timeout)

    local_dir = args.local_dir.expanduser().resolve()
    progress_file = (
        args.progress_file.expanduser().resolve()
        if args.progress_file is not None
        else local_dir / DEFAULT_PROGRESS_FILE
    )

    chunks = build_chunks(local_dir)
    completed = load_progress(progress_file)

    print(f"Repo: {args.repo_id}")
    print(f"Local dir: {local_dir}")
    print(f"Progress file: {progress_file}")
    print(f"Chunks: {len(chunks)} total, {len(completed)} already completed")

    if args.dry_run:
        for chunk in chunks:
            marker = "done" if chunk.path_in_repo in completed else "todo"
            print(f"[{marker}] {chunk.kind:6s} {chunk.path_in_repo}")
        return

    api = HubApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        visibility=args.visibility,
        exist_ok=True,
        token=args.token,
    )

    for index, chunk in enumerate(chunks, start=1):
        if chunk.path_in_repo in completed:
            print(f"[{index}/{len(chunks)}] skip {chunk.path_in_repo}")
            continue

        print(f"[{index}/{len(chunks)}] upload {chunk.path_in_repo}")
        for attempt in range(1, args.retries + 1):
            try:
                upload_chunk(
                    api=api,
                    chunk=chunk,
                    repo_id=args.repo_id,
                    token=args.token,
                    max_workers=args.max_workers,
                )
                completed.add(chunk.path_in_repo)
                save_progress(progress_file, args.repo_id, local_dir, completed)
                print(f"[{index}/{len(chunks)}] done {chunk.path_in_repo}")
                break
            except Exception as exc:
                if attempt == args.retries:
                    print(
                        f"[{index}/{len(chunks)}] failed {chunk.path_in_repo} "
                        f"after {attempt} attempts: {exc}"
                    )
                    raise
                sleep_s = args.backoff_seconds * (2 ** (attempt - 1))
                print(
                    f"[{index}/{len(chunks)}] retry {attempt}/{args.retries} "
                    f"for {chunk.path_in_repo}: {exc}"
                )
                print(f"Sleeping {sleep_s:.1f}s before retry...")
                time.sleep(sleep_s)

    print(f"Upload complete: https://www.modelscope.cn/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
