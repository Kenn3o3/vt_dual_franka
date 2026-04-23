#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." && pwd)

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 LOCAL_PATH [PATH_IN_REPO]" >&2
    echo "Example: $0 robot_workspace/data/mpd/put_cup_on_plate put_cup_on_plate" >&2
    echo "Example: $0 robot_workspace/data/mpd/put_cup_on_plate /" >&2
    exit 1
fi

LOCAL_ARG=$1

if [[ -e "$LOCAL_ARG" ]]; then
    LOCAL_PATH=$(cd -- "$(dirname -- "$LOCAL_ARG")" && pwd)/$(basename -- "$LOCAL_ARG")
elif [[ -e "$REPO_ROOT/$LOCAL_ARG" ]]; then
    LOCAL_PATH="$REPO_ROOT/$LOCAL_ARG"
else
    echo "Local path not found: $LOCAL_ARG" >&2
    echo "Also tried repository path: $REPO_ROOT/$LOCAL_ARG" >&2
    exit 1
fi

DEFAULT_REPO_PATH=$(basename -- "$LOCAL_PATH")
PATH_IN_REPO=${2:-$DEFAULT_REPO_PATH}
PATH_IN_REPO=${PATH_IN_REPO#/}
if [[ "$PATH_IN_REPO" == "." ]]; then
    PATH_IN_REPO=""
fi

MODELSCOPE_REPO_ID=${MODELSCOPE_REPO_ID:-}
MODELSCOPE_REPO_TYPE=${MODELSCOPE_REPO_TYPE:-dataset}
MODELSCOPE_TOKEN=${MODELSCOPE_TOKEN:-}
MODELSCOPE_REVISION=${MODELSCOPE_REVISION:-master}
MODELSCOPE_MAX_WORKERS=${MODELSCOPE_MAX_WORKERS:-4}
MODELSCOPE_DRY_RUN=${MODELSCOPE_DRY_RUN:-0}
MODELSCOPE_CREATE_REPO=${MODELSCOPE_CREATE_REPO:-1}
MODELSCOPE_VISIBILITY=${MODELSCOPE_VISIBILITY:-private}
MODELSCOPE_COMMIT_MESSAGE=${MODELSCOPE_COMMIT_MESSAGE:-"upload ${PATH_IN_REPO}"}
MODELSCOPE_IGNORE_PATTERNS=${MODELSCOPE_IGNORE_PATTERNS:-}
MODELSCOPE_BATCH_SIZE=${MODELSCOPE_BATCH_SIZE:-64}
MODELSCOPE_RESUME=${MODELSCOPE_RESUME:-1}
MODELSCOPE_RETRY_TIMES=${MODELSCOPE_RETRY_TIMES:-5}
MODELSCOPE_RETRY_DELAY_SECONDS=${MODELSCOPE_RETRY_DELAY_SECONDS:-20}

if [[ -z "$MODELSCOPE_REPO_ID" ]]; then
    echo "MODELSCOPE_REPO_ID is required." >&2
    exit 1
fi

echo "ModelScope upload"
echo "  local path : $LOCAL_PATH"
echo "  repo id    : $MODELSCOPE_REPO_ID"
echo "  repo type  : $MODELSCOPE_REPO_TYPE"
echo "  repo path  : ${PATH_IN_REPO:-.}"
echo "  revision   : $MODELSCOPE_REVISION"
echo "  batch size : $MODELSCOPE_BATCH_SIZE"
echo "  max workers: $MODELSCOPE_MAX_WORKERS"
echo "  resume     : $MODELSCOPE_RESUME"
echo "  dry run    : $MODELSCOPE_DRY_RUN"

export LOCAL_PATH
export PATH_IN_REPO
export MODELSCOPE_REPO_ID
export MODELSCOPE_REPO_TYPE
export MODELSCOPE_TOKEN
export MODELSCOPE_REVISION
export MODELSCOPE_MAX_WORKERS
export MODELSCOPE_DRY_RUN
export MODELSCOPE_CREATE_REPO
export MODELSCOPE_VISIBILITY
export MODELSCOPE_COMMIT_MESSAGE
export MODELSCOPE_IGNORE_PATTERNS
export MODELSCOPE_BATCH_SIZE
export MODELSCOPE_RESUME
export MODELSCOPE_RETRY_TIMES
export MODELSCOPE_RETRY_DELAY_SECONDS

python - <<'PY'
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Callable, Iterable, TypeVar

import requests
from modelscope.hub.api import HubApi
from modelscope.utils.repo_utils import RepoUtils

T = TypeVar("T")


def _parse_patterns(raw: str) -> list[str]:
    if not raw:
        return []
    patterns: list[str] = []
    for chunk in raw.replace("\n", ",").split(","):
        chunk = chunk.strip()
        if chunk:
            patterns.append(chunk)
    return patterns


def _build_ignore_patterns(extra_ignore: list[str]) -> list[str]:
    ignore_patterns = [
        ".modelscope_upload_progress.json",
        ".runtime/**",
        "**/.runtime/**",
        ".cache/**",
        "**/.cache/**",
        ".DS_Store",
        "**/.DS_Store",
    ]
    ignore_patterns.extend(extra_ignore)
    return ignore_patterns


def _list_local_files(local_root: Path, ignore_patterns: list[str]) -> list[tuple[str, Path]]:
    relpath_to_abspath = {
        path.relative_to(local_root).as_posix(): path
        for path in sorted(local_root.glob("**/*"))
        if path.is_file()
    }
    filtered_relpaths = list(
        RepoUtils.filter_repo_objects(
            relpath_to_abspath.keys(),
            ignore_patterns=ignore_patterns or None,
        )
    )
    return [(relpath, relpath_to_abspath[relpath]) for relpath in filtered_relpaths]


def _normalize_repo_path(path: str, path_in_repo: str) -> str:
    normalized = path.replace("\\", "/").lstrip("/")
    prefix = path_in_repo.strip("/")
    if prefix and normalized and normalized != prefix and not normalized.startswith(f"{prefix}/"):
        normalized = f"{prefix}/{normalized}"
    return normalized


def _list_remote_files(
    api: HubApi,
    *,
    repo_id: str,
    repo_type: str,
    revision: str,
    path_in_repo: str,
    token: str | None,
) -> set[str]:
    if repo_type != "dataset":
        raise ValueError(f"Remote resume currently supports dataset repos only, got: {repo_type}")

    root_path = f"/{path_in_repo.strip('/')}" if path_in_repo else "/"
    page_number = 1
    page_size = 100
    file_paths: set[str] = set()

    while True:
        dataset_files = api.get_dataset_files(
            repo_id=repo_id,
            revision=revision,
            root_path=root_path,
            recursive=True,
            page_number=page_number,
            page_size=page_size,
            token=token,
        )
        for entry in dataset_files:
            if (entry.get("Type") or entry.get("type")) == "tree":
                continue
            path = entry.get("Path") or entry.get("path") or entry.get("Name") or entry.get("name")
            if path:
                file_paths.add(_normalize_repo_path(str(path), path_in_repo))
        if len(dataset_files) < page_size:
            break
        page_number += 1

    return file_paths


def _is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, requests.exceptions.RequestException):
        return True

    message = str(exc)
    retryable_markers = [
        "502 Server Error",
        "503 Server Error",
        "504 Server Error",
        "NameResolutionError",
        "Temporary failure in name resolution",
        "Max retries exceeded",
        "Failed to resolve",
        "Connection aborted",
        "Connection reset",
        "Read timed out",
        "Remote end closed connection",
    ]
    return any(marker in message for marker in retryable_markers)


def _call_with_retries(
    action: Callable[[], T],
    *,
    label: str,
    retry_times: int,
    retry_delay_seconds: int,
) -> T:
    for attempt in range(1, retry_times + 1):
        try:
            return action()
        except Exception as exc:
            if attempt >= retry_times or not _is_retryable_error(exc):
                raise
            sleep_seconds = retry_delay_seconds * attempt
            print(
                f"{label} failed on attempt {attempt}/{retry_times}: {exc}\n"
                f"Retrying in {sleep_seconds}s...",
                flush=True,
            )
            time.sleep(sleep_seconds)
    raise RuntimeError(f"Unreachable retry loop for {label}")


def _chunked(items: list[tuple[str, Path]], chunk_size: int) -> Iterable[list[tuple[str, Path]]]:
    effective_chunk_size = max(1, chunk_size)
    for index in range(0, len(items), effective_chunk_size):
        yield items[index:index + effective_chunk_size]


def _stage_batch(local_root: Path, batch: list[tuple[str, Path]]) -> Path:
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=".modelscope-upload-",
            dir=str(local_root.parent),
        )
    )
    try:
        for relpath, source_path in batch:
            staged_path = staging_dir / relpath
            staged_path.parent.mkdir(parents=True, exist_ok=True)
            staged_path.symlink_to(source_path)
        return staging_dir
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


local_path = Path(os.environ["LOCAL_PATH"]).expanduser().resolve()
path_in_repo = os.environ["PATH_IN_REPO"].strip("/")
repo_id = os.environ["MODELSCOPE_REPO_ID"]
repo_type = os.environ["MODELSCOPE_REPO_TYPE"]
token = os.environ.get("MODELSCOPE_TOKEN") or None
revision = os.environ["MODELSCOPE_REVISION"]
max_workers = int(os.environ["MODELSCOPE_MAX_WORKERS"])
dry_run = os.environ["MODELSCOPE_DRY_RUN"] == "1"
create_repo = os.environ["MODELSCOPE_CREATE_REPO"] == "1"
visibility = os.environ["MODELSCOPE_VISIBILITY"]
commit_message = os.environ["MODELSCOPE_COMMIT_MESSAGE"]
extra_ignore = _parse_patterns(os.environ.get("MODELSCOPE_IGNORE_PATTERNS", ""))
batch_size = int(os.environ["MODELSCOPE_BATCH_SIZE"])
resume = os.environ["MODELSCOPE_RESUME"] == "1"
retry_times = int(os.environ["MODELSCOPE_RETRY_TIMES"])
retry_delay_seconds = int(os.environ["MODELSCOPE_RETRY_DELAY_SECONDS"])

if not local_path.exists():
    raise SystemExit(f"Local path does not exist: {local_path}")

ignore_patterns = _build_ignore_patterns(extra_ignore)

summary = {
    "local_path": str(local_path),
    "repo_id": repo_id,
    "repo_type": repo_type,
    "path_in_repo": path_in_repo,
    "revision": revision,
    "ignore_patterns": ignore_patterns,
    "batch_size": batch_size,
    "resume": resume,
    "retry_times": retry_times,
    "retry_delay_seconds": retry_delay_seconds,
}

api = HubApi()

if local_path.is_dir():
    local_files = _list_local_files(local_path, ignore_patterns)
    remote_paths: set[str] = set()

    if resume and not dry_run:
        repo_exists = _call_with_retries(
            lambda: api.repo_exists(repo_id, repo_type=repo_type, token=token),
            label="Repo existence check",
            retry_times=retry_times,
            retry_delay_seconds=retry_delay_seconds,
        )
        if repo_exists:
            remote_paths = _call_with_retries(
                lambda: _list_remote_files(
                    api,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    revision=revision,
                    path_in_repo=path_in_repo,
                    token=token,
                ),
                label="Remote file listing",
                retry_times=retry_times,
                retry_delay_seconds=retry_delay_seconds,
            )
    else:
        repo_exists = False

    remaining_files = [
        (relpath, source_path)
        for relpath, source_path in local_files
        if not resume or _normalize_repo_path(relpath, path_in_repo) not in remote_paths
    ]

    summary.update(
        {
            "total_local_files": len(local_files),
            "already_committed_files": len(local_files) - len(remaining_files),
            "remaining_files": len(remaining_files),
            "batch_count": 0 if not remaining_files else (len(remaining_files) - 1) // max(1, batch_size) + 1,
        }
    )

    print(json.dumps(summary, indent=2))

    if dry_run:
        raise SystemExit(0)

    if create_repo and not repo_exists:
        _call_with_retries(
            lambda: api.create_repo(
                repo_id,
                token=token,
                visibility=visibility,
                repo_type=repo_type,
                exist_ok=True,
                create_default_config=False,
            ),
            label="Repo creation",
            retry_times=retry_times,
            retry_delay_seconds=retry_delay_seconds,
        )

    if not remaining_files:
        print("No files left to upload.")
        raise SystemExit(0)

    batches = list(_chunked(remaining_files, batch_size))
    result = None
    for batch_index, batch in enumerate(batches, start=1):
        batch_commit_message = f"{commit_message} (batch {batch_index}/{len(batches)})"
        print(
            f"Uploading batch {batch_index}/{len(batches)} with {len(batch)} file(s)...",
            flush=True,
        )
        staging_dir = _stage_batch(local_path, batch)
        try:
            result = _call_with_retries(
                lambda: api.upload_folder(
                    repo_id=repo_id,
                    folder_path=str(staging_dir),
                    path_in_repo=path_in_repo,
                    commit_message=batch_commit_message,
                    commit_description="Batched dataset upload",
                    token=token,
                    repo_type=repo_type,
                    ignore_patterns=ignore_patterns,
                    max_workers=min(max_workers, len(batch)),
                    revision=revision,
                ),
                label=f"Batch {batch_index}/{len(batches)} upload",
                retry_times=retry_times,
                retry_delay_seconds=retry_delay_seconds,
            )
        finally:
            shutil.rmtree(staging_dir, ignore_errors=True)
else:
    summary.update({"total_local_files": 1, "remaining_files": 1, "batch_count": 1})
    print(json.dumps(summary, indent=2))

    if dry_run:
        raise SystemExit(0)

    if create_repo:
        _call_with_retries(
            lambda: api.create_repo(
                repo_id,
                token=token,
                visibility=visibility,
                repo_type=repo_type,
                exist_ok=True,
                create_default_config=False,
            ),
            label="Repo creation",
            retry_times=retry_times,
            retry_delay_seconds=retry_delay_seconds,
        )

    result = _call_with_retries(
        lambda: api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            token=token,
            repo_type=repo_type,
            commit_message=commit_message,
            revision=revision,
        ),
        label="File upload",
        retry_times=retry_times,
        retry_delay_seconds=retry_delay_seconds,
    )

print("Upload complete.")
print(result)
PY
