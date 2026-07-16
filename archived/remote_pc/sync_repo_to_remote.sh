#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/pfs_cuhk/kenny/vt_franka}"
SSH_PORT="${SSH_PORT:-}"
SSH_KEY="${SSH_KEY:-}"
DRY_RUN="${DRY_RUN:-0}"
SYNC_MODE="${SYNC_MODE:-rsync}"

if [[ -z "${REMOTE}" ]]; then
  echo "REMOTE is required, for example user@host" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SSH_ARGS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
if [[ -n "${SSH_PORT}" ]]; then SSH_ARGS+=(-p "${SSH_PORT}"); fi
if [[ -n "${SSH_KEY}" ]]; then SSH_ARGS+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes); fi

EXCLUDES=(
  --exclude=__pycache__
  --exclude='*.pyc'
  --exclude=.pytest_cache
  --exclude=.mypy_cache
  --exclude=.ruff_cache
  --exclude=.git
  --exclude=analysis
  --exclude=Log
  --exclude=UNTOUCHED_REFERENCE
  --exclude=robot_workspace/Log
  --exclude=robot_workspace/debug
  --exclude=robot_workspace/third_parties
  --exclude=robot_workspace/data/collect
  --exclude=robot_workspace/data/preprocess1
  --exclude=robot_workspace/data/prepared
  --exclude=robot_workspace/data/checkpoints
)

if [[ "${SYNC_MODE}" == "tar" ]]; then
  if [[ "${DRY_RUN}" == "1" ]]; then
    tar -C "${REPO_ROOT}" -cf /dev/null "${EXCLUDES[@]}" .
    echo "[dry-run] tar stream to ${REMOTE}:${REMOTE_ROOT}"
    exit 0
  fi
  tar -C "${REPO_ROOT}" -cf - "${EXCLUDES[@]}" . \
    | ssh "${SSH_ARGS[@]}" "${REMOTE}" "mkdir -p '${REMOTE_ROOT}' && tar -xf - -C '${REMOTE_ROOT}'"
  exit 0
fi

if [[ "${SYNC_MODE}" == "minimal" ]]; then
  RSYNC_ARGS=(-a --info=progress2 --partial
    --exclude=__pycache__
    --exclude='*.pyc'
    --exclude=.pytest_cache
    --exclude=.mypy_cache
    --exclude=.ruff_cache
    --exclude='robot_workspace/src/vt_dual_franka_workspace/policies/**/__pycache__'
  )
  if [[ "${DRY_RUN}" == "1" ]]; then
    RSYNC_ARGS+=(--dry-run)
  fi
  ssh "${SSH_ARGS[@]}" "${REMOTE}" "mkdir -p '${REMOTE_ROOT}'"
  rsync "${RSYNC_ARGS[@]}" -R -e "ssh ${SSH_ARGS[*]}" \
    "${REPO_ROOT}/./robot_workspace/src" \
    "${REPO_ROOT}/./robot_workspace/config" \
    "${REPO_ROOT}/./shared/src" \
    "${REPO_ROOT}/./remote_pc" \
    "${REPO_ROOT}/./docs/RUN" \
    "${REPO_ROOT}/./robot_workspace/pyproject.toml" \
    "${REMOTE}:${REMOTE_ROOT}/"
  exit 0
fi

RSYNC_ARGS=(-a --info=progress2 --partial)
RSYNC_ARGS+=("${EXCLUDES[@]}")
if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

rsync "${RSYNC_ARGS[@]}" -e "ssh ${SSH_ARGS[*]}" "${REPO_ROOT}/" "${REMOTE}:${REMOTE_ROOT}/"
