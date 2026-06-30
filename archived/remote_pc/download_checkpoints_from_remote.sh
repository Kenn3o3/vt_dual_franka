#!/usr/bin/env bash
set -euo pipefail

REMOTE="${REMOTE:-}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/pfs_cuhk/kenny/vt_franka}"
TASK_NAME="${TASK_NAME:-}"
SSH_PORT="${SSH_PORT:-}"
SSH_KEY="${SSH_KEY:-}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -z "${REMOTE}" || -z "${TASK_NAME}" ]]; then
  echo "REMOTE and TASK_NAME are required" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_PATH="${REPO_ROOT}/robot_workspace/data/checkpoints/${TASK_NAME}"
REMOTE_PATH="${REMOTE_ROOT}/robot_workspace/data/checkpoints/${TASK_NAME}"

SSH_ARGS=(-o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30 -o ServerAliveCountMax=6)
if [[ -n "${SSH_PORT}" ]]; then SSH_ARGS+=(-p "${SSH_PORT}"); fi
if [[ -n "${SSH_KEY}" ]]; then SSH_ARGS+=(-i "${SSH_KEY}" -o IdentitiesOnly=yes); fi

RSYNC_ARGS=(-a --info=progress2 --partial)
if [[ "${DRY_RUN}" == "1" ]]; then
  RSYNC_ARGS+=(--dry-run)
fi

rsync "${RSYNC_ARGS[@]}" -e "ssh ${SSH_ARGS[*]}" "${REMOTE}:${REMOTE_PATH}/" "${LOCAL_PATH}/"
