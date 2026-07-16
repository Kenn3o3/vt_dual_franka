#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:?usage: run_single_train_probe.sh MODEL GPU EPOCHS [BATCH_SIZE]}"
GPU="${2:?usage: run_single_train_probe.sh MODEL GPU EPOCHS [BATCH_SIZE]}"
EPOCHS="${3:?usage: run_single_train_probe.sh MODEL GPU EPOCHS [BATCH_SIZE]}"
BATCH_SIZE="${4:-64}"

REMOTE_ROOT="${REMOTE_ROOT:-/mnt/pfs_cuhk/kenny/vt_franka}"
TASK_NAME="${TASK_NAME:-usb_insertion}"
PROFILE_NAME="${PROFILE_NAME:-real_canonical_v1}"
PYTHON_BIN="${PYTHON_BIN:-/home/zlkenny/.conda/envs/isp/bin/python}"
RUN_NAME="${RUN_NAME:-${MODEL}_probe}"

cd "${REMOTE_ROOT}"
export PYTHONPATH="${REMOTE_ROOT}/robot_workspace/src:${REMOTE_ROOT}/shared/src:${PYTHONPATH:-}"

LOG_DIR="${REMOTE_ROOT}/robot_workspace/data/checkpoints/_remote_scheduler/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
MEM_PATH="${LOG_DIR}/${RUN_NAME}_mem.csv"

PREPARED_DIR="${REMOTE_ROOT}/robot_workspace/data/prepared/${TASK_NAME}/visuotactile/${PROFILE_NAME}/${MODEL}_${RUN_NAME}"
CHECKPOINT_DIR="${REMOTE_ROOT}/robot_workspace/data/checkpoints/${TASK_NAME}/visuotactile/${MODEL}"
BACKEND_DATASET_ROOT="${CHECKPOINT_DIR}/backend_dataset"
PREPROCESS1_ROOT="${REMOTE_ROOT}/robot_workspace/data/preprocess1/${TASK_NAME}/${PROFILE_NAME}"

echo "timestamp,index,used_mb,free_mb" > "${MEM_PATH}"
(
  while true; do
    ts="$(date +%s)"
    nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader,nounits \
      | awk -v ts="${ts}" -F, -v gpu="${GPU}" '
          {gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3)}
          $1 == gpu {print ts "," $1 "," $2 "," $3}
        ' >> "${MEM_PATH}"
    sleep 2
  done
) &
MONITOR_PID="$!"
cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

{
  echo "[probe] started_at=$(date -Is)"
  echo "[probe] model=${MODEL} gpu=${GPU} epochs=${EPOCHS} batch_size=${BATCH_SIZE}"
  echo "[probe] CUDA_VISIBLE_DEVICES=${GPU} training_device=cuda:0"
  echo "[probe] prepared_dir=${PREPARED_DIR}"
  echo "[probe] checkpoint_dir=${CHECKPOINT_DIR}"

  "${PYTHON_BIN}" -m vt_dual_franka_workspace.policies.visuotactile.prepare \
    --workspace-config "${REMOTE_ROOT}/robot_workspace/config/workspace.yaml" \
    --task-name "${TASK_NAME}" \
    --model "${MODEL}" \
    --raw-run-dir "${REMOTE_ROOT}/robot_workspace/data/collect/${TASK_NAME}" \
    --output-dir "${PREPARED_DIR}" \
    --source preprocess1 \
    --source-root "${PREPROCESS1_ROOT}" \
    --overwrite

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" -m vt_dual_franka_workspace.policies.visuotactile.train \
    --workspace-config "${REMOTE_ROOT}/robot_workspace/config/workspace.yaml" \
    --task-name "${TASK_NAME}" \
    --model "${MODEL}" \
    --dataset-dir "${PREPARED_DIR}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --backend-dataset-root "${BACKEND_DATASET_ROOT}" \
    --seed 0 \
    --batch-size "${BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --device "cuda:0" \
    --no-prepare \
    --overwrite

  peak_used_mb="$(awk -F, 'NR > 1 && $3 > max {max = $3} END {print max + 0}' "${MEM_PATH}")"
  echo "[probe] finished_at=$(date -Is)"
  echo "[probe] peak_used_mb=${peak_used_mb}"
} >> "${LOG_PATH}" 2>&1
