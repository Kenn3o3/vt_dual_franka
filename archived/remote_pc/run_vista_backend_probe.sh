#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-vista_so3}"
GPU="${2:-1}"
EPOCHS="${3:-1}"
BATCH_SIZE="${4:-64}"

REMOTE_ROOT="${REMOTE_ROOT:-/mnt/pfs_cuhk/kenny/vt_franka}"
TASK_NAME="${TASK_NAME:-usb_insertion}"
PYTHON_BIN="${PYTHON_BIN:-/home/zlkenny/.conda/envs/isp/bin/python}"
SOURCE_RUN_NAME="${SOURCE_RUN_NAME:-vista_so3_gpu1_epoch1_probe}"
RUN_NAME="${RUN_NAME:-${MODEL}_gpu${GPU}_epoch${EPOCHS}_backend_probe}"

cd "${REMOTE_ROOT}"
export PYTHONPATH="${REMOTE_ROOT}/robot_workspace/src:${REMOTE_ROOT}/shared/src:${PYTHONPATH:-}"

LOG_DIR="${REMOTE_ROOT}/robot_workspace/data/checkpoints/_remote_scheduler/logs"
mkdir -p "${LOG_DIR}"
LOG_PATH="${LOG_DIR}/${RUN_NAME}.log"
MEM_PATH="${LOG_DIR}/${RUN_NAME}_mem.csv"

BACKEND_DATASET_ROOT="${BACKEND_DATASET_ROOT:-${REMOTE_ROOT}/robot_workspace/data/checkpoints/${TASK_NAME}/visuotactile/${MODEL}/${SOURCE_RUN_NAME}/backend_dataset}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${REMOTE_ROOT}/robot_workspace/data/checkpoints/${TASK_NAME}/visuotactile/${MODEL}}"
CONFIG_NAME="train_vista"
if [[ "${MODEL}" == "vista_so2" ]]; then
  CONFIG_NAME="train_vista_so2"
fi

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
  echo "[vista_backend_probe] started_at=$(date -Is)"
  echo "[vista_backend_probe] model=${MODEL} physical_gpu=${GPU} epochs=${EPOCHS} batch_size=${BATCH_SIZE}"
  echo "[vista_backend_probe] CUDA_VISIBLE_DEVICES=${GPU} training.device=cuda:0"
  echo "[vista_backend_probe] backend_dataset_root=${BACKEND_DATASET_ROOT}"
  echo "[vista_backend_probe] checkpoint_dir=${CHECKPOINT_DIR}"
  mkdir -p "${CHECKPOINT_DIR}"

  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON_BIN}" \
    "${REMOTE_ROOT}/robot_workspace/src/vt_franka_workspace/policies/VISTA/train.py" \
    "--config-name=${CONFIG_NAME}" \
    "task_name=${TASK_NAME}" \
    "dataset_root=${BACKEND_DATASET_ROOT}" \
    "data_split=clean" \
    "n_demo=90" \
    "training.seed=0" \
    "training.device=cuda:0" \
    "hydra.run.dir=${CHECKPOINT_DIR}" \
    "hydra.sweep.dir=${CHECKPOINT_DIR}" \
    "multi_run.run_dir=${CHECKPOINT_DIR}" \
    "logging.mode=disabled" \
    "task=univtac_vista_lr" \
    "policy.crop_shape=null" \
    "policy.tactile_shape=[3,224,224]" \
    "task.shape_meta.obs.robot0_eye_in_hand_image.shape=[3,224,224]" \
    "task.dataset.shape_meta.obs.robot0_eye_in_hand_image.shape=[3,224,224]" \
    "task.shape_meta.obs.robot0_tactile_left_image.shape=[3,224,224]" \
    "task.dataset.shape_meta.obs.robot0_tactile_left_image.shape=[3,224,224]" \
    "task.shape_meta.obs.robot0_tactile_right_image.shape=[3,224,224]" \
    "task.dataset.shape_meta.obs.robot0_tactile_right_image.shape=[3,224,224]" \
    "dataloader.batch_size=${BATCH_SIZE}" \
    "val_dataloader.batch_size=${BATCH_SIZE}" \
    "training.num_epochs=${EPOCHS}"

  peak_used_mb="$(awk -F, 'NR > 1 && $3 > max {max = $3} END {print max + 0}' "${MEM_PATH}")"
  echo "[vista_backend_probe] finished_at=$(date -Is)"
  echo "[vista_backend_probe] peak_used_mb=${peak_used_mb}"
} >> "${LOG_PATH}" 2>&1
