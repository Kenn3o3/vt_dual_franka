#!/bin/bash
# MOTIF 消融实验启动脚本
#
# 对三个 MOTIF 核心机制进行消融实验，运行五组实验：
#   motif-full     : 完整 MOTIF（三个机制全开）
#   motif-no-td    : w/o M1  关闭物理时间编码，改用可学习位置嵌入
#   motif-no-vel   : w/o M2  关闭速度监督损失 L_vel（alpha_vel=0）
#   motif-no-dct   : w/o M3  关闭 DCT 系数空间，改用原始速度帧
#   motif-only-td  : 仅M1    只保留物理时间编码，关闭M2和M3
#
# 用法:
#   bash run_ablation_motif.sh [选项]
#
# 选项:
#   -t, --tasks   任务列表，逗号分隔（默认：四个 SOFA 任务）
#                 可选值：rope_threading, grasp_lift_touch, ligating_loop,
#                         bimanual_tissue_manipulation
#   -g, --gpus    GPU 编号列表，逗号分隔（默认：0,1,2,3）
#   -r, --retries 单个任务最大重试次数（默认：1）
#   -h, --help    显示此帮助信息
#
# 示例:
#   # 在全部 4 个任务上跑全部消融（16 个 job，4 卡）
#   bash run_ablation_motif.sh
#
#   # 只在 rope_threading 上跑，使用 GPU 0 和 1
#   bash run_ablation_motif.sh -t rope_threading -g 0,1
#
#   # 指定两个任务，4 卡并行
#   bash run_ablation_motif.sh -t rope_threading,grasp_lift_touch -g 0,1,2,3

# ── 环境变量（与 train_sofa_tasks.sh 保持一致）──────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOFA_ROOT="${SCRIPT_DIR}/sofa/SOFA_v23.12.01_Linux"
export SP3_SITE="${SCRIPT_DIR}/sofa/SofaPython3_v23.12.01_python-3.10_for-SOFA-v23.12.01_Linux/lib/python3/site-packages"
export SP3_LIB="${SCRIPT_DIR}/sofa/SofaPython3_v23.12.01_python-3.10_for-SOFA-v23.12.01_Linux/lib"
export SOFAPYTHON3_ROOT="${SCRIPT_DIR}/sofa/SofaPython3_v23.12.01_python-3.10_for-SOFA-v23.12.01_Linux"
export SOFA_LIB=$SOFA_ROOT/lib
export PYTHONPATH=$SP3_SITE${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$SP3_LIB:$SOFA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# ── 无头渲染设置（H100 + NVIDIA EGL）─────────────────────────────────────────
# PYGLET_HEADLESS_DEVICE 在 run_job() 中按 GPU 动态设置，以匹配 CUDA 设备。
export PYGLET_HEADLESS=1
export LD_PRELOAD="${SCRIPT_DIR}/fix_sofa_egl.so"

CONDA_PYTHON=/home/hasac_cover/miniconda3/envs/mpd/bin/python

# ── 消融条件定义 ──────────────────────────────────────────────────────────────
# 每条记录格式: "<method_name>|<hydra overrides (空格分隔)>"
ABLATION_CONDITIONS=(
    # "motif-full|"
    "motif-no-td|agent_config.model_config.inner_model_config.use_physical_time_encoding=false"
    "motif-no-vel|agent_config.model_config.alpha_vel=0.0"
    "motif-no-dct|agent_config.process_batch_config.motif_handler_config.use_dct=false agent_config.model_config.inner_model_config.motif_handler_config.use_dct=false"
    "motif-only-td|agent_config.model_config.alpha_vel=0.0 agent_config.process_batch_config.motif_handler_config.use_dct=false agent_config.model_config.inner_model_config.motif_handler_config.use_dct=false"
)

# ── 默认参数 ──────────────────────────────────────────────────────────────────
TASKS="rope_threading,grasp_lift_touch,bimanual_tissue_manipulation,ligating_loop"
GPUS="0,1,2,3"
MAX_RETRIES=1

# ── 参数解析 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tasks)   TASKS="$2";       shift 2 ;;
        -g|--gpus)    GPUS="$2";        shift 2 ;;
        -r|--retries) MAX_RETRIES="$2"; shift 2 ;;
        -h|--help)    sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "[ERROR] 未知参数: $1"; exit 1 ;;
    esac
done

IFS=',' read -ra TASK_LIST <<< "$TASKS"
IFS=',' read -ra GPU_LIST  <<< "$GPUS"
NUM_GPUS=${#GPU_LIST[@]}

# ── 验证任务 ──────────────────────────────────────────────────────────────────
VALID_TASKS="rope_threading grasp_lift_touch ligating_loop bimanual_tissue_manipulation"
for task in "${TASK_LIST[@]}"; do
    task="$(echo "$task" | tr -d '[:space:]')"
    if [[ ! " $VALID_TASKS " =~ " $task " ]]; then
        echo "[ERROR] 未知任务: '$task'"; echo "        合法任务: $VALID_TASKS"; exit 1
    fi
    cfg="$SCRIPT_DIR/conf/experiments/$task/train_motif_transformer.yaml"
    if [[ ! -f "$cfg" ]]; then
        echo "[ERROR] 找不到配置文件: $cfg"; exit 1
    fi
done

# ── 临时文件 ──────────────────────────────────────────────────────────────────
_RUN_ID="$$_$(echo $RANDOM | md5sum | cut -c1-6)"
_QUEUE_FILE="/tmp/mpd_ablation_queue_${_RUN_ID}"
_LOCK_FILE="/tmp/mpd_ablation_lock_${_RUN_ID}"
_QSIZE_FILE="/tmp/mpd_ablation_qsize_${_RUN_ID}"
declare -A BG_PIDS

cleanup() {
    echo ""
    echo "[SIGNAL] 捕获退出信号，终止所有训练进程..."
    for pid in "${BG_PIDS[@]}"; do
        [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && kill -TERM "$pid" 2>/dev/null
    done
    for pid in "${BG_PIDS[@]}"; do
        [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    done
    rm -f "${_QUEUE_FILE}" "${_LOCK_FILE}" "${_QSIZE_FILE}"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ── 构建任务队列（任务优先：task1|full, task1|no-td, ... task2|full, ...）────
> "$_QUEUE_FILE"
for task in "${TASK_LIST[@]}"; do
    t="$(echo "$task" | tr -d '[:space:]')"
    for cond in "${ABLATION_CONDITIONS[@]}"; do
        method_name="${cond%%|*}"
        overrides="${cond##*|}"
        echo "${t}|${method_name}|${overrides}" >> "$_QUEUE_FILE"
    done
done
NUM_JOBS=$(wc -l < "$_QUEUE_FILE")
echo "$NUM_JOBS" > "$_QSIZE_FILE"

# ── 打印计划 ──────────────────────────────────────────────────────────────────
echo "========================================================"
echo " MOTIF 消融实验"
echo "========================================================"
echo " 任务列表  : ${TASK_LIST[*]}"
echo " 消融条件  : full | no-td (w/o M1) | no-vel (w/o M2) | no-dct (w/o M3) | only-td (仅M1)"
echo " 总 job 数 : $NUM_JOBS"
echo " GPU 列表  : ${GPU_LIST[*]}"
echo " 最大重试  : $MAX_RETRIES 次"
echo " 结果目录  : $SCRIPT_DIR/results/{task}/{method_name}/"
echo "--------------------------------------------------------"
echo " 队列顺序:"
awk -F'|' '{printf "   %-40s  →  %s\n", $1, $2}' "$_QUEUE_FILE"
echo "--------------------------------------------------------"

# ── 单 job 执行（含重试）──────────────────────────────────────────────────────
run_job() {
    local gpu=$1
    local task=$2
    local method_name=$3
    local overrides=$4
    local attempt=0

    while (( attempt <= MAX_RETRIES )); do
        (( attempt++ ))
        echo " [GPU $gpu] 启动 $task / $method_name（第 ${attempt} 次尝试）"

        # 将 overrides 字符串拆分为独立参数
        read -ra override_args <<< "$overrides"

        CUDA_VISIBLE_DEVICES=$gpu \
        PYGLET_HEADLESS_DEVICE=$gpu \
            $CONDA_PYTHON "$SCRIPT_DIR/scripts/train.py" \
                --config-name "experiments/${task}/train_motif_transformer" \
                "method_name=${method_name}" \
                "${override_args[@]}" \
            2>&1 | grep -v \
                -e "the library has not been cleaned up" \
                -e "TopologyData:.*already has a TopologyDataHandler" \
                -e "createTopologyHandler should only be called once" \
                -e "totalMass value overriding the value"
        status=${PIPESTATUS[0]}

        if (( status == 0 )); then
            echo " [GPU $gpu] ✓ $task / $method_name 完成"
            return 0
        elif (( attempt <= MAX_RETRIES )); then
            echo " [GPU $gpu] ! $task / $method_name 失败 (exit=$status)，15 秒后重试..."
            sleep 15
        fi
    done

    echo " [GPU $gpu] ✗ $task / $method_name 最终失败（共 ${attempt} 次）"
    return $status
}

# ── GPU 工作进程 ──────────────────────────────────────────────────────────────
gpu_worker() {
    local gpu=$1
    local worker_id=$2

    echo " [Worker $worker_id - GPU $gpu] 工作进程启动"

    while true; do
        local tmp_job="/tmp/mpd_ablation_worker_${worker_id}_${_RUN_ID}"
        (
            flock -x 200
            if [[ -s "$_QUEUE_FILE" ]]; then
                head -n 1 "$_QUEUE_FILE" > "$tmp_job"
                sed -i '1d' "$_QUEUE_FILE"
                wc -l < "$_QUEUE_FILE" > "$_QSIZE_FILE"
            else
                rm -f "$tmp_job"
            fi
        ) 200>"$_LOCK_FILE"

        [[ -f "$tmp_job" ]] || break
        local job; job=$(cat "$tmp_job"); rm -f "$tmp_job"

        IFS='|' read -r task method_name overrides <<< "$job"
        remaining=$(cat "$_QSIZE_FILE" 2>/dev/null || echo "?")
        echo " [Worker $worker_id - GPU $gpu] 取到 job: $task / $method_name（队列剩余: $remaining）"

        # 错开 SOFA 初始化避免并发竞态
        (( worker_id > 1 )) && sleep $(( (worker_id - 1) * 15 ))

        run_job "$gpu" "$task" "$method_name" "$overrides"
    done

    echo " [Worker $worker_id - GPU $gpu] 工作进程结束"
}

# ── 启动 GPU 工作进程 ─────────────────────────────────────────────────────────
echo " 启动 $NUM_GPUS 个 GPU 工作进程..."
for (( idx=0; idx<NUM_GPUS; idx++ )); do
    gpu="$(echo "${GPU_LIST[$idx]}" | tr -d '[:space:]')"
    worker_id=$(( idx + 1 ))
    gpu_worker "$gpu" "$worker_id" &
    BG_PIDS[$gpu]=$!
done

echo "========================================================"
echo " 所有工作进程已启动，PID: ${BG_PIDS[*]}"
echo "========================================================"

# ── 等待结束 ──────────────────────────────────────────────────────────────────
FAILED_WORKERS=()
for gpu in "${!BG_PIDS[@]}"; do
    wait "${BG_PIDS[$gpu]}"
    [[ $? -ne 0 ]] && FAILED_WORKERS+=("GPU${gpu}")
done

rm -f "${_QUEUE_FILE}" "${_LOCK_FILE}" "${_QSIZE_FILE}"

echo "========================================================"
if [[ ${#FAILED_WORKERS[@]} -eq 0 ]]; then
    echo " 全部消融实验完成！"
    echo " 结果目录: $SCRIPT_DIR/results/{task}/motif-{full,no-td,no-vel,no-dct,only-td}/"
else
    echo " 以下工作进程异常退出: ${FAILED_WORKERS[*]}"
    exit 1
fi
echo "========================================================"
