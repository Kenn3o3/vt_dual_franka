#!/bin/bash
# MPD 多任务多算法多 GPU 并行训练启动脚本
#
# 用法:
#   bash train_sofa_tasks.sh [选项] [额外 Hydra 参数]
#
# 选项:
#   -t, --tasks   任务列表，逗号分隔（默认：所有 4 个 SOFA 任务）
#                 可选值：rope_threading, grasp_lift_touch, ligating_loop,
#                         bimanual_tissue_manipulation
#   -a, --algos   算法列表，逗号分隔（默认：prodmp_transformer）
#                 可选值：prodmp_transformer, fm_transformer, sfp_transformer,
#                         motif_transformer, motif_fm_transformer,
#                         dp_transformer, dp_unet1d, beso, ...
#                 支持多算法：-a motif_transformer,sfp_transformer,fm_transformer
#   -g, --gpus    GPU 编号列表，逗号分隔（默认：4,5,6,7）
#                 GPU 数量少于总任务数时循环复用
#   -r, --retries 单个任务最大重试次数（默认：1，即失败后重试 1 次）
#   -h, --help    显示此帮助信息
#
# 多任务多算法的执行顺序：任务优先（task1-algo1, task1-algo2, task2-algo1, ...）
# 即先完成某个任务在所有算法下的训练，再进入下一个任务。
#
# 示例:
#   # 跑全部 4 个任务（默认算法 prodmp_transformer）
#   bash train_sofa_tasks.sh
#
#   # 用 motif+sfp 两个算法跑全部 4 个任务（共 8 个 job，任务优先入队）
#   bash train_sofa_tasks.sh -a motif_transformer,sfp_transformer
#
#   # 只跑 2 个任务，用 sfp 算法，各自指定 GPU
#   bash train_sofa_tasks.sh -t rope_threading,grasp_lift_touch -a sfp_transformer -g 0,1
#
#   # 跑全部任务和算法的组合，4 卡并行
#   bash train_sofa_tasks.sh -a prodmp_transformer,fm_transformer,sfp_transformer,motif_transformer -g 4,5,6,7
#
#   # 跑单个任务在指定 GPU 上，并覆盖 epochs
#   bash train_sofa_tasks.sh -t bimanual_tissue_manipulation -g 2 epochs=100

# ── 环境变量 ──────────────────────────────────────────────────────────────────
# SCRIPT_DIR 为本脚本所在目录（即 mpd/），使 SOFA 路径与仓库位置无关
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SOFA_ROOT="${SCRIPT_DIR}/sofa/SOFA_v23.12.01_Linux"
export SP3_SITE="${SCRIPT_DIR}/sofa/SofaPython3_v23.12.01_python-3.10_for-SOFA-v23.12.01_Linux/lib/python3/site-packages"
export SP3_LIB="${SCRIPT_DIR}/sofa/SofaPython3_v23.12.01_python-3.10_for-SOFA-v23.12.01_Linux/lib"
export SOFAPYTHON3_ROOT="${SCRIPT_DIR}/sofa/SofaPython3_v23.12.01_python-3.10_for-SOFA-v23.12.01_Linux"
export SOFA_LIB=$SOFA_ROOT/lib
export PYTHONPATH=$SP3_SITE${PYTHONPATH:+:$PYTHONPATH}
export LD_LIBRARY_PATH=$SP3_LIB:$SOFA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

# ── 无头渲染设置（H100 + NVIDIA EGL）─────────────────────────────────────────
# H100 EGL (设备 0-7) 支持 OpenGL 4.6 兼容 profile，可直接 GPU 渲染。
# fix_sofa_egl.so 仅修复 GLX→EGL 重定向，不 stub 光照/FBO，保留完整渲染效果。
# （fix_sofa_headless.so 是 llvmpipe 专用版本，stub 了所有光照导致全黑帧）
# PYGLET_HEADLESS_DEVICE 在 run_task() 中按 GPU 动态设置，以匹配 CUDA 设备。
export PYGLET_HEADLESS=1
export LD_PRELOAD="${SCRIPT_DIR}/fix_sofa_egl.so"

CONDA_PYTHON=/home/hasac_cover/miniconda3/envs/mpd/bin/python

# ── 临时文件命名（含 PID + 随机后缀，保证唯一性）────────────────────────────
_script_pid=$$
_random_suffix=$(echo $RANDOM | md5sum | cut -c1-6)
_RUN_ID="${_script_pid}_${_random_suffix}"
_QUEUE_FILE="/tmp/mpd_train_queue_${_RUN_ID}"
_LOCK_FILE="/tmp/mpd_train_lock_${_RUN_ID}"
_QSIZE_FILE="/tmp/mpd_train_qsize_${_RUN_ID}"

# ── 后台进程 PID 表（gpu -> pid）────────────────────────────────────────────
declare -A BG_PIDS

# ── 信号处理：Ctrl-C / kill 时优雅退出 ──────────────────────────────────────
cleanup() {
    echo ""
    echo "[SIGNAL] 捕获退出信号，终止所有训练进程..."

    local pids_to_kill=("${BG_PIDS[@]}")
    for pid in "${pids_to_kill[@]}"; do
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            echo "  终止进程 $pid (SIGTERM)..."
            kill -TERM "$pid" 2>/dev/null
            sleep 1
            if kill -0 "$pid" 2>/dev/null; then
                echo "  强制杀死进程 $pid (SIGKILL)..."
                kill -KILL "$pid" 2>/dev/null
            fi
        fi
    done

    for pid in "${pids_to_kill[@]}"; do
        [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
    done

    echo "[SIGNAL] 清理临时文件..."
    rm -f "${_QUEUE_FILE}" "${_LOCK_FILE}" "${_QSIZE_FILE}"
    echo "[SIGNAL] 完成。"
    exit 1
}
trap cleanup SIGINT SIGTERM

# ── 默认参数 ──────────────────────────────────────────────────────────────────
TASKS="rope_threading,grasp_lift_touch,bimanual_tissue_manipulation,ligating_loop"
ALGOS="motif_transformer,prodmp_transformer,sfp_transformer,fm_transformer,dp_transformer"
GPUS="0,1,2,3,4"

# TASKS="grasp_lift_touch"
# ALGOS="motif_transformer,prodmp_transformer,sfp_transformer,fm_transformer,dp_transformer"
# GPUS="0,1,2,3,7"

# test
# TASKS="grasp_lift_touch"
# ALGOS="prodmp_transformer"
# GPUS="3"

MAX_RETRIES=1

# ── 参数解析 ──────────────────────────────────────────────────────────────────
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -t|--tasks)   TASKS="$2";       shift 2 ;;
        -a|--algos)   ALGOS="$2";       shift 2 ;;
        -g|--gpus)    GPUS="$2";        shift 2 ;;
        -r|--retries) MAX_RETRIES="$2"; shift 2 ;;
        -h|--help)    sed -n '2,42p' "$0"; exit 0 ;;
        *)            POSITIONAL_ARGS+=("$1"); shift ;;
    esac
done

IFS=',' read -ra TASK_LIST <<< "$TASKS"
IFS=',' read -ra ALGO_LIST <<< "$ALGOS"
IFS=',' read -ra GPU_LIST  <<< "$GPUS"
HYDRA_OVERRIDES=("${POSITIONAL_ARGS[@]}")
NUM_GPUS=${#GPU_LIST[@]}

# ── 验证任务和算法配置合法 ────────────────────────────────────────────────────
VALID_TASKS="rope_threading grasp_lift_touch ligating_loop bimanual_tissue_manipulation"
for task in "${TASK_LIST[@]}"; do
    task="$(echo "$task" | tr -d '[:space:]')"
    if [[ ! " $VALID_TASKS " =~ " $task " ]]; then
        echo "[ERROR] 未知任务: '$task'"; echo "        合法任务: $VALID_TASKS"; exit 1
    fi
    for algo in "${ALGO_LIST[@]}"; do
        algo="$(echo "$algo" | tr -d '[:space:]')"
        cfg="$SCRIPT_DIR/conf/experiments/$task/train_${algo}.yaml"
        if [[ ! -f "$cfg" ]]; then
            echo "[ERROR] 找不到配置文件: $cfg"
            echo "        请检查算法名 '$algo' 是否正确"
            echo "        可用配置: $(ls $SCRIPT_DIR/conf/experiments/$task/train_*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/train_//' | tr '\n' ' ')"
            exit 1
        fi
    done
done

# ── 验证 GPU 编号合法 ─────────────────────────────────────────────────────────
MAX_GPU_ID=$(($($CONDA_PYTHON -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1) - 1))
for gpu in "${GPU_LIST[@]}"; do
    gpu="$(echo "$gpu" | tr -d '[:space:]')"
    if ! [[ "$gpu" =~ ^[0-9]+$ ]] || [[ "$gpu" -gt "$MAX_GPU_ID" ]]; then
        echo "[ERROR] 无效 GPU 编号: '$gpu'（可用范围: 0-$MAX_GPU_ID）"; exit 1
    fi
done

# ── 构建任务队列文件（任务优先：task1|algo1, task1|algo2, task2|algo1, ...）──
> "$_QUEUE_FILE"
for task in "${TASK_LIST[@]}"; do
    t="$(echo "$task" | tr -d '[:space:]')"
    for algo in "${ALGO_LIST[@]}"; do
        a="$(echo "$algo" | tr -d '[:space:]')"
        echo "${t}|${a}" >> "$_QUEUE_FILE"
    done
done
NUM_JOBS=$(wc -l < "$_QUEUE_FILE")
echo "$NUM_JOBS" > "$_QSIZE_FILE"

# ── 打印任务计划 ──────────────────────────────────────────────────────────────
echo "========================================================"
echo " MPD 并行训练启动"
echo "========================================================"
echo " 任务列表  : ${TASK_LIST[*]}"
echo " 算法列表  : ${ALGO_LIST[*]}"
echo " 总 job 数 : $NUM_JOBS（任务优先入队）"
echo " GPU 列表  : ${GPU_LIST[*]}"
echo " 最大重试  : $MAX_RETRIES 次"
echo " 结果目录  : $SCRIPT_DIR/results/{task_name}/{method_name}/"
[[ ${#HYDRA_OVERRIDES[@]} -gt 0 ]] && echo " 额外参数  : ${HYDRA_OVERRIDES[*]}"
echo "--------------------------------------------------------"
echo " 队列顺序:"
cat "$_QUEUE_FILE" | awk -F'|' '{printf "   %s  →  %s\n", $1, $2}'
echo "--------------------------------------------------------"

# ── 单 job 执行函数（含重试）─────────────────────────────────────────────────
# 训练日志由 Hydra 写入 results/{task_name}/{method_name}/日期/时间/train.log
run_task() {
    local gpu=$1
    local task=$2
    local algo=$3
    local attempt=0
    local status=0

    while (( attempt <= MAX_RETRIES )); do
        (( attempt++ ))

        echo " [GPU $gpu] 启动 $task / $algo（第 ${attempt} 次尝试）"

        CUDA_VISIBLE_DEVICES=$gpu \
        PYGLET_HEADLESS_DEVICE=$gpu \
            $CONDA_PYTHON "$SCRIPT_DIR/scripts/train.py" \
                --config-name "experiments/${task}/train_${algo}" \
                "${HYDRA_OVERRIDES[@]}" \
            2>&1 | grep -v \
                -e "the library has not been cleaned up" \
                -e "TopologyData:.*already has a TopologyDataHandler" \
                -e "createTopologyHandler should only be called once" \
                -e "totalMass value overriding the value"
        status=${PIPESTATUS[0]}

        if (( status == 0 )); then
            echo " [GPU $gpu] ✓ $task / $algo 完成"
            return 0
        elif (( attempt <= MAX_RETRIES )); then
            echo " [GPU $gpu] ! $task / $algo 失败 (exit=$status)，15 秒后重试..."
            sleep 15
        fi
    done

    echo " [GPU $gpu] ✗ $task / $algo 最终失败 (exit=$status，共尝试 ${attempt} 次)"
    return $status
}

# ── GPU 工作进程：从队列取 job 直到队列为空 ──────────────────────────────────
gpu_worker() {
    local gpu=$1
    local worker_id=$2

    echo " [Worker $worker_id - GPU $gpu] 工作进程启动"

    while true; do
        # 用文件锁从队列头取一个 job（格式: task|algo）
        local job=""
        local tmp_job_file="/tmp/mpd_train_worker_${worker_id}_${_RUN_ID}"
        (
            flock -x 200
            if [[ -s "$_QUEUE_FILE" ]]; then
                head -n 1 "$_QUEUE_FILE" > "$tmp_job_file"
                sed -i '1d' "$_QUEUE_FILE"
                wc -l < "$_QUEUE_FILE" > "$_QSIZE_FILE"
            else
                rm -f "$tmp_job_file"
            fi
        ) 200>"$_LOCK_FILE"

        if [[ -f "$tmp_job_file" ]]; then
            job=$(cat "$tmp_job_file")
            rm -f "$tmp_job_file"
        else
            job=""
        fi

        # 队列已空，退出
        if [[ -z "$job" ]]; then
            echo " [Worker $worker_id - GPU $gpu] 队列已空，退出"
            break
        fi

        local task algo remaining
        task="${job%%|*}"
        algo="${job##*|}"
        remaining=$(cat "$_QSIZE_FILE" 2>/dev/null || echo "?")
        echo " [Worker $worker_id - GPU $gpu] 取到 job: $task / $algo（队列剩余: $remaining）"

        # 错开 SOFA 初始化避免并发竞态（非首个 worker 延迟 15 秒）
        if (( worker_id > 1 )); then
            sleep $(( (worker_id - 1) * 15 ))
        fi

        run_task "$gpu" "$task" "$algo"
    done

    echo " [Worker $worker_id - GPU $gpu] 工作进程结束"
}

# ── 为每个 GPU 启动一个工作进程 ──────────────────────────────────────────────
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

# ── 等待所有工作进程结束，汇报结果 ──────────────────────────────────────────
FAILED_WORKERS=()
for gpu in "${!BG_PIDS[@]}"; do
    pid=${BG_PIDS[$gpu]}
    wait "$pid"
    code=$?
    if [[ $code -ne 0 ]]; then
        FAILED_WORKERS+=("GPU$gpu(exit=$code)")
    fi
done

# ── 清理临时文件 ──────────────────────────────────────────────────────────────
rm -f "${_QUEUE_FILE}" "${_LOCK_FILE}" "${_QSIZE_FILE}"

echo "========================================================"
if [[ ${#FAILED_WORKERS[@]} -eq 0 ]]; then
    echo " 全部任务完成！"
else
    echo " 以下工作进程异常退出: ${FAILED_WORKERS[*]}"
    echo " 请查看 results/{task_name}/{method_name}/日期/时间/train.log 排查原因。"
    exit 1
fi
echo "========================================================"
