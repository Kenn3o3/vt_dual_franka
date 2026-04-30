#!/bin/bash
# Baseline 统一训练脚本 - 集成 DP/FM/SFP/ProDMP/MOTIF，支持单卡串行与多卡并行
#
# 串行用法:
#   bash baseline_train.sh [experiment] [model_type] [options]
#
# 并行用法 (多模型分发到多 GPU):
#   bash baseline_train.sh [experiment] --models m1,m2,m3 --gpus 0,1,2 [options]
#
# 选项:
#   -m, --models   LIST   模型类型列表，逗号分隔 → 触发并行模式
#   -g, --gpus     LIST   GPU 编号列表，逗号分隔（并行：每卡一个 worker）
#       --gpu      ID     CUDA_VISIBLE_DEVICES 值（串行，可含多卡 "0,1"）
#   -e, --epochs   N      训练轮数（默认: 1）
#   -d, --device   DEV    cuda/cpu（默认: cuda）
#       --swanlab-mode    cloud/local/disabled（默认: cloud）
#       --swanlab-entity  SwanLab workspace（默认: motif）
#       --swanlab-project SwanLab 项目名（默认: baseline-comparison）
#       --swanlab-group   实验组名
#       --suffix   STR    SwanLab run name 后缀
#       --batch-size N    Batch size
#       --lr RATE         学习率
#       --num-modes N     傅里叶模态数（MOTIF 专用）
#       --alpha-vel ALPHA 速度损失权重（MOTIF 专用）
#       --seed N          随机种子
#       --resume PATH     从 checkpoint 恢复
#   -r, --retries  N      最大重试次数（并行模式，默认: 1）
#       --dry-run         仅显示命令，不执行
#       --auto-yes        跳过交互确认（等价于 AUTO_YES=1）
#   -h, --help            显示帮助

export PYBULLET_EGL=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="mpd"
CONDA_PYTHON="/home/hasac_cover/miniconda3/envs/${CONDA_ENV}/bin/python"

# ============================================================================
# ★ 快速配置区 - 直接在此处修改默认值，无需每次传命令行参数
# ============================================================================

# 默认实验名（串行/并行通用）
DEFAULT_EXPERIMENT="obstacle_avoidance"

# 默认算法列表（并行模式默认跑这些模型；串行模式仅取第一个）
# 可选: dp, fm, sfp, prodmp_diffusion(别名: mpd), prodmp_fm, motif_diffusion(别名: motif), motif_fm

# ALGOS="motif,mpd,sfp,fm,dp"
ALGOS="motif"

# 默认 GPU 编号列表（并行: 每张卡一个 worker；串行: 取第一个作为 CUDA_VISIBLE_DEVICES）
GPUS="3"

# 最大重试次数（并行模式，任务失败后重试）
MAX_RETRIES=1

# 默认训练轮数
DEFAULT_EPOCHS="${EPOCHS:-3000}"

# 默认设备
DEFAULT_DEVICE="${DEVICE:-cuda}"

# 默认 SwanLab 模式
DEFAULT_SWANLAB_MODE="${SWANLAB_MODE:-cloud}"

# SwanLab 配置
SWANLAB_ENTITY="motif"
SWANLAB_PROJECT="baseline-comparison"

# ============================================================================
# 内部变量（由快速配置区派生，一般不需要修改）
# ============================================================================
DEFAULT_MODEL_TYPE="$(echo "$ALGOS" | cut -d',' -f1)"  # 串行模式默认取第一个算法
DEFAULT_GPU_IDS="$(echo "$GPUS" | cut -d',' -f1)"      # 串行模式默认取第一张卡

# ============================================================================
# 颜色输出
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# ============================================================================
# 辅助函数
# ============================================================================
print_header()  { echo -e "${BLUE}================================${NC}"; echo -e "${BLUE}$1${NC}"; echo -e "${BLUE}================================${NC}"; }
print_info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error()   { echo -e "${RED}[ERROR]${NC} $1"; }
print_config()  { echo -e "${CYAN}[CONFIG]${NC} $1"; }

check_conda_env() {
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        print_error "Conda 环境 '${CONDA_ENV}' 不存在！"
        print_info "请先创建环境: conda create -n ${CONDA_ENV} python=3.10"
        exit 1
    fi
}

check_gpu() {
    if ! nvidia-smi &>/dev/null; then
        print_warning "未检测到 NVIDIA GPU"
        return 1
    fi
    local cnt
    cnt=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    print_info "检测到 ${cnt} 个 GPU"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    return 0
}

check_cuda() {
    print_info "检查 CUDA 状态..."
    local cuda_available
    cuda_available=$("$CONDA_PYTHON" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | tail -1)
    if [ "$cuda_available" = "True" ]; then
        print_info "✅ PyTorch CUDA 可用"
        local cnt
        cnt=$("$CONDA_PYTHON" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1)
        print_info "可用 GPU 数量: ${cnt}"
        return 0
    else
        print_warning "⚠️  PyTorch CUDA 不可用，将回退到 CPU"
        return 1
    fi
}

get_config_name() {
    local model_type="$1" experiment="$2"
    case "$model_type" in
        dp)               echo "experiments/${experiment}/train_dp_transformer" ;;
        fm)               echo "experiments/${experiment}/train_fm_transformer" ;;
        sfp)              echo "experiments/${experiment}/train_sfp_transformer" ;;
        prodmp_diffusion) echo "experiments/${experiment}/train_prodmp_transformer" ;;
        prodmp_fm)        echo "experiments/${experiment}/train_prodmp_fm_transformer" ;;
        prodmp_diffusion|mpd) echo "experiments/${experiment}/train_prodmp_transformer" ;;
        motif_fm)         echo "experiments/${experiment}/train_motif_fm_transformer" ;;
        motif_diffusion|motif) echo "experiments/${experiment}/train_motif_transformer" ;;
        *)
            print_error "未知的模型类型: ${model_type}"
            echo "可用: dp, fm, sfp, prodmp_diffusion(=mpd), prodmp_fm, motif_diffusion(=motif), motif_fm" >&2
            exit 1
            ;;
    esac
}

get_model_description() {
    case "$1" in
        dp)               echo "Diffusion Policy (Transformer)" ;;
        fm)               echo "Flow Matching (Transformer)" ;;
        sfp)              echo "Streaming Flow Policy (Transformer)" ;;
        prodmp_diffusion) echo "ProDMP + Diffusion" ;;
        prodmp_fm)        echo "ProDMP + Flow Matching" ;;
        prodmp_diffusion|mpd) echo "ProDMP + Diffusion" ;;
        motif_fm)         echo "MOTIF + Flow Matching" ;;
        motif_diffusion|motif) echo "MOTIF + Diffusion" ;;
        *)                echo "Unknown" ;;
    esac
}

show_usage() {
    cat << EOF
${GREEN}Baseline 统一训练脚本 - 支持串行与多卡并行${NC}

串行用法:
    bash baseline_train.sh [experiment] [model_type] [options]

并行用法 (多模型按 GPU 队列并发，GPU 数少于模型数时循环复用):
    bash baseline_train.sh [experiment] --models m1,m2,m3 --gpus 0,1,2 [options]

参数:
    experiment          实验名称（默认: ${DEFAULT_EXPERIMENT}）
    model_type          模型类型（默认: ${DEFAULT_MODEL_TYPE}，仅串行模式）

模型类型:
    ${CYAN}纯 Transformer:${NC}
    dp                  Diffusion Policy
    fm                  Flow Matching
    sfp                 Streaming Flow Policy
    ${CYAN}ProDMP:${NC}
    prodmp_diffusion    ProDMP + Diffusion
    prodmp_fm           ProDMP + Flow Matching
    ${CYAN}MOTIF:${NC}
    motif_diffusion     MOTIF + Diffusion（集成自 motif_train.sh）
    motif_fm            MOTIF + Flow Matching（集成自 motif_train.sh）

选项:
    -m, --models   LIST   模型类型列表，逗号分隔 → 触发并行模式
    -g, --gpus     LIST   GPU 编号列表，逗号分隔（并行: 每卡一个 worker）
        --gpu      ID     CUDA_VISIBLE_DEVICES 值（串行，可含多卡 "0,1"）
    -e, --epochs   N      训练轮数（默认: ${DEFAULT_EPOCHS}）
    -d, --device   DEV    cuda/cpu（默认: ${DEFAULT_DEVICE}）
        --swanlab-mode    cloud/local/disabled（默认: ${DEFAULT_SWANLAB_MODE}）
        --swanlab-entity  SwanLab workspace（默认: ${SWANLAB_ENTITY}）
        --swanlab-project SwanLab 项目名（默认: ${SWANLAB_PROJECT}）
        --swanlab-group   实验组名
        --suffix   STR    SwanLab run name 后缀
        --batch-size N    Batch size
        --lr RATE         学习率
        --num-modes N     傅里叶模态数（MOTIF 专用，来自 motif_train.sh）
        --alpha-vel ALPHA 速度损失权重（MOTIF 专用，来自 motif_train.sh）
        --seed N          随机种子（来自 mpd_train.sh）
        --resume PATH     从 checkpoint 恢复（来自 mpd_train.sh）
    -r, --retries  N      最大重试次数（并行模式，默认: 1）
        --dry-run         仅显示命令，不执行（来自 motif_train.sh）
        --auto-yes        跳过交互确认（等价于 AUTO_YES=1）
    -h, --help            显示此帮助

示例:
    ${CYAN}# 串行: 单模型${NC}
    bash baseline_train.sh obstacle_avoidance dp --gpu 0 --epochs 3000

    ${CYAN}# 串行: MOTIF 带专有参数（来自 motif_train.sh）${NC}
    bash baseline_train.sh obstacle_avoidance motif_fm --gpu 0 \\
        --num-modes 16 --alpha-vel 1.0 --epochs 3000

    ${CYAN}# 串行: ProDMP 带种子和后缀（来自 mpd_train.sh）${NC}
    bash baseline_train.sh obstacle_avoidance prodmp_fm --gpu 0 \\
        --seed 42 --suffix v2 --epochs 3000

    ${CYAN}# 串行: 仅预览命令，不执行${NC}
    bash baseline_train.sh obstacle_avoidance motif_fm --gpu 0 --dry-run

    ${CYAN}# 并行: 3 模型分发到 3 卡${NC}
    bash baseline_train.sh obstacle_avoidance \\
        --models dp,fm,sfp --gpus 0,1,2 --epochs 3000

    ${CYAN}# 并行: 全量 5 个 baseline，4 卡循环分发（队列机制，无需 job 数 = GPU 数）${NC}
    bash baseline_train.sh obstacle_avoidance \\
        --models dp,fm,sfp,prodmp_fm,motif_fm --gpus 0,1,2,3 --epochs 3000

    ${CYAN}# 并行 + dry-run 预览全部命令${NC}
    bash baseline_train.sh obstacle_avoidance \\
        --models dp,motif_fm --gpus 0,1 --dry-run

    ${CYAN}# 环境变量覆盖${NC}
    EPOCHS=3000 SWANLAB_MODE=local bash baseline_train.sh obstacle_avoidance motif_fm

五个 Baseline 一键对比:
    bash baseline_train.sh <exp> --models dp,fm,sfp,prodmp_fm,motif_fm --gpus 0,1,2,3,4 --epochs 3000

EOF
}

# ============================================================================
# 构建训练命令字符串（不含 CUDA_VISIBLE_DEVICES，由调用方设置）
# ============================================================================
build_train_cmd() {
    local experiment="$1" model_type="$2"

    local config_name
    config_name=$(get_config_name "$model_type" "$experiment")

    local cmd="$CONDA_PYTHON $SCRIPT_DIR/scripts/train.py"
    cmd="$cmd --config-name=${config_name}"
    cmd="$cmd device=${DEVICE}"
    cmd="$cmd epochs=${EPOCHS}"
    cmd="$cmd swanlab.entity=${SWANLAB_ENTITY}"
    cmd="$cmd swanlab.project=${SWANLAB_PROJECT}"
    cmd="$cmd swanlab.mode=${SWANLAB_MODE}"

    # SwanLab run name: 模型-实验名[-后缀]
    local run_name="${model_type}-${experiment}"
    [[ -n "$SWANLAB_SUFFIX" ]] && run_name="${run_name}-${SWANLAB_SUFFIX}"
    cmd="$cmd +swanlab.run_name=${run_name}"

    [[ -n "$SWANLAB_GROUP" ]] && cmd="$cmd swanlab.group=${SWANLAB_GROUP}"
    [[ -n "$BATCH_SIZE" ]]    && cmd="$cmd data_loader_config.batch_size=${BATCH_SIZE}"
    [[ -n "$LEARNING_RATE" ]] && cmd="$cmd agent_config.lr=${LEARNING_RATE}"
    [[ -n "$SEED" ]]          && cmd="$cmd seed=${SEED}"
    [[ -n "$RESUME" ]]        && cmd="$cmd resume=${RESUME}"

    # MOTIF 专有参数（来自 motif_train.sh）
    if [[ "$model_type" == motif_* ]]; then
        cmd="$cmd method_name=motif"
        [[ -n "$NUM_MODES" ]] && cmd="$cmd agent_config.model_config.inner_model_config.motif_handler_config.num_modes=${NUM_MODES}"
        [[ -n "$ALPHA_VEL" ]] && cmd="$cmd agent_config.model_config.alpha_vel=${ALPHA_VEL}"
    fi

    [[ -n "$EXTRA_ARGS" ]] && cmd="$cmd $EXTRA_ARGS"

    echo "$cmd"
}

# ============================================================================
# 串行模式：执行单个训练任务
# ============================================================================
run_serial() {
    local experiment="$1" model_type="$2" gpu_ids="$3"

    local cmd
    cmd=$(build_train_cmd "$experiment" "$model_type")
    local full_cmd="CUDA_VISIBLE_DEVICES=${gpu_ids} ${cmd}"

    print_info "执行命令:"
    echo -e "${YELLOW}${full_cmd}${NC}"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        print_warning "Dry-run 模式，不执行训练"
        return 0
    fi

    if [[ "${AUTO_YES}" != "1" ]]; then
        read -p "是否开始训练? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
            print_warning "训练已取消"
            return 0
        fi
    else
        print_info "自动确认训练 (--auto-yes)"
    fi

    export CUDA_VISIBLE_DEVICES="${gpu_ids}"
    # 取第一个 GPU 编号给 EGL（PyBullet 渲染器只用一张卡）
    local first_gpu="${gpu_ids%%,*}"
    export EGL_VISIBLE_DEVICES="${first_gpu}"
    eval "$cmd"
    local exit_code=$?

    if (( exit_code == 0 )); then
        print_header "训练完成"
        print_info "模型: $(get_model_description "$model_type")"
        print_info "实验: ${experiment}"
    else
        print_error "训练失败 (exit=${exit_code})"
        exit "$exit_code"
    fi
}

# ============================================================================
# 并行模式：执行单个 job（含重试），借鉴 train_sofa_tasks.sh
# ============================================================================
_run_job() {
    local gpu="$1" task="$2" model="$3"
    local attempt=0 status=0

    # 在当前 worker subshell 里绑定 GPU
    # CUDA_VISIBLE_DEVICES: 控制 PyTorch 使用的 GPU（逻辑编号）
    # EGL_VISIBLE_DEVICES:  控制 PyBullet EGL 渲染器使用的 GPU（物理编号）
    # 两者必须同步，否则 PyBullet 渲染 worker 会全部堆在物理 GPU 0 上
    export CUDA_VISIBLE_DEVICES="$gpu"
    export EGL_VISIBLE_DEVICES="$gpu"

    local cmd
    cmd=$(build_train_cmd "$task" "$model")

    while (( attempt <= MAX_RETRIES )); do
        attempt=$(( attempt + 1 ))
        echo " [GPU ${gpu}] 启动 ${task} / ${model}（第 ${attempt} 次尝试）"

        eval "$cmd"
        status=$?

        if (( status == 0 )); then
            echo " [GPU ${gpu}] ✓ ${task} / ${model} 完成"
            return 0
        elif (( attempt <= MAX_RETRIES )); then
            echo " [GPU ${gpu}] ! ${task} / ${model} 失败 (exit=${status})，15 秒后重试..."
            sleep 15
        fi
    done

    echo " [GPU ${gpu}] ✗ ${task} / ${model} 最终失败 (exit=${status}，共 ${attempt} 次)"
    return "$status"
}

# ============================================================================
# 并行模式：GPU worker，从文件队列取 job 直到队列为空
# 设计来自 train_sofa_tasks.sh
# ============================================================================
_gpu_worker() {
    local gpu="$1" worker_id="$2"
    echo " [Worker ${worker_id} - GPU ${gpu}] 工作进程启动"

    while true; do
        local job="" tmp_job="/tmp/baseline_worker_${worker_id}_${_RUN_ID}"

        # 文件锁：原子地从队列头取一个 job
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

        if [[ -f "$tmp_job" ]]; then
            job=$(cat "$tmp_job")
            rm -f "$tmp_job"
        else
            job=""
        fi

        if [[ -z "$job" ]]; then
            echo " [Worker ${worker_id} - GPU ${gpu}] 队列已空，退出"
            break
        fi

        local task="${job%%|*}" model="${job##*|}"
        local remaining
        remaining=$(cat "$_QSIZE_FILE" 2>/dev/null || echo "?")
        echo " [Worker ${worker_id} - GPU ${gpu}] 取到 job: ${task} / ${model}（队列剩余: ${remaining}）"

        # 非首个 worker 错开启动，避免并发初始化竞态
        if (( worker_id > 1 )); then
            sleep $(( (worker_id - 1) * 5 ))
        fi

        _run_job "$gpu" "$task" "$model"
    done

    echo " [Worker ${worker_id} - GPU ${gpu}] 工作进程结束"
}

# ============================================================================
# 并行模式：主控函数
# ============================================================================
run_parallel() {
    local experiment="$1"
    shift
    local models=("$@")

    IFS=',' read -ra GPU_LIST <<< "$GPU_IDS"
    local num_gpus=${#GPU_LIST[@]}

    # 临时文件（PID + 随机后缀保证唯一性）
    local rnd
    rnd=$(echo $RANDOM | md5sum | cut -c1-6)
    _RUN_ID="${$}_${rnd}"
    _QUEUE_FILE="/tmp/baseline_queue_${_RUN_ID}"
    _LOCK_FILE="/tmp/baseline_lock_${_RUN_ID}"
    _QSIZE_FILE="/tmp/baseline_qsize_${_RUN_ID}"

    # 构建任务队列
    > "$_QUEUE_FILE"
    local model
    for model in "${models[@]}"; do
        echo "${experiment}|${model}" >> "$_QUEUE_FILE"
    done
    local num_jobs
    num_jobs=$(wc -l < "$_QUEUE_FILE")
    echo "$num_jobs" > "$_QSIZE_FILE"

    echo "========================================================"
    echo " Baseline 并行训练启动"
    echo "========================================================"
    printf " 实验       : %s\n" "$experiment"
    printf " 模型列表   : %s\n" "${models[*]}"
    printf " 总 job 数  : %s\n" "$num_jobs"
    printf " GPU 列表   : %s（%s 个 worker）\n" "${GPU_LIST[*]}" "$num_gpus"
    printf " 最大重试   : %s 次\n" "$MAX_RETRIES"
    printf " 训练轮数   : %s\n" "$EPOCHS"
    echo "--------------------------------------------------------"
    echo " 队列顺序:"
    awk -F'|' '{printf "   %-35s → %s\n", $1, $2}' "$_QUEUE_FILE"
    echo "--------------------------------------------------------"

    if [[ "$DRY_RUN" == true ]]; then
        print_warning "Dry-run 模式 - 预览各 job 命令:"
        while IFS='|' read -r task model; do
            echo ""
            print_info "${task} / ${model}:"
            echo -e "${YELLOW}  CUDA_VISIBLE_DEVICES=<GPU> $(build_train_cmd "$task" "$model")${NC}"
        done < "$_QUEUE_FILE"
        rm -f "$_QUEUE_FILE" "$_LOCK_FILE" "$_QSIZE_FILE"
        return 0
    fi

    # 后台进程 PID 表（gpu -> pid）
    declare -A _BG_PIDS

    # 信号处理：Ctrl-C / kill 时优雅退出
    _parallel_cleanup() {
        echo ""
        echo "[SIGNAL] 捕获退出信号，终止所有训练进程..."
        local pid
        for pid in "${_BG_PIDS[@]}"; do
            if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
                kill -TERM "$pid" 2>/dev/null
                sleep 1
                kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
            fi
        done
        for pid in "${_BG_PIDS[@]}"; do
            [[ -n "$pid" ]] && wait "$pid" 2>/dev/null || true
        done
        rm -f "$_QUEUE_FILE" "$_LOCK_FILE" "$_QSIZE_FILE"
        echo "[SIGNAL] 完成。"
        exit 1
    }
    trap _parallel_cleanup SIGINT SIGTERM

    # 为每个 GPU 启动一个 worker
    echo " 启动 ${num_gpus} 个 GPU 工作进程..."
    local idx
    for (( idx=0; idx<num_gpus; idx++ )); do
        local gpu
        gpu="$(echo "${GPU_LIST[$idx]}" | tr -d '[:space:]')"
        local wid=$(( idx + 1 ))
        _gpu_worker "$gpu" "$wid" &
        _BG_PIDS[$gpu]=$!
    done
    echo " 所有工作进程已启动，PID: ${_BG_PIDS[*]}"
    echo "========================================================"

    # 等待所有 worker 结束，汇报结果
    local failed=()
    local gpu
    for gpu in "${!_BG_PIDS[@]}"; do
        wait "${_BG_PIDS[$gpu]}"
        local code=$?
        (( code != 0 )) && failed+=("GPU${gpu}(exit=${code})")
    done

    trap - SIGINT SIGTERM
    rm -f "$_QUEUE_FILE" "$_LOCK_FILE" "$_QSIZE_FILE"

    echo "========================================================"
    if [[ ${#failed[@]} -eq 0 ]]; then
        echo " 全部任务完成！"
    else
        echo " 以下工作进程异常退出: ${failed[*]}"
        echo " 请查看 ${SCRIPT_DIR}/results/ 下对应的 train.log 排查原因。"
        exit 1
    fi
    echo "========================================================"
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    print_header "Baseline 统一训练脚本"

    # 位置参数：[experiment] [model_type]（非 - 开头视为位置参数）
    EXPERIMENT="$DEFAULT_EXPERIMENT"
    MODEL_TYPE="$DEFAULT_MODEL_TYPE"
    MODELS_LIST=""

    if [[ $# -gt 0 && "$1" != -* ]]; then
        EXPERIMENT="$1"; shift
    fi
    if [[ $# -gt 0 && "$1" != -* ]]; then
        MODEL_TYPE="$1"; shift
    fi

    # 选项默认值：命令行 > 顶部快速配置区 > 环境变量
    DEVICE="$DEFAULT_DEVICE"
    GPU_IDS="$GPUS"          # 来自顶部快速配置区
    EPOCHS="$DEFAULT_EPOCHS"
    SWANLAB_MODE="$DEFAULT_SWANLAB_MODE"
    SWANLAB_GROUP=""
    SWANLAB_SUFFIX=""
    BATCH_SIZE=""
    LEARNING_RATE=""
    NUM_MODES=""
    ALPHA_VEL=""
    SEED=""
    RESUME=""
    # MAX_RETRIES 已在顶部快速配置区定义，此处不覆盖
    DRY_RUN=false
    AUTO_YES="${AUTO_YES:-0}"
    EXTRA_ARGS=""

    # 选项解析（命令行参数优先级最高，覆盖顶部配置区）
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -m|--models)          MODELS_LIST="$2";       shift 2 ;;
            -g|--gpus)            GPU_IDS="$2";            shift 2 ;;
               --gpu)             GPU_IDS="$2";            shift 2 ;;
            -e|--epochs)          EPOCHS="$2";             shift 2 ;;
            -d|--device)          DEVICE="$2";             shift 2 ;;
               --swanlab-mode)    SWANLAB_MODE="$2";       shift 2 ;;
               --swanlab-entity)  SWANLAB_ENTITY="$2";     shift 2 ;;
               --swanlab-project) SWANLAB_PROJECT="$2";    shift 2 ;;
               --swanlab-group)   SWANLAB_GROUP="$2";      shift 2 ;;
               --suffix)          SWANLAB_SUFFIX="$2";     shift 2 ;;
               --batch-size)      BATCH_SIZE="$2";         shift 2 ;;
               --lr)              LEARNING_RATE="$2";      shift 2 ;;
               --num-modes)       NUM_MODES="$2";          shift 2 ;;
               --alpha-vel)       ALPHA_VEL="$2";          shift 2 ;;
               --seed)            SEED="$2";               shift 2 ;;
               --resume)          RESUME="$2";             shift 2 ;;
            -r|--retries)         MAX_RETRIES="$2";        shift 2 ;;
               --dry-run)         DRY_RUN=true;            shift   ;;
               --auto-yes)        AUTO_YES=1;              shift   ;;
            -h|--help)            show_usage; exit 0       ;;
            *)                    EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
        esac
    done

    if [[ "$EXPERIMENT" == "-h" || "$EXPERIMENT" == "--help" ]]; then
        show_usage; exit 0
    fi

    # 检查训练环境
    print_info "检查训练环境..."
    check_conda_env
    check_gpu || true
    check_cuda || DEVICE="cpu"

    # 未通过命令行 --models 指定时，回落到顶部快速配置区的 ALGOS
    # 当 ALGOS 包含多个模型时自动进入并行模式；只有一个模型时进入串行模式
    if [[ -z "$MODELS_LIST" ]]; then
        local algo_count
        algo_count=$(echo "$ALGOS" | tr ',' '\n' | grep -c .)
        if (( algo_count > 1 )); then
            MODELS_LIST="$ALGOS"
        fi
        # algo_count == 1 时保持串行，MODEL_TYPE 已由 DEFAULT_MODEL_TYPE 设置
    fi

    # 根据 --models 决定串行还是并行模式
    if [[ -n "$MODELS_LIST" ]]; then
        # ── 并行模式 ──────────────────────────────────────────────────────────
        IFS=',' read -ra MODELS_ARRAY <<< "$MODELS_LIST"

        print_header "并行训练配置"
        print_config "实验: ${EXPERIMENT}"
        print_config "模型列表: ${MODELS_ARRAY[*]}"
        print_config "GPU 列表: ${GPU_IDS}"
        print_config "训练轮数: ${EPOCHS}"
        print_config "最大重试: ${MAX_RETRIES}"
        print_config "SwanLab 模式: ${SWANLAB_MODE}"
        [[ -n "$SWANLAB_GROUP" ]] && print_config "SwanLab 组: ${SWANLAB_GROUP}"
        [[ -n "$NUM_MODES" ]]     && print_config "MOTIF 模态数: ${NUM_MODES}"
        [[ -n "$ALPHA_VEL" ]]     && print_config "MOTIF 速度权重: ${ALPHA_VEL}"
        [[ -n "$SEED" ]]          && print_config "随机种子: ${SEED}"
        echo ""

        run_parallel "$EXPERIMENT" "${MODELS_ARRAY[@]}"
    else
        # ── 串行模式 ──────────────────────────────────────────────────────────
        local model_desc
        model_desc=$(get_model_description "$MODEL_TYPE")

        print_header "串行训练配置"
        print_config "实验: ${EXPERIMENT}"
        print_config "模型: ${MODEL_TYPE} (${model_desc})"
        print_config "GPU / CUDA_VISIBLE_DEVICES: ${GPU_IDS}"
        print_config "设备: ${DEVICE}"
        print_config "训练轮数: ${EPOCHS}"
        print_config "SwanLab 模式: ${SWANLAB_MODE}"
        [[ -n "$SWANLAB_GROUP" ]] && print_config "SwanLab 组: ${SWANLAB_GROUP}"
        [[ -n "$NUM_MODES" ]]     && print_config "MOTIF 模态数: ${NUM_MODES}"
        [[ -n "$ALPHA_VEL" ]]     && print_config "MOTIF 速度权重: ${ALPHA_VEL}"
        [[ -n "$SEED" ]]          && print_config "随机种子: ${SEED}"
        [[ -n "$RESUME" ]]        && print_config "从 checkpoint 恢复: ${RESUME}"
        [[ -n "$EXTRA_ARGS" ]]    && print_config "额外参数: ${EXTRA_ARGS}"
        echo ""

        run_serial "$EXPERIMENT" "$MODEL_TYPE" "$GPU_IDS"
    fi
}

main "$@"
