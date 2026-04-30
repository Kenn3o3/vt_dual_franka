#!/bin/bash
# MOTIF v3.2 训练脚本
# 用法: bash motif_train.sh [experiment] [additional_args...]

set -e  # 遇到错误立即退出

# ============================================================================
# 配置部分
# ============================================================================
export PYBULLET_EGL=1

# 默认配置
DEFAULT_EXPERIMENT="obstacle_avoidance"
DEFAULT_DEVICE="cuda"
DEFAULT_GPU_IDS="0"
DEFAULT_EPOCHS=3000
DEFAULT_SWANLAB_MODE="cloud"

# SwanLab 配置
SWANLAB_ENTITY="motif"
SWANLAB_PROJECT="movement-primitive-diffusion"

# 环境配置
CONDA_ENV="mpd"

# ============================================================================
# 颜色输出
# ============================================================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 辅助函数
# ============================================================================

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_conda_env() {
    if ! conda env list | grep -q "^${CONDA_ENV} "; then
        print_error "Conda 环境 '${CONDA_ENV}' 不存在！"
        print_info "请先创建环境: conda create -n ${CONDA_ENV} python=3.10"
        exit 1
    fi
}

check_gpu() {
    if ! nvidia-smi &> /dev/null; then
        print_warning "未检测到 NVIDIA GPU"
        return 1
    fi
    
    local gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    print_info "检测到 ${gpu_count} 个 GPU"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    return 0
}

check_cuda() {
    print_info "检查 CUDA 状态..."
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
    local cuda_available=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | tail -1)
    
    if [ "$cuda_available" = "True" ]; then
        print_info "✅ PyTorch CUDA 可用"
        local device_count=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null | tail -1)
        print_info "可用 GPU 数量: ${device_count}"
        return 0
    else
        print_warning "⚠️  PyTorch CUDA 不可用"
        print_warning "将使用 CPU 模式训练（速度较慢）"
        return 1
    fi
}

show_usage() {
    cat << EOF
${GREEN}MOTIF v3.2 训练脚本${NC}

用法:
    bash motif_train.sh [experiment] [options]

参数:
    experiment    实验类型 (默认: ${DEFAULT_EXPERIMENT})

可用实验:
    - obstacle_avoidance            (避障任务)

选项:
    --device DEVICE              设备 (cuda/cpu, 默认: ${DEFAULT_DEVICE})
    --epochs N                   训练轮数 (默认: ${DEFAULT_EPOCHS})
    --gpu ID                     指定 GPU ID (例如: 0,1,2,3)
    --swanlab-mode MODE         SwanLab 模式 (cloud/local/disabled, 默认: ${DEFAULT_SWANLAB_MODE})
    --swanlab-entity ENTITY     SwanLab workspace (默认: ${SWANLAB_ENTITY})
    --swanlab-project PROJECT   SwanLab 项目名 (默认: ${SWANLAB_PROJECT})
    --swanlab-group GROUP       实验组名
    --batch-size N              批次大小
    --lr RATE                   学习率
    --num-modes M               傅里叶模态数 (默认: 16)
    --alpha-vel ALPHA           速度损失权重 (默认: 1.0)
    --seed N                    随机种子
    --resume PATH               从检查点恢复
    --dry-run                   仅显示命令，不执行
    --help, -h                  显示此帮助信息

示例:
    # 基础训练
    bash motif_train.sh

    # 指定实验
    bash motif_train.sh obstacle_avoidance

    # 使用特定 GPU
    bash motif_train.sh --gpu 0

    # 本地模式（不上传云端）
    bash motif_train.sh --swanlab-mode local

    # 完整配置
    bash motif_train.sh obstacle_avoidance \\
        --device cuda \\
        --gpu 0 \\
        --epochs 3000 \\
        --batch-size 256 \\
        --num-modes 16 \\
        --alpha-vel 1.0 \\
        --swanlab-mode cloud \\
        --swanlab-group motif-baseline

MOTIF v3.2 核心特性:
    ✓ 物理时间编码 (秒，非步骤索引)
    ✓ 傅里叶系数空间 (DCT-II)
    ✓ 状态条件查询 (t-conditioned masking)
    ✓ 速度监督损失 (L_FM + α·L_vel)

EOF
}

# ============================================================================
# 主程序
# ============================================================================

main() {
    print_header "MOTIF v3.2 训练"
    
    # 解析参数
    EXPERIMENT="${1:-$DEFAULT_EXPERIMENT}"
    shift 2>/dev/null || true
    
    # 默认值(优先使用环境变量)
    DEVICE="${DEVICE:-$DEFAULT_DEVICE}"
    EPOCHS="${EPOCHS:-$DEFAULT_EPOCHS}"
    GPU_IDS="${GPU_IDS:-$DEFAULT_GPU_IDS}"
    SWANLAB_MODE="${SWANLAB_MODE:-$DEFAULT_SWANLAB_MODE}"
    SWANLAB_GROUP=""
    BATCH_SIZE=""
    LEARNING_RATE=""
    NUM_MODES=""
    ALPHA_VEL=""
    SEED=""
    RESUME=""
    DRY_RUN=false
    EXTRA_ARGS=""
    
    # 解析选项
    while [[ $# -gt 0 ]]; do
        case $1 in
            --device)
                DEVICE="$2"
                shift 2
                ;;
            --epochs)
                EPOCHS="$2"
                shift 2
                ;;
            --gpu)
                GPU_IDS="$2"
                shift 2
                ;;
            --swanlab-mode)
                SWANLAB_MODE="$2"
                shift 2
                ;;
            --swanlab-entity)
                SWANLAB_ENTITY="$2"
                shift 2
                ;;
            --swanlab-project)
                SWANLAB_PROJECT="$2"
                shift 2
                ;;
            --swanlab-group)
                SWANLAB_GROUP="$2"
                shift 2
                ;;
            --batch-size)
                BATCH_SIZE="$2"
                shift 2
                ;;
            --lr)
                LEARNING_RATE="$2"
                shift 2
                ;;
            --num-modes)
                NUM_MODES="$2"
                shift 2
                ;;
            --alpha-vel)
                ALPHA_VEL="$2"
                shift 2
                ;;
            --seed)
                SEED="$2"
                shift 2
                ;;
            --resume)
                RESUME="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                EXTRA_ARGS="$EXTRA_ARGS $1"
                shift
                ;;
        esac
    done
    
    # 检查环境
    print_info "检查运行环境..."
    check_conda_env
    
    if [ "$DEVICE" = "cuda" ]; then
        check_gpu
        check_cuda || DEVICE="cpu"
    fi
    
    # 构建训练命令
    print_info "配置训练参数..."
    
    # 根据USE_FLOW_MATCHING选择配置文件
    if [ "${USE_FLOW_MATCHING:-0}" = "1" ]; then
        CONFIG_NAME="train_motif_fm_transformer"
        print_info "使用Flow Matching配置"
    else
        CONFIG_NAME="train_motif_transformer"
        print_info "使用Diffusion配置"
    fi
    
    CMD="python scripts/train.py"
    CMD="$CMD --config-name=experiments/${EXPERIMENT}/${CONFIG_NAME}"
    
    # 基础参数
    CMD="$CMD method_name=motif"  # 重要：设置保存路径为results/motif/
    CMD="$CMD device=${DEVICE}"
    CMD="$CMD epochs=${EPOCHS}"
    
    # SwanLab 配置
    CMD="$CMD swanlab.entity=${SWANLAB_ENTITY}"
    CMD="$CMD swanlab.project=${SWANLAB_PROJECT}"
    CMD="$CMD swanlab.mode=${SWANLAB_MODE}"
    CMD="$CMD +swanlab.run_name=motif-transformer-${EXPERIMENT}"
    
    if [ -n "$SWANLAB_GROUP" ]; then
        CMD="$CMD swanlab.group=${SWANLAB_GROUP}"
    fi
    
    # 可选参数
    if [ -n "$BATCH_SIZE" ]; then
        CMD="$CMD data_loader_config.batch_size=${BATCH_SIZE}"
    fi
    
    if [ -n "$LEARNING_RATE" ]; then
        CMD="$CMD agent_config.lr=${LEARNING_RATE}"
    fi
    
    if [ -n "$NUM_MODES" ]; then
        CMD="$CMD agent_config.model_config.inner_model_config.motif_handler_config.num_modes=${NUM_MODES}"
    fi
    
    if [ -n "$ALPHA_VEL" ]; then
        CMD="$CMD agent_config.model_config.alpha_vel=${ALPHA_VEL}"
    fi
    
    if [ -n "$SEED" ]; then
        CMD="$CMD seed=${SEED}"
    fi
    
    if [ -n "$RESUME" ]; then
        CMD="$CMD resume=${RESUME}"
    fi
    
    # 额外参数
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi
    
    # 设置 GPU
    if [ -n "$GPU_IDS" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_IDS"
        print_info "使用 GPU: $GPU_IDS"
    fi
    
    # 显示配置
    print_header "训练配置"
    echo -e "${GREEN}实验:${NC}        $EXPERIMENT"
    echo -e "${GREEN}算法:${NC}        MOTIF v3.2 Transformer"
    echo -e "${GREEN}设备:${NC}        $DEVICE"
    echo -e "${GREEN}训练轮数:${NC}    $EPOCHS"
    echo -e "${GREEN}SwanLab:${NC}     $SWANLAB_MODE ($SWANLAB_ENTITY/$SWANLAB_PROJECT)"
    if [ -n "$GPU_IDS" ]; then
        echo -e "${GREEN}GPU IDs:${NC}     $GPU_IDS"
    fi
    if [ -n "$BATCH_SIZE" ]; then
        echo -e "${GREEN}批次大小:${NC}    $BATCH_SIZE"
    fi
    if [ -n "$NUM_MODES" ]; then
        echo -e "${GREEN}傅里叶模态:${NC}  $NUM_MODES"
    fi
    if [ -n "$ALPHA_VEL" ]; then
        echo -e "${GREEN}速度损失权重:${NC} $ALPHA_VEL"
    fi
    echo ""
    
    # 显示完整命令
    print_info "执行命令:"
    echo -e "${YELLOW}$CMD${NC}"
    echo ""
    
    # 执行或仅显示
    if [ "$DRY_RUN" = true ]; then
        print_warning "Dry-run 模式，不执行训练"
        exit 0
    fi
    
    # 确认执行(如果设置了AUTO_YES环境变量则跳过)
    if [ "${AUTO_YES:-0}" != "1" ]; then
        read -p "是否开始训练? [Y/n] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]] && [[ -n $REPLY ]]; then
            print_warning "训练已取消"
            exit 0
        fi
    else
        print_info "自动确认训练 (AUTO_YES=1)"
    fi
    
    # 开始训练
    print_header "开始训练 MOTIF v3.2"
    
    # 激活 conda 环境
    eval "$(conda shell.bash hook)"
    conda activate ${CONDA_ENV}
    
    print_info "训练日志将保存到results/motif目录下的train.log"
    echo ""
    
    # 执行训练命令
    eval $CMD
    EXIT_CODE=$?
    
    # 检查训练结果
    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        print_header "训练完成！"
        print_info "日志已保存到: results/motif/YYYY-MM-DD/HH-MM-SS/train.log"
        print_info "查看结果: https://swanlab.cn/@${SWANLAB_ENTITY}/${SWANLAB_PROJECT}"
    else
        echo ""
        print_error "训练失败！"
        print_error "查看最新日志: ls -t results/motif/*/*/train.log | head -1"
        exit 1
    fi
}

# 运行主程序
main "$@"
