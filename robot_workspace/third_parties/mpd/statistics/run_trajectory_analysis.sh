#!/bin/bash
# 运行轨迹指标计算和可视化的便捷脚本

set -e  # 遇到错误时退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 获取脚本所在目录
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# 默认参数
CONDA_ENV="mpd"
METHODS=""
SKIP_COMPUTE=false
SKIP_VISUALIZE=false
OUTPUT_DIR="$PROJECT_DIR/_results/trajectory_metrics_plots"

# 帮助信息
show_help() {
    cat << EOF
用法: $0 [选项]

计算和可视化轨迹的平滑性指标 (Jerk, Energy, Path Length)

选项:
    -h, --help              显示此帮助信息
    -e, --env ENV           指定 conda 环境 (默认: mpd)
    -m, --methods M1 M2     只处理指定的方法 (默认: 所有方法)
    -o, --output DIR        输出目录 (默认: _results/trajectory_metrics_plots)
    --skip-compute          跳过计算步骤，只进行可视化
    --skip-visualize        跳过可视化步骤，只进行计算
    --compute-only          只计算，不可视化 (等同于 --skip-visualize)
    --visualize-only        只可视化，不计算 (等同于 --skip-compute)

示例:
    # 处理所有方法
    $0
    
    # 只处理特定方法
    $0 --methods mpd dp_transformer fm_transformer
    
    # 只计算不可视化
    $0 --compute-only
    
    # 只可视化已有数据
    $0 --visualize-only
    
    # 使用不同的conda环境
    $0 --env motif

EOF
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--env)
            CONDA_ENV="$2"
            shift 2
            ;;
        -m|--methods)
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^- ]]; do
                METHODS="$METHODS $1"
                shift
            done
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --skip-compute)
            SKIP_COMPUTE=true
            shift
            ;;
        --skip-visualize)
            SKIP_VISUALIZE=true
            shift
            ;;
        --compute-only)
            SKIP_VISUALIZE=true
            shift
            ;;
        --visualize-only)
            SKIP_COMPUTE=true
            shift
            ;;
        *)
            echo -e "${RED}错误: 未知选项 $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# 打印配置
echo -e "${GREEN}=== 轨迹指标分析 ===${NC}"
echo "项目目录: $PROJECT_DIR"
echo "Conda环境: $CONDA_ENV"
if [ -n "$METHODS" ]; then
    echo "处理方法:$METHODS"
else
    echo "处理方法: 所有方法"
fi
echo "输出目录: $OUTPUT_DIR"
echo ""

# 切换到项目目录
cd "$PROJECT_DIR"

# 步骤1: 计算指标
if [ "$SKIP_COMPUTE" = false ]; then
    echo -e "${YELLOW}步骤 1/2: 计算轨迹指标...${NC}"
    
    COMPUTE_CMD="python scripts/compute_trajectory_metrics.py --save_summary"
    if [ -n "$METHODS" ]; then
        COMPUTE_CMD="$COMPUTE_CMD --methods$METHODS"
    fi
    
    echo "运行命令: conda run -n $CONDA_ENV $COMPUTE_CMD"
    if conda run -n "$CONDA_ENV" $COMPUTE_CMD; then
        echo -e "${GREEN}✓ 指标计算完成${NC}"
    else
        echo -e "${RED}✗ 指标计算失败${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${YELLOW}跳过计算步骤${NC}"
    echo ""
fi

# 步骤2: 可视化
if [ "$SKIP_VISUALIZE" = false ]; then
    echo -e "${YELLOW}步骤 2/2: 生成可视化...${NC}"
    
    VIS_CMD="python scripts/visualize_trajectory_metrics.py --output_dir $OUTPUT_DIR"
    if [ -n "$METHODS" ]; then
        VIS_CMD="$VIS_CMD --methods$METHODS"
    fi
    
    echo "运行命令: conda run -n $CONDA_ENV $VIS_CMD"
    if conda run -n "$CONDA_ENV" $VIS_CMD; then
        echo -e "${GREEN}✓ 可视化完成${NC}"
    else
        echo -e "${RED}✗ 可视化失败${NC}"
        exit 1
    fi
    echo ""
else
    echo -e "${YELLOW}跳过可视化步骤${NC}"
    echo ""
fi

# 完成
echo -e "${GREEN}=== 分析完成! ===${NC}"
if [ "$SKIP_VISUALIZE" = false ]; then
    echo -e "结果保存在: ${GREEN}$OUTPUT_DIR${NC}"
    echo ""
    echo "主要输出文件:"
    echo "  - comparison/: 不同方法的对比图"
    echo "  - per_method/: 每个方法的详细指标图"
    echo "  - final_metrics_summary.json: 最终epoch的指标汇总"
fi
