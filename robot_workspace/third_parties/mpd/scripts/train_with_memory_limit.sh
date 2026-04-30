#!/bin/bash
# 带内存限制的训练启动器

MEMORY_LIMIT="60G"  # 设置最大内存限制
SCRIPT="${1:-train.py}"
shift

echo "启动训练（内存限制: $MEMORY_LIMIT）"

# 方法1: 使用systemd-run（推荐，需要sudo）
# sudo systemd-run --scope -p MemoryMax=$MEMORY_LIMIT python3 $SCRIPT "$@"

# 方法2: 使用ulimit（当前shell有效）
ulimit -v $((60*1024*1024))  # 60GB in KB
ulimit -a | grep "virtual memory"

# 方法3: 使用cgroups v2（手动）
# echo $$ | sudo tee /sys/fs/cgroup/my_training/cgroup.procs
# echo "$MEMORY_LIMIT" | sudo tee /sys/fs/cgroup/my_training/memory.max

python3 $SCRIPT "$@"
