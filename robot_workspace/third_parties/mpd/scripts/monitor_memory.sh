#!/bin/bash
# 实时监控内存使用

while true; do
    clear
    echo "=== 系统内存状态 $(date) ==="
    free -h
    echo ""
    echo "=== TOP 10 内存消耗进程 ==="
    ps aux --sort=-%mem | head -11
    echo ""
    echo "=== 内存压力指标 ==="
    echo -n "Swap使用: "; free | awk '/Swap/ {printf "%.1f%%\n", $3/$2*100}'
    echo -n "OOM Killer计数: "; dmesg | grep -c "Out of memory"
    echo -n "当前Swappiness: "; cat /proc/sys/vm/swappiness
    sleep 5
done
