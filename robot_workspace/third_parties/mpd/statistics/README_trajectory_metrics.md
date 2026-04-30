# 轨迹指标计算与可视化工具

## 功能说明

这两个脚本用于计算和可视化机器人轨迹的平滑性指标：

- **Jerk (加加速度/急动度)**: 衡量加速度变化的剧烈程度，越小越平滑
- **Energy (能量)**: 加速度的累积，越小越经济
- **Path Length (路径长度)**: 轨迹的总长度

## 使用方法

### 1. 计算轨迹指标

```bash
# 计算所有方法的所有epoch的指标
python statistics/compute_trajectory_metrics.py

# 只计算特定方法
python statistics/compute_trajectory_metrics.py --methods mpd dp_transformer fm_transformer

# 自定义时间步长 (默认 1/30)
python statistics/compute_trajectory_metrics.py --dt 0.033

# 保存汇总结果
python statistics/compute_trajectory_metrics.py --save_summary --output my_summary.json
```

**输出**:
- 在每个 `epoch_XXXX/` 文件夹下生成 `trajectory_metrics.json`
- 可选: 在 `_results/` 下生成汇总文件

### 2. 可视化指标

```bash
# 可视化所有方法的指标对比
python statistics/visualize_trajectory_metrics.py

# 只可视化特定方法
python statistics/visualize_trajectory_metrics.py --methods mpd dp_transformer

# 自定义输出路径
python statistics/visualize_trajectory_metrics.py --output_dir /path/to/output

# 只生成对比图，不生成单方法图
python statistics/visualize_trajectory_metrics.py --no_per_method

# 只生成单方法图，不生成对比图
python statistics/visualize_trajectory_metrics.py --no_comparison
```

**输出**:
- `trajectory_metrics_plots/comparison/`: 不同方法的指标对比图
- `trajectory_metrics_plots/per_method/`: 每个方法的多指标图
- `trajectory_metrics_plots/final_metrics_summary.json`: 最终epoch的指标汇总

## 指标说明

### trajectory_metrics.json 格式

```json
{
  "epoch": 1500,
  "method": "dp_transformer",
  "date": "2026-04-15",
  "time": "01-06-17",
  "total_episodes": 40,
  "success_rate": 0.375,
  
  // 所有轨迹的指标
  "all_mean_jerk": 245.67,
  "all_std_jerk": 89.23,
  "all_mean_energy": 1234.56,
  "all_std_energy": 456.78,
  "all_mean_path_length": 0.8543,
  "all_std_path_length": 0.1234,
  
  // 成功轨迹的指标
  "success_mean_jerk": 198.45,
  "success_std_jerk": 67.89,
  "success_mean_energy": 987.65,
  "success_std_energy": 345.67,
  "success_mean_path_length": 0.7821,
  "success_std_path_length": 0.0987,
  
  // 失败轨迹的指标
  "failed_mean_jerk": 276.89,
  "failed_std_jerk": 98.76,
  "failed_mean_energy": 1456.78,
  "failed_std_energy": 567.89,
  "failed_mean_path_length": 0.9012,
  "failed_std_path_length": 0.1456
}
```

## 指标解释

### Jerk (加加速度)
- **定义**: 加速度对时间的导数，即 `d³x/dt³`
- **意义**: 衡量运动的平滑程度
- **越小越好**: 低 Jerk 表示轨迹平滑，运动舒适
- **单位**: m/s³ (米/秒³)
- **应用**: 机器人轨迹规划中常用的平滑性指标，基于人类运动的 minimum jerk 原理

### Energy (能量)
- **定义**: 加速度幅值的累积和，即 `Σ||a(t)||`
- **意义**: 衡量运动的"努力程度"
- **越小越好**: 低能量表示运动效率高
- **单位**: m/s² (米/秒²)
- **应用**: 反映轨迹执行的能量消耗

### Path Length (路径长度)
- **定义**: 轨迹在空间中的总长度，即 `Σ||x(t+1) - x(t)||`
- **意义**: 衡量轨迹的直接性
- **评价**: 取决于任务，有时较短表示更直接，但也可能表示避障不够
- **单位**: m (米)
- **应用**: 评估轨迹效率

## 示例输出

### 对比图
- `all_mean_jerk.png`: 各方法的平均 Jerk 对比
- `all_mean_energy.png`: 各方法的平均能量对比
- `all_mean_path_length.png`: 各方法的平均路径长度对比
- `success_mean_jerk.png`: 成功轨迹的平均 Jerk 对比
- `success_rate.png`: 各方法的成功率对比

### 单方法图
- `mpd_metrics.png`: mpd 方法的所有指标随 epoch 变化
- `dp_transformer_metrics.png`: dp_transformer 方法的所有指标随 epoch 变化

## 注意事项

1. **时间步长**: 确保使用正确的 `dt` 参数（默认 1/30 秒）
2. **数据完整性**: 需要先运行训练生成 episode JSON 文件
3. **计算时间**: 处理大量 epoch 可能需要几分钟
4. **NaN 处理**: 如果某个 epoch 没有成功轨迹，相关指标会是 NaN

## 依赖包

```bash
pip install numpy matplotlib tqdm
```

## 常见问题

### Q: 为什么某些指标是 NaN？
A: 当某个类别（成功/失败）的轨迹数量为 0 时，相关指标无法计算。

### Q: 可以只计算特定 epoch 吗？
A: 目前脚本会处理所有找到的 epoch。如需筛选，可以修改脚本或手动删除不需要的 epoch 文件夹。

### Q: 如何比较不同方法的最终性能？
A: 查看 `final_metrics_summary.json` 或运行可视化脚本查看对比图。

## 扩展

如需添加其他指标（如曲率、角度变化等），可以在 `compute_trajectory_metrics.py` 的 `process_trajectory()` 函数中添加计算逻辑。
