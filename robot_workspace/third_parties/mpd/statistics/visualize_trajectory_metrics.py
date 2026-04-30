#!/usr/bin/env python3
"""
可视化轨迹指标随epoch的变化趋势
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse
from collections import defaultdict


def load_metrics(results_dir: Path, methods: List[str] = None) -> Dict:
    """
    加载所有方法的轨迹指标
    
    Args:
        results_dir: 结果文件夹路径
        methods: 要加载的方法列表
        
    Returns:
        metrics_dict: {method: {run_path: [metrics_list]}}
    """
    if methods is None:
        method_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    else:
        method_dirs = [results_dir / method for method in methods]
        method_dirs = [d for d in method_dirs if d.exists()]
    
    metrics_dict = defaultdict(lambda: defaultdict(list))
    
    for method_dir in method_dirs:
        method_name = method_dir.name
        
        # 查找所有 trajectory_metrics.json 文件
        for metrics_file in method_dir.rglob("trajectory_metrics.json"):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
            
            run_path = f"{metrics['date']}/{metrics['time']}"
            metrics_dict[method_name][run_path].append(metrics)
    
    # 按epoch排序
    for method in metrics_dict:
        for run_path in metrics_dict[method]:
            metrics_dict[method][run_path].sort(key=lambda x: x['epoch'])
    
    return metrics_dict


def plot_metrics_comparison(metrics_dict: Dict, output_dir: Path, 
                           metric_configs: List[Dict] = None):
    """
    绘制不同方法的指标对比图
    
    Args:
        metrics_dict: 指标字典
        output_dir: 输出文件夹
        metric_configs: 要绘制的指标配置
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if metric_configs is None:
        metric_configs = [
            # 原始指标（受轨迹长度影响）
            {'key': 'all_mean_jerk', 'title': 'Mean Jerk (Original)', 'ylabel': 'Jerk', 'lower_is_better': True},
            {'key': 'all_mean_energy', 'title': 'Mean Energy (Original)', 'ylabel': 'Energy', 'lower_is_better': True},
            {'key': 'all_mean_path_length', 'title': 'Mean Path Length', 'ylabel': 'Path Length (m)', 'lower_is_better': False},
            {'key': 'success_rate', 'title': 'Success Rate', 'ylabel': 'Success Rate', 'lower_is_better': False},
            # 归一化指标（更公平的比较）
            {'key': 'all_mean_jerk_per_step', 'title': 'Mean Jerk per Step (Normalized)', 'ylabel': 'Jerk/step', 'lower_is_better': True},
            {'key': 'all_energy_per_step', 'title': 'Energy per Step (Normalized)', 'ylabel': 'Energy/step', 'lower_is_better': True},
            {'key': 'all_energy_per_meter', 'title': 'Energy per Meter (Normalized)', 'ylabel': 'Energy/meter', 'lower_is_better': True},
            # 成功轨迹的归一化指标
            {'key': 'success_mean_jerk_per_step', 'title': 'Mean Jerk per Step (Success, Normalized)', 'ylabel': 'Jerk/step', 'lower_is_better': True},
            {'key': 'success_energy_per_step', 'title': 'Energy per Step (Success, Normalized)', 'ylabel': 'Energy/step', 'lower_is_better': True},
            # Mode熵值指标（多样性）
            {'key': 'all_mode_entropy', 'title': 'Mode Entropy (All Trajectories)', 'ylabel': 'Entropy [0-1]', 'lower_is_better': False},
            {'key': 'success_mode_entropy', 'title': 'Mode Entropy (Success Trajectories)', 'ylabel': 'Entropy [0-1]', 'lower_is_better': False},
            {'key': 'failed_mode_entropy', 'title': 'Mode Entropy (Failed Trajectories)', 'ylabel': 'Entropy [0-1]', 'lower_is_better': False},
        ]
    
    # 为每个指标绘制图表
    for config in metric_configs:
        metric_key = config['key']
        title = config['title']
        ylabel = config['ylabel']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # 绘制每个方法
        for method in sorted(metrics_dict.keys()):
            for run_path in metrics_dict[method]:
                metrics_list = metrics_dict[method][run_path]
                
                # 提取数据
                epochs = [m['epoch'] for m in metrics_list]
                values = []
                for m in metrics_list:
                    if metric_key in m and m[metric_key] is not None:
                        values.append(m[metric_key])
                    else:
                        values.append(np.nan)
                
                # 跳过全为NaN的数据
                if all(np.isnan(values)):
                    continue
                
                # 绘制曲线
                label = f"{method}"
                if len(metrics_dict[method]) > 1:
                    label += f" ({run_path.split('/')[0]})"
                
                ax.plot(epochs, values, marker='o', markersize=3, 
                       label=label, alpha=0.7, linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f"{metric_key}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_file}")
        plt.close()


def plot_multi_metrics_per_method(metrics_dict: Dict, output_dir: Path):
    """
    为每个方法绘制多指标对比图
    
    Args:
        metrics_dict: 指标字典
        output_dir: 输出文件夹
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for method in sorted(metrics_dict.keys()):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{method} - Trajectory Metrics (Normalized)', fontsize=16, fontweight='bold')
        
        metrics_to_plot = [
            ('all_mean_jerk_per_step', 'Jerk per Step', 'Jerk/step'),
            ('all_energy_per_step', 'Energy per Step', 'Energy/step'),
            ('all_energy_per_meter', 'Energy per Meter', 'Energy/m'),
            ('success_mean_jerk_per_step', 'Jerk per Step (Success)', 'Jerk/step'),
            ('success_energy_per_step', 'Energy per Step (Success)', 'Energy/step'),
            ('success_rate', 'Success Rate', 'Success Rate'),
        ]
        
        # 如果有熵值数据，使用包含熵值的布局
        has_entropy = any(
            'all_mode_entropy' in m 
            for run_metrics in metrics_dict[method].values() 
            for m in run_metrics
        )
        
        if has_entropy:
            fig, axes = plt.subplots(3, 3, figsize=(18, 15))
            fig.suptitle(f'{method} - Trajectory Metrics (Normalized + Entropy)', fontsize=16, fontweight='bold')
            metrics_to_plot = [
                ('all_mean_jerk_per_step', 'Jerk per Step', 'Jerk/step'),
                ('all_energy_per_step', 'Energy per Step', 'Energy/step'),
                ('all_energy_per_meter', 'Energy per Meter', 'Energy/m'),
                ('success_mean_jerk_per_step', 'Jerk per Step (Success)', 'Jerk/step'),
                ('success_energy_per_step', 'Energy per Step (Success)', 'Energy/step'),
                ('success_rate', 'Success Rate', 'Success Rate'),
                ('all_mode_entropy', 'Mode Entropy (All)', 'Entropy [0-1]'),
                ('success_mode_entropy', 'Mode Entropy (Success)', 'Entropy [0-1]'),
                ('failed_mode_entropy', 'Mode Entropy (Failed)', 'Entropy [0-1]'),
            ]
        
        for idx, (metric_key, title, ylabel) in enumerate(metrics_to_plot):
            rows = 3 if has_entropy else 2
            ax = axes[idx // 3, idx % 3] if has_entropy else axes[idx // 3, idx % 3]
            
            # 绘制每个运行
            for run_path in metrics_dict[method]:
                metrics_list = metrics_dict[method][run_path]
                
                epochs = [m['epoch'] for m in metrics_list]
                values = []
                for m in metrics_list:
                    if metric_key in m and m[metric_key] is not None:
                        values.append(m[metric_key])
                    else:
                        values.append(np.nan)
                
                if all(np.isnan(values)):
                    continue
                
                label = run_path.split('/')[0] if len(metrics_dict[method]) > 1 else None
                ax.plot(epochs, values, marker='o', markersize=4, 
                       label=label, alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Epoch', fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            if len(metrics_dict[method]) > 1:
                ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = output_dir / f"{method}_metrics.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_file}")
        plt.close()


def generate_summary_table(metrics_dict: Dict, output_dir: Path):
    """
    生成最终epoch的指标汇总表
    
    Args:
        metrics_dict: 指标字典
        output_dir: 输出文件夹
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集最后一个epoch的数据
    summary_data = []
    
    for method in sorted(metrics_dict.keys()):
        for run_path in metrics_dict[method]:
            metrics_list = metrics_dict[method][run_path]
            if len(metrics_list) == 0:
                continue
            
            # 获取最后一个epoch的数据
            last_metrics = metrics_list[-1]
            
            summary_data.append({
                'Method': method,
                'Run': run_path.split('/')[0],
                'Epoch': last_metrics['epoch'],
                'Success Rate': f"{last_metrics['success_rate']:.3f}",
                # 原始指标
                'Mean Jerk': f"{last_metrics.get('all_mean_jerk', np.nan):.4f}",
                'Mean Energy': f"{last_metrics.get('all_mean_energy', np.nan):.2f}",
                # 归一化指标（更公平）
                'Jerk/step': f"{last_metrics.get('all_mean_jerk_per_step', np.nan):.4f}",
                'Energy/step': f"{last_metrics.get('all_energy_per_step', np.nan):.4f}",
                'Energy/meter': f"{last_metrics.get('all_energy_per_meter', np.nan):.2f}",
                'Episode Length': f"{last_metrics.get('all_mean_episode_length', np.nan):.1f}",
                # Mode熵值指标
                'Mode Entropy (All)': f"{last_metrics.get('all_mode_entropy', np.nan):.4f}",
                'Mode Entropy (Success)': f"{last_metrics.get('success_mode_entropy', np.nan):.4f}",
                'Mode Entropy (Failed)': f"{last_metrics.get('failed_mode_entropy', np.nan):.4f}",
            })
    
    # 保存为JSON
    output_file = output_dir / "final_metrics_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"已保存: {output_file}")
    
    # 打印表格
    print("\n=== Final Epoch Metrics Summary ===")
    print("原始指标 (受轨迹长度影响):")
    print(f"{'Method':<20} {'Success':<10} {'Jerk':<12} {'Energy':<12} {'Ep.Len':<10}")
    print("-" * 80)
    for row in summary_data:
        print(f"{row['Method']:<20} {row['Success Rate']:<10} {row['Mean Jerk']:<12} "
              f"{row['Mean Energy']:<12} {row['Episode Length']:<10}")
    
    print("\n归一化指标 (消除轨迹长度影响，更公平):")
    print(f"{'Method':<20} {'Success':<10} {'Jerk/step':<12} {'Energy/step':<14} {'Energy/m':<12}")
    print("-" * 80)
    for row in summary_data:
        print(f"{row['Method']:<20} {row['Success Rate']:<10} {row['Jerk/step']:<12} "
              f"{row['Energy/step']:<14} {row['Energy/meter']:<12}")
    
    # 如果有熵值数据，打印熵值表
    if any('Mode Entropy (All)' in row for row in summary_data):
        print("\nMode熵值指标 (路径多样性，越高越好):")
        print(f"{'Method':<20} {'Success':<10} {'Entropy(All)':<15} {'Entropy(Success)':<18} {'Entropy(Failed)':<18}")
        print("-" * 90)
        for row in summary_data:
            print(f"{row['Method']:<20} {row['Success Rate']:<10} "
                  f"{row.get('Mode Entropy (All)', 'N/A'):<15} "
                  f"{row.get('Mode Entropy (Success)', 'N/A'):<18} "
                  f"{row.get('Mode Entropy (Failed)', 'N/A'):<18}")


def main():
    parser = argparse.ArgumentParser(description='可视化轨迹指标')
    parser.add_argument('--results_dir', type=str, default='/home/hasac_cover/gjn/mpd/results/obstacle_avoidance',
                       help='结果文件夹路径')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                       help='要可视化的方法列表 (默认: 所有方法)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出文件夹路径 (默认: results_dir/trajectory_metrics_plots)')
    parser.add_argument('--no_comparison', action='store_true',
                       help='不生成方法对比图')
    parser.add_argument('--no_per_method', action='store_true',
                       help='不生成每个方法的多指标图')
    parser.add_argument('--no_summary', action='store_true',
                       help='不生成汇总表')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if args.output_dir is None:
        output_dir = results_dir / "trajectory_metrics_plots"
    else:
        output_dir = Path(args.output_dir)
    
    print("正在加载指标数据...")
    metrics_dict = load_metrics(results_dir, args.methods)
    
    if len(metrics_dict) == 0:
        print("Error: 没有找到任何指标数据")
        print("请先运行 compute_trajectory_metrics.py 生成指标")
        return
    
    print(f"已加载 {len(metrics_dict)} 个方法的数据")
    for method in sorted(metrics_dict.keys()):
        print(f"  {method}: {len(metrics_dict[method])} 个实验运行")
    
    # 生成可视化
    if not args.no_comparison:
        print("\n生成方法对比图...")
        plot_metrics_comparison(metrics_dict, output_dir / "comparison")
    
    if not args.no_per_method:
        print("\n生成每个方法的多指标图...")
        plot_multi_metrics_per_method(metrics_dict, output_dir / "per_method")
    
    if not args.no_summary:
        print("\n生成汇总表...")
        generate_summary_table(metrics_dict, output_dir)
    
    print(f"\n所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
