#!/usr/bin/env python3
"""
计算各个方法各个epoch的轨迹指标：
- 平均 Jerk (加加速度/急动度)
- 平均能量 (Energy)
- 平均路径长度 (Path Length)
- Mode分布熵值 (Mode Entropy)

结果将保存在各个epoch文件夹下的 trajectory_metrics.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse
from tqdm import tqdm


def compute_velocity(positions: np.ndarray, dt: float = 1/30) -> np.ndarray:
    """
    计算速度
    
    Args:
        positions: shape (T, dim) - 位置序列
        dt: 时间步长
        
    Returns:
        velocity: shape (T-1, dim) - 速度序列
    """
    velocity = np.diff(positions, axis=0) / dt
    return velocity


def compute_acceleration(velocity: np.ndarray, dt: float = 1/30) -> np.ndarray:
    """
    计算加速度
    
    Args:
        velocity: shape (T, dim) - 速度序列
        dt: 时间步长
        
    Returns:
        acceleration: shape (T-1,) - 加速度幅值序列
    """
    accel_vec = np.diff(velocity, axis=0) / dt
    acceleration = np.linalg.norm(accel_vec, axis=-1)
    return acceleration


def compute_jerk(acceleration: np.ndarray, dt: float = 1/30) -> np.ndarray:
    """
    计算 Jerk (加加速度)
    
    Args:
        acceleration: shape (T,) - 加速度幅值序列
        dt: 时间步长
        
    Returns:
        jerk: shape (T-1,) - jerk 序列
    """
    jerk = np.abs(np.diff(acceleration)) / dt
    return jerk


def compute_path_length(positions: np.ndarray) -> float:
    """
    计算路径总长度
    
    Args:
        positions: shape (T, dim) - 位置序列
        
    Returns:
        path_length: float - 路径总长度
    """
    deltas = np.diff(positions, axis=0)
    distances = np.linalg.norm(deltas, axis=-1)
    path_length = np.sum(distances)
    return path_length


def compute_energy(acceleration: np.ndarray) -> float:
    """
    计算能量 (加速度的累积)
    
    Args:
        acceleration: shape (T,) - 加速度幅值序列
        
    Returns:
        energy: float - 总能量
    """
    energy = np.sum(acceleration)
    return energy


def compute_mode_from_trajectory(x: np.ndarray, y: np.ndarray, 
                                obs_lvl_origin: np.ndarray = np.array([0.5, -0.1, 0.0]),
                                obs_lvl_offset: np.ndarray = np.array([0.075, 0.18]),
                                obs_half_extend_y: float = 0.03) -> int:
    """
    从轨迹坐标重建mode
    
    Mode编码规则（9位二进制）:
    - [0:2]: level 1 (1个障碍物) - [左, 右]
    - [2:5]: level 2 (2个障碍物) - [上, 中, 下]  
    - [5:9]: level 3 (3个障碍物) - [上, 中上, 中下, 下]
    
    Args:
        x, y: 轨迹坐标数组
        obs_lvl_origin: 障碍物层原点
        obs_lvl_offset: 障碍物层偏移
        obs_half_extend_y: 障碍物y方向半径
        
    Returns:
        mode: mode的十进制编码 (0-511)
    """
    encoding = np.zeros(9, dtype=int)
    
    # 计算各层障碍物位置
    # Level 1: y=0
    l1_y = obs_lvl_origin[1] + 0 * obs_lvl_offset[1]
    l1_mid_x = obs_lvl_origin[0] + 0 * 0 * obs_lvl_offset[0]
    
    # Level 2: y=1
    l2_y = obs_lvl_origin[1] + 1 * obs_lvl_offset[1]
    l2_top_x = obs_lvl_origin[0] + (-1) * 1 * obs_lvl_offset[0]
    l2_bot_x = obs_lvl_origin[0] + 1 * 1 * obs_lvl_offset[0]
    
    # Level 3: y=2
    l3_y = obs_lvl_origin[1] + 2 * obs_lvl_offset[1]
    l3_top_x = obs_lvl_origin[0] + (-1) * 2 * obs_lvl_offset[0]
    l3_mid_x = obs_lvl_origin[0] + 0 * 2 * obs_lvl_offset[0]
    l3_bot_x = obs_lvl_origin[0] + 1 * 2 * obs_lvl_offset[0]
    
    l1_passed = False
    l2_passed = False
    l3_passed = False
    
    # 遍历轨迹，检测通过障碍物的方式
    for i in range(len(x)):
        eef_x, eef_y = x[i], y[i]
        
        # Level 1
        if not l1_passed:
            if abs(eef_y - l1_y) <= obs_half_extend_y:
                if eef_x < l1_mid_x:
                    encoding[0] = 1
                elif eef_x > l1_mid_x:
                    encoding[1] = 1
                l1_passed = True
        
        # Level 2
        if not l2_passed:
            if abs(eef_y - l2_y) <= obs_half_extend_y:
                if eef_x < l2_top_x:
                    encoding[2] = 1
                elif l2_top_x < eef_x < l2_bot_x:
                    encoding[3] = 1
                elif eef_x > l2_bot_x:
                    encoding[4] = 1
                l2_passed = True
        
        # Level 3
        if not l3_passed:
            if eef_y >= l3_y:
                if eef_x < l3_top_x:
                    encoding[5] = 1
                if l3_top_x < eef_x < l3_mid_x:
                    encoding[6] = 1
                elif l3_mid_x < eef_x < l3_bot_x:
                    encoding[7] = 1
                elif eef_x > l3_bot_x:
                    encoding[8] = 1
                l3_passed = True
        
        # 如果所有层都通过了，提前结束
        if l1_passed and l2_passed and l3_passed:
            break
    
    # 转换为十进制
    mode = int(encoding.dot(1 << np.arange(9)))
    return mode


def compute_mode_entropy(modes: List[int]) -> float:
    """
    计算mode分布的熵值
    
    Args:
        modes: mode列表
        
    Returns:
        entropy: 熵值 (归一化到[0,1]，以log(24)为基准)
    """
    if len(modes) == 0:
        return 0.0
    
    _, counts = np.unique(modes, return_counts=True)
    mode_dist = counts / np.sum(counts)
    entropy = -np.sum(mode_dist * (np.log(mode_dist) / np.log(24)))
    return float(entropy)


def compute_normalized_metrics(jerk: np.ndarray, acceleration: np.ndarray, 
                               path_length: float, episode_length: int) -> Dict:
    """
    计算归一化指标（消除轨迹长度的影响）
    
    Args:
        jerk: Jerk序列
        acceleration: 加速度序列
        path_length: 路径长度
        episode_length: episode长度（时间步数）
        
    Returns:
        normalized_metrics: 归一化指标字典
    """
    # 防止除零
    safe_length = max(episode_length, 1)
    safe_path_length = max(path_length, 1e-6)
    
    return {
        'mean_jerk_per_step': float(np.mean(jerk)),  # 每步平均Jerk
        'energy_per_step': float(np.sum(acceleration) / safe_length),  # 每步平均能量
        'energy_per_meter': float(np.sum(acceleration) / safe_path_length),  # 每米路径的能量
        'jerk_per_meter': float(np.mean(jerk) * safe_length / safe_path_length),  # 每米路径的Jerk
    }


def process_trajectory(traj_data: Dict, dt: float = 1/30) -> Dict:
    """
    处理单条轨迹并计算所有指标
    
    Args:
        traj_data: 轨迹数据字典
        dt: 时间步长
        
    Returns:
        metrics: 包含所有指标的字典
    """
    # 提取轨迹坐标
    x = np.array(traj_data['trajectory']['x'])
    y = np.array(traj_data['trajectory']['y'])
    positions = np.stack([x, y], axis=-1)  # shape: (T, 2)
    
    # 计算速度
    velocity = compute_velocity(positions, dt)
    
    # 计算加速度
    acceleration = compute_acceleration(velocity, dt)
    
    # 计算 jerk
    jerk = compute_jerk(acceleration, dt)
    
    # 计算路径长度
    path_length = compute_path_length(positions)
    
    # 计算能量
    energy = compute_energy(acceleration)
    
    # 计算归一化指标
    normalized = compute_normalized_metrics(jerk, acceleration, path_length, 
                                           traj_data['episode_length'])
    
    # 从轨迹重建mode（如果JSON中没有保存）
    if 'final_mode' in traj_data:
        final_mode = int(traj_data['final_mode'])
    else:
        final_mode = compute_mode_from_trajectory(x, y)
    
    # 返回指标
    metrics = {
        # 原始指标（总量）
        'mean_jerk': float(np.mean(jerk)),
        'std_jerk': float(np.std(jerk)),
        'max_jerk': float(np.max(jerk)),
        'energy': float(energy),
        'path_length': float(path_length),
        'episode_length': int(traj_data['episode_length']),
        'successful': bool(traj_data['successful']),
        'total_reward': float(traj_data.get('total_reward', 0)),
        'final_mode': final_mode,  # 从轨迹重建或读取
        # 归一化指标（消除长度影响）
        **normalized,
    }
    
    return metrics


def process_epoch(epoch_dir: Path, dt: float = 1/30) -> Dict:
    """
    处理单个epoch的所有轨迹
    
    Args:
        epoch_dir: epoch文件夹路径
        dt: 时间步长
        
    Returns:
        epoch_metrics: epoch级别的聚合指标
    """
    episodes_dir = epoch_dir / "episodes"
    
    if not episodes_dir.exists():
        return None
    
    # 查找所有episode JSON文件
    episode_files = sorted(episodes_dir.glob("*_episode_*.json"))
    
    if len(episode_files) == 0:
        return None
    
    all_metrics = []
    success_metrics = []
    failed_metrics = []
    
    # 处理每个episode
    for episode_file in episode_files:
        try:
            with open(episode_file, 'r') as f:
                traj_data = json.load(f)
            
            metrics = process_trajectory(traj_data, dt)
            all_metrics.append(metrics)
            
            if metrics['successful']:
                success_metrics.append(metrics)
            else:
                failed_metrics.append(metrics)
                
        except Exception as e:
            print(f"Warning: Failed to process {episode_file}: {e}")
            continue
    
    if len(all_metrics) == 0:
        return None
    
    # 聚合统计
    def aggregate_metrics(metrics_list: List[Dict], prefix: str = "") -> Dict:
        if len(metrics_list) == 0:
            return {}
        
        # 计算mode熵值（只对有mode信息的数据计算）
        modes = [m['final_mode'] for m in metrics_list if m.get('final_mode', -1) >= 0]
        mode_entropy = compute_mode_entropy(modes) if len(modes) > 0 else 0.0
        
        return {
            # 原始指标（受轨迹长度影响）
            f"{prefix}mean_jerk": float(np.mean([m['mean_jerk'] for m in metrics_list])),
            f"{prefix}std_jerk": float(np.std([m['mean_jerk'] for m in metrics_list])),
            f"{prefix}mean_energy": float(np.mean([m['energy'] for m in metrics_list])),
            f"{prefix}std_energy": float(np.std([m['energy'] for m in metrics_list])),
            f"{prefix}mean_path_length": float(np.mean([m['path_length'] for m in metrics_list])),
            f"{prefix}std_path_length": float(np.std([m['path_length'] for m in metrics_list])),
            f"{prefix}mean_episode_length": float(np.mean([m['episode_length'] for m in metrics_list])),
            # 归一化指标（消除轨迹长度影响，更公平）
            f"{prefix}mean_jerk_per_step": float(np.mean([m['mean_jerk_per_step'] for m in metrics_list])),
            f"{prefix}energy_per_step": float(np.mean([m['energy_per_step'] for m in metrics_list])),
            f"{prefix}energy_per_meter": float(np.mean([m['energy_per_meter'] for m in metrics_list])),
            f"{prefix}jerk_per_meter": float(np.mean([m['jerk_per_meter'] for m in metrics_list])),
            # Mode熵值指标
            f"{prefix}mode_entropy": mode_entropy,
            f"{prefix}count": len(metrics_list),
        }
    
    epoch_metrics = {
        'epoch': int(epoch_dir.name.replace('epoch_', '')),
        'total_episodes': len(all_metrics),
        'success_rate': len(success_metrics) / len(all_metrics) if len(all_metrics) > 0 else 0,
        **aggregate_metrics(all_metrics, prefix="all_"),
        **aggregate_metrics(success_metrics, prefix="success_"),
        **aggregate_metrics(failed_metrics, prefix="failed_"),
    }
    
    return epoch_metrics


def process_method(method_dir: Path, dt: float = 1/30, save_results: bool = True) -> List[Dict]:
    """
    处理单个方法的所有实验运行
    
    Args:
        method_dir: 方法文件夹路径
        dt: 时间步长
        save_results: 是否保存结果到各个epoch文件夹
        
    Returns:
        all_results: 所有epoch的指标列表
    """
    print(f"\n处理方法: {method_dir.name}")
    
    # 查找所有日期文件夹
    date_dirs = sorted([d for d in method_dir.iterdir() if d.is_dir()])
    
    all_results = []
    
    for date_dir in date_dirs:
        # 查找所有时间戳文件夹
        time_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()])
        
        for time_dir in time_dirs:
            print(f"  处理实验: {date_dir.name}/{time_dir.name}")
            
            # 查找所有epoch文件夹
            epoch_dirs = sorted([d for d in time_dir.iterdir() 
                               if d.is_dir() and d.name.startswith('epoch_')])
            
            if len(epoch_dirs) == 0:
                print(f"    Warning: 没有找到epoch文件夹")
                continue
            
            # 处理每个epoch
            for epoch_dir in tqdm(epoch_dirs, desc=f"    处理epochs", leave=False):
                epoch_metrics = process_epoch(epoch_dir, dt)
                
                if epoch_metrics is None:
                    continue
                
                # 添加元信息
                epoch_metrics['method'] = method_dir.name
                epoch_metrics['date'] = date_dir.name
                epoch_metrics['time'] = time_dir.name
                epoch_metrics['run_path'] = str(time_dir.relative_to(method_dir.parent.parent))
                
                all_results.append(epoch_metrics)
                
                # 保存到epoch文件夹
                if save_results:
                    output_file = epoch_dir / "trajectory_metrics.json"
                    with open(output_file, 'w') as f:
                        json.dump(epoch_metrics, f, indent=2)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='计算轨迹指标')
    parser.add_argument('--results_dir', type=str, default='/home/hasac_cover/gjn/mpd/results/obstacle_avoidance',
                       help='结果文件夹路径')
    parser.add_argument('--methods', type=str, nargs='+', default=None,
                       help='要处理的方法列表 (默认: 所有方法)')
    parser.add_argument('--dt', type=float, default=1/30,
                       help='时间步长 (默认: 1/30)')
    parser.add_argument('--save_summary', action='store_true',
                       help='是否保存汇总结果')
    parser.add_argument('--output', type=str, default='trajectory_metrics_summary.json',
                       help='汇总结果输出文件名')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    if not results_dir.exists():
        print(f"Error: 结果文件夹不存在: {results_dir}")
        return
    
    # 确定要处理的方法
    if args.methods is None:
        method_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    else:
        method_dirs = [results_dir / method for method in args.methods]
        method_dirs = [d for d in method_dirs if d.exists()]
    
    if len(method_dirs) == 0:
        print("Error: 没有找到要处理的方法文件夹")
        return
    
    # 处理所有方法
    all_results = []
    for method_dir in method_dirs:
        method_results = process_method(method_dir, dt=args.dt, save_results=True)
        all_results.extend(method_results)
    
    # 保存汇总结果
    if args.save_summary and len(all_results) > 0:
        output_path = results_dir / args.output
        with open(output_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n汇总结果已保存到: {output_path}")
        print(f"总共处理了 {len(all_results)} 个epoch")
    
    # 打印统计信息
    print("\n=== 处理完成 ===")
    methods_count = {}
    for result in all_results:
        method = result['method']
        methods_count[method] = methods_count.get(method, 0) + 1
    
    print(f"各方法处理的epoch数量:")
    for method, count in sorted(methods_count.items()):
        print(f"  {method}: {count} epochs")


if __name__ == "__main__":
    main()
