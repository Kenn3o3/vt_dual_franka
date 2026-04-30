#!/usr/bin/env python3
"""
计算 SOFA 手术任务各方法各 epoch 的轨迹指标：
- 平均 Jerk (加加速度)
- 平均能量 (Energy)
- 平均路径长度 (Path Length)
- 成功率 (Success Rate)
- 任务特定指标（tip distance、rope fraction、phase、distance to target 等）

支持四个任务：
- rope_threading
- ligating_loop
- grasp_lift_touch
- bimanual_tissue_manipulation

结果将保存在各个 epoch 文件夹下的 trajectory_metrics.json 和 episodes/summary.json
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from tqdm import tqdm


VALID_TASKS = [
    "rope_threading",
    "ligating_loop",
    "grasp_lift_touch",
    "bimanual_tissue_manipulation",
]


def get_positions(traj_data: Dict, task_name: str) -> np.ndarray:
    """
    从 trajectory 数据中提取 3D 位置序列。

    Args:
        traj_data: episode JSON 的 'trajectory' 子字典
        task_name: 任务名称

    Returns:
        positions: shape (T, 3) 的 numpy 数组
    """
    if task_name == "rope_threading":
        return np.array(traj_data["tool_pos_xyz"], dtype=float)
    elif task_name == "ligating_loop":
        return np.array(traj_data["tool_positions_xyz"], dtype=float)
    elif task_name in ("grasp_lift_touch", "bimanual_tissue_manipulation"):
        # 6DOF：[x1,y1,z1, x2,y2,z2]，取第一个工具的前 3 维
        raw = np.array(traj_data["tool_positions_6dof"], dtype=float)
        return raw[:, :3]
    else:
        raise ValueError(f"不支持的任务: {task_name}，有效任务: {VALID_TASKS}")


def get_secondary_positions(traj_data: Dict, task_name: str) -> Optional[np.ndarray]:
    """
    提取第二个工具的 3D 位置（仅限 6DOF 任务）。

    Returns:
        positions: shape (T, 3) 或 None
    """
    if task_name in ("grasp_lift_touch", "bimanual_tissue_manipulation"):
        raw = np.array(traj_data["tool_positions_6dof"], dtype=float)
        if raw.shape[1] >= 6:
            return raw[:, 3:6]
    return None


def compute_velocity(positions: np.ndarray, dt: float = 1 / 30) -> np.ndarray:
    """计算速度，返回 shape (T-1, dim)。"""
    return np.diff(positions, axis=0) / dt


def compute_acceleration(velocity: np.ndarray, dt: float = 1 / 30) -> np.ndarray:
    """计算加速度幅值，返回 shape (T-2,)。"""
    accel_vec = np.diff(velocity, axis=0) / dt
    return np.linalg.norm(accel_vec, axis=-1)


def compute_jerk(acceleration: np.ndarray, dt: float = 1 / 30) -> np.ndarray:
    """计算 Jerk，返回 shape (T-3,)。"""
    return np.abs(np.diff(acceleration)) / dt


def compute_path_length(positions: np.ndarray) -> float:
    """计算 3D 路径总长度。"""
    deltas = np.diff(positions, axis=0)
    return float(np.sum(np.linalg.norm(deltas, axis=-1)))


def compute_energy(acceleration: np.ndarray) -> float:
    """计算能量（加速度幅值之和）。"""
    return float(np.sum(acceleration))


def compute_normalized_metrics(
    jerk: np.ndarray,
    acceleration: np.ndarray,
    path_length: float,
    episode_length: int,
) -> Dict:
    """计算归一化指标（消除轨迹长度的影响）。"""
    safe_length = max(episode_length, 1)
    safe_path_length = max(path_length, 1e-6)

    return {
        "mean_jerk_per_step": float(np.mean(jerk)),
        "energy_per_step": float(np.sum(acceleration) / safe_length),
        "energy_per_meter": float(np.sum(acceleration) / safe_path_length),
        "jerk_per_meter": float(np.mean(jerk) * safe_length / safe_path_length),
    }


def get_task_specific_metrics(episode_data: Dict, task_name: str) -> Dict:
    """
    提取任务特定指标。

    Args:
        episode_data: 完整 episode JSON 字典
        task_name: 任务名称

    Returns:
        task_metrics: 任务特定指标字典
    """
    metrics = episode_data.get("metrics", {}) or {}
    traj = episode_data.get("trajectory", {})

    if task_name == "rope_threading":
        return {
            "min_tip_dist": float(metrics.get("min_tip_dist", np.nan)),
            "final_tip_dist": float(metrics.get("final_tip_dist", np.nan)),
            "max_rope_pts_passed": float(metrics.get("max_rope_pts_passed", np.nan)),
            "final_rope_pts_passed": float(metrics.get("final_rope_pts_passed", np.nan)),
            "max_rope_frac_passed": float(metrics.get("max_rope_frac_passed", np.nan)),
            "final_rope_frac_passed": float(metrics.get("final_rope_frac_passed", np.nan)),
        }

    elif task_name == "ligating_loop":
        overlap = traj.get("loop_marking_overlap", [])
        in_cavity = traj.get("loop_center_in_cavity", [])
        collisions = traj.get("instrument_shaft_collisions", [])
        return {
            "mean_loop_marking_overlap": float(np.mean(overlap)) if overlap else np.nan,
            "max_loop_marking_overlap": float(np.max(overlap)) if overlap else np.nan,
            "mean_loop_center_in_cavity": float(np.mean(in_cavity)) if in_cavity else np.nan,
            "total_shaft_collisions": float(collisions[-1]) if collisions else np.nan,
        }

    elif task_name == "grasp_lift_touch":
        collisions_total = traj.get("collisions_total", [])
        force_gb = traj.get("force_on_gallbladder", [])
        return {
            "final_phase": float(metrics.get("final_phase", np.nan)),
            "total_collisions": float(collisions_total[-1]) if collisions_total else np.nan,
            "mean_force_on_gallbladder": float(np.mean(force_gb)) if force_gb else np.nan,
        }

    elif task_name == "bimanual_tissue_manipulation":
        markers = traj.get("markers_at_target_count", [])
        force = traj.get("force_on_tissue", [])
        return {
            "final_distance_mean": float(metrics.get("final_distance_mean", np.nan)),
            "min_distance_mean": float(metrics.get("min_distance_mean", np.nan)),
            "final_distance_left": float(metrics.get("final_distance_left", np.nan)),
            "final_distance_right": float(metrics.get("final_distance_right", np.nan)),
            "max_markers_at_target": float(max(markers)) if markers else np.nan,
            "mean_force_on_tissue": float(np.mean(force)) if force else np.nan,
        }

    return {}


def process_episode(episode_file: Path, task_name: str, dt: float = 1 / 30) -> Dict:
    """
    处理单个 episode JSON，计算所有指标。

    Args:
        episode_file: episode JSON 文件路径
        task_name: 任务名称
        dt: 时间步长

    Returns:
        metrics: 包含公共指标 + 任务特定指标的字典
    """
    with open(episode_file, "r") as f:
        episode_data = json.load(f)

    traj = episode_data["trajectory"]
    positions = get_positions(traj, task_name)

    # 至少需要 4 个时间步才能计算 jerk
    if len(positions) < 4:
        return None

    velocity = compute_velocity(positions, dt)
    acceleration = compute_acceleration(velocity, dt)
    jerk = compute_jerk(acceleration, dt)

    path_length = compute_path_length(positions)
    energy = compute_energy(acceleration)
    episode_length = int(episode_data["episode_length"])

    normalized = compute_normalized_metrics(jerk, acceleration, path_length, episode_length)
    task_metrics = get_task_specific_metrics(episode_data, task_name)

    return {
        "mean_jerk": float(np.mean(jerk)),
        "std_jerk": float(np.std(jerk)),
        "max_jerk": float(np.max(jerk)),
        "energy": float(energy),
        "path_length": float(path_length),
        "episode_length": episode_length,
        "successful": bool(episode_data["successful"]),
        "total_reward": float(episode_data.get("total_reward", 0.0)),
        **normalized,
        **task_metrics,
    }


def aggregate_metrics(metrics_list: List[Dict], prefix: str, task_name: str) -> Dict:
    """
    对 episode 指标列表进行统计聚合。

    Args:
        metrics_list: episode 指标列表
        prefix: 输出键名前缀（"all_" / "success_" / "failed_"）
        task_name: 任务名称

    Returns:
        agg: 聚合后的指标字典
    """
    if len(metrics_list) == 0:
        return {}

    def safe_mean(key: str) -> float:
        vals = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
        return float(np.mean(vals)) if vals else float("nan")

    def safe_std(key: str) -> float:
        vals = [m[key] for m in metrics_list if key in m and not np.isnan(m[key])]
        return float(np.std(vals)) if vals else float("nan")

    result = {
        f"{prefix}mean_jerk": safe_mean("mean_jerk"),
        f"{prefix}std_jerk": safe_std("mean_jerk"),
        f"{prefix}mean_energy": safe_mean("energy"),
        f"{prefix}std_energy": safe_std("energy"),
        f"{prefix}mean_path_length": safe_mean("path_length"),
        f"{prefix}std_path_length": safe_std("path_length"),
        f"{prefix}mean_episode_length": safe_mean("episode_length"),
        f"{prefix}mean_jerk_per_step": safe_mean("mean_jerk_per_step"),
        f"{prefix}energy_per_step": safe_mean("energy_per_step"),
        f"{prefix}energy_per_meter": safe_mean("energy_per_meter"),
        f"{prefix}jerk_per_meter": safe_mean("jerk_per_meter"),
        f"{prefix}count": len(metrics_list),
    }

    # 任务特定指标的聚合
    task_keys = _get_task_specific_keys(task_name)
    for key in task_keys:
        result[f"{prefix}{key}"] = safe_mean(key)

    return result


def _get_task_specific_keys(task_name: str) -> List[str]:
    """返回任务特定指标的键名列表。"""
    if task_name == "rope_threading":
        return [
            "min_tip_dist",
            "final_tip_dist",
            "max_rope_pts_passed",
            "final_rope_pts_passed",
            "max_rope_frac_passed",
            "final_rope_frac_passed",
        ]
    elif task_name == "ligating_loop":
        return [
            "mean_loop_marking_overlap",
            "max_loop_marking_overlap",
            "mean_loop_center_in_cavity",
            "total_shaft_collisions",
        ]
    elif task_name == "grasp_lift_touch":
        return ["final_phase", "total_collisions", "mean_force_on_gallbladder"]
    elif task_name == "bimanual_tissue_manipulation":
        return [
            "final_distance_mean",
            "min_distance_mean",
            "final_distance_left",
            "final_distance_right",
            "max_markers_at_target",
            "mean_force_on_tissue",
        ]
    return []


def process_epoch(epoch_dir: Path, task_name: str, dt: float = 1 / 30) -> Optional[Dict]:
    """
    处理单个 epoch 的所有 episode，写入 trajectory_metrics.json 和 episodes/summary.json。

    Args:
        epoch_dir: epoch 文件夹路径
        task_name: 任务名称
        dt: 时间步长

    Returns:
        epoch_metrics: epoch 级别聚合指标，失败则返回 None
    """
    episodes_dir = epoch_dir / "episodes"
    if not episodes_dir.exists():
        return None

    episode_files = sorted(episodes_dir.glob("*_episode_*.json"))
    if len(episode_files) == 0:
        return None

    all_metrics: List[Dict] = []
    success_metrics: List[Dict] = []
    failed_metrics: List[Dict] = []

    for episode_file in episode_files:
        try:
            ep_metrics = process_episode(episode_file, task_name, dt)
            if ep_metrics is None:
                continue
            all_metrics.append(ep_metrics)
            if ep_metrics["successful"]:
                success_metrics.append(ep_metrics)
            else:
                failed_metrics.append(ep_metrics)
        except Exception as e:
            print(f"  Warning: 处理 {episode_file.name} 失败: {e}")
            continue

    if len(all_metrics) == 0:
        return None

    epoch_metrics = {
        "epoch": int(epoch_dir.name.replace("epoch_", "")),
        "total_episodes": len(all_metrics),
        "success_rate": len(success_metrics) / len(all_metrics),
        **aggregate_metrics(all_metrics, "all_", task_name),
        **aggregate_metrics(success_metrics, "success_", task_name),
        **aggregate_metrics(failed_metrics, "failed_", task_name),
    }

    # 写 episodes/summary.json
    summary = {
        "total": len(all_metrics),
        "successful": len(success_metrics),
        "failed": len(failed_metrics),
        "mean_length": float(np.mean([m["episode_length"] for m in all_metrics])),
        "success_rate": epoch_metrics["success_rate"],
    }
    with open(episodes_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return epoch_metrics


def process_method(
    method_dir: Path,
    task_name: str,
    dt: float = 1 / 30,
    save_results: bool = True,
) -> List[Dict]:
    """
    处理单个方法目录下的所有实验运行。

    Args:
        method_dir: 方法文件夹路径（含 date/time 子目录）
        task_name: 任务名称
        dt: 时间步长
        save_results: 是否将 trajectory_metrics.json 写回 epoch 目录

    Returns:
        all_results: 所有 epoch 的指标列表
    """
    print(f"\n处理方法: {method_dir.name}")

    date_dirs = sorted([d for d in method_dir.iterdir() if d.is_dir()])
    all_results: List[Dict] = []

    for date_dir in date_dirs:
        time_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()])

        for time_dir in time_dirs:
            print(f"  处理实验: {date_dir.name}/{time_dir.name}")

            epoch_dirs = sorted(
                [d for d in time_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")]
            )

            if len(epoch_dirs) == 0:
                print("    Warning: 没有找到 epoch 文件夹")
                continue

            for epoch_dir in tqdm(epoch_dirs, desc="    处理 epochs", leave=False):
                epoch_metrics = process_epoch(epoch_dir, task_name, dt)
                if epoch_metrics is None:
                    continue

                epoch_metrics["task"] = task_name
                epoch_metrics["method"] = method_dir.name
                epoch_metrics["date"] = date_dir.name
                epoch_metrics["time"] = time_dir.name
                epoch_metrics["run_path"] = str(
                    time_dir.relative_to(method_dir.parent.parent)
                )

                all_results.append(epoch_metrics)

                if save_results:
                    output_file = epoch_dir / "trajectory_metrics.json"
                    with open(output_file, "w") as f:
                        json.dump(epoch_metrics, f, indent=2)

    return all_results


def main():
    parser = argparse.ArgumentParser(description="计算 SOFA 任务轨迹指标")
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="结果文件夹路径（方法目录的父目录）。"
        "若未指定则使用 mpd/results/<task>",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=VALID_TASKS,
        help="任务名称",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="要处理的方法列表（默认：自动发现所有方法）",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=1 / 30,
        help="时间步长（默认：1/30）",
    )
    parser.add_argument(
        "--save_summary",
        action="store_true",
        help="是否在 results_dir 下保存汇总 JSON",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="trajectory_metrics_summary.json",
        help="汇总结果输出文件名",
    )

    args = parser.parse_args()

    # 确定 results_dir
    if args.results_dir is None:
        script_root = Path(__file__).resolve().parent.parent
        results_dir = script_root / "results" / args.task
    else:
        results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: 结果文件夹不存在: {results_dir}")
        return

    # 确定方法目录列表
    if args.methods is None:
        method_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    else:
        method_dirs = [results_dir / m for m in args.methods if (results_dir / m).exists()]

    if len(method_dirs) == 0:
        print("Error: 没有找到方法文件夹")
        return

    print(f"任务: {args.task}")
    print(f"结果目录: {results_dir}")
    print(f"方法数量: {len(method_dirs)}")

    all_results: List[Dict] = []
    for method_dir in method_dirs:
        method_results = process_method(method_dir, args.task, dt=args.dt, save_results=True)
        all_results.extend(method_results)

    if args.save_summary and len(all_results) > 0:
        output_path = results_dir / args.output
        with open(output_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n汇总结果已保存到: {output_path}")

    print("\n=== 处理完成 ===")
    methods_count: Dict[str, int] = {}
    for result in all_results:
        m = result["method"]
        methods_count[m] = methods_count.get(m, 0) + 1

    print("各方法处理的 epoch 数量:")
    for method, count in sorted(methods_count.items()):
        print(f"  {method}: {count} epochs")


if __name__ == "__main__":
    main()
