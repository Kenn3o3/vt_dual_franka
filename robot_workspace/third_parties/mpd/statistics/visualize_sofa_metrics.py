#!/usr/bin/env python3
"""
可视化 SOFA 手术任务的轨迹指标：

1. 每个 epoch 的汇总统计图（epoch_summary.png）
2. 每个 epoch 的 3D 轨迹速度着色图（trajectories_3d.png）
   每个 epoch 的 2D 轨迹速度着色图，仅 X-Y 分量（trajectories_2d.png）
3. 跨 epoch 的方法对比图（comparison/<metric>.png）
4. 每个方法的多指标综合图（per_method/<method>_metrics.png）
5. 最终 epoch 汇总表（final_metrics_summary.json）

所有图保存在各 epoch 目录下的 plots/ 子文件夹，
跨 epoch 对比图保存在 results_dir/trajectory_metrics_plots/。
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse
from collections import defaultdict
from tqdm import tqdm


VALID_TASKS = [
    "rope_threading",
    "ligating_loop",
    "grasp_lift_touch",
    "bimanual_tissue_manipulation",
]


# ─────────────────────────────────────────────────────────────────────────────
# Task-specific configuration
# ─────────────────────────────────────────────────────────────────────────────

def get_task_specific_metric_configs(task_name: str) -> List[Dict]:
    """返回任务特定指标的可视化配置。"""
    if task_name == "rope_threading":
        return [
            {"key": "all_max_rope_frac_passed", "title": "Max Rope Frac Passed (All)", "ylabel": "Fraction"},
            {"key": "success_max_rope_frac_passed", "title": "Max Rope Frac Passed (Success)", "ylabel": "Fraction"},
            {"key": "all_final_rope_frac_passed", "title": "Final Rope Frac Passed (All)", "ylabel": "Fraction"},
            {"key": "all_min_tip_dist", "title": "Min Tip Distance (All)", "ylabel": "Distance"},
        ]
    elif task_name == "ligating_loop":
        return [
            {"key": "all_max_loop_marking_overlap", "title": "Max Loop Overlap (All)", "ylabel": "Overlap"},
            {"key": "success_max_loop_marking_overlap", "title": "Max Loop Overlap (Success)", "ylabel": "Overlap"},
            {"key": "all_mean_loop_center_in_cavity", "title": "Loop Center in Cavity (All)", "ylabel": "Fraction"},
            {"key": "all_total_shaft_collisions", "title": "Shaft Collisions (All)", "ylabel": "Count"},
        ]
    elif task_name == "grasp_lift_touch":
        return [
            {"key": "all_final_phase", "title": "Final Phase (All)", "ylabel": "Phase"},
            {"key": "success_final_phase", "title": "Final Phase (Success)", "ylabel": "Phase"},
            {"key": "all_total_collisions", "title": "Total Collisions (All)", "ylabel": "Count"},
            {"key": "all_mean_force_on_gallbladder", "title": "Mean Force on Gallbladder (All)", "ylabel": "Force"},
        ]
    elif task_name == "bimanual_tissue_manipulation":
        return [
            {"key": "all_final_distance_mean", "title": "Final Distance Mean (All)", "ylabel": "Distance"},
            {"key": "success_final_distance_mean", "title": "Final Distance Mean (Success)", "ylabel": "Distance"},
            {"key": "all_min_distance_mean", "title": "Min Distance Mean (All)", "ylabel": "Distance"},
            {"key": "all_max_markers_at_target", "title": "Max Markers at Target (All)", "ylabel": "Count"},
        ]
    return []


def get_trajectory_positions(traj: Dict, task_name: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    提取主工具和（可选）次工具的 3D 位置序列。

    Returns:
        (primary_pos, secondary_pos) — secondary_pos 对非 6DOF 任务为 None
    """
    if task_name == "rope_threading":
        pos = np.array(traj["tool_pos_xyz"], dtype=float)
        return pos, None
    elif task_name == "ligating_loop":
        pos = np.array(traj["tool_positions_xyz"], dtype=float)
        return pos, None
    elif task_name in ("grasp_lift_touch", "bimanual_tissue_manipulation"):
        raw = np.array(traj["tool_positions_6dof"], dtype=float)
        return raw[:, :3], raw[:, 3:6] if raw.shape[1] >= 6 else None
    raise ValueError(f"不支持的任务: {task_name}")


def compute_speeds(positions: np.ndarray) -> np.ndarray:
    """计算每段的速度（相邻点欧式距离），shape (T-1,)。"""
    return np.linalg.norm(np.diff(positions, axis=0), axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Per-epoch plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_epoch_summary(epoch_metrics: Dict, task_name: str, output_path: Path) -> None:
    """
    绘制单个 epoch 的多指标汇总条形图，分 all / success / failed 三组展示。

    Args:
        epoch_metrics: trajectory_metrics.json 的内容
        task_name: 任务名称
        output_path: 保存路径
    """
    # 基础指标（有 all/success/failed 三组）
    base_metrics = [
        ("mean_episode_length", "Episode Length", "Steps"),
        ("mean_path_length", "Path Length", "Units"),
        ("mean_jerk_per_step", "Jerk / Step", "Jerk/step"),
        ("energy_per_step", "Energy / Step", "Energy/step"),
        ("energy_per_meter", "Energy / Meter", "Energy/m"),
    ]

    task_configs = get_task_specific_metric_configs(task_name)
    task_keys_for_summary = [
        ("max_rope_frac_passed", "Max Rope Frac", "Fraction"),
        ("min_tip_dist", "Min Tip Dist", "Distance"),
    ] if task_name == "rope_threading" else [
        ("max_loop_marking_overlap", "Max Loop Overlap", "Overlap"),
        ("total_shaft_collisions", "Shaft Collisions", "Count"),
    ] if task_name == "ligating_loop" else [
        ("final_phase", "Final Phase", "Phase"),
        ("total_collisions", "Total Collisions", "Count"),
    ] if task_name == "grasp_lift_touch" else [
        ("final_distance_mean", "Final Dist Mean", "Distance"),
        ("max_markers_at_target", "Max Markers", "Count"),
    ]

    all_panels = base_metrics + task_keys_for_summary

    n_panels = len(all_panels) + 1  # +1 for success_rate
    ncols = 3
    nrows = (n_panels + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
    axes = np.array(axes).flatten()

    epoch = epoch_metrics.get("epoch", "?")
    method = epoch_metrics.get("method", "")
    sr = epoch_metrics.get("success_rate", 0)
    total = epoch_metrics.get("total_episodes", 0)
    fig.suptitle(
        f"{task_name} | {method} | Epoch {epoch} | "
        f"Success Rate: {sr:.1%} ({epoch_metrics.get('success_count', epoch_metrics.get('all_count', total))} total)",
        fontsize=13, fontweight="bold",
    )

    colors = {"all": "#4C72B0", "success": "#55A868", "failed": "#C44E52"}

    # Panel 0: success rate
    ax = axes[0]
    ax.bar(["Success Rate"], [sr], color=colors["success"], edgecolor="white", width=0.4)
    ax.set_ylim(0, 1.05)
    ax.set_title("Success Rate", fontweight="bold")
    ax.set_ylabel("Rate")
    ax.text(0, sr + 0.02, f"{sr:.1%}", ha="center", va="bottom", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Remaining panels
    for i, (key, title, ylabel) in enumerate(all_panels):
        ax = axes[i + 1]
        labels, vals = [], []
        for group in ("all", "success", "failed"):
            full_key = f"{group}_{key}"
            val = epoch_metrics.get(full_key, np.nan)
            if not np.isnan(val):
                labels.append(group.capitalize())
                vals.append(val)

        if vals:
            bar_colors = [colors[g.lower()] for g in labels]
            bars = ax.bar(labels, vals, color=bar_colors, edgecolor="white", width=0.5)
            for bar, v in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{v:.3g}",
                    ha="center", va="bottom", fontsize=8,
                )
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused axes
    for j in range(i + 2, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _add_3d_trajectory(
    ax: "Axes3D",
    positions: np.ndarray,
    speeds: np.ndarray,
    global_norm: plt.Normalize,
    cmap: str = "plasma",
    alpha: float = 0.75,
    linewidth: float = 1.2,
) -> Line3DCollection:
    """向 3D 轴添加一条速度着色轨迹，返回 Line3DCollection 供 colorbar 使用。"""
    segments = [positions[i : i + 2] for i in range(len(positions) - 1)]
    lc = Line3DCollection(
        segments, cmap=cmap, norm=global_norm, alpha=alpha, linewidth=linewidth
    )
    lc.set_array(speeds)
    ax.add_collection3d(lc)
    return lc


def _add_2d_trajectory(
    ax: plt.Axes,
    positions: np.ndarray,
    speeds: np.ndarray,
    global_norm: plt.Normalize,
    cmap: str = "plasma",
    alpha: float = 0.75,
    linewidth: float = 1.2,
) -> LineCollection:
    """向 2D 轴添加一条速度着色轨迹（仅 X-Y 分量），返回 LineCollection 供 colorbar 使用。"""
    xy = positions[:, :2]
    segments = [xy[i : i + 2] for i in range(len(xy) - 1)]
    lc = LineCollection(segments, cmap=cmap, norm=global_norm, alpha=alpha, linewidth=linewidth)
    lc.set_array(speeds)
    ax.add_collection(lc)
    return lc


def plot_epoch_trajectories_2d(
    epoch_dir: Path,
    task_name: str,
    output_path: Path,
) -> None:
    """
    绘制单个 epoch 所有 episode 的 2D 轨迹（X-Y 平面），颜色表示速度。

    - 1-tool 任务：2 列（success / failed）
    - 2-tool 任务：2 行 × 2 列（tool1/tool2 各有 success/failed 列）
    - 所有子图共享 colorbar
    """
    episodes_dir = epoch_dir / "episodes"
    if not episodes_dir.exists():
        return

    episode_files = sorted(episodes_dir.glob("*_episode_*.json"))
    if not episode_files:
        return

    two_tools = task_name in ("grasp_lift_touch", "bimanual_tissue_manipulation")

    success_data: List[Tuple] = []
    failed_data: List[Tuple] = []
    all_speeds: List[float] = []

    for ep_file in episode_files:
        try:
            with open(ep_file, "r") as f:
                ep = json.load(f)
            traj = ep["trajectory"]
            pos1, pos2 = get_trajectory_positions(traj, task_name)
            if len(pos1) < 2:
                continue
            sp1 = compute_speeds(pos1)
            sp2 = compute_speeds(pos2) if pos2 is not None and len(pos2) >= 2 else None
            all_speeds.extend(sp1.tolist())
            if sp2 is not None:
                all_speeds.extend(sp2.tolist())
            entry = (pos1, pos2, sp1, sp2)
            if ep["successful"]:
                success_data.append(entry)
            else:
                failed_data.append(entry)
        except Exception:
            continue

    if not all_speeds:
        return

    global_norm = plt.Normalize(
        vmin=max(0, np.percentile(all_speeds, 2)),
        vmax=np.percentile(all_speeds, 98),
    )
    cmap = "plasma"

    def setup_ax_2d(ax: plt.Axes, title: str) -> None:
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("X", fontsize=8)
        ax.set_ylabel("Y", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.25)

    if two_tools:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        ax_s1, ax_s2 = axes[0]
        ax_f1, ax_f2 = axes[1]
        axes_pairs = [
            (ax_s1, ax_s2, success_data, "Success"),
            (ax_f1, ax_f2, failed_data, "Failed"),
        ]
    else:
        fig, (ax_s, ax_f) = plt.subplots(1, 2, figsize=(14, 6))
        axes_pairs = None
        axes_single = [(ax_s, success_data, "Success"), (ax_f, failed_data, "Failed")]

    last_lc = None

    if two_tools:
        for ax1, ax2, data, label in axes_pairs:
            setup_ax_2d(ax1, f"{label} — Tool 1 ({len(data)} ep)")
            setup_ax_2d(ax2, f"{label} — Tool 2 ({len(data)} ep)")
            for pos1, pos2, sp1, sp2 in data:
                lc = _add_2d_trajectory(ax1, pos1, sp1, global_norm, cmap)
                last_lc = lc
                if pos2 is not None and sp2 is not None:
                    lc2 = _add_2d_trajectory(ax2, pos2, sp2, global_norm, cmap)
                    last_lc = lc2
            for ax, pts_list in [(ax1, [d[0] for d in data]),
                                  (ax2, [d[1] for d in data if d[1] is not None])]:
                if pts_list:
                    all_pts = np.vstack(pts_list)
                    margin_x = max((all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.05, 1e-6)
                    margin_y = max((all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.05, 1e-6)
                    ax.set_xlim(all_pts[:, 0].min() - margin_x, all_pts[:, 0].max() + margin_x)
                    ax.set_ylim(all_pts[:, 1].min() - margin_y, all_pts[:, 1].max() + margin_y)
    else:
        for ax, data, label in axes_single:
            setup_ax_2d(ax, f"{label} ({len(data)} ep)")
            for pos1, _, sp1, _ in data:
                lc = _add_2d_trajectory(ax, pos1, sp1, global_norm, cmap)
                last_lc = lc
            if data:
                all_pts = np.vstack([d[0] for d in data])
                margin_x = max((all_pts[:, 0].max() - all_pts[:, 0].min()) * 0.05, 1e-6)
                margin_y = max((all_pts[:, 1].max() - all_pts[:, 1].min()) * 0.05, 1e-6)
                ax.set_xlim(all_pts[:, 0].min() - margin_x, all_pts[:, 0].max() + margin_x)
                ax.set_ylim(all_pts[:, 1].min() - margin_y, all_pts[:, 1].max() + margin_y)

    epoch_num = epoch_dir.name.replace("epoch_", "")
    fig.suptitle(
        f"{task_name} — Epoch {epoch_num} — 2D Trajectories XY (color = speed)",
        fontsize=12, fontweight="bold",
    )

    if last_lc is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=global_norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Speed (units/step)")

    plt.subplots_adjust(right=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_trajectories_3d(
    epoch_dir: Path,
    task_name: str,
    output_path: Path,
) -> None:
    """
    绘制单个 epoch 所有 episode 的 3D 轨迹，颜色表示速度。

    - 1-tool 任务：2 列（success / failed）
    - 2-tool 任务：2 行 × 2 列（tool1/tool2 各有 success/failed 列）
    - 所有子图共享 colorbar
    """
    episodes_dir = epoch_dir / "episodes"
    if not episodes_dir.exists():
        return

    episode_files = sorted(episodes_dir.glob("*_episode_*.json"))
    if not episode_files:
        return

    two_tools = task_name in ("grasp_lift_touch", "bimanual_tissue_manipulation")

    # 读取数据
    success_data: List[Tuple] = []  # list of (primary_pos, secondary_pos, speeds_primary, speeds_sec)
    failed_data: List[Tuple] = []

    all_speeds: List[float] = []

    for ep_file in episode_files:
        try:
            with open(ep_file, "r") as f:
                ep = json.load(f)
            traj = ep["trajectory"]
            pos1, pos2 = get_trajectory_positions(traj, task_name)
            if len(pos1) < 2:
                continue
            sp1 = compute_speeds(pos1)
            sp2 = compute_speeds(pos2) if pos2 is not None and len(pos2) >= 2 else None
            all_speeds.extend(sp1.tolist())
            if sp2 is not None:
                all_speeds.extend(sp2.tolist())
            entry = (pos1, pos2, sp1, sp2)
            if ep["successful"]:
                success_data.append(entry)
            else:
                failed_data.append(entry)
        except Exception:
            continue

    if not all_speeds:
        return

    global_norm = plt.Normalize(
        vmin=max(0, np.percentile(all_speeds, 2)),
        vmax=np.percentile(all_speeds, 98),
    )
    cmap = "plasma"

    # 布局
    if two_tools:
        # 2x2 网格：行=success/failed，列=tool1/tool2
        fig = plt.figure(figsize=(14, 10))
        ax_s1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax_s2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax_f1 = fig.add_subplot(2, 2, 3, projection="3d")
        ax_f2 = fig.add_subplot(2, 2, 4, projection="3d")
        axes_pairs = [
            (ax_s1, ax_s2, success_data, "Success"),
            (ax_f1, ax_f2, failed_data, "Failed"),
        ]
    else:
        fig = plt.figure(figsize=(14, 6))
        ax_s = fig.add_subplot(1, 2, 1, projection="3d")
        ax_f = fig.add_subplot(1, 2, 2, projection="3d")
        axes_pairs = None
        axes_single = [(ax_s, success_data, "Success"), (ax_f, failed_data, "Failed")]

    last_lc = None  # for colorbar

    def setup_ax(ax: "Axes3D", title: str) -> None:
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("X", fontsize=7)
        ax.set_ylabel("Y", fontsize=7)
        ax.set_zlabel("Z", fontsize=7)
        ax.tick_params(labelsize=6)

    if two_tools:
        for ax1, ax2, data, label in axes_pairs:
            setup_ax(ax1, f"{label} — Tool 1 ({len(data)} ep)")
            setup_ax(ax2, f"{label} — Tool 2 ({len(data)} ep)")
            for pos1, pos2, sp1, sp2 in data:
                lc = _add_3d_trajectory(ax1, pos1, sp1, global_norm, cmap)
                last_lc = lc
                if pos2 is not None and sp2 is not None:
                    lc2 = _add_3d_trajectory(ax2, pos2, sp2, global_norm, cmap)
                    last_lc = lc2
            # Auto-scale
            for ax, data_list, attr in [(ax1, [(d[0], d[2]) for d in data], 0),
                                         (ax2, [(d[1], d[3]) for d in data if d[1] is not None], 0)]:
                if data_list:
                    all_pts = np.vstack([p for p, _ in data_list if p is not None])
                    ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
                    ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
                    ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())
    else:
        for ax, data, label in axes_single:
            setup_ax(ax, f"{label} ({len(data)} ep)")
            for pos1, _, sp1, _ in data:
                lc = _add_3d_trajectory(ax, pos1, sp1, global_norm, cmap)
                last_lc = lc
            if data:
                all_pts = np.vstack([d[0] for d in data])
                ax.set_xlim(all_pts[:, 0].min(), all_pts[:, 0].max())
                ax.set_ylim(all_pts[:, 1].min(), all_pts[:, 1].max())
                ax.set_zlim(all_pts[:, 2].min(), all_pts[:, 2].max())

    epoch_num = epoch_dir.name.replace("epoch_", "")
    method = epoch_dir.parent.parent.parent.name  # date/../method
    fig.suptitle(
        f"{task_name} — Epoch {epoch_num} — 3D Trajectories (color = speed)",
        fontsize=12, fontweight="bold",
    )

    if last_lc is not None:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=global_norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cbar_ax, label="Speed (units/step)")

    plt.subplots_adjust(right=0.90)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_epoch_plots(
    results_dir: Path,
    task_name: str,
    methods: Optional[List[str]] = None,
) -> None:
    """
    遍历所有 epoch 目录，生成 plots/epoch_summary.png 和 plots/trajectories_3d.png。

    Args:
        results_dir: 包含各方法目录的根目录（如 mpd/results/rope_threading）
        task_name: 任务名称
        methods: 要处理的方法列表，None 表示全部
    """
    if methods is None:
        method_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    else:
        method_dirs = [results_dir / m for m in methods if (results_dir / m).is_dir()]

    for method_dir in method_dirs:
        for date_dir in sorted(d for d in method_dir.iterdir() if d.is_dir()):
            for time_dir in sorted(d for d in date_dir.iterdir() if d.is_dir()):
                epoch_dirs = sorted(
                    d for d in time_dir.iterdir()
                    if d.is_dir() and d.name.startswith("epoch_")
                )
                for epoch_dir in tqdm(
                    epoch_dirs,
                    desc=f"  {method_dir.name}/{date_dir.name}/{time_dir.name}",
                    leave=False,
                ):
                    metrics_file = epoch_dir / "trajectory_metrics.json"
                    plots_dir = epoch_dir / "plots"

                    # Summary bar chart
                    summary_out = plots_dir / "epoch_summary.png"
                    if metrics_file.exists():
                        with open(metrics_file, "r") as f:
                            epoch_metrics = json.load(f)
                        plot_epoch_summary(epoch_metrics, task_name, summary_out)

                    # 3D trajectory plot
                    traj3d_out = plots_dir / "trajectories_3d.png"
                    plot_epoch_trajectories_3d(epoch_dir, task_name, traj3d_out)

                    # 2D trajectory plot (X-Y plane)
                    traj2d_out = plots_dir / "trajectories_2d.png"
                    plot_epoch_trajectories_2d(epoch_dir, task_name, traj2d_out)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-epoch comparison plots
# ─────────────────────────────────────────────────────────────────────────────

def load_metrics(
    results_dir: Path,
    task_name: str,
    methods: Optional[List[str]] = None,
) -> Dict:
    """
    加载所有方法的 trajectory_metrics.json，结构同 visualize_trajectory_metrics.py。

    Returns:
        {method: {run_path: [metrics_list_sorted_by_epoch]}}
    """
    if methods is None:
        method_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir()])
    else:
        method_dirs = [results_dir / m for m in methods if (results_dir / m).is_dir()]

    metrics_dict = defaultdict(lambda: defaultdict(list))

    for method_dir in method_dirs:
        for metrics_file in method_dir.rglob("trajectory_metrics.json"):
            with open(metrics_file, "r") as f:
                m = json.load(f)
            # 只加载本任务的指标（有 task 字段时检查一致性）
            if m.get("task", task_name) != task_name:
                continue
            run_path = f"{m['date']}/{m['time']}"
            metrics_dict[method_dir.name][run_path].append(m)

    for method in metrics_dict:
        for run_path in metrics_dict[method]:
            metrics_dict[method][run_path].sort(key=lambda x: x["epoch"])

    return metrics_dict


def plot_metrics_comparison(
    metrics_dict: Dict,
    task_name: str,
    output_dir: Path,
) -> None:
    """每个指标绘制一张方法对比折线图，保存到 output_dir/comparison/。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    common_configs = [
        {"key": "success_rate", "title": "Success Rate", "ylabel": "Success Rate"},
        {"key": "all_mean_path_length", "title": "Mean Path Length (All)", "ylabel": "Path Length"},
        {"key": "all_mean_episode_length", "title": "Mean Episode Length (All)", "ylabel": "Steps"},
        {"key": "all_mean_jerk", "title": "Mean Jerk (All)", "ylabel": "Jerk"},
        {"key": "all_mean_jerk_per_step", "title": "Mean Jerk/Step (All)", "ylabel": "Jerk/step"},
        {"key": "all_energy_per_step", "title": "Energy/Step (All)", "ylabel": "Energy/step"},
        {"key": "all_energy_per_meter", "title": "Energy/Meter (All)", "ylabel": "Energy/m"},
        {"key": "success_mean_jerk_per_step", "title": "Jerk/Step (Success)", "ylabel": "Jerk/step"},
        {"key": "success_energy_per_step", "title": "Energy/Step (Success)", "ylabel": "Energy/step"},
    ]
    all_configs = common_configs + get_task_specific_metric_configs(task_name)

    for config in all_configs:
        metric_key = config["key"]
        fig, ax = plt.subplots(figsize=(12, 6))

        for method in sorted(metrics_dict.keys()):
            for run_path, metrics_list in metrics_dict[method].items():
                epochs = [m["epoch"] for m in metrics_list]
                values = [
                    m.get(metric_key, np.nan) if m.get(metric_key) is not None else np.nan
                    for m in metrics_list
                ]
                if all(np.isnan(v) for v in values):
                    continue
                label = method
                if len(metrics_dict[method]) > 1:
                    label += f" ({run_path.split('/')[0]})"
                ax.plot(epochs, values, marker="o", markersize=3, label=label, alpha=0.8, linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel(config["ylabel"], fontsize=12)
        ax.set_title(f"{task_name} — {config['title']}", fontsize=13, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        out = output_dir / f"{metric_key}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {out}")


def plot_multi_metrics_per_method(
    metrics_dict: Dict,
    task_name: str,
    output_dir: Path,
) -> None:
    """为每个方法绘制包含多个指标的综合对比图。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    panels = [
        ("success_rate", "Success Rate", "Rate"),
        ("all_mean_path_length", "Path Length (All)", "Units"),
        ("all_mean_episode_length", "Episode Length (All)", "Steps"),
        ("all_mean_jerk_per_step", "Jerk/Step (All)", "Jerk/step"),
        ("all_energy_per_step", "Energy/Step (All)", "Energy/step"),
        ("all_energy_per_meter", "Energy/Meter (All)", "Energy/m"),
    ]
    task_configs = get_task_specific_metric_configs(task_name)
    extra_panels = [(c["key"], c["title"].replace(f" ({task_name.replace('_', ' ')})", ""), c["ylabel"])
                    for c in task_configs[:3]]
    panels = panels + extra_panels

    ncols = 3
    nrows = (len(panels) + ncols - 1) // ncols

    for method in sorted(metrics_dict.keys()):
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 3.5))
        axes = np.array(axes).flatten()
        fig.suptitle(f"{task_name} — {method} — Metrics over Epochs", fontsize=13, fontweight="bold")

        for idx, (metric_key, title, ylabel) in enumerate(panels):
            ax = axes[idx]
            for run_path, metrics_list in metrics_dict[method].items():
                epochs = [m["epoch"] for m in metrics_list]
                values = [
                    m.get(metric_key, np.nan) if m.get(metric_key) is not None else np.nan
                    for m in metrics_list
                ]
                if all(np.isnan(v) for v in values):
                    continue
                label = run_path.split("/")[0] if len(metrics_dict[method]) > 1 else None
                ax.plot(epochs, values, marker="o", markersize=4, label=label, alpha=0.8, linewidth=2)
            ax.set_xlabel("Epoch", fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            if len(metrics_dict[method]) > 1:
                ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        for j in range(len(panels), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        out = output_dir / f"{method}_metrics.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  已保存: {out}")


def generate_summary_table(
    metrics_dict: Dict,
    task_name: str,
    output_dir: Path,
) -> None:
    """生成最终 epoch 的指标汇总表，保存为 JSON 并打印。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_data = []
    task_keys = get_task_specific_metric_configs(task_name)

    for method in sorted(metrics_dict.keys()):
        for run_path, metrics_list in metrics_dict[method].items():
            if not metrics_list:
                continue
            last = metrics_list[-1]
            row = {
                "Method": method,
                "Run": run_path.split("/")[0],
                "Epoch": last["epoch"],
                "Success Rate": f"{last['success_rate']:.3f}",
                "Ep Length": f"{last.get('all_mean_episode_length', np.nan):.1f}",
                "Path Length": f"{last.get('all_mean_path_length', np.nan):.4f}",
                "Jerk/Step": f"{last.get('all_mean_jerk_per_step', np.nan):.4f}",
                "Energy/Step": f"{last.get('all_energy_per_step', np.nan):.4f}",
                "Energy/Meter": f"{last.get('all_energy_per_meter', np.nan):.2f}",
            }
            for cfg in task_keys:
                k = cfg["key"]
                row[cfg["title"]] = f"{last.get(k, np.nan):.4g}"
            summary_data.append(row)

    out_file = output_dir / "final_metrics_summary.json"
    with open(out_file, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  已保存: {out_file}")

    print(f"\n=== {task_name} — Final Epoch Metrics Summary ===")
    print(f"{'Method':<25} {'Success':<10} {'Ep Len':<10} {'Path Len':<12} {'Jerk/step':<12} {'Energy/step':<14}")
    print("-" * 90)
    for row in summary_data:
        print(f"{row['Method']:<25} {row['Success Rate']:<10} {row['Ep Length']:<10} "
              f"{row['Path Length']:<12} {row['Jerk/Step']:<12} {row['Energy/Step']:<14}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="可视化 SOFA 任务轨迹指标")
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=VALID_TASKS,
        help="任务名称",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=None,
        help="结果文件夹路径（方法目录的父目录）。未指定则使用 mpd/results/<task>",
    )
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=None,
        help="要处理的方法列表（默认：自动发现所有方法）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="跨 epoch 对比图的输出目录（默认：results_dir/trajectory_metrics_plots）",
    )
    parser.add_argument(
        "--no_epoch_plots",
        action="store_true",
        help="跳过逐 epoch 的汇总图和 3D 轨迹图",
    )
    parser.add_argument(
        "--no_comparison",
        action="store_true",
        help="跳过方法对比折线图",
    )
    parser.add_argument(
        "--no_per_method",
        action="store_true",
        help="跳过每个方法的多指标综合图",
    )
    parser.add_argument(
        "--no_summary",
        action="store_true",
        help="跳过最终 epoch 汇总表",
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

    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "trajectory_metrics_plots"

    print(f"任务: {args.task}")
    print(f"结果目录: {results_dir}")

    # 1. 逐 epoch 可视化
    if not args.no_epoch_plots:
        print("\n生成逐 epoch 可视化图 (epoch_summary + trajectories_3d)...")
        generate_epoch_plots(results_dir, args.task, args.methods)

    # 2. 加载跨 epoch 指标
    print("\n加载 trajectory_metrics.json...")
    metrics_dict = load_metrics(results_dir, args.task, args.methods)

    if not metrics_dict:
        print("Warning: 没有找到 trajectory_metrics.json，请先运行 compute_sofa_metrics.py")
        return

    print(f"已加载 {len(metrics_dict)} 个方法")
    for m, runs in sorted(metrics_dict.items()):
        total_epochs = sum(len(v) for v in runs.values())
        print(f"  {m}: {len(runs)} 次实验, {total_epochs} epochs")

    # 3. 方法对比折线图
    if not args.no_comparison:
        print("\n生成方法对比图...")
        plot_metrics_comparison(metrics_dict, args.task, output_dir / "comparison")

    # 4. 每方法多指标综合图
    if not args.no_per_method:
        print("\n生成每方法多指标综合图...")
        plot_multi_metrics_per_method(metrics_dict, args.task, output_dir / "per_method")

    # 5. 汇总表
    if not args.no_summary:
        print("\n生成汇总表...")
        generate_summary_table(metrics_dict, args.task, output_dir)

    print(f"\n所有图表已保存到: {output_dir}")


if __name__ == "__main__":
    main()
