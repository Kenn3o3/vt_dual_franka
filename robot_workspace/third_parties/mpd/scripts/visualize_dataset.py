"""
Visualize the obstacle avoidance dataset trajectories,
reusing the same plotting logic as in the training evaluation
(ObstacleAvoidanceEnvVectorWorkspace._plot_all_trajectories).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Hard-coded scene parameters (from ObstacleAvoidanceScene / URDF files)
# ---------------------------------------------------------------------------
OBS_LVL_ORIGIN = np.array([0.5, -0.1, 0.0])
OBS_LVL_OFFSET = np.array([0.075, 0.18])
FINISH_LINE_X   = 0.4

# obstacle_0 (level-1 single obstacle): radius 0.03
# obstacle_1 (level-2/3 obstacles):     radius 0.025
OBSTACLES = [
    {"name": "obs_l1_mid", "cx": 0.5,   "cy": -0.1,  "r": 0.03},
    {"name": "obs_l2_top", "cx": 0.425, "cy":  0.08,  "r": 0.025},
    {"name": "obs_l2_bot", "cx": 0.575, "cy":  0.08,  "r": 0.025},
    {"name": "obs_l3_top", "cx": 0.35,  "cy":  0.26,  "r": 0.025},
    {"name": "obs_l3_mid", "cx": 0.5,   "cy":  0.26,  "r": 0.025},
    {"name": "obs_l3_bot", "cx": 0.65,  "cy":  0.26,  "r": 0.025},
]

GOAL_Y = OBS_LVL_ORIGIN[1] + 2.5 * OBS_LVL_OFFSET[1]  # 0.35
WORKSPACE = {"x_min": 0.293, "x_max": 0.707, "y_min": -0.3, "y_max": 0.38}


# ---------------------------------------------------------------------------
# Environment drawing helper (mirrors workspace code)
# ---------------------------------------------------------------------------
def draw_environment(ax: plt.Axes) -> None:
    """Draw workspace boundary, obstacles and goal line on *ax*."""
    ws = WORKSPACE
    ax.add_patch(plt.Rectangle(
        (ws["x_min"], ws["y_min"]),
        ws["x_max"] - ws["x_min"],
        ws["y_max"] - ws["y_min"],
        fill=False, edgecolor="gray", linewidth=2, linestyle="--",
    ))
    for obs in OBSTACLES:
        circle = plt.Circle(
            (obs["cx"], obs["cy"]), obs["r"],
            fill=True, facecolor="lightcoral", edgecolor="darkred",
            linewidth=1.5, alpha=0.6, zorder=3,
        )
        ax.add_patch(circle)
    ax.axhline(y=GOAL_Y, color="gold", linewidth=3, linestyle="-", zorder=5)


# ---------------------------------------------------------------------------
# Load dataset trajectories (agent_pos only)
# ---------------------------------------------------------------------------
def load_dataset(data_dir: Path) -> list[np.ndarray]:
    """Return list of (T, 2) arrays with *raw* agent_pos for every trajectory."""
    traj_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    trajs = []
    for traj_dir in traj_dirs:
        npz_path = traj_dir / "agent_pos.npz"
        if not npz_path.exists():
            continue
        data = np.load(str(npz_path))
        arr = data[data.files[0]]  # shape (T, 2)
        trajs.append(arr)
    return trajs


# ---------------------------------------------------------------------------
# Determine success: trajectory reaches above GOAL_Y
# ---------------------------------------------------------------------------
def is_success(traj: np.ndarray) -> bool:
    return bool(np.any(traj[:, 1] >= GOAL_Y))


# ---------------------------------------------------------------------------
# Plot all trajectories (mirrors _plot_all_trajectories in workspace)
# ---------------------------------------------------------------------------
def plot_all_trajectories(
    trajs: list[np.ndarray],
    success_state: list[bool],
    output_path: Path,
) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    ws = WORKSPACE
    x_margin = (ws["x_max"] - ws["x_min"]) * 0.05
    y_margin = (ws["y_max"] - ws["y_min"]) * 0.05

    for ax, flag, title_color, label in [
        (ax1, True,  "green", "Successful"),
        (ax2, False, "red",   "Failed"),
    ]:
        draw_environment(ax)
        subset = [t for t, s in zip(trajs, success_state) if s == flag]
        for traj in subset:
            timesteps = np.arange(len(traj))
            ax.scatter(traj[:, 0], traj[:, 1], c=timesteps, cmap="viridis",
                       s=20, alpha=0.5, edgecolors="none", zorder=10)
            ax.plot(traj[:, 0], traj[:, 1], "k-", alpha=0.2, linewidth=0.5, zorder=9)

        ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
        ax.set_title(f"{label} Trajectories (n={len(subset)})",
                     fontsize=14, color=title_color, fontweight="bold")
        ax.set_xlim(ws["x_min"] - x_margin, ws["x_max"] + x_margin)
        ax.set_ylim(ws["y_min"] - y_margin, ws["y_max"] + y_margin)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Plot individual trajectory (mirrors _plot_single_trajectory in workspace)
# ---------------------------------------------------------------------------
def plot_single_trajectory(
    traj: np.ndarray,
    episode_id: int,
    status: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    draw_environment(ax)

    timesteps = np.arange(len(traj))
    scatter = ax.scatter(traj[:, 0], traj[:, 1], c=timesteps, cmap="viridis",
                         s=60, alpha=0.8, edgecolors="black", linewidth=0.5, zorder=10)
    ax.plot(traj[:, 0], traj[:, 1], "k-", alpha=0.4, linewidth=1.5, zorder=9)
    ax.plot(traj[0, 0],  traj[0, 1],  "go", markersize=18, label="Start",
            markeredgecolor="black", markeredgewidth=2, zorder=11)
    ax.plot(traj[-1, 0], traj[-1, 1], "r*", markersize=22, label="End",
            markeredgecolor="black", markeredgewidth=2, zorder=11)

    plt.colorbar(scatter, ax=ax, label="Timestep", pad=0.02)

    ax.set_xlabel("X Position (m)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Y Position (m)", fontsize=13, fontweight="bold")
    color = "green" if status == "success" else "red"
    ax.set_title(f"Dataset Episode {episode_id:04d} - {status.upper()}\n"
                 f"Length: {len(traj)} steps",
                 fontsize=15, color=color, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    ws = WORKSPACE
    x_margin = (ws["x_max"] - ws["x_min"]) * 0.05
    y_margin = (ws["y_max"] - ws["y_min"]) * 0.05
    ax.set_xlim(ws["x_min"] - x_margin, ws["x_max"] + x_margin)
    ax.set_ylim(ws["y_min"] - y_margin, ws["y_max"] + y_margin)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Plot statistics (mirrors _plot_statistics in workspace)
# ---------------------------------------------------------------------------
def plot_statistics(
    trajs: list[np.ndarray],
    success_state: list[bool],
    output_path: Path,
) -> None:
    lengths = np.array([len(t) for t in trajs])
    success_arr = np.array(success_state)
    num_total = len(trajs)
    num_success = int(np.sum(success_arr))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].hist(lengths, bins=20, alpha=0.7, edgecolor="black")
    axes[0, 0].set_xlabel("Trajectory Length (steps)")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Trajectory Length Distribution")
    axes[0, 0].grid(True, alpha=0.3)

    # Final y position as proxy for goal distance
    final_y = np.array([t[-1, 1] for t in trajs])
    goal_dist = np.abs(final_y - GOAL_Y)
    axes[0, 1].hist(goal_dist, bins=20, alpha=0.7, edgecolor="black")
    axes[0, 1].set_xlabel("Final Distance to Goal Line (m)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Goal Distance Distribution")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].bar(["Success", "Failed"],
                   [num_success, num_total - num_success],
                   color=["green", "red"], alpha=0.7)
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title(f"Success Rate: {num_success / num_total:.1%}")
    axes[1, 0].grid(True, alpha=0.3, axis="y")

    # Path lengths
    path_lengths = []
    for traj in trajs:
        deltas = np.diff(traj, axis=0)
        path_lengths.append(float(np.sum(np.linalg.norm(deltas, axis=1))))
    path_lengths = np.array(path_lengths)
    axes[1, 1].boxplot(
        [path_lengths[success_arr], path_lengths[~success_arr]],
        tick_labels=["Success", "Failed"],
    )
    axes[1, 1].set_ylabel("Total Path Length (m)")
    axes[1, 1].set_title("Path Length by Outcome")
    axes[1, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Locate the repository root via this script's location
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    data_dir = repo_root / "data" / "obstacle_avoidance"

    if not data_dir.exists():
        print(f"ERROR: data directory not found at {data_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = repo_root / "results" / "dataset_visualization"
    episodes_dir = output_dir / "episodes"
    plots_dir = output_dir / "plots"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading trajectories from: {data_dir}")
    trajs = load_dataset(data_dir)
    print(f"Loaded {len(trajs)} trajectories")

    success_state = [is_success(t) for t in trajs]
    n_success = sum(success_state)
    n_failed = len(trajs) - n_success
    print(f"Success: {n_success}, Failed: {n_failed}, "
          f"Rate: {n_success / len(trajs):.1%}")

    # ---- Summary: all-trajectories plot (same as workspace evaluation) ----
    plot_all_trajectories(trajs, success_state, plots_dir / "all_trajectories.png")

    # ---- Summary statistics ----
    plot_statistics(trajs, success_state, plots_dir / "statistics.png")

    # ---- Per-trajectory plots ----
    print("Saving individual episode plots …")
    for i, (traj, success) in enumerate(zip(trajs, success_state)):
        status = "success" if success else "failed"
        fname = episodes_dir / f"{status}_episode_{i:04d}.png"
        plot_single_trajectory(traj, i, status, fname)

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
