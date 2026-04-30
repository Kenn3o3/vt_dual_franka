from typing import Optional

import numpy as np
import json
import hydra
from pathlib import Path

from omegaconf import DictConfig

from movement_primitive_diffusion.agents.base_agent import BaseAgent
from movement_primitive_diffusion.workspaces.base_vector_workspace import BaseVectorWorkspace
from movement_primitive_diffusion.workspaces.obstacle_avoidance.obstacle_avoidance_env import (
    Mode,
)
from movement_primitive_diffusion.workspaces.obstacle_avoidance.obstacle_avoidance_utils import (
  plotly_trajectories,
  plotly_trajectory_modes,
)
import swanlab
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

class ObstacleAvoidanceEnvVectorWorkspace(BaseVectorWorkspace):
    def __init__(
        self,
        env_config: DictConfig,
        t_act: int,
        num_parallel_envs: int,
        shared_memory: bool = False,
        async_vector_env: bool = True,
        num_upload_successful_videos: int = 5,
        num_upload_failed_videos: int = 5,
        video_dt: Optional[float] = None,
        show_images: bool = False,
        annotate_videos: bool = True,
        seed: Optional[int] = None,
    ):
        if video_dt is None:
            video_dt = env_config["control_dt"]

        self.dt = env_config["control_dt"]

        super().__init__(
            env_config=env_config,
            t_act=t_act,
            num_parallel_envs=num_parallel_envs,
            shared_memory=shared_memory,
            async_vector_env=async_vector_env,
            num_upload_successful_videos=num_upload_successful_videos,
            num_upload_failed_videos=num_upload_failed_videos,
            video_dt=video_dt,
            show_images=show_images,
            annotate_videos=annotate_videos,
        )

        self.seed = seed or np.random.randint(0, 2**32 - 1)
        self.trajectory_seeds: list[np.random.SeedSequence]

        # Discount factor for calculating return values from step rewards
        self.gamma = 1.0

    def render_function(self, caller_locals: dict) -> np.ndarray:
        return self.vector_env.call("render")

    def check_success_hook(self, caller_locals: dict) -> bool:
        return caller_locals["env_info"]["success"][caller_locals["env_index"]]

    def reset_env(self, caller_locals: dict) -> tuple[np.ndarray, dict]:
        # We initialize this list with None because we might have fewer trajectories left than we have parallel envs
        seeds = [None for _ in range(self.num_parallel_envs)]

        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            if (traj_idx := env_index_offset + env_index) < len(self.trajectory_seeds):
                seeds[env_index] = self.trajectory_seeds[traj_idx].entropy
            else:
                break

        obs, infos = self.vector_env.reset(seed=seeds)

        for env_index in range(self.num_parallel_envs):
            if (traj_idx := env_index_offset + env_index) < caller_locals["num_trajectories"]:
                self.hooks["eef_pos"][traj_idx, 0] = infos["eef_pos"][env_index][:2]
                self.hooks["eef_vel"][traj_idx, 0] = infos["eef_vel"][env_index][:2]
        return obs, infos


    def post_step_hook(self, caller_locals: dict) -> None:
        rewards = caller_locals["env_reward"]
        infos = caller_locals["env_info"]

        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            if (traj_idx := env_index_offset + env_index) < caller_locals["num_trajectories"]:
                t = int(self.hooks["episode_lengths"][traj_idx])
                if not caller_locals["done_buffer"][env_index]:
                    self.hooks["episode_lengths"][traj_idx] += 1
                # note that we keep overwriting the value at the last timestep when done
                # this works since AsyncVectorEnv does not reset the envs until all are done
                # a finished env just returns the last observation, info, and reward on subsequent steps
                if t <= self.hooks["episode_lengths"][traj_idx]:
                    # t+1 since the first timestep is the initial state, cf. reset
                    self.hooks["reward"][traj_idx, t + 1] = rewards[env_index]
                    self.hooks["eef_pos"][traj_idx, t + 1] = infos["eef_pos"][env_index][:2]
                    self.hooks["eef_vel"][traj_idx, t + 1] = infos["eef_vel"][env_index][:2]

    def post_episode_hook(self, caller_locals: dict) -> None:
        infos = caller_locals["env_info"]
        env_index_offset = caller_locals["episode_sequence_index"] * self.num_parallel_envs
        for env_index in range(self.num_parallel_envs):
            if (traj_idx := env_index_offset + env_index) < caller_locals["num_trajectories"]:
                distance = np.linalg.norm(
                    infos["eef_pos"][env_index][1] - infos["goal_pos"][env_index][1]
                ).mean()
                self.hooks["final_goal_distance"][traj_idx] = distance

                mode = Mode.from_encoding(infos["mode"][env_index])
                self.hooks["final_mode"][traj_idx] = mode

                truncated = caller_locals["env_truncated"][env_index]
                self.hooks["truncation_state"][traj_idx] = truncated

                success = infos["success"][env_index]
                self.hooks["success_state"][traj_idx] = success

    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        # Ensure create_vectorized_env() has run so self.envs_fns is available
        # before _get_environment_info() tries to access it.
        if self.vector_env is None:
            self.create_vectorized_env()

        # From a fixed start seed, create a seed list of length num_trajectories. These will be used to reset the envs
        seed_sequence = np.random.SeedSequence(self.seed)
        self.trajectory_seeds = seed_sequence.spawn(num_trajectories)

        self.hooks = {
            "episode_lengths": np.zeros(num_trajectories),
            "truncation_state": np.full(num_trajectories, False),
            "success_state": np.full(num_trajectories, False),
            "final_goal_distance": np.full(num_trajectories, np.inf),
            "reward": np.full((num_trajectories, self.time_limit + 1), np.nan),
            "eef_pos": np.full((num_trajectories, self.time_limit + 1, 2), np.nan),
            "eef_vel": np.full((num_trajectories, self.time_limit + 1, 2), np.nan),
            "final_mode": [Mode() for _ in range(num_trajectories)],
        }
        
        # Get environment info for visualization
        self.env_info = self._get_environment_info()

        # Call the parent's test agent function
        result_dict = super().test_agent(agent, num_trajectories)
        
        # Compute additional metrics including successful_mode_entropy
        additional_metrics = self._compute_test_metrics(num_trajectories)
        result_dict.update(additional_metrics)
        
        # Save trajectories and generate plots
        self._save_and_visualize_trajectories(num_trajectories)
        
        return result_dict
    
    def _get_environment_info(self):
        """Get environment configuration for visualization"""
        # Create a temporary env to extract scene info
        # For AsyncVectorEnv, we need to create a new env instance
        temp_env = self.envs_fns[0]()
        
        # Reset to load the scene
        temp_env.reset()
        
        # Workspace limits
        workspace_limits = temp_env.workspace_limits  # [[x_min, y_min], [x_max, y_max]]
        
        # Goal position
        goal_pos = temp_env.scene.get_goal()[:2]  # [x, y]
        
        # Obstacle positions and sizes (cylinders)
        obstacles = []
        for obs_name, obs_body in temp_env.scene.obstacles.items():
            pos = obs_body.pos()  # [x, y, z]
            aabb = obs_body.aabb()  # [[x_min, y_min, z_min], [x_max, y_max, z_max]]
            # Calculate radius from AABB (cylinder)
            radius = (aabb[1, 0] - aabb[0, 0]) / 2
            obstacles.append({
                'name': obs_name,
                'center_x': pos[0],
                'center_y': pos[1],
                'radius': radius,
            })
        
        # Close the temporary env
        temp_env.close()
        
        return {
            'workspace_limits': workspace_limits,
            'goal_pos': goal_pos,
            'obstacles': obstacles,
        }
    
    def _compute_test_metrics(self, num_trajectories: int) -> dict:
        """Compute test metrics including successful_mode_entropy"""
        result_dict = {}
        
        # episode length
        final_t = self.hooks["episode_lengths"].astype(int)
        result_dict["mean_episode_length"] = np.mean(self.hooks["episode_lengths"])

        # truncation rate
        result_dict["truncation_rate"] = np.mean(self.hooks["truncation_state"])

        # return
        rewards = self.hooks["reward"]
        episode_returns = np.array([reward * self.gamma**t for t, reward in enumerate(rewards)])
        result_dict["mean_return"] = np.nanmean(episode_returns)
        result_dict["min_return"] = np.nanmin(episode_returns)
        result_dict["max_return"] = np.nanmax(episode_returns)

        # reward
        result_dict["mean_max_reward"] = np.nanmean(np.nanmax(rewards, axis=-1))
        final_rewards = rewards[np.arange(num_trajectories), final_t + 1]
        result_dict["mean_final_reward"] = np.nanmean(final_rewards)

        # tool path length
        cartesian_tool_positions = self.hooks["eef_pos"]
        cartesian_tool_position_deltas = np.diff(cartesian_tool_positions, axis=-2)
        cartesian_tool_path_length = np.nansum(np.linalg.norm(cartesian_tool_position_deltas, axis=-1), axis=-1)
        result_dict["mean_tool_path_length"] = np.mean(cartesian_tool_path_length)
        result_dict["min_tool_path_length"] = np.min(cartesian_tool_path_length)
        result_dict["max_tool_path_length"] = np.max(cartesian_tool_path_length)

        # tool acceleration
        tool_velocity = self.hooks["eef_vel"]
        tool_acceleration = np.linalg.norm(np.diff(tool_velocity, axis=-2) / self.dt, axis=-1)
        result_dict["mean_tool_acceleration"] = np.nanmean(tool_acceleration)
        result_dict["min_tool_acceleration"] = np.nanmin(tool_acceleration)
        result_dict["max_tool_acceleration"] = np.nanmax(tool_acceleration)

        # tool energy
        tool_energy = np.nansum(tool_acceleration, axis=-1)
        result_dict["mean_tool_energy"] = np.mean(tool_energy)
        result_dict["min_tool_energy"] = np.min(tool_energy)
        result_dict["max_tool_energy"] = np.max(tool_energy)

        # tool jerk
        tool_jerk = np.zeros_like(tool_acceleration)
        tool_jerk[:, 1:] = np.abs(np.diff(tool_acceleration, axis=-1)) / self.dt
        result_dict["mean_tool_jerk"] = np.nanmean(tool_jerk)
        result_dict["min_tool_jerk"] = np.nanmin(tool_jerk)
        result_dict["max_tool_jerk"] = np.nanmax(tool_jerk)

        # goal distance
        result_dict["mean_final_goal_distance"] = np.mean(self.hooks["final_goal_distance"])

        # final modes
        final_modes = self.hooks["final_mode"]
        _, entropy = Mode.compute_distribution(final_modes)
        result_dict["mode_entropy"] = entropy
        # Only log arrays when SwanLab is disabled (for local analysis)
        if swanlab.get_run() is None:
            result_dict["modes_dec"] = np.array([mode.decode() for mode in final_modes])

        # successful modes
        success_state = self.hooks["success_state"]
        success_modes = [final_modes[i] for i in range(len(final_modes)) if success_state[i]]
        _, success_entropy = Mode.compute_distribution(success_modes)
        result_dict["successful_mode_entropy"] = success_entropy
        # Only log arrays when SwanLab is disabled (for local analysis)
        if swanlab.get_run() is None:
            result_dict["successful_modes_dec"] = np.array([mode.decode() for mode in success_modes])

        # trajectories
        # strip nan values from trajectories
        failed_trajs = [
            self.hooks["eef_pos"][i, :final_t[i]+1]
            for i in np.arange(num_trajectories)[np.logical_not(success_state)]
        ]
        success_trajs = [
            self.hooks["eef_pos"][i, :final_t[i]+1]
            for i in np.arange(num_trajectories)[success_state]
        ]
        # Log trajectory visualizations
        # Note: Plotly figures are skipped for SwanLab to avoid Chrome dependency
        # Only log numeric metrics
        if swanlab.get_run() is None:
            # Keep Plotly figures when not using SwanLab (for compatibility)
            failed_fig = plotly_trajectories(
                trajs=failed_trajs,
                traj_labels=[str(i) for i in np.arange(num_trajectories)[np.logical_not(success_state)]],
                title="Failed trajectories",
            )
            success_fig = plotly_trajectories(
                trajs=success_trajs,
                traj_labels=[str(i) for i in np.arange(num_trajectories)[success_state]],
                title="Successful trajectories",
            )
            modes_fig = plotly_trajectory_modes(
                traj_modes=final_modes,
                title="Trajectory Modes",
            )
            result_dict["failed_trajectories"] = failed_fig
            result_dict["successful_trajectories"] = success_fig
            result_dict["trajectory_modes"] = modes_fig

        return result_dict

    def get_result_dict_keys(self) -> list[str]:
        super_keys = super().get_result_dict_keys()
        return super_keys + [
            "mean_episode_length",
            "truncation_rate",
            "mean_return",
            "min_return",
            "max_return",
            "mean_max_reward",
            "mean_final_reward",
            "mean_final_goal_distance",
            "modes_dec",
            "mode_entropy",
            "successful_modes_dec",
            "successful_mode_entropy",
            "failed_trajectories",
            "successful_trajectories",
            "trajectory_modes",
            "mean_tool_path_length",
            "min_tool_path_length",
            "max_tool_path_length",
            "mean_tool_acceleration",
            "min_tool_acceleration",
            "max_tool_acceleration",
            "mean_tool_energy",
            "min_tool_energy",
            "max_tool_energy",
            "mean_tool_jerk",
            "min_tool_jerk",
            "max_tool_jerk",
        ]
    
    def _save_and_visualize_trajectories(self, num_trajectories: int):
        """Save trajectory data and generate visualization plots"""
        try:
            # Use Hydra output directory directly (respects method_name parameter)
            output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            
            # Get current epoch (default to 0 if not set)
            current_epoch = getattr(self, 'current_epoch', 0)
            
            # Create epoch-specific directory
            epoch_dir = output_dir / f"epoch_{current_epoch:04d}"
            episodes_dir = epoch_dir / "episodes"
            episodes_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"\n💾 Saving episodes data...")
            
            final_t = self.hooks["episode_lengths"].astype(int)
            success_state = self.hooks["success_state"]
            
            # Save individual episodes with unified naming
            for i in range(num_trajectories):
                status = "success" if success_state[i] else "failed"
                episode_length = int(final_t[i])
                
                eef_pos = self.hooks["eef_pos"][i, :episode_length+1]
                eef_vel = self.hooks["eef_vel"][i, :episode_length+1]
                rewards = self.hooks["reward"][i, :episode_length+1]
                
                # Save as JSON (human-readable) with status prefix
                json_data = {
                    'episode_id': i,
                    'epoch': current_epoch,
                    'status': status,
                    'episode_length': episode_length,
                    'successful': bool(success_state[i]),
                    'final_goal_distance': float(self.hooks["final_goal_distance"][i]),
                    'total_reward': float(np.nansum(rewards)),
                    'trajectory': {
                        'x': eef_pos[:, 0].tolist(),
                        'y': eef_pos[:, 1].tolist(),
                        'vx': eef_vel[:, 0].tolist(),
                        'vy': eef_vel[:, 1].tolist(),
                        'rewards': np.nan_to_num(rewards, nan=0.0).tolist(),
                    }
                }
                json_path = episodes_dir / f"{status}_episode_{i:04d}.json"
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)
                
                # Generate individual trajectory plot with time-colored path
                self._plot_single_trajectory(episodes_dir, i, eef_pos, rewards, status)
            
            # Save summary
            summary = {
                'total': num_trajectories,
                'successful': int(np.sum(success_state)),
                'failed': int(num_trajectories - np.sum(success_state)),
                'mean_length': float(np.mean(self.hooks["episode_lengths"])),
                'success_rate': float(np.sum(success_state) / num_trajectories),
            }
            with open(episodes_dir / "summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            print(f"✅ Saved {num_trajectories} episodes (Success: {summary['successful']}, Failed: {summary['failed']})")
            
            # Generate summary plots
            print(f"📊 Generating summary plots...")
            plot_dir = epoch_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            self._plot_all_trajectories(plot_dir, num_trajectories, final_t, success_state)
            self._plot_statistics(plot_dir, num_trajectories, success_state)
            print(f"✅ Epoch {current_epoch}: Plots saved to {plot_dir}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to save episodes/plots: {e}")
            import traceback
            traceback.print_exc()
    
    def _plot_single_trajectory(self, episodes_dir, episode_id, eef_pos, rewards, status):
        """Plot single trajectory with time-colored path and environment info"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        # Plot environment elements first (so trajectory is on top)
        env_info = self.env_info
        
        # 1. Draw workspace boundaries
        ws_limits = env_info['workspace_limits']
        ax.add_patch(plt.Rectangle(
            (ws_limits[0, 0], ws_limits[0, 1]),
            ws_limits[1, 0] - ws_limits[0, 0],
            ws_limits[1, 1] - ws_limits[0, 1],
            fill=False, edgecolor='gray', linewidth=2, linestyle='--', 
            label='Workspace'
        ))
        
        # 2. Draw obstacles (circles for top-down view of cylinders)
        for obs in env_info['obstacles']:
            circle = plt.Circle(
                (obs['center_x'], obs['center_y']),
                obs['radius'],
                fill=True, facecolor='lightcoral', edgecolor='darkred', 
                linewidth=2, alpha=0.7, zorder=3
            )
            ax.add_patch(circle)
        
        # Add single obstacle label
        if env_info['obstacles']:
            ax.plot([], [], 'o', color='lightcoral', markersize=10, 
                   markeredgecolor='darkred', markeredgewidth=2, label='Obstacles')
        
        # 3. Draw goal line
        goal_pos = env_info['goal_pos']
        ax.axhline(y=goal_pos[1], color='gold', linewidth=3, linestyle='-', 
                  label='Goal Line', zorder=5)
        
        # 4. Plot trajectory with color representing time
        timesteps = np.arange(len(eef_pos))
        scatter = ax.scatter(eef_pos[:, 0], eef_pos[:, 1], 
                           c=timesteps, cmap='viridis', 
                           s=60, alpha=0.8, edgecolors='black', linewidth=0.5,
                           zorder=10)
        
        # Draw lines connecting points
        ax.plot(eef_pos[:, 0], eef_pos[:, 1], 'k-', alpha=0.4, linewidth=1.5, zorder=9)
        
        # 5. Mark start and end
        ax.plot(eef_pos[0, 0], eef_pos[0, 1], 'go', markersize=18, 
               label='Start', markeredgecolor='black', markeredgewidth=2, zorder=11)
        ax.plot(eef_pos[-1, 0], eef_pos[-1, 1], 'r*', markersize=22, 
               label='End', markeredgecolor='black', markeredgewidth=2, zorder=11)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, label='Timestep', pad=0.02)
        
        # Labels and title
        ax.set_xlabel('X Position (m)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=13, fontweight='bold')
        color = 'green' if status == 'success' else 'red'
        ax.set_title(f'Episode {episode_id:04d} - {status.upper()}\n'
                    f'Length: {len(eef_pos)} steps, Total Reward: {np.nansum(rewards):.2f}',
                    fontsize=15, color=color, fontweight='bold')
        ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(episodes_dir / f'{status}_episode_{episode_id:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_all_trajectories(self, plot_dir, num_trajectories, final_t, success_state):
        """Plot all trajectories together with environment info"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        
        env_info = self.env_info
        
        # Helper function to draw environment on axis
        def draw_environment(ax):
            # Workspace boundaries
            ws_limits = env_info['workspace_limits']
            ax.add_patch(plt.Rectangle(
                (ws_limits[0, 0], ws_limits[0, 1]),
                ws_limits[1, 0] - ws_limits[0, 0],
                ws_limits[1, 1] - ws_limits[0, 1],
                fill=False, edgecolor='gray', linewidth=2, linestyle='--'
            ))
            
            # Obstacles (circles for top-down view of cylinders)
            for obs in env_info['obstacles']:
                circle = plt.Circle(
                    (obs['center_x'], obs['center_y']),
                    obs['radius'],
                    fill=True, facecolor='lightcoral', edgecolor='darkred', 
                    linewidth=1.5, alpha=0.6, zorder=3
                )
                ax.add_patch(circle)
            
            # Goal line
            goal_pos = env_info['goal_pos']
            ax.axhline(y=goal_pos[1], color='gold', linewidth=3, linestyle='-', zorder=5)
        
        # Set consistent axis limits based on workspace boundaries
        ws_limits = env_info['workspace_limits']
        x_min, x_max = ws_limits[0, 0], ws_limits[1, 0]
        y_min, y_max = ws_limits[0, 1], ws_limits[1, 1]
        # Add small margin
        x_margin = (x_max - x_min) * 0.05
        y_margin = (y_max - y_min) * 0.05
        
        # Successful trajectories
        draw_environment(ax1)
        for i in range(num_trajectories):
            if success_state[i]:
                traj = self.hooks["eef_pos"][i, :final_t[i]+1]
                timesteps = np.arange(len(traj))
                ax1.scatter(traj[:, 0], traj[:, 1], c=timesteps, cmap='viridis',
                          s=20, alpha=0.5, edgecolors='none', zorder=10)
                ax1.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.2, linewidth=0.5, zorder=9)
        ax1.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Successful Trajectories (n={np.sum(success_state)})', 
                     fontsize=14, color='green', fontweight='bold')
        ax1.set_xlim(x_min - x_margin, x_max + x_margin)
        ax1.set_ylim(y_min - y_margin, y_max + y_margin)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # Failed trajectories
        draw_environment(ax2)
        for i in range(num_trajectories):
            if not success_state[i]:
                traj = self.hooks["eef_pos"][i, :final_t[i]+1]
                timesteps = np.arange(len(traj))
                ax2.scatter(traj[:, 0], traj[:, 1], c=timesteps, cmap='viridis',
                          s=20, alpha=0.5, edgecolors='none', zorder=10)
                ax2.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.2, linewidth=0.5, zorder=9)
        ax2.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
        ax2.set_title(f'Failed Trajectories (n={num_trajectories - np.sum(success_state)})', 
                     fontsize=14, color='red', fontweight='bold')
        ax2.set_xlim(x_min - x_margin, x_max + x_margin)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'all_trajectories.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_statistics(self, plot_dir, num_trajectories, success_state):
        """Plot summary statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        axes[0, 0].hist(self.hooks["episode_lengths"], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Episode Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Episode Length Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(self.hooks["final_goal_distance"], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Final Goal Distance (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Goal Distance Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        success_count = np.sum(success_state)
        axes[1, 0].bar(['Success', 'Failed'], [success_count, num_trajectories - success_count], 
                      color=['green', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Success Rate: {success_count/num_trajectories:.1%}')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        total_rewards = np.nansum(self.hooks["reward"], axis=1)
        axes[1, 1].boxplot([total_rewards[success_state], total_rewards[~success_state]], 
                          labels=['Success', 'Failed'])
        axes[1, 1].set_ylabel('Total Reward')
        axes[1, 1].set_title('Reward by Outcome')
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(plot_dir / 'statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
