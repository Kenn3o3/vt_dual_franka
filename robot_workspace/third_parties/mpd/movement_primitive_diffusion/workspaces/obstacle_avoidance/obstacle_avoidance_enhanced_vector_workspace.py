"""
Enhanced Obstacle Avoidance Vector Workspace with trajectory logging and visualization
"""
from typing import Optional
import numpy as np
from omegaconf import DictConfig
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import hydra

from movement_primitive_diffusion.workspaces.obstacle_avoidance.obstacle_avoidance_vector_workspace import ObstacleAvoidanceEnvVectorWorkspace
from movement_primitive_diffusion.agents.base_agent import BaseAgent


class ObstacleAvoidanceEnhancedVectorWorkspace(ObstacleAvoidanceEnvVectorWorkspace):
    """
    Enhanced workspace that saves:
    1. Model output trajectories (actions, observations, positions)
    2. Trajectory visualization plots
    3. Detailed logs
    """
    
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
        save_trajectories: bool = True,
        save_plots: bool = True,
    ):
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
            seed=seed,
        )
        
        self.save_trajectories = save_trajectories
        self.save_plots = save_plots
        
    def test_agent(self, agent: BaseAgent, num_trajectories: int = 10) -> dict:
        """Override test_agent to save trajectories and plots"""
        # Call parent test_agent
        result = super().test_agent(agent, num_trajectories)
        
        # Get output directory
        base_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        self.trajectory_dir = base_dir / "trajectories"
        self.plot_dir = base_dir / "plots"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trajectory data
        if self.save_trajectories:
            self._save_trajectories(num_trajectories)
        
        # Generate plots
        if self.save_plots:
            self._generate_plots(num_trajectories)
        
        return result
    
    def _save_trajectories(self, num_trajectories: int):
        """Save individual trajectory data"""
        print("\n💾 Saving trajectory data...")
        
        final_t = self.hooks["episode_lengths"].astype(int)
        success_state = self.hooks["success_state"]
        
        trajectory_summary = {
            'total_trajectories': num_trajectories,
            'successful_trajectories': int(np.sum(success_state)),
            'failed_trajectories': int(num_trajectories - np.sum(success_state)),
            'mean_episode_length': float(np.mean(self.hooks["episode_lengths"])),
            'trajectories': []
        }
        
        for i in range(num_trajectories):
            status = "success" if success_state[i] else "failed"
            episode_length = int(final_t[i])
            
            # Extract trajectory data
            eef_pos = self.hooks["eef_pos"][i, :episode_length+1]
            eef_vel = self.hooks["eef_vel"][i, :episode_length+1]
            rewards = self.hooks["reward"][i, :episode_length+1]
            
            # Save as npz file
            npz_path = self.trajectory_dir / f"trajectory_{i:04d}_{status}.npz"
            np.savez(
                npz_path,
                eef_pos=eef_pos,
                eef_vel=eef_vel,
                rewards=rewards,
                episode_length=episode_length,
                successful=success_state[i],
                truncated=self.hooks["truncation_state"][i],
                final_goal_distance=self.hooks["final_goal_distance"][i],
                total_reward=np.nansum(rewards)
            )
            
            # Add to summary
            trajectory_summary['trajectories'].append({
                'id': i,
                'status': status,
                'episode_length': episode_length,
                'successful': bool(success_state[i]),
                'truncated': bool(self.hooks["truncation_state"][i]),
                'final_goal_distance': float(self.hooks["final_goal_distance"][i]),
                'total_reward': float(np.nansum(rewards)),
                'file': str(npz_path.name)
            })
        
        # Save summary
        summary_path = self.trajectory_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(trajectory_summary, f, indent=2)
        
        print(f"✅ Saved {num_trajectories} trajectories to {self.trajectory_dir}")
        print(f"   Success: {trajectory_summary['successful_trajectories']}, "
              f"Failed: {trajectory_summary['failed_trajectories']}")
    
    def _generate_plots(self, num_trajectories: int):
        """Generate visualization plots"""
        print("\n📊 Generating trajectory plots...")
        
        final_t = self.hooks["episode_lengths"].astype(int)
        success_state = self.hooks["success_state"]
        
        # Plot 1: Trajectory paths (2D)
        self._plot_trajectory_paths(num_trajectories, final_t, success_state)
        
        # Plot 2: Rewards over time
        self._plot_rewards(num_trajectories, final_t, success_state)
        
        # Plot 3: Velocity profiles
        self._plot_velocities(num_trajectories, final_t, success_state)
        
        # Plot 4: Statistics summary
        self._plot_statistics(num_trajectories, success_state)
        
        print(f"✅ Plots saved to {self.plot_dir}")
    
    def _plot_trajectory_paths(self, num_trajectories, final_t, success_state):
        """Plot 2D trajectory paths"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot successful trajectories
        for i in range(num_trajectories):
            if success_state[i]:
                traj = self.hooks["eef_pos"][i, :final_t[i]+1]
                ax1.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=2)
                ax1.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start' if i == 0 else '')
                ax1.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=12, label='End' if i == 0 else '')
        
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.set_title(f'Successful Trajectories (n={np.sum(success_state)})')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.axis('equal')
        
        # Plot failed trajectories
        for i in range(num_trajectories):
            if not success_state[i]:
                traj = self.hooks["eef_pos"][i, :final_t[i]+1]
                ax2.plot(traj[:, 0], traj[:, 1], alpha=0.6, linewidth=2)
                ax2.plot(traj[0, 0], traj[0, 1], 'go', markersize=8, label='Start' if i == 0 else '')
                ax2.plot(traj[-1, 0], traj[-1, 1], 'r*', markersize=12, label='End' if i == 0 else '')
        
        ax2.set_xlabel('X Position (m)')
        ax2.set_ylabel('Y Position (m)')
        ax2.set_title(f'Failed Trajectories (n={num_trajectories - np.sum(success_state)})')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.axis('equal')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'trajectory_paths.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_rewards(self, num_trajectories, final_t, success_state):
        """Plot rewards over time"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Cumulative rewards for successful trajectories
        for i in range(num_trajectories):
            if success_state[i]:
                rewards = self.hooks["reward"][i, :final_t[i]+1]
                cumsum_rewards = np.nancumsum(rewards)
                ax1.plot(cumsum_rewards, alpha=0.6, linewidth=1.5)
        
        ax1.set_xlabel('Timestep')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title(f'Successful Trajectories (n={np.sum(success_state)})')
        ax1.grid(True, alpha=0.3)
        
        # Cumulative rewards for failed trajectories
        for i in range(num_trajectories):
            if not success_state[i]:
                rewards = self.hooks["reward"][i, :final_t[i]+1]
                cumsum_rewards = np.nancumsum(rewards)
                ax2.plot(cumsum_rewards, alpha=0.6, linewidth=1.5)
        
        ax2.set_xlabel('Timestep')
        ax2.set_ylabel('Cumulative Reward')
        ax2.set_title(f'Failed Trajectories (n={num_trajectories - np.sum(success_state)})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'rewards_over_time.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_velocities(self, num_trajectories, final_t, success_state):
        """Plot velocity profiles"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Successful - X velocity
        for i in range(num_trajectories):
            if success_state[i]:
                vel = self.hooks["eef_vel"][i, :final_t[i]+1, 0]
                axes[0, 0].plot(vel, alpha=0.5, linewidth=1)
        axes[0, 0].set_xlabel('Timestep')
        axes[0, 0].set_ylabel('X Velocity (m/s)')
        axes[0, 0].set_title(f'Successful - X Velocity (n={np.sum(success_state)})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Successful - Y velocity
        for i in range(num_trajectories):
            if success_state[i]:
                vel = self.hooks["eef_vel"][i, :final_t[i]+1, 1]
                axes[0, 1].plot(vel, alpha=0.5, linewidth=1)
        axes[0, 1].set_xlabel('Timestep')
        axes[0, 1].set_ylabel('Y Velocity (m/s)')
        axes[0, 1].set_title(f'Successful - Y Velocity (n={np.sum(success_state)})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Failed - X velocity
        for i in range(num_trajectories):
            if not success_state[i]:
                vel = self.hooks["eef_vel"][i, :final_t[i]+1, 0]
                axes[1, 0].plot(vel, alpha=0.5, linewidth=1)
        axes[1, 0].set_xlabel('Timestep')
        axes[1, 0].set_ylabel('X Velocity (m/s)')
        axes[1, 0].set_title(f'Failed - X Velocity (n={num_trajectories - np.sum(success_state)})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Failed - Y velocity
        for i in range(num_trajectories):
            if not success_state[i]:
                vel = self.hooks["eef_vel"][i, :final_t[i]+1, 1]
                axes[1, 1].plot(vel, alpha=0.5, linewidth=1)
        axes[1, 1].set_xlabel('Timestep')
        axes[1, 1].set_ylabel('Y Velocity (m/s)')
        axes[1, 1].set_title(f'Failed - Y Velocity (n={num_trajectories - np.sum(success_state)})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'velocity_profiles.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_statistics(self, num_trajectories, success_state):
        """Plot summary statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Episode length distribution
        axes[0, 0].hist(self.hooks["episode_lengths"], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(np.mean(self.hooks["episode_lengths"]), color='r', linestyle='--', 
                          label=f'Mean: {np.mean(self.hooks["episode_lengths"]):.1f}')
        axes[0, 0].set_xlabel('Episode Length')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Episode Length Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Goal distance distribution
        axes[0, 1].hist(self.hooks["final_goal_distance"], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(np.mean(self.hooks["final_goal_distance"]), color='r', linestyle='--',
                          label=f'Mean: {np.mean(self.hooks["final_goal_distance"]):.3f}')
        axes[0, 1].set_xlabel('Final Goal Distance (m)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Final Goal Distance Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Success vs Failed
        success_count = np.sum(success_state)
        failed_count = num_trajectories - success_count
        axes[1, 0].bar(['Success', 'Failed'], [success_count, failed_count], 
                      color=['green', 'red'], alpha=0.7)
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title(f'Success vs Failed (Rate: {success_count/num_trajectories:.1%})')
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # Total rewards comparison
        total_rewards = np.nansum(self.hooks["reward"], axis=1)
        success_rewards = total_rewards[success_state]
        failed_rewards = total_rewards[~success_state]
        
        box_data = []
        labels = []
        if len(success_rewards) > 0:
            box_data.append(success_rewards)
            labels.append('Success')
        if len(failed_rewards) > 0:
            box_data.append(failed_rewards)
            labels.append('Failed')
        
        if len(box_data) > 0:
            axes[1, 1].boxplot(box_data, labels=labels)
            axes[1, 1].set_ylabel('Total Reward')
            axes[1, 1].set_title('Reward Distribution by Outcome')
            axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.plot_dir / 'statistics.png', dpi=150, bbox_inches='tight')
        plt.close()
