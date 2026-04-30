import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from .base import BaseVisualizer


class StatisticsVisualizer(BaseVisualizer):
    """Visualizer for statistics and metrics."""
    
    def __init__(self, dpi: int = 150, figsize: Tuple[float, float] = (12, 8)):
        """Initialize statistics visualizer.
        
        Args:
            dpi: Dots per inch for saved figures
            figsize: Figure size (width, height) in inches
        """
        super().__init__(dpi=dpi)
        self.figsize = figsize
    
    def plot_rewards(
        self,
        rewards: np.ndarray,
        title: str = "Rewards Over Time",
        xlabel: str = "Episode",
        ylabel: str = "Reward",
        show_mean: bool = True,
        **kwargs
    ):
        """Plot rewards over episodes.
        
        Args:
            rewards: Array of rewards, shape (num_episodes,)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            show_mean: Whether to show mean line
            **kwargs: Additional arguments passed to plot
        """
        self.close()
        
        self.fig, ax = plt.subplots(figsize=self.figsize)
        
        episodes = np.arange(len(rewards))
        ax.plot(episodes, rewards, alpha=0.6, label='Reward', **kwargs)
        
        if show_mean and len(rewards) > 0:
            mean_reward = np.mean(rewards)
            ax.axhline(y=mean_reward, color='r', linestyle='--', 
                      label=f'Mean: {mean_reward:.2f}')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.fig
    
    def plot_success_rate(
        self,
        success_flags: np.ndarray,
        window_size: int = 100,
        title: str = "Success Rate Over Time",
        **kwargs
    ):
        """Plot success rate with moving average.
        
        Args:
            success_flags: Boolean array indicating success, shape (num_episodes,)
            window_size: Window size for moving average
            title: Plot title
            **kwargs: Additional arguments
        """
        self.close()
        
        self.fig, ax = plt.subplots(figsize=self.figsize)
        
        episodes = np.arange(len(success_flags))
        
        # Plot individual successes/failures
        ax.scatter(episodes[success_flags], 
                  np.ones(np.sum(success_flags)),
                  color='green', marker='o', s=20, alpha=0.5, label='Success')
        ax.scatter(episodes[~success_flags], 
                  np.zeros(np.sum(~success_flags)),
                  color='red', marker='x', s=20, alpha=0.5, label='Failure')
        
        # Plot moving average
        if len(success_flags) >= window_size:
            moving_avg = np.convolve(success_flags.astype(float), 
                                    np.ones(window_size)/window_size, 
                                    mode='valid')
            ax.plot(episodes[window_size-1:], moving_avg, 
                   color='blue', linewidth=2, 
                   label=f'Success Rate (window={window_size})')
        
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success")
        ax.set_title(title)
        ax.set_ylim([-0.1, 1.1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.fig
    
    def plot_metrics_grid(
        self,
        metrics: Dict[str, np.ndarray],
        title: str = "Training Metrics",
        **kwargs
    ):
        """Plot multiple metrics in a grid layout.
        
        Args:
            metrics: Dictionary mapping metric names to arrays
            title: Overall plot title
            **kwargs: Additional arguments
        """
        self.close()
        
        num_metrics = len(metrics)
        if num_metrics == 0:
            raise ValueError("No metrics to plot")
        
        # Determine grid layout
        ncols = min(2, num_metrics)
        nrows = (num_metrics + ncols - 1) // ncols
        
        self.fig, axes = plt.subplots(nrows, ncols, figsize=(self.figsize[0], self.figsize[1] * nrows / 2))
        
        if num_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (name, values) in enumerate(metrics.items()):
            ax = axes[i]
            episodes = np.arange(len(values))
            ax.plot(episodes, values)
            ax.set_xlabel("Episode")
            ax.set_ylabel(name)
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(num_metrics, len(axes)):
            axes[i].axis('off')
        
        self.fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        return self.fig
    
    def plot_distribution(
        self,
        data: np.ndarray,
        title: str = "Data Distribution",
        bins: int = 30,
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        **kwargs
    ):
        """Plot histogram of data distribution.
        
        Args:
            data: Data array to plot
            title: Plot title
            bins: Number of histogram bins
            xlabel: X-axis label
            ylabel: Y-axis label
            **kwargs: Additional arguments passed to hist
        """
        self.close()
        
        self.fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.hist(data, bins=bins, alpha=0.7, edgecolor='black', **kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add mean and std lines
        mean_val = np.mean(data)
        std_val = np.std(data)
        ax.axvline(mean_val, color='r', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(mean_val - std_val, color='orange', linestyle='--', linewidth=1, 
                  label=f'Std: {std_val:.2f}')
        ax.axvline(mean_val + std_val, color='orange', linestyle='--', linewidth=1)
        
        ax.legend()
        plt.tight_layout()
        return self.fig
