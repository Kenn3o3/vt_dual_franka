import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Union, Tuple
from pathlib import Path

from .base import BaseVisualizer


class TrajectoryVisualizer(BaseVisualizer):
    """Visualizer for trajectory data using matplotlib."""
    
    def __init__(self, dpi: int = 150, figsize: Tuple[float, float] = (10, 8)):
        """Initialize trajectory visualizer.
        
        Args:
            dpi: Dots per inch for saved figures
            figsize: Figure size (width, height) in inches
        """
        super().__init__(dpi=dpi)
        self.figsize = figsize
    
    def plot_single_trajectory(
        self,
        positions: np.ndarray,
        title: str = "Trajectory",
        xlabel: str = "X",
        ylabel: str = "Y",
        color: str = 'blue',
        marker: str = 'o',
        **kwargs
    ):
        """Plot a single trajectory.
        
        Args:
            positions: Trajectory positions array of shape (T, 2) or (T, 3)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            color: Line/marker color
            marker: Marker style
            **kwargs: Additional arguments passed to plot
        """
        self.close()
        
        if positions.shape[1] == 2:
            self.fig, ax = plt.subplots(figsize=self.figsize)
            ax.plot(positions[:, 0], positions[:, 1], color=color, marker=marker, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
        elif positions.shape[1] == 3:
            from mpl_toolkits.mplot3d import Axes3D
            self.fig = plt.figure(figsize=self.figsize)
            ax = self.fig.add_subplot(111, projection='3d')
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                   color=color, marker=marker, **kwargs)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel("Z")
            ax.set_title(title)
        else:
            raise ValueError(f"Positions must have 2 or 3 dimensions, got {positions.shape[1]}")
        
        plt.tight_layout()
        return self.fig
    
    def plot_multiple_trajectories(
        self,
        trajectories: List[np.ndarray],
        title: str = "Trajectories",
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        **kwargs
    ):
        """Plot multiple trajectories on the same axes.
        
        Args:
            trajectories: List of trajectory arrays, each of shape (T, 2) or (T, 3)
            title: Plot title
            labels: Optional labels for each trajectory
            colors: Optional colors for each trajectory
            **kwargs: Additional arguments passed to plot
        """
        self.close()
        
        if not trajectories:
            raise ValueError("trajectories list is empty")
        
        # Check dimensionality
        ndim = trajectories[0].shape[1]
        
        if ndim == 2:
            self.fig, ax = plt.subplots(figsize=self.figsize)
            for i, traj in enumerate(trajectories):
                label = labels[i] if labels else f"Trajectory {i+1}"
                color = colors[i] if colors else None
                ax.plot(traj[:, 0], traj[:, 1], label=label, color=color, alpha=0.7, **kwargs)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        elif ndim == 3:
            from mpl_toolkits.mplot3d import Axes3D
            self.fig = plt.figure(figsize=self.figsize)
            ax = self.fig.add_subplot(111, projection='3d')
            for i, traj in enumerate(trajectories):
                label = labels[i] if labels else f"Trajectory {i+1}"
                color = colors[i] if colors else None
                ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], 
                       label=label, color=color, alpha=0.7, **kwargs)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(title)
            ax.legend()
        else:
            raise ValueError(f"Trajectories must have 2 or 3 dimensions, got {ndim}")
        
        plt.tight_layout()
        return self.fig
    
    def plot_with_start_end_markers(
        self,
        positions: np.ndarray,
        title: str = "Trajectory",
        start_marker: str = 'o',
        end_marker: str = 's',
        start_color: str = 'green',
        end_color: str = 'red',
        **kwargs
    ):
        """Plot trajectory with distinct start and end markers.
        
        Args:
            positions: Trajectory positions array of shape (T, 2)
            title: Plot title
            start_marker: Marker style for start point
            end_marker: Marker style for end point
            start_color: Color for start point
            end_color: Color for end point
            **kwargs: Additional arguments passed to plot
        """
        self.close()
        
        self.fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], color='blue', alpha=0.7, **kwargs)
        
        # Mark start and end
        ax.scatter(positions[0, 0], positions[0, 1], 
                  marker=start_marker, s=100, color=start_color, 
                  label='Start', zorder=5)
        ax.scatter(positions[-1, 0], positions[-1, 1], 
                  marker=end_marker, s=100, color=end_color, 
                  label='End', zorder=5)
        
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self.fig
