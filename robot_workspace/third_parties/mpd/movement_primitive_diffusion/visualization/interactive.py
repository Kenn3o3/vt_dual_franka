import numpy as np
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class InteractiveVisualizer:
    """Visualizer for interactive plots using plotly."""
    
    def __init__(self):
        """Initialize interactive visualizer."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("plotly is required for InteractiveVisualizer. Install with: pip install plotly")
        self.fig = None
    
    def plot_trajectories_2d(
        self,
        trajectories: List[np.ndarray],
        title: str = "Trajectories",
        labels: Optional[List[str]] = None,
        colors: Optional[List[str]] = None,
        obstacles: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> go.Figure:
        """Plot 2D trajectories with optional obstacles.
        
        Args:
            trajectories: List of trajectory arrays, each of shape (T, 2)
            title: Plot title
            labels: Optional labels for each trajectory
            colors: Optional colors for each trajectory
            obstacles: Optional list of obstacle specifications
            **kwargs: Additional arguments
            
        Returns:
            Plotly figure object
        """
        self.fig = go.Figure()
        
        # Plot trajectories
        for i, traj in enumerate(trajectories):
            label = labels[i] if labels else f"Trajectory {i+1}"
            color = colors[i] if colors else None
            
            self.fig.add_trace(go.Scatter(
                x=traj[:, 0],
                y=traj[:, 1],
                mode='lines+markers',
                name=label,
                line=dict(color=color) if color else None,
                marker=dict(size=4)
            ))
        
        # Add obstacles if provided
        if obstacles:
            for obs in obstacles:
                if obs['type'] == 'circle':
                    # Create circle using parametric equations
                    theta = np.linspace(0, 2*np.pi, 100)
                    x = obs['center'][0] + obs['radius'] * np.cos(theta)
                    y = obs['center'][1] + obs['radius'] * np.sin(theta)
                    
                    self.fig.add_trace(go.Scatter(
                        x=x, y=y,
                        mode='lines',
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.2)',
                        line=dict(color='red', width=2),
                        name=obs.get('name', 'Obstacle'),
                        showlegend=False
                    ))
        
        self.fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            hovermode='closest',
            showlegend=True
        )
        
        return self.fig
    
    def plot_trajectory_modes(
        self,
        trajectories: List[np.ndarray],
        mode_labels: List[int],
        title: str = "Trajectory Modes",
        **kwargs
    ) -> go.Figure:
        """Plot trajectories colored by mode.
        
        Args:
            trajectories: List of trajectory arrays
            mode_labels: Mode label for each trajectory
            title: Plot title
            **kwargs: Additional arguments
            
        Returns:
            Plotly figure object
        """
        self.fig = go.Figure()
        
        # Group trajectories by mode
        unique_modes = sorted(set(mode_labels))
        color_scale = px.colors.qualitative.Plotly
        
        for mode in unique_modes:
            mode_trajs = [traj for traj, label in zip(trajectories, mode_labels) if label == mode]
            color = color_scale[mode % len(color_scale)]
            
            for traj in mode_trajs:
                self.fig.add_trace(go.Scatter(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f"Mode {mode}",
                    showlegend=True,
                    legendgroup=f"mode_{mode}",
                ))
        
        self.fig.update_layout(
            title=title,
            xaxis_title="X",
            yaxis_title="Y",
            hovermode='closest'
        )
        
        return self.fig
    
    def save(self, path: Path, **kwargs):
        """Save figure to HTML file.
        
        Args:
            path: Output file path
            **kwargs: Additional arguments passed to write_html
        """
        if self.fig is None:
            raise ValueError("No figure to save. Call a plot method first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.fig.write_html(str(path), **kwargs)
    
    def show(self):
        """Display the figure."""
        if self.fig is None:
            raise ValueError("No figure to show. Call a plot method first.")
        self.fig.show()
