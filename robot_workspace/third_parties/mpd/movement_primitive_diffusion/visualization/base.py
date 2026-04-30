from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class BaseVisualizer(ABC):
    """Abstract base class for visualization components."""
    
    def __init__(self, dpi: int = 150):
        """Initialize visualizer.
        
        Args:
            dpi: Dots per inch for saved figures
        """
        self.dpi = dpi
        self.fig = None
    
    @abstractmethod
    def plot(self, *args, **kwargs):
        """Create the plot/visualization."""
        raise NotImplementedError
    
    def save(self, path: Path, **kwargs):
        """Save the current figure to file.
        
        Args:
            path: Output file path
            **kwargs: Additional arguments passed to savefig
        """
        if self.fig is None:
            raise ValueError("No figure to save. Call plot() first.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_kwargs = {'dpi': self.dpi, 'bbox_inches': 'tight'}
        save_kwargs.update(kwargs)
        
        self.fig.savefig(path, **save_kwargs)
    
    def close(self):
        """Close the current figure."""
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
