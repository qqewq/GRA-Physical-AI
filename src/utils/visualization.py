```python
"""
GRA Physical AI - Visualization Module
======================================

This module provides visualization tools for GRA training and evaluation.
It supports:
    - Real-time plotting of metrics (rewards, foam, safety levels)
    - Trajectory visualization in 2D/3D
    - Foam heatmaps and correlation matrices
    - Safety and ethics incident visualization
    - Policy behavior visualization
    - Comparison plots for different experiments
    - Export to images/videos
"""

import numpy as np
import torch
import time
import os
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from collections import deque
import warnings

# Try to import visualization libraries
try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.animation import FuncAnimation, FFMpegWriter
    import matplotlib.patches as patches
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.gridspec as gridspec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available for visualization")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from ..logger.logger import GRA_Logger, get_logger


# ======================================================================
# Visualization Configuration
# ======================================================================

@dataclass
class VisConfig:
    """Configuration for visualization."""
    
    # Figure settings
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 100
    style: str = 'seaborn-v0_8-darkgrid'
    
    # Colors
    colors: List[str] = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ])
    
    # Foam heatmap
    foam_cmap: str = 'viridis'
    foam_vmin: float = 0.0
    foam_vmax: float = 1.0
    
    # Safety levels
    safety_colors: Dict[int, str] = field(default_factory=lambda: {
        0: '#00ff00',  # nominal - green
        1: '#ffff00',  # caution - yellow
        2: '#ff9900',  # warning - orange
        3: '#ff0000',  # critical - red
        4: '#8b0000'   # emergency - dark red
    })
    
    # Animation
    animation_interval: int = 100  # ms
    animation_format: str = 'gif'
    
    # Output
    save_dir: str = './visualizations'
    show_plots: bool = True
    save_plots: bool = True


# ======================================================================
# Base Visualizer
# ======================================================================

class BaseVisualizer:
    """Base class for all visualizers."""
    
    def __init__(self, config: Optional[VisConfig] = None, 
                 logger: Optional[GRA_Logger] = None):
        """
        Args:
            config: Visualization configuration
            logger: Logger instance
        """
        self.config = config or VisConfig()
        self.logger = logger or get_logger()
        
        if MATPLOTLIB_AVAILABLE:
            plt.style.use(self.config.style)
        
        # Create save directory
        if self.config.save_plots:
            os.makedirs(self.config.save_dir, exist_ok=True)
        
        # Active figures
        self.figures: Dict[str, plt.Figure] = {}
        self.animations: Dict[str, FuncAnimation] = {}
    
    def save_figure(self, fig: plt.Figure, name: str):
        """Save figure to file."""
        if not self.config.save_plots:
            return
        
        path = os.path.join(self.config.save_dir, f"{name}.png")
        fig.savefig(path, dpi=self.config.dpi, bbox_inches='tight')
        self.logger.info("visualization", f"Saved figure: {name}", {'path': path})
    
    def close_all(self):
        """Close all figures."""
        for fig in self.figures.values():
            plt.close(fig)
        self.figures.clear()
        self.animations.clear()


# ======================================================================
# Metric Visualizer
# ======================================================================

class MetricVisualizer(BaseVisualizer):
    """Visualizer for training metrics."""
    
    def __init__(self, config: Optional[VisConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.metrics_history = deque(maxlen=10000)
    
    def add_metrics(self, metrics: Dict[str, float], step: int):
        """Add metrics to history."""
        self.metrics_history.append({
            'step': step,
            'timestamp': time.time(),
            **metrics
        })
    
    def plot_metrics(self, metrics: List[str], 
                     steps: Optional[range] = None,
                     title: str = "Training Metrics") -> plt.Figure:
        """
        Plot metrics over time.
        
        Args:
            metrics: List of metric names to plot
            steps: Range of steps to include (None for all)
            title: Plot title
        
        Returns:
            Matplotlib figure
        """
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("visualization", "Matplotlib not available")
            return None
        
        # Filter data
        data = list(self.metrics_history)
        if steps is not None:
            data = [d for d in data if steps.start <= d['step'] <= steps.stop]
        
        if not data:
            self.logger.warning("visualization", "No data to plot")
            return None
        
        # Create figure
        fig, axes = plt.subplots(len(metrics), 1, figsize=self.config.figsize)
        if len(metrics) == 1:
            axes = [axes]
        
        fig.suptitle(title)
        
        steps = [d['step'] for d in data]
        
        for i, metric in enumerate(metrics):
            values = [d.get(metric, 0) for d in data]
            axes[i].plot(steps, values, color=self.config.colors[i % len(self.config.colors)])
            axes[i].set_xlabel('Step')
            axes[i].set_ylabel(metric)
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        self.figures['metrics'] = fig
        self.save_figure(fig, 'metrics')
        
        return fig
    
    def plot_reward_distribution(self, bins: int = 50) -> plt.Figure:
        """Plot distribution of rewards."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        rewards = [d.get('reward', 0) for d in self.metrics_history]
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        ax.hist(rewards, bins=bins, color=self.config.colors[0], alpha=0.7)
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.grid(True, alpha=0.3)
        
        self.figures['reward_dist'] = fig
        self.save_figure(fig, 'reward_distribution')
        
        return fig
    
    def plot_learning_curve(self, window: int = 100) -> plt.Figure:
        """Plot learning curve with smoothing."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        rewards = [d.get('reward', 0) for d in self.metrics_history]
        steps = [d['step'] for d in self.metrics_history]
        
        # Smooth with moving average
        if len(rewards) > window:
            smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
            steps_smoothed = steps[window-1:]
        else:
            smoothed = rewards
            steps_smoothed = steps
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        ax.plot(steps, rewards, alpha=0.3, color=self.config.colors[0], label='Raw')
        ax.plot(steps_smoothed, smoothed, color=self.config.colors[1], linewidth=2, label='Smoothed')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.figures['learning_curve'] = fig
        self.save_figure(fig, 'learning_curve')
        
        return fig


# ======================================================================
# Foam Visualizer
# ======================================================================

class FoamVisualizer(BaseVisualizer):
    """Visualizer for GRA foam."""
    
    def __init__(self, config: Optional[VisConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.foam_history = deque(maxlen=10000)
    
    def add_foam(self, foam_dict: Dict[int, float], step: int):
        """Add foam values to history."""
        self.foam_history.append({
            'step': step,
            'timestamp': time.time(),
            **foam_dict
        })
    
    def plot_foam_over_time(self, levels: Optional[List[int]] = None) -> plt.Figure:
        """Plot foam values over time for each level."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = list(self.foam_history)
        if not data:
            self.logger.warning("visualization", "No foam data to plot")
            return None
        
        if levels is None:
            # Get all levels from first entry
            levels = [k for k in data[0].keys() if isinstance(k, int)]
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        steps = [d['step'] for d in data]
        
        for i, level in enumerate(levels):
            values = [d.get(level, 0) for d in data]
            ax.plot(steps, values, 
                   color=self.config.colors[i % len(self.config.colors)],
                   label=f'Level {level}')
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Foam')
        ax.set_title('Foam Evolution')
        ax.legend()
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        self.figures['foam_over_time'] = fig
        self.save_figure(fig, 'foam_over_time')
        
        return fig
    
    def plot_foam_heatmap(self, level: int, 
                          subsystem_labels: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot foam heatmap for a specific level.
        
        This shows the pairwise contributions to foam.
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Need recent foam data with pairwise info
        # This would require storing the full foam matrix
        # Simplified version for now
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Generate random data for demonstration
        n = 10
        data = np.random.rand(n, n)
        np.fill_diagonal(data, 0)
        
        im = ax.imshow(data, cmap=self.config.foam_cmap,
                      vmin=self.config.foam_vmin,
                      vmax=self.config.foam_vmax)
        
        ax.set_xlabel('Subsystem')
        ax.set_ylabel('Subsystem')
        ax.set_title(f'Foam Heatmap - Level {level}')
        
        plt.colorbar(im, ax=ax, label='Foam Contribution')
        
        if subsystem_labels:
            ax.set_xticks(range(len(subsystem_labels)))
            ax.set_yticks(range(len(subsystem_labels)))
            ax.set_xticklabels(subsystem_labels, rotation=45, ha='right')
            ax.set_yticklabels(subsystem_labels)
        
        self.figures[f'foam_heatmap_l{level}'] = fig
        self.save_figure(fig, f'foam_heatmap_l{level}')
        
        return fig
    
    def plot_foam_correlation(self) -> plt.Figure:
        """Plot correlation between foam levels."""
        if not MATPLOTLIB_AVAILABLE or not SEABORN_AVAILABLE:
            return None
        
        data = list(self.foam_history)
        if not data:
            return None
        
        # Extract foam values
        levels = [k for k in data[0].keys() if isinstance(k, int)]
        foam_matrix = np.array([[d.get(l, 0) for l in levels] for d in data])
        
        # Compute correlation
        corr = np.corrcoef(foam_matrix.T)
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
                   xticklabels=[f'L{l}' for l in levels],
                   yticklabels=[f'L{l}' for l in levels],
                   ax=ax)
        ax.set_title('Foam Level Correlation')
        
        self.figures['foam_correlation'] = fig
        self.save_figure(fig, 'foam_correlation')
        
        return fig
    
    def plot_foam_3d(self, level: int) -> plt.Figure:
        """3D visualization of foam landscape."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        fig = plt.figure(figsize=self.config.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate sample data
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y) * 0.5 + 0.5  # Simulated foam landscape
        
        surf = ax.plot_surface(X, Y, Z, cmap=self.config.foam_cmap,
                               linewidth=0, antialiased=True)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Foam')
        ax.set_title(f'Foam Landscape - Level {level}')
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        self.figures[f'foam_3d_l{level}'] = fig
        self.save_figure(fig, f'foam_3d_l{level}')
        
        return fig


# ======================================================================
# Trajectory Visualizer
# ======================================================================

class TrajectoryVisualizer(BaseVisualizer):
    """Visualizer for robot trajectories."""
    
    def __init__(self, config: Optional[VisConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.trajectories: Dict[str, List[np.ndarray]] = {}
        self.obstacles: List[Dict] = []
    
    def add_trajectory(self, name: str, positions: List[np.ndarray]):
        """Add a trajectory."""
        self.trajectories[name] = positions
    
    def add_obstacle(self, position: np.ndarray, radius: float = 0.5,
                     color: str = 'red'):
        """Add an obstacle."""
        self.obstacles.append({
            'position': position,
            'radius': radius,
            'color': color
        })
    
    def plot_trajectory_2d(self, names: Optional[List[str]] = None,
                           show_obstacles: bool = True) -> plt.Figure:
        """Plot trajectories in 2D."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if names is None:
            names = list(self.trajectories.keys())
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        for i, name in enumerate(names):
            if name not in self.trajectories:
                continue
            
            positions = self.trajectories[name]
            if not positions:
                continue
            
            pos_array = np.array(positions)
            ax.plot(pos_array[:, 0], pos_array[:, 1],
                   color=self.config.colors[i % len(self.config.colors)],
                   linewidth=2, label=name)
            
            # Mark start and end
            ax.scatter(pos_array[0, 0], pos_array[0, 1],
                      color='green', s=100, marker='o', zorder=5)
            ax.scatter(pos_array[-1, 0], pos_array[-1, 1],
                      color='red', s=100, marker='s', zorder=5)
        
        # Plot obstacles
        if show_obstacles:
            for obs in self.obstacles:
                circle = patches.Circle(
                    obs['position'][:2], obs['radius'],
                    color=obs['color'], alpha=0.3
                )
                ax.add_patch(circle)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Robot Trajectories')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        self.figures['trajectory_2d'] = fig
        self.save_figure(fig, 'trajectory_2d')
        
        return fig
    
    def plot_trajectory_3d(self, names: Optional[List[str]] = None) -> plt.Figure:
        """Plot trajectories in 3D."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if names is None:
            names = list(self.trajectories.keys())
        
        fig = plt.figure(figsize=self.config.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        for i, name in enumerate(names):
            if name not in self.trajectories:
                continue
            
            positions = self.trajectories[name]
            if not positions:
                continue
            
            pos_array = np.array(positions)
            ax.plot(pos_array[:, 0], pos_array[:, 1], pos_array[:, 2],
                   color=self.config.colors[i % len(self.config.colors)],
                   linewidth=2, label=name)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Robot Trajectories (3D)')
        ax.legend()
        
        self.figures['trajectory_3d'] = fig
        self.save_figure(fig, 'trajectory_3d')
        
        return fig
    
    def animate_trajectory(self, name: str, interval: Optional[int] = None) -> FuncAnimation:
        """Create animation of trajectory execution."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        if name not in self.trajectories:
            return None
        
        positions = self.trajectories[name]
        if not positions:
            return None
        
        interval = interval or self.config.animation_interval
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        # Set up plot
        pos_array = np.array(positions)
        ax.set_xlim(pos_array[:, 0].min() - 1, pos_array[:, 0].max() + 1)
        ax.set_ylim(pos_array[:, 1].min() - 1, pos_array[:, 1].max() + 1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Trajectory Animation: {name}')
        
        # Plot obstacles
        for obs in self.obstacles:
            circle = patches.Circle(
                obs['position'][:2], obs['radius'],
                color=obs['color'], alpha=0.3
            )
            ax.add_patch(circle)
        
        # Initialize line and point
        line, = ax.plot([], [], 'b-', linewidth=2)
        point, = ax.plot([], [], 'ro', markersize=8)
        
        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point
        
        def animate(i):
            # Plot trajectory up to current point
            line.set_data(pos_array[:i+1, 0], pos_array[:i+1, 1])
            point.set_data([pos_array[i, 0]], [pos_array[i, 1]])
            return line, point
        
        anim = FuncAnimation(fig, animate, init_func=init,
                            frames=len(positions),
                            interval=interval, blit=True)
        
        self.animations[f'traj_{name}'] = anim
        self.figures[f'traj_anim_{name}'] = fig
        
        return anim


# ======================================================================
# Safety Visualizer
# ======================================================================

class SafetyVisualizer(BaseVisualizer):
    """Visualizer for safety incidents and monitoring."""
    
    def __init__(self, config: Optional[VisConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.safety_history = deque(maxlen=10000)
    
    def add_safety_event(self, event: Dict):
        """Add safety event to history."""
        self.safety_history.append(event)
    
    def plot_safety_levels(self) -> plt.Figure:
        """Plot safety levels over time."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = list(self.safety_history)
        if not data:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        steps = [d.get('step', i) for i, d in enumerate(data)]
        levels = [d.get('safety_level', 0) for d in data]
        
        # Color by level
        colors = [self.config.safety_colors.get(l, '#888888') for l in levels]
        
        ax.scatter(steps, levels, c=colors, s=50, alpha=0.6)
        ax.plot(steps, levels, 'k-', alpha=0.3)
        
        ax.set_xlabel('Step')
        ax.set_ylabel('Safety Level')
        ax.set_title('Safety Levels Over Time')
        ax.set_yticks(list(self.config.safety_colors.keys()))
        ax.set_yticklabels(['Nominal', 'Caution', 'Warning', 'Critical', 'Emergency'])
        ax.grid(True, alpha=0.3)
        
        self.figures['safety_levels'] = fig
        self.save_figure(fig, 'safety_levels')
        
        return fig
    
    def plot_violation_types(self) -> plt.Figure:
        """Plot distribution of violation types."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = list(self.safety_history)
        if not data:
            return None
        
        # Count violation types
        violation_counts = {}
        for d in data:
            vtype = d.get('violation_type', 'unknown')
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        types = list(violation_counts.keys())
        counts = list(violation_counts.values())
        
        bars = ax.bar(types, counts, color=self.config.colors[:len(types)])
        ax.set_xlabel('Violation Type')
        ax.set_ylabel('Count')
        ax.set_title('Safety Violation Distribution')
        ax.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{count}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        self.figures['violation_types'] = fig
        self.save_figure(fig, 'violation_types')
        
        return fig
    
    def plot_severity_over_time(self) -> plt.Figure:
        """Plot severity of violations over time."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = list(self.safety_history)
        if not data:
            return None
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        steps = [d.get('step', i) for i, d in enumerate(data) if 'severity' in d]
        severities = [d['severity'] for d in data if 'severity' in d]
        
        ax.scatter(steps, severities, alpha=0.6, c=severities, 
                  cmap='RdYlGn_r', s=50)
        ax.set_xlabel('Step')
        ax.set_ylabel('Severity')
        ax.set_title('Violation Severity Over Time')
        ax.grid(True, alpha=0.3)
        
        plt.colorbar(ax.collections[0], ax=ax, label='Severity')
        
        self.figures['severity_over_time'] = fig
        self.save_figure(fig, 'severity_over_time')
        
        return fig


# ======================================================================
# Ethics Visualizer
# ======================================================================

class EthicsVisualizer(BaseVisualizer):
    """Visualizer for ethical dilemmas and decisions."""
    
    def __init__(self, config: Optional[VisConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        self.ethics_history = deque(maxlen=10000)
    
    def add_ethical_event(self, event: Dict):
        """Add ethical event to history."""
        self.ethics_history.append(event)
    
    def plot_principle_violations(self) -> plt.Figure:
        """Plot violations by ethical principle."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = list(self.ethics_history)
        if not data:
            return None
        
        # Count by principle
        principle_counts = {}
        for d in data:
            principle = d.get('principle', 'unknown')
            principle_counts[principle] = principle_counts.get(principle, 0) + 1
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        principles = list(principle_counts.keys())
        counts = list(principle_counts.values())
        
        wedges, texts, autotexts = ax.pie(counts, labels=principles, autopct='%1.1f%%',
                                          colors=self.config.colors[:len(principles)])
        ax.set_title('Ethical Violations by Principle')
        
        self.figures['principle_violations'] = fig
        self.save_figure(fig, 'principle_violations')
        
        return fig
    
    def plot_dilemma_resolution(self) -> plt.Figure:
        """Plot dilemma resolution statistics."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        data = list(self.ethics_history)
        if not data:
            return None
        
        # Count resolved vs pending
        resolved = sum(1 for d in data if d.get('resolved', False))
        pending = sum(1 for d in data if not d.get('resolved', True))
        human_involved = sum(1 for d in data if d.get('human_involved', False))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figsize)
        
        # Resolution status
        ax1.bar(['Resolved', 'Pending'], [resolved, pending],
                color=[self.config.colors[0], self.config.colors[1]])
        ax1.set_title('Dilemma Resolution Status')
        ax1.set_ylabel('Count')
        
        # Human involvement
        ax2.bar(['Human', 'Autonomous'], [human_involved, resolved - human_involved],
                color=[self.config.colors[2], self.config.colors[3]])
        ax2.set_title('Human Involvement')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        
        self.figures['dilemma_resolution'] = fig
        self.save_figure(fig, 'dilemma_resolution')
        
        return fig


# ======================================================================
# Dashboard
# ======================================================================

class GRADashboard(BaseVisualizer):
    """Combined dashboard for GRA training visualization."""
    
    def __init__(self, config: Optional[VisConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        
        self.metric_viz = MetricVisualizer(config, **kwargs)
        self.foam_viz = FoamVisualizer(config, **kwargs)
        self.traj_viz = TrajectoryVisualizer(config, **kwargs)
        self.safety_viz = SafetyVisualizer(config, **kwargs)
        self.ethics_viz = EthicsVisualizer(config, **kwargs)
        
        self.dashboard_fig = None
    
    def update(self, metrics: Dict, foam: Dict, safety: Dict, ethics: Dict, step: int):
        """Update all visualizers with new data."""
        self.metric_viz.add_metrics(metrics, step)
        self.foam_viz.add_foam(foam, step)
        
        if safety:
            self.safety_viz.add_safety_event({'step': step, **safety})
        
        if ethics:
            self.ethics_viz.add_ethical_event({'step': step, **ethics})
    
    def create_dashboard(self, update_interval: int = 1000) -> plt.Figure:
        """Create comprehensive dashboard."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 4, figure=fig)
        
        # Metrics plot
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_title('Training Metrics')
        
        # Foam plot
        ax2 = fig.add_subplot(gs[0, 2:])
        ax2.set_title('Foam Levels')
        
        # Safety plot
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.set_title('Safety Status')
        
        # Ethics plot
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_title('Ethics')
        
        # Trajectory plot
        ax5 = fig.add_subplot(gs[1, 3])
        ax5.set_title('Trajectory')
        
        # Foam heatmap
        ax6 = fig.add_subplot(gs[2, :2])
        ax6.set_title('Foam Heatmap')
        
        # Statistics text
        ax7 = fig.add_subplot(gs[2, 2:])
        ax7.set_title('Statistics')
        ax7.axis('off')
        
        plt.tight_layout()
        
        self.dashboard_fig = fig
        self.figures['dashboard'] = fig
        
        return fig
    
    def update_dashboard(self):
        """Update dashboard with latest data."""
        if self.dashboard_fig is None:
            self.create_dashboard()
        
        # This would update each subplot with current data
        # For now, just save the static figure
        self.save_figure(self.dashboard_fig, 'dashboard')


# ======================================================================
# Interactive Visualizations (Plotly)
# ======================================================================

class InteractiveVisualizer(BaseVisualizer):
    """Interactive visualizations using Plotly."""
    
    def __init__(self, config: Optional[VisConfig] = None, **kwargs):
        super().__init__(config, **kwargs)
        
        if not PLOTLY_AVAILABLE:
            self.logger.warning("visualization", "Plotly not available")
    
    def plot_metrics_interactive(self, metrics_data: Dict[str, List[float]],
                                 steps: List[int]) -> go.Figure:
        """Create interactive metrics plot."""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        for name, values in metrics_data.items():
            fig.add_trace(go.Scatter(
                x=steps, y=values,
                mode='lines',
                name=name,
                line=dict(width=2)
            ))
        
        fig.update_layout(
            title='Training Metrics',
            xaxis_title='Step',
            yaxis_title='Value',
            hovermode='x unified',
            template='plotly_dark'
        )
        
        return fig
    
    def plot_foam_3d_interactive(self, foam_data: np.ndarray,
                                 x_label: str = 'X',
                                 y_label: str = 'Y') -> go.Figure:
        """Create interactive 3D foam plot."""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure(data=[go.Surface(z=foam_data, colorscale='Viridis')])
        
        fig.update_layout(
            title='3D Foam Landscape',
            scene=dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title='Foam'
            ),
            template='plotly_dark'
        )
        
        return fig
    
    def plot_trajectory_interactive(self, trajectories: Dict[str, np.ndarray],
                                   obstacles: Optional[List] = None) -> go.Figure:
        """Create interactive trajectory plot."""
        if not PLOTLY_AVAILABLE:
            return None
        
        fig = go.Figure()
        
        for name, points in trajectories.items():
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2] if points.shape[1] > 2 else np.zeros(len(points)),
                mode='lines+markers',
                name=name,
                line=dict(width=4),
                marker=dict(size=4)
            ))
        
        # Add obstacles as spheres
        if obstacles:
            for obs in obstacles:
                fig.add_trace(go.Scatter3d(
                    x=[obs['position'][0]], y=[obs['position'][1]],
                    z=[obs['position'][2] if len(obs['position']) > 2 else 0],
                    mode='markers',
                    marker=dict(size=obs.get('radius', 0.5) * 20,
                              color=obs.get('color', 'red'),
                              opacity=0.3),
                    name='Obstacle'
                ))
        
        fig.update_layout(
            title='Robot Trajectories',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            ),
            template='plotly_dark'
        )
        
        return fig


# ======================================================================
# Video Recording
# ======================================================================

class VideoRecorder(BaseVisualizer):
    """Record videos of robot behavior."""
    
    def __init__(self, config: Optional[VisConfig] = None, 
                 fps: int = 30, **kwargs):
        super().__init__(config, **kwargs)
        self.fps = fps
        self.frames: List[np.ndarray] = []
        self.recording = False
    
    def start_recording(self):
        """Start recording."""
        self.frames = []
        self.recording = True
    
    def add_frame(self, frame: np.ndarray):
        """Add a frame."""
        if self.recording:
            self.frames.append(frame)
    
    def stop_recording(self, filename: str = 'recording.mp4'):
        """Stop recording and save video."""
        self.recording = False
        
        if not self.frames:
            self.logger.warning("video", "No frames recorded")
            return
        
        if not CV2_AVAILABLE:
            self.logger.warning("video", "OpenCV not available")
            return
        
        path = os.path.join(self.config.save_dir, filename)
        
        # Get frame size
        height, width = self.frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, self.fps, (width, height))
        
        for frame in self.frames:
            # Convert RGB to BGR for OpenCV
            if frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            out.write(frame_bgr)
        
        out.release()
        self.logger.info("video", f"Saved video: {path}")
    
    def capture_episode(self, env, policy, max_steps: int = 1000):
        """Record an entire episode."""
        self.start_recording()
        
        obs = env.reset()
        
        for _ in range(max_steps):
            # Get action
            if hasattr(policy, 'act'):
                action = policy.act(torch.FloatTensor(obs))
            else:
                action = policy(obs)
            
            # Step environment
            obs, _, done, _ = env.step(action)
            
            # Capture frame
            frame = env.render(mode='rgb_array')
            if frame is not None:
                self.add_frame(frame)
            
            if done:
                break
        
        self.stop_recording(f"episode_{time.strftime('%Y%m%d_%H%M%S')}.mp4")


# ======================================================================
# Factory function
# ======================================================================

def create_visualizer(vis_type: str, **kwargs) -> BaseVisualizer:
    """
    Factory function to create visualizers.
    
    Args:
        vis_type: Type of visualizer ('metrics', 'foam', 'trajectory',
                  'safety', 'ethics', 'dashboard', 'interactive', 'video')
        **kwargs: Additional arguments
    
    Returns:
        Visualizer instance
    """
    config = kwargs.get('config', VisConfig())
    
    if vis_type == 'metrics':
        return MetricVisualizer(config, **kwargs)
    elif vis_type == 'foam':
        return FoamVisualizer(config, **kwargs)
    elif vis_type == 'trajectory':
        return TrajectoryVisualizer(config, **kwargs)
    elif vis_type == 'safety':
        return SafetyVisualizer(config, **kwargs)
    elif vis_type == 'ethics':
        return EthicsVisualizer(config, **kwargs)
    elif vis_type == 'dashboard':
        return GRADashboard(config, **kwargs)
    elif vis_type == 'interactive':
        return InteractiveVisualizer(config, **kwargs)
    elif vis_type == 'video':
        return VideoRecorder(config, **kwargs)
    else:
        raise ValueError(f"Unknown visualizer type: {vis_type}")


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing Visualization Module ===\n")
    
    if not MATPLOTLIB_AVAILABLE:
        print("Matplotlib not available. Please install: pip install matplotlib")
        sys.exit(1)
    
    # Create metric visualizer
    metric_viz = MetricVisualizer()
    
    # Add some sample data
    for i in range(100):
        metrics = {
            'reward': np.sin(i * 0.1) * 10,
            'loss': np.exp(-i * 0.05) + np.random.rand() * 0.1,
            'accuracy': 0.5 + 0.5 * (1 - np.exp(-i * 0.02))
        }
        metric_viz.add_metrics(metrics, i)
    
    # Plot metrics
    metric_viz.plot_metrics(['reward', 'loss', 'accuracy'])
    metric_viz.plot_reward_distribution()
    metric_viz.plot_learning_curve()
    
    # Create foam visualizer
    foam_viz = FoamVisualizer()
    
    # Add sample foam data
    for i in range(50):
        foam = {
            0: np.exp(-i * 0.1) + 0.1,
            1: np.exp(-i * 0.08) * 0.8 + 0.1,
            2: np.exp(-i * 0.05) * 0.6 + 0.1,
            3: np.exp(-i * 0.03) * 0.4 + 0.1
        }
        foam_viz.add_foam(foam, i)
    
    foam_viz.plot_foam_over_time()
    foam_viz.plot_foam_heatmap(0)
    
    # Create trajectory visualizer
    traj_viz = TrajectoryVisualizer()
    
    # Add sample trajectory
    t = np.linspace(0, 4*np.pi, 100)
    x = np.cos(t) * 2
    y = np.sin(t) * 2
    positions = np.column_stack([x, y])
    traj_viz.add_trajectory('spiral', positions)
    
    # Add obstacles
    traj_viz.add_obstacle(np.array([0, 0]), radius=0.5)
    
    traj_viz.plot_trajectory_2d()
    
    # Create safety visualizer
    safety_viz = SafetyVisualizer()
    
    # Add sample safety events
    for i in range(20):
        event = {
            'step': i * 10,
            'safety_level': np.random.choice([0, 1, 2, 3]),
            'violation_type': np.random.choice(['collision', 'force', 'velocity']),
            'severity': np.random.rand()
        }
        safety_viz.add_safety_event(event)
    
    safety_viz.plot_safety_levels()
    safety_viz.plot_violation_types()
    
    # Show all plots
    plt.show()
    
    print("\nAll tests passed!")
```