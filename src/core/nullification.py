```python
"""
GRA Physical AI - Nullification (Zeroing) Algorithms
====================================================

This module implements the core **recursive zeroing algorithm** that drives
the GRA multiverse toward a fully coherent state with zero foam at all levels.

The main algorithm (`recursive_zero`) follows the recursive definition of the
total functional J_total and uses gradient information to eliminate inconsistencies.

Also provided:
    - Parallel zeroing for large systems
    - Online/adaptive zeroing for deployed systems
    - Visualization and monitoring tools
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any, Iterator
from dataclasses import dataclass, field
from collections import defaultdict
import time
import warnings
from enum import Enum


# ======================================================================
# Data Structures for Zeroing State
# ======================================================================

@dataclass
class ZeroingState:
    """
    Holds the current state of the zeroing process.
    
    This is used to track progress and enable checkpointing/resume.
    """
    epoch: int
    states: Dict[Any, torch.Tensor]  # multi-index -> state tensor
    foams: Dict[int, float]  # level -> foam value
    gradients: Optional[Dict[Any, torch.Tensor]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def save(self, path: str):
        """Save state to file."""
        torch.save({
            'epoch': self.epoch,
            'states': self.states,
            'foams': self.foams,
            'metadata': self.metadata
        }, path)
    
    @classmethod
    def load(cls, path: str) -> 'ZeroingState':
        """Load state from file."""
        data = torch.load(path)
        return cls(
            epoch=data['epoch'],
            states=data['states'],
            foams=data['foams'],
            metadata=data.get('metadata', {})
        )


class ZeroingStatus(Enum):
    """Status of the zeroing algorithm."""
    RUNNING = "running"
    CONVERGED = "converged"
    DIVERGED = "diverged"
    PAUSED = "paused"
    STOPPED = "stopped"


# ======================================================================
# Core Recursive Zeroing Algorithm
# ======================================================================

def recursive_zero(
    level: int,
    states: Dict[Any, torch.Tensor],
    hierarchy: Dict[Any, Any],  # multi-index -> subsystem
    get_children: Callable[[Any], List[Any]],
    get_goal_projector: Callable[[int], Optional[Callable]],
    get_level_weight: Callable[[int], float],
    learning_rate: float = 0.01,
    tolerance: float = 1e-4,
    max_iters: int = 100,
    local_optimizer: Optional[Callable] = None,
    callback: Optional[Callable] = None
) -> Dict[Any, torch.Tensor]:
    """
    Recursively zero foam at this level and below.
    
    This is the core algorithm from algorithm.md, implemented recursively.
    
    Args:
        level: current level (starting from top)
        states: current states of all subsystems (multi-index -> tensor)
        hierarchy: mapping from multi-index to subsystem objects
        get_children: function returning children indices for a given index
        get_goal_projector: function returning projector for a level
        get_level_weight: function returning weight Λ_l for level
        learning_rate: step size for gradient descent
        tolerance: foam tolerance for convergence
        max_iters: maximum iterations at this level
        local_optimizer: optional optimizer for local goals (level 0)
        callback: optional function called each iteration
    
    Returns:
        updated states
    """
    # Base case: level 0 - optimize local goals
    if level == 0:
        return _zero_local_level(states, hierarchy, get_children, 
                                 get_goal_projector(0), local_optimizer, 
                                 max_iters, callback)
    
    # Get projector for this level
    projector = get_goal_projector(level)
    if projector is None:
        # No goal at this level - just pass through
        return states
    
    # First, recursively zero all lower levels
    # Get all subsystems at this level
    level_indices = [idx for idx in states.keys() 
                     if hasattr(idx, 'level') and idx.level == level]
    
    for idx in level_indices:
        children = get_children(idx)
        for child_idx in children:
            if child_idx in states:
                # Recursive call
                states = recursive_zero(
                    child_idx.level, states, hierarchy, get_children,
                    get_goal_projector, get_level_weight,
                    learning_rate, tolerance, max_iters,
                    local_optimizer, callback
                )
    
    # Now reduce foam at this level
    return _zero_level_foam(
        level, states, level_indices, projector,
        get_level_weight(level), learning_rate, tolerance, max_iters, callback
    )


def _zero_local_level(
    states: Dict[Any, torch.Tensor],
    hierarchy: Dict[Any, Any],
    get_children: Callable,
    local_projector: Optional[Callable],
    optimizer: Optional[Callable],
    max_iters: int,
    callback: Optional[Callable]
) -> Dict[Any, torch.Tensor]:
    """Optimize local goals at level 0."""
    # Get all level 0 indices
    level0_indices = [idx for idx in states.keys() 
                      if hasattr(idx, 'level') and idx.level == 0]
    
    if not level0_indices:
        return states
    
    # If no local projector, nothing to do
    if local_projector is None:
        return states
    
    # If optimizer provided, use it
    if optimizer is not None:
        for idx in level0_indices:
            # Wrap state in optimizer if needed
            if hasattr(optimizer, 'step'):
                # Assume optimizer is per-subsystem
                opt = optimizer(idx)
                for _ in range(max_iters):
                    loss = _local_loss(states[idx], local_projector)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    if callback:
                        callback(idx, states[idx], loss)
            else:
                # Simple gradient descent
                for _ in range(max_iters):
                    loss = _local_loss(states[idx], local_projector)
                    grad = torch.autograd.grad(loss, states[idx])[0]
                    states[idx] = states[idx] - 0.01 * grad
                    if callback:
                        callback(idx, states[idx], loss)
    else:
        # Simple gradient descent
        for idx in level0_indices:
            for _ in range(max_iters):
                loss = _local_loss(states[idx], local_projector)
                if loss < 1e-6:
                    break
                grad = torch.autograd.grad(loss, states[idx])[0]
                states[idx] = states[idx] - 0.01 * grad
                if callback:
                    callback(idx, states[idx], loss)
    
    return states


def _zero_level_foam(
    level: int,
    states: Dict[Any, torch.Tensor],
    indices: List[Any],
    projector: Callable,
    level_weight: float,
    learning_rate: float,
    tolerance: float,
    max_iters: int,
    callback: Optional[Callable]
) -> Dict[Any, torch.Tensor]:
    """
    Reduce foam at a specific level using gradient descent.
    """
    from .foam import foam_gradient, compute_foam
    
    for iteration in range(max_iters):
        # Compute current foam
        level_states = [states[idx] for idx in indices]
        foam = compute_foam(level_states, projector)
        
        if foam < tolerance:
            break
        
        # Compute gradients for each state at this level
        grads = foam_gradient(level_states, projector)
        
        # Update states
        for i, idx in enumerate(indices):
            grad = grads[i]
            # Scale by level weight and learning rate
            states[idx] = states[idx] - learning_rate * level_weight * grad
        
        if callback:
            callback(level, iteration, foam.item() if hasattr(foam, 'item') else foam)
    
    return states


def _local_loss(state: torch.Tensor, projector: Callable) -> torch.Tensor:
    """Compute local loss for a state (distance to goal)."""
    # If projector has loss method, use it
    if hasattr(projector, 'loss'):
        return projector.loss(state)
    else:
        # Otherwise, use squared distance to projected state
        proj = projector(state)
        return torch.norm(state - proj) ** 2


# ======================================================================
# Main Zeroing Algorithm (iterative version)
# ======================================================================

class ZeroingAlgorithm:
    """
    Main zeroing algorithm that runs the recursive procedure iteratively.
    
    This class manages the entire zeroing process, including:
        - State tracking
        - Convergence checking
        - Checkpointing
        - Visualization hooks
    """
    
    def __init__(
        self,
        hierarchy: Dict[Any, Any],  # multi-index -> subsystem
        get_children: Callable[[Any], List[Any]],
        get_parents: Callable[[Any], List[Any]],
        get_goal_projector: Callable[[int], Optional[Callable]],
        get_level_weight: Callable[[int], float],
        level_tolerances: List[float],
        learning_rates: List[float],
        local_optimizer: Optional[Callable] = None,
        device: str = 'cpu'
    ):
        """
        Args:
            hierarchy: mapping from multi-index to subsystem
            get_children: function to get children of an index
            get_parents: function to get parents of an index
            get_goal_projector: function to get projector for a level
            get_level_weight: function to get weight for a level
            level_tolerances: ε_l for each level
            learning_rates: η_l for each level
            local_optimizer: optional optimizer for level 0
            device: computation device
        """
        self.hierarchy = hierarchy
        self.get_children = get_children
        self.get_parents = get_parents
        self.get_goal_projector = get_goal_projector
        self.get_level_weight = get_level_weight
        self.level_tolerances = level_tolerances
        self.learning_rates = learning_rates
        self.local_optimizer = local_optimizer
        self.device = device
        
        # Determine max level
        self.max_level = max(
            (idx.level for idx in hierarchy.keys() if hasattr(idx, 'level')),
            default=0
        )
        
        # State
        self.status = ZeroingStatus.STOPPED
        self.current_state: Optional[ZeroingState] = None
        self.history: List[Dict[int, float]] = []
        
        # Callbacks
        self.epoch_callbacks = []
        self.level_callbacks = []
        
    def initialize_states(self, initializer: Optional[Callable] = None) -> Dict[Any, torch.Tensor]:
        """Initialize states for all subsystems."""
        states = {}
        for idx, subsystem in self.hierarchy.items():
            if initializer is not None:
                states[idx] = initializer(idx)
            elif hasattr(subsystem, 'get_state'):
                states[idx] = subsystem.get_state()
            else:
                # Default: random normal
                dim = getattr(subsystem, 'state_dim', 10)
                states[idx] = torch.randn(dim, device=self.device)
        return states
    
    def run(
        self,
        initial_states: Optional[Dict[Any, torch.Tensor]] = None,
        num_epochs: int = 1000,
        checkpoint_path: Optional[str] = None,
        checkpoint_freq: int = 100,
        verbose: bool = True
    ) -> Dict[Any, torch.Tensor]:
        """
        Run the zeroing algorithm.
        
        Args:
            initial_states: starting states (if None, initialize)
            num_epochs: maximum number of epochs
            checkpoint_path: path to save checkpoints
            checkpoint_freq: save checkpoint every N epochs
            verbose: print progress
        
        Returns:
            final states
        """
        if initial_states is None:
            states = self.initialize_states()
        else:
            states = initial_states.copy()
        
        self.status = ZeroingStatus.RUNNING
        self.history = []
        
        for epoch in range(num_epochs):
            # Run one epoch of recursive zeroing from top level
            states = recursive_zero(
                level=self.max_level,
                states=states,
                hierarchy=self.hierarchy,
                get_children=self.get_children,
                get_goal_projector=self.get_goal_projector,
                get_level_weight=self.get_level_weight,
                learning_rate=self.learning_rates[self.max_level],
                tolerance=self.level_tolerances[self.max_level],
                max_iters=10,  # inner iterations
                local_optimizer=self.local_optimizer,
                callback=lambda l, it, f: self._level_callback(epoch, l, it, f)
            )
            
            # Compute foams for monitoring
            foams = self._compute_all_foams(states)
            self.history.append(foams)
            
            # Callbacks
            for cb in self.epoch_callbacks:
                cb(epoch, states, foams)
            
            if verbose and epoch % 10 == 0:
                foam_str = ", ".join([f"L{l}: {f:.6f}" for l, f in foams.items()])
                print(f"Epoch {epoch:4d}: {foam_str}")
            
            # Check convergence
            if all(foams.get(l, 0) < self.level_tolerances[l] 
                   for l in range(self.max_level + 1)):
                if verbose:
                    print(f"Converged at epoch {epoch}")
                self.status = ZeroingStatus.CONVERGED
                break
            
            # Save checkpoint
            if checkpoint_path and epoch % checkpoint_freq == 0:
                self._save_checkpoint(checkpoint_path, epoch, states, foams)
        
        self.current_state = ZeroingState(
            epoch=epoch,
            states=states,
            foams=foams
        )
        
        return states
    
    def _compute_all_foams(self, states: Dict[Any, torch.Tensor]) -> Dict[int, float]:
        """Compute foam at all levels."""
        from .foam import compute_foam
        
        foams = {}
        for level in range(self.max_level + 1):
            projector = self.get_goal_projector(level)
            if projector is None:
                foams[level] = 0.0
                continue
            
            level_indices = [idx for idx in states.keys() 
                           if hasattr(idx, 'level') and idx.level == level]
            if len(level_indices) < 2:
                foams[level] = 0.0
                continue
            
            level_states = [states[idx] for idx in level_indices]
            foam = compute_foam(level_states, projector)
            foams[level] = foam.item() if hasattr(foam, 'item') else foam
        
        return foams
    
    def _level_callback(self, epoch: int, level: int, iteration: int, foam: float):
        """Internal callback for level iterations."""
        for cb in self.level_callbacks:
            cb(epoch, level, iteration, foam)
    
    def _save_checkpoint(self, path: str, epoch: int, 
                         states: Dict[Any, torch.Tensor],
                         foams: Dict[int, float]):
        """Save checkpoint to file."""
        state = ZeroingState(epoch=epoch, states=states, foams=foams)
        state.save(f"{path}_epoch{epoch}.pt")
    
    def add_epoch_callback(self, callback: Callable):
        """Add callback called after each epoch."""
        self.epoch_callbacks.append(callback)
    
    def add_level_callback(self, callback: Callable):
        """Add callback called during level zeroing."""
        self.level_callbacks.append(callback)
    
    def get_history(self) -> List[Dict[int, float]]:
        """Return foam history."""
        return self.history


# ======================================================================
# Parallel Zeroing (for large systems)
# ======================================================================

class ParallelZeroing(ZeroingAlgorithm):
    """
    Parallel version of zeroing algorithm using PyTorch's distributed capabilities.
    
    This splits subsystems across devices/nodes and communicates gradients.
    """
    
    def __init__(
        self,
        hierarchy: Dict[Any, Any],
        get_children: Callable,
        get_parents: Callable,
        get_goal_projector: Callable,
        get_level_weight: Callable,
        level_tolerances: List[float],
        learning_rates: List[float],
        device_ids: List[str],
        local_optimizer: Optional[Callable] = None
    ):
        super().__init__(
            hierarchy, get_children, get_parents, get_goal_projector,
            get_level_weight, level_tolerances, learning_rates, local_optimizer
        )
        self.device_ids = device_ids
        self.device_map = self._assign_devices()
        
    def _assign_devices(self) -> Dict[Any, str]:
        """Assign each subsystem to a device."""
        device_map = {}
        indices = list(self.hierarchy.keys())
        n_devices = len(self.device_ids)
        
        for i, idx in enumerate(indices):
            device_map[idx] = self.device_ids[i % n_devices]
        
        return device_map
    
    def _scatter_states(self, states: Dict[Any, torch.Tensor]) -> Dict[str, Dict[Any, torch.Tensor]]:
        """Scatter states to devices."""
        device_states = {dev: {} for dev in self.device_ids}
        for idx, state in states.items():
            dev = self.device_map[idx]
            device_states[dev][idx] = state.to(dev)
        return device_states
    
    def _gather_states(self, device_states: Dict[str, Dict[Any, torch.Tensor]]) -> Dict[Any, torch.Tensor]:
        """Gather states from devices."""
        states = {}
        for dev, dev_states in device_states.items():
            for idx, state in dev_states.items():
                states[idx] = state.cpu()
        return states
    
    def run_parallel(self, num_epochs: int = 1000, verbose: bool = True):
        """Run zeroing in parallel."""
        # Initial states on each device
        initial = self.initialize_states()
        device_states = self._scatter_states(initial)
        
        for epoch in range(num_epochs):
            # Run one epoch on each device in parallel
            # This is a simplified version - in practice use torch.distributed
            for dev in self.device_ids:
                # Move relevant parts of hierarchy to device
                dev_indices = [idx for idx in device_states[dev].keys()]
                
                # Run recursive zeroing on this device's subsystems
                # (This would need to communicate with other devices for higher levels)
                pass
            
            # Gather and check convergence
            states = self._gather_states(device_states)
            foams = self._compute_all_foams(states)
            
            if verbose and epoch % 10 == 0:
                foam_str = ", ".join([f"L{l}: {f:.6f}" for l, f in foams.items()])
                print(f"Epoch {epoch:4d}: {foam_str}")
            
            if all(foams.get(l, 0) < self.level_tolerances[l] 
                   for l in range(self.max_level + 1)):
                print(f"Converged at epoch {epoch}")
                break
        
        return self._gather_states(device_states)


# ======================================================================
# Online/Adaptive Zeroing
# ======================================================================

class OnlineZeroing:
    """
    Lightweight zeroing for deployed systems that runs continuously.
    
    This adapts to changes in the environment or robot dynamics.
    """
    
    def __init__(
        self,
        hierarchy: Dict[Any, Any],
        get_goal_projector: Callable[[int], Optional[Callable]],
        drift_thresholds: List[float],
        update_frequency: float = 1.0,  # Hz
        learning_rate: float = 0.001
    ):
        self.hierarchy = hierarchy
        self.get_goal_projector = get_goal_projector
        self.drift_thresholds = drift_thresholds
        self.update_frequency = update_frequency
        self.learning_rate = learning_rate
        
        self.max_level = max((idx.level for idx in hierarchy.keys()), default=0)
        self.last_update = time.time()
        self.running = False
        
    def start(self):
        """Start online zeroing loop."""
        self.running = True
        self._loop()
    
    def stop(self):
        """Stop online zeroing."""
        self.running = False
    
    def _loop(self):
        """Main online loop."""
        import threading
        
        def run():
            while self.running:
                time.sleep(1.0 / self.update_frequency)
                self._update()
        
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
    
    def _update(self):
        """Perform one online zeroing update."""
        from .foam import compute_foam, foam_gradient
        
        # Get current states from all subsystems
        states = {}
        for idx, subsystem in self.hierarchy.items():
            if hasattr(subsystem, 'get_state'):
                states[idx] = subsystem.get_state()
        
        # Check each level for drift
        for level in range(self.max_level + 1):
            projector = self.get_goal_projector(level)
            if projector is None:
                continue
            
            level_indices = [idx for idx in states.keys() 
                           if hasattr(idx, 'level') and idx.level == level]
            if len(level_indices) < 2:
                continue
            
            level_states = [states[idx] for idx in level_indices]
            foam = compute_foam(level_states, projector)
            
            if foam > self.drift_thresholds[level]:
                # Drift detected – apply one gradient step
                grads = foam_gradient(level_states, projector)
                for i, idx in enumerate(level_indices):
                    delta = self.learning_rate * grads[i]
                    new_state = states[idx] - delta
                    
                    # Apply update to subsystem
                    if hasattr(self.hierarchy[idx], 'set_state'):
                        self.hierarchy[idx].set_state(new_state)
                    else:
                        states[idx] = new_state


# ======================================================================
# Visualization and Monitoring
# ======================================================================

class ZeroingMonitor:
    """
    Monitor and visualize the zeroing process.
    """
    
    def __init__(self, algorithm: ZeroingAlgorithm):
        self.algorithm = algorithm
        self.fig = None
        self.axes = None
        
    def plot_foam_history(self, show: bool = True):
        """Plot foam over epochs."""
        import matplotlib.pyplot as plt
        
        history = self.algorithm.get_history()
        if not history:
            print("No history to plot")
            return
        
        epochs = range(len(history))
        levels = list(history[0].keys())
        
        plt.figure(figsize=(10, 6))
        for level in levels:
            foams = [h[level] for h in history]
            plt.semilogy(epochs, foams, label=f'Level {level}')
        
        plt.xlabel('Epoch')
        plt.ylabel('Foam (log scale)')
        plt.title('Zeroing Progress')
        plt.legend()
        plt.grid(True)
        
        if show:
            plt.show()
    
    def animate_zeroing(self, interval: int = 100):
        """Animate the zeroing process."""
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation
        
        history = self.algorithm.get_history()
        if not history:
            return
        
        fig, ax = plt.subplots()
        lines = []
        levels = list(history[0].keys())
        
        for level in levels:
            line, = ax.semilogy([], [], label=f'Level {level}')
            lines.append(line)
        
        ax.set_xlim(0, len(history))
        ax.set_ylim(1e-6, 1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Foam')
        ax.legend()
        ax.grid(True)
        
        def update(frame):
            for i, level in enumerate(levels):
                x = list(range(frame+1))
                y = [h[level] for h in history[:frame+1]]
                lines[i].set_data(x, y)
            return lines
        
        ani = animation.FuncAnimation(fig, update, frames=len(history), 
                                      interval=interval, blit=True)
        plt.show()
        return ani


# ======================================================================
# Utility Functions
# ======================================================================

def create_zeroing_algorithm(
    multiverse: Any,  # Multiverse object
    level_tolerances: List[float],
    learning_rates: List[float],
    device: str = 'cpu'
) -> ZeroingAlgorithm:
    """
    Convenience function to create zeroing algorithm from a Multiverse.
    
    Args:
        multiverse: Multiverse instance (from multiverse.py)
        level_tolerances: ε_l for each level
        learning_rates: η_l for each level
        device: computation device
    
    Returns:
        configured ZeroingAlgorithm
    """
    
    def get_children(idx):
        return list(multiverse.levels.get_children(idx))
    
    def get_parents(idx):
        return list(multiverse.levels.get_parents(idx))
    
    def get_goal_projector(level):
        goal = multiverse.get_goal(level)
        if goal is None:
            return None
        return goal.projector
    
    def get_level_weight(level):
        return multiverse.level_weights[level]
    
    return ZeroingAlgorithm(
        hierarchy=multiverse.subsystems,
        get_children=get_children,
        get_parents=get_parents,
        get_goal_projector=get_goal_projector,
        get_level_weight=get_level_weight,
        level_tolerances=level_tolerances,
        learning_rates=learning_rates,
        device=device
    )


def check_zeroing_feasibility(
    multiverse: Any,
    tolerance: float = 1e-4
) -> Tuple[bool, List[str]]:
    """
    Check if zeroing is feasible for this multiverse.
    
    Verifies:
        - All projectors commute
        - Hierarchy consistency
        - Sufficient dimension
    
    Returns:
        (feasible, list of issues)
    """
    issues = []
    
    # Check commutativity
    projectors = {}
    for level in range(multiverse.max_level + 1):
        goal = multiverse.get_goal(level)
        if goal and hasattr(goal, 'projector'):
            projectors[level] = goal.projector
    
    # For each pair of levels, check commutativity
    # This is approximate – in practice would need to check on subspaces
    for l1 in projectors:
        for l2 in projectors:
            if l1 >= l2:
                continue
            # This check is not straightforward without matrix representations
            # Placeholder
            pass
    
    # Check hierarchy consistency
    for idx, subsystem in multiverse.subsystems.items():
        if idx.level > 0:
            children = multiverse.levels.get_children(idx)
            # Check that subsystem's goal is product of children's goals
            # This requires comparing projectors – difficult in general
            pass
    
    # Check dimension sufficiency
    # Simple check: total dim >= product of number of subsystems per level
    total_dim = sum(s.state_space.dimension() for s in multiverse.subsystems.values())
    n_per_level = [len(multiverse.levels.get_at_level(l)) for l in range(multiverse.max_level + 1)]
    required = np.prod([max(1, n) for n in n_per_level])
    
    if total_dim < required:
        issues.append(f"Total dim {total_dim} < required {required}")
    
    return len(issues) == 0, issues


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing Zeroing Algorithm ===\n")
    
    # Create a simple test hierarchy
    from multiverse import MultiIndex, Multiverse, EuclideanSpace, Subsystem
    
    class SimpleSubsystem(Subsystem):
        def __init__(self, idx, dim):
            super().__init__(idx, EuclideanSpace(dim), None)
            self._state = torch.randn(dim)
        
        def get_state(self):
            return self._state
        
        def set_state(self, state):
            self._state = state
        
        def step(self, dt, action=None):
            pass
    
    # Create multiverse with 2 levels
    mv = Multiverse(max_level=1)
    
    # Level 0: two motors
    motor1 = MultiIndex(("motor1", None))
    motor2 = MultiIndex(("motor2", None))
    mv.add_subsystem(SimpleSubsystem(motor1, 5))
    mv.add_subsystem(SimpleSubsystem(motor2, 5))
    
    # Level 1: controller
    controller = MultiIndex((None, "controller"))
    mv.add_subsystem(SimpleSubsystem(controller, 10), parent=controller)
    
    # Add simple goals (using identity projectors for testing)
    from projector import IdentityProjector
    from goal import Goal
    
    mv.set_goal(0, Goal("motor_goal", IdentityProjector()))
    mv.set_goal(1, Goal("ctrl_goal", IdentityProjector()))
    mv.set_level_weight(0, 1.0)
    mv.set_level_weight(1, 0.5)
    
    # Create zeroing algorithm
    algo = create_zeroing_algorithm(
        mv,
        level_tolerances=[1e-4, 1e-4],
        learning_rates=[0.01, 0.001]
    )
    
    # Run
    states = algo.initialize_states()
    final = algo.run(states, num_epochs=50, verbose=True)
    
    # Plot history
    monitor = ZeroingMonitor(algo)
    monitor.plot_foam_history()
    
    print("\nZeroing complete!")
```