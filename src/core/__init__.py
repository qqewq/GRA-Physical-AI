"""
GRA Physical AI - Core Module
============================

This module contains the fundamental building blocks of the GRA Meta‑zeroing framework:
- MultiIndex: hierarchical addressing of subsystems
- HilbertSpace: abstract representation of state spaces
- Projector: goal representation as subspace projection
- Goal: high‑level goal with projector and loss
- Subsystem: a component identified by MultiIndex
- Functional: recursive total functional J
- ZeroingAlgorithm: the core recursive zeroing procedure

All implementations are tensor‑based (PyTorch) for GPU acceleration and differentiability.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import OrderedDict
import json


# ======================================================================
# MultiIndex – Hierarchical Addressing
# ======================================================================

@dataclass(frozen=True)
class MultiIndex:
    """
    Hierarchical address of a subsystem in the GRA multiverse.
    
    A MultiIndex is a tuple of strings of length L (the level).
    Each element identifies the subsystem at that level.
    The last element (index L-1) is the highest level containing this subsystem.
    
    Examples:
        - (motor_name, None, None) – level 0 motor
        - (None, limb_name, None) – level 1 limb (contains motors)
        - (None, None, planner) – level 2 planner
    """
    indices: Tuple[Optional[str], ...]
    
    def __post_init__(self):
        # Ensure it's a tuple of strings or None
        object.__setattr__(self, 'indices', tuple(self.indices))
        for idx in self.indices:
            assert idx is None or isinstance(idx, str), "Each index must be string or None"
    
    @property
    def level(self) -> int:
        """Level = length of tuple - 1 (0‑based)."""
        return len(self.indices) - 1
    
    @property
    def is_root(self) -> bool:
        """True if this is the highest level (all None except last?)."""
        # In practice, root has no None beyond its level? We'll define later.
        return all(idx is None for idx in self.indices[:-1]) and self.indices[-1] is not None
    
    def contains(self, other: 'MultiIndex') -> bool:
        """
        Does this multi‑index contain `other` as a subsystem?
        That is, `other` is a prefix of `self`.
        """
        if len(other.indices) >= len(self.indices):
            return False
        return all(self.indices[i] == other.indices[i] or other.indices[i] is None 
                   for i in range(len(other.indices)))
    
    def __str__(self):
        return "/".join(str(i) if i is not None else "*" for i in self.indices)
    
    def __repr__(self):
        return f"MultiIndex({self.indices})"
    
    def to_json(self) -> str:
        return json.dumps(self.indices)
    
    @classmethod
    def from_json(cls, s: str) -> 'MultiIndex':
        return cls(tuple(json.loads(s)))


# ======================================================================
# HilbertSpace – Abstract State Space
# ======================================================================

class HilbertSpace(ABC):
    """
    Abstract base class for a Hilbert space of states.
    
    In practice, we use finite‑dimensional spaces with a metric (inner product).
    The state is represented as a torch.Tensor.
    """
    
    @abstractmethod
    def inner_product(self, psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Inner product ⟨ψ|φ⟩."""
        pass
    
    @abstractmethod
    def norm(self, psi: torch.Tensor) -> torch.Tensor:
        """Norm of a state."""
        return torch.sqrt(self.inner_product(psi, psi))
    
    @abstractmethod
    def zero_state(self) -> torch.Tensor:
        """Return the zero vector in this space."""
        pass
    
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the space (or approximate if infinite)."""
        pass


class EuclideanSpace(HilbertSpace):
    """Simple Euclidean space with standard dot product."""
    
    def __init__(self, dim: int):
        self.dim = dim
        
    def inner_product(self, psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return torch.dot(psi.flatten(), phi.flatten())
    
    def zero_state(self) -> torch.Tensor:
        return torch.zeros(self.dim)
    
    def dimension(self) -> int:
        return self.dim


# ======================================================================
# Projector – Goal Representation
# ======================================================================

class Projector(ABC):
    """
    Abstract base class for a projector onto a goal subspace.
    
    A projector P satisfies:
        P^2 = P
        P† = P (self‑adjoint)
        Range(P) = goal‑satisfying subspace
    """
    
    @abstractmethod
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply projector to a state: return P|ψ⟩.
        For differentiable projectors, this should support gradient computation.
        """
        pass
    
    @abstractmethod
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        """
        Loss function measuring distance to goal subspace.
        Should be 0 iff state is in the subspace.
        Typically: loss = ‖(I - P)|ψ⟩‖²
        """
        pass


class IdentityProjector(Projector):
    """Identity projector – always satisfied."""
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return state
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=state.device)


class HardThresholdProjector(Projector):
    """
    Projector based on a threshold: projects to 1 if above threshold, else 0.
    Not differentiable – use for evaluation only.
    """
    
    def __init__(self, threshold: float):
        self.threshold = threshold
        
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return (state > self.threshold).float()
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        return torch.mean((state - self(state)) ** 2)


class SoftThresholdProjector(Projector):
    """
    Differentiable approximation of threshold projector using sigmoid.
    """
    
    def __init__(self, threshold: float, temperature: float = 1.0):
        self.threshold = threshold
        self.temperature = temperature
        
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid((state - self.threshold) / self.temperature)
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        return torch.mean((state - self(state)) ** 2)


class CompositeProjector(Projector):
    """Projector that is the tensor product of several projectors."""
    
    def __init__(self, projectors: List[Projector]):
        self.projectors = projectors
        
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        # Assume state is already flattened concatenation of subspaces
        # This is tricky – in practice, we'd need to split state
        # For now, return state (placeholder)
        return state
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        # Would need to split state per projector
        return total_loss


# ======================================================================
# Goal – High‑level Goal with Projector and Loss
# ======================================================================

class Goal(ABC):
    """
    A goal at a given level, with its projector and loss function.
    """
    
    def __init__(self, name: str, projector: Projector):
        self.name = name
        self.projector = projector
        
    @abstractmethod
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        """Loss for this goal (delegates to projector.loss by default)."""
        return self.projector.loss(state)
    
    def project(self, state: torch.Tensor) -> torch.Tensor:
        """Project state onto goal subspace."""
        return self.projector(state)


# ======================================================================
# Subsystem – A Component in the Multiverse
# ======================================================================

class Subsystem(ABC):
    """
    A component of the GRA multiverse, identified by a MultiIndex.
    
    Each subsystem has:
        - A state space (HilbertSpace)
        - A current state vector
        - A goal (Projector) for its level
        - Children (lower‑level subsystems) that it contains
        - Parents (higher‑level systems) that contain it
    """
    
    def __init__(self, multi_index: MultiIndex, state_space: HilbertSpace, goal: Goal):
        self.multi_index = multi_index
        self.state_space = state_space
        self.goal = goal
        self.children: List[MultiIndex] = []  # multi‑indices of subsystems this contains
        self.parents: List[MultiIndex] = []   # multi‑indices of systems that contain this
        self._state: Optional[torch.Tensor] = None
        
    @abstractmethod
    def get_state(self) -> torch.Tensor:
        """Return current state of this subsystem."""
        if self._state is None:
            self._state = self.state_space.zero_state()
        return self._state
    
    @abstractmethod
    def set_state(self, new_state: torch.Tensor):
        """Set the state of this subsystem (for planning/zeroing)."""
        assert new_state.shape == (self.state_space.dimension(),)
        self._state = new_state
    
    @abstractmethod
    def step(self, dt: float):
        """Update subsystem dynamics (if any)."""
        pass
    
    @property
    def level(self) -> int:
        return self.multi_index.level
    
    def __repr__(self):
        return f"Subsystem({self.multi_index}, level={self.level})"


class SimpleSubsystem(Subsystem):
    """
    Simple implementation where state is stored directly.
    """
    
    def get_state(self) -> torch.Tensor:
        if self._state is None:
            self._state = self.state_space.zero_state()
        return self._state
    
    def set_state(self, new_state: torch.Tensor):
        self._state = new_state.clone()
    
    def step(self, dt: float):
        # No dynamics by default
        pass


# ======================================================================
# Functional – Recursive Total Functional J
# ======================================================================

class Functional:
    """
    Implements the recursive total functional J_total.
    
    J^(0)(ψ) = J_local(ψ, G₀)
    J^(l)(ψ) = Σ_{children} J^(l-1)(ψ_child) + Φ^(l)(ψ, G_l)
    
    J_total = Σ_l Λ_l Σ_{subsystems at level l} J^(l)
    """
    
    def __init__(self, hierarchy: Dict[MultiIndex, Subsystem], 
                 goals: Dict[int, Goal],
                 level_weights: List[float],
                 local_loss_fns: Dict[MultiIndex, Callable]):
        """
        Args:
            hierarchy: mapping from multi‑index to Subsystem
            goals: mapping from level to Goal for that level
            level_weights: Λ_l for each level
            local_loss_fns: mapping from multi‑index to function computing J_local
        """
        self.hierarchy = hierarchy
        self.goals = goals
        self.level_weights = level_weights
        self.local_loss_fns = local_loss_fns
        
        # Precompute level structure
        self.by_level: Dict[int, List[MultiIndex]] = {}
        for idx in hierarchy:
            l = idx.level
            if l not in self.by_level:
                self.by_level[l] = []
            self.by_level[l].append(idx)
            
        self.max_level = max(self.by_level.keys())
        
    def compute_foam(self, level: int, states: Dict[MultiIndex, torch.Tensor]) -> torch.Tensor:
        """
        Compute foam at given level: Φ^(l) = Σ_{a≠b} |⟨ψ_a|P_G|ψ_b⟩|²
        """
        if level not in self.by_level:
            return torch.tensor(0.0)
        
        indices = self.by_level[level]
        if len(indices) < 2:
            return torch.tensor(0.0)
        
        goal = self.goals.get(level)
        if goal is None:
            return torch.tensor(0.0)
        
        foam = 0.0
        for i, a in enumerate(indices):
            for j, b in enumerate(indices):
                if i >= j:
                    continue
                psi_a = states[a]
                psi_b = states[b]
                
                # Compute ⟨ψ_a| P_G |ψ_b⟩
                proj_b = goal.project(psi_b)
                overlap = torch.dot(psi_a.flatten(), proj_b.flatten())
                foam += overlap ** 2
                
        return foam
    
    def compute_level_functional(self, level: int, 
                                  idx: MultiIndex,
                                  states: Dict[MultiIndex, torch.Tensor],
                                  cache: Dict[Tuple[int, MultiIndex], torch.Tensor]) -> torch.Tensor:
        """
        Compute J^(l)(ψ^(idx)) recursively.
        """
        if (level, idx) in cache:
            return cache[(level, idx)]
        
        subsystem = self.hierarchy[idx]
        
        if level == 0:
            # Base case
            if idx in self.local_loss_fns:
                result = self.local_loss_fns[idx](states[idx])
            else:
                result = subsystem.goal.loss(states[idx])
            cache[(level, idx)] = result
            return result
        
        # Recursive case: sum over children + foam
        children_sum = 0.0
        for child_idx in subsystem.children:
            child_level = child_idx.level
            children_sum += self.compute_level_functional(child_level, child_idx, states, cache)
        
        # Foam at this level for this subsystem? Wait, foam is per level, not per subsystem.
        # The definition says: Φ^(l) depends on all states at level l, not just one subsystem.
        # So we need to compute foam separately and then add it to each subsystem's functional.
        # We'll handle foam outside.
        
        result = children_sum  # + foam will be added separately
        cache[(level, idx)] = result
        return result
    
    def total(self, states: Dict[MultiIndex, torch.Tensor]) -> torch.Tensor:
        """
        Compute J_total = Σ_l Λ_l Σ_{idx at level l} J^(l)(idx)
        """
        # First compute all J^(l) recursively (without foam)
        cache = {}
        for l in range(self.max_level + 1):
            for idx in self.by_level.get(l, []):
                self.compute_level_functional(l, idx, states, cache)
        
        # Now add foam at each level
        total = 0.0
        for l in range(self.max_level + 1):
            level_foam = self.compute_foam(l, states)
            
            # Add foam to each subsystem's functional at this level
            for idx in self.by_level.get(l, []):
                j_l = cache.get((l, idx), torch.tensor(0.0))
                total += self.level_weights[l] * (j_l + level_foam / len(self.by_level.get(l, [1])))
                
        return total
    
    def gradient(self, states: Dict[MultiIndex, torch.Tensor]) -> Dict[MultiIndex, torch.Tensor]:
        """
        Compute gradients of J_total w.r.t. each subsystem's state.
        
        Returns: dict mapping multi‑index → gradient tensor
        """
        # Requires autograd – we'll implement a simple version for now
        # In practice, use torch.autograd.grad
        grads = {}
        
        # For each subsystem, we need:
        # ∂J_total/∂ψ_a = Λ_l * ∂Φ^(l)/∂ψ_a + Σ_{levels > l} contributions
        
        # We'll implement the full formula from multiverse.md:
        # ∂J_total/∂ψ^(a) = Λ_l ∂Φ^(l)/∂ψ^(a) + Σ_{b > a} Λ_{l+1} ∂Φ^(l+1)/∂ψ^(a)
        
        for l in range(self.max_level + 1):
            for a in self.by_level.get(l, []):
                grad = torch.zeros_like(states[a])
                
                # Contribution from own level's foam
                if l in self.goals:
                    goal = self.goals[l]
                    # ∂Φ^(l)/∂ψ^(a) = 2 Σ_{b≠a} ⟨ψ_a|P_G|ψ_b⟩ · P_G|ψ_b⟩
                    for b in self.by_level.get(l, []):
                        if a == b:
                            continue
                        proj_b = goal.project(states[b])
                        overlap = torch.dot(states[a].flatten(), proj_b.flatten())
                        grad += 2 * overlap * proj_b.flatten()
                
                # Contributions from higher levels
                for higher_l in range(l+1, self.max_level + 1):
                    if higher_l not in self.goals:
                        continue
                    goal_h = self.goals[higher_l]
                    # Find all subsystems at level higher_l that contain a
                    for b in self.by_level.get(higher_l, []):
                        if self.hierarchy[b].multi_index.contains(a):
                            # ∂Φ^(higher_l)/∂ψ^(a) – need chain rule through the fact that
                            # ψ^(b) depends on ψ^(a). This is complex.
                            # For now, approximate as zero.
                            pass
                
                grads[a] = grad.view(states[a].shape)
                
        return grads


# ======================================================================
# ZeroingAlgorithm – Core Recursive Zeroing
# ======================================================================

class ZeroingAlgorithm:
    """
    Implements the recursive zeroing algorithm from algorithm.md.
    
    Usage:
        zeroing = ZeroingAlgorithm(hierarchy, goals, level_weights, local_loss_fns)
        states = zeroing.zero(states_init, num_epochs=100)
    """
    
    def __init__(self, hierarchy: Dict[MultiIndex, Subsystem],
                 goals: Dict[int, Goal],
                 level_weights: List[float],
                 local_loss_fns: Dict[MultiIndex, Callable],
                 learning_rates: List[float],
                 tolerances: List[float]):
        """
        Args:
            hierarchy: mapping multi‑index → Subsystem
            goals: mapping level → Goal
            level_weights: Λ_l
            local_loss_fns: J_local for each subsystem
            learning_rates: η_l for each level
            tolerances: ε_l for each level
        """
        self.functional = Functional(hierarchy, goals, level_weights, local_loss_fns)
        self.hierarchy = hierarchy
        self.goals = goals
        self.learning_rates = learning_rates
        self.tolerances = tolerances
        
        # Organize by level
        self.by_level: Dict[int, List[MultiIndex]] = {}
        for idx in hierarchy:
            l = idx.level
            if l not in self.by_level:
                self.by_level[l] = []
            self.by_level[l].append(idx)
        self.max_level = max(self.by_level.keys())
        
    def zero_level(self, level: int, 
                   states: Dict[MultiIndex, torch.Tensor],
                   max_iters: int = 100) -> Dict[MultiIndex, torch.Tensor]:
        """
        Recursively zero foam at this level and below.
        Implements the algorithm from algorithm.md.
        """
        if level == 0:
            # Base case: just optimize local goals
            for idx in self.by_level.get(0, []):
                for _ in range(max_iters):
                    loss = self.functional.local_loss_fns.get(idx, 
                                                              self.hierarchy[idx].goal.loss)(states[idx])
                    if loss < self.tolerances[0]:
                        break
                    # Simple gradient descent on local loss
                    # In practice, use optimizer
                    grad = torch.autograd.grad(loss, states[idx], retain_graph=True)[0]
                    states[idx] = states[idx] - self.learning_rates[0] * grad
            return states
        
        # Recursive case: first zero all lower levels
        for idx in self.by_level.get(level, []):
            for child_idx in self.hierarchy[idx].children:
                states = self.zero_level(child_idx.level, states, max_iters)
        
        # Now reduce foam at this level
        for iteration in range(max_iters):
            # Compute foam at this level
            foam = self.functional.compute_foam(level, states)
            if foam < self.tolerances[level]:
                break
            
            # Compute gradients w.r.t. each state at this level
            for a in self.by_level.get(level, []):
                # Simplified gradient: just from foam at this level
                goal = self.goals.get(level)
                if goal is None:
                    continue
                    
                grad = torch.zeros_like(states[a])
                for b in self.by_level.get(level, []):
                    if a == b:
                        continue
                    proj_b = goal.project(states[b])
                    overlap = torch.dot(states[a].flatten(), proj_b.flatten())
                    grad += 2 * overlap * proj_b.flatten()
                
                states[a] = states[a] - self.learning_rates[level] * grad.view(states[a].shape)
        
        return states
    
    def zero(self, initial_states: Dict[MultiIndex, torch.Tensor], 
             num_epochs: int = 1000,
             callback: Optional[Callable] = None) -> Dict[MultiIndex, torch.Tensor]:
        """
        Run full multiverse zeroing.
        
        Args:
            initial_states: starting states for all subsystems
            num_epochs: number of zeroing epochs
            callback: optional function called each epoch with (epoch, states, foams)
            
        Returns:
            zeroed states
        """
        states = initial_states.copy()
        
        for epoch in range(num_epochs):
            # Recursive zeroing from top level
            states = self.zero_level(self.max_level, states)
            
            # Compute all foams for monitoring
            foams = {l: self.functional.compute_foam(l, states).item() 
                     for l in range(self.max_level + 1)}
            
            if callback:
                callback(epoch, states, foams)
            
            # Check convergence
            if all(foams[l] < self.tolerances[l] for l in foams):
                print(f"Converged at epoch {epoch}")
                break
                
        return states


# ======================================================================
# Utility Functions
# ======================================================================

def tensor_product(states: List[torch.Tensor]) -> torch.Tensor:
    """
    Compute tensor product of multiple state vectors.
    For efficiency, we return the Kronecker product.
    """
    result = states[0]
    for s in states[1:]:
        result = torch.kron(result, s)
    return result


def partial_trace(full_state: torch.Tensor, dims: List[int], keep: List[int]) -> torch.Tensor:
    """
    Partial trace over subsystems.
    
    Args:
        full_state: density matrix or state vector
        dims: dimensions of each subsystem
        keep: indices of subsystems to keep
    
    Returns:
        reduced state
    """
    # For pure states, |ψ⟩⟨ψ| is needed – simplified version
    # In practice, use einsum
    n = len(dims)
    keep_dims = [dims[i] for i in keep]
    keep_indices = keep
    
    # Reshape into tensor
    psi_tensor = full_state.view(*dims)
    
    # Contract over traced-out indices
    # This is a simplified placeholder
    return psi_tensor


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    # Create a simple two‑level hierarchy
    # Level 0: two motors
    motor1_idx = MultiIndex(("motor1", None, None))
    motor2_idx = MultiIndex(("motor2", None, None))
    # Level 1: navigator
    nav_idx = MultiIndex((None, "navigator", None))
    
    # Create subsystems
    motor1 = SimpleSubsystem(motor1_idx, EuclideanSpace(2), 
                              Goal("motor_goal", SoftThresholdProjector(0.5)))
    motor2 = SimpleSubsystem(motor2_idx, EuclideanSpace(2),
                              Goal("motor_goal", SoftThresholdProjector(0.5)))
    nav = SimpleSubsystem(nav_idx, EuclideanSpace(6),
                          Goal("nav_goal", IdentityProjector()))
    
    # Set children
    nav.children = [motor1_idx, motor2_idx]
    
    hierarchy = {
        motor1_idx: motor1,
        motor2_idx: motor2,
        nav_idx: nav
    }
    
    # Goals
    goals = {
        0: Goal("motor_goal", SoftThresholdProjector(0.5)),
        1: Goal("nav_goal", IdentityProjector())
    }
    
    # Local loss functions
    def motor_loss(state):
        return (state[1] - state[0]) ** 2
    
    local_losses = {
        motor1_idx: motor_loss,
        motor2_idx: motor_loss,
        nav_idx: lambda s: torch.sum(s**2)  # placeholder
    }
    
    # Level weights
    level_weights = [1.0, 0.5]
    
    # Learning rates
    learning_rates = [0.01, 0.001]
    
    # Tolerances
    tolerances = [0.01, 0.01]
    
    # Create zeroing algorithm
    zeroing = ZeroingAlgorithm(hierarchy, goals, level_weights, local_losses,
                               learning_rates, tolerances)
    
    # Initial states
    states = {
        motor1_idx: torch.tensor([0.5, 0.3]),  # cmd, actual
        motor2_idx: torch.tensor([0.5, 0.4]),
        nav_idx: torch.randn(6)
    }
    
    # Define callback
    def print_callback(epoch, states, foams):
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: foams = {foams}")
    
    # Run zeroing
    final_states = zeroing.zero(states, num_epochs=100, callback=print_callback)
    
    print("Zeroing complete!")
    for idx, state in final_states.items():
        print(f"{idx}: {state}")