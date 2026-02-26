```python
"""
GRA Physical AI - Multiverse Module
===================================

This module defines the core structures for the GRA multiverse:
- MultiIndex: hierarchical addressing of subsystems
- HilbertSpace: abstract and concrete state spaces
- TensorProductSpace: composition of spaces
- Multiverse: container for all subsystems and their relationships
- LevelStructure: organization of subsystems by level

These classes form the foundation for building hierarchical AI systems.
All implementations support PyTorch tensors for GPU acceleration and differentiability.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Set, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
import json
import warnings


# ======================================================================
# MultiIndex – Hierarchical Addressing
# ======================================================================

@dataclass(frozen=True, order=True)
class MultiIndex:
    """
    Hierarchical address of a subsystem in the GRA multiverse.
    
    A MultiIndex is a tuple of strings (or None) of length L (the level count).
    Each element identifies the subsystem at that level, with None indicating
    "wildcard" or "not applicable at this level".
    
    The last element (index L-1) is the highest level containing this subsystem.
    
    Examples:
        - (motor_name, None, None, None, None) – level 0 motor
        - (None, limb_name, None, None, None) – level 1 limb
        - (None, None, planner_name, None, None) – level 2 planner
        - (None, None, None, None, ethics) – level 4 ethics supervisor
        - (robot1, motor_left, limb_arm, planner, ethics) – full 5-level index
    
    Properties:
        - level: the depth (length-1)
        - is_root: whether this is the top-most subsystem
        - ancestors: all containing subsystems
        - descendants: all contained subsystems (if known from Multiverse)
    """
    indices: Tuple[Optional[str], ...]
    
    def __post_init__(self):
        # Ensure it's a tuple of strings or None
        object.__setattr__(self, 'indices', tuple(self.indices))
        for idx in self.indices:
            assert idx is None or isinstance(idx, str), f"Each index must be string or None, got {type(idx)}"
    
    @property
    def level(self) -> int:
        """Level of this subsystem (0 = lowest, increasing)."""
        return len(self.indices) - 1
    
    @property
    def max_level(self) -> int:
        """Maximum possible level (length-1)."""
        return len(self.indices) - 1
    
    @property
    def is_root(self) -> bool:
        """True if this is the highest level (all lower levels are None, last is not None)."""
        return all(idx is None for idx in self.indices[:-1]) and self.indices[-1] is not None
    
    @property
    def is_wildcard(self) -> bool:
        """True if all indices are None (used for matching)."""
        return all(idx is None for idx in self.indices)
    
    def contains(self, other: 'MultiIndex') -> bool:
        """
        Does this multi‑index contain `other` as a subsystem?
        That is, `other` is a prefix of `self` with matching non‑None values.
        
        Example:
            a = (robot1, None, None, None)
            b = (robot1, left_motor, None, None)
            a.contains(b) → True
        """
        if len(other.indices) >= len(self.indices):
            return False
        
        for i in range(len(other.indices)):
            if other.indices[i] is not None and self.indices[i] != other.indices[i]:
                return False
        return True
    
    def is_ancestor_of(self, other: 'MultiIndex') -> bool:
        """Alias for contains."""
        return self.contains(other)
    
    def is_descendant_of(self, other: 'MultiIndex') -> bool:
        """Check if this is a descendant of other."""
        return other.contains(self)
    
    def common_ancestor(self, other: 'MultiIndex') -> Optional['MultiIndex']:
        """
        Find the lowest common ancestor of two multi‑indices.
        Returns None if they share no common ancestor.
        """
        min_len = min(len(self.indices), len(other.indices))
        common = []
        for i in range(min_len):
            if self.indices[i] == other.indices[i] and self.indices[i] is not None:
                common.append(self.indices[i])
            elif self.indices[i] is None and other.indices[i] is not None:
                # One has wildcard – still compatible if the other's value matches previous?
                # This is ambiguous. For simplicity, stop when mismatch.
                break
            elif self.indices[i] is not None and other.indices[i] is None:
                break
            else:
                # Both None – continue
                common.append(None)
        
        if not common:
            return None
        
        # Pad with None to full length
        while len(common) < len(self.indices):
            common.append(None)
        
        return MultiIndex(tuple(common))
    
    def prefix(self, level: int) -> 'MultiIndex':
        """Return the prefix up to given level (inclusive)."""
        assert 0 <= level <= self.level
        return MultiIndex(self.indices[:level+1] + (None,) * (self.max_level - level))
    
    def with_level(self, level: int, value: Optional[str]) -> 'MultiIndex':
        """Return a new MultiIndex with the given level set to value."""
        new_indices = list(self.indices)
        new_indices[level] = value
        return MultiIndex(tuple(new_indices))
    
    def __str__(self) -> str:
        parts = []
        for i, idx in enumerate(self.indices):
            if idx is not None:
                parts.append(f"L{i}:{idx}")
        return "/".join(parts) if parts else "*"
    
    def __repr__(self) -> str:
        return f"MultiIndex({self.indices})"
    
    def to_json(self) -> str:
        return json.dumps(self.indices)
    
    @classmethod
    def from_json(cls, s: str) -> 'MultiIndex':
        return cls(tuple(json.loads(s)))
    
    @classmethod
    def from_string(cls, s: str, levels: int) -> 'MultiIndex':
        """
        Create from string like "L0:motor/L1:limb/L2:planner".
        Missing levels become None.
        """
        indices = [None] * (levels + 1)
        parts = s.split('/')
        for part in parts:
            if part.startswith('L'):
                level_str, value = part.split(':', 1)
                level = int(level_str[1:])
                indices[level] = value
        return cls(tuple(indices))


# ======================================================================
# HilbertSpace – Abstract State Space
# ======================================================================

class HilbertSpace(ABC):
    """
    Abstract base class for a Hilbert space of states.
    
    In practice, we work with finite-dimensional spaces where states are
    represented as PyTorch tensors. Each space defines:
        - Inner product (for computing overlaps)
        - Zero state
        - Dimension
        - Basis (optional)
    
    All operations should be differentiable where possible.
    """
    
    @abstractmethod
    def inner_product(self, psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """
        Inner product ⟨ψ|φ⟩.
        
        Args:
            psi, phi: state vectors in this space
        
        Returns:
            scalar tensor (complex for quantum, real for classical)
        """
        pass
    
    def norm(self, psi: torch.Tensor) -> torch.Tensor:
        """Norm of a state: √⟨ψ|ψ⟩."""
        return torch.sqrt(self.inner_product(psi, psi))
    
    def distance(self, psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        """Distance between two states: ‖ψ - φ‖."""
        diff = psi - phi
        return self.norm(diff)
    
    @abstractmethod
    def zero_state(self) -> torch.Tensor:
        """Return the zero vector in this space."""
        pass
    
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of the space (finite)."""
        pass
    
    @abstractmethod
    def random_state(self) -> torch.Tensor:
        """Return a random state (for initialization)."""
        pass
    
    def project(self, psi: torch.Tensor, subspace: 'HilbertSpace') -> torch.Tensor:
        """
        Project state onto a subspace (if subspace is a subset).
        Default implementation uses basis if available.
        """
        raise NotImplementedError
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(dim={self.dimension()})"


class EuclideanSpace(HilbertSpace):
    """
    Euclidean space ℝⁿ with standard dot product.
    Most common for classical robotics states.
    """
    
    def __init__(self, dim: int, dtype: torch.dtype = torch.float32):
        self._dim = dim
        self.dtype = dtype
        
    def inner_product(self, psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return torch.dot(psi.flatten().to(self.dtype), 
                        phi.flatten().to(self.dtype))
    
    def zero_state(self) -> torch.Tensor:
        return torch.zeros(self._dim, dtype=self.dtype)
    
    def dimension(self) -> int:
        return self._dim
    
    def random_state(self) -> torch.Tensor:
        return torch.randn(self._dim, dtype=self.dtype)
    
    def __repr__(self) -> str:
        return f"EuclideanSpace(dim={self._dim})"


class ComplexSpace(HilbertSpace):
    """
    Complex Hilbert space ℂⁿ with Hermitian inner product.
    For quantum robotics or complex representations.
    """
    
    def __init__(self, dim: int, dtype: torch.dtype = torch.complex64):
        self._dim = dim
        self.dtype = dtype
        
    def inner_product(self, psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # ⟨ψ|φ⟩ = ψ† φ
        return torch.dot(psi.flatten().conj(), phi.flatten())
    
    def zero_state(self) -> torch.Tensor:
        return torch.zeros(self._dim, dtype=self.dtype)
    
    def dimension(self) -> int:
        return self._dim
    
    def random_state(self) -> torch.Tensor:
        real = torch.randn(self._dim)
        imag = torch.randn(self._dim)
        return torch.complex(real, imag)
    
    def __repr__(self) -> str:
        return f"ComplexSpace(dim={self._dim})"


class TensorProductSpace(HilbertSpace):
    """
    Tensor product of multiple Hilbert spaces.
    
    This represents the combined space of several subsystems.
    The total state is the tensor product of individual states.
    """
    
    def __init__(self, spaces: List[HilbertSpace]):
        self.spaces = spaces
        self._dim = np.prod([s.dimension() for s in spaces]).astype(int)
        
    def inner_product(self, psi: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        # For product states, inner product factorizes
        # But psi and phi may not be product states – fallback to flatten
        return torch.dot(psi.flatten(), phi.flatten())
    
    def zero_state(self) -> torch.Tensor:
        # Zero in tensor product = tensor product of zeros
        # But that's just a single zero – we need the zero vector in the full space
        return torch.zeros(self._dim, dtype=self.spaces[0].dtype)
    
    def dimension(self) -> int:
        return self._dim
    
    def random_state(self) -> torch.Tensor:
        # Random product state
        states = [s.random_state() for s in self.spaces]
        return self.tensor_product(states)
    
    def tensor_product(self, states: List[torch.Tensor]) -> torch.Tensor:
        """
        Combine states from each subspace into a full tensor product state.
        """
        result = states[0]
        for s in states[1:]:
            result = torch.kron(result, s)
        return result
    
    def factorize(self, full_state: torch.Tensor) -> List[torch.Tensor]:
        """
        Attempt to factorize a full state into product components.
        If the state is entangled, this will be an approximation.
        """
        # Simple implementation: reshape and take first component
        # In practice, use SVD
        factors = []
        remaining = full_state
        for i, space in enumerate(self.spaces):
            dim = space.dimension()
            # Reshape to (dim, -1)
            reshaped = remaining.view(dim, -1)
            # Take first column as approximate factor
            factor = reshaped[:, 0] / torch.norm(reshaped[:, 0])
            factors.append(factor)
            # Not truly reversible – this is a simplification
        return factors
    
    def __repr__(self) -> str:
        return f"TensorProductSpace({self.spaces})"


# ======================================================================
# LevelStructure – Organization by Level
# ======================================================================

class LevelStructure:
    """
    Organizes subsystems by their level in the hierarchy.
    
    Provides efficient queries for:
        - All subsystems at a given level
        - Children/parent relationships
        - Subsystems containing a given index
        - Common ancestors
    """
    
    def __init__(self, max_level: int):
        """
        Args:
            max_level: highest level number (e.g., 4 for 5 levels)
        """
        self.max_level = max_level
        self.by_level: Dict[int, Set[MultiIndex]] = defaultdict(set)
        self.all_indices: Set[MultiIndex] = set()
        self.parent_map: Dict[MultiIndex, Set[MultiIndex]] = defaultdict(set)
        self.child_map: Dict[MultiIndex, Set[MultiIndex]] = defaultdict(set)
        
    def add_subsystem(self, idx: MultiIndex, parent: Optional[MultiIndex] = None):
        """Add a subsystem to the structure."""
        assert idx.level == len(idx.indices) - 1
        assert idx.level <= self.max_level
        
        self.by_level[idx.level].add(idx)
        self.all_indices.add(idx)
        
        if parent is not None:
            self.parent_map[idx].add(parent)
            self.child_map[parent].add(idx)
    
    def get_at_level(self, level: int) -> Set[MultiIndex]:
        """Get all subsystems at a given level."""
        return self.by_level.get(level, set())
    
    def get_children(self, idx: MultiIndex) -> Set[MultiIndex]:
        """Get all direct children of a subsystem."""
        return self.child_map.get(idx, set())
    
    def get_parents(self, idx: MultiIndex) -> Set[MultiIndex]:
        """Get all direct parents of a subsystem."""
        return self.parent_map.get(idx, set())
    
    def get_ancestors(self, idx: MultiIndex, include_self: bool = False) -> Set[MultiIndex]:
        """Get all ancestors (containing subsystems)."""
        ancestors = set()
        if include_self:
            ancestors.add(idx)
        
        current = idx
        while True:
            parents = self.get_parents(current)
            if not parents:
                break
            # In a tree, there should be at most one parent
            current = next(iter(parents))
            ancestors.add(current)
        
        return ancestors
    
    def get_descendants(self, idx: MultiIndex, include_self: bool = False) -> Set[MultiIndex]:
        """Get all descendants (contained subsystems)."""
        descendants = set()
        if include_self:
            descendants.add(idx)
        
        to_process = list(self.get_children(idx))
        while to_process:
            current = to_process.pop()
            descendants.add(current)
            to_process.extend(self.get_children(current))
        
        return descendants
    
    def get_all_indices(self) -> Set[MultiIndex]:
        """Get all subsystems in the structure."""
        return self.all_indices
    
    def is_leaf(self, idx: MultiIndex) -> bool:
        """True if subsystem has no children."""
        return len(self.get_children(idx)) == 0
    
    def is_root(self, idx: MultiIndex) -> bool:
        """True if subsystem has no parents."""
        return len(self.get_parents(idx)) == 0
    
    def get_root(self) -> Optional[MultiIndex]:
        """Get the top‑level subsystem (assuming tree, not DAG)."""
        for idx in self.all_indices:
            if self.is_root(idx):
                return idx
        return None
    
    def __repr__(self) -> str:
        return f"LevelStructure(levels={dict(self.by_level)})"


# ======================================================================
# Multiverse – Container for All Subsystems
# ======================================================================

class Multiverse:
    """
    Container for the entire GRA multiverse.
    
    Holds:
        - All subsystems, indexed by MultiIndex
        - Their state spaces
        - Their relationships (parent/child)
        - Goals for each level
    
    Provides methods for:
        - Navigating the hierarchy
        - Computing foam at each level
        - Accessing states in a structured way
        - Serialization
    """
    
    def __init__(self, name: str = "Multiverse", max_level: int = 4):
        """
        Args:
            name: identifier for this multiverse
            max_level: highest level number (e.g., 4 for G₀…G₄)
        """
        self.name = name
        self.max_level = max_level
        
        # Subsystems: MultiIndex -> Subsystem object
        self.subsystems: Dict[MultiIndex, 'Subsystem'] = {}
        
        # Level structure (for efficient queries)
        self.levels = LevelStructure(max_level)
        
        # Goals: level -> Goal object
        self.goals: Dict[int, 'Goal'] = {}
        
        # Level weights Λ_l
        self.level_weights: List[float] = [1.0] * (max_level + 1)
        
        # Cache for computed properties
        self._cache: Dict[str, Any] = {}
        
    def add_subsystem(self, subsystem: 'Subsystem', parent: Optional[MultiIndex] = None):
        """
        Add a subsystem to the multiverse.
        
        Args:
            subsystem: the subsystem to add
            parent: optional parent multi‑index (if known)
        """
        idx = subsystem.multi_index
        assert idx.level == subsystem.level
        assert idx not in self.subsystems, f"Duplicate subsystem {idx}"
        
        self.subsystems[idx] = subsystem
        self.levels.add_subsystem(idx, parent)
        
        # Clear cache
        self._cache.clear()
    
    def get_subsystem(self, idx: MultiIndex) -> Optional['Subsystem']:
        """Get subsystem by multi‑index."""
        return self.subsystems.get(idx)
    
    def get_state(self, idx: MultiIndex) -> Optional[torch.Tensor]:
        """Get current state of a subsystem."""
        sub = self.get_subsystem(idx)
        return sub.get_state() if sub else None
    
    def set_state(self, idx: MultiIndex, state: torch.Tensor):
        """Set state of a subsystem."""
        sub = self.get_subsystem(idx)
        if sub:
            sub.set_state(state)
    
    def get_all_states(self) -> Dict[MultiIndex, torch.Tensor]:
        """Get states of all subsystems."""
        return {idx: sub.get_state() for idx, sub in self.subsystems.items()}
    
    def get_states_at_level(self, level: int) -> Dict[MultiIndex, torch.Tensor]:
        """Get states of all subsystems at a given level."""
        return {idx: self.subsystems[idx].get_state() 
                for idx in self.levels.get_at_level(level)}
    
    def set_goal(self, level: int, goal: 'Goal'):
        """Set the goal for a given level."""
        assert 0 <= level <= self.max_level
        self.goals[level] = goal
    
    def get_goal(self, level: int) -> Optional['Goal']:
        """Get goal for a level."""
        return self.goals.get(level)
    
    def set_level_weight(self, level: int, weight: float):
        """Set Λ_l for a level."""
        assert 0 <= level <= self.max_level
        self.level_weights[level] = weight
    
    def compute_foam(self, level: int) -> torch.Tensor:
        """
        Compute foam at given level: Φ^(l) = Σ_{a≠b} |⟨ψ_a|P_G|ψ_b⟩|²
        """
        if level not in self.goals:
            return torch.tensor(0.0)
        
        indices = list(self.levels.get_at_level(level))
        if len(indices) < 2:
            return torch.tensor(0.0)
        
        goal = self.goals[level]
        foam = 0.0
        
        for i, a in enumerate(indices):
            psi_a = self.subsystems[a].get_state()
            for j, b in enumerate(indices):
                if i >= j:
                    continue
                psi_b = self.subsystems[b].get_state()
                
                # Compute ⟨ψ_a| P_G |ψ_b⟩
                proj_b = goal.project(psi_b)
                
                # Use the inner product from the state space of a
                space_a = self.subsystems[a].state_space
                overlap = space_a.inner_product(psi_a, proj_b)
                foam += torch.abs(overlap) ** 2
        
        return foam
    
    def compute_all_foams(self) -> Dict[int, torch.Tensor]:
        """Compute foam at all levels."""
        return {l: self.compute_foam(l) for l in range(self.max_level + 1)}
    
    def get_hierarchy_graph(self) -> Dict[MultiIndex, List[MultiIndex]]:
        """
        Return the parent/child relationships as a directed graph.
        Useful for visualization.
        """
        graph = {}
        for idx in self.subsystems:
            children = list(self.levels.get_children(idx))
            if children:
                graph[idx] = children
        return graph
    
    def validate(self) -> List[str]:
        """
        Validate the multiverse structure.
        Returns list of warnings/errors.
        """
        issues = []
        
        # Check that all indices have correct level
        for idx in self.subsystems:
            if idx.level != len(idx.indices) - 1:
                issues.append(f"Index {idx} has inconsistent level")
        
        # Check that parent/child relationships are consistent
        for idx in self.subsystems:
            for parent in self.levels.get_parents(idx):
                if not parent.contains(idx):
                    issues.append(f"Parent {parent} does not contain child {idx}")
        
        # Check that all levels have goals
        for l in range(self.max_level + 1):
            if l not in self.goals:
                issues.append(f"Level {l} has no goal")
        
        # Check level weights length
        if len(self.level_weights) != self.max_level + 1:
            issues.append(f"Level weights length {len(self.level_weights)} != {self.max_level + 1}")
        
        return issues
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'max_level': self.max_level,
            'subsystems': {str(idx): sub.to_dict() for idx, sub in self.subsystems.items()},
            'goals': {str(l): str(g) for l, g in self.goals.items()},
            'level_weights': self.level_weights,
            'structure': {
                'parents': {str(idx): [str(p) for p in self.levels.get_parents(idx)]
                           for idx in self.subsystems},
                'children': {str(idx): [str(c) for c in self.levels.get_children(idx)]
                            for idx in self.subsystems}
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Multiverse':
        """Deserialize from dictionary."""
        mv = cls(name=data['name'], max_level=data['max_level'])
        mv.level_weights = data['level_weights']
        
        # Subsystems would need to be reconstructed – depends on implementation
        # This is a placeholder
        return mv
    
    def __repr__(self) -> str:
        return f"Multiverse(name='{self.name}', levels={self.max_level+1}, subsystems={len(self.subsystems)})"


# ======================================================================
# Subsystem Base Class
# ======================================================================

class Subsystem(ABC):
    """
    Base class for any subsystem in the GRA multiverse.
    
    Each subsystem has:
        - A multi‑index identifying it
        - A state space (HilbertSpace)
        - A current state tensor
        - A goal (optional, can be overridden by level goal)
        - Dynamics (step method)
        - Methods for getting/setting state
    """
    
    def __init__(self, 
                 multi_index: MultiIndex,
                 state_space: HilbertSpace,
                 goal: Optional['Goal'] = None):
        """
        Args:
            multi_index: unique identifier
            state_space: the Hilbert space for this subsystem's states
            goal: optional specific goal for this subsystem (if different from level goal)
        """
        self.multi_index = multi_index
        self.state_space = state_space
        self.goal = goal
        
        # Internal state
        self._state: Optional[torch.Tensor] = None
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
        
    @property
    def level(self) -> int:
        return self.multi_index.level
    
    @abstractmethod
    def get_state(self) -> torch.Tensor:
        """Return current state."""
        if self._state is None:
            self._state = self.state_space.zero_state()
        return self._state
    
    @abstractmethod
    def set_state(self, new_state: torch.Tensor):
        """Set state (for zeroing/planning)."""
        assert new_state.shape == (self.state_space.dimension(),)
        self._state = new_state.clone()
    
    @abstractmethod
    def step(self, dt: float, action: Optional[torch.Tensor] = None):
        """
        Update subsystem dynamics.
        
        Args:
            dt: time step
            action: optional external input
        """
        pass
    
    def reset(self):
        """Reset to zero state."""
        self._state = self.state_space.zero_state()
    
    def randomize(self):
        """Set to random state."""
        self._state = self.state_space.random_state()
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        return {
            'multi_index': self.multi_index.to_json(),
            'level': self.level,
            'state_space': repr(self.state_space),
            'goal': repr(self.goal) if self.goal else None,
            'metadata': self.metadata
        }
    
    def __repr__(self) -> str:
        return f"Subsystem({self.multi_index}, space={self.state_space})"


# ======================================================================
# Utility Functions
# ======================================================================

def create_multi_index_from_path(path: List[Optional[str]], max_level: int) -> MultiIndex:
    """
    Create a MultiIndex from a list of values, padding with None.
    
    Example:
        create_multi_index_from_path(['motor1', 'limb'], max_level=4)
        → ('motor1', 'limb', None, None, None)
    """
    assert len(path) <= max_level + 1
    indices = list(path) + [None] * (max_level + 1 - len(path))
    return MultiIndex(tuple(indices))


def multi_index_product(indices: List[MultiIndex]) -> List[MultiIndex]:
    """
    Compute all combinations of multi‑indices? Not really product.
    This is more for generating all children of a parent.
    """
    # Placeholder – in practice, would generate all combinations of values
    return indices


def match_pattern(pattern: MultiIndex, target: MultiIndex) -> bool:
    """
    Check if target matches pattern (pattern's non‑None values must match).
    """
    if len(pattern.indices) != len(target.indices):
        return False
    for p, t in zip(pattern.indices, target.indices):
        if p is not None and p != t:
            return False
    return True


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    # Create a simple multiverse for a robot with 5 levels
    mv = Multiverse(name="TestRobot", max_level=4)
    
    # Create multi‑indices
    motor_left = MultiIndex(("motor_left", None, None, None, None))
    motor_right = MultiIndex(("motor_right", None, None, None, None))
    limb_arm = MultiIndex((None, "arm", None, None, None))
    planner = MultiIndex((None, None, "planner", None, None))
    ethics = MultiIndex((None, None, None, None, "ethics"))
    
    # Create simple subsystem classes (would be defined elsewhere)
    class SimpleMotor(Subsystem):
        def get_state(self): return super().get_state()
        def set_state(self, s): super().set_state(s)
        def step(self, dt, action=None): pass
    
    class SimpleLimb(Subsystem):
        def get_state(self): return super().get_state()
        def set_state(self, s): super().set_state(s)
        def step(self, dt, action=None): pass
    
    # Add subsystems
    mv.add_subsystem(SimpleMotor(motor_left, EuclideanSpace(2)))
    mv.add_subsystem(SimpleMotor(motor_right, EuclideanSpace(2)), parent=limb_arm)
    mv.add_subsystem(SimpleLimb(limb_arm, TensorProductSpace([EuclideanSpace(2), EuclideanSpace(2)])), parent=planner)
    mv.add_subsystem(SimpleLimb(planner, EuclideanSpace(6)), parent=ethics)
    mv.add_subsystem(SimpleLimb(ethics, EuclideanSpace(1)))
    
    # Set goals (placeholder)
    # mv.set_goal(0, some_goal)
    
    # Validate
    issues = mv.validate()
    print("Validation issues:", issues)
    
    # Explore hierarchy
    print("\nHierarchy:")
    for level in range(5):
        print(f"Level {level}: {mv.levels.get_at_level(level)}")
    
    print("\nChildren of arm:", mv.levels.get_children(limb_arm))
    print("Parents of motor_left:", mv.levels.get_parents(motor_left))
    
    # Test contains
    print(f"\nDoes {limb_arm} contain {motor_left}? {limb_arm.contains(motor_left)}")
    print(f"Does {limb_arm} contain {planner}? {limb_arm.contains(planner)}")
    
    # Test common ancestor
    common = motor_left.common_ancestor(motor_right)
    print(f"Common ancestor of {motor_left} and {motor_right}: {common}")
```