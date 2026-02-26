```python
"""
GRA Physical AI - Foam Computation Module
=========================================

This module provides functions and classes for computing **foam** – 
the measure of inconsistency between subsystems at the same level.

Foam at level l is defined as:
    Φ⁽ˡ⁾(Ψ⁽ˡ⁾, G_l) = Σ_{a≠b, dim(a)=dim(b)=l} |⟨Ψ^(a)| P_{G_l} |Ψ^(b)⟩|²

This module implements:
    - Basic foam computation for a collection of states
    - Batched and GPU‑accelerated versions
    - Approximate foam via sampling
    - Foam gradients for zeroing
    - Foam visualization tools
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable, Any
from dataclasses import dataclass, field
import warnings
import math


# ======================================================================
# Core Foam Computation
# ======================================================================

def compute_foam(states: Union[List[torch.Tensor], Dict[Any, torch.Tensor]],
                 projector: Callable[[torch.Tensor], torch.Tensor],
                 inner_product: Optional[Callable] = None) -> torch.Tensor:
    """
    Compute foam Φ = Σ_{i≠j} |⟨ψ_i| P |ψ_j⟩|².
    
    Args:
        states: list of state tensors, or dict mapping keys to states
        projector: function that applies P to a state
        inner_product: function computing ⟨ψ|φ⟩ (default: dot product)
    
    Returns:
        scalar foam tensor
    """
    # Convert dict to list if needed
    if isinstance(states, dict):
        keys = list(states.keys())
        state_list = [states[k] for k in keys]
    else:
        state_list = states
        keys = list(range(len(state_list)))
    
    n = len(state_list)
    if n < 2:
        return torch.tensor(0.0, device=state_list[0].device if state_list else None)
    
    # Default inner product: dot product after flattening
    if inner_product is None:
        inner_product = lambda x, y: torch.dot(x.flatten(), y.flatten())
    
    device = state_list[0].device
    foam = torch.tensor(0.0, device=device)
    
    # Project all states (can be done in parallel)
    proj_states = [projector(s) for s in state_list]
    
    # Compute all pairs
    for i in range(n):
        for j in range(i+1, n):
            overlap = inner_product(state_list[i], proj_states[j])
            foam = foam + torch.abs(overlap) ** 2
    
    return foam


def compute_foam_batched(states: torch.Tensor,
                         projector_matrix: Optional[torch.Tensor] = None,
                         projector_fn: Optional[Callable] = None) -> torch.Tensor:
    """
    Batched foam computation using matrix operations.
    
    Args:
        states: tensor of shape (n, d) where n = number of subsystems, d = state dim
        projector_matrix: matrix representation of P, shape (d, d)
        projector_fn: alternative function to apply P (if matrix not available)
    
    Returns:
        scalar foam
    """
    n, d = states.shape
    
    if n < 2:
        return torch.tensor(0.0, device=states.device)
    
    if projector_matrix is not None:
        # Project all states: (n, d) @ (d, d) -> (n, d)
        proj = states @ projector_matrix.T
    elif projector_fn is not None:
        # Apply function to each state (slower)
        proj = torch.stack([projector_fn(s) for s in states])
    else:
        raise ValueError("Either projector_matrix or projector_fn must be provided")
    
    # Compute all inner products: (n, d) @ (n, d).T -> (n, n)
    overlaps = states @ proj.T
    
    # Zero out diagonal and take squared magnitude
    mask = 1 - torch.eye(n, device=states.device)
    off_diag = overlaps * mask
    foam = torch.sum(off_diag ** 2) / 2  # divide by 2 because we count each pair twice
    
    return foam


# ======================================================================
# Gradient of Foam
# ======================================================================

def foam_gradient(states: Union[List[torch.Tensor], torch.Tensor],
                  projector: Callable[[torch.Tensor], torch.Tensor],
                  index: Optional[int] = None) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Compute gradient of foam w.r.t. each state.
    
    For a single state ψ_i, the gradient is:
        ∂Φ/∂ψ_i = 2 Σ_{j≠i} ⟨ψ_i|P|ψ_j⟩ · P|ψ_j⟩
    
    Args:
        states: list of states or batched tensor (n, d)
        projector: function that applies P to a state
        index: if provided, compute gradient only for that state
    
    Returns:
        if index is None: list of gradients for all states
        else: gradient for the specified state
    """
    if isinstance(states, torch.Tensor) and states.dim() == 2:
        # Batched case
        n, d = states.shape
        proj = torch.stack([projector(s) for s in states])
        
        # Compute overlaps matrix
        overlaps = torch.zeros(n, n, device=states.device)
        for i in range(n):
            for j in range(n):
                if i != j:
                    overlaps[i, j] = torch.dot(states[i].flatten(), proj[j].flatten())
        
        if index is not None:
            # Gradient for one state
            grad = torch.zeros_like(states[index])
            for j in range(n):
                if j != index:
                    grad = grad + 2 * overlaps[index, j] * proj[j]
            return grad
        else:
            # Gradients for all states
            grads = []
            for i in range(n):
                grad = torch.zeros_like(states[i])
                for j in range(n):
                    if j != i:
                        grad = grad + 2 * overlaps[i, j] * proj[j]
                grads.append(grad)
            return grads
    
    else:
        # List case
        state_list = list(states) if isinstance(states, list) else states
        n = len(state_list)
        proj = [projector(s) for s in state_list]
        
        # Precompute overlaps
        overlaps = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    overlaps[i][j] = torch.dot(state_list[i].flatten(), proj[j].flatten())
        
        if index is not None:
            grad = torch.zeros_like(state_list[index])
            for j in range(n):
                if j != index:
                    grad = grad + 2 * overlaps[index][j] * proj[j]
            return grad
        else:
            grads = []
            for i in range(n):
                grad = torch.zeros_like(state_list[i])
                for j in range(n):
                    if j != i:
                        grad = grad + 2 * overlaps[i][j] * proj[j]
                grads.append(grad)
            return grads


def foam_gradient_batched(states: torch.Tensor,
                          projector_matrix: torch.Tensor) -> torch.Tensor:
    """
    Batched computation of foam gradients.
    
    Args:
        states: tensor of shape (n, d)
        projector_matrix: tensor of shape (d, d)
    
    Returns:
        gradients tensor of shape (n, d)
    """
    n, d = states.shape
    
    # Project all states
    proj = states @ projector_matrix.T  # (n, d)
    
    # Compute overlaps matrix
    overlaps = states @ proj.T  # (n, n)
    
    # Zero out diagonal
    mask = 1 - torch.eye(n, device=states.device)
    overlaps = overlaps * mask
    
    # Gradients: for each i, sum over j of 2 * overlaps[i,j] * proj[j]
    # This is matrix multiplication: (overlaps) @ proj
    grads = 2 * overlaps @ proj  # (n, d)
    
    return grads


# ======================================================================
# Approximate Foam (for large N)
# ======================================================================

def compute_foam_sampled(states: Union[List[torch.Tensor], Dict[Any, torch.Tensor]],
                         projector: Callable[[torch.Tensor], torch.Tensor],
                         num_samples: int = 100,
                         inner_product: Optional[Callable] = None) -> Tuple[torch.Tensor, float]:
    """
    Approximate foam by random sampling of pairs.
    
    Args:
        states: list or dict of states
        projector: projector function
        num_samples: number of random pairs to sample
        inner_product: inner product function
    
    Returns:
        (estimated foam, standard error)
    """
    if isinstance(states, dict):
        keys = list(states.keys())
        state_list = [states[k] for k in keys]
    else:
        state_list = list(states)
        keys = list(range(len(state_list)))
    
    n = len(state_list)
    if n < 2:
        return torch.tensor(0.0), 0.0
    
    if inner_product is None:
        inner_product = lambda x, y: torch.dot(x.flatten(), y.flatten())
    
    device = state_list[0].device
    proj_states = [projector(s) for s in state_list]
    
    samples = []
    for _ in range(num_samples):
        # Sample two distinct indices uniformly
        i = torch.randint(0, n, (1,)).item()
        j = torch.randint(0, n-1, (1,)).item()
        if j >= i:
            j += 1
        
        overlap = inner_product(state_list[i], proj_states[j])
        samples.append((overlap ** 2).item())
    
    samples = torch.tensor(samples, device=device)
    mean = samples.mean()
    std_err = samples.std() / math.sqrt(num_samples)
    
    # Scale by number of pairs
    total_pairs = n * (n - 1) / 2
    estimated_foam = mean * total_pairs
    
    return estimated_foam, std_err.item()


def compute_foam_blocked(states: Union[List[torch.Tensor], torch.Tensor],
                         projector: Callable[[torch.Tensor], torch.Tensor],
                         block_size: int = 100) -> torch.Tensor:
    """
    Compute foam using blocking to manage memory.
    Useful when n is too large for full matrix.
    
    Args:
        states: list of states or batched tensor
        projector: projector function
        block_size: size of blocks for partial computation
    
    Returns:
        foam scalar
    """
    if isinstance(states, torch.Tensor) and states.dim() == 2:
        n, d = states.shape
        state_list = [states[i] for i in range(n)]
    else:
        state_list = list(states)
        n = len(state_list)
    
    if n < 2:
        return torch.tensor(0.0, device=state_list[0].device)
    
    device = state_list[0].device
    proj_list = [projector(s) for s in state_list]
    
    foam = torch.tensor(0.0, device=device)
    
    # Process in blocks to avoid O(n²) memory
    for i_start in range(0, n, block_size):
        i_end = min(i_start + block_size, n)
        
        for j_start in range(i_start, n, block_size):
            j_end = min(j_start + block_size, n)
            
            for i in range(i_start, i_end):
                for j in range(max(j_start, i+1), j_end):
                    if i != j:
                        overlap = torch.dot(state_list[i].flatten(), proj_list[j].flatten())
                        foam = foam + overlap ** 2
    
    return foam


# ======================================================================
# Foam Analysis Tools
# ======================================================================

@dataclass
class FoamReport:
    """Container for foam analysis results."""
    
    level: int
    total_foam: float
    mean_pairwise: float
    max_pairwise: float
    min_pairwise: float
    std_pairwise: float
    num_pairs: int
    contributions: Optional[Dict[Tuple[Any, Any], float]] = None
    
    def __str__(self) -> str:
        s = f"Level {self.level} Foam Report:\n"
        s += f"  Total foam: {self.total_foam:.6f}\n"
        s += f"  Number of pairs: {self.num_pairs}\n"
        s += f"  Mean pairwise: {self.mean_pairwise:.6f}\n"
        s += f"  Max pairwise: {self.max_pairwise:.6f}\n"
        s += f"  Min pairwise: {self.min_pairwise:.6f}\n"
        s += f"  Std deviation: {self.std_pairwise:.6f}\n"
        return s


def analyze_foam(states: Union[List[torch.Tensor], Dict[Any, torch.Tensor]],
                 projector: Callable[[torch.Tensor], torch.Tensor],
                 level: int = 0,
                 inner_product: Optional[Callable] = None,
                 return_contributions: bool = False) -> FoamReport:
    """
    Detailed analysis of foam contributions.
    
    Returns:
        FoamReport with statistics and optional pair contributions
    """
    if isinstance(states, dict):
        keys = list(states.keys())
        state_list = [states[k] for k in keys]
        id_map = {i: keys[i] for i in range(len(keys))}
    else:
        state_list = list(states)
        keys = list(range(len(state_list)))
        id_map = {i: i for i in range(len(keys))}
    
    n = len(state_list)
    if n < 2:
        return FoamReport(level, 0.0, 0.0, 0.0, 0.0, 0.0, 0)
    
    if inner_product is None:
        inner_product = lambda x, y: torch.dot(x.flatten(), y.flatten())
    
    device = state_list[0].device
    proj_states = [projector(s) for s in state_list]
    
    pair_values = []
    contributions = {} if return_contributions else None
    
    for i in range(n):
        for j in range(i+1, n):
            overlap = inner_product(state_list[i], proj_states[j])
            val = (overlap ** 2).item()
            pair_values.append(val)
            
            if return_contributions:
                contributions[(id_map[i], id_map[j])] = val
    
    pair_tensor = torch.tensor(pair_values, device=device)
    
    return FoamReport(
        level=level,
        total_foam=pair_tensor.sum().item(),
        mean_pairwise=pair_tensor.mean().item(),
        max_pairwise=pair_tensor.max().item(),
        min_pairwise=pair_tensor.min().item(),
        std_pairwise=pair_tensor.std().item(),
        num_pairs=len(pair_values),
        contributions=contributions
    )


def foam_heatmap(states: Union[List[torch.Tensor], torch.Tensor],
                 projector: Callable[[torch.Tensor], torch.Tensor],
                 labels: Optional[List[str]] = None) -> np.ndarray:
    """
    Compute matrix of pairwise contributions for visualization.
    
    Returns:
        numpy array of shape (n, n) with values |⟨ψ_i|P|ψ_j⟩|²
    """
    if isinstance(states, torch.Tensor) and states.dim() == 2:
        n = states.shape[0]
        state_list = [states[i] for i in range(n)]
    else:
        state_list = list(states)
        n = len(state_list)
    
    proj_list = [projector(s) for s in state_list]
    
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                overlap = torch.dot(state_list[i].flatten(), proj_list[j].flatten())
                matrix[i, j] = (overlap ** 2).item()
    
    return matrix


# ======================================================================
# Level‑Specific Foam (integrates with multiverse)
# ======================================================================

class LevelFoamComputer:
    """
    Computes foam for a specific level in a GRA multiverse.
    
    Caches projector applications for efficiency.
    """
    
    def __init__(self, level: int, projector: Callable[[torch.Tensor], torch.Tensor]):
        self.level = level
        self.projector = projector
        self.cache: Dict[int, torch.Tensor] = {}  # index -> projected state
        
    def compute(self, states: Dict[Any, torch.Tensor], 
                use_cache: bool = True) -> torch.Tensor:
        """
        Compute foam for given states.
        
        Args:
            states: mapping from identifier to state tensor
            use_cache: whether to reuse cached projected states
        
        Returns:
            foam scalar
        """
        ids = list(states.keys())
        n = len(ids)
        
        if n < 2:
            return torch.tensor(0.0, device=next(iter(states.values())).device)
        
        # Project states (with caching)
        proj = {}
        for i, idx in enumerate(ids):
            if use_cache and i in self.cache:
                proj[i] = self.cache[i]
            else:
                proj[i] = self.projector(states[idx])
                if use_cache:
                    self.cache[i] = proj[i]
        
        foam = torch.tensor(0.0, device=proj[0].device)
        for i in range(n):
            for j in range(i+1, n):
                overlap = torch.dot(states[ids[i]].flatten(), proj[j].flatten())
                foam = foam + overlap ** 2
        
        return foam
    
    def clear_cache(self):
        """Clear cached projected states."""
        self.cache.clear()


# ======================================================================
# Multi‑Level Foam (for entire hierarchy)
# ======================================================================

class MultiLevelFoam:
    """
    Manages foam computation for all levels in a hierarchy.
    
    This integrates with the Multiverse class.
    """
    
    def __init__(self, 
                 get_states_fn: Callable[[int], Dict[Any, torch.Tensor]],
                 get_projector_fn: Callable[[int], Optional[Callable]]):
        """
        Args:
            get_states_fn: function that returns states for a given level
            get_projector_fn: function that returns projector for a given level
        """
        self.get_states = get_states_fn
        self.get_projector = get_projector_fn
        self.level_computers: Dict[int, LevelFoamComputer] = {}
        
    def compute_level(self, level: int, use_cache: bool = True) -> torch.Tensor:
        """Compute foam for a single level."""
        projector = self.get_projector(level)
        if projector is None:
            return torch.tensor(0.0)
        
        if level not in self.level_computers:
            self.level_computers[level] = LevelFoamComputer(level, projector)
        
        states = self.get_states(level)
        return self.level_computers[level].compute(states, use_cache)
    
    def compute_all(self, levels: Optional[List[int]] = None, 
                    use_cache: bool = True) -> Dict[int, torch.Tensor]:
        """Compute foam for all levels."""
        if levels is None:
            # Need to know max level – will be set externally
            return {}
        
        return {l: self.compute_level(l, use_cache) for l in levels}
    
    def clear_cache(self, level: Optional[int] = None):
        """Clear cache for one level or all."""
        if level is not None:
            if level in self.level_computers:
                self.level_computers[level].clear_cache()
        else:
            for comp in self.level_computers.values():
                comp.clear_cache()


# ======================================================================
# Utility Functions
# ======================================================================

def normalize_foam(foam: torch.Tensor, num_pairs: int) -> torch.Tensor:
    """
    Normalize foam by number of pairs to get average pairwise inconsistency.
    """
    if num_pairs > 0:
        return foam / num_pairs
    return foam


def foam_to_probability(foam: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Convert foam to a probability (e.g., for sampling).
    Lower foam → higher probability.
    """
    return torch.exp(-foam / temperature)


def is_zero_foam(foam: torch.Tensor, tolerance: float = 1e-6) -> bool:
    """Check if foam is effectively zero."""
    return torch.abs(foam) < tolerance


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing Foam Computation ===\n")
    
    # Create some test states
    n_states = 5
    dim = 10
    states = [torch.randn(dim) for _ in range(n_states)]
    
    # Simple projector: identity
    identity_proj = lambda x: x
    
    # Compute foam
    foam = compute_foam(states, identity_proj)
    print(f"Foam with identity projector: {foam.item():.6f}")
    
    # Batched version
    states_batch = torch.stack(states)
    # Create random projector matrix
    P = torch.randn(dim, dim)
    P = P @ P.T  # make symmetric (but not necessarily idempotent)
    foam_batch = compute_foam_batched(states_batch, projector_matrix=P)
    print(f"Batched foam: {foam_batch.item():.6f}")
    
    # Gradient
    grads = foam_gradient(states, identity_proj)
    print(f"\nGradient for state 0 shape: {grads[0].shape}")
    
    # Batched gradient
    grads_batch = foam_gradient_batched(states_batch, torch.eye(dim))
    print(f"Batched gradients shape: {grads_batch.shape}")
    
    # Sampled approximation
    foam_est, stderr = compute_foam_sampled(states, identity_proj, num_samples=20)
    print(f"\nSampled foam: {foam_est.item():.6f} ± {stderr:.6f}")
    
    # Detailed analysis
    report = analyze_foam(states, identity_proj, return_contributions=True)
    print(f"\n{report}")
    
    # Heatmap
    heatmap = foam_heatmap(states, identity_proj)
    print(f"Heatmap shape: {heatmap.shape}")
    
    print("\nAll tests passed!")
```