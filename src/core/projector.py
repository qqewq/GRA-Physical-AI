```python
"""
GRA Physical AI - Projector Module
==================================

This module defines the base classes and implementations for **projectors** – 
the mathematical representation of goals in the GRA framework.

A projector P is a linear operator satisfying:
    P² = P (idempotent)
    P† = P (self-adjoint)

Projectors map any state to the closest state in the goal‑satisfying subspace.
They are used to:
    - Define goals (as subspaces)
    - Compute foam (via off‑diagonal elements)
    - Project states during zeroing

All implementations support:
    - Differentiable operations (for gradient‑based zeroing)
    - GPU acceleration via PyTorch
    - Composition (products, sums, tensor products)
"""

import torch
import numpy as np
from typing import Optional, Union, List, Callable, Tuple, Dict, Any
from abc import ABC, abstractmethod
from enum import Enum
import warnings


# ======================================================================
# Base Projector Class
# ======================================================================

class Projector(ABC):
    """
    Abstract base class for all projectors in the GRA framework.
    
    A projector defines a subspace (the goal) and provides methods to:
        - Project any state onto that subspace
        - Compute the distance (loss) from a state to the subspace
        - Check if a state satisfies the goal
    
    Subclasses must implement:
        - __call__(self, state): apply the projector
        - loss(self, state): compute distance to subspace
    
    Optionally, subclasses may implement:
        - satisfies(self, state): check if state is in subspace
        - orthogonal(self, state): compute component orthogonal to subspace
        - gramian(self, states): compute Gram matrix of projected states
    """
    
    @abstractmethod
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        """
        Apply projector to state: return P|ψ⟩.
        
        For differentiable projectors, this should support gradient computation
        via PyTorch's autograd.
        
        Args:
            state: input state tensor of shape (dim,) or (batch, dim)
        
        Returns:
            projected state of same shape
        """
        pass
    
    @abstractmethod
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute loss = ‖(I - P)|ψ⟩‖² = distance² to goal subspace.
        
        This should be 0 iff state satisfies the goal.
        
        Args:
            state: input state tensor of shape (dim,) or (batch, dim)
        
        Returns:
            scalar loss (if state is 1D) or batch of losses (if state is 2D)
        """
        pass
    
    def satisfies(self, state: torch.Tensor, tolerance: float = 1e-6) -> Union[bool, torch.Tensor]:
        """
        Check if state satisfies the goal (is in the subspace).
        
        Args:
            state: input state
            tolerance: numerical tolerance
        
        Returns:
            bool or boolean tensor for batches
        """
        loss_val = self.loss(state)
        if loss_val.dim() == 0:
            return loss_val < tolerance
        else:
            return loss_val < tolerance
    
    def orthogonal(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute component orthogonal to goal subspace: (I - P)|ψ⟩.
        
        Args:
            state: input state
        
        Returns:
            orthogonal component
        """
        return state - self(state)
    
    def gramian(self, states: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute Gram matrix G_{ij} = ⟨ψ_i| P |ψ_j⟩.
        
        This is used in foam computation.
        
        Args:
            states: list of state tensors
        
        Returns:
            matrix of overlaps of shape (n, n)
        """
        n = len(states)
        device = states[0].device
        
        # Project all states
        proj_states = [self(s) for s in states]
        
        # Compute inner products
        gram = torch.zeros(n, n, device=device)
        for i in range(n):
            for j in range(n):
                gram[i, j] = torch.dot(states[i].flatten(), proj_states[j].flatten())
        
        return gram
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
    
    def __str__(self) -> str:
        return repr(self)


# ======================================================================
# Identity and Zero Projectors
# ======================================================================

class IdentityProjector(Projector):
    """
    Identity projector: P = I.
    
    The goal is the entire space – always satisfied.
    Useful for levels with no constraints, or as a building block.
    """
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return state
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            return torch.tensor(0.0, device=state.device)
        else:
            return torch.zeros(state.shape[0], device=state.device)
    
    def __repr__(self) -> str:
        return "IdentityProjector()"


class ZeroProjector(Projector):
    """
    Zero projector: P = 0.
    
    The goal is the empty subspace – never satisfied.
    Useful for debugging or as a placeholder.
    """
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(state)
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        # Distance to empty set is the norm of the state
        if state.dim() == 1:
            return torch.norm(state) ** 2
        else:
            return torch.norm(state, dim=1) ** 2
    
    def __repr__(self) -> str:
        return "ZeroProjector()"


# ======================================================================
# Linear Projectors (based on matrices)
# ======================================================================

class MatrixProjector(Projector):
    """
    Projector defined by an explicit projection matrix.
    
    For a subspace spanned by orthonormal basis vectors U (matrix of shape dim × k),
    the projector is P = U U^T (real) or U U^† (complex).
    
    This class stores the matrix explicitly.
    """
    
    def __init__(self, matrix: torch.Tensor):
        """
        Args:
            matrix: projection matrix of shape (dim, dim) satisfying P² = P, P† = P
        """
        self.matrix = matrix
        self._check_properties()
    
    def _check_properties(self, tolerance: float = 1e-4):
        """Verify that matrix is a valid projector."""
        # Check P² = P
        p2 = self.matrix @ self.matrix
        diff = torch.norm(p2 - self.matrix).item()
        if diff > tolerance:
            warnings.warn(f"Matrix may not be idempotent: ||P² - P|| = {diff}")
        
        # Check P† = P (Hermitian)
        if not torch.allclose(self.matrix, self.matrix.T.conj(), rtol=tolerance):
            warnings.warn("Matrix may not be Hermitian")
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        # Handle batched states
        if state.dim() == 1:
            return self.matrix @ state
        else:
            # state shape: (batch, dim)
            return state @ self.matrix.T
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            return torch.norm(state - self(state)) ** 2
        else:
            return torch.norm(state - self(state), dim=1) ** 2
    
    @classmethod
    def from_basis(cls, basis: torch.Tensor):
        """
        Create projector from orthonormal basis vectors.
        
        Args:
            basis: matrix of shape (dim, k) with orthonormal columns
        
        Returns:
            MatrixProjector P = basis @ basis.T
        """
        # Ensure orthonormal (optional check)
        k = basis.shape[1]
        for i in range(k):
            for j in range(k):
                if i == j:
                    assert abs(torch.dot(basis[:, i], basis[:, j]) - 1.0) < 1e-4
                else:
                    assert abs(torch.dot(basis[:, i], basis[:, j])) < 1e-4
        
        matrix = basis @ basis.T
        return cls(matrix)
    
    def __repr__(self) -> str:
        return f"MatrixProjector(shape={self.matrix.shape})"


# ======================================================================
# Threshold Projectors (for inequality constraints)
# ======================================================================

class ThresholdProjector(Projector):
    """
    Projector based on a threshold: projects to 1 if value >= threshold, else 0.
    
    This represents goals like "temperature ≤ limit" or "velocity ≥ minimum".
    The state is interpreted as a scalar or vector of values.
    
    For vectors, the projection is applied elementwise.
    """
    
    def __init__(self, threshold: float, 
                 compare: str = "ge",  # "ge", "le", "gt", "lt"
                 hard: bool = True,
                 temperature: float = 1.0):
        """
        Args:
            threshold: threshold value
            compare: comparison type ("ge", "le", "gt", "lt")
            hard: if True, use hard threshold (non‑differentiable); 
                  if False, use sigmoid approximation
            temperature: temperature for soft threshold (only if hard=False)
        """
        self.threshold = threshold
        self.compare = compare
        self.hard = hard
        self.temperature = temperature
        
        # Define comparison function
        if compare == "ge":
            self.compare_fn = lambda x: x >= threshold
        elif compare == "le":
            self.compare_fn = lambda x: x <= threshold
        elif compare == "gt":
            self.compare_fn = lambda x: x > threshold
        elif compare == "lt":
            self.compare_fn = lambda x: x < threshold
        else:
            raise ValueError(f"Unknown compare: {compare}")
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        if self.hard:
            # Hard threshold (non‑differentiable)
            return self.compare_fn(state).float()
        else:
            # Soft threshold (differentiable)
            if self.compare in ["ge", "gt"]:
                # x >= threshold -> sigmoid((x - threshold)/temperature)
                return torch.sigmoid((state - self.threshold) / self.temperature)
            else:
                # x <= threshold -> sigmoid((threshold - x)/temperature)
                return torch.sigmoid((self.threshold - state) / self.temperature)
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        """
        Loss = squared distance to threshold.
        For ge/gt: loss = max(0, threshold - x)²
        For le/lt: loss = max(0, x - threshold)²
        """
        if self.compare in ["ge", "gt"]:
            # Want x >= threshold
            violation = torch.relu(self.threshold - state)
        else:
            # Want x <= threshold
            violation = torch.relu(state - self.threshold)
        
        if state.dim() == 1:
            return torch.sum(violation ** 2)
        else:
            return torch.sum(violation ** 2, dim=1)
    
    def __repr__(self) -> str:
        mode = "hard" if self.hard else f"soft(T={self.temperature})"
        return f"ThresholdProjector({self.threshold}, {self.compare}, {mode})"


# ======================================================================
# Range Projector (interval constraints)
# ======================================================================

class RangeProjector(Projector):
    """
    Projector for interval constraints: low ≤ x ≤ high.
    
    Projects values into the interval [low, high].
    """
    
    def __init__(self, low: float, high: float, hard: bool = True, temperature: float = 1.0):
        """
        Args:
            low: lower bound
            high: upper bound
            hard: if True, use hard clipping; if False, use soft plus penalty
            temperature: temperature for soft projections
        """
        assert low <= high
        self.low = low
        self.high = high
        self.hard = hard
        self.temperature = temperature
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        if self.hard:
            # Hard clipping
            return torch.clamp(state, self.low, self.high)
        else:
            # Soft projection: sigmoid to [low, high]
            # Map from ℝ to [low, high] via sigmoid
            normalized = torch.sigmoid(state / self.temperature)
            return self.low + (self.high - self.low) * normalized
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        """
        Loss = distance outside interval squared.
        """
        lower_violation = torch.relu(self.low - state)
        upper_violation = torch.relu(state - self.high)
        violation = lower_violation + upper_violation
        
        if state.dim() == 1:
            return torch.sum(violation ** 2)
        else:
            return torch.sum(violation ** 2, dim=1)
    
    def __repr__(self) -> str:
        mode = "hard" if self.hard else f"soft(T={self.temperature})"
        return f"RangeProjector([{self.low}, {self.high}], {mode})"


# ======================================================================
# Norm Projector (for spherical constraints)
# ======================================================================

class NormProjector(Projector):
    """
    Projector for norm constraints: ‖x‖ ≤ max_norm.
    
    Projects onto the ball of radius max_norm.
    """
    
    def __init__(self, max_norm: float, p: int = 2):
        """
        Args:
            max_norm: maximum allowed norm
            p: norm type (1, 2, inf, etc.)
        """
        self.max_norm = max_norm
        self.p = p
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        # Handle batched states
        if state.dim() == 1:
            norm = torch.norm(state, p=self.p)
            if norm <= self.max_norm:
                return state
            else:
                return state * (self.max_norm / norm)
        else:
            # Batch case
            norms = torch.norm(state, p=self.p, dim=1, keepdim=True)
            scale = torch.where(norms <= self.max_norm, 
                                torch.ones_like(norms),
                                self.max_norm / norms)
            return state * scale
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        """
        Loss = max(0, ‖x‖ - max_norm)²
        """
        if state.dim() == 1:
            norm = torch.norm(state, p=self.p)
            violation = torch.relu(norm - self.max_norm)
            return violation ** 2
        else:
            norms = torch.norm(state, p=self.p, dim=1)
            violation = torch.relu(norms - self.max_norm)
            return violation ** 2
    
    def __repr__(self) -> str:
        return f"NormProjector(max_norm={self.max_norm}, p={self.p})"


# ======================================================================
# Composite Projectors
# ======================================================================

class CompositeProjector(Projector):
    """
    Base class for projectors composed of multiple sub‑projectors.
    """
    
    def __init__(self, projectors: List[Projector]):
        self.projectors = projectors
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        # To be overridden by subclasses
        raise NotImplementedError
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        # Sum of losses (by default)
        total_loss = 0.0
        for p in self.projectors:
            total_loss = total_loss + p.loss(state)
        return total_loss
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.projectors})"


class ProductProjector(CompositeProjector):
    """
    Product of projectors: P = P₁ P₂ ... Pₙ.
    
    Assumes projectors commute (otherwise order matters).
    """
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        result = state
        for p in self.projectors:
            result = p(result)
        return result


class SumProjector(CompositeProjector):
    """
    Weighted sum of projectors: P = Σ w_i P_i.
    
    Note: This is generally not a projector (may not be idempotent).
    Use only for approximations.
    """
    
    def __init__(self, projectors: List[Projector], weights: Optional[List[float]] = None):
        super().__init__(projectors)
        if weights is None:
            self.weights = [1.0] * len(projectors)
        else:
            assert len(weights) == len(projectors)
            self.weights = weights
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(state)
        for w, p in zip(self.weights, self.projectors):
            result = result + w * p(state)
        return result
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        # Weighted sum of losses
        total_loss = 0.0
        for w, p in zip(self.weights, self.projectors):
            total_loss = total_loss + w * p.loss(state)
        return total_loss


class IntersectionProjector(CompositeProjector):
    """
    Projector onto the intersection of subspaces.
    
    For commuting projectors, the intersection projector is the product.
    For non‑commuting, this is more complex – we use alternating projections.
    """
    
    def __init__(self, projectors: List[Projector], max_iter: int = 100):
        super().__init__(projectors)
        self.max_iter = max_iter
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        # Von Neumann alternating projections
        current = state
        for _ in range(self.max_iter):
            new_current = current
            for p in self.projectors:
                new_current = p(new_current)
            if torch.allclose(current, new_current):
                break
            current = new_current
        return current
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        # Distance to intersection = norm(state - P_intersection(state))
        proj = self(state)
        return torch.norm(state - proj) ** 2


# ======================================================================
# Learned Projectors (using neural networks)
# ======================================================================

class LearnedProjector(Projector):
    """
    Projector learned from data using a neural network.
    
    The network should map any state to the closest state in the goal subspace.
    This is useful for complex, non‑linear goals (e.g., "natural motion").
    """
    
    def __init__(self, network: torch.nn.Module, loss_fn: Optional[Callable] = None):
        """
        Args:
            network: neural net that maps input state → projected state
            loss_fn: optional custom loss function (default: MSE between input and output)
        """
        self.network = network
        self.loss_fn = loss_fn or (lambda x, x_proj: torch.mean((x - x_proj) ** 2))
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        proj = self(state)
        return self.loss_fn(state, proj)
    
    def __repr__(self) -> str:
        return f"LearnedProjector(net={self.network.__class__.__name__})"


# ======================================================================
# Differentiable Projector Wrapper
# ======================================================================

class DifferentiableProjector(Projector):
    """
    Wrapper to make any projector differentiable via straight‑through estimator.
    
    For non‑differentiable operations (e.g., hard threshold), this uses the
    identity for backward pass (straight‑through estimator).
    """
    
    def __init__(self, base_projector: Projector):
        self.base = base_projector
    
    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        # Forward pass uses base projector
        proj = self.base(state)
        
        # For backward pass, use identity (straight‑through)
        if state.requires_grad:
            proj = state + (proj - state).detach()
        
        return proj
    
    def loss(self, state: torch.Tensor) -> torch.Tensor:
        # Loss may still be non‑differentiable – wrap it
        loss_val = self.base.loss(state)
        if state.requires_grad and not loss_val.requires_grad:
            # Create a differentiable copy
            loss_val = loss_val + 0.0 * state.sum()
        return loss_val
    
    def __repr__(self) -> str:
        return f"DifferentiableProjector({self.base})"


# ======================================================================
# Utility Functions
# ======================================================================

def is_projector(matrix: torch.Tensor, tolerance: float = 1e-4) -> bool:
    """Check if a matrix is a valid projector."""
    # Check P² = P
    p2 = matrix @ matrix
    if torch.norm(p2 - matrix) > tolerance:
        return False
    
    # Check P† = P
    if not torch.allclose(matrix, matrix.T.conj(), rtol=tolerance):
        return False
    
    return True


def projector_from_subspace(basis: torch.Tensor) -> MatrixProjector:
    """Create projector from basis vectors (orthonormal)."""
    return MatrixProjector.from_basis(basis)


def projector_to_subspace(subspace: torch.Tensor) -> MatrixProjector:
    """Alias for from_basis."""
    return projector_from_subspace(subspace)


def random_projector(dim: int, rank: Optional[int] = None) -> MatrixProjector:
    """
    Generate a random orthogonal projector of given rank.
    
    Args:
        dim: dimension of space
        rank: rank of projector (default: dim//2)
    
    Returns:
        random projector
    """
    if rank is None:
        rank = dim // 2
    
    # Generate random orthonormal basis
    Q, _ = torch.linalg.qr(torch.randn(dim, rank))
    
    return MatrixProjector.from_basis(Q)


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    # Test basic projectors
    print("=== Testing Basic Projectors ===\n")
    
    # Identity
    I = IdentityProjector()
    x = torch.randn(5)
    print(f"Identity: {I(x)}")
    print(f"Loss: {I.loss(x)}")
    
    # Threshold
    T = ThresholdProjector(0.5, compare="ge", hard=False)
    x = torch.tensor([0.2, 0.8, 0.5, 1.0])
    print(f"\nThreshold (soft): {T(x)}")
    print(f"Loss: {T.loss(x)}")
    
    # Range
    R = RangeProjector(0.0, 1.0)
    x = torch.tensor([-0.5, 0.5, 1.5])
    print(f"\nRange: {R(x)}")
    print(f"Loss: {R.loss(x)}")
    
    # Norm
    N = NormProjector(1.0)
    x = torch.tensor([0.5, 0.5, 0.5])  # norm ≈ 0.87
    print(f"\nNorm (within): {N(x)}")
    x = torch.tensor([1.0, 1.0, 1.0])  # norm ≈ 1.73
    print(f"Norm (exceed): {N(x)}")
    print(f"Loss: {N.loss(x)}")
    
    # Matrix projector from basis
    basis = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).T  # 2D basis in 2D space
    M = MatrixProjector.from_basis(basis)
    x = torch.randn(2)
    print(f"\nMatrix projector: {M(x)}")
    print(f"Is projector? {is_projector(M.matrix)}")
    
    # Composite
    P1 = ThresholdProjector(0.0, compare="ge")
    P2 = ThresholdProjector(1.0, compare="le")
    intersection = IntersectionProjector([P1, P2])
    x = torch.tensor([-0.5, 0.5, 1.5])
    print(f"\nIntersection of [0,1]: {intersection(x)}")
    print(f"Loss: {intersection.loss(x)}")
    
    # Differentiable wrapper
    hard_T = ThresholdProjector(0.5, compare="ge", hard=True)
    soft_T = DifferentiableProjector(hard_T)
    x = torch.tensor([0.2, 0.8], requires_grad=True)
    y = soft_T(x)
    loss = y.sum()
    loss.backward()
    print(f"\nDifferentiable hard threshold:")
    print(f"  Forward: {y}")
    print(f"  Gradient: {x.grad}")
    
    print("\nAll tests passed!")
```