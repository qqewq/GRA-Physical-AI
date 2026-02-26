```python
#!/usr/bin/env python3
"""
GRA Physical AI - Core Module Tests
====================================

This module contains unit tests for the core GRA components:
    - MultiIndex
    - HilbertSpace
    - Projector
    - Goal
    - Subsystem
    - Functional
    - ZeroingAlgorithm
    - Foam computation

Run with: pytest test_core.py -v
"""

import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.multiverse import MultiIndex, EuclideanSpace, TensorProductSpace, Multiverse
from core.projector import (
    IdentityProjector, ZeroProjector, ThresholdProjector, RangeProjector,
    NormProjector, MatrixProjector, IntersectionProjector, DifferentiableProjector
)
from core.goal import Goal
from core.subsystem import SimpleSubsystem
from core.foam import compute_foam, foam_gradient, compute_foam_batched, analyze_foam
from core.nullification import ZeroingAlgorithm, recursive_zero


# ======================================================================
# MultiIndex Tests
# ======================================================================

class TestMultiIndex:
    """Tests for MultiIndex class."""
    
    def test_creation(self):
        """Test MultiIndex creation."""
        idx = MultiIndex(("robot", "left_motor", None, None))
        assert idx.level == 3
        assert idx.indices[0] == "robot"
        assert idx.indices[1] == "left_motor"
        assert idx.indices[2] is None
        assert idx.indices[3] is None
    
    def test_contains(self):
        """Test contains method."""
        parent = MultiIndex(("robot", None, None))
        child = MultiIndex(("robot", "left_motor", None))
        unrelated = MultiIndex(("other", None, None))
        
        assert parent.contains(child)
        assert not child.contains(parent)
        assert not parent.contains(unrelated)
    
    def test_equality(self):
        """Test equality."""
        idx1 = MultiIndex(("a", "b", None))
        idx2 = MultiIndex(("a", "b", None))
        idx3 = MultiIndex(("a", "c", None))
        
        assert idx1 == idx2
        assert idx1 != idx3
    
    def test_string_representation(self):
        """Test string representation."""
        idx = MultiIndex(("robot", "motor", None))
        assert str(idx) == "L0:robot/L1:motor"
        
        idx = MultiIndex((None, None, None))
        assert str(idx) == "*"


# ======================================================================
# HilbertSpace Tests
# ======================================================================

class TestHilbertSpace:
    """Tests for Hilbert spaces."""
    
    def test_euclidean_space(self):
        """Test Euclidean space."""
        space = EuclideanSpace(5)
        assert space.dimension() == 5
        
        v1 = space.random_state()
        v2 = space.random_state()
        
        inner = space.inner_product(v1, v2)
        assert inner.shape == torch.Size([])
        
        norm = space.norm(v1)
        assert norm >= 0
    
    def test_tensor_product_space(self):
        """Test tensor product space."""
        space1 = EuclideanSpace(2)
        space2 = EuclideanSpace(3)
        product = TensorProductSpace([space1, space2])
        
        assert product.dimension() == 6
        
        state = product.random_state()
        assert state.shape == torch.Size([6])


# ======================================================================
# Projector Tests
# ======================================================================

class TestProjector:
    """Tests for projectors."""
    
    def test_identity_projector(self):
        """Test identity projector."""
        proj = IdentityProjector()
        x = torch.randn(5)
        
        assert torch.allclose(proj(x), x)
        assert proj.loss(x) == 0
    
    def test_zero_projector(self):
        """Test zero projector."""
        proj = ZeroProjector()
        x = torch.randn(5)
        
        assert torch.allclose(proj(x), torch.zeros_like(x))
        assert proj.loss(x) > 0
    
    def test_threshold_projector_hard(self):
        """Test hard threshold projector."""
        proj = ThresholdProjector(threshold=0.5, compare='ge', hard=True)
        x = torch.tensor([0.2, 0.8, 0.5])
        
        result = proj(x)
        expected = torch.tensor([0.0, 1.0, 1.0])
        assert torch.allclose(result, expected)
    
    def test_threshold_projector_soft(self):
        """Test soft threshold projector."""
        proj = ThresholdProjector(threshold=0.5, compare='ge', hard=False, temperature=0.1)
        x = torch.tensor([0.2, 0.8, 0.5])
        
        result = proj(x)
        # Should be between 0 and 1
        assert torch.all(result >= 0) and torch.all(result <= 1)
        
        # Loss should be differentiable
        loss = proj.loss(x)
        assert loss.requires_grad == x.requires_grad
    
    def test_range_projector(self):
        """Test range projector."""
        proj = RangeProjector(low=-1.0, high=1.0, hard=True)
        x = torch.tensor([-2.0, 0.5, 2.0])
        
        result = proj(x)
        expected = torch.tensor([-1.0, 0.5, 1.0])
        assert torch.allclose(result, expected)
        
        loss = proj.loss(x)
        expected_loss = torch.tensor(1.0**2 + 1.0**2)  # (-2 to -1) + (2 to 1)
        assert torch.allclose(loss, expected_loss)
    
    def test_norm_projector(self):
        """Test norm projector."""
        proj = NormProjector(max_norm=1.0)
        
        # Within norm
        x1 = torch.tensor([0.5, 0.5])
        assert torch.allclose(proj(x1), x1)
        assert proj.loss(x1) == 0
        
        # Exceeds norm
        x2 = torch.tensor([1.0, 1.0])
        result = proj(x2)
        assert torch.norm(result) <= 1.0
        assert proj.loss(x2) > 0
    
    def test_matrix_projector(self):
        """Test matrix projector."""
        # Create a simple 2D subspace
        basis = torch.tensor([[1.0, 0.0]]).T
        proj = MatrixProjector.from_basis(basis)
        
        x = torch.tensor([1.0, 2.0])
        result = proj(x)
        
        # Should project onto x-axis
        expected = torch.tensor([1.0, 0.0])
        assert torch.allclose(result, expected)
    
    def test_intersection_projector(self):
        """Test intersection projector."""
        # Two interval constraints
        p1 = RangeProjector(low=-1.0, high=1.0)
        p2 = RangeProjector(low=0.0, high=2.0)
        
        intersection = IntersectionProjector([p1, p2])
        
        x = torch.tensor([-0.5, 1.5])
        result = intersection(x)
        
        # Should project to intersection [0,1]
        expected = torch.tensor([0.0, 1.0])
        assert torch.allclose(result, expected, atol=1e-2)
    
    def test_differentiable_projector(self):
        """Test differentiable wrapper."""
        hard_proj = ThresholdProjector(threshold=0.5, compare='ge', hard=True)
        diff_proj = DifferentiableProjector(hard_proj)
        
        x = torch.tensor([0.2, 0.8], requires_grad=True)
        y = diff_proj(x)
        loss = y.sum()
        loss.backward()
        
        # Gradient should exist (straight-through)
        assert x.grad is not None
        assert torch.allclose(x.grad, torch.ones_like(x))


# ======================================================================
# Foam Computation Tests
# ======================================================================

class TestFoam:
    """Tests for foam computation."""
    
    def test_foam_identical_states(self):
        """Test foam with identical states."""
        states = [torch.ones(5) for _ in range(3)]
        proj = IdentityProjector()
        
        foam = compute_foam(states, proj)
        assert foam == 0  # All overlaps should be 1, but off-diagonal in eigenbasis
    
    def test_foam_orthogonal_states(self):
        """Test foam with orthogonal states."""
        states = [
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
            torch.tensor([0.0, 0.0, 1.0])
        ]
        proj = IdentityProjector()
        
        foam = compute_foam(states, proj)
        assert foam == 0  # Orthogonal -> no off-diagonal
    
    def test_foam_mixed_states(self):
        """Test foam with mixed states."""
        states = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.5, 0.5]),
        ]
        proj = IdentityProjector()
        
        foam = compute_foam(states, proj)
        expected = (1.0 * 0.5) ** 2  # dot product squared
        assert torch.allclose(foam, torch.tensor(expected))
    
    def test_foam_batched(self):
        """Test batched foam computation."""
        states = torch.tensor([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        proj_matrix = torch.eye(2)
        
        foam = compute_foam_batched(states, projector_matrix=proj_matrix)
        
        # Manual computation
        overlaps = states @ states.T
        mask = 1 - torch.eye(3)
        off_diag = overlaps * mask
        expected = (off_diag ** 2).sum() / 2
        
        assert torch.allclose(foam, expected)
    
    def test_foam_gradient(self):
        """Test foam gradient computation."""
        states = [
            torch.tensor([1.0, 0.0], requires_grad=True),
            torch.tensor([0.5, 0.5], requires_grad=True),
        ]
        proj = IdentityProjector()
        
        grads = foam_gradient(states, proj)
        
        assert len(grads) == 2
        assert grads[0].shape == states[0].shape
        
        # Check one gradient numerically
        foam = compute_foam(states, proj)
        foam.backward()
        
        assert torch.allclose(states[0].grad, grads[0], rtol=1e-4)
    
    def test_foam_analysis(self):
        """Test foam analysis."""
        states = [
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.5, 0.5]),
            torch.tensor([0.0, 1.0]),
        ]
        proj = IdentityProjector()
        
        report = analyze_foam(states, proj, return_contributions=True)
        
        assert report.num_pairs == 3
        assert report.total_foam > 0
        assert report.mean_pairwise > 0
        assert report.max_pairwise >= report.min_pairwise
        assert report.contributions is not None
        assert len(report.contributions) == 3


# ======================================================================
# Subsystem Tests
# ======================================================================

class TestSubsystem:
    """Tests for Subsystem class."""
    
    def test_simple_subsystem(self):
        """Test SimpleSubsystem."""
        idx = MultiIndex(("test", None))
        space = EuclideanSpace(5)
        
        sub = SimpleSubsystem(idx, space, None)
        
        state = sub.get_state()
        assert state.shape == torch.Size([5])
        assert torch.allclose(state, torch.zeros(5))
        
        new_state = torch.randn(5)
        sub.set_state(new_state)
        assert torch.allclose(sub.get_state(), new_state)


# ======================================================================
# Goal Tests
# ======================================================================

class TestGoal:
    """Tests for Goal class."""
    
    def test_goal_creation(self):
        """Test goal creation."""
        proj = IdentityProjector()
        goal = Goal("test_goal", proj)
        
        assert goal.name == "test_goal"
        assert goal.projector == proj
        
        x = torch.randn(5)
        assert torch.allclose(goal.project(x), x)
        assert goal.loss(x) == 0


# ======================================================================
# Multiverse Tests
# ======================================================================

class TestMultiverse:
    """Tests for Multiverse class."""
    
    def test_multiverse_creation(self):
        """Test multiverse creation."""
        mv = Multiverse(name="test", max_level=2)
        assert mv.name == "test"
        assert mv.max_level == 2
        assert len(mv.subsystems) == 0
    
    def test_add_subsystem(self):
        """Test adding subsystems."""
        mv = Multiverse(max_level=2)
        
        idx1 = MultiIndex(("robot", None, None))
        sub1 = SimpleSubsystem(idx1, EuclideanSpace(5), None)
        
        idx2 = MultiIndex(("robot", "motor", None))
        sub2 = SimpleSubsystem(idx2, EuclideanSpace(3), None)
        
        mv.add_subsystem(sub1)
        mv.add_subsystem(sub2, parent=idx1)
        
        assert len(mv.subsystems) == 2
        assert mv.get_subsystem(idx1) is sub1
        assert mv.get_subsystem(idx2) is sub2
    
    def test_foam_computation(self):
        """Test foam computation in multiverse."""
        mv = Multiverse(max_level=1)
        
        # Add two subsystems at level 0
        idx1 = MultiIndex(("sub1", None))
        sub1 = SimpleSubsystem(idx1, EuclideanSpace(2), None)
        sub1.set_state(torch.tensor([1.0, 0.0]))
        
        idx2 = MultiIndex(("sub2", None))
        sub2 = SimpleSubsystem(idx2, EuclideanSpace(2), None)
        sub2.set_state(torch.tensor([0.5, 0.5]))
        
        mv.add_subsystem(sub1)
        mv.add_subsystem(sub2)
        
        # Add goal
        goal = Goal("test", IdentityProjector())
        mv.set_goal(0, goal)
        
        foam = mv.compute_foam(0)
        expected = (1.0 * 0.5) ** 2  # dot product
        
        assert torch.allclose(foam, torch.tensor(expected))


# ======================================================================
# Zeroing Algorithm Tests
# ======================================================================

class TestZeroing:
    """Tests for zeroing algorithm."""
    
    def test_recursive_zero_base(self):
        """Test base case of recursive zero."""
        states = {
            MultiIndex(("sub", None)): torch.tensor([1.0, 0.0])
        }
        hierarchy = {}
        
        def get_children(x): return []
        def get_goal_projector(l): return IdentityProjector()
        def get_level_weight(l): return 1.0
        
        new_states = recursive_zero(
            level=0,
            states=states,
            hierarchy=hierarchy,
            get_children=get_children,
            get_goal_projector=get_goal_projector,
            get_level_weight=get_level_weight,
            max_iters=10
        )
        
        assert len(new_states) == 1
        assert torch.allclose(new_states[MultiIndex(("sub", None))], states[MultiIndex(("sub", None))])
    
    def test_zeroing_algorithm_creation(self):
        """Test zeroing algorithm creation."""
        algo = ZeroingAlgorithm(
            hierarchy={},
            get_children=lambda x: [],
            get_parents=lambda x: [],
            get_goal_projector=lambda x: None,
            get_level_weight=lambda x: 1.0,
            level_tolerances=[0.01],
            learning_rates=[0.01]
        )
        
        assert algo.max_level == 0
        assert algo.status.value == "stopped"


# ======================================================================
# Integration Tests
# ======================================================================

class TestIntegration:
    """Integration tests for multiple components."""
    
    def test_simple_two_level_system(self):
        """Test a simple two-level system."""
        # Create indices
        motor1_idx = MultiIndex(("motor1", None, None))
        motor2_idx = MultiIndex(("motor2", None, None))
        nav_idx = MultiIndex((None, "navigator", None))
        
        # Create spaces
        motor_space = EuclideanSpace(2)  # [cmd, actual]
        nav_space = EuclideanSpace(6)    # [x, y, theta, target_x, target_y, target_theta]
        
        # Create subsystems
        motor1 = SimpleSubsystem(motor1_idx, motor_space, None)
        motor2 = SimpleSubsystem(motor2_idx, motor_space, None)
        nav = SimpleSubsystem(nav_idx, nav_space, None)
        
        # Set initial states
        motor1.set_state(torch.tensor([0.5, 0.3]))
        motor2.set_state(torch.tensor([0.5, 0.4]))
        nav.set_state(torch.randn(6))
        
        # Create multiverse
        mv = Multiverse(max_level=2)
        mv.add_subsystem(motor1)
        mv.add_subsystem(motor2)
        mv.add_subsystem(nav)
        
        # Create goals
        class MotorGoal(Goal):
            def loss(self, state):
                return (state[1] - state[0]) ** 2
        
        class NavGoal(Goal):
            def loss(self, state):
                return torch.norm(state[:2] - state[3:5])
        
        mv.set_goal(0, MotorGoal("motor_goal", IdentityProjector()))
        mv.set_goal(1, NavGoal("nav_goal", IdentityProjector()))
        
        # Compute foam
        foam0 = mv.compute_foam(0)
        foam1 = mv.compute_foam(1)
        
        assert foam0 >= 0
        assert foam1 >= 0


# ======================================================================
# Run tests
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```