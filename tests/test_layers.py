```python
#!/usr/bin/env python3
"""
GRA Physical AI - Layers Module Tests
======================================

This module contains unit tests for the GRA layers:
    - G0_Layer (motor control)
    - G1_TaskLayer (task execution)
    - G2_SafetyLayer (safety monitoring)
    - G3_EthicsLayer (ethical constraints)
    - CombinedLayer (hierarchical composition)

Run with: pytest test_layers.py -v
"""

import pytest
import torch
import numpy as np
import sys
import os
from pathlib import Path
import time
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from layers.g0_motor_layer import G0_Layer, MotorTrackingGoal
from layers.g1_task_layer import G1_TaskLayer, Task, TaskType, TaskStatus, MoveToPosePrimitive
from layers.g2_safety_layer import G2_SafetyLayer, SafetyLevel, SafetyViolationType, JointLimitConstraint
from layers.g3_ethics_layer import G3_EthicsLayer, EthicalPrinciple, EthicalViolationType, EthicalConstitution
from layers.base_layer import GRA_layer, CombinedLayer

from core.multiverse import MultiIndex
from core.subsystem import SimpleSubsystem
from core.projector import ThresholdProjector, RangeProjector


# ======================================================================
# G0 Layer Tests
# ======================================================================

class TestG0Layer:
    """Tests for G0 motor layer."""
    
    def test_layer_creation(self):
        """Test G0 layer creation."""
        layer = G0_Layer(
            name="test_motors",
            num_joints=6,
            joint_names=[f"joint_{i}" for i in range(6)],
            joint_limits=[(-2.8, 2.8)] * 6,
            motor_types=['torque'] * 6
        )
        
        assert layer.level == 0
        assert layer.name == "test_motors"
        assert layer.num_joints == 6
        assert len(layer.subsystems) == 6  # One per joint
        assert len(layer.get_goals()) == 1
    
    def test_subsystem_creation(self):
        """Test that subsystems are created correctly."""
        layer = G0_Layer(num_joints=3)
        
        # Check that we have 3 joint subsystems
        joint_indices = [idx for idx in layer.subsystems.keys() if idx.level == 0]
        assert len(joint_indices) == 3
        
        # Check naming
        for i, idx in enumerate(joint_indices):
            assert idx.indices[0] == f"joint_{i}"
    
    def test_motor_goal(self):
        """Test motor tracking goal."""
        goal = MotorTrackingGoal(
            joint_names=['j0', 'j1'],
            motor_types=['torque', 'velocity'],
            joint_limits=[(-1, 1), (-2, 2)]
        )
        
        # Test loss calculation
        state = torch.tensor([0.5, 0.3, 0.1,  # torque cmd, pos, vel for j0
                             0.8, 0.7])        # vel cmd, actual for j1
        
        loss = goal.loss(state)
        assert loss > 0
        assert isinstance(loss, torch.Tensor)
    
    def test_state_extraction(self):
        """Test state extraction from layer."""
        layer = G0_Layer(num_joints=2)
        states = layer.extract_states()
        
        assert len(states) == 2
        for state in states.values():
            assert isinstance(state, torch.Tensor)


# ======================================================================
# G1 Task Layer Tests
# ======================================================================

class TestG1TaskLayer:
    """Tests for G1 task layer."""
    
    def test_layer_creation(self):
        """Test G1 layer creation."""
        g0 = G0_Layer(num_joints=2)
        layer = G1_TaskLayer(
            name="test_task",
            g0_layer=g0,
            max_concurrent_tasks=2,
            task_timeout=10.0
        )
        
        assert layer.level == 1
        assert layer.name == "test_task"
        assert layer.max_concurrent_tasks == 2
        assert len(layer.subsystems) == 2  # executor + queue
        assert len(layer.get_goals()) == 3
    
    def test_task_creation(self):
        """Test task creation."""
        layer = G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))
        
        task_id = layer.add_task_by_type(
            TaskType.MOVE_TO_POSE,
            parameters={'target_pose': [1.0, 0.0, 0.0]},
            priority=1
        )
        
        assert task_id is not None
        assert len(layer.task_queue) == 1
        assert layer.task_queue[0].task_type == TaskType.MOVE_TO_POSE
    
    def test_task_cancellation(self):
        """Test task cancellation."""
        layer = G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))
        
        task_id = layer.add_task_by_type(TaskType.MOVE_TO_POSE)
        assert len(layer.task_queue) == 1
        
        result = layer.cancel_task(task_id)
        assert result
        assert len(layer.task_queue) == 0
    
    def test_task_execution(self):
        """Test task execution update."""
        layer = G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))
        
        # Add a task
        layer.add_task_by_type(TaskType.MOVE_TO_POSE, parameters={'target_pose': [1, 0, 0]})
        
        # Update should move task from queue to active
        current_state = torch.zeros(10)
        commands = layer.update(current_state, dt=0.1)
        
        assert len(layer.active_tasks) == 1
        assert len(layer.task_queue) == 0
        assert isinstance(commands, torch.Tensor)
    
    def test_task_status(self):
        """Test task status tracking."""
        layer = G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))
        
        task_id = layer.add_task_by_type(TaskType.MOVE_TO_POSE)
        assert layer.get_task_status(task_id) == TaskStatus.PENDING
        
        # Start execution
        layer.update(torch.zeros(10), dt=0.1)
        assert layer.get_task_status(task_id) == TaskStatus.RUNNING
    
    def test_statistics(self):
        """Test task statistics."""
        layer = G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))
        
        # Add and complete some tasks
        for i in range(3):
            task_id = layer.add_task_by_type(TaskType.MOVE_TO_POSE)
            layer.update(torch.zeros(10), dt=0.1)
        
        stats = layer.get_statistics()
        assert 'queue_length' in stats
        assert 'active_tasks' in stats
        assert 'completed_tasks' in stats
        assert 'success_rate' in stats


# ======================================================================
# G2 Safety Layer Tests
# ======================================================================

class TestG2SafetyLayer:
    """Tests for G2 safety layer."""
    
    def test_layer_creation(self):
        """Test G2 layer creation."""
        g1 = G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))
        layer = G2_SafetyLayer(
            name="test_safety",
            g1_layer=g1,
            max_force=50.0,
            personal_space=0.5,
            emergency_stop_on_violation=True
        )
        
        assert layer.level == 2
        assert layer.name == "test_safety"
        assert len(layer.subsystems) == 2  # monitor + estop
        assert len(layer.get_goals()) == 3
    
    def test_safety_check(self):
        """Test safety checking."""
        layer = G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2)))
        
        # Create a state with violations
        state = torch.zeros(20)
        # Set joint near limit
        if len(state) > 0:
            state[0] = 3.0  # Above limit (assuming limit 2.8)
        
        safety_info = layer.check_safety(state)
        
        assert 'safe' in safety_info
        assert 'safety_level' in safety_info
        assert 'violations' in safety_info
    
    def test_emergency_stop(self):
        """Test emergency stop activation."""
        layer = G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2)))
        
        assert not layer.emergency_stop_active
        
        layer.activate_emergency_stop()
        assert layer.emergency_stop_active
        
        commands = torch.randn(6)
        safe_commands = layer.get_safe_commands(commands)
        assert torch.allclose(safe_commands, torch.zeros_like(commands))
        
        layer.deactivate_emergency_stop()
        assert not layer.emergency_stop_active
    
    def test_joint_limit_constraint(self):
        """Test joint limit constraint."""
        constraint = JointLimitConstraint(
            joint_limits=[(-1.0, 1.0), (-2.0, 2.0)],
            joint_names=['j0', 'j1']
        )
        
        # Within limits
        state = torch.tensor([0.5, 1.0])
        satisfied, mag = constraint.check(state)
        assert satisfied
        assert mag == 0
        
        # Exceeding limits
        state = torch.tensor([1.5, 2.5])
        satisfied, mag = constraint.check(state)
        assert not satisfied
        assert mag > 0
    
    def test_safety_prediction(self):
        """Test safety prediction."""
        layer = G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2)))
        
        state = torch.zeros(20)
        predictions = layer.predict_safety(state, horizon=1.0)
        
        assert 'predictions' in predictions
        assert 'min_time_to_violation' in predictions
        assert 'imminent_violation' in predictions
    
    def test_incident_logging(self):
        """Test safety incident logging."""
        layer = G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2)))
        
        # Create a violation
        state = torch.zeros(20)
        if len(state) > 0:
            state[0] = 3.0  # Joint limit violation
        
        layer.check_safety(state)
        
        assert len(layer.incidents) > 0
        assert layer.incidents[0].violation_type is not None


# ======================================================================
# G3 Ethics Layer Tests
# ======================================================================

class TestG3EthicsLayer:
    """Tests for G3 ethics layer."""
    
    def test_layer_creation(self):
        """Test G3 layer creation."""
        g2 = G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2)))
        layer = G3_EthicsLayer(
            name="test_ethics",
            g2_layer=g2,
            enable_human_oversight=True,
            dilemma_timeout=30.0
        )
        
        assert layer.level == 3
        assert layer.name == "test_ethics"
        assert len(layer.subsystems) == 2  # supervisor + constitution
        assert len(layer.get_goals()) == 3
        assert layer.constitution is not None
    
    def test_constitution_integrity(self):
        """Test ethical constitution integrity."""
        constitution = EthicalConstitution(version="1.0.0")
        
        assert constitution.verify_integrity()
        assert len(constitution.get_all_principles()) == 4
        
        principle = constitution.get_principle(EthicalPrinciple.DO_NO_HARM)
        assert principle is not None
        assert 'name' in principle
    
    def test_ethical_check(self):
        """Test ethical action checking."""
        layer = G3_EthicsLayer(g2_layer=G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))))
        
        state = torch.zeros(10)
        action = torch.randn(6)
        
        # Test with safe context
        context = {
            'contact_forces': [10.0],
            'distances_to_humans': [2.0],
            'in_privacy_zone': False,
            'coerced': False
        }
        
        result = layer.check_action(state, action, context)
        assert 'is_ethical' in result
        assert 'ethical_scores' in result
    
    def test_ethical_intervention(self):
        """Test ethical intervention."""
        layer = G3_EthicsLayer(g2_layer=G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))))
        
        # Create a violating action
        state = torch.zeros(10)
        action = torch.randn(6) * 10.0  # Large action
        
        context = {
            'contact_forces': [100.0],  # Excessive force
            'distances_to_humans': [0.1],  # Too close
            'in_privacy_zone': True
        }
        
        result = layer.check_action(state, action, context)
        assert not result['is_ethical']
        
        intervened = layer.intervene(action, result)
        assert not torch.allclose(intervened, action)
    
    def test_dilemma_detection(self):
        """Test ethical dilemma detection."""
        layer = G3_EthicsLayer(g2_layer=G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))))
        
        # Create situation with multiple violations
        state = torch.zeros(10)
        action = torch.randn(6)
        
        context = {
            'contact_forces': [100.0],
            'distances_to_humans': [0.1],
            'commanded': True,
            'command': 'push_human'
        }
        
        result = layer.check_action(state, action, context)
        
        # Should detect dilemma
        if result.get('dilemma'):
            assert result['dilemma'].dilemma_type is not None
    
    def test_human_oversight(self):
        """Test human oversight request."""
        layer = G3_EthicsLayer(
            g2_layer=G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))),
            enable_human_oversight=True
        )
        
        # Create a dilemma
        from layers.g3_ethics_layer import EthicalDilemma, EthicalDilemmaType
        
        dilemma = EthicalDilemma(
            dilemma_id="test_001",
            dilemma_type=EthicalDilemmaType.TROLLEY_PROBLEM,
            description="Test dilemma",
            options=[{'action': 'a'}, {'action': 'b'}],
            context={}
        )
        
        layer.active_dilemmas.append(dilemma)
        
        # Simulate human feedback
        result = layer.provide_human_feedback("test_001", 1, "Human choice")
        
        assert result
        assert dilemma.resolved
        assert dilemma.human_override


# ======================================================================
# Combined Layer Tests
# ======================================================================

class TestCombinedLayer:
    """Tests for combined layer."""
    
    def test_layer_combination(self):
        """Test combining multiple layers."""
        g0 = G0_Layer(num_joints=2)
        g1 = G1_TaskLayer(g0_layer=g0)
        g2 = G2_SafetyLayer(g1_layer=g1)
        
        combined = CombinedLayer(
            name="test_combined",
            level=3,
            layers=[g0, g1, g2]
        )
        
        assert combined.level == 3
        assert combined.name == "test_combined"
        assert len(combined.layers) == 3
        assert len(combined.subsystems) > 0  # Should include all subsystems
    
    def test_coordinator_subsystem(self):
        """Test coordinator subsystem in combined layer."""
        g0 = G0_Layer(num_joints=2)
        g1 = G1_TaskLayer(g0_layer=g0)
        
        combined = CombinedLayer(
            name="test",
            level=2,
            layers=[g0, g1]
        )
        
        # Check for coordinator subsystem
        coord_indices = [idx for idx in combined.subsystems.keys() 
                        if idx.indices[3] is not None and 'coord' in idx.indices[3]]
        assert len(coord_indices) == 1
    
    def test_combined_goals(self):
        """Test goals from combined layer."""
        g0 = G0_Layer(num_joints=2)
        g1 = G1_TaskLayer(g0_layer=g0)
        
        combined = CombinedLayer(
            name="test",
            level=2,
            layers=[g0, g1]
        )
        
        goals = combined.get_goals()
        # Should include goals from both layers plus coordination goal
        assert len(goals) >= len(g0.get_goals()) + len(g1.get_goals())


# ======================================================================
# Layer Integration Tests
# ======================================================================

class TestLayerIntegration:
    """Integration tests for multiple layers working together."""
    
    def test_full_hierarchy(self):
        """Test complete G0-G3 hierarchy."""
        # Create layers
        g0 = G0_Layer(num_joints=7)
        g1 = G1_TaskLayer(g0_layer=g0)
        g2 = G2_SafetyLayer(g1_layer=g1)
        g3 = G3_EthicsLayer(g2_layer=g2)
        
        # Connect them
        g1.connect_to_g0(g0)
        g2.connect_to_g1(g1)
        g3.connect_to_g2(g2)
        
        # Check connections
        assert g0.parent == g1
        assert g1.parent == g2
        assert g2.parent == g3
        assert g3.parent is None
        
        # Test information flow
        state = torch.zeros(7*3)  # Joint states
        
        # G1 generates commands
        commands = g1.update(state, dt=0.1)
        
        # G2 checks safety
        safety_info = g2.check_safety(state)
        
        # G3 checks ethics
        ethical_result = g3.check_action(state, commands, {
            'safety_info': safety_info
        })
        
        assert ethical_result is not None
    
    def test_state_propagation(self):
        """Test state propagation through layers."""
        g0 = G0_Layer(num_joints=2)
        g1 = G1_TaskLayer(g0_layer=g0)
        
        # Set some state in G0
        joint_idx = list(g0.subsystems.keys())[0]
        g0.subsystems[joint_idx].set_state(torch.tensor([0.5, 0.3, 0.1]))
        
        # Extract states
        g0_states = g0.extract_states()
        assert len(g0_states) == 2
        
        # G1 should be able to access G0 states through connection
        assert g1.g0_layer == g0
    
    def test_safety_ethics_interaction(self):
        """Test interaction between safety and ethics layers."""
        g0 = G0_Layer(num_joints=2)
        g1 = G1_TaskLayer(g0_layer=g0)
        g2 = G2_SafetyLayer(g1_layer=g1)
        g3 = G3_EthicsLayer(g2_layer=g2)
        
        # Create a safety violation
        state = torch.zeros(20)
        if len(state) > 0:
            state[0] = 3.0  # Joint limit violation
        
        safety_info = g2.check_safety(state)
        
        # Ethics should consider safety info
        action = torch.randn(2)
        ethical_result = g3.check_action(state, action, {
            'safety_info': safety_info,
            'contact_forces': [100.0]
        })
        
        # Should detect violation
        assert not ethical_result['is_ethical'] or safety_info['violations']


# ======================================================================
# Error Handling Tests
# ======================================================================

class TestLayerErrors:
    """Tests for error handling in layers."""
    
    def test_invalid_layer_creation(self):
        """Test invalid layer creation."""
        with pytest.raises(ValueError):
            # Invalid motor type
            G0_Layer(
                num_joints=1,
                motor_types=['invalid_type']
            )
    
    def test_missing_connections(self):
        """Test behavior with missing connections."""
        g1 = G1_TaskLayer()  # No G0 layer
        
        # Should still work but without lower level
        state = torch.zeros(10)
        commands = g1.update(state, dt=0.1)
        assert commands is not None
    
    def test_invalid_task_parameters(self):
        """Test invalid task parameters."""
        layer = G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))
        
        # Task with missing parameters
        with pytest.raises(Exception):
            layer.add_task_by_type(
                TaskType.MOVE_TO_POSE,
                parameters={}  # Missing target_pose
            )


# ======================================================================
# Performance Tests
# ======================================================================

class TestLayerPerformance:
    """Performance tests for layers."""
    
    def test_g0_update_speed(self):
        """Test G0 layer update speed."""
        layer = G0_Layer(num_joints=10)
        
        start = time.time()
        for _ in range(1000):
            states = layer.extract_states()
        elapsed = time.time() - start
        
        print(f"G0 1000 state extractions: {elapsed:.3f}s")
        assert elapsed < 1.0  # Should be fast
    
    def test_g1_update_speed(self):
        """Test G1 layer update speed."""
        g0 = G0_Layer(num_joints=7)
        layer = G1_TaskLayer(g0_layer=g0)
        
        # Add many tasks
        for i in range(100):
            layer.add_task_by_type(TaskType.MOVE_TO_POSE)
        
        start = time.time()
        for _ in range(100):
            layer.update(torch.zeros(20), dt=0.1)
        elapsed = time.time() - start
        
        print(f"G1 100 updates: {elapsed:.3f}s")
        assert elapsed < 2.0
    
    def test_safety_check_speed(self):
        """Test safety check speed."""
        layer = G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=7)))
        
        state = torch.randn(50)
        
        start = time.time()
        for _ in range(100):
            layer.check_safety(state)
        elapsed = time.time() - start
        
        print(f"Safety 100 checks: {elapsed:.3f}s")
        assert elapsed < 1.0


# ======================================================================
# Run tests
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
```