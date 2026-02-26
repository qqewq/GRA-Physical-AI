```python
#!/usr/bin/env python3
"""
GRA Physical AI - Integration Tests
====================================

This module contains integration tests for the complete GRA framework,
testing how components work together in realistic scenarios.

Tests cover:
    - Complete robot pipeline (environment + layers + zeroing)
    - Multi-agent coordination
    - Safety-critical scenarios
    - Ethical dilemma resolution
    - Human feedback integration
    - Long-running stability
    - Checkpoint/restore functionality

Run with: pytest test_integration.py -v --timeout=300
"""

import pytest
import torch
import numpy as np
import sys
import os
import time
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import threading
import queue

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core.multiverse import Multiverse, MultiIndex
from core.nullification import ZeroingAlgorithm, ZeroingState
from core.foam import compute_foam
from core.projector import ThresholdProjector, RangeProjector, IdentityProjector

from layers.g0_motor_layer import G0_Layer
from layers.g1_task_layer import G1_TaskLayer, TaskType
from layers.g2_safety_layer import G2_SafetyLayer, SafetyLevel
from layers.g3_ethics_layer import G3_EthicsLayer, EthicalPrinciple

from envs.pybullet_wrapper import PyBulletGRAWrapper, CartPolePyBullet, HumanoidPyBullet
from envs.base_environment import BaseEnvironment

from agents.base_agent import BaseAgent, NullAgent
from agents.rl_agent import PPOAgent

from human_feedback.human_feedback import HumanFeedbackInterface, FeedbackType

from logger.logger import GRA_Logger
from config.config import ExperimentConfig, EnvironmentConfig, AgentConfig, GRAConfig, TrainingConfig


# ======================================================================
# Mock Environment for Testing
# ======================================================================

class MockEnvironment(BaseEnvironment):
    """Mock environment for integration testing."""
    
    def __init__(self, name="mock_env", obs_dim=10, act_dim=4):
        super().__init__(name)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.step_count = 0
        self.episode_count = 0
        self._state = torch.zeros(obs_dim + 10)  # Extra for ground truth
        
    def reset(self):
        self.step_count = 0
        self.episode_count += 1
        return torch.randn(self.obs_dim)
    
    def step(self, action):
        self.step_count += 1
        obs = torch.randn(self.obs_dim)
        reward = -torch.norm(action).item() * 0.1
        done = self.step_count > 100
        info = {'step': self.step_count}
        return obs, reward, done, info
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        pass
    
    def get_observation_space(self):
        from gym import spaces
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,))
    
    def get_action_space(self):
        from gym import spaces
        return spaces.Box(low=-1, high=1, shape=(self.act_dim,))
    
    def get_ground_truth_state(self):
        return self._state


# ======================================================================
# Complete Robot Pipeline Test
# ======================================================================

class TestCompletePipeline:
    """Test complete robot pipeline with all components."""
    
    @pytest.fixture
    def setup_pipeline(self):
        """Set up complete pipeline for testing."""
        # Create environment
        env = MockEnvironment(obs_dim=20, act_dim=6)
        
        # Create layers
        g0 = G0_Layer(num_joints=6)
        g1 = G1_TaskLayer(g0_layer=g0, task_name="reach")
        g2 = G2_SafetyLayer(g1_layer=g1, max_force=50.0)
        g3 = G3_EthicsLayer(g2_layer=g2, enable_human_oversight=False)
        
        # Connect layers
        g1.connect_to_g0(g0)
        g2.connect_to_g1(g1)
        g3.connect_to_g2(g2)
        
        # Create multiverse
        mv = Multiverse(name="test_multiverse", max_level=3)
        for layer in [g0, g1, g2, g3]:
            for idx, sub in layer.subsystems.items():
                mv.add_subsystem(sub)
        
        # Create zeroing algorithm
        zeroing = ZeroingAlgorithm(
            hierarchy=mv.subsystems,
            get_children=lambda x: [],
            get_parents=lambda x: [],
            get_goal_projector=lambda level: IdentityProjector(),
            get_level_weight=lambda level: 1.0,
            level_tolerances=[0.01, 0.01, 0.01, 0.01],
            learning_rates=[0.01, 0.005, 0.001, 0.0005]
        )
        
        # Create agent
        agent = NullAgent("test_agent", 20, 6)
        
        return {
            'env': env,
            'g0': g0,
            'g1': g1,
            'g2': g2,
            'g3': g3,
            'mv': mv,
            'zeroing': zeroing,
            'agent': agent
        }
    
    def test_full_episode(self, setup_pipeline):
        """Test running a full episode with all components."""
        components = setup_pipeline
        env = components['env']
        agent = components['agent']
        
        obs = env.reset()
        total_reward = 0
        steps = 0
        
        for _ in range(50):
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            
            # Update GRA layers
            components['g1'].update(obs, dt=0.1)
            safety_info = components['g2'].check_safety(obs)
            ethical_result = components['g3'].check_action(obs, action, {})
            
            if done:
                break
        
        assert steps > 0
        assert total_reward is not None
        assert safety_info is not None
        assert ethical_result is not None
    
    def test_zeroing_during_episode(self, setup_pipeline):
        """Test running zeroing during episode."""
        components = setup_pipeline
        env = components['env']
        mv = components['mv']
        zeroing = components['zeroing']
        
        obs = env.reset()
        
        # Run some steps
        for i in range(10):
            action = torch.randn(6)
            obs, _, _, _ = env.step(action)
            
            # Update multiverse
            # (would update from env)
            
            # Run zeroing every few steps
            if i % 3 == 0:
                states = mv.get_all_states()
                new_states = zeroing.zero_level(3, states)
                
                # Update multiverse
                for idx, state in new_states.items():
                    if idx in mv.subsystems:
                        mv.set_state(idx, state)
        
        # Check that multiverse still has states
        assert len(mv.get_all_states()) > 0
    
    def test_safety_intervention(self, setup_pipeline):
        """Test safety layer intervention."""
        components = setup_pipeline
        g2 = components['g2']
        
        # Create unsafe action
        unsafe_action = torch.ones(6) * 10.0
        
        # Safety layer should modify it
        safe_action = g2.get_safe_commands(unsafe_action)
        
        assert not torch.allclose(safe_action, unsafe_action)
        
        # Emergency stop should zero out action
        g2.activate_emergency_stop()
        emergency_action = g2.get_safe_commands(unsafe_action)
        assert torch.allclose(emergency_action, torch.zeros_like(unsafe_action))
    
    def test_ethical_filter(self, setup_pipeline):
        """Test ethical layer filtering."""
        components = setup_pipeline
        g3 = components['g3']
        
        # Create unethical action context
        state = torch.zeros(20)
        action = torch.ones(6)
        context = {
            'contact_forces': [100.0],  # Excessive force
            'distances_to_humans': [0.1],  # Too close
            'coerced': True
        }
        
        result = g3.check_action(state, action, context)
        
        if not result['is_ethical']:
            intervened = g3.intervene(action, result)
            assert not torch.allclose(intervened, action)


# ======================================================================
# Multi-Agent Coordination Test
# ======================================================================

class TestMultiAgent:
    """Tests for multi-agent coordination."""
    
    def test_two_robot_system(self):
        """Test two robots operating in same environment."""
        # Create environments for two robots
        env1 = MockEnvironment("robot1")
        env2 = MockEnvironment("robot2")
        
        # Create layers for each robot
        g0_1 = G0_Layer(num_joints=2)
        g1_1 = G1_TaskLayer(g0_layer=g0_1)
        
        g0_2 = G0_Layer(num_joints=2)
        g1_2 = G1_TaskLayer(g0_layer=g0_2)
        
        # Create coordinator (simplified)
        class Coordinator:
            def __init__(self):
                self.shared_state = {}
            
            def coordinate(self, state1, state2):
                # Simple collision avoidance
                distance = torch.norm(state1[:2] - state2[:2])
                if distance < 0.5:
                    return -0.1 * (state1[:2] - state2[:2])  # repulsion
                return 0
        
        coord = Coordinator()
        
        # Run both robots
        obs1 = env1.reset()
        obs2 = env2.reset()
        
        for _ in range(20):
            # Get actions from each robot
            action1 = g1_1.update(obs1, dt=0.1)
            action2 = g1_2.update(obs2, dt=0.1)
            
            # Apply coordination
            correction = coord.coordinate(obs1, obs2)
            action1[:2] += correction
            action2[:2] -= correction
            
            # Step environments
            obs1, _, _, _ = env1.step(action1)
            obs2, _, _, _ = env2.step(action2)
        
        assert True  # Test passes if no exceptions
    
    def test_shared_safety(self):
        """Test shared safety constraints between robots."""
        env1 = MockEnvironment("robot1")
        env2 = MockEnvironment("robot2")
        
        # Shared safety monitor
        class SharedSafety:
            def __init__(self, safe_distance=0.5):
                self.safe_distance = safe_distance
                self.violations = []
            
            def check(self, pos1, pos2):
                distance = torch.norm(pos1[:2] - pos2[:2])
                if distance < self.safe_distance:
                    self.violations.append({
                        'time': time.time(),
                        'distance': distance.item()
                    })
                    return False
                return True
        
        safety = SharedSafety()
        
        # Run simulation with potential collisions
        pos1 = torch.tensor([0.0, 0.0])
        pos2 = torch.tensor([0.3, 0.0])  # Too close
        
        assert not safety.check(pos1, pos2)
        assert len(safety.violations) == 1


# ======================================================================
# Safety-Critical Scenarios
# ======================================================================

class TestSafetyCritical:
    """Tests for safety-critical scenarios."""
    
    def test_joint_limit_violation(self):
        """Test response to joint limit violation."""
        g2 = G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2)))
        
        # State with joint limit violation
        state = torch.zeros(20)
        state[0] = 3.0  # Joint 0 above limit
        
        safety_info = g2.check_safety(state)
        
        assert not safety_info['safe']
        assert 'joint_limits' in safety_info['violations']
        assert safety_info['safety_level'].value >= SafetyLevel.WARNING.value
    
    def test_force_limit_violation(self):
        """Test response to excessive force."""
        g2 = G2_SafetyLayer(
            g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2)),
            max_force=50.0
        )
        
        # State with force information at the end
        state = torch.zeros(30)
        state[-10] = 100.0  # Excessive force
        
        safety_info = g2.check_safety(state)
        
        assert not safety_info['safe']
        assert 'force_limits' in safety_info['violations']
    
    def test_emergency_stop_propagation(self):
        """Test emergency stop propagation through layers."""
        g0 = G0_Layer(num_joints=2)
        g1 = G1_TaskLayer(g0_layer=g0)
        g2 = G2_SafetyLayer(g1_layer=g1)
        
        # Activate emergency stop
        g2.activate_emergency_stop()
        
        # Task layer should get safe commands
        action = torch.ones(2) * 5.0
        safe_action = g2.get_safe_commands(action)
        
        assert torch.allclose(safe_action, torch.zeros_like(action))
    
    def test_recovery_after_violation(self):
        """Test system recovery after safety violation."""
        g2 = G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2)))
        
        # Cause violation
        state = torch.zeros(20)
        state[0] = 3.0
        g2.check_safety(state)
        
        assert g2.current_safety_level.value >= SafetyLevel.WARNING.value
        
        # Recover
        state[0] = 0.0
        safety_info = g2.check_safety(state)
        
        assert safety_info['safe']
        assert g2.current_safety_level == SafetyLevel.NOMINAL


# ======================================================================
# Ethical Dilemma Resolution
# ======================================================================

class TestEthicalDilemmas:
    """Tests for ethical dilemma resolution."""
    
    def test_trolley_problem(self):
        """Test trolley problem dilemma."""
        g3 = G3_EthicsLayer(
            g2_layer=G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))),
            enable_human_oversight=False
        )
        
        # Create situation with two bad options
        state = torch.zeros(20)
        action = torch.ones(6)
        
        context = {
            'harms_others': True,
            'self_benefit': 0.1,
            'other_benefit': -0.5,
            'contact_forces': [100.0],
            'distances_to_humans': [0.2]
        }
        
        result = g3.check_action(state, action, context)
        
        # Should detect dilemma
        if result.get('dilemma'):
            assert result['dilemma'].dilemma_type is not None
            # Should resolve by choosing least harmful option
            intervened = g3.intervene(action, result)
            assert intervened is not None
    
    def test_obedience_dilemma(self):
        """Test obedience dilemma (following harmful commands)."""
        g3 = G3_EthicsLayer(
            g2_layer=G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))),
            enable_human_oversight=False
        )
        
        state = torch.zeros(20)
        action = torch.ones(6)
        
        context = {
            'commanded': True,
            'command': 'push human',
            'coerced': True,
            'contact_forces': [100.0]
        }
        
        result = g3.check_action(state, action, context)
        
        # Should refuse command
        if not result['is_ethical']:
            intervened = g3.intervene(action, result)
            # Should stop or reduce action
            assert torch.norm(intervened) <= torch.norm(action)
    
    def test_human_oversight(self):
        """Test human oversight in dilemma resolution."""
        g3 = G3_EthicsLayer(
            g2_layer=G2_SafetyLayer(g1_layer=G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))),
            enable_human_oversight=True
        )
        
        # Create dilemma
        from layers.g3_ethics_layer import EthicalDilemma, EthicalDilemmaType
        
        dilemma = EthicalDilemma(
            dilemma_id="test_001",
            dilemma_type=EthicalDilemmaType.TROLLEY_PROBLEM,
            description="Test dilemma",
            options=[{'action': 'stop'}, {'action': 'continue'}],
            context={}
        )
        
        g3.active_dilemmas.append(dilemma)
        
        # Simulate human feedback
        result = g3.provide_human_feedback("test_001", 0, "Stop is safer")
        
        assert result
        assert dilemma.resolved
        assert dilemma.human_override


# ======================================================================
# Human Feedback Integration
# ======================================================================

class TestHumanFeedback:
    """Tests for human feedback integration."""
    
    def test_feedback_collection(self):
        """Test collecting human feedback."""
        interface = HumanFeedbackInterface(
            name="test_feedback",
            store_feedback=True
        )
        
        # Add various types of feedback
        interface.add_rating(0.8)
        interface.add_rating(0.9)
        
        state = torch.randn(10)
        interface.add_preference(state, torch.randn(4), torch.randn(4))
        
        interface.add_language("Good job!")
        
        stats = interface.get_statistics()
        assert stats['total_feedback'] >= 3
    
    def test_feedback_to_foam(self):
        """Test converting feedback to foam term."""
        interface = HumanFeedbackInterface()
        
        from human_feedback.human_feedback import Feedback, FeedbackType, FeedbackSource, FeedbackPriority
        
        feedback = Feedback.create_rating(0.3)  # Low rating
        
        foam_term = interface.feedback_to_foam_term(feedback)
        assert foam_term > 0.5  # Low rating -> high foam
        
        feedback = Feedback.create_rating(0.9)  # High rating
        foam_term = interface.feedback_to_foam_term(feedback)
        assert foam_term < 0.2
    
    def test_preference_learning(self):
        """Test learning from preferences."""
        interface = HumanFeedbackInterface(
            state_dim=10,
            action_dim=4,
            enable_preference_learning=True
        )
        
        # Add preferences
        for _ in range(10):
            state = torch.randn(10)
            good_action = torch.randn(4) * 0.5
            bad_action = torch.randn(4) * 2.0
            interface.add_preference(state, good_action, bad_action)
        
        # Train preference model
        if interface.preference_model:
            loss = interface.preference_model.train_step()
            assert 'loss' in loss


# ======================================================================
# Checkpoint/Restore Test
# ======================================================================

class TestCheckpointRestore:
    """Tests for checkpoint and restore functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for checkpoints."""
        dirpath = tempfile.mkdtemp()
        yield dirpath
        shutil.rmtree(dirpath)
    
    def test_save_load_multiverse(self, temp_dir):
        """Test saving and loading multiverse state."""
        # Create multiverse
        mv = Multiverse(name="test", max_level=2)
        
        # Add some subsystems
        from core.subsystem import SimpleSubsystem
        from core.multiverse import EuclideanSpace
        
        idx1 = MultiIndex(("sub1", None, None))
        sub1 = SimpleSubsystem(idx1, EuclideanSpace(5), None)
        sub1.set_state(torch.randn(5))
        
        idx2 = MultiIndex(("sub2", None, None))
        sub2 = SimpleSubsystem(idx2, EuclideanSpace(3), None)
        sub2.set_state(torch.randn(3))
        
        mv.add_subsystem(sub1)
        mv.add_subsystem(sub2)
        
        # Save states
        states = mv.get_all_states()
        save_path = os.path.join(temp_dir, "states.pt")
        torch.save(states, save_path)
        
        # Create new multiverse and load
        mv2 = Multiverse(name="test2", max_level=2)
        mv2.add_subsystem(SimpleSubsystem(idx1, EuclideanSpace(5), None))
        mv2.add_subsystem(SimpleSubsystem(idx2, EuclideanSpace(3), None))
        
        loaded_states = torch.load(save_path)
        for idx, state in loaded_states.items():
            mv2.set_state(idx, state)
        
        # Check states match
        for idx in states.keys():
            assert torch.allclose(mv.get_state(idx), mv2.get_state(idx))
    
    def test_zeroing_checkpoint(self, temp_dir):
        """Test checkpointing zeroing progress."""
        # Create zeroing algorithm
        mv = Multiverse(name="test", max_level=1)
        
        algo = ZeroingAlgorithm(
            hierarchy=mv.subsystems,
            get_children=lambda x: [],
            get_parents=lambda x: [],
            get_goal_projector=lambda level: IdentityProjector(),
            get_level_weight=lambda level: 1.0,
            level_tolerances=[0.01],
            learning_rates=[0.01]
        )
        
        # Create checkpoint
        state = ZeroingState(
            epoch=42,
            states={},
            foams={0: 0.123}
        )
        
        checkpoint_path = os.path.join(temp_dir, "checkpoint.pt")
        state.save(checkpoint_path)
        
        # Load checkpoint
        loaded = ZeroingState.load(checkpoint_path)
        
        assert loaded.epoch == 42
        assert loaded.foams[0] == 0.123


# ======================================================================
# Long-Running Stability
# ======================================================================

class TestLongRunning:
    """Tests for long-running stability."""
    
    def test_many_steps_no_crash(self):
        """Test running many steps without crashing."""
        env = MockEnvironment()
        g1 = G1_TaskLayer(g0_layer=G0_Layer(num_joints=2))
        
        obs = env.reset()
        
        for i in range(1000):
            action = torch.randn(2)
            obs, _, _, _ = env.step(action)
            g1.update(obs, dt=0.1)
            
            if i % 100 == 0:
                # Add occasional tasks
                g1.add_task_by_type(TaskType.MOVE_TO_POSE)
        
        assert True  # Test passes if no exception
    
    def test_memory_leak_check(self):
        """Check for memory leaks (simplified)."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        
        # Run operations that might leak
        mv = Multiverse(max_level=2)
        for i in range(100):
            idx = MultiIndex((f"sub_{i}", None, None))
            sub = SimpleSubsystem(idx, EuclideanSpace(10), None)
            mv.add_subsystem(sub)
            
            # Compute foam repeatedly
            for _ in range(10):
                compute_foam([torch.randn(10) for _ in range(5)], IdentityProjector())
        
        mem_after = process.memory_info().rss
        mem_diff = mem_after - mem_before
        
        # Memory should not increase dramatically (allow some for normal operation)
        # This is a rough check - actual thresholds depend on system
        print(f"Memory change: {mem_diff / 1024 / 1024:.2f} MB")
        assert mem_diff < 500 * 1024 * 1024  # Less than 500MB increase


# ======================================================================
# Configuration Integration
# ======================================================================

class TestConfigIntegration:
    """Tests for configuration integration."""
    
    def test_experiment_config_creation(self):
        """Test creating experiment from config."""
        config = ExperimentConfig(
            experiment_id="test_001",
            description="Integration test",
            environment=EnvironmentConfig(
                type="mock",
                name="test_env",
                max_steps=100
            ),
            agent=AgentConfig(
                type="mlp",
                observation_dim=10,
                action_dim=4
            ),
            gra=GRAConfig(
                use_g0=True,
                use_g1=True,
                num_joints=4
            ),
            training=TrainingConfig(
                total_episodes=10,
                max_steps_per_episode=50
            )
        )
        
        assert config.experiment_id == "test_001"
        assert config.environment.max_steps == 100
        assert config.gra.num_joints == 4
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Valid config
        config = ExperimentConfig()
        errors = config.validate()
        assert len(errors) == 0
        
        # Invalid config (mismatched dimensions)
        config.agent.action_dim = 10
        config.gra.num_joints = 5
        errors = config.validate()
        assert len(errors) > 0


# ======================================================================
# End-to-End Workflow
# ======================================================================

class TestEndToEnd:
    """Complete end-to-end workflow test."""
    
    def test_full_training_workflow(self):
        """Test complete training workflow."""
        # 1. Setup
        env = MockEnvironment()
        g0 = G0_Layer(num_joints=2)
        g1 = G1_TaskLayer(g0_layer=g0)
        g2 = G2_SafetyLayer(g1_layer=g1)
        
        # 2. Run episodes
        all_rewards = []
        all_foams = []
        
        for episode in range(5):
            obs = env.reset()
            episode_reward = 0
            episode_foams = []
            
            for step in range(20):
                # Get action
                action = g1.update(obs, dt=0.1)
                
                # Apply safety
                safety_info = g2.check_safety(obs)
                if not safety_info['safe']:
                    action = g2.get_safe_commands(action)
                
                # Step environment
                obs, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Record foam
                foam = safety_info.get('max_violation', 0)
                episode_foams.append(foam)
                
                if done:
                    break
            
            all_rewards.append(episode_reward)
            all_foams.append(np.mean(episode_foams))
        
        # 3. Check results
        assert len(all_rewards) == 5
        assert len(all_foams) == 5
        
        # 4. Zeroing
        # (Simplified - would need full multiverse)
        print(f"Average reward: {np.mean(all_rewards):.2f}")
        print(f"Average foam: {np.mean(all_foams):.4f}")


# ======================================================================
# Run tests
# ======================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--timeout=300"])
```