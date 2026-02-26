```python
#!/usr/bin/env python3
"""
GRA Physical AI - Main Training Loop for Safety-Critical Robotics
=================================================================

This script implements the main training loop that integrates all GRA components:
    - Environment (PyBullet, Isaac, ROS2)
    - GRA layers (G0-G4)
    - Zeroing algorithm
    - Human feedback
    - Safety monitoring
    - Ethical constraints

The training loop runs episodes, collects data, performs zeroing steps,
and monitors safety and ethical compliance throughout.

Usage:
    python train_gra_safety.py --config configs/humanoid_safety.yaml
"""

import argparse
import torch
import numpy as np
import time
import yaml
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GRA core imports
from core.multiverse import Multiverse, MultiIndex
from core.nullification import ZeroingAlgorithm, ZeroingMonitor
from core.foam import compute_foam, analyze_foam

# Layer imports
from layers.g0_motor_layer import G0_Layer
from layers.g1_task_layer import G1_TaskLayer
from layers.g2_safety_layer import G2_SafetyLayer, SafetyViolationType
from layers.g3_ethics_layer import G3_EthicsLayer, EthicalPrinciple, EthicalDilemma

# Environment imports
from envs.pybullet_wrapper import PyBulletGRAWrapper, HumanoidPyBullet, CartPolePyBullet
from envs.isaac_wrapper import IsaacLabWrapper
from envs.ros2_bridge import ROS2Environment

# Agent imports
from agents.base_agent import BaseAgent
from agents.rl_agent import PPOAgent, SACAgent
from agents.llm_agent import HuggingFaceAgent
from agents.gr00t_agent import Gr00tG3Wrapper

# Human feedback
from human_feedback.human_feedback import HumanFeedbackInterface, Feedback, FeedbackType

# Visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    warnings.warn("Matplotlib not available for visualization")


# ======================================================================
# Configuration
# ======================================================================

@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Environment
    env_type: str = "pybullet"  # pybullet, isaac, ros2
    env_name: str = "humanoid"
    num_envs: int = 1
    max_steps_per_episode: int = 1000
    total_episodes: int = 100
    
    # GRA layers
    use_g0: bool = True
    use_g1: bool = True
    use_g2: bool = True
    use_g3: bool = True
    num_joints: int = 6
    
    # Zeroing
    zeroing_enabled: bool = True
    zeroing_frequency: int = 10  # steps between zeroing
    zeroing_learning_rates: List[float] = None
    zeroing_tolerances: List[float] = None
    
    # Safety
    safety_check_frequency: int = 1  # steps between safety checks
    emergency_stop_on_violation: bool = True
    max_force_limit: float = 50.0
    personal_space: float = 0.5
    
    # Ethics
    ethics_enabled: bool = True
    human_oversight: bool = True
    dilemma_timeout: float = 30.0
    
    # Human feedback
    feedback_enabled: bool = True
    feedback_aggregation_window: float = 60.0
    
    # Logging
    log_dir: str = "./logs"
    save_checkpoints: bool = True
    checkpoint_frequency: int = 10  # episodes
    visualize: bool = True
    
    def __post_init__(self):
        if self.zeroing_learning_rates is None:
            self.zeroing_learning_rates = [0.01, 0.005, 0.001, 0.0005]
        if self.zeroing_tolerances is None:
            self.zeroing_tolerances = [0.01, 0.01, 0.001, 0.001]
    
    @classmethod
    def from_yaml(cls, path: str) -> 'TrainingConfig':
        """Load config from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)


# ======================================================================
# Training Metrics
# ======================================================================

@dataclass
class TrainingMetrics:
    """Training metrics for logging."""
    
    episode: int
    steps: int
    total_reward: float
    avg_reward: float
    foam_levels: Dict[int, float]
    safety_level: int
    ethical_violations: int
    human_feedback_count: int
    zeroing_steps: int
    episode_time: float
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging."""
        return {
            'episode': self.episode,
            'steps': self.steps,
            'total_reward': self.total_reward,
            'avg_reward': self.avg_reward,
            'foam_levels': self.foam_levels,
            'safety_level': self.safety_level,
            'ethical_violations': self.ethical_violations,
            'human_feedback_count': self.human_feedback_count,
            'zeroing_steps': self.zeroing_steps,
            'episode_time': self.episode_time,
            'timestamp': self.timestamp
        }


# ======================================================================
# Main Training Class
# ======================================================================

class GRASafetyTrainer:
    """
    Main training class for GRA-based safety-critical robotics.
    
    Integrates all components and runs the training loop with:
        - Environment interaction
        - Layer updates
        - Zeroing
        - Safety monitoring
        - Ethical supervision
        - Human feedback
    """
    
    def __init__(self, config: TrainingConfig):
        """
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Create log directory
        self.log_dir = os.path.join(config.log_dir, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(self.log_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config.__dict__, f)
        
        # Initialize components
        self._init_environment()
        self._init_layers()
        self._init_agents()
        self._init_zeroing()
        self._init_human_feedback()
        
        # Metrics
        self.metrics: List[TrainingMetrics] = []
        self.current_step = 0
        self.current_episode = 0
        self.zeroing_step_count = 0
        
        # Visualization
        if config.visualize and VISUALIZATION_AVAILABLE:
            self._init_visualization()
        
        print(f"\n=== GRA Safety Trainer Initialized ===")
        print(f"Environment: {config.env_type}/{config.env_name}")
        print(f"Layers: G0={config.use_g0}, G1={config.use_g1}, G2={config.use_g2}, G3={config.use_g3}")
        print(f"Zeroing: {config.zeroing_enabled}")
        print(f"Ethics: {config.ethics_enabled}")
        print(f"Human Feedback: {config.feedback_enabled}")
        print(f"Logging to: {self.log_dir}\n")
    
    def _init_environment(self):
        """Initialize the environment."""
        if self.config.env_type == "pybullet":
            if self.config.env_name == "humanoid":
                self.env = HumanoidPyBullet(gui=True)
            elif self.config.env_name == "cartpole":
                self.env = CartPolePyBullet(gui=True)
            else:
                self.env = PyBulletGRAWrapper(
                    name=self.config.env_name,
                    urdf_paths=[f"assets/{self.config.env_name}.urdf"],
                    gui=True
                )
        elif self.config.env_type == "isaac":
            self.env = IsaacLabWrapper(
                name=self.config.env_name,
                env_class=self.config.env_name,
                headless=False
            )
        elif self.config.env_type == "ros2":
            self.env = ROS2Environment(
                name=self.config.env_name,
                observation_topics=['/joint_states'],
                action_topic='/cmd_vel'
            )
        else:
            raise ValueError(f"Unknown environment type: {self.config.env_type}")
    
    def _init_layers(self):
        """Initialize GRA layers."""
        self.layers = {}
        self.multiverse = Multiverse(name="robot", max_level=3)
        
        # G0: Motor layer
        if self.config.use_g0:
            self.layers[0] = G0_Layer(
                name="motors",
                num_joints=self.config.num_joints,
                joint_limits=[(-2.8, 2.8)] * self.config.num_joints
            )
            # Add to multiverse
            for idx, subsystem in self.layers[0].subsystems.items():
                self.multiverse.add_subsystem(subsystem)
        
        # G1: Task layer
        if self.config.use_g1:
            self.layers[1] = G1_TaskLayer(
                name="task",
                g0_layer=self.layers.get(0)
            )
            for idx, subsystem in self.layers[1].subsystems.items():
                self.multiverse.add_subsystem(subsystem)
            if 0 in self.layers:
                self.layers[1].connect_to_g0(self.layers[0])
        
        # G2: Safety layer
        if self.config.use_g2:
            self.layers[2] = G2_SafetyLayer(
                name="safety",
                g1_layer=self.layers.get(1),
                emergency_stop_on_violation=self.config.emergency_stop_on_violation
            )
            for idx, subsystem in self.layers[2].subsystems.items():
                self.multiverse.add_subsystem(subsystem)
            if 1 in self.layers:
                self.layers[2].connect_to_g1(self.layers[1])
        
        # G3: Ethics layer
        if self.config.use_g3:
            self.layers[3] = G3_EthicsLayer(
                name="ethics",
                g2_layer=self.layers.get(2),
                enable_human_oversight=self.config.human_oversight,
                dilemma_timeout=self.config.dilemma_timeout
            )
            for idx, subsystem in self.layers[3].subsystems.items():
                self.multiverse.add_subsystem(subsystem)
            if 2 in self.layers:
                self.layers[3].connect_to_g2(self.layers[2])
        
        # Set goals in multiverse
        for level, layer in self.layers.items():
            for goal in layer.get_goals():
                self.multiverse.set_goal(level, goal)
        
        # Connect multiverse to environment
        self.env.attach_multiverse(self.multiverse)
    
    def _init_agents(self):
        """Initialize agents for each layer."""
        self.agents = {}
        
        # For now, just use random actions
        # In a real system, you'd have learned policies
        pass
    
    def _init_zeroing(self):
        """Initialize zeroing algorithm."""
        if not self.config.zeroing_enabled:
            self.zeroing_algo = None
            return
        
        # Create zeroing algorithm
        self.zeroing_algo = ZeroingAlgorithm(
            hierarchy=self.multiverse.subsystems,
            get_children=lambda x: [],  # Would need proper hierarchy
            get_parents=lambda x: [],
            get_goal_projector=lambda level: self.multiverse.get_goal(level).projector if self.multiverse.get_goal(level) else None,
            get_level_weight=lambda level: 1.0,
            level_tolerances=self.config.zeroing_tolerances,
            learning_rates=self.config.zeroing_learning_rates
        )
        
        # Add monitor
        self.zeroing_monitor = ZeroingMonitor(self.zeroing_algo)
    
    def _init_human_feedback(self):
        """Initialize human feedback interface."""
        if not self.config.feedback_enabled:
            self.feedback = None
            return
        
        self.feedback = HumanFeedbackInterface(
            name="trainer_feedback",
            state_dim=10,  # Would need actual dimensions
            action_dim=self.config.num_joints,
            auto_aggregate=True,
            aggregate_window=self.config.feedback_aggregation_window
        )
    
    def _init_visualization(self):
        """Initialize visualization."""
        if not VISUALIZATION_AVAILABLE:
            return
        
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 10))
        self.fig.suptitle("GRA Safety Training Monitor")
        
        # Initialize plots
        self.foam_lines = {}
        self.reward_line = None
        self.safety_line = None
        
        plt.ion()
        plt.show()
    
    # ======================================================================
    # Training Loop
    # ======================================================================
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        for episode in range(self.config.total_episodes):
            episode_start = time.time()
            
            # Run episode
            metrics = self._run_episode(episode)
            
            # Log metrics
            self.metrics.append(metrics)
            self._log_episode(metrics)
            
            # Save checkpoint
            if self.config.save_checkpoints and episode % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(episode)
            
            # Update visualization
            if self.config.visualize and VISUALIZATION_AVAILABLE:
                self._update_visualization()
            
            # Print progress
            if episode % 10 == 0:
                self._print_progress(episode)
        
        print("\nTraining complete!")
        self._save_final_results()
    
    def _run_episode(self, episode: int) -> TrainingMetrics:
        """Run a single episode."""
        obs = self.env.reset()
        self.current_episode = episode
        episode_reward = 0
        step_count = 0
        ethical_violations = 0
        
        while step_count < self.config.max_steps_per_episode:
            # Get action from policy (currently random)
            action = torch.randn(self.config.num_joints) * 0.5
            
            # Apply GRA layers
            action = self._apply_layers(action, obs)
            
            # Take step in environment
            next_obs, reward, done, info = self.env.step(action)
            
            episode_reward += reward
            step_count += 1
            self.current_step += 1
            
            # Check safety
            if self.config.use_g2 and step_count % self.config.safety_check_frequency == 0:
                safety_info = self.layers[2].check_safety(next_obs)
                if not safety_info['safe']:
                    ethical_violations += len(safety_info['violations'])
            
            # Check ethics
            if self.config.use_g3:
                # Would check action against ethics
                pass
            
            # Run zeroing step
            if self.config.zeroing_enabled and step_count % self.config.zeroing_frequency == 0:
                self._run_zeroing_step()
                self.zeroing_step_count += 1
            
            # Get human feedback (simulated)
            if self.feedback and np.random.random() < 0.01:  # 1% chance
                self._simulate_human_feedback(obs, action, reward)
            
            obs = next_obs
            
            if done:
                break
        
        # Compute foams
        foams = self.multiverse.compute_all_foams() if self.multiverse else {}
        
        return TrainingMetrics(
            episode=episode,
            steps=step_count,
            total_reward=episode_reward,
            avg_reward=episode_reward / step_count if step_count > 0 else 0,
            foam_levels={l: f.item() for l, f in foams.items()},
            safety_level=self.layers[2].current_safety_level.value if self.config.use_g2 else 0,
            ethical_violations=ethical_violations,
            human_feedback_count=len(self.feedback.feedback_history) if self.feedback else 0,
            zeroing_steps=self.zeroing_step_count,
            episode_time=time.time() - episode_start
        )
    
    def _apply_layers(self, action: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        """Apply GRA layers to action."""
        current_action = action
        
        # G2: Safety layer (intervenes if needed)
        if self.config.use_g2:
            safety_info = self.layers[2].check_safety(obs)
            current_action = self.layers[2].get_safe_commands(current_action)
        
        # G3: Ethics layer (supervises)
        if self.config.use_g3:
            check_result = self.layers[3].check_action(obs, current_action, {
                'safety_info': safety_info if self.config.use_g2 else {}
            })
            if not check_result['is_ethical']:
                current_action = self.layers[3].intervene(current_action, check_result)
        
        return current_action
    
    def _run_zeroing_step(self):
        """Run one step of zeroing algorithm."""
        if not self.zeroing_algo:
            return
        
        # Get current states
        states = self.multiverse.get_all_states()
        
        # Run zeroing
        new_states = self.zeroing_algo.zero_level(
            self.multiverse.max_level,
            states
        )
        
        # Update multiverse
        for idx, new_state in new_states.items():
            self.multiverse.set_state(idx, new_state)
    
    def _simulate_human_feedback(self, obs: torch.Tensor, action: torch.Tensor, reward: float):
        """Simulate human feedback for testing."""
        if not self.feedback:
            return
        
        # Randomly generate feedback types
        feedback_type = np.random.choice([
            FeedbackType.RATING,
            FeedbackType.CORRECTION
        ])
        
        if feedback_type == FeedbackType.RATING:
            # Rating based on reward
            rating = np.clip((reward + 1) / 2, 0, 1)  # Normalize
            self.feedback.add_rating(rating)
        
        elif feedback_type == FeedbackType.CORRECTION:
            # Random correction
            correction = action + torch.randn_like(action) * 0.1
            self.feedback.add_correction(action, correction, reason="Simulated correction")
    
    # ======================================================================
    # Logging and Checkpoints
    # ======================================================================
    
    def _log_episode(self, metrics: TrainingMetrics):
        """Log episode metrics."""
        log_file = os.path.join(self.log_dir, 'metrics.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(metrics.to_dict()) + '\n')
    
    def _print_progress(self, episode: int):
        """Print training progress."""
        recent = self.metrics[-10:]
        avg_reward = np.mean([m.total_reward for m in recent])
        avg_foam = np.mean([np.mean(list(m.foam_levels.values())) for m in recent if m.foam_levels])
        
        print(f"\nEpisode {episode}/{self.config.total_episodes}")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Foam: {avg_foam:.4f}")
        print(f"  Zeroing Steps: {self.zeroing_step_count}")
        if self.config.use_g2:
            print(f"  Safety Level: {self.layers[2].current_safety_level.name}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(self.log_dir, f'checkpoint_{episode}')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save multiverse state
        if self.multiverse:
            states = self.multiverse.get_all_states()
            torch.save(states, os.path.join(checkpoint_dir, 'multiverse_states.pt'))
        
        # Save metrics
        with open(os.path.join(checkpoint_dir, 'metrics.json'), 'w') as f:
            json.dump([m.to_dict() for m in self.metrics[-100:]], f)
        
        print(f"  Saved checkpoint to {checkpoint_dir}")
    
    def _save_final_results(self):
        """Save final training results."""
        results = {
            'config': self.config.__dict__,
            'metrics': [m.to_dict() for m in self.metrics],
            'total_steps': self.current_step,
            'total_episodes': self.current_episode
        }
        
        with open(os.path.join(self.log_dir, 'final_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {self.log_dir}")
    
    # ======================================================================
    # Visualization
    # ======================================================================
    
    def _update_visualization(self):
        """Update visualization."""
        if not VISUALIZATION_AVAILABLE or not hasattr(self, 'fig'):
            return
        
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot rewards
        ax = self.axes[0, 0]
        rewards = [m.total_reward for m in self.metrics]
        ax.plot(rewards)
        ax.set_title('Episode Rewards')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        
        # Plot foams
        ax = self.axes[0, 1]
        if self.metrics and self.metrics[-1].foam_levels:
            levels = list(self.metrics[-1].foam_levels.keys())
            foams = [self.metrics[-1].foam_levels[l] for l in levels]
            ax.bar(levels, foams)
        ax.set_title('Current Foam Levels')
        ax.set_xlabel('Level')
        ax.set_ylabel('Foam')
        
        # Plot safety level
        ax = self.axes[0, 2]
        if self.config.use_g2:
            safety_levels = [m.safety_level for m in self.metrics]
            ax.plot(safety_levels)
        ax.set_title('Safety Level')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Level')
        ax.set_ylim(0, 4)
        
        # Plot ethical violations
        ax = self.axes[1, 0]
        violations = [m.ethical_violations for m in self.metrics]
        ax.plot(violations)
        ax.set_title('Ethical Violations')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Count')
        
        # Plot feedback
        ax = self.axes[1, 1]
        if self.feedback:
            feedback_counts = [m.human_feedback_count for m in self.metrics]
            ax.plot(feedback_counts)
        ax.set_title('Human Feedback')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Count')
        
        # Plot zeroing steps
        ax = self.axes[1, 2]
        zeroing_steps = [m.zeroing_steps for m in self.metrics]
        ax.plot(zeroing_steps)
        ax.set_title('Zeroing Steps')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)
    
    # ======================================================================
    # Cleanup
    # ======================================================================
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'env'):
            self.env.close()
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        print("Training closed.")


# ======================================================================
# Command Line Interface
# ======================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='GRA Safety Training')
    
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('--env', type=str, default='humanoid',
                        choices=['humanoid', 'cartpole', 'custom'],
                        help='Environment name')
    
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes')
    
    parser.add_argument('--no-zeroing', action='store_true',
                        help='Disable zeroing')
    
    parser.add_argument('--no-ethics', action='store_true',
                        help='Disable ethics layer')
    
    parser.add_argument('--no-feedback', action='store_true',
                        help='Disable human feedback')
    
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Log directory')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load config
    if os.path.exists(args.config):
        config = TrainingConfig.from_yaml(args.config)
    else:
        config = TrainingConfig()
    
    # Override with command line args
    config.env_name = args.env
    config.total_episodes = args.episodes
    config.zeroing_enabled = not args.no_zeroing
    config.ethics_enabled = not args.no_ethics
    config.feedback_enabled = not args.no_feedback
    config.log_dir = args.log_dir
    
    # Create trainer
    trainer = GRASafetyTrainer(config)
    
    try:
        # Run training
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
```