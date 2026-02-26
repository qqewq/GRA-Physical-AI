```python
#!/usr/bin/env python3
"""
GRA Physical AI - Policy Fine-tuning with GR00T-Mimic
=====================================================

This script fine-tunes policies using demonstrations collected from humans.
It implements the GR00T-Mimic approach: using human demonstrations to
initialize and refine policies within the GRA safety-ethics framework.

Key features:
    - Behavioral cloning from demonstrations
    - Reinforcement learning with demonstration replay
    - Inverse reinforcement learning from preferences
    - Safe policy updates with GRA constraints
    - Ethical alignment via human feedback
    - Integration with GR00T foundation models
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import yaml
import json
import os
import sys
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GRA core imports
from core.multiverse import Multiverse, MultiIndex
from core.nullification import ZeroingAlgorithm
from core.foam import compute_foam

# Layer imports
from layers.g0_motor_layer import G0_Layer
from layers.g1_task_layer import G1_TaskLayer
from layers.g2_safety_layer import G2_SafetyLayer
from layers.g3_ethics_layer import G3_EthicsLayer

# Agent imports
from agents.base_agent import BaseAgent
from agents.rl_agent import PPOAgent, SACAgent
from agents.gr00t_agent import Gr00tBaseWrapper, Gr00tPolicyAgent

# Environment imports
from envs.pybullet_wrapper import PyBulletGRAWrapper, HumanoidPyBullet

# Human feedback
from human_feedback.human_feedback import HumanFeedbackInterface, FeedbackType

# Dataset handling
try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False
    warnings.warn("h5py not available for dataset loading")

try:
    import gym
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False


# ======================================================================
# Dataset Classes
# ======================================================================

class DemonstrationDataset(torch.utils.data.Dataset):
    """PyTorch dataset for demonstration data."""
    
    def __init__(self, demos_path: str, mode: str = 'bc'):
        """
        Args:
            demos_path: Path to demonstration file (HDF5 or NPZ)
            mode: 'bc' for behavioral cloning, 'irl' for inverse RL
        """
        self.mode = mode
        self.data = self._load_demos(demos_path)
        
    def _load_demos(self, path: str) -> List[Dict]:
        """Load demonstrations from file."""
        data = []
        
        if path.endswith('.hdf5') and H5PY_AVAILABLE:
            with h5py.File(path, 'r') as f:
                num_demos = f.attrs.get('num_demos', 0)
                for i in range(num_demos):
                    demo_group = f[f'demonstrations/demo_{i}']
                    steps_group = demo_group['steps']
                    
                    observations = []
                    actions = []
                    
                    for step_name in sorted(steps_group.keys()):
                        step = steps_group[step_name]
                        if 'observation' in step and 'action' in step:
                            obs = np.array(step['observation'])
                            act = np.array(step['action'])
                            observations.append(obs)
                            actions.append(act)
                    
                    if observations:
                        data.append({
                            'observations': np.stack(observations),
                            'actions': np.stack(actions)
                        })
        
        elif path.endswith('.npz'):
            npz = np.load(path, allow_pickle=True)
            if 'observations' in npz and 'actions' in npz:
                data.append({
                    'observations': npz['observations'],
                    'actions': npz['actions']
                })
            else:
                # Multiple demos in one file
                for key in npz.files:
                    if key.startswith('demo_'):
                        demo = npz[key].item()
                        if 'observations' in demo and 'actions' in demo:
                            data.append(demo)
        
        elif path.endswith('.json'):
            with open(path, 'r') as f:
                json_data = json.load(f)
                for demo in json_data.get('demonstrations', []):
                    steps = demo.get('steps', [])
                    if steps:
                        observations = [s['observation'] for s in steps if s.get('observation')]
                        actions = [s['action'] for s in steps if s.get('action')]
                        if observations and actions:
                            data.append({
                                'observations': np.array(observations),
                                'actions': np.array(actions)
                            })
        
        print(f"Loaded {len(data)} demonstrations from {path}")
        return data
    
    def __len__(self) -> int:
        return sum(len(d['observations']) for d in self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get (observation, action) pair."""
        # Find which demo and which step
        cum_len = 0
        for demo in self.data:
            demo_len = len(demo['observations'])
            if idx < cum_len + demo_len:
                step_idx = idx - cum_len
                obs = demo['observations'][step_idx]
                act = demo['actions'][step_idx]
                return torch.FloatTensor(obs), torch.FloatTensor(act)
            cum_len += demo_len
        
        raise IndexError("Index out of range")
    
    def get_demonstrations(self) -> List[Dict]:
        """Get all demonstrations."""
        return self.data


class PreferenceDataset(torch.utils.data.Dataset):
    """Dataset for preference learning (pairwise comparisons)."""
    
    def __init__(self, preferences: List[Dict]):
        """
        Args:
            preferences: List of preference tuples
                Each tuple: (state, action_a, action_b, preference)
                preference = 0 if A preferred, 1 if B preferred
        """
        self.preferences = preferences
    
    def __len__(self) -> int:
        return len(self.preferences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pref = self.preferences[idx]
        return (torch.FloatTensor(pref['state']),
                torch.FloatTensor(pref['action_a']),
                torch.FloatTensor(pref['action_b']),
                torch.tensor(pref['preference']))


# ======================================================================
# Loss Functions
# ======================================================================

def behavioral_cloning_loss(policy: nn.Module, 
                           observations: torch.Tensor,
                           target_actions: torch.Tensor) -> torch.Tensor:
    """
    Behavioral cloning loss (MSE between predicted and target actions).
    
    Args:
        policy: Policy network
        observations: Batch of observations
        target_actions: Batch of target actions
    
    Returns:
        Loss value
    """
    predicted_actions = policy(observations)
    return nn.MSELoss()(predicted_actions, target_actions)


def preference_loss(reward_model: nn.Module,
                   states: torch.Tensor,
                   actions_a: torch.Tensor,
                   actions_b: torch.Tensor,
                   preferences: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry preference loss for reward learning.
    
    Args:
        reward_model: Reward model (maps state-action to scalar)
        states: Batch of states
        actions_a: First actions
        actions_b: Second actions
        preferences: 0 if A preferred, 1 if B preferred
    
    Returns:
        Loss value
    """
    # Compute rewards
    sa_a = torch.cat([states, actions_a], dim=-1)
    sa_b = torch.cat([states, actions_b], dim=-1)
    
    r_a = reward_model(sa_a).squeeze()
    r_b = reward_model(sa_b).squeeze()
    
    # Bradley-Terry model
    logits = r_a - r_b
    loss = nn.BCEWithLogitsLoss()(logits, preferences.float())
    
    return loss


def gra_constraint_loss(policy_state: torch.Tensor,
                       multiverse: Multiverse,
                       zeroing_algo: ZeroingAlgorithm) -> torch.Tensor:
    """
    GRA constraint loss to ensure policy respects GRA layers.
    
    Args:
        policy_state: Current policy parameters (flattened)
        multiverse: GRA multiverse
        zeroing_algo: Zeroing algorithm
    
    Returns:
        Constraint violation loss
    """
    # Update policy parameters in multiverse
    # This assumes policy is a subsystem in multiverse
    # For now, return dummy loss
    return torch.tensor(0.0)


# ======================================================================
# Fine-tuning Algorithms
# ======================================================================

class BehavioralCloning:
    """
    Behavioral cloning fine-tuning.
    
    Trains policy to mimic demonstrations via supervised learning.
    """
    
    def __init__(self,
                 policy: nn.Module,
                 dataset: DemonstrationDataset,
                 learning_rate: float = 1e-4,
                 batch_size: int = 64,
                 device: str = 'cuda'):
        """
        Args:
            policy: Policy network to fine-tune
            dataset: Demonstration dataset
            learning_rate: Learning rate
            batch_size: Batch size
            device: Computation device
        """
        self.policy = policy.to(device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.device = device
        
        self.optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        self.loss_history = []
    
    def train_step(self) -> Dict[str, float]:
        """Perform one training step."""
        self.policy.train()
        
        try:
            obs, actions = next(self.train_iter)
        except (StopIteration, AttributeError):
            self.train_iter = iter(self.dataloader)
            obs, actions = next(self.train_iter)
        
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        
        loss = behavioral_cloning_loss(self.policy, obs, actions)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        loss_val = loss.item()
        self.loss_history.append(loss_val)
        
        return {'bc_loss': loss_val}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0
        
        for obs, actions in self.dataloader:
            obs = obs.to(self.device)
            actions = actions.to(self.device)
            
            loss = behavioral_cloning_loss(self.policy, obs, actions)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        self.loss_history.append(avg_loss)
        
        return {'bc_loss': avg_loss}
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_history': self.loss_history
        }, path)


class ReinforcementLearningFineTuner:
    """
    Reinforcement learning fine-tuning with demonstration replay.
    
    Combines RL with demonstration replay to improve policy.
    """
    
    def __init__(self,
                 rl_agent: BaseAgent,
                 dataset: DemonstrationDataset,
                 demo_replay_ratio: float = 0.25,
                 demo_replay_weight: float = 1.0,
                 device: str = 'cuda'):
        """
        Args:
            rl_agent: RL agent to fine-tune
            dataset: Demonstration dataset
            demo_replay_ratio: Fraction of batch from demonstrations
            demo_replay_weight: Weight for demonstration loss
            device: Computation device
        """
        self.agent = rl_agent
        self.dataset = dataset
        self.demo_replay_ratio = demo_replay_ratio
        self.demo_replay_weight = demo_replay_weight
        self.device = device
        
        # Create dataloader for demonstrations
        self.demo_loader = torch.utils.data.DataLoader(
            dataset, batch_size=64, shuffle=True
        )
        self.demo_iter = iter(self.demo_loader)
    
    def train_step(self, env_samples: List[Dict]) -> Dict[str, float]:
        """
        Perform one training step with mixed RL and demonstration data.
        
        Args:
            env_samples: Batch of environment samples
                Each sample: {'state', 'action', 'reward', 'next_state', 'done'}
        
        Returns:
            Dictionary of loss metrics
        """
        # Get demonstration batch
        try:
            demo_obs, demo_actions = next(self.demo_iter)
        except StopIteration:
            self.demo_iter = iter(self.demo_loader)
            demo_obs, demo_actions = next(self.demo_iter)
        
        demo_obs = demo_obs.to(self.device)
        demo_actions = demo_actions.to(self.device)
        
        # Prepare environment batch
        env_states = torch.stack([s['state'] for s in env_samples]).to(self.device)
        env_actions = torch.stack([s['action'] for s in env_samples]).to(self.device)
        env_rewards = torch.tensor([s['reward'] for s in env_samples]).to(self.device)
        env_next_states = torch.stack([s['next_state'] for s in env_samples]).to(self.device)
        env_dones = torch.tensor([s['done'] for s in env_samples]).to(self.device)
        
        # Compute RL loss (varies by algorithm)
        if hasattr(self.agent, 'compute_loss'):
            rl_loss_dict = self.agent.compute_loss(
                env_states, env_actions, env_rewards, env_next_states, env_dones
            )
            rl_loss = sum(rl_loss_dict.values())
        else:
            # Fallback: behavioral cloning on demonstrations only
            rl_loss = torch.tensor(0.0)
            rl_loss_dict = {}
        
        # Compute demonstration loss (behavioral cloning)
        demo_pred_actions = self.agent.policy_network(demo_obs)
        demo_loss = nn.MSELoss()(demo_pred_actions, demo_actions)
        
        # Combined loss
        total_loss = rl_loss + self.demo_replay_weight * demo_loss
        
        # Optimize
        if hasattr(self.agent, 'optimizer'):
            self.agent.optimizer.zero_grad()
            total_loss.backward()
            self.agent.optimizer.step()
        
        return {
            'rl_loss': rl_loss.item() if isinstance(rl_loss, torch.Tensor) else 0.0,
            'demo_loss': demo_loss.item(),
            'total_loss': total_loss.item(),
            **{f'rl_{k}': v.item() for k, v in rl_loss_dict.items()}
        }


class InverseReinforcementLearning:
    """
    Inverse Reinforcement Learning from preferences.
    
    Learns a reward function from human preferences, then uses RL
    to optimize policy under that reward.
    """
    
    def __init__(self,
                 policy: nn.Module,
                 preference_dataset: PreferenceDataset,
                 state_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [64, 64],
                 learning_rate: float = 1e-4,
                 device: str = 'cuda'):
        """
        Args:
            policy: Policy network
            preference_dataset: Dataset of preferences
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden dimensions for reward model
            learning_rate: Learning rate
            device: Computation device
        """
        self.policy = policy.to(device)
        self.preference_dataset = preference_dataset
        self.device = device
        
        # Build reward model
        self.reward_model = self._build_reward_model(
            state_dim, action_dim, hidden_dims
        ).to(device)
        
        self.policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=learning_rate)
        
        self.dataloader = torch.utils.data.DataLoader(
            preference_dataset, batch_size=32, shuffle=True
        )
        
        self.reward_loss_history = []
    
    def _build_reward_model(self, state_dim: int, action_dim: int,
                           hidden_dims: List[int]) -> nn.Module:
        """Build neural network for reward prediction."""
        layers = []
        input_dim = state_dim + action_dim
        
        for hidden in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.ReLU())
            input_dim = hidden
        
        layers.append(nn.Linear(input_dim, 1))  # Single reward value
        
        return nn.Sequential(*layers)
    
    def train_reward_step(self) -> Dict[str, float]:
        """Train reward model on preferences."""
        self.reward_model.train()
        
        try:
            states, actions_a, actions_b, preferences = next(self.reward_iter)
        except (StopIteration, AttributeError):
            self.reward_iter = iter(self.dataloader)
            states, actions_a, actions_b, preferences = next(self.reward_iter)
        
        states = states.to(self.device)
        actions_a = actions_a.to(self.device)
        actions_b = actions_b.to(self.device)
        preferences = preferences.to(self.device)
        
        loss = preference_loss(self.reward_model, states, actions_a, actions_b, preferences)
        
        self.reward_optimizer.zero_grad()
        loss.backward()
        self.reward_optimizer.step()
        
        loss_val = loss.item()
        self.reward_loss_history.append(loss_val)
        
        return {'reward_loss': loss_val}
    
    def compute_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute reward for state-action pair."""
        sa = torch.cat([state, action], dim=-1)
        return self.reward_model(sa)
    
    def policy_gradient_step(self, env_samples: List[Dict]) -> Dict[str, float]:
        """
        Update policy using rewards from learned reward model.
        
        Args:
            env_samples: Batch of environment samples
        
        Returns:
            Loss metrics
        """
        states = torch.stack([s['state'] for s in env_samples]).to(self.device)
        actions = torch.stack([s['action'] for s in env_samples]).to(self.device)
        
        # Compute rewards using learned model
        with torch.no_grad():
            rewards = self.compute_reward(states, actions)
        
        # Simple policy gradient (would use proper RL algorithm in practice)
        log_probs = self.policy.get_log_prob(states, actions)
        loss = -(log_probs * rewards).mean()
        
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        return {'pg_loss': loss.item()}


# ======================================================================
# GR00T-Mimic Fine-tuning
# ======================================================================

class GR00TMimicFineTuner:
    """
    Main fine-tuning class for GR00T-Mimic.
    
    Combines multiple fine-tuning approaches within GRA framework.
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load policy
        self._load_policy()
        
        # Load datasets
        self._load_datasets()
        
        # Initialize fine-tuning algorithms
        self._init_algorithms()
        
        # Initialize GRA components
        self._init_gra()
        
        # Logging
        self.log_dir = self.config.get('log_dir', './logs')
        os.makedirs(self.log_dir, exist_ok=True)
        
        print(f"\n=== GR00T-Mimic Fine-tuning Initialized ===")
        print(f"Policy: {self.config.get('policy_type', 'gr00t')}")
        print(f"Dataset: {self.config.get('demos_path', '')}")
        print(f"Method: {self.config.get('method', 'bc')}")
        print(f"Device: {self.device}")
        print(f"Logging to: {self.log_dir}")
    
    def _load_policy(self):
        """Load policy model."""
        policy_type = self.config.get('policy_type', 'mlp')
        
        if policy_type == 'gr00t':
            from agents.gr00t_agent import Gr00tBaseWrapper
            self.policy = Gr00tBaseWrapper(
                name="gr00t_policy",
                model_path=self.config.get('gr00t_model_path'),
                device=self.device
            )
        elif policy_type == 'mlp':
            from agents.rl_agent import PPOAgent
            # Simplified - would need proper dimensions
            self.policy = nn.Sequential(
                nn.Linear(self.config.get('state_dim', 10), 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, self.config.get('action_dim', 6))
            )
        else:
            raise ValueError(f"Unknown policy type: {policy_type}")
    
    def _load_datasets(self):
        """Load demonstration datasets."""
        demos_path = self.config.get('demos_path')
        if demos_path and os.path.exists(demos_path):
            self.demo_dataset = DemonstrationDataset(
                demos_path, mode=self.config.get('method', 'bc')
            )
        else:
            print(f"Warning: Demo path {demos_path} not found")
            self.demo_dataset = None
        
        # Load preferences if available
        prefs_path = self.config.get('preferences_path')
        if prefs_path and os.path.exists(prefs_path):
            with open(prefs_path, 'r') as f:
                pref_data = json.load(f)
            self.preference_dataset = PreferenceDataset(pref_data.get('preferences', []))
        else:
            self.preference_dataset = None
    
    def _init_algorithms(self):
        """Initialize fine-tuning algorithms."""
        method = self.config.get('method', 'bc')
        
        if method == 'bc':
            if self.demo_dataset is None:
                raise ValueError("BC requires demonstration dataset")
            
            self.finetuner = BehavioralCloning(
                policy=self.policy,
                dataset=self.demo_dataset,
                learning_rate=self.config.get('learning_rate', 1e-4),
                batch_size=self.config.get('batch_size', 64),
                device=self.device
            )
        
        elif method == 'rl_demo':
            if self.demo_dataset is None:
                raise ValueError("RL+demo requires demonstration dataset")
            
            # Would need to create RL agent
            rl_agent = None  # Placeholder
            self.finetuner = ReinforcementLearningFineTuner(
                rl_agent=rl_agent,
                dataset=self.demo_dataset,
                demo_replay_ratio=self.config.get('demo_replay_ratio', 0.25),
                demo_replay_weight=self.config.get('demo_replay_weight', 1.0),
                device=self.device
            )
        
        elif method == 'irl':
            if self.preference_dataset is None:
                raise ValueError("IRL requires preference dataset")
            
            self.finetuner = InverseReinforcementLearning(
                policy=self.policy,
                preference_dataset=self.preference_dataset,
                state_dim=self.config.get('state_dim', 10),
                action_dim=self.config.get('action_dim', 6),
                hidden_dims=self.config.get('hidden_dims', [64, 64]),
                learning_rate=self.config.get('learning_rate', 1e-4),
                device=self.device
            )
    
    def _init_gra(self):
        """Initialize GRA components for safe fine-tuning."""
        self.use_gra = self.config.get('use_gra', False)
        
        if not self.use_gra:
            return
        
        # Create GRA layers
        self.g0 = G0_Layer(num_joints=self.config.get('num_joints', 6))
        self.g1 = G1_TaskLayer(g0_layer=self.g0)
        self.g2 = G2_SafetyLayer(g1_layer=self.g1)
        self.g3 = G3_EthicsLayer(g2_layer=self.g2)
        
        # Create multiverse
        self.multiverse = Multiverse(max_level=3)
        for layer in [self.g0, self.g1, self.g2, self.g3]:
            for idx, subsystem in layer.subsystems.items():
                self.multiverse.add_subsystem(subsystem)
        
        # Create zeroing algorithm
        self.zeroing = ZeroingAlgorithm(
            hierarchy=self.multiverse.subsystems,
            get_children=lambda x: [],
            get_parents=lambda x: [],
            get_goal_projector=lambda level: None,
            get_level_weight=lambda level: 1.0,
            level_tolerances=[0.01, 0.01, 0.001, 0.001],
            learning_rates=[0.01, 0.005, 0.001, 0.0005]
        )
    
    def train(self, num_steps: int = 10000):
        """Main training loop."""
        print(f"\nStarting fine-tuning for {num_steps} steps...")
        
        step = 0
        while step < num_steps:
            if hasattr(self.finetuner, 'train_step'):
                # Online training
                metrics = self.finetuner.train_step()
                step += 1
            elif hasattr(self.finetuner, 'train_epoch'):
                # Epoch-based training
                metrics = self.finetuner.train_epoch()
                step += len(self.finetuner.dataloader)
            
            # Logging
            if step % 100 == 0:
                self._log_progress(step, metrics)
            
            # Save checkpoint
            if step % 1000 == 0:
                self.save_checkpoint(step)
        
        print("\nFine-tuning complete!")
        self.save_checkpoint(num_steps, final=True)
    
    def _log_progress(self, step: int, metrics: Dict[str, float]):
        """Log training progress."""
        metric_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"Step {step:6d}: {metric_str}")
    
    def save_checkpoint(self, step: int, final: bool = False):
        """Save model checkpoint."""
        if final:
            path = os.path.join(self.log_dir, f"policy_final.pt")
        else:
            path = os.path.join(self.log_dir, f"policy_step_{step}.pt")
        
        if hasattr(self.policy, 'save'):
            self.policy.save(path)
        else:
            torch.save({
                'policy_state_dict': self.policy.state_dict(),
                'step': step
            }, path)
        
        print(f"Saved checkpoint to {path}")
    
    def evaluate(self, env, num_episodes: int = 10):
        """Evaluate fine-tuned policy."""
        self.policy.eval()
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            total_reward = 0
            length = 0
            
            while not done:
                if isinstance(obs, np.ndarray):
                    obs_tensor = torch.FloatTensor(obs).to(self.device)
                else:
                    obs_tensor = obs.to(self.device)
                
                with torch.no_grad():
                    action = self.policy(obs_tensor)
                
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                
                obs, reward, done, _ = env.step(action)
                total_reward += reward
                length += 1
            
            episode_rewards.append(total_reward)
            episode_lengths.append(length)
            
            print(f"Episode {ep+1}: reward={total_reward:.2f}, length={length}")
        
        return {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_length': np.mean(episode_lengths)
        }


# ======================================================================
# Command Line Interface
# ======================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Fine-tune policies with GR00T-Mimic')
    
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file')
    
    parser.add_argument('--demos', type=str,
                        help='Path to demonstrations file')
    
    parser.add_argument('--method', type=str, choices=['bc', 'rl_demo', 'irl'],
                        help='Fine-tuning method')
    
    parser.add_argument('--steps', type=int, default=10000,
                        help='Number of training steps')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    
    parser.add_argument('--device', type=str, default='cuda',
                        help='Computation device')
    
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Log directory')
    
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate after training')
    
    parser.add_argument('--eval-episodes', type=int, default=10,
                        help='Number of evaluation episodes')
    
    args = parser.parse_args()
    
    # Override config with command line args
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    if args.demos:
        config['demos_path'] = args.demos
    if args.method:
        config['method'] = args.method
    if args.lr:
        config['learning_rate'] = args.lr
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.device:
        config['device'] = args.device
    if args.log_dir:
        config['log_dir'] = args.log_dir
    
    # Create fine-tuner
    finetuner = GR00TMimicFineTuner(args.config)
    
    # Train
    finetuner.train(num_steps=args.steps)
    
    # Evaluate if requested
    if args.eval:
        from envs.pybullet_wrapper import CartPolePyBullet
        env = CartPolePyBullet(gui=False)
        results = finetuner.evaluate(env, num_episodes=args.eval_episodes)
        print(f"\nEvaluation results: {results}")
        env.close()


if __name__ == "__main__":
    main()
```