```python
"""
GRA Physical AI - Reinforcement Learning Agent Module
=====================================================

This module provides wrappers for Reinforcement Learning (RL) policies to integrate them
into the GRA framework as agents. RL agents are particularly important for:
    - Learning low-level motor control (G0)
    - Developing navigation policies (G1/G2)
    - Adapting to new tasks (G3)
    - Online zeroing and adaptation

The module supports:
    - Multiple RL algorithm types (PPO, SAC, DQN, etc.)
    - Integration with popular RL libraries (Stable-Baselines3, RLlib)
    - Differentiable policies for end-to-end zeroing
    - Experience replay integration for GRA
    - Policy distillation for hierarchical levels
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import warnings
import os
from collections import deque
import random

# Try to import common RL libraries
try:
    from stable_baselines3 import PPO, SAC, TD3, A2C, DQN
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.base_class import BaseAlgorithm
    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    warnings.warn("Stable-Baselines3 not installed. Install with: pip install stable-baselines3")

try:
    import gym
    import gym.spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

from ..core.base_agent import BaseAgent, DifferentiableAgentWrapper
from ..core.multiverse import MultiIndex


# ======================================================================
# Experience Replay Buffer (for GRA integration)
# ======================================================================

@dataclass
class Experience:
    """Single experience tuple for RL."""
    observation: torch.Tensor
    action: torch.Tensor
    reward: float
    next_observation: torch.Tensor
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)


class ReplayBuffer:
    """
    Experience replay buffer that integrates with GRA.
    
    Stores experiences and provides methods for:
        - Sampling batches for training
        - Computing statistics for foam calculation
        - Prioritized sampling based on GRA metrics
    """
    
    def __init__(self, capacity: int = 10000, prioritized: bool = False):
        """
        Args:
            capacity: Maximum number of experiences
            prioritized: Use prioritized replay (based on GRA foam)
        """
        self.capacity = capacity
        self.prioritized = prioritized
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: deque = deque(maxlen=capacity)
        self.position = 0
        
    def push(self, exp: Experience, priority: float = 1.0):
        """Add experience to buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
            self.priorities.append(priority)
        else:
            self.buffer[self.position] = exp
            self.priorities[self.position] = priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch of experiences."""
        if self.prioritized:
            # Sample based on priorities
            probs = np.array(self.priorities) / sum(self.priorities)
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            return [self.buffer[i] for i in indices]
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            return [self.buffer[i] for i in indices]
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized replay."""
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = prio
    
    def compute_gra_stats(self) -> Dict[str, float]:
        """Compute statistics useful for GRA foam calculation."""
        if len(self.buffer) == 0:
            return {}
        
        # Compute reward statistics
        rewards = [exp.reward for exp in self.buffer]
        
        # Compute action consistency (for foam)
        actions = torch.stack([exp.action for exp in self.buffer if hasattr(exp.action, 'shape')])
        if len(actions) > 0:
            action_mean = actions.mean(dim=0)
            action_std = actions.std(dim=0)
        else:
            action_mean = action_std = torch.tensor(0.0)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'buffer_size': len(self.buffer),
            'action_consistency': 1.0 / (1.0 + action_std.mean().item())
        }
    
    def __len__(self) -> int:
        return len(self.buffer)


# ======================================================================
# Base RL Agent
# ======================================================================

class RLAgent(BaseAgent):
    """
    Base class for Reinforcement Learning agents.
    
    Provides common interface for RL policies, supporting:
        - Action selection (deterministic/stochastic)
        - Learning from experience
        - Value estimation
        - Integration with GRA zeroing
    """
    
    def __init__(
        self,
        name: str,
        observation_space: Any,
        action_space: Any,
        policy_network: Optional[nn.Module] = None,
        value_network: Optional[nn.Module] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        device: str = 'cuda',
        **kwargs
    ):
        """
        Args:
            name: Agent name
            observation_space: Gym observation space
            action_space: Gym action space
            policy_network: Optional policy network
            value_network: Optional value network
            learning_rate: Learning rate
            gamma: Discount factor
            device: Computation device
            **kwargs: Additional arguments
        """
        # Determine dimensions from spaces
        if GYM_AVAILABLE:
            if isinstance(observation_space, gym.spaces.Box):
                obs_dim = observation_space.shape[0]
            elif isinstance(observation_space, gym.spaces.Discrete):
                obs_dim = observation_space.n
            else:
                obs_dim = kwargs.get('observation_dim', 10)
            
            if isinstance(action_space, gym.spaces.Box):
                act_dim = action_space.shape[0]
            elif isinstance(action_space, gym.spaces.Discrete):
                act_dim = action_space.n
            else:
                act_dim = kwargs.get('action_dim', 4)
        else:
            obs_dim = kwargs.get('observation_dim', 10)
            act_dim = kwargs.get('action_dim', 4)
        
        super().__init__(name, obs_dim, act_dim, device, **kwargs)
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Networks
        self.policy_network = policy_network
        self.value_network = value_network
        
        # Experience replay
        self.replay_buffer = ReplayBuffer(
            capacity=kwargs.get('buffer_capacity', 10000),
            prioritized=kwargs.get('prioritized_replay', False)
        )
        
        # Optimizer
        self.optimizer = None
        if policy_network is not None:
            params = list(policy_network.parameters())
            if value_network is not None:
                params.extend(value_network.parameters())
            self.optimizer = torch.optim.Adam(params, lr=learning_rate)
        
        # Training stats
        self.episode_rewards: List[float] = []
        self.current_episode_reward = 0.0
        self.training_steps = 0
        
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Select action using policy network.
        
        Args:
            observation: Current observation
            deterministic: If True, return mean/mode; else sample
        
        Returns:
            Action tensor
        """
        if self.policy_network is None:
            # Random action
            if isinstance(self.action_space, gym.spaces.Box):
                low = torch.tensor(self.action_space.low, device=self.device)
                high = torch.tensor(self.action_space.high, device=self.device)
                return low + (high - low) * torch.rand(self.action_dim, device=self.device)
            else:
                return torch.randint(0, self.action_dim, (1,), device=self.device)[0]
        
        observation = observation.to(self.device)
        
        with torch.set_grad_enabled(self.training):
            action = self.policy_network(observation)
            
            if not deterministic and hasattr(self.policy_network, 'sample'):
                action = self.policy_network.sample(observation)
        
        return action
    
    def learn(self, batch: List[Experience]) -> Dict[str, float]:
        """
        Perform one learning step from a batch of experiences.
        
        Args:
            batch: List of experiences
        
        Returns:
            Dictionary of loss metrics
        """
        if self.optimizer is None:
            return {}
        
        # To be implemented by subclasses
        raise NotImplementedError
    
    def store_experience(self, exp: Experience):
        """Store experience in replay buffer."""
        self.replay_buffer.push(exp)
        
        # Track episode reward
        self.current_episode_reward += exp.reward
        if exp.done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
    
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """
        Update policy using sampled batch from replay buffer.
        
        Returns:
            Loss metrics
        """
        if len(self.replay_buffer) < batch_size:
            return {}
        
        batch = self.replay_buffer.sample(batch_size)
        return self.learn(batch)
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return policy and value network parameters."""
        params = {}
        if self.policy_network is not None:
            for name, param in self.policy_network.named_parameters():
                params[f'policy.{name}'] = param
        if self.value_network is not None:
            for name, param in self.value_network.named_parameters():
                params[f'value.{name}'] = param
        return params
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set network parameters."""
        if self.policy_network is not None:
            policy_params = {k.replace('policy.', ''): v 
                           for k, v in params.items() if k.startswith('policy.')}
            for name, param in self.policy_network.named_parameters():
                if name in policy_params:
                    param.data.copy_(policy_params[name].to(self.device))
        
        if self.value_network is not None:
            value_params = {k.replace('value.', ''): v 
                          for k, v in params.items() if k.startswith('value.')}
            for name, param in self.value_network.named_parameters():
                if name in value_params:
                    param.data.copy_(value_params[name].to(self.device))
    
    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """Return internal state (e.g., LSTM hidden states)."""
        return {}
    
    def set_internal_state(self, state: Dict[str, torch.Tensor]):
        """Set internal state."""
        pass
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get performance metrics for GRA foam calculation."""
        metrics = {
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0.0,
            'training_steps': self.training_steps,
        }
        
        # Add replay buffer stats
        buffer_stats = self.replay_buffer.compute_gra_stats()
        metrics.update(buffer_stats)
        
        return metrics


# ======================================================================
# PPO Agent (Proximal Policy Optimization)
# ======================================================================

class PPOAgent(RLAgent):
    """
    Proximal Policy Optimization agent.
    
    Implements the PPO algorithm with clipped surrogate objective.
    """
    
    def __init__(
        self,
        name: str,
        observation_space: Any,
        action_space: Any,
        policy_network: Optional[nn.Module] = None,
        value_network: Optional[nn.Module] = None,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        **kwargs
    ):
        """
        Args:
            name: Agent name
            observation_space: Gym observation space
            action_space: Gym action space
            policy_network: Policy network (should output mean and log_std)
            value_network: Value network
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            **kwargs: Additional arguments
        """
        super().__init__(name, observation_space, action_space, 
                        policy_network, value_network, **kwargs)
        
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # For storing rollouts
        self.rollout_buffer: List[Experience] = []
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Select action and store log probability."""
        observation = observation.to(self.device)
        
        if self.policy_network is None:
            return super().act(observation, deterministic)
        
        with torch.set_grad_enabled(self.training):
            if hasattr(self.policy_network, 'get_action'):
                # Network returns (action, log_prob)
                action, log_prob = self.policy_network.get_action(observation, deterministic)
                self._internal_state['last_log_prob'] = log_prob
            else:
                # Simple network: just action
                action = self.policy_network(observation)
                self._internal_state['last_log_prob'] = torch.tensor(0.0)
        
        return action
    
    def learn(self, batch: List[Experience]) -> Dict[str, float]:
        """PPO update step."""
        if len(batch) == 0:
            return {}
        
        # Prepare batch tensors
        obs = torch.stack([exp.observation for exp in batch]).to(self.device)
        actions = torch.stack([exp.action for exp in batch]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], device=self.device)
        next_obs = torch.stack([exp.next_observation for exp in batch]).to(self.device)
        dones = torch.tensor([exp.done for exp in batch], device=self.device, dtype=torch.float32)
        
        # Compute advantages (simplified GAE)
        with torch.no_grad():
            if self.value_network is not None:
                values = self.value_network(obs).squeeze()
                next_values = self.value_network(next_obs).squeeze()
            else:
                values = torch.zeros_like(rewards)
                next_values = torch.zeros_like(rewards)
        
        # TD error
        td_target = rewards + self.gamma * next_values * (1 - dones)
        advantages = td_target - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get action log probs
        if hasattr(self.policy_network, 'evaluate_actions'):
            log_probs, entropy = self.policy_network.evaluate_actions(obs, actions)
        else:
            # Simplified: assume Gaussian
            action_mean = self.policy_network(obs)
            log_probs = -0.5 * ((actions - action_mean) ** 2).sum(dim=1)
            entropy = torch.tensor(0.0)
        
        # PPO clipped objective
        ratio = torch.exp(log_probs - self._internal_state.get('last_log_prob', log_probs.detach()))
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        if self.value_network is not None:
            value_pred = self.value_network(obs).squeeze()
            value_loss = nn.MSELoss()(value_pred, td_target)
        else:
            value_loss = torch.tensor(0.0)
        
        # Entropy bonus
        entropy_loss = -entropy.mean() * self.entropy_coef if entropy != 0 else torch.tensor(0.0)
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss + entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.get_parameters_list(), self.max_grad_norm)
        self.optimizer.step()
        
        self.training_steps += 1
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item() if isinstance(value_loss, torch.Tensor) else 0.0,
            'entropy': entropy.mean().item() if entropy != 0 else 0.0,
            'total_loss': total_loss.item()
        }
    
    def get_parameters_list(self) -> List[torch.Tensor]:
        """Get list of all parameters for gradient clipping."""
        params = []
        if self.policy_network is not None:
            params.extend(self.policy_network.parameters())
        if self.value_network is not None:
            params.extend(self.value_network.parameters())
        return params


# ======================================================================
# SAC Agent (Soft Actor-Critic)
# ======================================================================

class SACAgent(RLAgent):
    """
    Soft Actor-Critic agent.
    
    Implements maximum entropy RL for stable learning.
    """
    
    def __init__(
        self,
        name: str,
        observation_space: Any,
        action_space: Any,
        policy_network: Optional[nn.Module] = None,
        q_network1: Optional[nn.Module] = None,
        q_network2: Optional[nn.Module] = None,
        value_network: Optional[nn.Module] = None,
        target_value_network: Optional[nn.Module] = None,
        alpha: float = 0.2,
        tau: float = 0.005,
        target_update_interval: int = 1,
        **kwargs
    ):
        """
        Args:
            name: Agent name
            observation_space: Gym observation space
            action_space: Gym action space
            policy_network: Policy network
            q_network1: First Q-network
            q_network2: Second Q-network (for double Q)
            value_network: Value network
            target_value_network: Target value network
            alpha: Temperature parameter for entropy
            tau: Soft update coefficient
            target_update_interval: How often to update target network
            **kwargs: Additional arguments
        """
        super().__init__(name, observation_space, action_space,
                        policy_network, value_network, **kwargs)
        
        self.q_network1 = q_network1
        self.q_network2 = q_network2
        self.target_value_network = target_value_network or value_network
        self.alpha = alpha
        self.tau = tau
        self.target_update_interval = target_update_interval
        
        # Create optimizers for Q networks
        if q_network1 is not None:
            self.q_optimizer1 = torch.optim.Adam(q_network1.parameters(), lr=learning_rate)
        if q_network2 is not None:
            self.q_optimizer2 = torch.optim.Adam(q_network2.parameters(), lr=learning_rate)
    
    def learn(self, batch: List[Experience]) -> Dict[str, float]:
        """SAC update step."""
        if len(batch) == 0:
            return {}
        
        # Prepare batch
        obs = torch.stack([exp.observation for exp in batch]).to(self.device)
        actions = torch.stack([exp.action for exp in batch]).to(self.device)
        rewards = torch.tensor([exp.reward for exp in batch], device=self.device).unsqueeze(1)
        next_obs = torch.stack([exp.next_observation for exp in batch]).to(self.device)
        dones = torch.tensor([exp.done for exp in batch], device=self.device, dtype=torch.float32).unsqueeze(1)
        
        # Update Q networks
        with torch.no_grad():
            # Target value
            if self.target_value_network is not None:
                next_v = self.target_value_network(next_obs)
            else:
                # Use current value network
                next_v = self.value_network(next_obs) if self.value_network else torch.zeros_like(rewards)
            
            # Target Q value
            target_q = rewards + self.gamma * (1 - dones) * next_v
        
        # Q1 update
        if self.q_network1 is not None:
            q1 = self.q_network1(obs, actions)
            q1_loss = nn.MSELoss()(q1, target_q)
            
            self.q_optimizer1.zero_grad()
            q1_loss.backward()
            self.q_optimizer1.step()
        else:
            q1_loss = torch.tensor(0.0)
        
        # Q2 update
        if self.q_network2 is not None:
            q2 = self.q_network2(obs, actions)
            q2_loss = nn.MSELoss()(q2, target_q)
            
            self.q_optimizer2.zero_grad()
            q2_loss.backward()
            self.q_optimizer2.step()
        else:
            q2_loss = torch.tensor(0.0)
        
        # Policy update (delayed)
        if self.training_steps % 2 == 0 and self.policy_network is not None:
            # Sample new actions from policy
            if hasattr(self.policy_network, 'sample'):
                new_actions, log_probs = self.policy_network.sample(obs)
            else:
                new_actions = self.policy_network(obs)
                log_probs = torch.zeros_like(new_actions)
            
            # Q value of new actions (use min of two Qs)
            if self.q_network1 is not None and self.q_network2 is not None:
                q1_new = self.q_network1(obs, new_actions)
                q2_new = self.q_network2(obs, new_actions)
                q_new = torch.min(q1_new, q2_new)
            else:
                q_new = torch.zeros_like(rewards)
            
            # Policy loss: α * log π - Q
            policy_loss = (self.alpha * log_probs - q_new).mean()
            
            # Update policy
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
        
        # Update target value network
        if self.training_steps % self.target_update_interval == 0:
            if self.target_value_network is not None and self.value_network is not None:
                for target_param, param in zip(self.target_value_network.parameters(),
                                               self.value_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.training_steps += 1
        
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'policy_loss': policy_loss.item() if self.training_steps % 2 == 0 else 0.0
        }


# ======================================================================
# Stable-Baselines3 Wrapper
# ======================================================================

class SB3Agent(RLAgent):
    """
    Wrapper for Stable-Baselines3 agents.
    
    Allows using trained SB3 models within GRA.
    """
    
    def __init__(
        self,
        name: str,
        sb3_model: Union[BaseAlgorithm, str],
        observation_space: Optional[Any] = None,
        action_space: Optional[Any] = None,
        device: str = 'cuda',
        **kwargs
    ):
        """
        Args:
            name: Agent name
            sb3_model: SB3 model or path to saved model
            observation_space: Gym observation space (if loading from path)
            action_space: Gym action space (if loading from path)
            device: Computation device
            **kwargs: Additional arguments
        """
        if not SB3_AVAILABLE:
            raise ImportError("Stable-Baselines3 not installed")
        
        # Load model if path provided
        if isinstance(sb3_model, str):
            self.model = BaseAlgorithm.load(sb3_model)
        else:
            self.model = sb3_model
        
        # Get spaces from model
        if observation_space is None:
            observation_space = self.model.observation_space
        if action_space is None:
            action_space = self.model.action_space
        
        super().__init__(name, observation_space, action_space, device=device, **kwargs)
        
        # Set to eval mode
        self.model.policy.eval()
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Use SB3 model to predict action."""
        # Convert to numpy for SB3
        if isinstance(observation, torch.Tensor):
            obs_np = observation.cpu().numpy()
        else:
            obs_np = np.array(observation)
        
        # Predict
        action, _ = self.model.predict(obs_np, deterministic=deterministic)
        
        return torch.tensor(action, device=self.device)
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract parameters from SB3 model."""
        params = {}
        for name, param in self.model.policy.named_parameters():
            params[f'sb3.{name}'] = param.data
        return params
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set parameters in SB3 model (use with caution)."""
        for name, param in self.model.policy.named_parameters():
            key = f'sb3.{name}'
            if key in params:
                param.data.copy_(params[key].cpu())


# ======================================================================
# Hierarchical RL Agent
# ======================================================================

class HierarchicalRLAgent(RLAgent):
    """
    Hierarchical RL agent with multiple levels.
    
    Combines multiple RL agents at different levels, matching GRA hierarchy.
    """
    
    def __init__(
        self,
        name: str,
        levels: Dict[int, RLAgent],
        coordinator: Optional[RLAgent] = None,
        **kwargs
    ):
        """
        Args:
            name: Agent name
            levels: Dictionary mapping level -> RLAgent
            coordinator: Optional meta-controller
            **kwargs: Additional arguments
        """
        # Use first level's observation/action spaces
        first_level = min(levels.keys())
        first_agent = levels[first_level]
        
        super().__init__(
            name,
            first_agent.observation_space,
            first_agent.action_space,
            **kwargs
        )
        
        self.levels = levels
        self.coordinator = coordinator
        self.current_level = first_level
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Select action using appropriate level."""
        # Coordinator decides which level to use
        if self.coordinator is not None:
            level_choice = self.coordinator.act(observation, deterministic)
            if isinstance(level_choice, torch.Tensor) and level_choice.numel() == 1:
                self.current_level = int(level_choice.item())
        
        # Use selected level's agent
        if self.current_level in self.levels:
            return self.levels[self.current_level].act(observation, deterministic)
        else:
            return super().act(observation, deterministic)
    
    def learn(self, batch: List[Experience]) -> Dict[str, float]:
        """Learn in all levels."""
        losses = {}
        for level, agent in self.levels.items():
            level_losses = agent.learn(batch)
            for k, v in level_losses.items():
                losses[f'level{level}_{k}'] = v
        
        if self.coordinator is not None:
            coord_losses = self.coordinator.learn(batch)
            for k, v in coord_losses.items():
                losses[f'coord_{k}'] = v
        
        return losses
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Get parameters from all levels."""
        params = {}
        for level, agent in self.levels.items():
            level_params = agent.get_parameters()
            for name, param in level_params.items():
                params[f'level{level}.{name}'] = param
        
        if self.coordinator is not None:
            coord_params = self.coordinator.get_parameters()
            for name, param in coord_params.items():
                params[f'coord.{name}'] = param
        
        return params
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set parameters for all levels."""
        for level, agent in self.levels.items():
            level_params = {k.replace(f'level{level}.', ''): v 
                          for k, v in params.items() if k.startswith(f'level{level}.')}
            agent.set_parameters(level_params)
        
        if self.coordinator is not None:
            coord_params = {k.replace('coord.', ''): v 
                          for k, v in params.items() if k.startswith('coord.')}
            self.coordinator.set_parameters(coord_params)


# ======================================================================
# Utility Functions
# ======================================================================

def create_rl_agent(
    level: int,
    name: str,
    algorithm: str = 'ppo',
    observation_space: Optional[Any] = None,
    action_space: Optional[Any] = None,
    **kwargs
) -> RLAgent:
    """
    Factory function to create RL agent for specific GRA level.
    
    Args:
        level: GRA level (0-4)
        name: Agent name
        algorithm: 'ppo', 'sac', 'sb3', etc.
        observation_space: Gym observation space
        action_space: Gym action space
        **kwargs: Additional arguments
    
    Returns:
        Configured RL agent
    """
    # Create simple networks if not provided
    if 'policy_network' not in kwargs and observation_space is not None and action_space is not None:
        if GYM_AVAILABLE:
            if isinstance(observation_space, gym.spaces.Box):
                obs_dim = observation_space.shape[0]
            else:
                obs_dim = 10
            
            if isinstance(action_space, gym.spaces.Box):
                act_dim = action_space.shape[0]
                is_discrete = False
            else:
                act_dim = action_space.n
                is_discrete = True
            
            # Simple MLP policy
            class PolicyNetwork(nn.Module):
                def __init__(self, obs_dim, act_dim, discrete):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(obs_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU()
                    )
                    self.mean = nn.Linear(256, act_dim)
                    self.log_std = nn.Parameter(torch.zeros(act_dim))
                    self.discrete = discrete
                
                def forward(self, x):
                    features = self.net(x)
                    return self.mean(features)
                
                def get_action(self, x, deterministic):
                    mean = self.forward(x)
                    if deterministic:
                        return mean, torch.tensor(0.0)
                    std = torch.exp(self.log_std)
                    dist = torch.distributions.Normal(mean, std)
                    action = dist.rsample()
                    log_prob = dist.log_prob(action).sum(dim=-1)
                    return action, log_prob
                
                def evaluate_actions(self, x, actions):
                    mean = self.forward(x)
                    std = torch.exp(self.log_std)
                    dist = torch.distributions.Normal(mean, std)
                    log_probs = dist.log_prob(actions).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)
                    return log_probs, entropy
            
            kwargs['policy_network'] = PolicyNetwork(obs_dim, act_dim, is_discrete)
            
            # Value network
            class ValueNetwork(nn.Module):
                def __init__(self, obs_dim):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(obs_dim, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 1)
                    )
                
                def forward(self, x):
                    return self.net(x)
            
            kwargs['value_network'] = ValueNetwork(obs_dim)
    
    # Create agent based on algorithm
    if algorithm.lower() == 'ppo':
        return PPOAgent(name, observation_space, action_space, **kwargs)
    elif algorithm.lower() == 'sac':
        return SACAgent(name, observation_space, action_space, **kwargs)
    elif algorithm.lower() == 'sb3':
        # Need to provide SB3 model
        if 'sb3_model' in kwargs:
            return SB3Agent(name, kwargs.pop('sb3_model'), observation_space, action_space, **kwargs)
        else:
            raise ValueError("SB3 agent requires 'sb3_model' argument")
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def rl_to_subsystem(agent: RLAgent, multi_index: MultiIndex) -> Any:
    """
    Convert RL agent to GRA subsystem.
    """
    from ..core.subsystem import Subsystem
    
    class RLSubsystem(Subsystem):
        def __init__(self, agent, multi_index):
            super().__init__(multi_index, None, None)
            self.agent = agent
        
        def get_state(self):
            return self.agent.get_state()
        
        def set_state(self, state):
            self.agent.set_state(state)
        
        def step(self, dt, action=None):
            # RL agent stepping is handled by environment
            pass
        
        def get_metrics(self):
            return self.agent.get_performance_metrics()
    
    return RLSubsystem(agent, multi_index)


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing RL Agents ===\n")
    
    if not GYM_AVAILABLE:
        print("Gym not available. Skipping tests.")
        exit()
    
    # Create simple environment
    env = gym.make('CartPole-v1')
    
    # Test PPO agent
    print("Creating PPO agent...")
    ppo_agent = create_rl_agent(
        level=0,
        name="ppo_test",
        algorithm='ppo',
        observation_space=env.observation_space,
        action_space=env.action_space,
        device='cpu'
    )
    
    # Test acting
    obs = env.reset()
    obs_tensor = torch.tensor(obs, dtype=torch.float32)
    action = ppo_agent.act(obs_tensor)
    print(f"  Action shape: {action.shape}")
    
    # Test experience storage
    next_obs, reward, done, _ = env.step(action.item() if action.numel() == 1 else action.numpy())
    exp = Experience(
        observation=obs_tensor,
        action=action,
        reward=reward,
        next_observation=torch.tensor(next_obs, dtype=torch.float32),
        done=done
    )
    ppo_agent.store_experience(exp)
    print(f"  Replay buffer size: {len(ppo_agent.replay_buffer)}")
    
    # Test learning
    for _ in range(10):
        obs = env.reset()
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        for _ in range(100):
            action = ppo_agent