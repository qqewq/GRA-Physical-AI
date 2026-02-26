```python
"""
GRA Physical AI - Base Agent Class
==================================

This module defines the foundational abstract base class for all agents in the GRA framework.
An agent is any subsystem that generates actions based on observations and maintains
learnable parameters.

The BaseAgent class provides:
    - Standard interface for act, learn, get_state, set_state
    - Integration with the GRA multiverse (state as parameters + internal state)
    - Support for differentiable and non-differentiable agents
    - Serialization and device management
    - Training/evaluation mode switching

All concrete agents (neural networks, classical controllers, hierarchical)
should inherit from this class.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import warnings


# ======================================================================
# Base Agent Class
# ======================================================================

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the GRA framework.
    
    An agent is a component that:
        - Receives observations from the environment
        - Produces actions
        - Maintains internal state (parameters, memory, beliefs)
        - Can be trained/optimized
        - Can be placed in a GRA multiverse as a subsystem
    
    The agent's "state" in GRA terms is a concatenation of:
        - All learnable parameters (flattened)
        - Any internal state (e.g., RNN hidden states, integrator values)
    
    This state is what gets zeroed during the GRA process.
    """
    
    def __init__(
        self,
        name: str,
        observation_dim: int,
        action_dim: int,
        device: Union[str, torch.device] = 'cpu',
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize a base agent.
        
        Args:
            name: Unique identifier for this agent
            observation_dim: Dimension of observations
            action_dim: Dimension of actions
            device: Computation device ('cpu', 'cuda', etc.)
            dtype: Data type for tensors
        """
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.device = torch.device(device)
        self.dtype = dtype
        
        # Training mode
        self.training = True
        
        # Metadata storage
        self.metadata: Dict[str, Any] = {}
        
        # Statistics (can be used for logging)
        self.stats: Dict[str, List[float]] = {
            'actions': [],
            'losses': [],
            'grad_norms': []
        }
        
    # ======================================================================
    # Core abstract methods
    # ======================================================================
    
    @abstractmethod
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Produce an action given an observation.
        
        This is the forward pass of the agent. For stochastic policies,
        `deterministic=False` enables sampling.
        
        Args:
            observation: Input observation tensor.
                Shape can be (obs_dim,) for single observation or
                (batch_size, obs_dim) for batched observations.
            deterministic: If False and agent is stochastic, sample from distribution.
                           If True, return mean/mode.
        
        Returns:
            Action tensor of shape (action_dim,) or (batch_size, action_dim)
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Return all trainable parameters as a dictionary mapping names to tensors.
        
        This is used to extract the agent's state for the GRA multiverse.
        The keys should be consistent across calls.
        
        Returns:
            Dictionary of parameter name -> parameter tensor
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """
        Set trainable parameters from a dictionary.
        
        This is used during zeroing to update the agent's state.
        
        Args:
            params: Dictionary of parameter name -> new tensor values
        """
        pass
    
    @abstractmethod
    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """
        Return internal state (non-parameter, e.g., RNN hidden states).
        
        This is part of the agent's full state in the multiverse.
        
        Returns:
            Dictionary of state name -> state tensor
        """
        pass
    
    @abstractmethod
    def set_internal_state(self, state: Dict[str, torch.Tensor]):
        """
        Set internal state from a dictionary.
        
        Args:
            state: Dictionary of state name -> new state tensor
        """
        pass
    
    # ======================================================================
    # Concrete methods for GRA integration
    # ======================================================================
    
    def get_state(self) -> torch.Tensor:
        """
        Get the full agent state as a single flattened tensor.
        
        This is what the GRA multiverse uses as the subsystem state.
        The state is the concatenation of:
            - All parameters (flattened)
            - All internal state variables (flattened)
        
        Returns:
            1D tensor containing all state information
        """
        # Get parameters
        params = self.get_parameters()
        param_list = [p.flatten() for p in params.values()]
        
        # Get internal state
        internal = self.get_internal_state()
        internal_list = [s.flatten() for s in internal.values()]
        
        # Concatenate
        if param_list or internal_list:
            return torch.cat(param_list + internal_list)
        else:
            # No state – return empty tensor
            return torch.tensor([], device=self.device, dtype=self.dtype)
    
    def set_state(self, state_vector: torch.Tensor):
        """
        Set the full agent state from a flattened tensor.
        
        This is the inverse of `get_state()`. The state vector must have
        the same length and structure as returned by `get_state()`.
        
        Args:
            state_vector: 1D tensor containing all state information
        """
        # Get parameter and internal state structures
        params = self.get_parameters()
        internal = self.get_internal_state()
        
        # Calculate sizes
        param_sizes = {name: p.numel() for name, p in params.items()}
        internal_sizes = {name: s.numel() for name, s in internal.items()}
        
        total_size = sum(param_sizes.values()) + sum(internal_sizes.values())
        
        if len(state_vector) != total_size:
            raise ValueError(
                f"State vector length {len(state_vector)} does not match "
                f"expected size {total_size}"
            )
        
        # Extract parameters
        idx = 0
        new_params = {}
        for name, size in param_sizes.items():
            new_params[name] = state_vector[idx:idx+size].reshape(params[name].shape)
            idx += size
        
        # Extract internal state
        new_internal = {}
        for name, size in internal_sizes.items():
            new_internal[name] = state_vector[idx:idx+size].reshape(internal[name].shape)
            idx += size
        
        # Set them
        self.set_parameters(new_params)
        self.set_internal_state(new_internal)
    
    def state_size(self) -> int:
        """
        Return the total dimension of the agent's state vector.
        
        Useful for allocating spaces in the multiverse.
        """
        params = self.get_parameters()
        internal = self.get_internal_state()
        
        return sum(p.numel() for p in params.values()) + \
               sum(s.numel() for s in internal.values())
    
    # ======================================================================
    # Training utilities
    # ======================================================================
    
    def train_mode(self, mode: bool = True):
        """
        Set training mode.
        
        Some agents (e.g., dropout, batch norm) behave differently
        during training vs evaluation.
        
        Args:
            mode: True for training, False for evaluation
        """
        self.training = mode
        
        # If agent is a PyTorch module, set its mode
        if isinstance(self, torch.nn.Module):
            self.train(mode)
    
    def zero_grad(self):
        """Zero out gradients of all parameters."""
        for param in self.get_parameters().values():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
    
    def get_gradients(self) -> Dict[str, torch.Tensor]:
        """Return gradients of all parameters (if available)."""
        grads = {}
        for name, param in self.get_parameters().items():
            if param.grad is not None:
                grads[name] = param.grad.clone()
        return grads
    
    def apply_gradients(self, grads: Dict[str, torch.Tensor], lr: float = 0.01):
        """
        Apply gradients manually (simple SGD).
        
        Args:
            grads: Dictionary of gradient tensors
            lr: Learning rate
        """
        params = self.get_parameters()
        for name, grad in grads.items():
            if name in params:
                params[name].data = params[name].data - lr * grad.to(params[name].device)
    
    # ======================================================================
    # Loss functions (can be overridden)
    # ======================================================================
    
    def compute_loss(self, observation: torch.Tensor, 
                     action: torch.Tensor,
                     target: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute a loss for the agent.
        
        This is used during zeroing for level 0 local optimization.
        Default implementation returns 0 (no local loss).
        Override for task-specific losses.
        
        Args:
            observation: Input observation
            action: Action taken (or to evaluate)
            target: Optional target (e.g., desired action)
        
        Returns:
            Scalar loss tensor
        """
        if target is not None:
            # Simple MSE between action and target
            return torch.nn.functional.mse_loss(action, target)
        return torch.tensor(0.0, device=self.device)
    
    # ======================================================================
    # Serialization
    # ======================================================================
    
    def save(self, path: str):
        """
        Save agent to file.
        
        Args:
            path: Path to save file
        """
        data = {
            'name': self.name,
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'device': str(self.device),
            'dtype': str(self.dtype),
            'metadata': self.metadata,
            'parameters': self.get_parameters(),
            'internal_state': self.get_internal_state(),
            'stats': self.stats
        }
        
        torch.save(data, path)
    
    def load(self, path: str):
        """
        Load agent from file.
        
        Args:
            path: Path to load file
        """
        data = torch.load(path, map_location=self.device)
        
        # Check compatibility
        assert data['name'] == self.name, f"Name mismatch: {data['name']} vs {self.name}"
        assert data['observation_dim'] == self.observation_dim
        assert data['action_dim'] == self.action_dim
        
        self.metadata = data.get('metadata', {})
        self.stats = data.get('stats', {'actions': [], 'losses': [], 'grad_norms': []})
        
        self.set_parameters(data['parameters'])
        self.set_internal_state(data.get('internal_state', {}))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary (for JSON serialization)."""
        return {
            'class': self.__class__.__name__,
            'name': self.name,
            'observation_dim': self.observation_dim,
            'action_dim': self.action_dim,
            'device': str(self.device),
            'dtype': str(self.dtype),
            'metadata': self.metadata,
            'state_size': self.state_size()
        }
    
    # ======================================================================
    # Device management
    # ======================================================================
    
    def to(self, device: Union[str, torch.device]) -> 'BaseAgent':
        """
        Move agent to a different device.
        
        Args:
            device: Target device
        
        Returns:
            self (for chaining)
        """
        self.device = torch.device(device)
        
        # Move parameters
        params = self.get_parameters()
        for name, param in params.items():
            params[name] = param.to(self.device)
        self.set_parameters(params)
        
        # Move internal state
        internal = self.get_internal_state()
        for name, state in internal.items():
            internal[name] = state.to(self.device)
        self.set_internal_state(internal)
        
        return self
    
    def cpu(self) -> 'BaseAgent':
        """Move agent to CPU."""
        return self.to('cpu')
    
    def cuda(self, device: Optional[int] = None) -> 'BaseAgent':
        """Move agent to CUDA."""
        return self.to(f'cuda:{device}' if device is not None else 'cuda')
    
    # ======================================================================
    # Statistics and logging
    # ======================================================================
    
    def log_action(self, action: torch.Tensor):
        """Record an action for statistics."""
        if hasattr(action, 'item'):
            self.stats['actions'].append(action.item())
        # Keep only last 1000
        if len(self.stats['actions']) > 1000:
            self.stats['actions'] = self.stats['actions'][-1000:]
    
    def log_loss(self, loss: float):
        """Record a loss value."""
        self.stats['losses'].append(loss)
        if len(self.stats['losses']) > 1000:
            self.stats['losses'] = self.stats['losses'][-1000:]
    
    def log_grad_norm(self, norm: float):
        """Record gradient norm."""
        self.stats['grad_norms'].append(norm)
        if len(self.stats['grad_norms']) > 1000:
            self.stats['grad_norms'] = self.stats['grad_norms'][-1000:]
    
    def get_statistics(self) -> Dict[str, float]:
        """Compute summary statistics."""
        stats = {}
        if self.stats['actions']:
            actions = torch.tensor(self.stats['actions'])
            stats['action_mean'] = actions.mean().item()
            stats['action_std'] = actions.std().item()
        
        if self.stats['losses']:
            losses = torch.tensor(self.stats['losses'])
            stats['loss_mean'] = losses.mean().item()
            stats['loss_min'] = losses.min().item()
            stats['loss_max'] = losses.max().item()
        
        if self.stats['grad_norms']:
            norms = torch.tensor(self.stats['grad_norms'])
            stats['grad_norm_mean'] = norms.mean().item()
            stats['grad_norm_max'] = norms.max().item()
        
        return stats
    
    def reset_statistics(self):
        """Clear accumulated statistics."""
        self.stats = {'actions': [], 'losses': [], 'grad_norms': []}
    
    # ======================================================================
    # Magic methods
    # ======================================================================
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"obs_dim={self.observation_dim}, "
            f"act_dim={self.action_dim}, "
            f"state_size={self.state_size()}, "
            f"device={self.device})"
        )
    
    def __str__(self) -> str:
        return self.__repr__()


# ======================================================================
# Wrapper for non-differentiable agents
# ======================================================================

class DifferentiableAgentWrapper(BaseAgent):
    """
    Wrapper that makes any agent differentiable via straight-through estimators.
    
    This allows non-differentiable agents (e.g., classical controllers) to be
    used in gradient-based zeroing. The wrapper uses the identity function
    in the backward pass.
    """
    
    def __init__(self, base_agent: BaseAgent):
        """
        Args:
            base_agent: The agent to wrap (may be non-differentiable)
        """
        super().__init__(
            name=f"diff_{base_agent.name}",
            observation_dim=base_agent.observation_dim,
            action_dim=base_agent.action_dim,
            device=base_agent.device,
            dtype=base_agent.dtype
        )
        self.base_agent = base_agent
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Forward pass with straight-through gradient estimator.
        
        If the observation requires gradients, we detach the base agent's
        action and add a residual connection that passes gradients through.
        """
        observation = observation.to(self.device)
        
        # Forward pass through base agent (detached)
        with torch.no_grad():
            action_base = self.base_agent.act(observation, deterministic)
        
        # If we need gradients, use straight-through estimator
        if observation.requires_grad:
            # action = observation + (action_base - observation).detach()
            # This makes ∂action/∂observation = I
            action = observation[..., :self.action_dim] + \
                     (action_base - observation[..., :self.action_dim]).detach()
        else:
            action = action_base
        
        return action
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return base agent's parameters."""
        return self.base_agent.get_parameters()
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set base agent's parameters."""
        self.base_agent.set_parameters(params)
    
    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """Return base agent's internal state."""
        return self.base_agent.get_internal_state()
    
    def set_internal_state(self, state: Dict[str, torch.Tensor]):
        """Set base agent's internal state."""
        self.base_agent.set_internal_state(state)
    
    def train_mode(self, mode: bool = True):
        """Set training mode for base agent."""
        super().train_mode(mode)
        self.base_agent.train_mode(mode)
    
    def to(self, device: Union[str, torch.device]) -> 'DifferentiableAgentWrapper':
        """Move both wrapper and base agent to device."""
        super().to(device)
        self.base_agent.to(device)
        return self


# ======================================================================
# Null agent (for testing)
# ======================================================================

class NullAgent(BaseAgent):
    """
    Agent that does nothing (returns zero actions).
    Useful for testing and as a placeholder.
    """
    
    def __init__(self, name: str, observation_dim: int, action_dim: int, **kwargs):
        super().__init__(name, observation_dim, action_dim, **kwargs)
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Return zero actions."""
        if observation.dim() == 1:
            return torch.zeros(self.action_dim, device=self.device, dtype=self.dtype)
        else:
            return torch.zeros(observation.shape[0], self.action_dim, 
                              device=self.device, dtype=self.dtype)
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """No parameters."""
        return {}
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Nothing to set."""
        pass
    
    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """No internal state."""
        return {}
    
    def set_internal_state(self, state: Dict[str, torch.Tensor]):
        """Nothing to set."""
        pass


# ======================================================================
# Agent registry (for factory pattern)
# ======================================================================

class AgentRegistry:
    """
    Registry for creating agents by name.
    Useful for loading from configuration files.
    """
    
    _registry: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an agent class."""
        def decorator(agent_class):
            cls._registry[name] = agent_class
            return agent_class
        return decorator
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseAgent:
        """
        Create an agent from a configuration dictionary.
        
        Example config:
            {
                'type': 'MLPAgent',
                'name': 'policy',
                'observation_dim': 10,
                'action_dim': 4,
                'hidden_dims': [256, 256]
            }
        """
        agent_type = config.pop('type')
        if agent_type not in cls._registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
        
        agent_class = cls._registry[agent_type]
        return agent_class(**config)


# ======================================================================
# Example usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing BaseAgent ===\n")
    
    # Create a simple test agent
    class TestAgent(BaseAgent):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.W = torch.nn.Parameter(torch.randn(self.observation_dim, self.action_dim))
            self.b = torch.nn.Parameter(torch.zeros(self.action_dim))
            self.hidden = torch.zeros(self.action_dim)  # internal state
        
        def act(self, observation, deterministic=True):
            return observation @ self.W + self.b
        
        def get_parameters(self):
            return {'W': self.W, 'b': self.b}
        
        def set_parameters(self, params):
            self.W.data.copy_(params['W'])
            self.b.data.copy_(params['b'])
        
        def get_internal_state(self):
            return {'hidden': self.hidden}
        
        def set_internal_state(self, state):
            self.hidden = state['hidden']
    
    agent = TestAgent(
        name="test",
        observation_dim=5,
        action_dim=2,
        device='cpu'
    )
    
    print(f"Agent: {agent}")
    print(f"State size: {agent.state_size()}")
    
    # Test act
    obs = torch.randn(5)
    action = agent.act(obs)
    print(f"Action: {action}")
    
    # Test state extraction
    state = agent.get_state()
    print(f"State vector length: {len(state)}")
    
    # Test state setting
    new_state = torch.randn_like(state)
    agent.set_state(new_state)
    new_action = agent.act(obs)
    print(f"Action after setting random state: {new_action}")
    
    # Test device movement
    if torch.cuda.is_available():
        agent.cuda()
        print(f"Moved to {agent.device}")
    
    # Test DifferentiableAgentWrapper
    diff_agent = DifferentiableAgentWrapper(agent)
    obs.requires_grad = True
    action = diff_agent.act(obs)
    loss = action.sum()
    loss.backward()
    print(f"Gradient through differentiable wrapper: {obs.grad is not None}")
    
    print("\nAll tests passed!")
```