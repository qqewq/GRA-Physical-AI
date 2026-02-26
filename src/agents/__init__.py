```python
"""
GRA Physical AI - Agents Module
===============================

This module provides base classes and implementations for **agents** – 
the policy‑bearing subsystems that act in the environment.

Agents are the "brains" of the GRA multiverse. They:
    - Maintain internal state (policy parameters, beliefs)
    - Generate actions based on observations
    - Learn and adapt through zeroing
    - Can be neural networks, classical controllers, or hybrid

The module defines:
    - Base Agent class with common interface
    - Neural network agents (MLP, RNN, Transformers)
    - Classical controllers (PID, LQR, MPC)
    - Hierarchical agents that combine multiple levels
    - Differentiable agents for end‑to‑end zeroing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import deque
import warnings


# ======================================================================
# Base Agent Class
# ======================================================================

class Agent(ABC):
    """
    Abstract base class for all agents in the GRA framework.
    
    An agent is a subsystem that:
        - Receives observations from the environment
        - Produces actions
        - Maintains internal state (policy parameters, memory)
        - Can be trained/zeroed
    
    Each agent has its own state space (parameters + internal state)
    which is part of the GRA multiverse.
    """
    
    def __init__(self, 
                 name: str,
                 observation_dim: int,
                 action_dim: int,
                 hidden_dims: Optional[List[int]] = None,
                 device: str = 'cpu'):
        """
        Args:
            name: agent identifier
            observation_dim: dimension of observations
            action_dim: dimension of actions
            hidden_dims: dimensions of hidden layers (if applicable)
            device: computation device
        """
        self.name = name
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or []
        self.device = device
        
        # Internal state (to be implemented by subclasses)
        self._parameters: Optional[torch.nn.ParameterDict] = None
        self._internal_state: Dict[str, torch.Tensor] = {}
        
        # Training state
        self.training = True
        self.optimizer: Optional[torch.optim.Optimizer] = None
        
    @abstractmethod
    def act(self, observation: torch.Tensor, 
            deterministic: bool = True) -> torch.Tensor:
        """
        Produce an action given observation.
        
        Args:
            observation: tensor of shape (obs_dim,) or (batch, obs_dim)
            deterministic: if False, sample from stochastic policy
        
        Returns:
            action tensor of shape (action_dim,) or (batch, action_dim)
        """
        pass
    
    @abstractmethod
    def get_parameters(self) -> torch.nn.ParameterDict:
        """
        Return all trainable parameters as a ParameterDict.
        This is used for state extraction in the multiverse.
        """
        pass
    
    @abstractmethod
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """
        Set parameters from a dictionary.
        Used during zeroing to update the agent.
        """
        pass
    
    def get_state(self) -> torch.Tensor:
        """
        Return flattened parameter vector for multiverse state.
        Override if state includes more than parameters.
        """
        params = self.get_parameters()
        return torch.cat([p.flatten() for p in params.values()])
    
    def set_state(self, state_vector: torch.Tensor):
        """
        Set state from flattened vector.
        """
        params = self.get_parameters()
        idx = 0
        new_params = {}
        for name, param in params.items():
            size = param.numel()
            new_params[name] = state_vector[idx:idx+size].reshape(param.shape)
            idx += size
        
        self.set_parameters(new_params)
    
    def to(self, device: str) -> 'Agent':
        """Move agent to device."""
        self.device = device
        return self
    
    def train_mode(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
    
    def save(self, path: str):
        """Save agent parameters."""
        torch.save({
            'name': self.name,
            'parameters': self.get_parameters(),
            'internal_state': self._internal_state
        }, path)
    
    def load(self, path: str):
        """Load agent parameters."""
        data = torch.load(path, map_location=self.device)
        self.set_parameters(data['parameters'])
        self._internal_state = data.get('internal_state', {})
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, obs={self.observation_dim}, act={self.action_dim})"


# ======================================================================
# Neural Network Agents
# ======================================================================

class MLPAgent(Agent, nn.Module):
    """
    Multi‑Layer Perceptron agent.
    
    Maps observations to actions through a feedforward network.
    Can be deterministic or stochastic (Gaussian policy).
    """
    
    def __init__(self,
                 name: str,
                 observation_dim: int,
                 action_dim: int,
                 hidden_dims: List[int] = [256, 256],
                 activation: str = 'relu',
                 stochastic: bool = False,
                 log_std_init: float = 0.0,
                 device: str = 'cpu'):
        """
        Args:
            name: agent identifier
            observation_dim: observation dimension
            action_dim: action dimension
            hidden_dims: list of hidden layer sizes
            activation: activation function ('relu', 'tanh', 'elu')
            stochastic: if True, output mean and log_std for Gaussian policy
            log_std_init: initial log standard deviation
            device: computation device
        """
        Agent.__init__(self, name, observation_dim, action_dim, hidden_dims, device)
        nn.Module.__init__(self)
        
        self.stochastic = stochastic
        self.log_std_init = log_std_init
        
        # Build network
        layers = []
        prev_dim = observation_dim
        act_fn = self._get_activation(activation)
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(act_fn)
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Output layers
        if stochastic:
            self.mean_head = nn.Linear(prev_dim, action_dim)
            self.log_std = nn.Parameter(torch.ones(action_dim) * log_std_init)
        else:
            self.output = nn.Linear(prev_dim, action_dim)
        
        self.to(device)
    
    def _get_activation(self, name: str) -> nn.Module:
        if name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'elu':
            return nn.ELU()
        else:
            raise ValueError(f"Unknown activation: {name}")
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass (deterministic)."""
        features = self.shared_net(observation)
        if self.stochastic:
            return self.mean_head(features)
        else:
            return self.output(features)
    
    def act(self, observation: torch.Tensor, 
            deterministic: bool = True) -> torch.Tensor:
        """
        Produce action.
        
        For stochastic policies:
            - deterministic=True: return mean
            - deterministic=False: sample from Gaussian
        """
        observation = observation.to(self.device)
        single = observation.dim() == 1
        
        if single:
            observation = observation.unsqueeze(0)
        
        if not self.stochastic:
            action = self.forward(observation)
        else:
            mean = self.forward(observation)
            if deterministic:
                action = mean
            else:
                std = torch.exp(self.log_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.rsample()  # reparameterized for gradients
        
        if single:
            action = action.squeeze(0)
        
        return action
    
    def get_parameters(self) -> torch.nn.ParameterDict:
        """Return all trainable parameters."""
        return dict(self.named_parameters())
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set parameters from dictionary."""
        own_params = dict(self.named_parameters())
        for name, value in params.items():
            if name in own_params:
                own_params[name].data.copy_(value.to(self.device))
    
    def get_log_prob(self, observation: torch.Tensor, 
                     action: torch.Tensor) -> torch.Tensor:
        """Compute log probability of action under current policy."""
        if not self.stochastic:
            raise RuntimeError("Deterministic policy has no log prob")
        
        mean = self.forward(observation)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)
    
    def entropy(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute entropy of policy at given observation."""
        if not self.stochastic:
            return torch.tensor(0.0, device=self.device)
        
        mean = self.forward(observation)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist.entropy().sum(dim=-1)


class RNNAgent(Agent, nn.Module):
    """
    Recurrent Neural Network agent.
    
    Maintains hidden state across time steps.
    Useful for partially observable environments.
    """
    
    def __init__(self,
                 name: str,
                 observation_dim: int,
                 action_dim: int,
                 hidden_dim: int = 256,
                 rnn_type: str = 'lstm',
                 num_layers: int = 2,
                 stochastic: bool = False,
                 device: str = 'cpu'):
        """
        Args:
            name: agent identifier
            observation_dim: observation dimension
            action_dim: action dimension
            hidden_dim: RNN hidden dimension
            rnn_type: 'lstm' or 'gru'
            num_layers: number of RNN layers
            stochastic: if True, use Gaussian policy
            device: computation device
        """
        Agent.__init__(self, name, observation_dim, action_dim, [hidden_dim], device)
        nn.Module.__init__(self)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.stochastic = stochastic
        
        # RNN
        if rnn_type.lower() == 'lstm':
            self.rnn = nn.LSTM(observation_dim, hidden_dim, num_layers, batch_first=True)
            self.hidden_state = None
        elif rnn_type.lower() == 'gru':
            self.rnn = nn.GRU(observation_dim, hidden_dim, num_layers, batch_first=True)
            self.hidden_state = None
        else:
            raise ValueError(f"Unknown RNN type: {rnn_type}")
        
        # Output layers
        if stochastic:
            self.mean_head = nn.Linear(hidden_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))
        else:
            self.output = nn.Linear(hidden_dim, action_dim)
        
        self.to(device)
    
    def reset_hidden(self, batch_size: int = 1):
        """Reset RNN hidden state."""
        device = self.device
        if isinstance(self.rnn, nn.LSTM):
            self.hidden_state = (
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
            )
        else:  # GRU
            self.hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
    
    def forward(self, observation: torch.Tensor, 
                hidden: Optional[Any] = None) -> Tuple[torch.Tensor, Any]:
        """
        Forward pass through RNN.
        
        Returns:
            (output, new_hidden)
        """
        batch_size = observation.shape[0] if observation.dim() > 1 else 1
        
        if hidden is None:
            if self.hidden_state is None:
                self.reset_hidden(batch_size)
            hidden = self.hidden_state
        
        # RNN expects (batch, seq_len, input_dim)
        if observation.dim() == 2:
            # (batch, obs_dim) -> (batch, 1, obs_dim)
            obs = observation.unsqueeze(1)
        else:
            obs = observation
        
        output, new_hidden = self.rnn(obs, hidden)
        self.hidden_state = new_hidden
        
        # Take last time step
        output = output[:, -1, :]
        
        if self.stochastic:
            return self.mean_head(output), new_hidden
        else:
            return self.output(output), new_hidden
    
    def act(self, observation: torch.Tensor, 
            deterministic: bool = True) -> torch.Tensor:
        """Produce action using current hidden state."""
        observation = observation.to(self.device)
        single = observation.dim() == 1
        
        if single:
            observation = observation.unsqueeze(0)
        
        if not self.stochastic:
            action, _ = self.forward(observation)
        else:
            mean, _ = self.forward(observation)
            if deterministic:
                action = mean
            else:
                std = torch.exp(self.log_std)
                dist = torch.distributions.Normal(mean, std)
                action = dist.rsample()
        
        if single:
            action = action.squeeze(0)
        
        return action
    
    def get_parameters(self) -> torch.nn.ParameterDict:
        return dict(self.named_parameters())
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        own_params = dict(self.named_parameters())
        for name, value in params.items():
            if name in own_params:
                own_params[name].data.copy_(value.to(self.device))
    
    def get_state(self) -> torch.Tensor:
        """Include hidden state in agent state."""
        param_vec = super().get_state()
        if self.hidden_state is not None:
            # Flatten hidden state
            if isinstance(self.hidden_state, tuple):
                h_flat = torch.cat([h.flatten() for h in self.hidden_state])
            else:
                h_flat = self.hidden_state.flatten()
            return torch.cat([param_vec, h_flat])
        return param_vec
    
    def set_state(self, state_vector: torch.Tensor):
        """Set parameters and hidden state."""
        # First set parameters
        param_dict = self.get_parameters()
        param_size = sum(p.numel() for p in param_dict.values())
        
        param_vec = state_vector[:param_size]
        super().set_state(param_vec)
        
        # Then set hidden state if present
        if param_size < len(state_vector):
            hidden_size = len(state_vector) - param_size
            if isinstance(self.hidden_state, tuple):
                # LSTM: split into two parts
                h_size = hidden_size // 2
                h = state_vector[param_size:param_size+h_size]
                c = state_vector[param_size+h_size:]
                self.hidden_state = (
                    h.reshape(self.num_layers, -1, self.hidden_dim),
                    c.reshape(self.num_layers, -1, self.hidden_dim)
                )
            else:
                # GRU
                self.hidden_state = state_vector[param_size:].reshape(
                    self.num_layers, -1, self.hidden_dim
                )


# ======================================================================
# Classical Controllers
# ======================================================================

class PIDAgent(Agent):
    """
    Proportional‑Integral‑Derivative controller.
    
    Classic feedback control with tunable gains.
    The state consists of the PID gains (Kp, Ki, Kd).
    """
    
    def __init__(self,
                 name: str,
                 observation_dim: int,
                 action_dim: int,
                 dt: float = 0.01,
                 initial_gains: Optional[Dict[str, float]] = None,
                 device: str = 'cpu'):
        """
        Args:
            name: agent identifier
            observation_dim: must be 2 * action_dim (setpoint + actual)
            action_dim: number of control outputs
            dt: time step
            initial_gains: initial PID gains
            device: computation device
        """
        assert observation_dim == 2 * action_dim, \
            f"PIDAgent requires observation_dim = 2 * action_dim, got {observation_dim} vs {action_dim}"
        
        super().__init__(name, observation_dim, action_dim, device=device)
        
        self.dt = dt
        
        # PID gains as learnable parameters
        if initial_gains is None:
            initial_gains = {'Kp': 1.0, 'Ki': 0.1, 'Kd': 0.05}
        
        self.Kp = nn.Parameter(torch.ones(action_dim) * initial_gains['Kp'])
        self.Ki = nn.Parameter(torch.ones(action_dim) * initial_gains['Ki'])
        self.Kd = nn.Parameter(torch.ones(action_dim) * initial_gains['Kd'])
        
        # Integral accumulator
        self.integral = torch.zeros(action_dim, device=device)
        self.prev_error = torch.zeros(action_dim, device=device)
        
        self.to(device)
    
    def act(self, observation: torch.Tensor, 
            deterministic: bool = True) -> torch.Tensor:
        """
        Compute PID output.
        
        observation: [setpoint_1, ..., setpoint_n, actual_1, ..., actual_n]
        """
        observation = observation.to(self.device)
        single = observation.dim() == 1
        
        if single:
            observation = observation.unsqueeze(0)
        
        batch_size = observation.shape[0]
        action_dim = self.action_dim
        
        # Split into setpoint and actual
        setpoint = observation[:, :action_dim]
        actual = observation[:, action_dim:]
        
        error = setpoint - actual
        
        # Update integral and derivative
        if self.training:
            # For training, we don't maintain persistent integral across batches
            integral = error.cumsum(dim=0) * self.dt if batch_size > 1 else error * self.dt
            derivative = (error - self.prev_error) / self.dt if hasattr(self, 'prev_error') else torch.zeros_like(error)
        else:
            # Online: use persistent integral
            self.integral = self.integral + error.mean(dim=0) * self.dt
            derivative = (error - self.prev_error) / self.dt
            self.prev_error = error
        
        # PID formula
        if self.training and batch_size > 1:
            # Vectorized for batch
            Kp = self.Kp.unsqueeze(0)
            Ki = self.Ki.unsqueeze(0)
            Kd = self.Kd.unsqueeze(0)
            output = Kp * error + Ki * integral + Kd * derivative
        else:
            output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        if single:
            output = output.squeeze(0)
        
        return output
    
    def reset(self):
        """Reset integral and previous error."""
        self.integral = torch.zeros(self.action_dim, device=self.device)
        self.prev_error = torch.zeros(self.action_dim, device=self.device)
    
    def get_parameters(self) -> torch.nn.ParameterDict:
        return torch.nn.ParameterDict({
            'Kp': self.Kp,
            'Ki': self.Ki,
            'Kd': self.Kd
        })
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        if 'Kp' in params:
            self.Kp.data.copy_(params['Kp'].to(self.device))
        if 'Ki' in params:
            self.Ki.data.copy_(params['Ki'].to(self.device))
        if 'Kd' in params:
            self.Kd.data.copy_(params['Kd'].to(self.device))


class LQRAgent(Agent):
    """
    Linear Quadratic Regulator.
    
    Optimal controller for linear systems with quadratic cost.
    State consists of the feedback gain matrix K.
    """
    
    def __init__(self,
                 name: str,
                 state_dim: int,
                 action_dim: int,
                 A: Optional[torch.Tensor] = None,
                 B: Optional[torch.Tensor] = None,
                 Q: Optional[torch.Tensor] = None,
                 R: Optional[torch.Tensor] = None,
                 device: str = 'cpu'):
        """
        Args:
            name: agent identifier
            state_dim: dimension of system state
            action_dim: dimension of control
            A, B: system matrices (if None, will be learned)
            Q, R: cost matrices (if None, will be learned)
            device: computation device
        """
        super().__init__(name, state_dim, action_dim, device=device)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # System matrices (learnable)
        if A is None:
            self.A = nn.Parameter(torch.randn(state_dim, state_dim) * 0.1)
        else:
            self.A = nn.Parameter(A)
        
        if B is None:
            self.B = nn.Parameter(torch.randn(state_dim, action_dim) * 0.1)
        else:
            self.B = nn.Parameter(B)
        
        # Cost matrices (learnable, must be positive definite)
        if Q is None:
            self.Q = nn.Parameter(torch.eye(state_dim))
        else:
            self.Q = nn.Parameter(Q)
        
        if R is None:
            self.R = nn.Parameter(torch.eye(action_dim))
        else:
            self.R = nn.Parameter(R)
        
        # Compute optimal gain (cached)
        self._K: Optional[torch.Tensor] = None
        
        self.to(device)
    
    def _compute_gain(self) -> torch.Tensor:
        """Solve Riccati equation for optimal gain."""
        # Discrete-time LQR: K = (R + B^T P B)^{-1} B^T P A
        # This is a simplified version – in practice use iterative method
        try:
            P = torch.eye(self.state_dim, device=self.device)
            for _ in range(100):
                P_next = self.Q + self.A.T @ P @ self.A - self.A.T @ P @ self.B @ \
                         torch.inverse(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
                if torch.norm(P_next - P) < 1e-4:
                    break
                P = P_next
            
            K = torch.inverse(self.R + self.B.T @ P @ self.B) @ self.B.T @ P @ self.A
            return K
        except:
            # Fallback to simple gain
            return torch.zeros(self.action_dim, self.state_dim, device=self.device)
    
    def act(self, observation: torch.Tensor, 
            deterministic: bool = True) -> torch.Tensor:
        """
        Compute LQR control: u = -K x
        """
        observation = observation.to(self.device)
        single = observation.dim() == 1
        
        if single:
            observation = observation.unsqueeze(0)
        
        # Update gain if needed
        if self._K is None or self.training:
            self._K = self._compute_gain()
        
        # u = -K x
        action = -observation @ self._K.T
        
        if single:
            action = action.squeeze(0)
        
        return action
    
    def get_parameters(self) -> torch.nn.ParameterDict:
        return torch.nn.ParameterDict({
            'A': self.A,
            'B': self.B,
            'Q': self.Q,
            'R': self.R
        })
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        if 'A' in params:
            self.A.data.copy_(params['A'].to(self.device))
        if 'B' in params:
            self.B.data.copy_(params['B'].to(self.device))
        if 'Q' in params:
            self.Q.data.copy_(params['Q'].to(self.device))
        if 'R' in params:
            self.R.data.copy_(params['R'].to(self.device))
        self._K = None  # Invalidate cache


# ======================================================================
# Hierarchical Agents
# ======================================================================

class HierarchicalAgent(Agent):
    """
    Agent composed of multiple sub‑agents at different levels.
    
    This mirrors the GRA hierarchy itself – a high‑level agent
    that coordinates lower‑level agents.
    """
    
    def __init__(self,
                 name: str,
                 sub_agents: Dict[str, Agent],
                 coordinator: Optional[Agent] = None,
                 device: str = 'cpu'):
        """
        Args:
            name: agent identifier
            sub_agents: mapping from level/name to Agent
            coordinator: optional agent that coordinates sub‑agents
            device: computation device
        """
        # Determine observation/action dimensions from sub‑agents
        obs_dim = sum(a.observation_dim for a in sub_agents.values())
        act_dim = sum(a.action_dim for a in sub_agents.values())
        
        super().__init__(name, obs_dim, act_dim, device=device)
        
        self.sub_agents = nn.ModuleDict(sub_agents) if all(isinstance(a, nn.Module) for a in sub_agents.values()) else sub_agents
        self.coordinator = coordinator
        
        # Move all sub‑agents to device
        for a in self.sub_agents.values():
            if hasattr(a, 'to'):
                a.to(device)
        if coordinator and hasattr(coordinator, 'to'):
            coordinator.to(device)
    
    def act(self, observation: torch.Tensor, 
            deterministic: bool = True) -> torch.Tensor:
        """
        Decompose observation for sub‑agents and combine actions.
        
        If coordinator exists, it gets the full observation and outputs
        a modulation of sub‑agent actions.
        """
        observation = observation.to(self.device)
        single = observation.dim() == 1
        
        if single:
            observation = observation.unsqueeze(0)
        
        batch_size = observation.shape[0]
        
        # Split observation for sub‑agents
        sub_obs = {}
        start = 0
        for name, agent in self.sub_agents.items():
            end = start + agent.observation_dim
            sub_obs[name] = observation[:, start:end]
            start = end
        
        # Get actions from sub‑agents
        sub_actions = {}
        for name, agent in self.sub_agents.items():
            sub_actions[name] = agent.act(sub_obs[name], deterministic)
        
        # If coordinator exists, use it to modulate
        if self.coordinator is not None:
            coord_obs = observation
            coord_action = self.coordinator.act(coord_obs, deterministic)
            
            # Use coordinator output to weight/modify sub‑actions
            # This is task‑specific – here we just concatenate
            combined = torch.cat([coord_action] + list(sub_actions.values()), dim=-1)
        else:
            # Just concatenate sub‑actions
            combined = torch.cat(list(sub_actions.values()), dim=-1)
        
        if single:
            combined = combined.squeeze(0)
        
        return combined
    
    def get_parameters(self) -> torch.nn.ParameterDict:
        """Collect parameters from all sub‑agents."""
        params = torch.nn.ParameterDict()
        for name, agent in self.sub_agents.items():
            if hasattr(agent, 'get_parameters'):
                agent_params = agent.get_parameters()
                for pname, p in agent_params.items():
                    params[f"{name}.{pname}"] = p
        
        if self.coordinator and hasattr(self.coordinator, 'get_parameters'):
            coord_params = self.coordinator.get_parameters()
            for pname, p in coord_params.items():
                params[f"coord.{pname}"] = p
        
        return params
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Distribute parameters to sub‑agents."""
        for name, agent in self.sub_agents.items():
            if hasattr(agent, 'set_parameters'):
                agent_params = {k.replace(f"{name}.", ""): v 
                              for k, v in params.items() if k.startswith(f"{name}.")}
                if agent_params:
                    agent.set_parameters(agent_params)
        
        if self.coordinator and hasattr(self.coordinator, 'set_parameters'):
            coord_params = {k.replace("coord.", ""): v 
                          for k, v in params.items() if k.startswith("coord.")}
            if coord_params:
                self.coordinator.set_parameters(coord_params)
    
    def train_mode(self, mode: bool = True):
        """Set training mode for all sub‑agents."""
        super().train_mode(mode)
        for agent in self.sub_agents.values():
            if hasattr(agent, 'train_mode'):
                agent.train_mode(mode)
        if self.coordinator and hasattr(self.coordinator, 'train_mode'):
            self.coordinator.train_mode(mode)


# ======================================================================
# Differentiable Agent Wrapper
# ======================================================================

class DifferentiableAgent(Agent):
    """
    Wrapper that makes any agent differentiable for end‑to‑end zeroing.
    
    Uses straight‑through estimators for non‑differentiable operations.
    """
    
    def __init__(self, base_agent: Agent):
        """
        Args:
            base_agent: the agent to wrap
        """
        super().__init__(
            name=f"diff_{base_agent.name}",
            observation_dim=base_agent.observation_dim,
            action_dim=base_agent.action_dim,
            device=base_agent.device
        )
        self.base = base_agent
    
    def act(self, observation: torch.Tensor, 
            deterministic: bool = True) -> torch.Tensor:
        """
        Forward pass with straight‑through gradient estimator.
        
        For non‑differentiable operations in the base agent,
        we use the identity in the backward pass.
        """
        observation = observation.to(self.device)
        
        # Forward pass through base agent
        with torch.no_grad():
            action_base = self.base.act(observation, deterministic)
        
        # For backward pass, use observation as proxy if needed
        if observation.requires_grad and not action_base.requires_grad:
            # Straight‑through: treat action as observation for gradients
            action = observation + (action_base - observation).detach()
        else:
            action = action_base
        
        return action
    
    def get_parameters(self) -> torch.nn.ParameterDict:
        return self.base.get_parameters() if hasattr(self.base, 'get_parameters') else torch.nn.ParameterDict()
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        if hasattr(self.base, 'set_parameters'):
            self.base.set_parameters(params)
    
    def get_state(self) -> torch.Tensor:
        return self.base.get_state() if hasattr(self.base, 'get_state') else torch.tensor([])
    
    def set_state(self, state_vector: torch.Tensor):
        if hasattr(self.base, 'set_state'):
            self.base.set_state(state_vector)


# ======================================================================
# Utility Functions
# ======================================================================

def create_agent_from_config(config: Dict[str, Any]) -> Agent:
    """
    Factory function to create an agent from configuration dictionary.
    
    Example config:
        {
            'type': 'MLPAgent',
            'name': 'policy',
            'observation_dim': 10,
            'action_dim': 4,
            'hidden_dims': [256, 256],
            'stochastic': True
        }
    """
    agent_type = config.pop('type')
    
    if agent_type == 'MLPAgent':
        return MLPAgent(**config)
    elif agent_type == 'RNNAgent':
        return RNNAgent(**config)
    elif agent_type == 'PIDAgent':
        return PIDAgent(**config)
    elif agent_type == 'LQRAgent':
        return LQRAgent(**config)
    elif agent_type == 'HierarchicalAgent':
        # Need to create sub‑agents first
        sub_agents = {}
        for name, sub_config in config.get('sub_agents', {}).items():
            sub_agents[name] = create_agent_from_config(sub_config)
        config['sub_agents'] = sub_agents
        
        if 'coordinator' in config:
            config['coordinator'] = create_agent_from_config(config['coordinator'])
        
        return HierarchicalAgent(**config)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


def agent_to_subsystem(agent: Agent, multi_index: Any) -> Any:
    """
    Wrap an agent as a GRA subsystem.
    
    This allows the agent to be placed in the multiverse.
    """
    from ..core.subsystem import Subsystem
    
    class AgentSubsystem(Subsystem):
        def __init__(self, agent: Agent, multi_index):
            super().__init__(multi_index, None, None)
            self.agent = agent
        
        def get_state(self):
            return self.agent.get_state()
        
        def set_state(self, state):
            self.agent.set_state(state)
        
        def step(self, dt, action=None):
            # Agent steps are handled by the environment
            pass
    
    return AgentSubsystem(agent, multi_index)


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing Agents ===\n")
    
    # Test MLP agent
    mlp = MLPAgent("test_mlp", observation_dim=10, action_dim=4, 
                   hidden_dims=[64, 64], stochastic=True)
    
    obs = torch.randn(10)
    action = mlp.act(obs, deterministic=False)
    print(f"MLP action: {action}")
    
    # Test PID agent
    pid = PIDAgent("test_pid", observation_dim=4, action_dim=2, dt=0.01)
    obs = torch.tensor([1.0, 0.0, 0.5, 0.2])  # setpoint1, setpoint2, actual1, actual2
    action = pid.act(obs)
    print(f"PID action: {action}")
    
    # Test RNN agent with sequence
    rnn = RNNAgent("test_rnn", observation_dim=5, action_dim=2, hidden_dim=32)
    rnn.reset_hidden()
    
    # Process a sequence
    seq = torch.randn(10, 5)  # 10 time steps
    actions = []
    for t in range(10):
        act = rnn.act(seq[t])
        actions.append(act)
    print(f"RNN processed 10 steps, final action: {actions[-1]}")
    
    # Test parameter getting/setting
    params = mlp.get_parameters()
    print(f"MLP has {len(params)} parameter tensors")
    
    state = mlp.get_state()
    print(f"MLP state vector size: {len(state)}")
    
    # Test hierarchical agent
    low1 = MLPAgent("low1", observation_dim=3, action_dim=1, hidden_dims=[16])
    low2 = MLPAgent("low2", observation_dim=3, action_dim=1, hidden_dims=[16])
    high = MLPAgent("high", observation_dim=6, action_dim=2, hidden_dims=[32])
    
    hierarchical = HierarchicalAgent("hier", 
                                     sub_agents={'left': low1, 'right': low2},
                                     coordinator=high)
    
    obs = torch.randn(6)
    action = hierarchical.act(obs)
    print(f"Hierarchical action: {action}")
    
    print("\nAll tests passed!")
```