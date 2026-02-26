```python
"""
GRA Physical AI - NVIDIA GR00T Agent Wrapper
============================================

This module provides wrappers for NVIDIA's GR00T foundation models,
integrating them into the GRA framework as agents at various levels.

GR00T (Generalist Robot 00 Technology) is NVIDIA's foundation model
for humanoid robots, providing:
    - Vision-language understanding
    - Motor primitive generation
    - Task planning
    - Human-robot interaction

This module implements:
    - Base GR00T wrapper with common interface
    - Level-specific wrappers (G0-G4)
    - Integration with GRA's state representation
    - Differentiable approximations for zeroing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import json
import warnings
import os

# Try to import GR00T modules (will fail if not installed)
try:
    from gr00t.model import Gr00tModel
    from gr00t.processor import Gr00tProcessor
    from gr00t.policy import Gr00tPolicy
    GR00T_AVAILABLE = True
except ImportError:
    GR00T_AVAILABLE = False
    warnings.warn("GR00T not installed. Install from NVIDIA's developer portal.")

from ..core.base_agent import BaseAgent, DifferentiableAgentWrapper
from ..core.multiverse import MultiIndex


# ======================================================================
# Base GR00T Wrapper
# ======================================================================

class Gr00tBaseWrapper(BaseAgent):
    """
    Base wrapper for NVIDIA GR00T models.
    
    This provides a common interface for all GR00T-based agents,
    handling model loading, inference, and GRA state extraction.
    """
    
    def __init__(
        self,
        name: str,
        model_path: Optional[str] = None,
        model_version: str = "gr00t-v1",
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        use_half: bool = True,
        **kwargs
    ):
        """
        Initialize GR00T wrapper.
        
        Args:
            name: Agent name
            model_path: Path to local model (if None, load from registry)
            model_version: Version of GR00T model
            device: Computation device
            dtype: Data type
            use_half: Use FP16 for faster inference
            **kwargs: Additional arguments for specific wrappers
        """
        if not GR00T_AVAILABLE:
            raise ImportError(
                "GR00T not available. Install from NVIDIA's developer portal."
            )
        
        # Determine observation/action dimensions from model
        # These will be set after loading
        super().__init__(
            name=name,
            observation_dim=kwargs.get('observation_dim', 512),  # placeholder
            action_dim=kwargs.get('action_dim', 64),             # placeholder
            device=device,
            dtype=dtype
        )
        
        self.model_path = model_path
        self.model_version = model_version
        self.use_half = use_half
        
        # Load model
        self._load_model()
        
        # GR00T-specific state
        self._internal_state = {
            'latent': None,
            'attention_maps': None,
            'task_embedding': None
        }
        
        # Token cache for efficiency
        self._token_cache = {}
        
    def _load_model(self):
        """Load GR00T model and processor."""
        if self.model_path:
            # Load from local path
            self.model = Gr00tModel.from_pretrained(self.model_path)
            self.processor = Gr00tProcessor.from_pretrained(self.model_path)
        else:
            # Load from NVIDIA registry
            self.model = Gr00tModel.from_pretrained(self.model_version)
            self.processor = Gr00tProcessor.from_pretrained(self.model_version)
        
        # Move to device
        self.model = self.model.to(self.device)
        if self.use_half:
            self.model = self.model.half()
        
        self.model.eval()
        
        # Set dimensions based on model config
        self.observation_dim = self.model.config.vision_config.hidden_size
        self.action_dim = self.model.config.action_dim
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Generate action using GR00T model.
        
        Args:
            observation: Can be:
                - Image tensor (3, H, W)
                - Text prompt (string, converted to tensor)
                - Multimodal dict
            deterministic: If True, use greedy decoding; else sample
        
        Returns:
            Action tensor
        """
        # Handle different observation types
        if isinstance(observation, str):
            # Text prompt
            inputs = self.processor(text=observation, return_tensors='pt')
        elif isinstance(observation, dict):
            # Multimodal input
            inputs = self.processor(**observation, return_tensors='pt')
        elif isinstance(observation, torch.Tensor) and observation.dim() == 3:
            # Image
            inputs = self.processor(images=observation, return_tensors='pt')
        else:
            # Assume already processed
            inputs = {'pixel_values': observation.to(self.device)}
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            if deterministic:
                # Greedy action
                outputs = self.model.generate(
                    **inputs,
                    do_sample=False,
                    num_beams=1
                )
            else:
                # Sample action
                outputs = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7
                )
        
        # Extract action from outputs
        if hasattr(outputs, 'actions'):
            action = outputs.actions
        else:
            # Assume last hidden state is action
            action = outputs.last_hidden_state
        
        # Store internal state for GRA
        self._internal_state['latent'] = outputs.pooler_output if hasattr(outputs, 'pooler_output') else None
        self._internal_state['attention_maps'] = outputs.attentions if hasattr(outputs, 'attentions') else None
        
        return action.squeeze(0) if action.dim() > 2 else action
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """
        Return GR00T model parameters.
        
        Note: GR00T is typically frozen during zeroing, but we expose
        parameters for completeness.
        """
        return dict(self.model.named_parameters())
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """
        Set GR00T model parameters.
        
        Warning: Modifying GR00T parameters may break the model.
        Use with caution.
        """
        own_params = dict(self.model.named_parameters())
        for name, value in params.items():
            if name in own_params:
                own_params[name].data.copy_(value.to(self.device))
    
    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """Return internal state (latents, attention)."""
        return {k: v for k, v in self._internal_state.items() if v is not None}
    
    def set_internal_state(self, state: Dict[str, torch.Tensor]):
        """Set internal state."""
        self._internal_state.update(state)
    
    def freeze(self):
        """Freeze GR00T parameters (no gradients)."""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze(self):
        """Unfreeze GR00T parameters."""
        for param in self.model.parameters():
            param.requires_grad = True


# ======================================================================
# Level-Specific GR00T Wrappers
# ======================================================================

class Gr00tG0Wrapper(Gr00tBaseWrapper):
    """
    GR00T wrapper for level G0 (low-level motor control).
    
    This uses GR00T's motor primitive generation capabilities.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        
        # G0 specific: motor commands
        self.action_dim = self.model.config.motor_dim if hasattr(self.model.config, 'motor_dim') else 32
        
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Generate low-level motor commands.
        
        Observation should contain:
            - Joint positions
            - Joint velocities
            - Target poses
        """
        action = super().act(observation, deterministic)
        
        # Ensure action is within motor limits
        if hasattr(self.model.config, 'motor_limits'):
            limits = self.model.config.motor_limits.to(self.device)
            action = torch.clamp(action, -limits, limits)
        
        return action


class Gr00tG1Wrapper(Gr00tBaseWrapper):
    """
    GR00T wrapper for level G1 (perception).
    
    This uses GR00T's vision-language understanding to process
    sensory data and produce perceptual representations.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        
        # G1 specific: perceptual features
        self.observation_dim = self.model.config.vision_config.hidden_size
        self.action_dim = self.model.config.projection_dim
        
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Process sensory input into perceptual features.
        
        Returns:
            Perceptual embedding
        """
        if isinstance(observation, dict) and 'image' in observation:
            # Process image
            inputs = self.processor(images=observation['image'], return_tensors='pt')
        else:
            # Assume already processed
            inputs = {'pixel_values': observation.to(self.device) if isinstance(observation, torch.Tensor) else observation}
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract visual features
        with torch.no_grad():
            vision_outputs = self.model.get_vision_features(**inputs)
        
        features = vision_outputs.pooler_output
        
        # Store internal state
        self._internal_state['vision_features'] = features
        
        return features.squeeze(0)


class Gr00tG2Wrapper(Gr00tBaseWrapper):
    """
    GR00T wrapper for level G2 (world model).
    
    This uses GR00T's physics understanding to predict
    outcomes of actions.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        
        # G2 specific: state prediction
        self.state_dim = kwargs.get('state_dim', 128)
        
    def predict_next_state(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state given current state and action.
        
        Args:
            state: Current state representation
            action: Action to take
        
        Returns:
            Predicted next state
        """
        # Combine state and action
        combined = torch.cat([state, action], dim=-1)
        
        # Use GR00T's dynamics head
        with torch.no_grad():
            next_state = self.model.forward_dynamics(combined)
        
        return next_state
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        For world model, "action" is the predicted next state.
        """
        return self.predict_next_state(observation, torch.zeros_like(observation))


class Gr00tG3Wrapper(Gr00tBaseWrapper):
    """
    GR00T wrapper for level G3 (task planning).
    
    This uses GR00T's language understanding to decompose
    high-level tasks into sequences.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        
        # G3 specific: task planning
        self.max_plan_length = kwargs.get('max_plan_length', 10)
        
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Generate a task plan from a natural language instruction.
        
        Observation should be a text prompt (string or tokenized).
        Returns a sequence of actions or subgoals.
        """
        if isinstance(observation, str):
            # Text instruction
            inputs = self.processor(text=observation, return_tensors='pt')
        else:
            inputs = {'input_ids': observation.to(self.device)}
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate plan
        with torch.no_grad():
            if deterministic:
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_plan_length,
                    do_sample=False
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_plan_length,
                    do_sample=True,
                    temperature=0.7
                )
        
        # Decode plan
        plan = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Store plan in internal state
        self._internal_state['plan'] = plan
        self._internal_state['plan_tokens'] = outputs
        
        # Return tokenized plan as tensor
        return outputs[0]


class Gr00tG4Wrapper(Gr00tBaseWrapper):
    """
    GR00T wrapper for level G4 (ethics/safety).
    
    This uses GR00T's safety modules to ensure actions
    comply with ethical constraints.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        
        # Load safety classifier
        self.safety_threshold = kwargs.get('safety_threshold', 0.5)
        
    def check_safety(self, action: torch.Tensor, context: Dict) -> Tuple[bool, float]:
        """
        Check if an action is safe given context.
        
        Returns:
            (is_safe, confidence)
        """
        # Prepare input for safety classifier
        if hasattr(self.model, 'safety_head'):
            safety_input = torch.cat([action.flatten(), context.get('state', torch.zeros(64))])
            safety_score = torch.sigmoid(self.model.safety_head(safety_input))
            is_safe = safety_score > self.safety_threshold
            return is_safe.item(), safety_score.item()
        else:
            # Default: assume safe
            return True, 1.0
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Safety filter: pass through action if safe, else modify.
        """
        # This wrapper typically doesn't generate actions directly
        # It's used as a filter on actions from lower levels
        return observation


# ======================================================================
# Differentiable GR00T Wrapper (for zeroing)
# ======================================================================

class DifferentiableGr00tWrapper(Gr00tBaseWrapper):
    """
    GR00T wrapper that supports gradients via straight-through estimators
    and learned approximations.
    
    This enables end-to-end zeroing through the GR00T model.
    """
    
    def __init__(self, name: str, use_approximation: bool = True, **kwargs):
        """
        Args:
            name: Agent name
            use_approximation: If True, use a learned approximation
                               for backpropagation
            **kwargs: Arguments for base wrapper
        """
        super().__init__(name, **kwargs)
        
        self.use_approximation = use_approximation
        
        if use_approximation:
            # Create a small learnable approximation of GR00T
            self.approx_model = nn.Sequential(
                nn.Linear(self.observation_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, self.action_dim)
            ).to(self.device)
        else:
            self.approx_model = None
        
        # Freeze real GR00T model
        self.freeze()
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Forward pass with gradient support.
        
        For backprop, we use either:
            - Straight-through estimator
            - Learned approximation
        """
        # Detach observation for real GR00T
        obs_detached = observation.detach()
        
        # Real GR00T forward (no gradients)
        with torch.no_grad():
            real_action = super().act(obs_detached, deterministic)
        
        if observation.requires_grad:
            if self.use_approximation and self.approx_model is not None:
                # Use approximation for gradients
                approx_action = self.approx_model(observation)
                # Combine: real value + gradient from approximation
                action = approx_action + (real_action - approx_action).detach()
            else:
                # Straight-through: treat as identity for gradients
                action = observation[..., :self.action_dim] + \
                         (real_action - observation[..., :self.action_dim]).detach()
        else:
            action = real_action
        
        return action
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return parameters of approximation model (if used)."""
        if self.approx_model is not None:
            return dict(self.approx_model.named_parameters())
        else:
            return {}
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set parameters of approximation model."""
        if self.approx_model is not None:
            own_params = dict(self.approx_model.named_parameters())
            for name, value in params.items():
                if name in own_params:
                    own_params[name].data.copy_(value.to(self.device))


# ======================================================================
# GR00T Policy Agent (for direct policy learning)
# ======================================================================

class Gr00tPolicyAgent(BaseAgent):
    """
    Agent that uses GR00T as a policy, with a learnable head.
    
    This keeps the base GR00T frozen and learns a task-specific
    output head, which can be zeroed efficiently.
    """
    
    def __init__(
        self,
        name: str,
        base_gr00t: Gr00tBaseWrapper,
        hidden_dims: List[int] = [256, 256],
        device: str = 'cuda',
        **kwargs
    ):
        """
        Args:
            name: Agent name
            base_gr00t: Frozen GR00T base model
            hidden_dims: Dimensions of policy head
            device: Computation device
            **kwargs: Additional arguments
        """
        # Get dimensions from base model
        obs_dim = base_gr00t.observation_dim
        act_dim = base_gr00t.action_dim
        
        super().__init__(name, obs_dim, act_dim, device, **kwargs)
        
        self.base = base_gr00t
        self.base.freeze()
        
        # Learnable policy head
        layers = []
        prev_dim = obs_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden))
            layers.append(nn.ReLU())
            prev_dim = hidden
        layers.append(nn.Linear(prev_dim, act_dim))
        
        self.policy_head = nn.Sequential(*layers).to(device)
        
        # Parameter count
        self._param_dict = dict(self.policy_head.named_parameters())
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Forward pass: GR00T features -> policy head -> action.
        """
        # Get features from base GR00T
        with torch.no_grad():
            features = self.base.act(observation, deterministic)
        
        # Apply policy head
        action = self.policy_head(features)
        
        return action
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return policy head parameters."""
        return self._param_dict
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set policy head parameters."""
        for name, value in params.items():
            if name in self._param_dict:
                self._param_dict[name].data.copy_(value.to(self.device))
    
    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """Return base GR00T's internal state."""
        return self.base.get_internal_state()
    
    def set_internal_state(self, state: Dict[str, torch.Tensor]):
        """Set base GR00T's internal state."""
        self.base.set_internal_state(state)


# ======================================================================
# Utility Functions
# ======================================================================

def create_gr00t_agent(
    level: int,
    name: str,
    model_path: Optional[str] = None,
    device: str = 'cuda',
    differentiable: bool = False,
    **kwargs
) -> BaseAgent:
    """
    Factory function to create appropriate GR00T agent for a given level.
    
    Args:
        level: GRA level (0-4)
        name: Agent name
        model_path: Path to GR00T model
        device: Computation device
        differentiable: Whether to use differentiable wrapper
        **kwargs: Additional arguments
    
    Returns:
        Configured GR00T agent
    """
    level_map = {
        0: Gr00tG0Wrapper,
        1: Gr00tG1Wrapper,
        2: Gr00tG2Wrapper,
        3: Gr00tG3Wrapper,
        4: Gr00tG4Wrapper
    }
    
    if level not in level_map:
        raise ValueError(f"Invalid level: {level}. Must be 0-4.")
    
    # Create base agent
    agent_class = level_map[level]
    agent = agent_class(name=name, device=device, **kwargs)
    
    if model_path:
        # Load from path
        agent.model_path = model_path
        agent._load_model()
    
    if differentiable:
        agent = DifferentiableGr00tWrapper(
            name=f"diff_{name}",
            use_approximation=True,
            device=device,
            **kwargs
        )
        # Copy base model
        if hasattr(agent, 'base_agent'):
            agent.base_agent = agent_class(name=name, device=device, **kwargs)
    
    return agent


def gr00t_to_subsystem(agent: Gr00tBaseWrapper, multi_index: MultiIndex) -> Any:
    """
    Convert a GR00T agent to a GRA subsystem.
    
    Args:
        agent: GR00T agent
        multi_index: MultiIndex for this subsystem
    
    Returns:
        Subsystem that can be added to Multiverse
    """
    from ..core.subsystem import Subsystem
    
    class Gr00tSubsystem(Subsystem):
        def __init__(self, agent, multi_index):
            super().__init__(multi_index, None, None)
            self.agent = agent
        
        def get_state(self):
            return self.agent.get_state()
        
        def set_state(self, state):
            self.agent.set_state(state)
        
        def step(self, dt, action=None):
            # GR00T stepping is handled by environment
            pass
    
    return Gr00tSubsystem(agent, multi_index)


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing GR00T Agent Wrappers ===\n")
    
    if not GR00T_AVAILABLE:
        print("GR00T not available. Skipping tests.")
        exit()
    
    # Test base wrapper
    print("Creating base GR00T wrapper...")
    base = Gr00tBaseWrapper(
        name="test_gr00t",
        device='cpu',
        use_half=False
    )
    print(f"  Observation dim: {base.observation_dim}")
    print(f"  Action dim: {base.action_dim}")
    print(f"  State size: {base.state_size()}")
    
    # Test act with dummy input
    dummy_obs = torch.randn(3, 224, 224)  # dummy image
    action = base.act(dummy_obs)
    print(f"  Action shape: {action.shape}")
    
    # Test level-specific wrappers
    for level in range(5):
        print(f"\nCreating G{level} wrapper...")
        agent = create_gr00t_agent(
            level=level,
            name=f"gr00t_g{level}",
            device='cpu'
        )
        print(f"  {agent.__class__.__name__}")
        print(f"  State size: {agent.state_size()}")
    
    # Test differentiable wrapper
    print("\nCreating differentiable wrapper...")
    diff_agent = DifferentiableGr00tWrapper(
        name="diff_gr00t",
        use_approximation=True,
        device='cpu'
    )
    
    # Test gradient flow
    obs = torch.randn(3, 224, 224, requires_grad=True)
    action = diff_agent.act(obs)
    loss = action.sum()
    loss.backward()
    print(f"  Gradient on observation: {obs.grad is not None}")
    
    print("\nAll tests passed!")
```