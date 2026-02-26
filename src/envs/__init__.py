```python
"""
GRA Physical AI - Environments Module
=====================================

This module provides interfaces and wrappers for various simulation and real-world
environments, allowing them to be used within the GRA framework.

Environments are where agents act and receive observations. They provide:
    - Step function (action -> next observation, reward, done)
    - Reset function
    - Observation and action spaces
    - Access to ground truth state (for foam computation)
    - Integration with the GRA multiverse

Supported environment types:
    - OpenAI Gym / Gymnasium
    - PyBullet
    - MuJoCo
    - NVIDIA Isaac Sim
    - ROS 2 based environments
    - Custom environments
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
import warnings

# Try to import common environment libraries
try:
    import gym
    import gym.spaces
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False
    warnings.warn("Gym not installed. Install with: pip install gym")

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

try:
    import mujoco
    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False

try:
    import rospy
    from std_msgs.msg import Float32MultiArray
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False

from ..core.multiverse import MultiIndex, Multiverse
from ..core.base_agent import BaseAgent


# ======================================================================
# Base Environment Interface
# ======================================================================

class BaseEnvironment(ABC):
    """
    Abstract base class for all environments in the GRA framework.
    
    Provides a unified interface for stepping, resetting, and accessing
    environment properties. Also supports integration with GRA multiverse
    for foam computation and zeroing.
    """
    
    def __init__(
        self,
        name: str,
        dt: float = 0.01,
        max_episode_steps: Optional[int] = None,
        seed: Optional[int] = None
    ):
        """
        Args:
            name: Environment name
            dt: Simulation time step (seconds)
            max_episode_steps: Maximum steps per episode
            seed: Random seed
        """
        self.name = name
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.seed = seed
        
        self.current_step = 0
        self.episode_reward = 0.0
        self.episode_count = 0
        
        # GRA multiverse integration
        self.multiverse: Optional[Multiverse] = None
        self.agent_map: Dict[MultiIndex, BaseAgent] = {}
        
    @abstractmethod
    def reset(self) -> torch.Tensor:
        """
        Reset environment to initial state.
        
        Returns:
            Initial observation as torch tensor
        """
        pass
    
    @abstractmethod
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Take one step in the environment.
        
        Args:
            action: Action tensor from agent
        
        Returns:
            observation: Next observation
            reward: Reward for this step
            done: Whether episode ended
            info: Additional information
        """
        pass
    
    @abstractmethod
    def render(self, mode: str = 'human'):
        """Render the environment."""
        pass
    
    @abstractmethod
    def close(self):
        """Clean up environment resources."""
        pass
    
    @abstractmethod
    def get_observation_space(self) -> Any:
        """Get observation space (Gym space or tuple of dimensions)."""
        pass
    
    @abstractmethod
    def get_action_space(self) -> Any:
        """Get action space (Gym space or tuple of dimensions)."""
        pass
    
    @abstractmethod
    def get_ground_truth_state(self) -> torch.Tensor:
        """
        Get ground truth state of the environment (for foam computation).
        
        Returns:
            Tensor containing full state information
        """
        pass
    
    # ======================================================================
    # GRA Integration
    # ======================================================================
    
    def attach_multiverse(self, multiverse: Multiverse):
        """Attach GRA multiverse to this environment."""
        self.multiverse = multiverse
    
    def register_agent(self, multi_index: MultiIndex, agent: BaseAgent):
        """Register an agent that acts in this environment."""
        self.agent_map[multi_index] = agent
    
    def get_agent_actions(self) -> Dict[MultiIndex, torch.Tensor]:
        """Get actions from all registered agents."""
        actions = {}
        for idx, agent in self.agent_map.items():
            # Get observation for this agent
            obs = self.get_observation_for_agent(idx)
            # Get action
            actions[idx] = agent.act(obs)
        return actions
    
    def get_observation_for_agent(self, multi_index: MultiIndex) -> torch.Tensor:
        """
        Get observation specific to an agent.
        Override for multi-agent environments.
        """
        # Default: return full observation
        return self.get_ground_truth_state()
    
    def compute_environment_foam(self) -> Dict[int, float]:
        """
        Compute foam contributions from environment dynamics.
        
        This can be overridden to add environment-specific foam terms.
        """
        return {}
    
    # ======================================================================
    # Utility Methods
    # ======================================================================
    
    def run_episode(
        self,
        agent: Optional[BaseAgent] = None,
        max_steps: Optional[int] = None,
        render: bool = False
    ) -> Dict[str, Any]:
        """
        Run a complete episode.
        
        Args:
            agent: Agent to use (if None, use registered agents)
            max_steps: Maximum steps (defaults to self.max_episode_steps)
            render: Whether to render
        
        Returns:
            Dictionary with episode statistics
        """
        obs = self.reset()
        total_reward = 0.0
        steps = 0
        max_steps = max_steps or self.max_episode_steps or 1000
        
        while steps < max_steps:
            if agent is not None:
                # Single agent
                action = agent.act(obs)
            else:
                # Multi-agent: get actions from all registered agents
                actions = self.get_agent_actions()
                # For now, just use first action
                action = next(iter(actions.values())) if actions else torch.zeros(self.get_action_space().shape[0])
            
            obs, reward, done, info = self.step(action)
            total_reward += reward
            steps += 1
            
            if render:
                self.render()
            
            if done:
                break
        
        self.episode_count += 1
        
        return {
            'episode': self.episode_count,
            'steps': steps,
            'total_reward': total_reward,
            'avg_reward': total_reward / steps if steps > 0 else 0
        }
    
    def seed(self, seed: int):
        """Set random seed."""
        self.seed = seed
        if hasattr(self, '_seed'):
            self._seed(seed)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ======================================================================
# Gym Environment Wrapper
# ======================================================================

class GymEnvironment(BaseEnvironment):
    """
    Wrapper for OpenAI Gym / Gymnasium environments.
    
    Supports any Gym-compatible environment.
    """
    
    def __init__(
        self,
        env_id: str,
        name: Optional[str] = None,
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            env_id: Gym environment ID (e.g., 'CartPole-v1')
            name: Custom name (defaults to env_id)
            render_mode: Rendering mode ('human', 'rgb_array', etc.)
            **kwargs: Additional arguments for gym.make
        """
        if not GYM_AVAILABLE:
            raise ImportError("Gym not installed")
        
        super().__init__(name or env_id, **kwargs)
        
        self.env_id = env_id
        self.render_mode = render_mode
        
        # Create environment
        self.env = gym.make(env_id, render_mode=render_mode, **kwargs)
        
        # Set seed if provided
        if self.seed is not None:
            self.env.seed(self.seed)
        
        # Get spaces
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        
        # State cache
        self._last_obs = None
        self._state = None
    
    def reset(self) -> torch.Tensor:
        """Reset Gym environment."""
        obs, info = self.env.reset() if hasattr(self.env, 'reset') else (self.env.reset(), {})
        self._last_obs = obs
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Convert to tensor
        if isinstance(obs, np.ndarray):
            return torch.tensor(obs, dtype=torch.float32)
        elif isinstance(obs, (int, float)):
            return torch.tensor([obs], dtype=torch.float32)
        else:
            return torch.tensor(obs, dtype=torch.float32)
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Step Gym environment."""
        # Convert action to numpy
        if isinstance(action, torch.Tensor):
            action_np = action.cpu().numpy()
        else:
            action_np = action
        
        # Ensure correct shape
        if isinstance(self.action_space, gym.spaces.Discrete):
            if action_np.ndim > 0:
                action_np = action_np.item()
        
        # Step
        obs, reward, terminated, truncated, info = self.env.step(action_np)
        done = terminated or truncated
        
        self._last_obs = obs
        self.current_step += 1
        self.episode_reward += reward
        
        # Convert to tensor
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
        elif isinstance(obs, (int, float)):
            obs_tensor = torch.tensor([obs], dtype=torch.float32)
        else:
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
        
        return obs_tensor, reward, done, info
    
    def render(self, mode: Optional[str] = None):
        """Render Gym environment."""
        if mode is None:
            mode = self.render_mode
        return self.env.render(mode)
    
    def close(self):
        """Close Gym environment."""
        self.env.close()
    
    def get_observation_space(self) -> Any:
        return self.observation_space
    
    def get_action_space(self) -> Any:
        return self.action_space
    
    def get_ground_truth_state(self) -> torch.Tensor:
        """Get ground truth state (for Gym, this is just the observation)."""
        if self._last_obs is None:
            return torch.tensor([])
        return torch.tensor(self._last_obs, dtype=torch.float32).flatten()
    
    def _seed(self, seed: int):
        """Set seed."""
        self.env.seed(seed)


# ======================================================================
# PyBullet Environment Wrapper
# ======================================================================

class PyBulletEnvironment(BaseEnvironment):
    """
    Wrapper for PyBullet simulations.
    
    Provides access to physics simulation and robot models.
    """
    
    def __init__(
        self,
        name: str,
        urdf_path: str,
        gui: bool = True,
        dt: float = 1/240,
        **kwargs
    ):
        """
        Args:
            name: Environment name
            urdf_path: Path to robot URDF file
            gui: Enable PyBullet GUI
            dt: Simulation time step
            **kwargs: Additional arguments
        """
        if not PYBULLET_AVAILABLE:
            raise ImportError("PyBullet not installed")
        
        super().__init__(name, dt=dt, **kwargs)
        
        self.urdf_path = urdf_path
        self.gui = gui
        
        # Connect to PyBullet
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(dt)
        
        # Load plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1], useFixedBase=False)
        
        # Get joint info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_names = []
        self.joint_indices = {}
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            name = info[1].decode('utf-8')
            self.joint_names.append(name)
            self.joint_indices[name] = i
        
        # State
        self.joint_positions = np.zeros(self.num_joints)
        self.joint_velocities = np.zeros(self.num_joints)
        self.joint_torques = np.zeros(self.num_joints)
        
        # Observation and action spaces
        self.observation_dim = self.num_joints * 2  # positions + velocities
        self.action_dim = self.num_joints  # torques
        
        # For rendering
        self.camera_params = {
            'distance': 2.0,
            'yaw': 50,
            'pitch': -35,
            'target': [0, 0, 0]
        }
    
    def reset(self) -> torch.Tensor:
        """Reset PyBullet simulation."""
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Reload plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF(self.urdf_path, basePosition=[0, 0, 0.1], useFixedBase=False)
        
        # Reset joint states to zero
        for i in range(self.num_joints):
            p.resetJointState(self.robot_id, i, 0, 0)
        
        # Step to stabilize
        for _ in range(10):
            p.stepSimulation()
        
        self._update_state()
        self.current_step = 0
        self.episode_reward = 0.0
        
        return self.get_observation()
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Apply torques and step simulation."""
        # Convert action to numpy
        if isinstance(action, torch.Tensor):
            torques = action.cpu().numpy()
        else:
            torques = action
        
        # Apply torques to joints
        for i in range(self.num_joints):
            p.setJointMotorControl2(
                self.robot_id, i,
                p.TORQUE_CONTROL,
                force=torques[i] if i < len(torques) else 0.0
            )
        
        # Step simulation
        p.stepSimulation()
        
        # Update state
        self._update_state()
        
        # Get observation
        obs = self.get_observation()
        
        # Compute reward (simple: minimize energy and stay upright)
        reward = -np.sum(torques**2) * 0.01
        
        # Check if done (fell over)
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        done = pos[2] < 0.3  # height threshold
        
        self.current_step += 1
        self.episode_reward += reward
        
        info = {
            'joint_positions': self.joint_positions.copy(),
            'joint_velocities': self.joint_velocities.copy(),
            'joint_torques': self.joint_torques.copy()
        }
        
        return obs, reward, done, info
    
    def _update_state(self):
        """Update internal state from PyBullet."""
        for i in range(self.num_joints):
            state = p.getJointState(self.robot_id, i)
            self.joint_positions[i] = state[0]
            self.joint_velocities[i] = state[1]
            self.joint_torques[i] = state[3]
    
    def get_observation(self) -> torch.Tensor:
        """Get current observation."""
        obs = np.concatenate([self.joint_positions, self.joint_velocities])
        return torch.tensor(obs, dtype=torch.float32)
    
    def render(self, mode: str = 'human'):
        """Render with PyBullet GUI."""
        if self.gui and mode == 'human':
            # Update camera
            p.resetDebugVisualizerCamera(
                cameraDistance=self.camera_params['distance'],
                cameraYaw=self.camera_params['yaw'],
                cameraPitch=self.camera_params['pitch'],
                cameraTargetPosition=self.camera_params['target']
            )
    
    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect()
    
    def get_observation_space(self) -> Any:
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,)
        )
    
    def get_action_space(self) -> Any:
        return gym.spaces.Box(
            low=-10, high=10, shape=(self.action_dim,)
        )
    
    def get_ground_truth_state(self) -> torch.Tensor:
        """Get full ground truth state."""
        state = np.concatenate([
            self.joint_positions,
            self.joint_velocities,
            self.joint_torques,
            p.getBasePositionAndOrientation(self.robot_id)[0]  # base position
        ])
        return torch.tensor(state, dtype=torch.float32)
    
    def set_camera(self, distance=None, yaw=None, pitch=None, target=None):
        """Set camera parameters."""
        if distance is not None:
            self.camera_params['distance'] = distance
        if yaw is not None:
            self.camera_params['yaw'] = yaw
        if pitch is not None:
            self.camera_params['pitch'] = pitch
        if target is not None:
            self.camera_params['target'] = target


# ======================================================================
# MuJoCo Environment Wrapper
# ======================================================================

class MuJoCoEnvironment(BaseEnvironment):
    """
    Wrapper for MuJoCo simulations.
    
    Provides high-performance physics simulation.
    """
    
    def __init__(
        self,
        name: str,
        xml_path: str,
        dt: float = 0.005,
        **kwargs
    ):
        """
        Args:
            name: Environment name
            xml_path: Path to MuJoCo XML file
            dt: Simulation time step
            **kwargs: Additional arguments
        """
        if not MUJOCO_AVAILABLE:
            raise ImportError("MuJoCo not installed")
        
        super().__init__(name, dt=dt, **kwargs)
        
        self.xml_path = xml_path
        
        # Load model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Set time step
        self.model.opt.timestep = dt
        
        # Dimensions
        self.nq = self.model.nq  # position dimensions
        self.nv = self.model.nv  # velocity dimensions
        self.nu = self.model.nu  # control dimensions
        
        self.observation_dim = self.nq + self.nv
        self.action_dim = self.nu
        
        # For rendering
        self.viewer = None
    
    def reset(self) -> torch.Tensor:
        """Reset MuJoCo simulation."""
        mujoco.mj_resetData(self.model, self.data)
        
        # Random initial state
        self.data.qpos[:] = np.random.randn(self.nq) * 0.1
        self.data.qvel[:] = np.random.randn(self.nv) * 0.1
        
        mujoco.mj_forward(self.model, self.data)
        
        self.current_step = 0
        self.episode_reward = 0.0
        
        return self.get_observation()
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Apply controls and step simulation."""
        # Convert action to numpy
        if isinstance(action, torch.Tensor):
            ctrl = action.cpu().numpy()
        else:
            ctrl = action
        
        # Apply controls
        self.data.ctrl[:] = ctrl[:self.nu]
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        obs = self.get_observation()
        
        # Compute reward (simple: minimize control)
        reward = -np.sum(ctrl**2) * 0.01
        
        # Check if done (height threshold)
        done = self.data.qpos[2] < 0.3  # assuming z is at index 2
        
        self.current_step += 1
        self.episode_reward += reward
        
        info = {
            'qpos': self.data.qpos.copy(),
            'qvel': self.data.qvel.copy()
        }
        
        return obs, reward, done, info
    
    def get_observation(self) -> torch.Tensor:
        """Get current observation."""
        obs = np.concatenate([self.data.qpos, self.data.qvel])
        return torch.tensor(obs, dtype=torch.float32)
    
    def render(self, mode: str = 'human'):
        """Render with MuJoCo viewer."""
        if mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif mode == 'rgb_array':
            # Render to numpy array
            renderer = mujoco.Renderer(self.model)
            renderer.update_scene(self.data)
            return renderer.render()
    
    def close(self):
        """Close MuJoCo viewer."""
        if self.viewer is not None:
            self.viewer.close()
    
    def get_observation_space(self) -> Any:
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.observation_dim,)
        )
    
    def get_action_space(self) -> Any:
        return gym.spaces.Box(
            low=-self.model.actuator_ctrlrange[:, 0],
            high=self.model.actuator_ctrlrange[:, 1],
            shape=(self.action_dim,)
        )
    
    def get_ground_truth_state(self) -> torch.Tensor:
        """Get full ground truth state."""
        state = np.concatenate([self.data.qpos, self.data.qvel, self.data.qacc])
        return torch.tensor(state, dtype=torch.float32)


# ======================================================================
# Multi-Agent Environment Wrapper
# ======================================================================

class MultiAgentEnvironment(BaseEnvironment):
    """
    Environment that supports multiple agents.
    
    This wraps a base environment and provides separate observations
    and actions for each agent.
    """
    
    def __init__(
        self,
        base_env: BaseEnvironment,
        agent_ids: List[str],
        observation_fn: Optional[Callable] = None,
        reward_fn: Optional[Callable] = None,
        done_fn: Optional[Callable] = None,
        **kwargs
    ):
        """
        Args:
            base_env: Underlying environment
            agent_ids: List of agent identifiers
            observation_fn: Function mapping (env_state, agent_id) -> observation
            reward_fn: Function mapping (env_state, agent_id, action) -> reward
            done_fn: Function mapping (env_state) -> done
            **kwargs: Additional arguments
        """
        super().__init__(f"multi_{base_env.name}", base_env.dt, **kwargs)
        
        self.base_env = base_env
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        
        # Default functions
        self.observation_fn = observation_fn or self._default_observation
        self.reward_fn = reward_fn or self._default_reward
        self.done_fn = done_fn or self._default_done
        
        # Agent-specific states
        self.agent_observations: Dict[str, torch.Tensor] = {}
        self.agent_rewards: Dict[str, float] = {}
        self.agent_dones: Dict[str, bool] = {}
        
        # Spaces (will be set after first reset)
        self.observation_spaces = {}
        self.action_spaces = {}
    
    def _default_observation(self, state: torch.Tensor, agent_id: str) -> torch.Tensor:
        """Default: give full state to all agents."""
        return state
    
    def _default_reward(self, state: torch.Tensor, agent_id: str, action: torch.Tensor) -> float:
        """Default: global reward."""
        return 0.0
    
    def _default_done(self, state: torch.Tensor) -> bool:
        """Default: episode ends when base env ends."""
        return False
    
    def reset(self) -> torch.Tensor:
        """Reset environment and return first agent's observation."""
        base_obs = self.base_env.reset()
        state = self.base_env.get_ground_truth_state()
        
        # Generate observations for all agents
        for agent_id in self.agent_ids:
            self.agent_observations[agent_id] = self.observation_fn(state, agent_id)
            self.agent_rewards[agent_id] = 0.0
            self.agent_dones[agent_id] = False
        
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Return first agent's observation (for compatibility)
        return self.agent_observations[self.agent_ids[0]]
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Step with a single action (for compatibility).
        Use step_multi for multi-agent steps.
        """
        # If single action, assume it's for first agent
        actions = {self.agent_ids[0]: action}
        return self.step_multi(actions)
    
    def step_multi(self, actions: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Step with actions for all agents.
        
        Args:
            actions: Dictionary mapping agent_id -> action tensor
        
        Returns:
            observation: First agent's observation (for compatibility)
            total_reward: Sum of all agent rewards
            done: Whether episode ended
            info: Additional information
        """
        # Take base step (using first agent's action as representative)
        # In practice, you'd design the environment to accept multi-agent actions
        first_action = next(iter(actions.values())) if actions else torch.zeros(1)
        base_obs, base_reward, base_done, base_info = self.base_env.step(first_action)
        
        # Get new state
        state = self.base_env.get_ground_truth_state()
        
        # Update agent-specific observations and rewards
        total_reward = 0.0
        agent_infos = {}
        
        for agent_id in self.agent_ids:
            # Get action for this agent (or zero)
            agent_action = actions.get(agent_id, torch.zeros(self.get_action_space(agent_id).shape[0]))
            
            # Compute observation
            self.agent_observations[agent_id] = self.observation_fn(state, agent_id)
            
            # Compute reward
            reward = self.reward_fn(state, agent_id, agent_action)
            self.agent_rewards[agent_id] = reward
            total_reward += reward
            
            # Check done
            self.agent_dones[agent_id] = self.done_fn(state)
            
            agent_infos[agent_id] = {
                'reward': reward,
                'observation': self.agent_observations[agent_id]
            }
        
        # Overall done (if any agent is done or base done)
        done = base_done or any(self.agent_dones.values())
        
        self.current_step += 1
        self.episode_reward += total_reward
        
        info = {
            'base_info': base_info,
            'agent_infos': agent_infos,
            'state': state
        }
        
        return self.agent_observations[self.agent_ids[0]], total_reward, done, info
    
    def get_observation_for_agent(self, multi_index: MultiIndex) -> torch.Tensor:
        """Get observation for a specific agent (for GRA integration)."""
        # Extract agent ID from multi-index
        agent_id = str(multi_index.indices[0])  # Assuming first level is agent ID
        return self.agent_observations.get(agent_id, torch.zeros(1))
    
    def get_observation_space(self, agent_id: Optional[str] = None) -> Any:
        """Get observation space for an agent."""
        if agent_id is None:
            agent_id = self.agent_ids[0]
        
        if agent_id in self.observation_spaces:
            return self.observation_spaces[agent_id]
        
        # Default: same as base env
        return self.base_env.get_observation_space()
    
    def get_action_space(self, agent_id: Optional[str] = None) -> Any:
        """Get action space for an agent."""
        if agent_id is None:
            agent_id = self.agent_ids[0]
        
        if agent_id in self.action_spaces:
            return self.action_spaces[agent_id]
        
        # Default: same as base env
        return self.base_env.get_action_space()
    
    def render(self, mode: str = 'human'):
        """Render base environment."""
        self.base_env.render(mode)
    
    def close(self):
        """Close base environment."""
        self.base_env.close()
    
    def get_ground_truth_state(self) -> torch.Tensor:
        """Get ground truth state from base environment."""
        return self.base_env.get_ground_truth_state()
    
    def compute_environment_foam(self) -> Dict[int, float]:
        """
        Compute foam from agent disagreements.
        
        For multi-agent environments, foam can measure:
            - Consistency of agent observations
            - Agreement on world state
            - Coordination metrics
        """
        foam = {}
        
        # Check observation consistency
        if len(self.agent_observations) > 1:
            obs_list = list(self.agent_observations.values())
            obs_diff = 0.0
            for i in range(len(obs_list)):
                for j in range(i+1, len(obs_list)):
                    diff = torch.norm(obs_list[i] - obs_list[j])
                    obs_diff += diff.item()
            
            foam[1] = obs_diff / (len(obs_list) * (len(obs_list) - 1) / 2)
        
        return foam


# ======================================================================
# ROS 2 Environment Wrapper
# ======================================================================

class ROS2Environment(BaseEnvironment):
    """
    Wrapper for ROS 2 based environments.
    
    Communicates with ROS topics for observations and actions.
    """
    
    def __init__(
        self,
        name: str,
        observation_topics: List[str],
        action_topic: str,
        dt: float = 0.1,
        **kwargs
    ):
        """
        Args:
            name: Environment name
            observation_topics: List of ROS topics to subscribe to
            action_topic: ROS topic to publish actions
            dt: Time step
            **kwargs: Additional arguments
        """
        if not ROS_AVAILABLE:
            raise ImportError("ROS not available")
        
        super().__init__(name, dt=dt, **kwargs)
        
        self.observation_topics = observation_topics
        self.action_topic = action_topic
        
        # Initialize ROS node
        rospy.init_node(f'gra_env_{name}', anonymous=True)
        
        # Subscribers
        self.observation_data = {}
        self.subscribers = []
        for topic in observation_topics:
            sub = rospy.Subscriber(topic, Float32MultiArray, self._observation_callback, callback_args=topic)
            self.subscribers.append(sub)
        
        # Publisher
        self.action_pub = rospy.Publisher(action_topic, Float32MultiArray, queue_size=10)
        
        # Rate
        self.rate = rospy.Rate(1.0 / dt)
        
        # State
        self._last_obs = None
        self._step_count = 0
    
    def _observation_callback(self, msg, topic):
        """Store observation data."""
        self.observation_data[topic] = np.array(msg.data)
    
    def reset(self) -> torch.Tensor:
        """Reset ROS environment (send reset signal)."""
        # In ROS, reset might involve a service call
        # For now, just wait for first observations
        while len(self.observation_data) < len(self.observation_topics):
            rospy.sleep(0.1)
        
        self._step_count = 0
        return self._get_observation_tensor()
    
    def _get_observation_tensor(self) -> torch.Tensor:
        """Combine all observations into one tensor."""
        all_obs = []
        for topic in self.observation_topics:
            if topic in self.observation_data:
                all_obs.extend(self.observation_data[topic])
        return torch.tensor(all_obs, dtype=torch.float32)
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Publish action and wait for next observations."""
        # Create and publish action message
        msg = Float32MultiArray()
        if isinstance(action, torch.Tensor):
            msg.data = action.cpu().numpy().tolist()
        else:
            msg.data = action.tolist() if hasattr(action, 'tolist') else [action]
        self.action_pub.publish(msg)
        
        # Wait for next observations
        self.observation_data.clear()
        timeout = rospy.Time.now() + rospy.Duration(1.0)
        while len(self.observation_data) < len(self.observation_topics) and rospy.Time.now() < timeout:
            self.rate.sleep()
        
        # Get observation
        obs = self._get_observation_tensor()
        
        # Compute reward (to be defined by user)
        reward = 0.0
        
        # Check if done (to be defined)
        done = False
        
        self._step_count += 1
        self.episode_reward += reward
        
        info = {
            'step': self._step_count,
            'observation_topics': self.observation_data.copy()
        }
        
        return obs, reward, done, info
    
    def render(self, mode: str = 'human'):
        """ROS environments typically don't render directly."""
        pass
    
    def close(self):
        """Shutdown ROS node."""
        rospy.signal_shutdown("Environment closed")
    
    def get_observation_space(self) -> Any:
        """Observation space (unknown a priori)."""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(100,))  # placeholder
    
    def get_action_space(self) -> Any:
        """Action space (unknown a priori)."""
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(10,))  # placeholder
    
    def get_ground_truth_state(self) -> torch.Tensor:
        """Get ground truth state (same as observation for ROS)."""
        return self._get_observation_tensor()


# ======================================================================
# Environment Registry
# ======================================================================

class EnvironmentRegistry:
    """
    Registry for creating environments by name.
    Useful for loading from configuration files.
    """
    
    _registry: Dict[str, type] = {
        'gym': GymEnvironment,
        'pybullet': PyBulletEnvironment,
        'mujoco': MuJoCoEnvironment,
        'multi': MultiAgentEnvironment,
        'ros2': ROS2Environment
    }
    
    @classmethod
    def register(cls, name: str, env_class: type):
        """Register a new environment type."""
        cls._registry[name] = env_class
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> BaseEnvironment:
        """
        Create an environment from a configuration dictionary.
        
        Example config:
            {
                'type': 'gym',
                'env_id': 'CartPole-v1',
                'name': 'cartpole'
            }
        """
        env_type = config.pop('type')
        if env_type not in cls._registry:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        env_class = cls._registry[env_type]
        return env_class(**config)


# ======================================================================
# Utility Functions
# ======================================================================

def create_environment(
    env_type: str,
    name: str,
    **kwargs
) -> BaseEnvironment:
    """
    Convenience function to create an environment.
    
    Args:
        env_type: 'gym', 'pybullet', 'mujoco', 'multi', 'ros2'
        name: Environment name
        **kwargs: Environment-specific arguments
    
    Returns:
        Environment instance
    """
    registry = EnvironmentRegistry()
    return registry.create({'type': env_type, 'name': name, **kwargs})


def env_to_multiverse(env: BaseEnvironment) -> Multiverse:
    """
    Create a GRA multiverse from an environment.
    
    This extracts the environment's ground truth state and creates
    appropriate subsystems for foam computation.
    """
    # Create multiverse
    mv = Multiverse(name=f"{env.name}_multiverse", max_level=1)
    
    # Add environment as a subsystem at level 0
    from ..core.subsystem import Subsystem
    
    class EnvironmentSubsystem(Subsystem):
        def __init__(self, env, multi_index):
            super().__init__(multi_index, None, None)
            self.env = env
        
        def get_state(self):
            return self.env.get_ground_truth_state()
        
        def set_state(self, state):
            # Cannot set environment state directly
            pass
        
        def step(self, dt, action=None):
            pass
    
    env_idx = MultiIndex((env.name, None))
    mv.add_subsystem(EnvironmentSubsystem(env, env_idx))
    
    return mv


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing Environments ===\n")
    
    # Test Gym environment
    if GYM_AVAILABLE:
        print("Testing Gym environment...")
        env = GymEnvironment('CartPole-v1', name='cartpole')
        obs = env.reset()
        print(f"  Observation shape: {obs.shape}")
        
        action = torch.tensor([0])
        next_obs, reward, done, info = env.step(action)
        print(f"  Reward: {reward}")
        env.close()
    
    # Test PyBullet environment
    if PYBULLET_AVAILABLE:
        print("\nTesting PyBullet environment...")
        try:
            env = PyBulletEnvironment('test_robot', 'r2d2.urdf', gui=False)
            obs = env.reset()
            print(f"  Observation shape: {obs.shape}")
            
            action = torch.zeros(env.action_dim)
            next_obs, reward, done, info = env.step(action)
            print(f"  Reward: {reward}")
            env.close()
        except Exception as e:
            print(f"  PyBullet test failed: {e}")
    
    # Test MuJoCo environment
    if MUJOCO_AVAILABLE:
        print("\nTesting MuJoCo environment...")
        try:
            # Use a simple MuJoCo model
            env = MuJoCoEnvironment('test_humanoid', 'humanoid.xml', gui=False)
            obs = env.reset()
            print(f"  Observation shape: {obs.shape}")
            
            action = torch.zeros(env.action_dim)
            next_obs, reward, done, info = env.step(action)
            print(f"  Reward: {reward}")
            env.close()
        except Exception as e:
            print(f"  MuJoCo test failed: {e}")
    
    # Test episode run
    if GYM_AVAILABLE:
        print("\nRunning episode...")
        env = GymEnvironment('CartPole-v1')
        from ..agents.base_agent import NullAgent
        agent = NullAgent('test', env.observation_space.shape[0], env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1)
        
        stats = env.run_episode(agent, max_steps=100)
        print(f"  Episode stats: {stats}")
        env.close()
    
    print("\nAll tests completed!")
```