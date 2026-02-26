```python
"""
GRA Physical AI - NVIDIA Isaac Sim / Isaac Lab Wrapper
=======================================================

This module provides wrappers for NVIDIA's Isaac Sim and Isaac Lab simulation platforms,
enabling GRA-based robots to be trained and zeroed in high-fidelity physics simulations.

Isaac Sim / Isaac Lab provide:
    - Photorealistic rendering
    - Accurate physics simulation (PhysX)
    - ROS 2 integration
    - GPU-accelerated training
    - Large-scale synthetic data generation

This wrapper implements:
    - GRA-compatible interface for Isaac environments
    - Access to ground truth state for foam computation
    - Multi-agent support
    - Domain randomization integration
    - Checkpoint/restore for zeroing
"""

import torch
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import time
import os
import json

# Try to import Isaac modules
try:
    from omni.isaac.kit import SimulationApp
    import omni.isaac.core.utils.prims as prim_utils
    import omni.isaac.core.utils.stage as stage_utils
    import omni.isaac.core.utils.numpy as np_utils
    import omni.isaac.core.utils.torch as torch_utils
    import omni.physx as physx
    import carb
    ISAAC_SIM_AVAILABLE = True
except ImportError:
    ISAAC_SIM_AVAILABLE = False
    warnings.warn("Isaac Sim not available. Install from NVIDIA Omniverse.")

try:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import ObservationManager, ActionManager, RewardManager
    from isaaclab.utils import configclass
    from isaaclab_tasks.utils import get_checkpoint_path, load_checkpoint
    ISAAC_LAB_AVAILABLE = True
except ImportError:
    ISAAC_LAB_AVAILABLE = False
    warnings.warn("Isaac Lab not available. Install from NVIDIA's GitHub.")

from ..core.base_environment import BaseEnvironment
from ..core.multiverse import MultiIndex, Multiverse
from ..core.base_agent import BaseAgent


# ======================================================================
# Isaac Sim Base Wrapper
# ======================================================================

class IsaacSimWrapper(BaseEnvironment):
    """
    Base wrapper for NVIDIA Isaac Sim.
    
    Provides access to Isaac Sim's physics and rendering capabilities.
    """
    
    def __init__(
        self,
        name: str,
        usd_path: str,
        headless: bool = False,
        dt: float = 1/60,
        physics_dt: float = 1/60,
        render_dt: float = 1/30,
        **kwargs
    ):
        """
        Args:
            name: Environment name
            usd_path: Path to USD scene file
            headless: Run without GUI
            dt: Control time step
            physics_dt: Physics simulation step
            render_dt: Rendering step
            **kwargs: Additional arguments
        """
        if not ISAAC_SIM_AVAILABLE:
            raise ImportError("Isaac Sim not available")
        
        super().__init__(name, dt=dt, **kwargs)
        
        self.usd_path = usd_path
        self.headless = headless
        self.physics_dt = physics_dt
        self.render_dt = render_dt
        
        # Launch Isaac Sim app
        self._launch_simulation_app()
        
        # Stage and prims
        self.stage = None
        self.prim_paths = {}
        self.articulation_views = {}
        self.sensor_views = {}
        
        # Load USD
        self._load_stage()
        
        # Physics
        self.physics_context = None
        self._setup_physics()
        
        # Timestep counters
        self.physics_step_count = 0
        self.render_step_count = 0
        self.control_step_count = 0
        
        # State
        self._ground_truth_state = {}
        self._last_observation = None
        
        # ROS bridge (optional)
        self.ros_bridge = None
        
        print(f"Isaac Sim environment '{name}' initialized")
    
    def _launch_simulation_app(self):
        """Launch the Isaac Sim application."""
        # Configure simulation app
        from omni.isaac.kit import SimulationApp
        
        config = {
            'headless': self.headless,
            'renderer': 'RayTracedLighting' if not self.headless else 'Noop',
            'width': 1280,
            'height': 720,
            'window_width': 1280,
            'window_height': 720,
            'display_options': 3286,  # Show all
        }
        
        self.simulation_app = SimulationApp(config)
        print("Isaac Sim app launched")
    
    def _load_stage(self):
        """Load USD stage."""
        from omni.isaac.core.utils.stage import open_stage
        
        self.stage = open_stage(self.usd_path)
        stage_utils.update_stage()
        print(f"Loaded stage: {self.usd_path}")
        
        # Get all prims
        from pxr import Usd, UsdGeom
        
        self.prim_paths = {}
        for prim in Usd.PrimRange(self.stage.GetPseudoRoot()):
            if prim.IsValid() and prim.GetTypeName():
                path = str(prim.GetPath())
                self.prim_paths[path] = prim
    
    def _setup_physics(self):
        """Set up physics simulation."""
        import omni.physics.tensors.impl.api as physx
        
        self.physics_context = physx.create_simulation_context(
            dt=self.physics_dt,
            use_gpu=True,
            gpu_device=0
        )
        
        # Set up articulation views
        self._create_articulation_views()
        
        print(f"Physics initialized with dt={self.physics_dt}")
    
    def _create_articulation_views(self):
        """Create articulation views for robots."""
        from omni.isaac.core.articulations import ArticulationView
        
        # Find robot prims
        from pxr import Usd, UsdShade
        
        robot_paths = []
        for path, prim in self.prim_paths.items():
            # Check if prim is a robot (has articulation API)
            if prim.HasAPI(UsdShade.MaterialBindingAPI):
                # Simplified detection
                if 'robot' in path.lower() or 'arm' in path.lower():
                    robot_paths.append(path)
        
        # Create articulation views
        for i, path in enumerate(robot_paths):
            view = ArticulationView(
                prim_paths_expr=path,
                name=f"robot_{i}",
                reset_xform_properties=False
            )
            self.articulation_views[f"robot_{i}"] = view
        
        print(f"Created {len(self.articulation_views)} articulation views")
    
    def reset(self) -> torch.Tensor:
        """Reset simulation to initial state."""
        # Reset physics
        self.physics_context.reset()
        
        # Reset all articulations
        for name, view in self.articulation_views.items():
            view.initialize()
            view.reset()
        
        # Step to stabilize
        for _ in range(10):
            self.step_physics()
        
        self.control_step_count = 0
        self.physics_step_count = 0
        self.render_step_count = 0
        
        # Get initial observation
        obs = self.get_observation()
        self._last_observation = obs
        
        return obs
    
    def step_physics(self, num_steps: int = 1):
        """Step physics simulation."""
        for _ in range(num_steps):
            self.physics_context.step()
            self.physics_step_count += 1
    
    def render(self, mode: str = 'human'):
        """Render a frame."""
        if not self.headless and mode == 'human':
            self.simulation_app.update()
            self.render_step_count += 1
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Apply actions and step simulation.
        
        Args:
            action: Action tensor (concatenated joint commands)
        
        Returns:
            observation, reward, done, info
        """
        # Apply actions to articulation views
        self._apply_actions(action)
        
        # Step physics (multiple steps per control step)
        steps_per_control = int(self.dt / self.physics_dt)
        self.step_physics(steps_per_control)
        
        # Render if needed
        if self.control_step_count % int(self.render_dt / self.dt) == 0:
            self.render()
        
        # Get observation
        obs = self.get_observation()
        
        # Compute reward
        reward = self._compute_reward()
        
        # Check if done
        done = self._check_done()
        
        self.control_step_count += 1
        self.episode_reward += reward
        self._last_observation = obs
        
        info = {
            'control_step': self.control_step_count,
            'physics_step': self.physics_step_count,
            'render_step': self.render_step_count,
            'ground_truth': self.get_ground_truth_state()
        }
        
        return obs, reward, done, info
    
    def _apply_actions(self, action: torch.Tensor):
        """Apply actions to robots."""
        # This is robot-specific - need to map actions to joint commands
        # Placeholder implementation
        idx = 0
        for name, view in self.articulation_views.items():
            num_joints = view.num_joints
            joint_commands = action[idx:idx+num_joints]
            view.set_joint_position_targets(joint_commands)
            idx += num_joints
    
    def get_observation(self) -> torch.Tensor:
        """Get current observation (robot states, sensors)."""
        observations = []
        
        # Joint states
        for name, view in self.articulation_views.items():
            joint_pos = view.get_joint_positions()
            joint_vel = view.get_joint_velocities()
            observations.extend(joint_pos)
            observations.extend(joint_vel)
        
        # Sensor data (cameras, lidar) - would need to be implemented
        
        return torch.tensor(observations, dtype=torch.float32)
    
    def _compute_reward(self) -> float:
        """Compute reward (to be overridden)."""
        return 0.0
    
    def _check_done(self) -> bool:
        """Check if episode is done (to be overridden)."""
        return False
    
    def get_observation_space(self) -> Any:
        """Get observation space dimensions."""
        # Compute total observation dimension
        obs_dim = 0
        for name, view in self.articulation_views.items():
            obs_dim += view.num_joints * 2  # pos + vel
        
        from gym import spaces
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    
    def get_action_space(self) -> Any:
        """Get action space dimensions."""
        # Total joint commands
        act_dim = 0
        for name, view in self.articulation_views.items():
            act_dim += view.num_joints
        
        from gym import spaces
        return spaces.Box(low=-np.pi, high=np.pi, shape=(act_dim,))
    
    def get_ground_truth_state(self) -> torch.Tensor:
        """Get full ground truth state of simulation."""
        state = []
        
        # Joint states
        for name, view in self.articulation_views.items():
            joint_pos = view.get_joint_positions()
            joint_vel = view.get_joint_velocities()
            joint_torque = view.get_applied_joint_torques()
            state.extend(joint_pos)
            state.extend(joint_vel)
            state.extend(joint_torque)
        
        return torch.tensor(state, dtype=torch.float32)
    
    def close(self):
        """Close Isaac Sim."""
        self.simulation_app.close()
        print("Isaac Sim closed")
    
    def add_ros_bridge(self, bridge_config: Dict):
        """Add ROS 2 bridge for communication."""
        try:
            from omni.isaac.ros_bridge import ROSBaseBridge
            self.ros_bridge = ROSBaseBridge(bridge_config)
            print("ROS bridge added")
        except ImportError:
            warnings.warn("ROS bridge not available")
    
    def get_prim_path(self, name: str) -> Optional[str]:
        """Get prim path by name."""
        for path in self.prim_paths:
            if name in path:
                return path
        return None
    
    def set_camera_view(self, camera_path: str, view_matrix: Optional[np.ndarray] = None):
        """Set camera view."""
        from pxr import UsdGeom, Gf
        camera_prim = self.stage.GetPrimAtPath(camera_path)
        if camera_prim:
            camera = UsdGeom.Camera(camera_prim)
            if view_matrix is not None:
                # Convert to Gf.Matrix4d
                gf_matrix = Gf.Matrix4d(view_matrix.tolist())
                camera.SetTransform(gf_matrix)


# ======================================================================
# Isaac Lab Environment Wrapper
# ======================================================================

class IsaacLabWrapper(BaseEnvironment):
    """
    Wrapper for NVIDIA Isaac Lab environments.
    
    Isaac Lab provides GPU-accelerated reinforcement learning environments
    with manager-based design.
    """
    
    def __init__(
        self,
        name: str,
        env_class: str,
        cfg: Optional[Any] = None,
        headless: bool = False,
        device: str = 'cuda',
        **kwargs
    ):
        """
        Args:
            name: Environment name
            env_class: Isaac Lab environment class or name
            cfg: Environment configuration
            headless: Run without rendering
            device: Computation device
            **kwargs: Additional arguments
        """
        if not ISAAC_LAB_AVAILABLE:
            raise ImportError("Isaac Lab not available")
        
        super().__init__(name, **kwargs)
        
        self.env_class = env_class
        self.headless = headless
        self.device = device
        
        # Create configuration
        if cfg is None:
            cfg = self._create_default_config()
        self.cfg = cfg
        
        # Create environment
        self.env = self._create_env()
        
        # Get dimensions
        self.num_envs = self.env.num_envs if hasattr(self.env, 'num_envs') else 1
        self.observation_dim = self.env.observation_manager.group_obs_dim['policy'][0]
        self.action_dim = self.env.action_manager.action_term_dim
        
        # State tracking
        self.current_obs = None
        self.current_reward = None
        self.current_done = None
        
        print(f"Isaac Lab environment '{name}' initialized with {self.num_envs} parallel envs")
    
    def _create_default_config(self):
        """Create default configuration."""
        from isaaclab.envs import ManagerBasedEnvCfg
        
        @configclass
        class DefaultEnvCfg(ManagerBasedEnvCfg):
            def __init__(self):
                # Simulation settings
                self.decimation = 2
                self.episode_length_s = 10.0
                
                # Viewer settings
                self.viewer = None
                
                # Physics settings
                self.physics_dt = 1/60
                self.rendering_dt = 1/30
        
        return DefaultEnvCfg()
    
    def _create_env(self):
        """Create Isaac Lab environment."""
        from isaaclab.envs import ManagerBasedEnv
        
        if isinstance(self.env_class, str):
            # Try to import from isaaclab_tasks
            try:
                from isaaclab_tasks.utils import import_env
                env_class = import_env(self.env_class)
            except ImportError:
                # Fall back to manager-based env
                env_class = ManagerBasedEnv
        else:
            env_class = self.env_class
        
        # Create environment
        env = env_class(self.cfg)
        
        return env
    
    def reset(self) -> torch.Tensor:
        """Reset environment."""
        obs_dict = self.env.reset()
        
        # Extract observation
        if isinstance(obs_dict, dict):
            self.current_obs = obs_dict['policy']
        else:
            self.current_obs = obs_dict
        
        self.current_step = 0
        self.episode_reward = 0.0
        
        return self.current_obs
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Step environment."""
        # Ensure action is on correct device
        if isinstance(action, torch.Tensor):
            action = action.to(self.device)
        
        # Step
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        
        # Extract observation
        if isinstance(obs_dict, dict):
            self.current_obs = obs_dict['policy']
        else:
            self.current_obs = obs_dict
        
        self.current_reward = reward
        self.current_done = done
        self.current_step += 1
        self.episode_reward += reward.mean().item() if isinstance(reward, torch.Tensor) else reward
        
        return self.current_obs, reward.mean().item() if isinstance(reward, torch.Tensor) else reward, done, info
    
    def render(self, mode: str = 'human'):
        """Render environment."""
        if not self.headless and hasattr(self.env, 'render'):
            return self.env.render(mode)
    
    def close(self):
        """Close environment."""
        if hasattr(self.env, 'close'):
            self.env.close()
    
    def get_observation_space(self) -> Any:
        """Get observation space."""
        from gym import spaces
        return spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_dim,))
    
    def get_action_space(self) -> Any:
        """Get action space."""
        from gym import spaces
        return spaces.Box(low=-1.0, high=1.0, shape=(self.action_dim,))
    
    def get_ground_truth_state(self) -> torch.Tensor:
        """Get ground truth state (from environment)."""
        if hasattr(self.env, 'get_ground_truth_state'):
            return self.env.get_ground_truth_state()
        
        # Fallback: return observation
        return self.current_obs if self.current_obs is not None else torch.zeros(self.observation_dim)
    
    def get_num_envs(self) -> int:
        """Get number of parallel environments."""
        return self.num_envs
    
    def set_seed(self, seed: int):
        """Set random seed."""
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)


# ======================================================================
# Isaac Lab Arena Wrapper (for specific robot benchmarks)
# ======================================================================

class IsaacLabArenaWrapper(IsaacLabWrapper):
    """
    Wrapper for Isaac Lab Arena benchmarks.
    
    Provides specific robot environments from the Isaac Lab Arena collection.
    """
    
    def __init__(
        self,
        name: str,
        task_name: str,
        robot_name: str,
        headless: bool = False,
        device: str = 'cuda',
        **kwargs
    ):
        """
        Args:
            name: Environment name
            task_name: Arena task name (e.g., 'Isaac-Lift-Cube-Franka-v0')
            robot_name: Robot name (e.g., 'franka_panda')
            headless: Run without rendering
            device: Computation device
            **kwargs: Additional arguments
        """
        self.task_name = task_name
        self.robot_name = robot_name
        
        # Import arena task
        try:
            from isaaclab_tasks.manager_based.classic.cartpole import mdp
            from isaaclab_tasks.manager_based.manipulation.lift import mdp as lift_mdp
            from isaaclab_tasks.manager_based.locomotion.velocity import mdp as velocity_mdp
            
            if 'lift' in task_name.lower():
                from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
                self.task_cfg = LiftEnvCfg()
            elif 'cartpole' in task_name.lower():
                from isaaclab_tasks.manager_based.classic.cartpole.cartpole_env_cfg import CartpoleEnvCfg
                self.task_cfg = CartpoleEnvCfg()
            elif 'velocity' in task_name.lower():
                from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import VelocityEnvCfg
                self.task_cfg = VelocityEnvCfg()
            else:
                raise ValueError(f"Unknown task: {task_name}")
                
        except ImportError:
            warnings.warn(f"Task {task_name} not found in isaaclab_tasks")
            self.task_cfg = None
        
        super().__init__(name, None, self.task_cfg, headless, device, **kwargs)
    
    def _create_env(self):
        """Create arena environment."""
        from isaaclab.envs import ManagerBasedRLEnv
        
        # Set robot name in config
        if hasattr(self.task_cfg, 'scene'):
            if hasattr(self.task_cfg.scene, 'robot'):
                self.task_cfg.scene.robot = self.robot_name
        
        # Create environment
        env = ManagerBasedRLEnv(self.task_cfg)
        
        return env


# ======================================================================
# GR00T Integration with Isaac
# ======================================================================

class GR00TIsaacWrapper(IsaacLabWrapper):
    """
    Wrapper that integrates GR00T foundation model with Isaac environments.
    
    This provides a high-level interface for GR00T-powered robots in Isaac.
    """
    
    def __init__(
        self,
        name: str,
        gr00t_agent: Any,  # GR00T agent from agents.gr00t_agent
        base_env_cfg: Any,
        headless: bool = False,
        device: str = 'cuda',
        **kwargs
    ):
        """
        Args:
            name: Environment name
            gr00t_agent: GR00T agent instance
            base_env_cfg: Base environment configuration
            headless: Run without rendering
            device: Computation device
            **kwargs: Additional arguments
        """
        super().__init__(name, None, base_env_cfg, headless, device, **kwargs)
        
        self.gr00t_agent = gr00t_agent
        
        # GR00T-specific observation processing
        self.vision_encoder = None
        self.language_encoder = None
        
        print(f"GR00T-Isaac environment '{name}' initialized")
    
    def get_gr00t_observation(self) -> Dict:
        """
        Get observation in format expected by GR00T.
        
        Returns:
            Dictionary with 'image', 'text', and/or 'state' keys
        """
        obs = {}
        
        # Get camera images
        if hasattr(self.env, 'render_rgb'):
            obs['image'] = self.env.render_rgb()
        
        # Get robot state
        obs['state'] = self.get_ground_truth_state()
        
        # Get task description (if available)
        if hasattr(self.env, 'get_task_description'):
            obs['text'] = self.env.get_task_description()
        
        return obs
    
    def step_gr00t(self, instruction: Optional[str] = None) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Step using GR00T agent with optional language instruction.
        
        Args:
            instruction: Optional text instruction
        
        Returns:
            observation, reward, done, info
        """
        # Get GR00T-compatible observation
        gr00t_obs = self.get_gr00t_observation()
        
        if instruction:
            gr00t_obs['text'] = instruction
        
        # Get action from GR00T agent
        action = self.gr00t_agent.act(gr00t_obs)
        
        # Step environment
        return self.step(action)


# ======================================================================
# Domain Randomization for Isaac
# ======================================================================

@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization in Isaac."""
    
    # Physics randomization
    friction_range: Tuple[float, float] = (0.5, 1.5)
    restitution_range: Tuple[float, float] = (0.0, 0.5)
    mass_scale_range: Tuple[float, float] = (0.8, 1.2)
    gravity_range: Tuple[float, float] = (-10.0, -9.5)
    
    # Appearance randomization
    lighting_range: Tuple[float, float] = (0.5, 1.5)
    texture_randomization: bool = False
    
    # Robot randomization
    joint_offset_range: Tuple[float, float] = (-0.1, 0.1)
    joint_damping_range: Tuple[float, float] = (0.8, 1.2)
    
    # Timing
    randomize_every_n_steps: int = 100


class DomainRandomizedIsaacWrapper(IsaacSimWrapper):
    """
    Isaac Sim wrapper with domain randomization for robust zeroing.
    """
    
    def __init__(
        self,
        name: str,
        usd_path: str,
        dr_config: Optional[DomainRandomizationConfig] = None,
        **kwargs
    ):
        super().__init__(name, usd_path, **kwargs)
        
        self.dr_config = dr_config or DomainRandomizationConfig()
        self.randomization_step = 0
        
        # Store original parameters
        self.original_params = self._get_physics_params()
    
    def _get_physics_params(self) -> Dict:
        """Get current physics parameters."""
        params = {}
        
        # Get friction from all prims
        for path, prim in self.prim_paths.items():
            if prim.HasAPI('PhysicsMaterialAPI'):
                from pxr import UsdPhysics
                material = UsdPhysics.MaterialAPI(prim)
                friction = material.GetFrictionAttr().Get()
                params[f'friction_{path}'] = friction
        
        return params
    
    def _randomize_physics(self):
        """Apply domain randomization to physics parameters."""
        import random
        
        # Randomize friction
        for path, prim in self.prim_paths.items():
            if prim.HasAPI('PhysicsMaterialAPI'):
                from pxr import UsdPhysics, Sdf
                material = UsdPhysics.MaterialAPI(prim)
                friction_attr = material.GetFrictionAttr()
                if friction_attr:
                    new_friction = random.uniform(*self.dr_config.friction_range)
                    friction_attr.Set(new_friction)
        
        # Randomize gravity
        new_gravity = random.uniform(*self.dr_config.gravity_range)
        physx.set_gravity([0, 0, new_gravity])
        
        print(f"Domain randomization applied at step {self.randomization_step}")
    
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Step with periodic domain randomization."""
        # Apply randomization periodically
        if self.control_step_count % self.dr_config.randomize_every_n_steps == 0:
            self._randomize_physics()
            self.randomization_step += 1
        
        return super().step(action)
    
    def reset_randomization(self):
        """Reset to original parameters."""
        # Restore original parameters
        # (implementation would restore saved values)
        pass


# ======================================================================
# Isaac to GRA Multiverse Bridge
# ======================================================================

def isaac_to_multiverse(
    env: IsaacSimWrapper,
    robot_names: Optional[List[str]] = None
) -> Multiverse:
    """
    Create a GRA multiverse from an Isaac environment.
    
    This creates subsystems for:
        - Each robot joint (level 0)
        - Each robot (level 1)
        - Environment (level 2)
        - Task (level 3)
        - Safety (level 4)
    """
    mv = Multiverse(name=f"{env.name}_multiverse", max_level=4)
    
    from ..core.subsystem import Subsystem
    
    class IsaacJointSubsystem(Subsystem):
        def __init__(self, idx, env, robot_name, joint_idx):
            super().__init__(idx, None, None)
            self.env = env
            self.robot_name = robot_name
            self.joint_idx = joint_idx
        
        def get_state(self):
            # Get joint state from environment
            # This would need to be implemented based on how joint states are accessed
            return torch.zeros(3)  # placeholder
        
        def set_state(self, state):
            pass
        
        def step(self, dt, action=None):
            pass
    
    class IsaacRobotSubsystem(Subsystem):
        def __init__(self, idx, env, robot_name):
            super().__init__(idx, None, None)
            self.env = env
            self.robot_name = robot_name
        
        def get_state(self):
            # Get full robot state
            return env.get_ground_truth_state()
        
        def set_state(self, state):
            pass
        
        def step(self, dt, action=None):
            pass
    
    # Add robot subsystems
    if robot_names:
        for robot_name in robot_names:
            # Robot level (G1)
            robot_idx = MultiIndex((robot_name, None, None, None, None))
            mv.add_subsystem(IsaacRobotSubsystem(robot_idx, env, robot_name))
            
            # Joint level (G0) - would need joint info
            # for joint in robot_joints:
            #     joint_idx = MultiIndex((joint, robot_name, None, None, None))
            #     mv.add_subsystem(IsaacJointSubsystem(joint_idx, env, robot_name, joint_idx))
    
    # Add environment level (G2)
    env_idx = MultiIndex((None, None, env.name, None, None))
    mv.add_subsystem(IsaacRobotSubsystem(env_idx, env, env.name))
    
    return mv


# ======================================================================
# Utility Functions
# ======================================================================

def create_isaac_environment(
    env_type: str,
    name: str,
    **kwargs
) -> BaseEnvironment:
    """
    Create Isaac environment by type.
    
    Args:
        env_type: 'sim', 'lab', 'arena', 'gr00t'
        name: Environment name
        **kwargs: Environment-specific arguments
    
    Returns:
        Isaac environment wrapper
    """
    if env_type == 'sim':
        return IsaacSimWrapper(name, **kwargs)
    elif env_type == 'lab':
        return IsaacLabWrapper(name, **kwargs)
    elif env_type == 'arena':
        return IsaacLabArenaWrapper(name, **kwargs)
    elif env_type == 'gr00t':
        return GR00TIsaacWrapper(name, **kwargs)
    else:
        raise ValueError(f"Unknown Isaac environment type: {env_type}")


def load_isaac_checkpoint(env: IsaacLabWrapper, checkpoint_path: str):
    """Load checkpoint into Isaac Lab environment."""
    if hasattr(env.env, 'load_checkpoint'):
        env.env.load_checkpoint(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        warnings.warn("Environment does not support checkpoint loading")


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing Isaac Environment Wrappers ===\n")
    
    if not ISAAC_SIM_AVAILABLE:
        print("Isaac Sim not available. Please install from NVIDIA Omniverse.")
        exit()
    
    # Test Isaac Sim wrapper (with dummy USD)
    print("Testing Isaac Sim wrapper...")
    try:
        # This would need a valid USD file
        env = IsaacSimWrapper(
            name="test_isaac",
            usd_path="dummy.usd",  # Replace with actual USD
            headless=True
        )
        print("  Isaac Sim wrapper created")
        env.close()
    except Exception as e:
        print(f"  Isaac Sim test failed: {e}")
    
    # Test Isaac Lab wrapper
    if ISAAC_LAB_AVAILABLE:
        print("\nTesting Isaac Lab wrapper...")
        try:
            env = IsaacLabWrapper(
                name="test_lab",
                env_class="Isaac-Cartpole-v0",
                headless=True
            )
            print(f"  Observation dim: {env.observation_dim}")
            print(f"  Action dim: {env.action_dim}")
            env.close()
        except Exception as e:
            print(f"  Isaac Lab test failed: {e}")
    
    print("\nNote: These tests require proper Isaac Sim/Lab installation and valid USD files.")
    print("See NVIDIA documentation for setup instructions.")
```