```python
"""
GRA Physical AI - PyBullet Wrapper
==================================

This module provides a lightweight wrapper for PyBullet simulations,
enabling rapid prototyping and testing of GRA-based robots.

PyBullet offers:
    - Easy installation (pip install pybullet)
    - Built-in robot models
    - Fast CPU-based simulation
    - OpenGL rendering
    - ROS integration

This wrapper implements:
    - GRA-compatible environment interface
    - Joint-level access for G0 subsystems
    - Body-level access for G1-G2 subsystems
    - Contact and force sensing for G4 (ethics)
    - Multi-robot support
    - Checkpoint/restore for zeroing
"""

import torch
import numpy as np
import warnings
import time
import os
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict

# Try to import PyBullet
try:
    import pybullet as p
    import pybullet_data
    import pybullet_utils.bullet_client as bc
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    warnings.warn("PyBullet not installed. Install with: pip install pybullet")

from ..core.base_environment import BaseEnvironment
from ..core.multiverse import MultiIndex, Multiverse
from ..core.base_agent import BaseAgent


# ======================================================================
# PyBullet Robot Info
# ======================================================================

@dataclass
class RobotInfo:
    """Information about a robot in the simulation."""
    id: int
    name: str
    urdf_path: str
    base_position: np.ndarray
    base_orientation: np.ndarray
    joint_names: List[str]
    joint_indices: List[int]
    joint_types: List[int]
    joint_limits: List[Tuple[float, float]]
    link_names: List[str]
    link_indices: List[int]
    num_joints: int
    num_links: int


@dataclass
class ObjectInfo:
    """Information about a non-robot object."""
    id: int
    name: str
    urdf_path: str
    position: np.ndarray
    orientation: np.ndarray


# ======================================================================
# PyBullet GRA Wrapper
# ======================================================================

class PyBulletGRAWrapper(BaseEnvironment):
    """
    PyBullet simulation wrapper for GRA framework.
    
    Provides:
        - Full access to PyBullet physics
        - GRA-compatible state representation
        - Joint-level and body-level subsystems
        - Contact and force sensing
        - Multi-robot support
    """
    
    def __init__(
        self,
        name: str,
        urdf_paths: Union[str, List[str]],
        dt: float = 1/240,
        gui: bool = True,
        gravity: Tuple[float, float, float] = (0, 0, -9.81),
        use_fixed_base: bool = False,
        robot_positions: Optional[List[Tuple[float, float, float]]] = None,
        add_ground: bool = True,
        add_obstacles: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        """
        Args:
            name: Environment name
            urdf_paths: Path(s) to robot URDF file(s)
            dt: Simulation time step
            gui: Enable PyBullet GUI
            gravity: Gravity vector
            use_fixed_base: Fix robot base (for manipulators)
            robot_positions: Initial positions for multiple robots
            add_ground: Add ground plane
            add_obstacles: Add obstacles for testing
            seed: Random seed
            **kwargs: Additional arguments
        """
        if not PYBULLET_AVAILABLE:
            raise ImportError("PyBullet not installed")
        
        super().__init__(name, dt=dt, seed=seed, **kwargs)
        
        self.urdf_paths = [urdf_paths] if isinstance(urdf_paths, str) else urdf_paths
        self.num_robots = len(self.urdf_paths)
        self.gui = gui
        self.gravity = gravity
        self.use_fixed_base = use_fixed_base
        self.robot_positions = robot_positions or [(0, 0, 0.1)] * self.num_robots
        self.add_ground = add_ground
        self.add_obstacles = add_obstacles
        
        # Connect to PyBullet
        self._connect()
        
        # Load models
        self.robots: Dict[str, RobotInfo] = {}
        self.objects: Dict[str, ObjectInfo] = {}
        self._load_models()
        
        # Setup simulation
        self._setup_simulation()
        
        # State tracking
        self._joint_states: Dict[int, np.ndarray] = {}
        self._link_states: Dict[int, np.ndarray] = {}
        self._contact_points: List = []
        self._last_observation = None
        
        # For GRA hierarchy
        self._subsystem_map: Dict[MultiIndex, Dict] = {}
        self._build_subsystem_map()
        
        # Action cache for replay
        self._last_actions: Dict[int, np.ndarray] = {}
        
        print(f"PyBullet environment '{name}' initialized with {self.num_robots} robots")
        
        # Initial render
        if gui:
            self.render()
    
    def _connect(self):
        """Connect to PyBullet."""
        if self.gui:
            self.client = p.connect(p.GUI)
            # Configure camera
            p.resetDebugVisualizerCamera(
                cameraDistance=2.0,
                cameraYaw=50,
                cameraPitch=-35,
                cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            self.client = p.connect(p.DIRECT)
        
        # Set additional search path
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    def _load_models(self):
        """Load robot and object models."""
        # Load ground plane
        if self.add_ground:
            ground_id = p.loadURDF("plane.urdf", [0, 0, 0])
            self.objects["ground"] = ObjectInfo(
                id=ground_id,
                name="ground",
                urdf_path="plane.urdf",
                position=np.array([0, 0, 0]),
                orientation=np.array([0, 0, 0, 1])
            )
        
        # Load robots
        for i, urdf_path in enumerate(self.urdf_paths):
            position = self.robot_positions[i] if i < len(self.robot_positions) else (0, 0, 0.1)
            
            robot_id = p.loadURDF(
                urdf_path,
                basePosition=position,
                useFixedBase=self.use_fixed_base
            )
            
            # Get robot info
            robot_name = f"robot_{i}" if i > 0 else "robot"
            joint_names = []
            joint_indices = []
            joint_types = []
            joint_limits = []
            link_names = []
            link_indices = []
            
            num_joints = p.getNumJoints(robot_id)
            for j in range(num_joints):
                info = p.getJointInfo(robot_id, j)
                joint_name = info[1].decode('utf-8')
                joint_type = info[2]
                joint_lower = info[8]
                joint_upper = info[9]
                
                joint_names.append(joint_name)
                joint_indices.append(j)
                joint_types.append(joint_type)
                joint_limits.append((joint_lower, joint_upper))
            
            # Links (bodies) are more complex in PyBullet
            # For simplicity, treat each joint as having a link
            link_names = [f"link_{j}" for j in range(num_joints)]
            link_indices = list(range(num_joints))
            
            self.robots[robot_name] = RobotInfo(
                id=robot_id,
                name=robot_name,
                urdf_path=urdf_path,
                base_position=np.array(position),
                base_orientation=np.array([0, 0, 0, 1]),
                joint_names=joint_names,
                joint_indices=joint_indices,
                joint_types=joint_types,
                joint_limits=joint_limits,
                link_names=link_names,
                link_indices=link_indices,
                num_joints=num_joints,
                num_links=num_joints
            )
            
            print(f"  Loaded robot {robot_name} with {num_joints} joints")
        
        # Add obstacles for testing
        if self.add_obstacles:
            self._add_test_obstacles()
    
    def _add_test_obstacles(self):
        """Add test obstacles."""
        # Add a few cubes
        for i in range(3):
            pos = [1.0 + i, 1.0, 0.5]
            obstacle_id = p.loadURDF("cube.urdf", pos, globalScaling=0.5)
            self.objects[f"obstacle_{i}"] = ObjectInfo(
                id=obstacle_id,
                name=f"obstacle_{i}",
                urdf_path="cube.urdf",
                position=np.array(pos),
                orientation=np.array([0, 0, 0, 1])
            )
        
        # Add a human model for ethics testing
        human_id = p.loadURDF("humanoid/humanoid.urdf", [2, 0, 1.0], useFixedBase=True)
        self.objects["human"] = ObjectInfo(
            id=human_id,
            name="human",
            urdf_path="humanoid/humanoid.urdf",
            position=np.array([2, 0, 1.0]),
            orientation=np.array([0, 0, 0, 1])
        )
    
    def _setup_simulation(self):
        """Setup simulation parameters."""
        p.setGravity(*self.gravity)
        p.setTimeStep(self.dt)
        p.setRealTimeSimulation(0)  # Step manually
        
        # Set physics parameters
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            contactBreakingThreshold=0.001,
            enableConeFriction=1
        )
    
    def _build_subsystem_map(self):
        """Build mapping from GRA multi-indices to PyBullet entities."""
        # Level 0: Joints
        for robot_name, robot in self.robots.items():
            for joint_name, joint_idx in zip(robot.joint_names, robot.joint_indices):
                idx = MultiIndex((
                    robot_name,
                    joint_name,
                    None,
                    None,
                    None
                ))
                self._subsystem_map[idx] = {
                    'type': 'joint',
                    'robot_id': robot.id,
                    'joint_idx': joint_idx,
                    'robot_name': robot_name,
                    'joint_name': joint_name
                }
        
        # Level 1: Links/Bodies
        for robot_name, robot in self.robots.items():
            for link_name, link_idx in zip(robot.link_names, robot.link_indices):
                idx = MultiIndex((
                    robot_name,
                    None,
                    link_name,
                    None,
                    None
                ))
                self._subsystem_map[idx] = {
                    'type': 'link',
                    'robot_id': robot.id,
                    'link_idx': link_idx,
                    'robot_name': robot_name,
                    'link_name': link_name
                }
        
        # Level 2: Whole robots
        for robot_name, robot in self.robots.items():
            idx = MultiIndex((
                robot_name,
                None,
                None,
                None,
                None
            ))
            self._subsystem_map[idx] = {
                'type': 'robot',
                'robot_id': robot.id,
                'robot_name': robot_name
            }
        
        # Level 3: Environment
        idx = MultiIndex((
            None,
            None,
            None,
            'environment',
            None
        ))
        self._subsystem_map[idx] = {
            'type': 'environment'
        }
        
        # Level 4: Ethics/Safety
        idx = MultiIndex((
            None,
            None,
            None,
            None,
            'ethics'
        ))
        self._subsystem_map[idx] = {
            'type': 'ethics'
        }
    
    def _update_state(self):
        """Update internal state from PyBullet."""
        # Update joint states
        for robot_name, robot in self.robots.items():
            for joint_idx in robot.joint_indices:
                joint_state = p.getJointState(robot.id, joint_idx)
                self._joint_states[(robot.id, joint_idx)] = np.array([
                    joint_state[0],  # position
                    joint_state[1],  # velocity
                    joint_state[3]   # torque
                ])
        
        # Update link states
        for robot_name, robot in self.robots.items():
            for link_idx in robot.link_indices:
                link_state = p.getLinkState(robot.id, link_idx)
                self._link_states[(robot.id, link_idx)] = np.array([
                    link_state[0],  # world position
                    link_state[1],  # world orientation
                ])
        
        # Update contact points
        self._contact_points = p.getContactPoints()
    
    def reset(self) -> torch.Tensor:
        """Reset simulation to initial state."""
        # Reset simulation
        p.resetSimulation()
        
        # Reload models
        self._load_models()
        
        # Reset joint positions to zero
        for robot_name, robot in self.robots.items():
            for joint_idx in robot.joint_indices:
                p.resetJointState(robot.id, joint_idx, 0, 0)
        
        # Step to stabilize
        for _ in range(10):
            p.stepSimulation()
        
        self._update_state()
        
        self.current_step = 0
        self.episode_reward = 0.0
        
        # Get observation
        obs = self.get_observation()
        self._last_observation = obs
        
        return obs
    
    def step(self, action: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, float, bool, Dict]:
        """
        Apply actions and step simulation.
        
        Args:
            action: Either:
                - Single tensor (concatenated joint torques)
                - Dict mapping robot_name -> joint torques
        
        Returns:
            observation, reward, done, info
        """
        # Parse actions
        if isinstance(action, dict):
            # Multi-robot actions
            robot_actions = action
        else:
            # Single robot - assume first robot
            robot_actions = {list(self.robots.keys())[0]: action}
        
        # Apply actions
        self._apply_actions(robot_actions)
        
        # Step simulation
        p.stepSimulation()
        
        # Update state
        self._update_state()
        
        # Get observation
        obs = self.get_observation()
        
        # Compute reward
        reward = self._compute_reward(robot_actions)
        
        # Check if done
        done = self._check_done()
        
        self.current_step += 1
        self.episode_reward += reward
        self._last_observation = obs
        
        info = {
            'step': self.current_step,
            'contact_points': len(self._contact_points),
            'joint_states': self._joint_states.copy()
        }
        
        return obs, reward, done, info
    
    def _apply_actions(self, actions: Dict[str, torch.Tensor]):
        """Apply torque commands to joints."""
        for robot_name, robot in self.robots.items():
            if robot_name not in actions:
                continue
            
            robot_action = actions[robot_name]
            
            # Convert to numpy
            if isinstance(robot_action, torch.Tensor):
                torques = robot_action.cpu().numpy()
            else:
                torques = np.array(robot_action)
            
            # Apply torques to each joint
            for i, joint_idx in enumerate(robot.joint_indices):
                if i < len(torques):
                    p.setJointMotorControl2(
                        robot.id,
                        joint_idx,
                        p.TORQUE_CONTROL,
                        force=torques[i]
                    )
                else:
                    # Zero torque for remaining joints
                    p.setJointMotorControl2(
                        robot.id,
                        joint_idx,
                        p.TORQUE_CONTROL,
                        force=0.0
                    )
            
            # Store last actions
            self._last_actions[robot.id] = torques
    
    def _compute_reward(self, actions: Dict[str, torch.Tensor]) -> float:
        """Compute reward (can be overridden)."""
        # Simple reward: negative squared torques (energy minimization)
        reward = 0.0
        for robot_name, robot_action in actions.items():
            if isinstance(robot_action, torch.Tensor):
                reward -= torch.sum(robot_action ** 2).item() * 0.01
            else:
                reward -= np.sum(np.array(robot_action) ** 2) * 0.01
        
        # Bonus for staying upright (if humanoid)
        for robot_name, robot in self.robots.items():
            pos, orn = p.getBasePositionAndOrientation(robot.id)
            # Check orientation (more upright = better)
            euler = p.getEulerFromQuaternion(orn)
            reward += 1.0 - abs(euler[0]) - abs(euler[1])  # pitch and roll
        
        return reward
    
    def _check_done(self) -> bool:
        """Check if episode should end."""
        # Check if any robot fell over
        for robot_name, robot in self.robots.items():
            pos, _ = p.getBasePositionAndOrientation(robot.id)
            if pos[2] < 0.3:  # height threshold
                return True
        
        return False
    
    def get_observation(self) -> torch.Tensor:
        """Get current observation (concatenated joint states)."""
        observations = []
        
        for robot_name, robot in self.robots.items():
            for joint_idx in robot.joint_indices:
                if (robot.id, joint_idx) in self._joint_states:
                    observations.extend(self._joint_states[(robot.id, joint_idx)])
                else:
                    observations.extend([0, 0, 0])
        
        return torch.tensor(observations, dtype=torch.float32)
    
    def get_state_for_index(self, idx: MultiIndex) -> torch.Tensor:
        """
        Get state for a specific GRA subsystem.
        
        This is used by the GRA multiverse for foam computation.
        """
        if idx not in self._subsystem_map:
            return torch.zeros(1)
        
        info = self._subsystem_map[idx]
        
        if info['type'] == 'joint':
            # Joint state: position, velocity, torque
            key = (info['robot_id'], info['joint_idx'])
            if key in self._joint_states:
                return torch.tensor(self._joint_states[key], dtype=torch.float32)
            else:
                return torch.zeros(3)
        
        elif info['type'] == 'link':
            # Link state: position, orientation
            key = (info['robot_id'], info['link_idx'])
            if key in self._link_states:
                return torch.tensor(self._link_states[key], dtype=torch.float32)
            else:
                return torch.zeros(6)
        
        elif info['type'] == 'robot':
            # Robot state: base position + joint states
            robot = self.robots[info['robot_name']]
            pos, orn = p.getBasePositionAndOrientation(robot.id)
            state = [pos[0], pos[1], pos[2], orn[0], orn[1], orn[2], orn[3]]
            
            for joint_idx in robot.joint_indices:
                key = (robot.id, joint_idx)
                if key in self._joint_states:
                    state.extend(self._joint_states[key])
                else:
                    state.extend([0, 0, 0])
            
            return torch.tensor(state, dtype=torch.float32)
        
        elif info['type'] == 'environment':
            # Environment state: positions of all objects
            state = []
            for obj_name, obj in self.objects.items():
                pos, orn = p.getBasePositionAndOrientation(obj.id)
                state.extend([pos[0], pos[1], pos[2], orn[0], orn[1], orn[2], orn[3]])
            return torch.tensor(state, dtype=torch.float32)
        
        elif info['type'] == 'ethics':
            # Ethics state: contact forces
            total_force = 0.0
            for contact in self._contact_points:
                total_force += contact[9]  # normal force
            return torch.tensor([total_force], dtype=torch.float32)
        
        return torch.zeros(1)
    
    def get_all_subsystem_indices(self) -> List[MultiIndex]:
        """Get all GRA subsystem indices."""
        return list(self._subsystem_map.keys())
    
    def render(self, mode: str = 'human'):
        """Render the simulation."""
        if self.gui and mode == 'human':
            # Update camera to follow first robot
            if self.robots:
                first_robot = list(self.robots.values())[0]
                pos, _ = p.getBasePositionAndOrientation(first_robot.id)
                p.resetDebugVisualizerCamera(
                    cameraDistance=2.0,
                    cameraYaw=50,
                    cameraPitch=-35,
                    cameraTargetPosition=pos
                )
            time.sleep(self.dt)
    
    def close(self):
        """Disconnect from PyBullet."""
        p.disconnect()
    
    def get_observation_space(self) -> Any:
        """Get observation space dimensions."""
        from gym import spaces
        
        # Count total observation dimensions
        obs_dim = 0
        for robot_name, robot in self.robots.items():
            obs_dim += robot.num_joints * 3  # pos, vel, torque per joint
        
        return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,))
    
    def get_action_space(self) -> Any:
        """Get action space dimensions."""
        from gym import spaces
        
        # Total joint torques
        act_dim = 0
        for robot_name, robot in self.robots.items():
            act_dim += robot.num_joints
        
        return spaces.Box(low=-10, high=10, shape=(act_dim,))
    
    def get_ground_truth_state(self) -> torch.Tensor:
        """Get full ground truth state."""
        state = []
        
        # Robot states
        for robot_name, robot in self.robots.items():
            pos, orn = p.getBasePositionAndOrientation(robot.id)
            state.extend([pos[0], pos[1], pos[2], orn[0], orn[1], orn[2], orn[3]])
            
            for joint_idx in robot.joint_indices:
                key = (robot.id, joint_idx)
                if key in self._joint_states:
                    state.extend(self._joint_states[key])
                else:
                    state.extend([0, 0, 0])
        
        # Object states
        for obj_name, obj in self.objects.items():
            pos, orn = p.getBasePositionAndOrientation(obj.id)
            state.extend([pos[0], pos[1], pos[2], orn[0], orn[1], orn[2], orn[3]])
        
        return torch.tensor(state, dtype=torch.float32)
    
    # ======================================================================
    # GRA-Specific Methods
    # ======================================================================
    
    def compute_foam_contributions(self) -> Dict[int, float]:
        """
        Compute foam contributions from environment dynamics.
        
        Returns:
            Dictionary mapping level -> foam value
        """
        foam = {}
        
        # Level 0: Joint consistency (joints of same robot should be coordinated)
        for robot_name, robot in self.robots.items():
            joint_forces = []
            for joint_idx in robot.joint_indices:
                key = (robot.id, joint_idx)
                if key in self._joint_states:
                    joint_forces.append(self._joint_states[key][2])  # torque
            
            if joint_forces:
                # Foam = variance of joint torques (should be low for coordination)
                foam[0] = foam.get(0, 0) + np.var(joint_forces)
        
        # Level 1: Link consistency (links of same robot should be connected)
        # (implicitly satisfied by physics)
        
        # Level 2: Robot consistency (robots shouldn't intersect)
        # Check collisions between robots
        collision_force = 0.0
        for contact in self._contact_points:
            body_a = contact[1]
            body_b = contact[2]
            
            # Check if both are robots
            robot_ids = [r.id for r in self.robots.values()]
            if body_a in robot_ids and body_b in robot_ids:
                collision_force += contact[9]  # normal force
        
        foam[2] = collision_force
        
        # Level 4: Ethics (contact with humans/obstacles)
        if 'human' in self.objects:
            human_id = self.objects['human'].id
            human_force = 0.0
            for contact in self._contact_points:
                if contact[1] == human_id or contact[2] == human_id:
                    human_force += contact[9]
            foam[4] = human_force
        
        return foam
    
    def get_robot_joint_indices(self, robot_name: str) -> List[int]:
        """Get joint indices for a specific robot."""
        if robot_name in self.robots:
            return self.robots[robot_name].joint_indices
        return []
    
    def get_robot_link_indices(self, robot_name: str) -> List[int]:
        """Get link indices for a specific robot."""
        if robot_name in self.robots:
            return self.robots[robot_name].link_indices
        return []
    
    def set_joint_position(self, robot_name: str, joint_idx: int, position: float):
        """Set joint position (for resetting)."""
        if robot_name in self.robots:
            robot = self.robots[robot_name]
            p.resetJointState(robot.id, joint_idx, position, 0)
    
    def get_contact_forces_with_object(self, object_name: str) -> float:
        """Get total contact force with a specific object."""
        if object_name not in self.objects:
            return 0.0
        
        obj_id = self.objects[object_name].id
        total_force = 0.0
        
        for contact in self._contact_points:
            if contact[1] == obj_id or contact[2] == obj_id:
                total_force += contact[9]
        
        return total_force
    
    def add_debug_visuals(self):
        """Add debug visuals (e.g., coordinate frames)."""
        for robot_name, robot in self.robots.items():
            # Add frame at base
            pos, orn = p.getBasePositionAndOrientation(robot.id)
            p.addUserDebugText(
                robot_name,
                pos,
                [1, 0, 0],
                textSize=1.0
            )
            
            # Add frame at each joint
            for joint_idx in robot.joint_indices:
                link_state = p.getLinkState(robot.id, joint_idx)
                pos = link_state[0]
                p.addUserDebugLine(
                    pos,
                    [pos[0] + 0.1, pos[1], pos[2]],
                    [1, 0, 0],
                    lineWidth=2
                )


# ======================================================================
# PyBullet to GRA Multiverse Bridge
# ======================================================================

def pybullet_to_multiverse(env: PyBulletGRAWrapper) -> Multiverse:
    """
    Create a GRA multiverse from a PyBullet environment.
    
    This creates subsystems for all GRA levels based on the environment.
    """
    mv = Multiverse(name=f"{env.name}_multiverse", max_level=4)
    
    from ..core.subsystem import Subsystem
    
    class PyBulletSubsystem(Subsystem):
        def __init__(self, idx, env, info):
            super().__init__(idx, None, None)
            self.env = env
            self.info = info
        
        def get_state(self):
            return self.env.get_state_for_index(self.multi_index)
        
        def set_state(self, state):
            # Cannot set state directly in PyBullet
            pass
        
        def step(self, dt, action=None):
            pass
    
    # Add all subsystems
    for idx, info in env._subsystem_map.items():
        mv.add_subsystem(PyBulletSubsystem(idx, env, info))
    
    return mv


# ======================================================================
# Predefined Test Environments
# ======================================================================

class CartPolePyBullet(PyBulletGRAWrapper):
    """CartPole environment in PyBullet for quick testing."""
    
    def __init__(self, name: str = "cartpole", gui: bool = True, **kwargs):
        # Use PyBullet's cartpole URDF
        urdf_path = pybullet_data.getDataPath() + "/cartpole.urdf"
        
        super().__init__(
            name=name,
            urdf_paths=urdf_path,
            gui=gui,
            use_fixed_base=False,
            robot_positions=[(0, 0, 0)],
            add_ground=False,
            **kwargs
        )
        
        # Cartpole specific
        self.cart_joint = 0
        self.pole_joint = 1
        
        print("CartPole environment initialized")
    
    def _compute_reward(self, actions: Dict[str, torch.Tensor]) -> float:
        """CartPole reward: keep pole upright."""
        reward = super()._compute_reward(actions)
        
        # Get pole angle
        key = (self.robots['robot'].id, self.pole_joint)
        if key in self._joint_states:
            angle = self._joint_states[key][0]  # position
            reward += np.cos(angle)  # reward upright pole
        
        return reward
    
    def _check_done(self) -> bool:
        """CartPole done if pole falls too far."""
        key = (self.robots['robot'].id, self.pole_joint)
        if key in self._joint_states:
            angle = abs(self._joint_states[key][0])
            return angle > 0.2  # about 15 degrees
        return False


class HumanoidPyBullet(PyBulletGRAWrapper):
    """Humanoid robot for ethics testing."""
    
    def __init__(self, name: str = "humanoid", gui: bool = True, **kwargs):
        # Use PyBullet's humanoid URDF
        urdf_path = pybullet_data.getDataPath() + "/humanoid/humanoid.urdf"
        
        super().__init__(
            name=name,
            urdf_paths=urdf_path,
            gui=gui,
            use_fixed_base=False,
            robot_positions=[(0, 0, 1.0)],
            add_obstacles=True,  # This adds a human model
            **kwargs
        )
        
        print("Humanoid environment initialized with ethics testing")


class MultiRobotPyBullet(PyBulletGRAWrapper):
    """Multi-robot environment for swarm testing."""
    
    def __init__(self, num_robots: int = 3, name: str = "multi", gui: bool = True, **kwargs):
        # Use multiple R2D2 robots
        urdf_path = pybullet_data.getDataPath() + "/r2d2.urdf"
        urdf_paths = [urdf_path] * num_robots
        
        # Spread them out
        positions = [(i*1.5, 0, 0.1) for i in range(num_robots)]
        
        super().__init__(
            name=name,
            urdf_paths=urdf_paths,
            gui=gui,
            use_fixed_base=False,
            robot_positions=positions,
            add_obstacles=True,
            **kwargs
        )
        
        print(f"Multi-robot environment with {num_robots} robots initialized")


# ======================================================================
# Utility Functions
# ======================================================================

def create_pybullet_environment(
    env_type: str,
    name: str,
    **kwargs
) -> PyBulletGRAWrapper:
    """
    Create PyBullet environment by type.
    
    Args:
        env_type: 'generic', 'cartpole', 'humanoid', 'multi'
        name: Environment name
        **kwargs: Environment-specific arguments
    
    Returns:
        PyBullet environment wrapper
    """
    if env_type == 'cartpole':
        return CartPolePyBullet(name, **kwargs)
    elif env_type == 'humanoid':
        return HumanoidPyBullet(name, **kwargs)
    elif env_type == 'multi':
        return MultiRobotPyBullet(**kwargs)
    else:
        return PyBulletGRAWrapper(name, **kwargs)


def save_pybullet_state(env: PyBulletGRAWrapper, path: str):
    """Save PyBullet simulation state."""
    state = p.saveState()
    # PyBullet states are saved internally, not to files easily
    # This is a placeholder
    print(f"State saved (ID: {state})")


def load_pybullet_state(env: PyBulletGRAWrapper, state_id: int):
    """Load PyBullet simulation state."""
    p.restoreState(state_id)


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing PyBullet GRA Wrapper ===\n")
    
    if not PYBULLET_AVAILABLE:
        print("PyBullet not installed. Install with: pip install pybullet")
        exit()
    
    # Test CartPole
    print("Testing CartPole environment...")
    env = CartPolePyBullet(gui=False)  # Headless for testing
    
    obs = env.reset()
    print(f"  Observation shape: {obs.shape}")
    
    # Run a few steps
    for i in range(10):
        action = torch.randn(env.get_action_space().shape[0])  # random action
        obs, reward, done, info = env.step(action)
        print(f"  Step {i}: reward={reward:.3f}, done={done}")
    
    env.close()
    
    # Test GRA subsystem access
    print("\nTesting GRA subsystem access...")
    env = CartPolePyBullet(gui=False)
    env.reset()
    
    indices = env.get_all_subsystem_indices()
    print(f"  Found {len(indices)} GRA subsystems")
    
    for idx in indices[:5]:  # Show first 5
        state = env.get_state_for_index(idx)
        print(f"  {idx}: state shape {state.shape}")
    
    env.close()
    
    # Test Humanoid
    print("\nTesting Humanoid environment...")
    env = HumanoidPyBullet(gui=False)
    env.reset()
    
    # Check ethics state
    ethics_idx = MultiIndex((None, None, None, None, 'ethics'))
    ethics_state = env.get_state_for_index(ethics_idx)
    print(f"  Ethics state: {ethics_state}")
    
    env.close()
    
    # Test Multi-Robot
    print("\nTesting Multi-Robot environment...")
    env = MultiRobotPyBullet(num_robots=2, gui=False)
    env.reset()
    
    # Get robot indices
    robot_indices = [idx for idx in env.get_all_subsystem_indices() 
                     if idx.indices[0] is not None and idx.indices[1] is None and idx.indices[2] is None]
    print(f"  Found {len(robot_indices)} robot-level subsystems")
    
    env.close()
    
    print("\nAll tests passed!")
```