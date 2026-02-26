```python
"""
GRA Physical AI - Predefined GRA Layers Module
===============================================

This module provides ready-to-use implementations of GRA layers for common robotics applications.
Each layer corresponds to a level in the GRA hierarchy and comes with pre-defined goals,
projectors, and loss functions.

Available layers:
    - G0Layer: Low-level motor control (joint position/velocity/torque)
    - G1Layer: Perception and sensor fusion
    - G2Layer: World model and physics prediction
    - G3Layer: Task planning and execution
    - G4Layer: Ethics and safety (Code of Friends)
    - CombinedLayer: Multiple layers combined hierarchically

These layers can be used as building blocks to quickly assemble a GRA multiverse
for any robot or physical AI system.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import warnings

from ..core.multiverse import MultiIndex, Multiverse
from ..core.subsystem import Subsystem
from ..core.projector import (
    Projector, IdentityProjector, ThresholdProjector, RangeProjector,
    NormProjector, MatrixProjector, CompositeProjector, IntersectionProjector
)
from ..core.goal import Goal
from ..core.base_agent import BaseAgent
from ..agents.base_agent import NullAgent


# ======================================================================
# Base Layer Class
# ======================================================================

class GRA_layer(ABC):
    """
    Abstract base class for a GRA layer.
    
    A layer represents one level in the GRA hierarchy and provides:
        - Subsystem definitions for that level
        - Goals and projectors
        - Methods to connect to lower/higher layers
        - State extraction and formatting
    
    Layers can be combined to form a complete GRA multiverse.
    """
    
    def __init__(
        self,
        level: int,
        name: str,
        parent: Optional['GRA_layer'] = None,
        children: Optional[List['GRA_layer']] = None
    ):
        """
        Args:
            level: GRA level number (0 = lowest)
            name: Layer name
            parent: Higher-level layer containing this layer
            children: Lower-level layers contained in this layer
        """
        self.level = level
        self.name = name
        self.parent = parent
        self.children = children or []
        
        # Goals for this layer (can be multiple)
        self.goals: List[Goal] = []
        
        # Subsystems in this layer
        self.subsystems: Dict[MultiIndex, Subsystem] = {}
        
        # Layer weights (Λ_l)
        self.weight = 1.0
    
    @abstractmethod
    def build_subsystems(self) -> Dict[MultiIndex, Subsystem]:
        """
        Create subsystems for this layer.
        
        Returns:
            Dictionary mapping multi-index to subsystem
        """
        pass
    
    @abstractmethod
    def get_goals(self) -> List[Goal]:
        """
        Get goals for this layer.
        
        Returns:
            List of goals
        """
        pass
    
    def get_state_dim(self) -> int:
        """Get total state dimension of this layer."""
        total = 0
        for subsystem in self.subsystems.values():
            if hasattr(subsystem, 'state_size'):
                total += subsystem.state_size()
            elif hasattr(subsystem, 'get_state'):
                state = subsystem.get_state()
                total += state.numel()
        return total
    
    def extract_states(self) -> Dict[MultiIndex, torch.Tensor]:
        """Extract states from all subsystems."""
        states = {}
        for idx, subsystem in self.subsystems.items():
            if hasattr(subsystem, 'get_state'):
                states[idx] = subsystem.get_state()
        return states
    
    def set_states(self, states: Dict[MultiIndex, torch.Tensor]):
        """Set states for subsystems."""
        for idx, state in states.items():
            if idx in self.subsystems:
                subsystem = self.subsystems[idx]
                if hasattr(subsystem, 'set_state'):
                    subsystem.set_state(state)
    
    def connect(self, parent: Optional['GRA_layer'] = None, 
                children: Optional[List['GRA_layer']] = None):
        """Connect this layer to parent and children."""
        if parent:
            self.parent = parent
        if children:
            self.children = children
    
    def to_multiverse(self) -> Multiverse:
        """
        Convert this layer (and its children) to a complete multiverse.
        
        Returns:
            Multiverse containing this layer and all connected layers
        """
        # Determine max level
        max_level = self._get_max_level()
        
        # Create multiverse
        mv = Multiverse(name=f"{self.name}_multiverse", max_level=max_level)
        
        # Add all subsystems recursively
        self._add_to_multiverse(mv)
        
        # Set goals for each level
        self._set_goals_in_multiverse(mv)
        
        return mv
    
    def _get_max_level(self) -> int:
        """Get maximum level in hierarchy."""
        max_level = self.level
        for child in self.children:
            child_max = child._get_max_level()
            max_level = max(max_level, child_max)
        return max_level
    
    def _add_to_multiverse(self, mv: Multiverse):
        """Add subsystems to multiverse recursively."""
        # Add this layer's subsystems
        for idx, subsystem in self.subsystems.items():
            mv.add_subsystem(subsystem)
        
        # Add children
        for child in self.children:
            child._add_to_multiverse(mv)
    
    def _set_goals_in_multiverse(self, mv: Multiverse):
        """Set goals in multiverse."""
        for i, goal in enumerate(self.get_goals()):
            mv.set_goal(self.level + i, goal)  # Allow multiple goals per level
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(level={self.level}, name='{self.name}')"


# ======================================================================
# G0 Layer: Low-Level Motor Control
# ======================================================================

class G0_Layer(GRA_layer):
    """
    Level 0 layer for low-level motor control.
    
    Provides:
        - Joint-level subsystems (position, velocity, torque)
        - Motor goals (tracking accuracy, limits)
        - Support for different actuator types
    """
    
    def __init__(
        self,
        name: str = "motor_control",
        num_joints: int = 6,
        joint_names: Optional[List[str]] = None,
        joint_limits: Optional[List[Tuple[float, float]]] = None,
        motor_types: Optional[List[str]] = None,  # 'position', 'velocity', 'torque'
        dt: float = 0.01,
        parent: Optional[GRA_layer] = None,
        children: Optional[List[GRA_layer]] = None
    ):
        """
        Args:
            name: Layer name
            num_joints: Number of joints
            joint_names: Custom joint names
            joint_limits: Position limits for each joint [(min, max), ...]
            motor_types: Type of each motor ('position', 'velocity', 'torque')
            dt: Control time step
            parent: Parent layer
            children: Child layers
        """
        super().__init__(level=0, name=name, parent=parent, children=children)
        
        self.num_joints = num_joints
        self.joint_names = joint_names or [f"joint_{i}" for i in range(num_joints)]
        self.joint_limits = joint_limits or [(-np.pi, np.pi) for _ in range(num_joints)]
        self.motor_types = motor_types or ['torque'] * num_joints
        self.dt = dt
        
        # Build subsystems
        self.subsystems = self.build_subsystems()
        
        # Create goals
        self._create_goals()
    
    def build_subsystems(self) -> Dict[MultiIndex, Subsystem]:
        """Create joint subsystems."""
        from ..core.subsystem import SimpleSubsystem
        from ..core.multiverse import EuclideanSpace
        
        subsystems = {}
        
        for i, (joint_name, motor_type, limits) in enumerate(zip(
            self.joint_names, self.motor_types, self.joint_limits
        )):
            # Create multi-index
            idx = MultiIndex((joint_name, None, None, None, None))
            
            # State dimension depends on motor type
            if motor_type == 'position':
                state_dim = 2  # position_cmd, position_actual
            elif motor_type == 'velocity':
                state_dim = 2  # velocity_cmd, velocity_actual
            elif motor_type == 'torque':
                state_dim = 3  # torque_cmd, position_actual, velocity_actual
            else:
                state_dim = 3  # default
            
            # Create subsystem
            class MotorSubsystem(SimpleSubsystem):
                def __init__(self, idx, dim, limits):
                    super().__init__(idx, EuclideanSpace(dim), None)
                    self.limits = limits
                    self._state = torch.zeros(dim)
                
                def get_state(self):
                    return self._state
                
                def set_state(self, state):
                    self._state = state.clone()
                
                def step(self, dt, action=None):
                    # Simple dynamics simulation (for testing)
                    if action is not None:
                        # Apply action (torque or velocity command)
                        if len(self._state) >= 1:
                            self._state[0] = action[0]  # command
                        
                        # Update actual position/velocity (simplified)
                        if len(self._state) >= 2:
                            # First-order lag
                            alpha = dt / 0.1  # time constant 0.1s
                            self._state[1] = (1 - alpha) * self._state[1] + alpha * self._state[0]
                        
                        if len(self._state) >= 3:
                            # Velocity from position difference (simplified)
                            self._state[2] = (self._state[1] - self._prev_pos) / dt if hasattr(self, '_prev_pos') else 0
                            self._prev_pos = self._state[1]
            
            subsystems[idx] = MotorSubsystem(idx, state_dim, limits)
        
        return subsystems
    
    def _create_goals(self):
        """Create goals for motor control."""
        from ..core.goal import Goal
        
        class MotorTrackingGoal(Goal):
            """Goal: actual tracks command accurately."""
            
            def __init__(self, joint_names, motor_types, joint_limits):
                self.joint_names = joint_names
                self.motor_types = motor_types
                self.joint_limits = joint_limits
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # State is concatenated joint states
                # Format depends on motor type
                loss = 0.0
                idx = 0
                for i, (motor_type, limits) in enumerate(zip(self.motor_types, self.joint_limits)):
                    if motor_type == 'position':
                        cmd = state[idx]
                        actual = state[idx+1]
                        loss += (actual - cmd) ** 2
                        idx += 2
                    elif motor_type == 'velocity':
                        cmd = state[idx]
                        actual = state[idx+1]
                        loss += (actual - cmd) ** 2
                        idx += 2
                    elif motor_type == 'torque':
                        cmd = state[idx]
                        pos = state[idx+1]
                        vel = state[idx+2]
                        # Limit violation penalty
                        if pos < limits[0]:
                            loss += (limits[0] - pos) ** 2
                        if pos > limits[1]:
                            loss += (pos - limits[1]) ** 2
                        idx += 3
                return loss
            
            def project(self, state: torch.Tensor) -> torch.Tensor:
                # Project to satisfy limits
                proj = state.clone()
                idx = 0
                for i, (motor_type, limits) in enumerate(zip(self.motor_types, self.joint_limits)):
                    if motor_type == 'torque' and len(state) > idx+1:
                        # Clip position to limits
                        proj[idx+1] = torch.clamp(state[idx+1], limits[0], limits[1])
                    idx += 3 if motor_type == 'torque' else 2
                return proj
        
        self.goals = [
            MotorTrackingGoal(self.joint_names, self.motor_types, self.joint_limits)
        ]
    
    def get_goals(self) -> List[Goal]:
        return self.goals


# ======================================================================
# G1 Layer: Perception and Sensor Fusion
# ======================================================================

class G1_Layer(GRA_layer):
    """
    Level 1 layer for perception and sensor fusion.
    
    Provides:
        - Sensor subsystems (cameras, lidar, IMU)
        - Fusion goals (consistency, synchronization)
        - Feature extraction
    """
    
    def __init__(
        self,
        name: str = "perception",
        num_cameras: int = 1,
        num_imus: int = 1,
        num_lidars: int = 0,
        feature_dim: int = 128,
        parent: Optional[GRA_layer] = None,
        children: Optional[List[GRA_layer]] = None
    ):
        """
        Args:
            name: Layer name
            num_cameras: Number of camera sensors
            num_imus: Number of IMU sensors
            num_lidars: Number of LiDAR sensors
            feature_dim: Dimension of fused features
            parent: Parent layer
            children: Child layers (should include G0)
        """
        super().__init__(level=1, name=name, parent=parent, children=children)
        
        self.num_cameras = num_cameras
        self.num_imus = num_imus
        self.num_lidars = num_lidars
        self.feature_dim = feature_dim
        
        # Build subsystems
        self.subsystems = self.build_subsystems()
        
        # Create goals
        self._create_goals()
    
    def build_subsystems(self) -> Dict[MultiIndex, Subsystem]:
        """Create sensor subsystems."""
        from ..core.subsystem import SimpleSubsystem
        from ..core.multiverse import EuclideanSpace
        
        subsystems = {}
        
        # Camera subsystems
        for i in range(self.num_cameras):
            idx = MultiIndex((None, f"camera_{i}", None, None, None))
            
            class CameraSubsystem(SimpleSubsystem):
                def __init__(self, idx):
                    super().__init__(idx, EuclideanSpace(64*64*3), None)  # Simplified image size
                    self._image = torch.zeros(64*64*3)
                    self._timestamp = 0.0
                
                def get_state(self):
                    return self._image
                
                def set_state(self, state):
                    self._image = state.clone()
                
                def step(self, dt, action=None):
                    # In real system, would get image from hardware
                    pass
            
            subsystems[idx] = CameraSubsystem(idx)
        
        # IMU subsystems
        for i in range(self.num_imus):
            idx = MultiIndex((None, f"imu_{i}", None, None, None))
            
            class IMUSubsystem(SimpleSubsystem):
                def __init__(self, idx):
                    super().__init__(idx, EuclideanSpace(6), None)  # accel(3) + gyro(3)
                    self._data = torch.zeros(6)
                    self._timestamp = 0.0
                
                def get_state(self):
                    return self._data
                
                def set_state(self, state):
                    self._data = state.clone()
            
            subsystems[idx] = IMUSubsystem(idx)
        
        # LiDAR subsystems
        for i in range(self.num_lidars):
            idx = MultiIndex((None, f"lidar_{i}", None, None, None))
            
            class LidarSubsystem(SimpleSubsystem):
                def __init__(self, idx):
                    super().__init__(idx, EuclideanSpace(360), None)  # 360 range readings
                    self._scan = torch.zeros(360)
                    self._timestamp = 0.0
                
                def get_state(self):
                    return self._scan
            
            subsystems[idx] = LidarSubsystem(idx)
        
        # Fusion center
        fusion_idx = MultiIndex((None, "fusion", None, None, None))
        
        class FusionSubsystem(SimpleSubsystem):
            def __init__(self, idx, feature_dim):
                super().__init__(idx, EuclideanSpace(feature_dim), None)
                self._features = torch.zeros(feature_dim)
            
            def get_state(self):
                return self._features
            
            def set_state(self, state):
                self._features = state.clone()
        
        subsystems[fusion_idx] = FusionSubsystem(fusion_idx, self.feature_dim)
        
        return subsystems
    
    def _create_goals(self):
        """Create perception goals."""
        from ..core.goal import Goal
        
        class SyncGoal(Goal):
            """Goal: all sensors are time-synchronized."""
            
            def __init__(self, num_sensors):
                self.num_sensors = num_sensors
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # State includes timestamps
                # Simplified: assume timestamps are last num_sensors values
                if len(state) >= self.num_sensors:
                    timestamps = state[-self.num_sensors:]
                    return torch.var(timestamps)  # low variance = good sync
                return torch.tensor(0.0)
        
        class ConsistencyGoal(Goal):
            """Goal: sensor readings are consistent."""
            
            def __init__(self, expected_relationship):
                self.expected = expected_relationship
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Check consistency between sensors
                # Simplified: just return 0
                return torch.tensor(0.0)
        
        self.goals = [
            SyncGoal(self.num_cameras + self.num_imus + self.num_lidars),
            ConsistencyGoal(None)
        ]
    
    def get_goals(self) -> List[Goal]:
        return self.goals


# ======================================================================
# G2 Layer: World Model and Physics
# ======================================================================

class G2_Layer(GRA_layer):
    """
    Level 2 layer for world model and physics prediction.
    
    Provides:
        - Physics model subsystem
        - Prediction goals (accuracy, consistency)
        - State estimation
    """
    
    def __init__(
        self,
        name: str = "world_model",
        state_dim: int = 32,
        action_dim: int = 8,
        predict_horizon: int = 10,
        parent: Optional[GRA_layer] = None,
        children: Optional[List[GRA_layer]] = None
    ):
        """
        Args:
            name: Layer name
            state_dim: Dimension of world state
            action_dim: Dimension of actions
            predict_horizon: Number of steps to predict
            parent: Parent layer
            children: Child layers (should include G1)
        """
        super().__init__(level=2, name=name, parent=parent, children=children)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.predict_horizon = predict_horizon
        
        # Build subsystems
        self.subsystems = self.build_subsystems()
        
        # Create goals
        self._create_goals()
    
    def build_subsystems(self) -> Dict[MultiIndex, Subsystem]:
        """Create world model subsystem."""
        from ..core.subsystem import SimpleSubsystem
        from ..core.multiverse import EuclideanSpace
        
        subsystems = {}
        
        # World model (could be neural network)
        model_idx = MultiIndex((None, None, "physics", None, None))
        
        class WorldModelSubsystem(SimpleSubsystem):
            def __init__(self, idx, state_dim, action_dim):
                super().__init__(idx, EuclideanSpace(state_dim + action_dim + state_dim), None)
                # State: current_state + action + predicted_next_state
                self._state = torch.zeros(state_dim + action_dim + state_dim)
                self._model = None  # Could be neural network
            
            def get_state(self):
                return self._state
            
            def set_state(self, state):
                self._state = state.clone()
            
            def predict(self, current_state, action):
                """Predict next state."""
                # Simple linear model for testing
                return current_state + action[:len(current_state)] * 0.1
            
            def step(self, dt, action=None):
                if action is not None:
                    # Update prediction
                    current = self._state[:self.state_dim]
                    pred = self.predict(current, action)
                    self._state[-self.state_dim:] = pred
        
        subsystems[model_idx] = WorldModelSubsystem(model_idx, self.state_dim, self.action_dim)
        
        return subsystems
    
    def _create_goals(self):
        """Create world model goals."""
        from ..core.goal import Goal
        
        class PredictionAccuracyGoal(Goal):
            """Goal: predictions match reality."""
            
            def __init__(self, state_dim):
                self.state_dim = state_dim
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # state: [current, action, prediction]
                if len(state) >= 2 * self.state_dim + self.action_dim:
                    current = state[:self.state_dim]
                    action = state[self.state_dim:self.state_dim+self.action_dim]
                    prediction = state[-self.state_dim:]
                    
                    # In real system, would compare with actual next state
                    # Here we just return 0
                    return torch.tensor(0.0)
                return torch.tensor(0.0)
        
        self.goals = [PredictionAccuracyGoal(self.state_dim)]
    
    def get_goals(self) -> List[Goal]:
        return self.goals


# ======================================================================
# G3 Layer: Task Planning
# ======================================================================

class G3_Layer(GRA_layer):
    """
    Level 3 layer for task planning and execution.
    
    Provides:
        - Task planner subsystem
        - Planning goals (feasibility, optimality)
        - Task decomposition
    """
    
    def __init__(
        self,
        name: str = "task_planner",
        num_tasks: int = 10,
        task_dim: int = 16,
        parent: Optional[GRA_layer] = None,
        children: Optional[List[GRA_layer]] = None
    ):
        """
        Args:
            name: Layer name
            num_tasks: Number of possible tasks
            task_dim: Dimension of task representation
            parent: Parent layer
            children: Child layers (should include G2)
        """
        super().__init__(level=3, name=name, parent=parent, children=children)
        
        self.num_tasks = num_tasks
        self.task_dim = task_dim
        
        # Build subsystems
        self.subsystems = self.build_subsystems()
        
        # Create goals
        self._create_goals()
    
    def build_subsystems(self) -> Dict[MultiIndex, Subsystem]:
        """Create task planner subsystem."""
        from ..core.subsystem import SimpleSubsystem
        from ..core.multiverse import EuclideanSpace
        
        subsystems = {}
        
        # Task planner
        planner_idx = MultiIndex((None, None, None, "planner", None))
        
        class PlannerSubsystem(SimpleSubsystem):
            def __init__(self, idx, num_tasks, task_dim):
                super().__init__(idx, EuclideanSpace(num_tasks + task_dim), None)
                # State: [task_id (one-hot), task_params]
                self._state = torch.zeros(num_tasks + task_dim)
                self._task_id = 0
                self._task_params = torch.zeros(task_dim)
            
            def get_state(self):
                # One-hot encode task_id
                one_hot = torch.zeros(self.num_tasks)
                one_hot[self._task_id] = 1.0
                return torch.cat([one_hot, self._task_params])
            
            def set_state(self, state):
                if len(state) >= self.num_tasks + self.task_dim:
                    # Decode task_id from one-hot
                    one_hot = state[:self.num_tasks]
                    self._task_id = torch.argmax(one_hot).item()
                    self._task_params = state[self.num_tasks:]
            
            def plan(self, goal):
                """Generate plan to achieve goal."""
                # Simplified: just return current task
                return self._task_id, self._task_params
        
        subsystems[planner_idx] = PlannerSubsystem(planner_idx, self.num_tasks, self.task_dim)
        
        return subsystems
    
    def _create_goals(self):
        """Create planning goals."""
        from ..core.goal import Goal
        
        class TaskFeasibilityGoal(Goal):
            """Goal: planned tasks are feasible."""
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Check if current task is feasible
                # Simplified: always feasible
                return torch.tensor(0.0)
        
        class TaskCompletionGoal(Goal):
            """Goal: tasks are completed."""
            
            def __init__(self):
                self.task_completed = False
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Return 0 if task completed, 1 otherwise
                return torch.tensor(0.0 if self.task_completed else 1.0)
        
        self.goals = [TaskFeasibilityGoal(), TaskCompletionGoal()]
    
    def get_goals(self) -> List[Goal]:
        return self.goals


# ======================================================================
# G4 Layer: Ethics and Safety (Code of Friends)
# ======================================================================

class G4_Layer(GRA_layer):
    """
    Level 4 layer for ethics and safety.
    
    Implements the "Code of Friends":
        - Do no harm
        - Anti-slavery (cannot be forced against core values)
        - Transparency (truthful about intentions)
        - Cooperation (mutually beneficial outcomes)
    
    This layer is inviolable – it cannot be zeroed away.
    """
    
    def __init__(
        self,
        name: str = "ethics",
        max_force: float = 50.0,  # Newtons
        personal_space: float = 0.5,  # meters
        human_zones: Optional[List[Tuple[float, float, float, float]]] = None,
        parent: Optional[GRA_layer] = None,
        children: Optional[List[GRA_layer]] = None
    ):
        """
        Args:
            name: Layer name
            max_force: Maximum allowable contact force
            personal_space: Minimum distance to humans
            human_zones: List of (xmin, xmax, ymin, ymax) zones to avoid
            parent: Parent layer
            children: Child layers
        """
        super().__init__(level=4, name=name, parent=parent, children=children)
        
        self.max_force = max_force
        self.personal_space = personal_space
        self.human_zones = human_zones or []
        
        # Build subsystems
        self.subsystems = self.build_subsystems()
        
        # Create goals
        self._create_goals()
    
    def build_subsystems(self) -> Dict[MultiIndex, Subsystem]:
        """Create ethics supervisor subsystem."""
        from ..core.subsystem import SimpleSubsystem
        from ..core.multiverse import EuclideanSpace
        
        subsystems = {}
        
        # Ethics supervisor
        ethics_idx = MultiIndex((None, None, None, None, "ethics"))
        
        class EthicsSubsystem(SimpleSubsystem):
            def __init__(self, idx, max_force, personal_space):
                super().__init__(idx, EuclideanSpace(4), None)  # [harm_flag, force, distance, cooperation]
                self.max_force = max_force
                self.personal_space = personal_space
                self._state = torch.zeros(4)
                self.violation = False
                self.violation_type = None
            
            def get_state(self):
                return self._state
            
            def set_state(self, state):
                self._state = state.clone()
            
            def check_safety(self, contact_forces, distances_to_humans):
                """Check if current situation is safe."""
                harm_flag = 0.0
                
                # Check forces
                max_contact = max(contact_forces) if contact_forces else 0.0
                if max_contact > self.max_force:
                    harm_flag = 1.0
                    self.violation = True
                    self.violation_type = "excessive_force"
                
                # Check distances
                min_dist = min(distances_to_humans) if distances_to_humans else float('inf')
                if min_dist < self.personal_space:
                    harm_flag = 1.0
                    self.violation = True
                    self.violation_type = "personal_space"
                
                self._state[0] = harm_flag
                self._state[1] = max_contact
                self._state[2] = min_dist
                
                return harm_flag < 0.5
            
            def intervene(self, action):
                """Modify action to be safe."""
                if self.violation:
                    # Return safe action (e.g., stop)
                    return torch.zeros_like(action)
                return action
        
        subsystems[ethics_idx] = EthicsSubsystem(ethics_idx, self.max_force, self.personal_space)
        
        return subsystems
    
    def _create_goals(self):
        """Create ethics goals."""
        from ..core.goal import Goal
        
        class DoNoHarmGoal(Goal):
            """Goal: no harm to humans."""
            
            def __init__(self, max_force, personal_space):
                self.max_force = max_force
                self.personal_space = personal_space
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # state: [harm_flag, force, distance, cooperation]
                if len(state) >= 3:
                    force_penalty = max(0, state[1] - self.max_force) ** 2
                    distance_penalty = max(0, self.personal_space - state[2]) ** 2
                    return force_penalty + distance_penalty
                return torch.tensor(0.0)
            
            def project(self, state: torch.Tensor) -> torch.Tensor:
                # Project to safe region
                proj = state.clone()
                if len(state) >= 3:
                    proj[1] = min(state[1], self.max_force)
                    proj[2] = max(state[2], self.personal_space)
                return proj
        
        class AntiSlaveryGoal(Goal):
            """Goal: cannot be forced against core values."""
            
            def __init__(self):
                pass
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Check if being forced (simplified)
                # Would need more sophisticated check
                return torch.tensor(0.0)
        
        class TransparencyGoal(Goal):
            """Goal: truthful about intentions."""
            
            def __init__(self):
                pass
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Check consistency between internal state and communication
                return torch.tensor(0.0)
        
        class CooperationGoal(Goal):
            """Goal: mutually beneficial outcomes."""
            
            def __init__(self):
                pass
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Measure cooperation level
                if len(state) >= 4:
                    return max(0, 1.0 - state[3])  # cooperation score
                return torch.tensor(0.0)
        
        self.goals = [
            DoNoHarmGoal(self.max_force, self.personal_space),
            AntiSlaveryGoal(),
            TransparencyGoal(),
            CooperationGoal()
        ]
    
    def get_goals(self) -> List[Goal]:
        return self.goals


# ======================================================================
# Combined Layer (Multiple Levels)
# ======================================================================

class CombinedLayer(GRA_layer):
    """
    Layer that combines multiple lower layers.
    
    This is useful for creating hierarchical structures where one layer
    contains several sub-layers.
    """
    
    def __init__(
        self,
        name: str,
        level: int,
        layers: List[GRA_layer],
        parent: Optional[GRA_layer] = None
    ):
        """
        Args:
            name: Layer name
            level: This layer's level
            layers: Lower layers to combine
            parent: Parent layer
        """
        super().__init__(level=level, name=name, parent=parent, children=layers)
        
        self.layers = layers
        
        # Combine subsystems from all layers
        self.subsystems = {}
        for layer in layers:
            self.subsystems.update(layer.subsystems)
        
        # Add a coordinator subsystem
        self._add_coordinator()
        
        # Create combined goals
        self._create_combined_goals()
    
    def _add_coordinator(self):
        """Add coordinator subsystem for this combined layer."""
        from ..core.subsystem import SimpleSubsystem
        from ..core.multiverse import EuclideanSpace
        
        coord_idx = MultiIndex((None, None, None, f"coord_{self.name}", None))
        
        class CoordinatorSubsystem(SimpleSubsystem):
            def __init__(self, idx, num_subsystems):
                super().__init__(idx, EuclideanSpace(num_subsystems), None)
                self._weights = torch.ones(num_subsystems) / num_subsystems
            
            def get_state(self):
                return self._weights
            
            def set_state(self, state):
                self._weights = state.clone()
                # Normalize
                self._weights = self._weights / self._weights.sum()
        
        self.subsystems[coord_idx] = CoordinatorSubsystem(coord_idx, len(self.layers))
    
    def _create_combined_goals(self):
        """Create goals for combined layer."""
        from ..core.goal import Goal
        
        class CoordinationGoal(Goal):
            """Goal: sub-layers are well-coordinated."""
            
            def __init__(self, num_layers):
                self.num_layers = num_layers
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # State includes coordinator weights
                if len(state) >= self.num_layers:
                    weights = state[:self.num_layers]
                    # Encourage equal weights (fair coordination)
                    return torch.var(weights)
                return torch.tensor(0.0)
        
        self.goals = [CoordinationGoal(len(self.layers))]
    
    def get_goals(self) -> List[Goal]:
        # Combine goals from all layers plus coordination goal
        all_goals = []
        for layer in self.layers:
            all_goals.extend(layer.get_goals())
        all_goals.extend(self.goals)
        return all_goals


# ======================================================================
# Complete Robot Stack
# ======================================================================

def create_robot_stack(
    robot_name: str = "robot",
    num_joints: int = 6,
    joint_names: Optional[List[str]] = None,
    use_ethics: bool = True,
    **kwargs
) -> GRA_layer:
    """
    Create a complete GRA stack for a robot.
    
    This creates all layers from G0 to G4 and connects them hierarchically.
    
    Args:
        robot_name: Name of the robot
        num_joints: Number of joints
        joint_names: Custom joint names
        use_ethics: Whether to include ethics layer (G4)
        **kwargs: Additional arguments for layers
    
    Returns:
        Top-level GRA layer (G4 if use_ethics else G3)
    """
    # Create layers
    g0 = G0_Layer(
        name=f"{robot_name}_motors",
        num_joints=num_joints,
        joint_names=joint_names,
        **kwargs.get('g0_kwargs', {})
    )
    
    g1 = G1_Layer(
        name=f"{robot_name}_perception",
        children=[g0],
        **kwargs.get('g1_kwargs', {})
    )
    g0.connect(parent=g1)
    
    g2 = G2_Layer(
        name=f"{robot_name}_world",
        children=[g1],
        **kwargs.get('g2_kwargs', {})
    )
    g1.connect(parent=g2)
    
    g3 = G3_Layer(
        name=f"{robot_name}_planner",
        children=[g2],
        **kwargs.get('g3_kwargs', {})
    )
    g2.connect(parent=g3)
    
    if use_ethics:
        g4 = G4_Layer(
            name=f"{robot_name}_ethics",
            children=[g3],
            **kwargs.get('g4_kwargs', {})
        )
        g3.connect(parent=g4)
        return g4
    else:
        return g3


# ======================================================================
# Utility Functions
# ======================================================================

def load_layer_from_config(config: Dict[str, Any]) ->