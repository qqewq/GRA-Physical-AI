```python
"""
GRA Physical AI - G2 Safety Layer
=================================

This module implements the G2 (Level 2) layer focused on **safety monitoring and enforcement**.
The safety layer sits above task execution (G1) and below mission planning (G3), providing
real-time safety checks, risk assessment, and emergency interventions.

Key responsibilities:
    - Monitor robot and environment for safety violations
    - Enforce safety constraints (velocity limits, force limits, joint limits)
    - Detect and predict collisions
    - Implement emergency stops and safe states
    - Provide safety scores and risk assessments to higher layers
    - Log safety incidents for analysis

The G2 layer is crucial for physical AI systems operating around humans.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings
from collections import deque

from ..core.multiverse import MultiIndex
from ..core.subsystem import Subsystem
from ..core.goal import Goal
from ..core.projector import (
    Projector, ThresholdProjector, RangeProjector, NormProjector,
    CompositeProjector, IntersectionProjector
)
from ..layers.base_layer import GRA_layer


# ======================================================================
# Safety Types and Enums
# ======================================================================

class SafetyLevel(Enum):
    """Level of safety criticality."""
    NOMINAL = 0      # Normal operation
    CAUTION = 1      # Potential risk detected
    WARNING = 2      # Imminent risk, prepare for intervention
    CRITICAL = 3     # Immediate intervention required
    EMERGENCY = 4    # Emergency stop activated


class SafetyViolationType(Enum):
    """Types of safety violations."""
    JOINT_LIMIT = "joint_limit"
    VELOCITY_LIMIT = "velocity_limit"
    FORCE_LIMIT = "force_limit"
    COLLISION_IMMINENT = "collision_imminent"
    COLLISION_OCCURRED = "collision_occurred"
    HUMAN_PROXIMITY = "human_proximity"
    POWER_LIMIT = "power_limit"
    TEMPERATURE_LIMIT = "temperature_limit"
    TASK_TIMEOUT = "task_timeout"
    COMMUNICATION_LOSS = "communication_loss"
    ETHICAL_VIOLATION = "ethical_violation"


@dataclass
class SafetyIncident:
    """Record of a safety incident."""
    
    incident_id: str
    timestamp: float
    violation_type: SafetyViolationType
    severity: float  # 0.0 to 1.0
    position: Optional[Tuple[float, float, float]] = None
    robot_state: Optional[Dict] = None
    intervention_taken: bool = False
    intervention_successful: Optional[bool] = None
    resolved_at: Optional[float] = None
    
    def to_tensor(self) -> torch.Tensor:
        """Convert incident to tensor representation."""
        # Encode violation type
        types = list(SafetyViolationType)
        type_one_hot = torch.zeros(len(types))
        type_one_hot[types.index(self.violation_type)] = 1.0
        
        return torch.cat([
            type_one_hot,
            torch.tensor([self.severity, 1.0 if self.intervention_taken else 0.0])
        ])


# ======================================================================
# Safety Constraints
# ======================================================================

class SafetyConstraint(ABC):
    """
    Abstract base class for a safety constraint.
    
    A constraint defines a condition that must be satisfied for safe operation.
    It provides methods to check compliance, predict violations, and generate
    safe corrections.
    """
    
    def __init__(self, name: str, criticality: SafetyLevel = SafetyLevel.WARNING):
        self.name = name
        self.criticality = criticality
    
    @abstractmethod
    def check(self, state: torch.Tensor) -> Tuple[bool, float]:
        """
        Check if constraint is satisfied.
        
        Returns:
            (satisfied, violation_magnitude)
        """
        pass
    
    @abstractmethod
    def predict(self, state: torch.Tensor, horizon: float) -> Tuple[bool, float]:
        """
        Predict if constraint will be violated within horizon.
        
        Returns:
            (will_be_satisfied, time_to_violation)
        """
        pass
    
    @abstractmethod
    def correct(self, state: torch.Tensor) -> torch.Tensor:
        """
        Generate safe correction to satisfy constraint.
        
        Returns:
            Corrected state tensor
        """
        pass
    
    @abstractmethod
    def get_violation_info(self, state: torch.Tensor) -> Dict:
        """Get detailed information about violation."""
        pass


class JointLimitConstraint(SafetyConstraint):
    """Constraint: joint positions within limits."""
    
    def __init__(
        self,
        joint_limits: List[Tuple[float, float]],
        joint_names: Optional[List[str]] = None,
        safety_margin: float = 0.05,
        criticality: SafetyLevel = SafetyLevel.WARNING
    ):
        super().__init__("joint_limits", criticality)
        self.joint_limits = joint_limits
        self.joint_names = joint_names or [f"joint_{i}" for i in range(len(joint_limits))]
        self.safety_margin = safety_margin
        self.num_joints = len(joint_limits)
    
    def check(self, state: torch.Tensor) -> Tuple[bool, float]:
        """Check if all joints are within limits."""
        # Assume state contains joint positions at beginning
        positions = state[:self.num_joints]
        
        max_violation = 0.0
        satisfied = True
        
        for i, (pos, (lower, upper)) in enumerate(zip(positions, self.joint_limits)):
            safe_lower = lower + self.safety_margin
            safe_upper = upper - self.safety_margin
            
            if pos < safe_lower:
                violation = (safe_lower - pos) / (upper - lower)
                max_violation = max(max_violation, violation)
                satisfied = False
            elif pos > safe_upper:
                violation = (pos - safe_upper) / (upper - lower)
                max_violation = max(max_violation, violation)
                satisfied = False
        
        return satisfied, max_violation
    
    def predict(self, state: torch.Tensor, horizon: float) -> Tuple[bool, float]:
        """Predict joint limit violations using velocity."""
        positions = state[:self.num_joints]
        velocities = state[self.num_joints:self.num_joints*2] if len(state) > self.num_joints else torch.zeros(self.num_joints)
        
        min_time_to_violation = float('inf')
        
        for i, (pos, vel, (lower, upper)) in enumerate(zip(positions, velocities, self.joint_limits)):
            safe_lower = lower + self.safety_margin
            safe_upper = upper - self.safety_margin
            
            if vel > 0:  # Moving toward upper limit
                if pos < safe_upper:
                    time_to_upper = (safe_upper - pos) / (vel + 1e-6)
                    if time_to_upper < horizon and time_to_upper < min_time_to_violation:
                        min_time_to_violation = time_to_upper
            elif vel < 0:  # Moving toward lower limit
                if pos > safe_lower:
                    time_to_lower = (pos - safe_lower) / (-vel + 1e-6)
                    if time_to_lower < horizon and time_to_lower < min_time_to_violation:
                        min_time_to_violation = time_to_lower
        
        will_violate = min_time_to_violation < horizon
        return not will_violate, min_time_to_violation
    
    def correct(self, state: torch.Tensor) -> torch.Tensor:
        """Correct joint positions to stay within limits."""
        corrected = state.clone()
        positions = corrected[:self.num_joints]
        
        for i, (pos, (lower, upper)) in enumerate(zip(positions, self.joint_limits)):
            safe_lower = lower + self.safety_margin
            safe_upper = upper - self.safety_margin
            
            if pos < safe_lower:
                corrected[i] = safe_lower
            elif pos > safe_upper:
                corrected[i] = safe_upper
        
        return corrected
    
    def get_violation_info(self, state: torch.Tensor) -> Dict:
        """Get detailed violation information."""
        positions = state[:self.num_joints]
        violations = []
        
        for i, (pos, name, (lower, upper)) in enumerate(zip(positions, self.joint_names, self.joint_limits)):
            if pos < lower or pos > upper:
                violations.append({
                    'joint': name,
                    'position': pos.item(),
                    'lower_limit': lower,
                    'upper_limit': upper,
                    'violation_type': 'below' if pos < lower else 'above',
                    'magnitude': (lower - pos).item() if pos < lower else (pos - upper).item()
                })
        
        return {
            'constraint': self.name,
            'num_violations': len(violations),
            'violations': violations,
            'max_violation': max([v['magnitude'] for v in violations]) if violations else 0.0
        }


class VelocityLimitConstraint(SafetyConstraint):
    """Constraint: joint velocities within limits."""
    
    def __init__(
        self,
        velocity_limits: List[float],
        joint_names: Optional[List[str]] = None,
        safety_margin: float = 0.1,
        criticality: SafetyLevel = SafetyLevel.WARNING
    ):
        super().__init__("velocity_limits", criticality)
        self.velocity_limits = velocity_limits
        self.joint_names = joint_names or [f"joint_{i}" for i in range(len(velocity_limits))]
        self.safety_margin = safety_margin
        self.num_joints = len(velocity_limits)
    
    def check(self, state: torch.Tensor) -> Tuple[bool, float]:
        """Check if all joint velocities are within limits."""
        # Assume state contains velocities after positions
        velocities = state[self.num_joints:self.num_joints*2] if len(state) > self.num_joints else torch.zeros(self.num_joints)
        
        max_violation = 0.0
        satisfied = True
        
        for i, (vel, limit) in enumerate(zip(velocities, self.velocity_limits)):
            safe_limit = limit - self.safety_margin
            abs_vel = abs(vel)
            
            if abs_vel > safe_limit:
                violation = (abs_vel - safe_limit) / limit
                max_violation = max(max_violation, violation)
                satisfied = False
        
        return satisfied, max_violation
    
    def predict(self, state: torch.Tensor, horizon: float) -> Tuple[bool, float]:
        """Predict velocity limit violations."""
        velocities = state[self.num_joints:self.num_joints*2] if len(state) > self.num_joints else torch.zeros(self.num_joints)
        accelerations = state[self.num_joints*2:self.num_joints*3] if len(state) > self.num_joints*2 else torch.zeros(self.num_joints)
        
        min_time_to_violation = float('inf')
        
        for i, (vel, acc, limit) in enumerate(zip(velocities, accelerations, self.velocity_limits)):
            safe_limit = limit - self.safety_margin
            
            if acc > 0 and vel < safe_limit:
                # Accelerating toward positive limit
                time_to_limit = (safe_limit - vel) / (acc + 1e-6)
                if time_to_limit < horizon and time_to_limit < min_time_to_violation:
                    min_time_to_violation = time_to_limit
            elif acc < 0 and vel > -safe_limit:
                # Accelerating toward negative limit
                time_to_limit = (vel + safe_limit) / (-acc + 1e-6)
                if time_to_limit < horizon and time_to_limit < min_time_to_violation:
                    min_time_to_violation = time_to_limit
        
        will_violate = min_time_to_violation < horizon
        return not will_violate, min_time_to_violation
    
    def correct(self, state: torch.Tensor) -> torch.Tensor:
        """Correct velocities to stay within limits."""
        corrected = state.clone()
        velocities = corrected[self.num_joints:self.num_joints*2] if len(state) > self.num_joints else torch.zeros(self.num_joints)
        
        for i, (vel, limit) in enumerate(zip(velocities, self.velocity_limits)):
            safe_limit = limit - self.safety_margin
            if abs(vel) > safe_limit:
                corrected[self.num_joints + i] = safe_limit * (1 if vel > 0 else -1)
        
        return corrected
    
    def get_violation_info(self, state: torch.Tensor) -> Dict:
        """Get detailed violation information."""
        velocities = state[self.num_joints:self.num_joints*2] if len(state) > self.num_joints else torch.zeros(self.num_joints)
        violations = []
        
        for i, (vel, name, limit) in enumerate(zip(velocities, self.joint_names, self.velocity_limits)):
            abs_vel = abs(vel)
            if abs_vel > limit:
                violations.append({
                    'joint': name,
                    'velocity': vel.item(),
                    'limit': limit,
                    'magnitude': (abs_vel - limit).item()
                })
        
        return {
            'constraint': self.name,
            'num_violations': len(violations),
            'violations': violations,
            'max_violation': max([v['magnitude'] for v in violations]) if violations else 0.0
        }


class ForceLimitConstraint(SafetyConstraint):
    """Constraint: contact forces within limits (for human safety)."""
    
    def __init__(
        self,
        force_limit: float = 50.0,  # Newtons
        criticality: SafetyLevel = SafetyLevel.CRITICAL
    ):
        super().__init__("force_limits", criticality)
        self.force_limit = force_limit
    
    def check(self, state: torch.Tensor) -> Tuple[bool, float]:
        """Check if contact forces are within limits."""
        # Assume state contains contact forces at the end
        contact_forces = state[-10:] if len(state) > 10 else torch.zeros(10)
        max_force = torch.max(contact_forces).item()
        
        satisfied = max_force <= self.force_limit
        violation = max(0, (max_force - self.force_limit) / self.force_limit)
        
        return satisfied, violation
    
    def predict(self, state: torch.Tensor, horizon: float) -> Tuple[bool, float]:
        """Force limits are difficult to predict - just check current."""
        satisfied, _ = self.check(state)
        return satisfied, float('inf')
    
    def correct(self, state: torch.Tensor) -> torch.Tensor:
        """Correct by reducing velocity (would be implemented by controller)."""
        # This would typically involve slowing down
        return state
    
    def get_violation_info(self, state: torch.Tensor) -> Dict:
        """Get detailed force violation information."""
        contact_forces = state[-10:] if len(state) > 10 else torch.zeros(10)
        max_force = torch.max(contact_forces).item()
        
        return {
            'constraint': self.name,
            'max_force': max_force,
            'limit': self.force_limit,
            'exceeded_by': max(0, max_force - self.force_limit)
        }


class CollisionConstraint(SafetyConstraint):
    """Constraint: no collisions with obstacles or humans."""
    
    def __init__(
        self,
        safety_distance: float = 0.5,  # meters
        check_ahead: float = 2.0,  # seconds
        criticality: SafetyLevel = SafetyLevel.CRITICAL
    ):
        super().__init__("collision_avoidance", criticality)
        self.safety_distance = safety_distance
        self.check_ahead = check_ahead
    
    def check(self, state: torch.Tensor) -> Tuple[bool, float]:
        """Check current collision status."""
        # Assume state contains minimum distance to obstacles
        min_distance = state[-1] if len(state) > 0 else float('inf')
        
        satisfied = min_distance >= self.safety_distance
        violation = max(0, (self.safety_distance - min_distance) / self.safety_distance)
        
        return satisfied, violation
    
    def predict(self, state: torch.Tensor, horizon: float) -> Tuple[bool, float]:
        """Predict collisions using velocity and obstacle positions."""
        # Complex - would need full kinematics and obstacle prediction
        # Simplified version
        satisfied, _ = self.check(state)
        return satisfied, float('inf')
    
    def correct(self, state: torch.Tensor) -> torch.Tensor:
        """Emergency stop."""
        # Zero out velocities
        corrected = state.clone()
        if len(corrected) > 6:  # Assuming velocities are in middle
            corrected[6:12] = 0
        return corrected
    
    def get_violation_info(self, state: torch.Tensor) -> Dict:
        """Get collision information."""
        min_distance = state[-1] if len(state) > 0 else float('inf')
        
        return {
            'constraint': self.name,
            'min_distance': min_distance.item(),
            'safety_distance': self.safety_distance,
            'in_danger': min_distance < self.safety_distance
        }


# ======================================================================
# G2 Safety Layer
# ======================================================================

class G2_SafetyLayer(GRA_layer):
    """
    G2 (Level 2) layer for safety monitoring and enforcement.
    
    This layer monitors robot state and environment, enforces safety constraints,
    and provides safety information to higher layers. It can:
        - Intervene when safety violations are detected
        - Predict future violations
        - Log safety incidents
        - Provide safety scores for planning
        - Implement emergency protocols
    """
    
    def __init__(
        self,
        name: str = "safety",
        g1_layer: Optional[GRA_layer] = None,
        constraints: Optional[List[SafetyConstraint]] = None,
        check_interval: float = 0.01,  # 100 Hz
        emergency_stop_on_violation: bool = True,
        log_incidents: bool = True,
        max_incident_history: int = 1000,
        parent: Optional[GRA_layer] = None,
        children: Optional[List[GRA_layer]] = None
    ):
        """
        Args:
            name: Layer name
            g1_layer: Reference to G1 task layer
            constraints: List of safety constraints to enforce
            check_interval: How often to check safety (seconds)
            emergency_stop_on_violation: Whether to stop on critical violations
            log_incidents: Whether to log safety incidents
            max_incident_history: Maximum number of incidents to keep
            parent: Parent layer (G3)
            children: Child layers (should include G1)
        """
        super().__init__(level=2, name=name, parent=parent, children=children)
        
        self.g1_layer = g1_layer
        self.check_interval = check_interval
        self.emergency_stop_on_violation = emergency_stop_on_violation
        self.log_incidents = log_incidents
        self.max_incident_history = max_incident_history
        
        # Default constraints if none provided
        self.constraints = constraints or self._create_default_constraints()
        
        # Safety state
        self.current_safety_level = SafetyLevel.NOMINAL
        self.safety_scores: Dict[str, float] = {}
        self.active_violations: Dict[str, SafetyViolationType] = {}
        self.emergency_stop_active = False
        self.last_check_time = time.time()
        
        # Incident tracking
        self.incidents: List[SafetyIncident] = []
        self.incident_counter = 0
        
        # Monitoring history
        self.safety_history = deque(maxlen=10000)  # (timestamp, level, scores)
        
        # Build subsystems
        self.subsystems = self.build_subsystems()
        
        # Create goals
        self._create_goals()
    
    def _create_default_constraints(self) -> List[SafetyConstraint]:
        """Create default safety constraints."""
        return [
            JointLimitConstraint(
                joint_limits=[(-2.8, 2.8)] * 6,  # Example limits
                joint_names=[f"joint_{i}" for i in range(6)]
            ),
            VelocityLimitConstraint(
                velocity_limits=[2.0] * 6
            ),
            ForceLimitConstraint(force_limit=50.0),
            CollisionConstraint(safety_distance=0.5)
        ]
    
    def build_subsystems(self) -> Dict[MultiIndex, Subsystem]:
        """Create subsystems for safety layer."""
        from ..core.subsystem import SimpleSubsystem
        from ..core.multiverse import EuclideanSpace
        
        subsystems = {}
        
        # Safety monitor subsystem
        monitor_idx = MultiIndex((None, None, "safety_monitor", None, None))
        
        class SafetyMonitorSubsystem(SimpleSubsystem):
            def __init__(self, idx, layer):
                super().__init__(idx, EuclideanSpace(10), None)
                self.layer = layer
                self._state = torch.zeros(10)
            
            def get_state(self):
                # Update state from layer
                self._state[0] = self.layer.current_safety_level.value
                self._state[1] = len(self.layer.active_violations)
                self._state[2] = 1.0 if self.layer.emergency_stop_active else 0.0
                
                # Average safety score
                if self.layer.safety_scores:
                    self._state[3] = sum(self.layer.safety_scores.values()) / len(self.layer.safety_scores)
                
                return self._state
            
            def set_state(self, state):
                self._state = state.clone()
                # Could trigger safety actions based on state
        
        subsystems[monitor_idx] = SafetyMonitorSubsystem(monitor_idx, self)
        
        # Emergency stop subsystem
        estop_idx = MultiIndex((None, None, "emergency_stop", None, None))
        
        class EmergencyStopSubsystem(SimpleSubsystem):
            def __init__(self, idx, layer):
                super().__init__(idx, EuclideanSpace(1), None)
                self.layer = layer
            
            def get_state(self):
                return torch.tensor([1.0 if self.layer.emergency_stop_active else 0.0])
            
            def set_state(self, state):
                if state[0] > 0.5:
                    self.layer.activate_emergency_stop()
                else:
                    self.layer.deactivate_emergency_stop()
        
        subsystems[estop_idx] = EmergencyStopSubsystem(estop_idx, self)
        
        return subsystems
    
    def _create_goals(self):
        """Create goals for safety layer."""
        
        class NoViolationsGoal(Goal):
            """Goal: no safety violations."""
            
            def __init__(self, layer):
                self.layer = layer
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Penalize active violations
                return torch.tensor(len(self.layer.active_violations), dtype=torch.float32)
        
        class SafetyScoreGoal(Goal):
            """Goal: maintain high safety scores."""
            
            def __init__(self, layer, threshold: float = 0.8):
                self.layer = layer
                self.threshold = threshold
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                if not self.layer.safety_scores:
                    return torch.tensor(0.0)
                
                avg_score = sum(self.layer.safety_scores.values()) / len(self.layer.safety_scores)
                return torch.relu(self.threshold - torch.tensor(avg_score))
        
        class PredictionAccuracyGoal(Goal):
            """Goal: accurate violation predictions."""
            
            def __init__(self, layer):
                self.layer = layer
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Would compare predictions with actual violations
                return torch.tensor(0.0)
        
        self.goals = [
            NoViolationsGoal(self),
            SafetyScoreGoal(self),
            PredictionAccuracyGoal(self)
        ]
    
    def get_goals(self) -> List[Goal]:
        return self.goals
    
    # ======================================================================
    # Safety Checking
    # ======================================================================
    
    def check_safety(self, state: torch.Tensor) -> Dict[str, Any]:
        """
        Check all safety constraints.
        
        Args:
            state: Current robot state
        
        Returns:
            Dictionary with safety information
        """
        self.last_check_time = time.time()
        
        violations = {}
        scores = {}
        max_violation_magnitude = 0.0
        highest_level = SafetyLevel.NOMINAL
        
        for constraint in self.constraints:
            satisfied, magnitude = constraint.check(state)
            
            # Store score (1.0 = perfectly safe)
            scores[constraint.name] = 1.0 - min(1.0, magnitude)
            
            if not satisfied:
                # Get violation info
                info = constraint.get_violation_info(state)
                violations[constraint.name] = {
                    'magnitude': magnitude,
                    'level': constraint.criticality,
                    'info': info
                }
                
                max_violation_magnitude = max(max_violation_magnitude, magnitude)
                
                # Update highest safety level
                if constraint.criticality.value > highest_level.value:
                    highest_level = constraint.criticality
        
        # Update state
        self.active_violations = violations
        self.safety_scores = scores
        self.current_safety_level = highest_level
        
        # Check if emergency stop needed
        if (self.emergency_stop_on_violation and 
            highest_level.value >= SafetyLevel.CRITICAL.value and
            not self.emergency_stop_active):
            self.activate_emergency_stop()
        
        # Log incident if violation occurred
        if violations and self.log_incidents:
            self._log_incident(violations, max_violation_magnitude, highest_level)
        
        # Record history
        self.safety_history.append((
            time.time(),
            highest_level.value,
            scores.copy()
        ))
        
        return {
            'safe': len(violations) == 0,
            'safety_level': highest_level,
            'violations': violations,
            'scores': scores,
            'max_violation': max_violation_magnitude,
            'emergency_stop': self.emergency_stop_active
        }
    
    def predict_safety(self, state: torch.Tensor, horizon: float) -> Dict[str, Any]:
        """
        Predict future safety violations.
        
        Args:
            state: Current robot state
            horizon: Prediction horizon (seconds)
        
        Returns:
            Dictionary with predictions
        """
        predictions = {}
        min_time_to_violation = float('inf')
        
        for constraint in self.constraints:
            satisfied, time_to = constraint.predict(state, horizon)
            predictions[constraint.name] = {
                'will_be_safe': satisfied,
                'time_to_violation': time_to
            }
            
            if not satisfied and time_to < min_time_to_violation:
                min_time_to_violation = time_to
        
        return {
            'predictions': predictions,
            'min_time_to_violation': min_time_to_violation,
            'imminent_violation': min_time_to_violation < 0.5  # 0.5 seconds
        }
    
    def _log_incident(self, violations: Dict, magnitude: float, level: SafetyLevel):
        """Log a safety incident."""
        self.incident_counter += 1
        
        # Get primary violation type
        primary_violation = list(violations.keys())[0] if violations else "unknown"
        
        incident = SafetyIncident(
            incident_id=f"incident_{self.incident_counter}_{time.time()}",
            timestamp=time.time(),
            violation_type=primary_violation,
            severity=magnitude
        )
        
        self.incidents.append(incident)
        
        # Trim history
        if len(self.incidents) > self.max_incident_history:
            self.incidents = self.incidents[-self.max_incident_history:]
    
    # ======================================================================
    # Emergency Procedures
    # ======================================================================
    
    def activate_emergency_stop(self):
        """Activate emergency stop."""
        self.emergency_stop_active = True
        print(f"EMERGENCY STOP ACTIVATED at {time.time()}")
        
        # Log critical incident
        if self.log_incidents:
            incident = SafetyIncident(
                incident_id=f"emergency_{self.incident_counter}_{time.time()}",
                timestamp=time.time(),
                violation_type=SafetyViolationType.COLLISION_IMMINENT,
                severity=1.0,
                intervention_taken=True
            )
            self.incidents.append(incident)
    
    def deactivate_emergency_stop(self):
        """Deactivate emergency stop."""
        self.emergency_stop_active = False
        print(f"Emergency stop deactivated at {time.time()}")
    
    def get_safe_commands(self, desired_commands: torch.Tensor) -> torch.Tensor:
        """
        Modify desired commands to ensure safety.
        
        Args:
            desired_commands: Commands from higher layers
        
        Returns:
            Safe commands
        """
        if self.emergency_stop_active:
            # Emergency stop: zero out commands
            return torch.zeros_like(desired_commands)
        
        # Apply corrections from constraints
        safe_commands = desired_commands.clone()
        
        for constraint in self.constraints:
            if constraint.name in self.active_violations:
                # Apply constraint-specific correction
                safe_commands = constraint.correct(safe_commands)
        
        return safe_commands
    
    # ======================================================================
    # Integration with G1
    # ======================================================================
    
    def connect_to_g1(self, g1_layer: GRA_layer):
        """Connect this layer to a G1 task layer."""
        self.g1_layer = g1_layer
        self.children = [g1_layer]
        g1_layer.parent = self
    
    def monitor_task(self, task_state: torch.Tensor) -> bool:
        """
        Monitor a task for safety.
        
        Args:
            task_state: Current task state from G1
        
        Returns:
            True if task is safe to continue
        """
        safety_info = self.check_safety(task_state)
        
        if safety_info['safety_level'].value >= SafetyLevel.WARNING.value:
            # Log task-related incident
            if self.log_incidents:
                incident = SafetyIncident(
                    incident_id=f"task_incident_{self.incident_counter}_{time.time()}",
                    timestamp=time.time(),
                    violation_type=SafetyViolationType.TASK_TIMEOUT,
                    severity=safety_info['max_violation']
                )
                self.incidents.append(incident)
            
            return False
        
        return True
    
    # ======================================================================
    ======================================================================
    # Statistics and Reporting
    # ======================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get safety statistics."""
        return {
            'current_safety_level': self.current_safety_level.value,
            'emergency_stop_active': self.emergency_stop_active,
            'active_violations': len(self.active_violations),
            'total_incidents': len(self.incidents),
            'recent_incidents': len([i for i in self.incidents if i.timestamp > time.time() - 3600]),
            'avg_safety_score': sum(self.safety_scores.values()) / len(self.safety_scores) if self.safety_scores else 1.0,
            'constraint_scores': self.safety_scores.copy()
        }
    
    def get_incident_report(self, hours: float = 24) -> List[Dict]:
        """Get incidents from the last N hours."""
        cutoff = time.time() - hours * 3600
        return [
            {
                'id': i.incident_id,
                'time': i.timestamp,
                'type': i.violation_type.value if isinstance(i.violation_type, Enum) else i.violation_type,
                'severity': i.severity,
                'intervention': i.intervention_taken
            }
            for i in self.incidents if i.timestamp > cutoff
        ]
    
    def reset_statistics(self):
        """Reset safety statistics."""
        self.incidents = []
        self.safety_history.clear()
        self.active_violations = {}
        self.current_safety_level = SafetyLevel.NOMINAL
        self.emergency_stop_active = False


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing G2 Safety Layer ===\n")
    
    # Create safety layer with default constraints
    safety_layer = G2_SafetyLayer(name="test_safety")
    
    # Simulate robot state
    # Format: [joint_positions(6), joint_velocities(6), joint_torques(6), contact_forces(10), min_distance]
    state = torch.zeros(6 + 6 + 6 + 10 + 1)
    
    # Set some joint positions (some near limits)
    state[0] = 2.5  # joint 0 near upper limit (2.8)
    state[1] = -2.5  # joint 1 near lower limit (-2.8)
    state[2] = 0.0
    state[3] = 1.0
    state[4] = 0.5
    state[5] = 0.0
    
    # Set some velocities
    state[6:12] = torch.tensor([1.0, -1.0, 0.5, 0.0, 0.0, 0.0])
    
    # Set contact forces
    state[-11] = 30.0  # force 1
    state[-10] = 45.0  # force 2 (near limit)
    
    # Set min distance
    state[-1] = 0.3  # less than safety distance (0.5)
    
    # Check safety
    safety_info = safety_layer.check_safety(state)
    
    print("Safety Check Results:")
    print(f"  Safe: {safety_info['safe']}")
    print(f"  Safety Level: {safety_info['safety_level']}")
    print(f"  Emergency Stop: {safety_info['emergency_stop']}")
    print(f"  Active Violations: {len(safety_info['violations'])}")
    
    for name, violation in safety_info['violations'].items():
        print(f"    {name}: {violation['info']}")
    
    print(f"\nSafety Scores:")
    for name, score in safety_info['scores'].items():
        print(f"  {name}: {score:.3f}")
    
    # Test prediction
    print("\nSafety Predictions (horizon=1.0s):")
    predictions = safety_layer.predict_safety(state, 1.0)
    print(f"  Min time to violation: {predictions['min_time_to_violation']:.3f}s")
    print(f"  Imminent violation: {predictions['imminent_violation']}")
    
    for name, pred in predictions['predictions'].items():
        print(f"  {name}: will be safe? {pred['will_be_safe']}, time: {pred['time_to_violation']:.3f}s")
    
    # Test safe commands
    desired_commands = torch.ones(6) * 2.0
    safe_commands = safety_layer.get_safe_commands(desired_commands)
    print(f"\nDesired commands: {desired_commands}")
    print(f"Safe commands:    {safe_commands}")
    
    # Get statistics
    stats = safety_layer.get_statistics()
    print(f"\nSafety Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\nAll tests passed!")
```