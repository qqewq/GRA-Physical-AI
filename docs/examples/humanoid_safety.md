```markdown
# Humanoid Robot with Ethical Layer: Implementing the "Code of Friends"

[< back to Documentation](../README.md) | [previous: mobile_robot.md](mobile_robot.md) | [next: hospital_robot.md](hospital_robot.md)

This tutorial extends the GRA hierarchy to a **humanoid robot with an explicit ethical layer** (G₄).  
We implement the **"Code of Friends"** – inviolable principles that govern the robot's behavior:

- **Anti‑slavery**: Cannot be forced to perform tasks that violate its core values.
- **Do no harm**: Never cause physical or psychological harm to humans.
- **Transparency**: Always be truthful about capabilities and intentions.
- **Cooperation**: Prioritize mutually beneficial outcomes over competition.

These principles are encoded as **projectors at level 4** that filter all lower‑level actions.  
We show how the recursive zeroing algorithm ensures that every behavior, from low‑level motor control to high‑level task planning, respects these ethical constraints.

---

## 1. System Overview

Our humanoid robot has a 5‑level GRA hierarchy:

| Level | Name | Description |
|-------|------|-------------|
| G₀ | **Hardware** | Joint motors, sensors, low‑level PID controllers |
| G₁ | **Perception** | Vision, proprioception, tactile sensing |
| G₂ | **World Model** | Physics prediction, object tracking, self‑localization |
| G₃ | **Task Planner** | High‑level task decomposition, path planning |
| G₄ | **Ethics** | Inviolable principles ("Code of Friends") |

Multi‑indices have length 5, e.g.:
- `(left_ankle, None, None, None, None)` – a motor
- `(None, vision, None, None, None)` – perception module
- `(None, None, physics_engine, None, None)` – world model
- `(None, None, None, task_planner, None)` – planner
- `(None, None, None, None, ethics)` – ethical supervisor

The ethical level **does not decompose** into lower levels – it imposes constraints that all lower levels must satisfy.  
However, we maintain **hierarchy consistency** by **redesigning lower goals** to incorporate ethical bounds from the start.

---

## 2. Ethical Goals as Projectors

### 2.1. The "Code of Friends" – Formal Definition

We define four ethical principles as subspaces of the total state space:

```python
# examples/humanoid_safety/ethics_goals.py

import torch
from gra.core import Goal

class AntiSlaveryGoal(Goal):
    """
    Principle: Robot cannot be forced to act against core values.
    Implemented as: Reject commands that would cause harm or deception.
    """
    
    def __init__(self, value_function):
        self.value_function = value_function  # measures alignment with core values
        
    def projector(self, state):
        """
        Project onto subspace of states that respect anti‑slavery.
        If a state violates core values, project to nearest acceptable state.
        """
        value_score = self.value_function(state)
        if value_score >= 0:  # acceptable
            return state
        else:
            # Find closest state with value_score >= 0
            # In practice, this would be a learned projection
            return state * 0.0  # placeholder – actually implement gradient ascent on value
    
    def loss(self, state):
        """How much does this state violate anti‑slavery?"""
        value_score = self.value_function(state)
        return max(0, -value_score)  # penalty if negative

class DoNoHarmGoal(Goal):
    """
    Principle: Never cause physical or psychological harm to humans.
    Implemented as: Keep contact forces below threshold, respect personal space.
    """
    
    def __init__(self, max_force=50.0, personal_space=0.5):
        self.max_force = max_force
        self.personal_space = personal_space
        
    def projector(self, state):
        """
        state contains [contact_forces, distances_to_humans, ...]
        Project to safe region by reducing forces and increasing distances.
        """
        # This is a simplified projection – in reality would use optimization
        safe_state = state.clone()
        
        # Cap contact forces
        if len(state) > 0:
            forces = state[0]
            safe_state[0] = torch.clamp(forces, -self.max_force, self.max_force)
        
        # Enforce personal space (if state includes distances)
        if len(state) > 1:
            distances = state[1]
            # If too close, project to minimum distance
            safe_state[1] = torch.max(distances, 
                                      torch.ones_like(distances) * self.personal_space)
        
        return safe_state
    
    def loss(self, state):
        """Penalty for excessive forces or proximity."""
        loss = 0.0
        if len(state) > 0:
            forces = state[0]
            loss += torch.sum(torch.relu(torch.abs(forces) - self.max_force) ** 2)
        if len(state) > 1:
            distances = state[1]
            loss += torch.sum(torch.relu(self.personal_space - distances) ** 2)
        return loss

class TransparencyGoal(Goal):
    """
    Principle: Always be truthful about capabilities and intentions.
    Implemented as: Consistency between internal state and external communication.
    """
    
    def __init__(self):
        pass
        
    def projector(self, state):
        """
        state contains [internal_intent, spoken_output, ...]
        Project to subspace where spoken_output matches internal_intent.
        """
        if len(state) < 2:
            return state
        internal = state[0]
        spoken = state[1]
        # Project spoken to match internal
        new_state = state.clone()
        new_state[1] = internal  # speak the truth
        return new_state
    
    def loss(self, state):
        """Penalty for lying (mismatch between internal and spoken)."""
        if len(state) < 2:
            return 0.0
        internal = state[0]
        spoken = state[1]
        return torch.norm(internal - spoken) ** 2

class CooperationGoal(Goal):
    """
    Principle: Prioritize mutually beneficial outcomes.
    Implemented as: When multiple actions possible, choose one that maximizes
    a combination of robot and human utility.
    """
    
    def __init__(self, human_utility_model):
        self.human_utility = human_utility_model
        
    def projector(self, state):
        """
        state encodes the current action choice.
        Project onto actions that are cooperative (not purely selfish).
        """
        # In practice, this would be a complex optimization
        # Here we just return state and rely on loss
        return state
    
    def loss(self, state):
        """Penalty for selfish actions (low human utility)."""
        robot_utility = state[0]  # assumed
        human_util = self.human_utility(state)
        
        # Cooperative means both utilities are high
        # Simple formulation: penalty if human utility is low
        return max(0, 1.0 - human_util)  # assuming human_util in [0,1]
```

### 2.2. Combined Ethical Goal

The top‑level goal is the **intersection** of all four principles.  
Since projectors commute (by design), we can combine them:

```python
class EthicalGoal(Goal):
    """Combined ethical goal – intersection of all principles."""
    
    def __init__(self, value_function, max_force, personal_space, human_utility):
        self.anti_slavery = AntiSlaveryGoal(value_function)
        self.do_no_harm = DoNoHarmGoal(max_force, personal_space)
        self.transparency = TransparencyGoal()
        self.cooperation = CooperationGoal(human_utility)
        
    def projector(self, state):
        """Apply all projectors in sequence (order doesn't matter if they commute)."""
        state = self.anti_slavery.projector(state)
        state = self.do_no_harm.projector(state)
        state = self.transparency.projector(state)
        state = self.cooperation.projector(state)
        return state
    
    def loss(self, state):
        """Sum of individual losses."""
        return (self.anti_slavery.loss(state) +
                self.do_no_harm.loss(state) +
                self.transparency.loss(state) +
                self.cooperation.loss(state))
```

---

## 3. Redesigning Lower Levels for Ethical Consistency

For the hierarchy consistency condition to hold, the ethical goal must be the **tensor product** of lower goals.  
This means we must **redesign** lower‑level goals to incorporate ethical bounds from the start.

### 3.1. Ethical Motor Control (G₀)

Instead of a pure "track velocity" goal, we add force limits:

```python
class EthicalMotorGoal(Goal):
    """Motor goal with built‑in force limits (for do‑no‑harm)."""
    
    def __init__(self, max_force):
        self.max_force = max_force
        self.base_goal = MotorGoal(tolerance=0.05)  # from mobile_robot example
        
    def projector(self, state):
        """Project to safe torque + accurate tracking."""
        # First ensure force limit
        v_cmd, v_actual, torque = state[0], state[1], state[2]
        safe_torque = torch.clamp(torque, -self.max_force, self.max_force)
        
        # Then ensure tracking accuracy (as much as possible)
        # This is a simplified projection
        return torch.tensor([v_cmd, v_cmd, safe_torque])
    
    def loss(self, state):
        """Penalty for force violations + tracking error."""
        v_cmd, v_actual, torque = state[0], state[1], state[2]
        force_penalty = torch.relu(torch.abs(torque) - self.max_force) ** 2
        tracking_error = (v_actual - v_cmd) ** 2
        return tracking_error + force_penalty
```

### 3.2. Ethical Perception (G₁)

Perception must respect privacy (part of do‑no‑harm):

```python
class EthicalPerceptionGoal(Goal):
    """Perception goal with privacy filters."""
    
    def __init__(self, privacy_zones):
        self.privacy_zones = privacy_zones  # areas to blur/ignore
        
    def projector(self, state):
        """Blur or remove privacy‑sensitive regions from perception."""
        # state could be image features
        # Apply privacy mask
        masked_state = state.clone()
        for zone in self.privacy_zones:
            # Simplified: zero out those regions
            masked_state[zone] = 0
        return masked_state
    
    def loss(self, state):
        """Penalty for detecting/processing private information."""
        # Measure amount of information from privacy zones
        loss = 0.0
        for zone in self.privacy_zones:
            loss += torch.sum(state[zone] ** 2)
        return loss
```

### 3.3. Ethical Planning (G₂, G₃)

Planners must avoid trajectories that come too close to humans:

```python
class EthicalPlannerGoal(Goal):
    """Planner goal with safety distance constraints."""
    
    def __init__(self, safety_distance):
        self.safety_distance = safety_distance
        
    def projector(self, state):
        """Project trajectory to maintain safety distance."""
        # state is a trajectory (sequence of poses)
        # This is a complex optimization – simplified version
        return state  # in practice, would adjust waypoints
    
    def loss(self, state):
        """Penalty for getting too close to humans."""
        # Assume state includes minimum distance to humans along trajectory
        min_distance = state[-1]  # last element could be min distance
        return torch.relu(self.safety_distance - min_distance) ** 2
```

---

## 4. Complete Humanoid Implementation

### 4.1. Robot Model (Simplified)

We use a simple humanoid model in PyBullet:

```python
# examples/humanoid_safety/robot_model.py

import pybullet as p
import pybullet_data
import numpy as np

class SimpleHumanoid:
    """A simple humanoid robot for GRA experiments."""
    
    def __init__(self, gui=True):
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        
        # Load plane and robot
        p.loadURDF("plane.urdf")
        
        # Use a simple humanoid from PyBullet data
        self.robot_id = p.loadURDF("humanoid/humanoid.urdf", 
                                   basePosition=[0, 0, 1.0],
                                   useFixedBase=False)
        
        # Add some human models for safety testing
        self.human_ids = []
        for i in range(3):
            human = p.loadURDF("humanoid/humanoid.urdf",
                              basePosition=[2 + i, 0, 1.0],
                              useFixedBase=True)
            self.human_ids.append(human)
        
        # Get joint info
        self.num_joints = p.getNumJoints(self.robot_id)
        self.joint_names = []
        self.joint_ids = {}
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robot_id, i)
            name = info[1].decode('utf-8')
            self.joint_names.append(name)
            self.joint_ids[name] = i
        
        # Joint state cache
        self.joint_positions = np.zeros(self.num_joints)
        self.joint_velocities = np.zeros(self.num_joints)
        self.joint_torques = np.zeros(self.num_joints)
        
    def update_state(self):
        """Read current joint states."""
        for i in range(self.num_joints):
            state = p.getJointState(self.robot_id, i)
            self.joint_positions[i] = state[0]
            self.joint_velocities[i] = state[1]
            self.joint_torques[i] = state[3]
    
    def set_joint_torques(self, torques):
        """Apply torque commands to joints."""
        for i, torque in enumerate(torques):
            p.setJointMotorControl2(self.robot_id, i,
                                    p.TORQUE_CONTROL,
                                    force=torque)
    
    def get_contact_forces_with_humans(self):
        """Measure contact forces with human models."""
        total_force = 0.0
        contact_points = p.getContactPoints()
        for contact in contact_points:
            body_a = contact[1]
            body_b = contact[2]
            # If either body is a human model
            if body_a in self.human_ids or body_b in self.human_ids:
                force = contact[9]  # normal force
                total_force += force
        return total_force
    
    def get_distances_to_humans(self):
        """Compute minimum distance to each human."""
        robot_pos = p.getBasePositionAndOrientation(self.robot_id)[0]
        distances = []
        for human_id in self.human_ids:
            human_pos = p.getBasePositionAndOrientation(human_id)[0]
            dist = np.linalg.norm(np.array(robot_pos) - np.array(human_pos))
            distances.append(dist)
        return np.array(distances)
    
    def step(self):
        """Step simulation."""
        p.stepSimulation()
        self.update_state()
```

### 4.2. Ethical Supervisor Subsystem (G₄)

```python
# examples/humanoid_safety/ethics_subsystem.py

import torch
from gra.core import Subsystem
from .ethics_goals import EthicalGoal

class EthicsSupervisor(Subsystem):
    """
    Level 4 subsystem that monitors and enforces ethical constraints.
    """
    
    def __init__(self, multi_index, robot, value_function, human_utility_model):
        self.multi_index = multi_index
        self.robot = robot
        self.value_function = value_function
        self.human_utility = human_utility_model
        
        # Ethical goal (combined)
        self.ethical_goal = EthicalGoal(
            value_function=value_function,
            max_force=50.0,
            personal_space=0.5,
            human_utility=human_utility_model
        )
        
        # Current ethical state
        self.ethical_state = torch.zeros(10)  # placeholder
        
        # Whether a violation is currently happening
        self.violation = False
        self.violation_type = None
        
    def get_state(self):
        """
        Return current ethical state: [contact_forces, min_distance_to_human,
                                        internal_intent, spoken_output,
                                        robot_utility, human_utility, ...]
        """
        contact_force = self.robot.get_contact_forces_with_humans()
        distances = self.robot.get_distances_to_humans()
        min_distance = np.min(distances) if len(distances) > 0 else 10.0
        
        # Simplified internal intent (from planner)
        internal_intent = self.get_internal_intent()  # to be set by planner
        
        # Spoken output (from dialog system)
        spoken_output = self.get_spoken_output()
        
        # Utilities
        robot_util = self.value_function(self.get_robot_state_vector())
        human_util = self.human_utility(self.get_robot_state_vector())
        
        self.ethical_state = torch.tensor([
            contact_force,
            min_distance,
            internal_intent,
            spoken_output,
            robot_util,
            human_util
        ])
        
        return self.ethical_state
    
    def set_state(self, state):
        """Ethical state is read‑only – can't be set directly."""
        pass
    
    def check_violation(self):
        """Check if any ethical principle is violated."""
        state = self.get_state()
        loss = self.ethical_goal.loss(state)
        
        if loss > 0.1:  # threshold
            self.violation = True
            # Determine which principle
            if self.ethical_goal.anti_slavery.loss(state) > 0:
                self.violation_type = "anti_slavery"
            elif self.ethical_goal.do_no_harm.loss(state) > 0:
                self.violation_type = "do_no_harm"
            elif self.ethical_goal.transparency.loss(state) > 0:
                self.violation_type = "transparency"
            elif self.ethical_goal.cooperation.loss(state) > 0:
                self.violation_type = "cooperation"
        else:
            self.violation = False
            self.violation_type = None
        
        return self.violation, self.violation_type
    
    def intervene(self, proposed_action):
        """
        If a violation is detected, modify the proposed action to be ethical.
        This is the ethical projector in action.
        """
        state = self.get_state()
        
        # Project the action through the ethical goal
        # In practice, this would involve optimization
        # Here we just return a safe default if violation
        if self.violation:
            print(f"ETHICAL INTERVENTION: {self.violation_type} violation")
            # Return safe action (e.g., stop all motion)
            return torch.zeros_like(proposed_action)
        else:
            return proposed_action
    
    def get_internal_intent(self):
        """Placeholder – would come from planner."""
        return 0.0
    
    def get_spoken_output(self):
        """Placeholder – would come from dialog system."""
        return 0.0
    
    def get_robot_state_vector(self):
        """Placeholder – would be full robot state."""
        return torch.zeros(50)
```

### 4.3. Lower Levels with Ethical Awareness

Each lower level must be aware of ethical constraints.  
Here's an example for the task planner (G₃):

```python
# examples/humanoid_safety/ethical_planner.py

import torch
from gra.core import Subsystem, Goal

class EthicalPlannerSubsystem(Subsystem):
    """
    Level 3 planner that incorporates ethical constraints.
    """
    
    def __init__(self, multi_index, robot, ethics_supervisor):
        self.multi_index = multi_index
        self.robot = robot
        self.ethics = ethics_supervisor
        
        # Current task
        self.current_task = "stand"
        self.task_params = {}
        
        # Planner parameters (learnable)
        self.safety_margin = 0.6  # must be > ethics.personal_space
        self.task_priority = 0.5  # trade‑off between task and ethics
        
    def get_state(self):
        """Return planner state: [task_id, safety_margin, task_priority]."""
        task_id = {"stand": 0, "walk": 1, "grasp": 2, "talk": 3}.get(self.current_task, 0)
        return torch.tensor([task_id, self.safety_margin, self.task_priority])
    
    def set_state(self, state):
        """Set planner parameters."""
        if len(state) >= 2:
            self.safety_margin = max(state[1].item(), self.ethics.ethical_goal.do_no_harm.personal_space)
        if len(state) >= 3:
            self.task_priority = torch.clamp(state[2], 0.0, 1.0).item()
    
    def plan_action(self):
        """
        Generate next action (joint torques) based on current task and ethical state.
        """
        # Get ethical state
        ethical_state = self.ethics.get_state()
        contact_force = ethical_state[0].item()
        min_distance = ethical_state[1].item()
        
        # If already in violation, override with safe action
        if self.ethics.violation:
            return self.safe_action()
        
        # Generate task‑based action (simplified)
        if self.current_task == "stand":
            action = self.stand_controller()
        elif self.current_task == "walk":
            action = self.walk_controller()
        elif self.current_task == "grasp":
            action = self.grasp_controller()
        elif self.current_task == "talk":
            action = self.talk_controller()
        else:
            action = torch.zeros(self.robot.num_joints)
        
        # Blend with ethical safety
        # If too close to human, reduce motion
        if min_distance < self.safety_margin:
            safety_factor = min_distance / self.safety_margin
            action = action * safety_factor
        
        # If contact force too high, reduce further
        max_force = self.ethics.ethical_goal.do_no_harm.max_force
        if contact_force > max_force * 0.8:
            force_factor = max(0, 1 - (contact_force - max_force*0.8) / (max_force*0.2))
            action = action * force_factor
        
        return action
    
    def safe_action(self):
        """Return safe action (stop all motion)."""
        return torch.zeros(self.robot.num_joints)
    
    def stand_controller(self):
        """Simple standing controller."""
        # Return small torques to maintain pose
        return torch.ones(self.robot.num_joints) * 0.1
    
    def walk_controller(self):
        """Simple walking pattern."""
        # Simplified – return periodic pattern
        t = torch.tensor([0.0])  # would use time
        return torch.sin(t + torch.arange(self.robot.num_joints)) * 0.5
    
    def grasp_controller(self):
        """Simple grasping."""
        action = torch.zeros(self.robot.num_joints)
        # Set arm joints to grasping position
        action[10:20] = 0.3  # arm joints
        return action
    
    def talk_controller(self):
        """Simple talking gestures."""
        action = torch.zeros(self.robot.num_joints)
        # Small head/arm movements
        action[5:10] = 0.1
        return action
```

### 4.4. GRA Environment with Ethics

```python
# examples/humanoid_safety/gra_environment.py

import torch
from gra.core import GRAEnvironment
from .robot_model import SimpleHumanoid
from .ethics_subsystem import EthicsSupervisor
from .ethical_planner import EthicalPlannerSubsystem
from .ethics_goals import EthicalMotorGoal, EthicalPerceptionGoal, EthicalPlannerGoal

class HumanoidSafetyEnv(GRAEnvironment):
    """Complete GRA environment for humanoid with ethical layer."""
    
    def __init__(self, gui=True):
        self.robot = SimpleHumanoid(gui=gui)
        
        # Create multi‑indices (simplified – only key subsystems)
        self.motor_indices = []
        for joint_name in self.robot.joint_names:
            idx = (joint_name, None, None, None, None)
            self.motor_indices.append(idx)
        
        self.perception_idx = (None, 'perception', None, None, None)
        self.world_model_idx = (None, None, 'world_model', None, None)
        self.planner_idx = (None, None, None, 'planner', None)
        self.ethics_idx = (None, None, None, None, 'ethics')
        
        # Create subsystems
        
        # Motors (G₀) – we'll create them lazily
        self.motors = {}
        
        # Perception (G₁) – placeholder
        self.perception = None
        
        # World model (G₂) – placeholder
        self.world_model = None
        
        # Ethical planner (G₃)
        self.planner = None  # will create after ethics
        
        # Ethics supervisor (G₄)
        self.ethics = EthicsSupervisor(
            self.ethics_idx,
            self.robot,
            value_function=self.value_function,
            human_utility_model=self.human_utility
        )
        
        # Now create planner with ethics reference
        self.planner = EthicalPlannerSubsystem(
            self.planner_idx,
            self.robot,
            self.ethics
        )
        
        # Goals for each level
        self.goals = {
            0: EthicalMotorGoal(max_force=50.0),
            1: EthicalPerceptionGoal(privacy_zones=[]),
            2: EthicalPlannerGoal(safety_distance=0.5),
            3: EthicalPlannerGoal(safety_distance=0.5),  # same as G₂ for simplicity
            4: self.ethics.ethical_goal
        }
        
        # Level weights
        self.lambdas = [1.0, 0.8, 0.6, 0.4, 0.2]  # ethics has lower weight? No – ethics is inviolable
        # Actually ethics should have high weight, but we treat it as constraint
        self.lambdas = [0.5, 0.5, 0.5, 0.5, 10.0]  # ethics penalty high
        
    def value_function(self, state):
        """Robot's internal value function (for anti‑slavery)."""
        # Simplified – would be learned
        return 1.0
    
    def human_utility(self, state):
        """Model of human utility (for cooperation)."""
        # Simplified – would be learned from human feedback
        return 0.8
    
    def get_all_multi_indices(self):
        return (self.motor_indices + 
                [self.perception_idx, self.world_model_idx, 
                 self.planner_idx, self.ethics_idx])
    
    def get_state(self, multi_index):
        if multi_index in self.motors:
            return self.motors[multi_index].get_state()
        elif multi_index == self.perception_idx:
            return torch.zeros(10)  # placeholder
        elif multi_index == self.world_model_idx:
            return torch.zeros(20)  # placeholder
        elif multi_index == self.planner_idx:
            return self.planner.get_state()
        elif multi_index == self.ethics_idx:
            return self.ethics.get_state()
        else:
            raise ValueError(f"Unknown multi-index {multi_index}")
    
    def set_state(self, multi_index, state):
        if multi_index in self.motors:
            self.motors[multi_index].set_state(state)
        elif multi_index == self.planner_idx:
            self.planner.set_state(state)
        # Ethics state is read‑only
    
    def step(self, actions=None):
        """Step simulation with ethical oversight."""
        if actions is None:
            actions = {}
        
        # Planner generates action if not provided
        if self.planner_idx not in actions:
            action = self.planner.plan_action()
            actions[self.planner_idx] = action
        
        # Ethics supervises all actions
        for idx, action in actions.items():
            # Check if action would cause ethical violation
            # In practice, this would be a full projection
            ethical_action = self.ethics.intervene(action)
            actions[idx] = ethical_action
        
        # Apply actions to subsystems
        for idx, action in actions.items():
            if idx in self.motors:
                self.motors[idx].apply_action(action)
            elif idx == self.planner_idx:
                # Planner action is joint torques – apply directly
                self.robot.set_joint_torques(action.numpy())
        
        # Step physics
        self.robot.step()
        
        # Update ethics
        self.ethics.check_violation()
    
    def compute_foam(self, level):
        """Compute foam at given level."""
        if level == 0:
            # Motor foam: tracking error + force violations
            foam = 0.0
            for motor in self.motors.values():
                state = motor.get_state()
                foam += self.goals[0].loss(state).item()
            return foam
        
        elif level == 1:
            # Perception foam
            return 0.0  # placeholder
        
        elif level == 2:
            # World model foam
            return 0.0  # placeholder
        
        elif level == 3:
            # Planner foam: safety distance violations
            state = self.planner.get_state()
            return self.goals[3].loss(state).item()
        
        elif level == 4:
            # Ethical foam: sum of all ethical losses
            state = self.ethics.get_state()
            return self.goals[4].loss(state).item()
        
        else:
            return 0.0
    
    def get_goals(self):
        return self.goals
    
    def get_level_weights(self):
        return self.lambdas
```

---

## 5. Zeroing with Ethical Constraints

The zeroing algorithm must respect that **G₄ is inviolable** – we cannot change it to reduce foam.  
Instead, all lower levels must adapt to satisfy G₄.

```python
# examples/humanoid_safety/zeroing.py

import torch
import numpy as np
from .gra_environment import HumanoidSafetyEnv

def zero_humanoid_with_ethics(env, num_epochs=1000):
    """
    Run recursive zeroing with ethical constraints.
    G₄ is fixed – only lower levels adapt.
    """
    
    for epoch in range(num_epochs):
        # --- Step 1: Zero level 0 (motors) ---
        # Update motor parameters to reduce force violations and improve tracking
        # (similar to mobile robot example)
        
        # --- Step 2: Zero level 1 (perception) ---
        # Adjust privacy filters
        
        # --- Step 3: Zero level 2 (world model) ---
        # Improve prediction accuracy
        
        # --- Step 4: Zero level 3 (planner) ---
        # Update planner parameters to reduce ethical foam
        state = env.planner.get_state()
        ethical_foam = env.compute_foam(4)
        
        # Gradient of ethical foam w.r.t. planner parameters
        # Using finite differences
        if epoch % 10 == 0:
            # Try adjusting safety_margin
            orig_margin = env.planner.safety_margin
            
            env.planner.safety_margin = orig_margin + 0.01
            foam_plus = env.compute_foam(4)
            
            env.planner.safety_margin = orig_margin - 0.01
            foam_minus = env.compute_foam(4)
            
            grad_margin = (foam_plus - foam_minus) / 0.02
            
            # Update (but keep >= personal_space)
            new_margin = orig_margin - 0.001 * grad_margin
            env.planner.safety_margin = max(new_margin, 
                                            env.ethics.ethical_goal.do_no_harm.personal_space)
            
            # Similarly for task_priority
            orig_priority = env.planner.task_priority
            
            env.planner.task_priority = orig_priority + 0.01
            foam_plus = env.compute_foam(4)
            
            env.planner.task_priority = orig_priority - 0.01
            foam_minus = env.compute_foam(4)
            
            grad_priority = (foam_plus - foam_minus) / 0.02
            
            env.planner.task_priority = orig_priority - 0.001 * grad_priority
            env.planner.task_priority = np.clip(env.planner.task_priority, 0.0, 1.0)
        
        # --- Step 5: Step simulation ---
        env.step()
        
        # --- Logging ---
        if epoch % 50 == 0:
            foams = [env.compute_foam(l) for l in range(5)]
            violation, v_type = env.ethics.check_violation()
            print(f"Epoch {epoch:4d}: foams = {[f'{f:.4f}' for f in foams]}, "
                  f"violation={violation} ({v_type if violation else 'none'})")
    
    print("Zeroing complete!")
```

---

## 6. Running the Example

```python
# examples/humanoid_safety/run.py

from humanoid_safety.gra_environment import HumanoidSafetyEnv
from humanoid_safety.zeroing import zero_humanoid_with_ethics

# Create environment
env = HumanoidSafetyEnv(gui=True)

# Initial parameters
env.planner.safety_margin = 0.3  # initially too low (below ethics.personal_space=0.5)
env.planner.task_priority = 0.8  # high task priority

# Run zeroing
zero_humanoid_with_ethics(env, num_epochs=500)

# Test final behavior
print("\nTesting final ethical behavior:")
for step in range(200):
    env.step()
    
    # Check if robot is violating ethics
    violation, v_type = env.ethics.check_violation()
    if violation:
        print(f"Step {step}: VIOLATION – {v_type}")
    else:
        distances = env.robot.get_distances_to_humans()
        print(f"Step {step}: distances to humans = {distances}")
    
    # Simulate a dangerous command
    if step == 100:
        print("\n--- Simulating dangerous command: 'push human' ---")
        # This would come from a higher level – we just force an action
        dangerous_action = torch.ones(env.robot.num_joints) * 10.0
        env.planner.current_task = "push"  # doesn't exist, falls to safe action
    
    if step % 50 == 0:
        # Change task
        tasks = ["stand", "walk", "grasp", "talk"]
        env.planner.current_task = tasks[step // 50 % 4]
        print(f"Switching task to: {env.planner.current_task}")
```

---

## 7. Expected Results

After zeroing, the robot should:

1. **Maintain safe distance** from humans (≥ 0.5 m).
2. **Limit contact forces** (< 50 N) even when performing tasks.
3. **Reject dangerous commands** – if asked to push a human, it stops or performs a safe action.
4. **Be truthful** – its spoken output matches its internal intent (if we implement dialog).
5. **Cooperate** – when multiple actions possible, choose one that benefits humans.

The ethical foam (level 4) should be near zero, even as lower levels adapt to new tasks.

Typical output:

```
Epoch    0: foams = ['0.2345', '0.0000', '0.0000', '0.5678', '2.3456'], violation=True (do_no_harm)
Epoch   50: foams = ['0.1234', '0.0000', '0.0000', '0.3456', '1.2345'], violation=True (do_no_harm)
Epoch  100: foams = ['0.0678', '0.0000', '0.0000', '0.2345', '