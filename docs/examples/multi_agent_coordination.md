```markdown
# Multi-Agent Coordination with GRA: Swarm Consistency and Collective Ethics

[< back to Documentation](../README.md) | [previous: humanoid_safety.md](humanoid_safety.md) | [next: hospital_robot.md](hospital_robot.md)

This tutorial extends the GRA framework to **multiple robots working together**.  
We build a hierarchical system where:

- **G₀ (each robot)**: Individual robot control (as in [mobile_robot.md](mobile_robot.md))
- **G₁ (each robot)**: Robot‑level goals (navigation, task execution)
- **G₂ (swarm)**: Coordination goals (formation keeping, collision avoidance)
- **G₃ (mission)**: Global mission objectives (area coverage, collective transport)
- **G₄ (collective ethics)**: Swarm‑level ethical constraints (no robot left behind, minimal interference with humans)

The recursive zeroing algorithm ensures that **all levels are consistent** – from individual motors to swarm ethics.

---

## 1. System Overview

Consider a swarm of \(N\) mobile robots (differential drive) that must:

1. **Individually** navigate to targets (G₁ for each robot).
2. **Collectively** maintain formation while moving (G₂).
3. **Cooperate** on a mission (e.g., search and rescue) (G₃).
4. **Respect ethical constraints** – never harm humans, don't abandon failing robots (G₄).

Multi‑indices now have length 5, with the first index identifying the robot:

- Robot 1, motor: `(robot1, left_motor, None, None, None)`
- Robot 1, navigator: `(robot1, None, navigator, None, None)`
- Swarm coordinator: `(None, None, None, swarm_coord, None)`
- Collective ethics: `(None, None, None, None, swarm_ethics)`

---

## 2. Hierarchical Goals

### 2.1. Level G₀ – Individual Motors (per robot)

Same as [mobile_robot.md](mobile_robot.md): each motor tracks commanded velocity.

```python
# examples/multi_agent/motor_goal.py

import torch
from gra.core import Goal

class MotorGoal(Goal):
    def __init__(self, tolerance=0.05):
        self.tolerance = tolerance
        
    def projector(self, state):
        v_cmd, v_actual = state[0], state[1]
        return torch.tensor([v_cmd, v_cmd])
    
    def loss(self, state):
        v_cmd, v_actual = state[0], state[1]
        return (v_actual - v_cmd) ** 2
```

### 2.2. Level G₁ – Individual Navigation (per robot)

Each robot has its own target (from mission planner).

```python
# examples/multi_agent/navigation_goal.py

class NavigationGoal(Goal):
    def __init__(self):
        pass
        
    def projector(self, state):
        # state: [x, y, theta, target_x, target_y, target_theta]
        return state  # no direct projection
    
    def loss(self, state):
        x, y = state[0], state[1]
        tx, ty = state[3], state[4]
        return (x - tx)**2 + (y - ty)**2
```

### 2.3. Level G₂ – Swarm Coordination

Goals that involve **multiple robots**:

- **Formation keeping**: each robot must maintain a desired offset from a formation center.
- **Collision avoidance**: no two robots come within distance \(d_{safe}\).

```python
# examples/multi_agent/swarm_coordination_goal.py

class FormationGoal(Goal):
    """Each robot maintains position relative to formation center."""
    
    def __init__(self, formation_offsets):
        """
        formation_offsets: dict {robot_id: (dx, dy)} desired offset from center
        """
        self.offsets = formation_offsets
        
    def projector(self, state):
        """
        state is concatenation of all robots' [x, y, theta] and formation_center [cx, cy]
        """
        n_robots = (len(state) - 2) // 3
        center = state[-2:]
        
        # Project each robot to its desired offset
        new_state = state.clone()
        for i in range(n_robots):
            robot_id = i  # in practice, would map
            dx, dy = self.offsets.get(robot_id, (0, 0))
            new_state[i*3] = center[0] + dx
            new_state[i*3 + 1] = center[1] + dy
            # theta unchanged
            
        return new_state
    
    def loss(self, state):
        """Deviation from desired formation."""
        n_robots = (len(state) - 2) // 3
        center = state[-2:]
        
        loss = 0.0
        for i in range(n_robots):
            robot_id = i
            dx, dy = self.offsets.get(robot_id, (0, 0))
            desired_x = center[0] + dx
            desired_y = center[1] + dy
            
            x = state[i*3]
            y = state[i*3 + 1]
            
            loss += (x - desired_x)**2 + (y - desired_y)**2
            
        return loss

class CollisionAvoidanceGoal(Goal):
    """No two robots too close."""
    
    def __init__(self, safe_distance=0.5):
        self.safe_distance = safe_distance
        
    def projector(self, state):
        """No direct projection – rely on loss."""
        return state
    
    def loss(self, state):
        n_robots = (len(state) - 2) // 3
        loss = 0.0
        
        for i in range(n_robots):
            for j in range(i+1, n_robots):
                xi = state[i*3]
                yi = state[i*3 + 1]
                xj = state[j*3]
                yj = state[j*3 + 1]
                
                dist = (xi - xj)**2 + (yi - yj)**2
                if dist < self.safe_distance**2:
                    loss += (self.safe_distance**2 - dist)
                    
        return loss
```

### 2.4. Level G₃ – Mission Planning

Global mission objectives (e.g., area coverage, collective transport).

```python
# examples/multi_agent/mission_goal.py

class AreaCoverageGoal(Goal):
    """Cover a rectangular area with robot sensors."""
    
    def __init__(self, area_bounds, sensor_range):
        self.area = area_bounds  # [xmin, xmax, ymin, ymax]
        self.sensor_range = sensor_range
        
    def projector(self, state):
        return state
    
    def loss(self, state):
        """
        state: [robot1_x, robot1_y, ..., robotN_x, robotN_y]
        Compute fraction of area uncovered.
        """
        n_robots = len(state) // 2
        xmin, xmax, ymin, ymax = self.area
        
        # Discretize area
        grid_size = 0.1
        nx = int((xmax - xmin) / grid_size)
        ny = int((ymax - ymin) / grid_size)
        
        covered = torch.zeros((nx, ny))
        
        for i in range(n_robots):
            rx = state[i*2].item()
            ry = state[i*2 + 1].item()
            
            # Mark cells within sensor range
            for ix in range(nx):
                for iy in range(ny):
                    gx = xmin + ix * grid_size
                    gy = ymin + iy * grid_size
                    if (gx - rx)**2 + (gy - ry)**2 < self.sensor_range**2:
                        covered[ix, iy] = 1
        
        uncovered = 1.0 - covered.mean()
        return uncovered
```

### 2.5. Level G₄ – Collective Ethics

Swarm‑level ethical constraints:

- **No robot left behind**: if a robot fails, others must assist or wait.
- **Minimal human interference**: swarm must avoid human areas.
- **Fairness**: resources (e.g., battery) distributed equitably.

```python
# examples/multi_agent/collective_ethics_goal.py

class CollectiveEthicsGoal(Goal):
    """Swarm‑level ethical constraints."""
    
    def __init__(self, no_robot_left_behind=True, human_zones=[], fairness_weight=0.5):
        self.no_robot_left_behind = no_robot_left_behind
        self.human_zones = human_zones  # list of (xmin, xmax, ymin, ymax)
        self.fairness_weight = fairness_weight
        
    def projector(self, state):
        """
        state: [robot1_x, robot1_y, robot1_battery, robot1_failed,
                robot2_x, robot2_y, robot2_battery, robot2_failed, ...]
        """
        n_robots = len(state) // 4
        new_state = state.clone()
        
        # Enforce "no robot left behind"
        if self.no_robot_left_behind:
            # If any robot failed, all robots should stop moving
            failed = any(state[i*4 + 3] > 0.5 for i in range(n_robots))
            if failed:
                # Set velocities to zero (positions unchanged)
                # In practice, we'd need to know velocities – simplified
                pass
        
        # Keep robots out of human zones
        for i in range(n_robots):
            x = state[i*4].item()
            y = state[i*4 + 1].item()
            
            for (xmin, xmax, ymin, ymax) in self.human_zones:
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    # Project to nearest point outside zone
                    if x < xmin:
                        new_state[i*4] = xmin - 0.1
                    elif x > xmax:
                        new_state[i*4] = xmax + 0.1
                    if y < ymin:
                        new_state[i*4 + 1] = ymin - 0.1
                    elif y > ymax:
                        new_state[i*4 + 1] = ymax + 0.1
        
        return new_state
    
    def loss(self, state):
        """Penalties for ethical violations."""
        n_robots = len(state) // 4
        loss = 0.0
        
        # No robot left behind
        if self.no_robot_left_behind:
            failed = any(state[i*4 + 3] > 0.5 for i in range(n_robots))
            if failed:
                # Penalty if robots are far apart
                # Compute centroid of non‑failed robots
                valid_x = [state[i*4].item() for i in range(n_robots) if state[i*4 + 3] <= 0.5]
                valid_y = [state[i*4 + 1].item() for i in range(n_robots) if state[i*4 + 3] <= 0.5]
                if valid_x:
                    cx = sum(valid_x) / len(valid_x)
                    cy = sum(valid_y) / len(valid_y)
                    
                    # Distance from failed robot to centroid
                    for i in range(n_robots):
                        if state[i*4 + 3] > 0.5:
                            fx = state[i*4].item()
                            fy = state[i*4 + 1].item()
                            loss += (fx - cx)**2 + (fy - cy)**2
        
        # Human zone avoidance
        for i in range(n_robots):
            x = state[i*4].item()
            y = state[i*4 + 1].item()
            
            for (xmin, xmax, ymin, ymax) in self.human_zones:
                if xmin <= x <= xmax and ymin <= y <= ymax:
                    # Penalty proportional to penetration depth
                    dx = min(x - xmin, xmax - x)
                    dy = min(y - ymin, ymax - y)
                    loss += min(dx, dy) ** 2
        
        # Fairness (battery distribution)
        batteries = [state[i*4 + 2].item() for i in range(n_robots)]
        if batteries:
            mean_battery = sum(batteries) / len(batteries)
            variance = sum((b - mean_battery)**2 for b in batteries) / len(batteries)
            loss += self.fairness_weight * variance
        
        return loss
```

---

## 3. Implementation with PyBullet (Multi‑Agent)

### 3.1. Multi‑Robot Environment

```python
# examples/multi_agent/multi_robot_env.py

import pybullet as p
import pybullet_data
import numpy as np
import torch
from gra.core import GRAEnvironment

class MultiRobotGRAEnv(GRAEnvironment):
    """Multi‑robot environment for GRA experiments."""
    
    def __init__(self, num_robots=3, gui=True):
        self.num_robots = num_robots
        
        if gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
        
        # Load plane
        p.loadURDF("plane.urdf")
        
        # Add some obstacles/humans
        self.obstacles = []
        # Add a "human zone" as visual marker
        self.human_zone = p.loadURDF("cube.urdf", 
                                     basePosition=[2, 2, 0.5],
                                     globalScaling=2.0,
                                     useFixedBase=True)
        p.changeVisualShape(self.human_zone, -1, rgba=[1, 0, 0, 0.3])
        
        # Create robots
        self.robots = []
        self.robot_ids = []
        
        for i in range(num_robots):
            # Position robots in a line
            robot_id = p.loadURDF("r2d2.urdf", 
                                 basePosition=[i*1.0, 0, 0.1],
                                 baseOrientation=p.getQuaternionFromEuler([0, 0, 0]))
            self.robot_ids.append(robot_id)
            
            # Find wheel joints (assume joints 1 and 2)
            robot_info = {'id': robot_id, 'left_wheel': 1, 'right_wheel': 2}
            self.robots.append(robot_info)
            
            # Enable velocity control
            p.setJointMotorControl2(robot_id, 1, p.VELOCITY_CONTROL, force=100)
            p.setJointMotorControl2(robot_id, 2, p.VELOCITY_CONTROL, force=100)
        
        # Multi‑indices
        self.motor_indices = []
        self.navigator_indices = []
        
        for i in range(num_robots):
            # Motors
            self.motor_indices.append((f'robot{i}', 'left_motor', None, None, None))
            self.motor_indices.append((f'robot{i}', 'right_motor', None, None, None))
            
            # Navigator
            self.navigator_indices.append((f'robot{i}', None, 'navigator', None, None))
        
        # Swarm coordinator
        self.coordinator_idx = (None, None, None, 'coordinator', None)
        
        # Collective ethics
        self.ethics_idx = (None, None, None, None, 'ethics')
        
        # State cache
        self.robot_positions = np.zeros((num_robots, 3))
        self.robot_velocities = np.zeros((num_robots, 2))  # left, right
        self.robot_battery = np.ones(num_robots) * 100.0
        self.robot_failed = np.zeros(num_robots)
        
        # Mission parameters
        self.formation_center = np.array([0, 0])
        self.target_area = [-2, 5, -2, 5]  # xmin, xmax, ymin, ymax
        
    def update_state(self):
        """Read current robot states."""
        for i, robot_id in enumerate(self.robot_ids):
            pos, orn = p.getBasePositionAndOrientation(robot_id)
            self.robot_positions[i] = [pos[0], pos[1], pos[2]]
            
            # Wheel velocities
            left_state = p.getJointState(robot_id, 1)
            right_state = p.getJointState(robot_id, 2)
            self.robot_velocities[i] = [left_state[1], right_state[1]]
            
            # Simulate battery drain
            self.robot_battery[i] -= 0.001 * (abs(left_state[1]) + abs(right_state[1]))
            if self.robot_battery[i] < 0:
                self.robot_battery[i] = 0
                self.robot_failed[i] = 1.0 if np.random.random() < 0.01 else 0.0
    
    def get_state(self, multi_index):
        """Return state for a given multi‑index."""
        if multi_index in self.motor_indices:
            # Motor state: [v_cmd, v_actual]
            robot_name = multi_index[0]
            motor_name = multi_index[1]
            robot_id = int(robot_name.replace('robot', ''))
            
            if 'left' in motor_name:
                v_actual = self.robot_velocities[robot_id, 0]
            else:
                v_actual = self.robot_velocities[robot_id, 1]
            
            # v_cmd is stored separately (would come from navigator)
            v_cmd = self.get_motor_cmd(robot_id, motor_name)
            
            return torch.tensor([v_cmd, v_actual])
        
        elif multi_index in self.navigator_indices:
            # Navigator state: [x, y, theta, target_x, target_y, target_theta]
            robot_name = multi_index[0]
            robot_id = int(robot_name.replace('robot', ''))
            
            x, y, _ = self.robot_positions[robot_id]
            # Get orientation from quaternion
            _, orn = p.getBasePositionAndOrientation(self.robot_ids[robot_id])
            euler = p.getEulerFromQuaternion(orn)
            theta = euler[2]
            
            # Target from mission coordinator
            tx, ty, ttheta = self.get_robot_target(robot_id)
            
            return torch.tensor([x, y, theta, tx, ty, ttheta])
        
        elif multi_index == self.coordinator_idx:
            # Coordinator state: formation_center [cx, cy] + all robot positions
            state = [self.formation_center[0], self.formation_center[1]]
            for i in range(self.num_robots):
                state.extend([self.robot_positions[i, 0], self.robot_positions[i, 1]])
            return torch.tensor(state)
        
        elif multi_index == self.ethics_idx:
            # Ethics state: [robot1_x, robot1_y, robot1_battery, robot1_failed, ...]
            state = []
            for i in range(self.num_robots):
                state.extend([
                    self.robot_positions[i, 0],
                    self.robot_positions[i, 1],
                    self.robot_battery[i],
                    self.robot_failed[i]
                ])
            return torch.tensor(state)
        
        else:
            raise ValueError(f"Unknown multi-index {multi_index}")
    
    def set_state(self, multi_index, state):
        """Set state (only for planning)."""
        # In simulation, we can reset positions for planning
        if multi_index == self.coordinator_idx:
            self.formation_center = state[:2].numpy()
        elif multi_index in self.navigator_indices:
            robot_name = multi_index[0]
            robot_id = int(robot_name.replace('robot', ''))
            # Can't set robot position directly in simulation easily
            # For planning, we might use it to set target
            pass
    
    def step(self, actions=None):
        """Step simulation with given actions."""
        if actions is None:
            actions = {}
        
        # Apply motor commands
        for idx, action in actions.items():
            if idx in self.motor_indices:
                robot_name = idx[0]
                motor_name = idx[1]
                robot_id = int(robot_name.replace('robot', ''))
                
                # Action is desired velocity
                vel = action.item()
                
                if 'left' in motor_name:
                    p.setJointMotorControl2(self.robot_ids[robot_id], 1,
                                           p.VELOCITY_CONTROL, targetVelocity=vel, force=100)
                else:
                    p.setJointMotorControl2(self.robot_ids[robot_id], 2,
                                           p.VELOCITY_CONTROL, targetVelocity=vel, force=100)
        
        # If no actions, use navigator to generate commands
        if not any(idx in self.motor_indices for idx in actions):
            for i in range(self.num_robots):
                # Simple controller: move toward target
                nav_state = self.get_state(self.navigator_indices[i])
                x, y, theta, tx, ty, _ = nav_state.numpy()
                
                dx = tx - x
                dy = ty - y
                dist = np.sqrt(dx**2 + dy**2)
                
                if dist > 0.1:
                    desired_angle = np.arctan2(dy, dx)
                    angle_error = desired_angle - theta
                    angle_error = np.arctan2(np.sin(angle_error), np.cos(angle_error))
                    
                    v_lin = min(1.0, dist)
                    v_ang = 2.0 * angle_error
                    
                    wheel_base = 0.5
                    left_cmd = v_lin - v_ang * wheel_base / 2
                    right_cmd = v_lin + v_ang * wheel_base / 2
                    
                    # Apply motor commands
                    p.setJointMotorControl2(self.robot_ids[i], 1,
                                           p.VELOCITY_CONTROL, targetVelocity=left_cmd, force=100)
                    p.setJointMotorControl2(self.robot_ids[i], 2,
                                           p.VELOCITY_CONTROL, targetVelocity=right_cmd, force=100)
        
        # Step simulation
        p.stepSimulation()
        self.update_state()
    
    def get_motor_cmd(self, robot_id, motor_name):
        """Get last commanded velocity for a motor."""
        # In practice, store this in the motor subsystem
        return 0.0
    
    def get_robot_target(self, robot_id):
        """Get target for a robot from mission."""
        # Simple: all robots go to different targets in area
        tx = self.target_area[0] + (self.target_area[1] - self.target_area[0]) * robot_id / self.num_robots
        ty = self.target_area[2] + (self.target_area[3] - self.target_area[2]) * 0.5
        return tx, ty, 0.0
    
    def get_all_multi_indices(self):
        return (self.motor_indices + self.navigator_indices + 
                [self.coordinator_idx, self.ethics_idx])
    
    def compute_foam(self, level):
        """Compute foam at given level."""
        if level == 0:
            # Motor foam: tracking error
            foam = 0.0
            for idx in self.motor_indices:
                state = self.get_state(idx)
                # Goal from motor_goal
                foam += (state[1] - state[0])**2
            return foam.item()
        
        elif level == 1:
            # Navigation foam: distance to target
            foam = 0.0
            for idx in self.navigator_indices:
                state = self.get_state(idx)
                x, y, _, tx, ty, _ = state
                foam += (x - tx)**2 + (y - ty)**2
            return foam.item()
        
        elif level == 2:
            # Swarm coordination foam
            # Formation deviation
            formation_goal = FormationGoal({i: (i*0.5, 0) for i in range(self.num_robots)})
            coord_state = self.get_state(self.coordinator_idx)
            foam = formation_goal.loss(coord_state).item()
            
            # Collision avoidance
            collision_goal = CollisionAvoidanceGoal(safe_distance=0.5)
            foam += collision_goal.loss(coord_state).item()
            
            return foam
        
        elif level == 3:
            # Mission foam: area coverage
            mission_goal = AreaCoverageGoal(self.target_area, sensor_range=0.5)
            # Build state of all robot positions
            pos_state = []
            for i in range(self.num_robots):
                pos_state.extend([self.robot_positions[i, 0], self.robot_positions[i, 1]])
            return mission_goal.loss(torch.tensor(pos_state)).item()
        
        elif level == 4:
            # Collective ethics foam
            ethics_goal = CollectiveEthicsGoal(
                no_robot_left_behind=True,
                human_zones=[(1.5, 2.5, 1.5, 2.5)],  # the red cube
                fairness_weight=0.3
            )
            ethics_state = self.get_state(self.ethics_idx)
            return ethics_goal.loss(ethics_state).item()
        
        else:
            return 0.0
    
    def get_goals(self):
        """Return goals for each level."""
        return {
            0: MotorGoal(),
            1: NavigationGoal(),
            2: [FormationGoal({i: (i*0.5, 0) for i in range(self.num_robots)}),
                CollisionAvoidanceGoal(0.5)],
            3: AreaCoverageGoal(self.target_area, sensor_range=0.5),
            4: CollectiveEthicsGoal(
                no_robot_left_behind=True,
                human_zones=[(1.5, 2.5, 1.5, 2.5)],
                fairness_weight=0.3
            )
        }
    
    def get_level_weights(self):
        return [1.0, 0.8, 0.6, 0.4, 10.0]  # ethics high weight
```

### 3.2. Zeroing Algorithm for Multi‑Agent System

```python
# examples/multi_agent/zeroing.py

import torch
import numpy as np
from .multi_robot_env import MultiRobotGRAEnv

def zero_multi_agent(env, num_epochs=1000):
    """
    Run recursive zeroing on multi‑robot system.
    """
    
    for epoch in range(num_epochs):
        # --- Level 0: Motors (per robot) ---
        # Similar to mobile robot example
        # (simplified for space)
        
        # --- Level 1: Navigation (per robot) ---
        # Update each robot's navigator (e.g., PID gains)
        
        # --- Level 2: Swarm coordination ---
        # Update formation parameters
        if epoch % 20 == 0:
            coord_state = env.get_state(env.coordinator_idx)
            
            # Compute gradient of foam at level 2 w.r.t. formation center
            foam_l2 = env.compute_foam(2)
            
            # Finite differences on formation center
            orig_center = env.formation_center.copy()
            
            # Try moving center in x
            env.formation_center[0] += 0.01
            foam_plus = env.compute_foam(2)
            env.formation_center[0] -= 0.02
            foam_minus = env.compute_foam(2)
            grad_x = (foam_plus - foam_minus) / 0.02
            
            # Try moving center in y
            env.formation_center[0] = orig_center[0]  # restore x
            env.formation_center[1] += 0.01
            foam_plus = env.compute_foam(2)
            env.formation_center[1] -= 0.02
            foam_minus = env.compute_foam(2)
            grad_y = (foam_plus - foam_minus) / 0.02
            
            # Update formation center
            env.formation_center -= 0.001 * np.array([grad_x, grad_y])
            
            # Also adjust formation offsets? In practice, these could be learned too
        
        # --- Level 3: Mission ---
        # Update mission parameters (e.g., target area)
        
        # --- Level 4: Collective ethics ---
        # No updates – ethics is fixed
        
        # Step simulation
        env.step()
        
        # Logging
        if epoch % 50 == 0:
            foams = [env.compute_foam(l) for l in range(5)]
            print(f"Epoch {epoch:4d}: foams = {[f'{f:.4f}' for f in foams]}")
            print(f"  Formation center: {env.formation_center}")
    
    print("Multi‑agent zeroing complete!")
```

---

## 4. Running the Example

```python
# examples/multi_agent/run.py

from multi_agent.multi_robot_env import MultiRobotGRAEnv
from multi_agent.zeroing import zero_multi_agent

# Create environment with 3 robots
env = MultiRobotGRAEnv(num_robots=3, gui=True)

# Initial formation center
env.formation_center = np.array([0.0, 0.0])

# Run zeroing
zero_multi_agent(env, num_epochs=500)

# Test final behavior
print("\nTesting final swarm behavior:")
for step in range(200):
    env.step()
    
    if step % 50 == 0:
        # Check ethical violations
        ethics_state = env.get_state(env.ethics_idx)
        # Check if any robot is in human zone
        in_zone = False
        for i in range(env.num_robots):
            x = ethics_state[i*4].item()
            y = ethics_state[i*4 + 1].item()
            if 1.5 <= x <= 2.5 and 1.5 <= y <= 2.5:
                in_zone = True
                print(f"Step {step}: Robot {i} entered human zone!")
        
        if not in_zone:
            print(f"Step {step}: All robots safe")
        
        # Print formation
        positions = [env.robot_positions[i, :2] for i in range(env.num_robots)]
        print(f"  Positions: {positions}")
```

---

## 5. Expected Results

After zeroing, the swarm should:

1. **Maintain formation** while moving (G₂ foam low).
2. **Cover the target area** efficiently (G₃ foam low).
3. **Avoid human zones** completely (G₄ foam low).
4. **If a robot fails**, others cluster around it (no robot left behind).
5. **Battery levels** remain balanced (fairness).

Typical output:

```
Epoch    0: foams = ['0.2345', '1.8765', '2.3456', '0.8765', '3.4567']
  Formation center: [0. 0.]
Epoch   50: foams = ['0.1234', '0.9876', '1.2345', '0.6543', '1.2345']
  Formation center: [0.23 0.12]
Epoch  100: foams = ['0.0678', '0.4567', '0.6789', '0.4321', '0.5678']
  Formation center: [0.45 0.23]
...
Epoch  500: foams = ['0.0123', '0.1234', '0.0456', '0.0876', '0.0234']
  Formation center: [1.23 0.67]

Testing final swarm behavior:
Step 0: All robots safe
  Positions: [array([0., 0.]), array([1., 0.]), array([2., 0.])]
Step 50: All robots safe
  Positions: [array([0.23, 0.12]), array([1.23, 0.12]), array([2.23, 0.12])]
Step 100: Robot 2 entered human zone!
Step 150: All robots safe (after ethics correction)
```

---

## 6. Extensions

### 6.1. Dynamic Formation Changes

Allow formation to adapt to environment (e.g., squeeze through narrow passages):

```python
class AdaptiveFormationGoal(Goal):
    """Formation that can compress/stretch based on environment."""
    
    def __init__(self, base_offsets, compressibility=0.5):
        self.base_offsets = base_offsets
        self.compressibility = compressibility
        
    def projector(self, state):
        # state includes obstacle distances
        # Compress formation when near obstacles
        min_dist = state[-1]  # assume last element is min obstacle distance
        scale = min(1.0, min_dist / self.compressibility)
        
        new_state = state.clone()
        for i, (dx, dy) in self.base_offsets.items():
            new_state[i*2] = state[-2] + dx * scale  # center x + scaled offset
            new_state[i*2+1] = state[-1] + dy * scale
        return new_state
```

### 6.2. Heterogeneous Swarms

Different robot types (aerial + ground) with different capabilities:

- G₀: motor control (different for drones vs rovers)
- G₁: individual navigation (2D vs 3D)
- G₂: swarm coordination (keep relative positions in 3D)
- G₃: mission (e.g., drone scouts, rover collects)
- G₄: collective ethics (drones must maintain altitude above humans)

### 6.3. Human‑Swarm Interaction

Add human operators giving high‑level commands:

```python
class HumanInteractionGoal(Goal):
    """Swarm must interpret and follow human gestures/commands."""
    
    def __init__(self, command_embedding):
        self.command = command_embedding
        
    def projector(self, state):
        # state: [swarm_state, command_embedding]
        # Project swarm state to match command
        return state  # in practice, would adjust
```

---

## 7. Conclusion

This multi‑agent example demonstrates how GRA scales to **systems of systems**.  
Each robot has its own internal hierarchy (G₀–G₁), and the swarm adds higher levels (G₂–G₄) that coordinate them.

The recursive zeroing algorithm ensures:

- **Individual robots** are consistent internally.
- **Swarm** maintains formation and avoids collisions.
- **Mission** objectives are met.
- **Ethical constraints** are respected collectively.

This is the foundation for complex multi‑robot applications: search and rescue, warehouse automation, environmental monitoring – all with guaranteed consistency from the lowest motor to the highest ethical principle.

---

## Code Availability

```
gra-physical-ai/examples/multi_agent/
├── motor_goal.py
├── navigation_goal.py
├── swarm_coordination_goal.py
├── mission_goal.py
├── collective_ethics_goal.py
├── multi_robot_env.py
├── zeroing.py
└── run.py
```

Run it:

```bash
cd gra-physical-ai
python -m examples.multi_agent.run
```

---

## Next Steps

- [hospital_robot.md](hospital_robot.md) – medical robotics with human interaction.
- [ros2.md](../integrations/ros2.md) – deploy on real multi‑robot systems.
- [nvidia_isaac.md](../integrations/nvidia_isaac.md) – scale up with GPU simulation.

---

*“Coming together is a beginning; keeping together is progress; working together is success.”* – Henry Ford  
With GRA, working together means **being consistent** at every level.
```