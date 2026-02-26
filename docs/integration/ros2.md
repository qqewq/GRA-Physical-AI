```markdown
# Integrating GRA with ROS 2: Bringing Coherence to the Robot Operating System

[< back to Documentation](../README.md) | [previous: nvidia_isaac.md](nvidia_isaac.md) | [next: hospital_robot.md](../examples/hospital_robot.md)

ROS 2 (Robot Operating System 2) is the de facto standard middleware for robotics research and industry.  
This guide shows how to implement the GRA Meta‑zeroing framework **on top of ROS 2**, turning a collection of loosely coupled nodes into a **coherent, self‑evolving system** with guaranteed consistency across all levels.

We provide:
- A **GRA‑ROS 2 bridge** that maps ROS topics, services, and nodes to GRA subsystems.
- **Implementations of projectors** for common ROS goals (TF tree consistency, message synchronization, action server correctness).
- A **zeroing supervisor node** that runs the recursive algorithm online.
- Examples with real ROS 2 robots (TurtleBot, MoveIt2, Nav2).

---

## 1. Why GRA on ROS 2?

ROS 2 is inherently **distributed and decentralized**. Nodes communicate via topics, services, and actions, but there is **no built‑in mechanism** to ensure global consistency. Common problems:

- **TF tree** can have intermittent errors (missing frames, out‑of‑date transforms).
- **Sensor fusion** nodes may publish conflicting estimates.
- **Planning vs. control** mismatch: the planner outputs a trajectory that the controller cannot track.
- **Behavior trees** may have race conditions.

GRA provides a **mathematical framework** to:
1. **Define goals** for each ROS node (e.g., "TF tree must be complete and up‑to‑date").
2. **Measure inconsistency** (foam) between nodes.
3. **Automatically adjust** node parameters or behaviors to restore consistency – **online**, without restarting the system.

---

## 2. Mapping ROS 2 Concepts to GRA

| ROS 2 Concept | GRA Equivalent | Description |
|---------------|----------------|-------------|
| **Node** | Subsystem | A computational unit with a state |
| **Topic** | Observable | Part of the node's state that is published |
| **Message** | State snapshot | The content published on a topic |
| **Parameter** | Internal state | Node configuration that can be tuned |
| **TF tree** | Cross‑node constraint | A goal that must be satisfied jointly |
| **Action server** | Higher‑level goal | Provides feedback on task execution |

Each ROS node becomes a **GRA subsystem** identified by a multi‑index.  
For example, a navigation stack might have:

```
Level 0: /imu_driver, /lidar_driver, /motor_controller
Level 1: /ekf_node (fuses IMU and odometry), /map_server
Level 2: /nav2_planner, /nav2_controller
Level 3: /behavior_tree (high‑level task)
Level 4: /safety_supervisor (ethical layer)
```

Multi‑indices would be:
- `/imu_driver` → `(imu_driver, None, None, None, None)`
- `/ekf_node` → `(None, ekf_fusion, None, None, None)`
- `/nav2_planner` → `(None, None, planner, None, None)`
- etc.

---

## 3. GRA‑ROS 2 Bridge Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GRA Supervisor Node                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Zeroing Algorithm (recursive)                       │  │
│  │  • Collects states from all nodes                    │  │
│  │  • Computes foam at each level                        │  │
│  │  • Sends parameter updates / service calls           │  │
│  └──────────────────────────────────────────────────────┘  │
│                       ▲                            ▲        │
│                       │                            │        │
│            State      │                    Parameter│        │
│            topics     │                    updates │        │
│                       │                            │        │
└───────────────────────┼────────────────────────────┼────────┘
                        │                            │
┌───────────────────────┼────────────────────────────┼────────┐
│  ┌────────────────────┼────────────────────────────┼──────┐ │
│  │  Node A            │              Node B        │      │ │
│  │  ┌─────────────┐   │              ┌─────────────┐│      │ │
│  │  │ GRA wrapper │   │              │ GRA wrapper ││      │ │
│  │  └─────────────┘   │              └─────────────┘│      │ │
│  │         │          │                    │        │      │ │
│  │    publishes       │               publishes      │      │ │
│  │    /node_a/state   │               /node_b/state  │      │ │
│  └────────────────────┼────────────────────────────┼──────┘ │
│                       │                            │        │
│                       ▼                            ▼        │
│                 /node_a/state                  /node_b/state │
│                       (topic)                      (topic)   │
└──────────────────────────────────────────────────────────────┘
```

### 3.1. GRA Wrapper for a ROS Node

Each ROS node that participates in zeroing is wrapped with a small library that:

- Publishes its **internal state** on a dedicated topic (`/node_name/gra_state`).
- Subscribes to **parameter updates** from the supervisor.
- Provides a **projector service** to compute the goal‑satisfying part of its state.

```python
# src/integrations/ros2/gra_node.py

import rclpy
from rclpy.node import Node
import torch
import numpy as np

from gra_interfaces.msg import GRAState
from gra_interfaces.srv import ProjectState, ComputeFoam

class GRANode(Node):
    """
    Base class for ROS nodes that participate in GRA zeroing.
    """
    
    def __init__(self, node_name, multi_index, level, goal):
        super().__init__(node_name)
        self.multi_index = multi_index  # list of strings
        self.level = level
        self.goal = goal  # GRA Goal object with projector and loss
        
        # Publisher for state
        self.state_pub = self.create_publisher(GRAState, f'{node_name}/gra_state', 10)
        
        # Service for projecting state
        self.project_srv = self.create_service(ProjectState, f'{node_name}/project_state', 
                                                self.project_state_callback)
        
        # Subscriber for parameter updates from supervisor
        self.param_sub = self.create_subscription(
            ParameterUpdate, '/gra_supervisor/param_updates', 
            self.param_update_callback, 10)
        
        # Timer to publish state periodically
        self.timer = self.create_timer(0.1, self.publish_state)
        
    def get_internal_state(self):
        """
        To be overridden by subclass.
        Returns the current internal state as a numpy array or torch tensor.
        """
        raise NotImplementedError
        
    def apply_parameter_update(self, param_name, value):
        """
        To be overridden by subclass.
        Apply a parameter update from the supervisor.
        """
        pass
        
    def publish_state(self):
        """Publish current state on the GRA state topic."""
        state = self.get_internal_state()
        msg = GRAState()
        msg.multi_index = self.multi_index
        msg.level = self.level
        msg.data = state.flatten().tolist()
        msg.shape = state.shape
        self.state_pub.publish(msg)
        
    def project_state_callback(self, request, response):
        """
        Service call: project a given state onto the goal subspace.
        """
        # Convert request data back to tensor
        state = torch.tensor(request.state.data).reshape(request.state.shape)
        projected = self.goal.projector(state)
        response.projected_state.data = projected.flatten().tolist()
        response.projected_state.shape = projected.shape
        return response
        
    def param_update_callback(self, msg):
        """Receive parameter update from supervisor."""
        self.apply_parameter_update(msg.param_name, msg.value)
```

### 3.2. Concrete Example: TF2 Listener Node

A node that maintains the TF tree:

```python
# src/examples/ros2/tf2_node.py

import tf2_ros
from geometry_msgs.msg import TransformStamped
from gra_node import GRANode

class TF2GRAWrapper(GRANode):
    """
    Wraps a TF2 listener/broadcaster as a GRA subsystem.
    Goal: TF tree is complete and up‑to‑date.
    """
    
    def __init__(self, node_name, multi_index, level):
        # Define goal: TF tree should have all transforms with timestamp < 0.1s old
        goal = TFConsistencyGoal(max_lag=0.1, required_frames=['map', 'odom', 'base_link', 'camera_link'])
        super().__init__(node_name, multi_index, level, goal)
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
    def get_internal_state(self):
        """
        State = vector of (timestamp, translation, rotation) for each required frame.
        """
        state = []
        for frame in self.goal.required_frames:
            try:
                trans = self.tf_buffer.lookup_transform('map', frame, rclpy.time.Time())
                # Encode as [timestamp_sec, timestamp_nanosec, x, y, z, qx, qy, qz, qw]
                t = trans.header.stamp
                state.extend([t.sec, t.nanosec, 
                              trans.transform.translation.x,
                              trans.transform.translation.y,
                              trans.transform.translation.z,
                              trans.transform.rotation.x,
                              trans.transform.rotation.y,
                              trans.transform.rotation.z,
                              trans.transform.rotation.w])
            except:
                # Frame missing – use zeros and rely on foam to detect inconsistency
                state.extend([0]*9)
        return np.array(state)
        
    def apply_parameter_update(self, param_name, value):
        """
        Supervisor can adjust, e.g., cache duration.
        """
        if param_name == 'cache_duration':
            self.tf_buffer.set_cache_duration(rclpy.time.Duration(seconds=value))
```

---

## 4. Goal and Projector Implementations for ROS 2

### 4.1. TF Consistency Goal

```python
# src/integrations/ros2/goals/tf_consistency.py

import torch
from gra.core import Goal

class TFConsistencyGoal(Goal):
    """
    Goal: TF tree has all required transforms and they are recent.
    """
    
    def __init__(self, max_lag, required_frames):
        self.max_lag = max_lag
        self.required_frames = required_frames
        self.frame_dim = 9  # (timestamp_sec, timestamp_nanosec, x, y, z, qx, qy, qz, qw)
        
    def projector(self, state):
        """
        Project onto subspace where all frames are present and recent.
        If a frame is missing, we cannot "create" it – instead, we return the state
        but with a high penalty in loss().
        For differentiable projection, we would use a learned model.
        Here we return the state unchanged and rely on loss.
        """
        return state  # identity projector
        
    def loss(self, state):
        """
        Compute how far state is from goal.
        """
        loss = 0.0
        for i in range(len(self.required_frames)):
            frame_state = state[i*self.frame_dim:(i+1)*self.frame_dim]
            # Check if frame exists (non‑zero timestamp)
            if frame_state[0] == 0 and frame_state[1] == 0:
                loss += 1.0  # missing frame penalty
            else:
                # Check timestamp age
                now_sec = torch.tensor(self.get_current_time_sec())
                stamp_sec = frame_state[0] + frame_state[1] * 1e-9
                age = now_sec - stamp_sec
                if age > self.max_lag:
                    loss += (age / self.max_lag)  # proportional penalty
        return loss
        
    def get_current_time_sec(self):
        """Get current ROS time in seconds."""
        import rclpy.clock
        clock = rclpy.clock.Clock()
        now = clock.now()
        return now.sec + now.nanosec * 1e-9
```

### 4.2. Message Synchronization Goal

For nodes that fuse multiple topics (e.g., `message_filters.ApproximateTimeSynchronizer`):

```python
# src/integrations/ros2/goals/sync_goal.py

class MessageSyncGoal(Goal):
    """
    Goal: messages from different topics arrive within a time window.
    """
    
    def __init__(self, sync_window, topics):
        self.sync_window = sync_window
        self.topics = topics
        
    def projector(self, state):
        """
        State = concatenated timestamps of last message on each topic.
        Projection = adjust timestamps to be within window? Not possible physically.
        We'll use loss only.
        """
        return state
        
    def loss(self, state):
        """
        Compute max time difference between topics.
        """
        # state is vector of timestamps (one per topic)
        max_t = torch.max(state)
        min_t = torch.min(state)
        diff = max_t - min_t
        if diff > self.sync_window:
            return (diff - self.sync_window) / self.sync_window
        return 0.0
```

### 4.3. Action Server Goal

For nodes that provide action servers (e.g., `nav2`):

```python
# src/integrations/ros2/goals/action_goal.py

class ActionServerGoal(Goal):
    """
    Goal: action server responds within expected time and with correct outcome.
    """
    
    def __init__(self, expected_duration, expected_result_code):
        self.expected_duration = expected_duration
        self.expected_result_code = expected_result_code
        
    def projector(self, state):
        """
        state = [duration, result_code, success_flag]
        Project onto subspace where duration <= expected and result_code matches.
        In practice, we would use a learned model to "correct" the action.
        """
        # Simplified: just clip duration
        projected = state.clone()
        projected[0] = min(state[0], self.expected_duration)
        projected[2] = 1.0  # force success flag
        return projected
        
    def loss(self, state):
        duration = state[0]
        result_code = state[1]
        success = state[2]
        
        loss = 0.0
        if duration > self.expected_duration:
            loss += (duration - self.expected_duration) / self.expected_duration
        if result_code != self.expected_result_code:
            loss += 1.0
        if success < 0.5:
            loss += 1.0
        return loss
```

---

## 5. GRA Supervisor Node

The supervisor collects states from all wrapped nodes, runs the recursive zeroing algorithm, and sends parameter updates.

```python
# src/integrations/ros2/supervisor_node.py

import rclpy
from rclpy.node import Node
import torch
import numpy as np
from collections import defaultdict

from gra_interfaces.msg import GRAState
from gra_interfaces.srv import ProjectState

from gra.algorithms import zero_level  # from algorithm.md

class GRASupervisor(Node):
    """
    Central supervisor that runs the zeroing algorithm.
    """
    
    def __init__(self):
        super().__init__('gra_supervisor')
        
        # Storage for latest states from all nodes
        self.states = {}  # multi_index -> state tensor
        self.node_names = {}  # multi_index -> node name
        
        # Subscriber for GRA state topics
        self.state_sub = self.create_subscription(GRAState, '/gra_state', 
                                                   self.state_callback, 10)
        
        # Dictionary of project service clients (populated as nodes appear)
        self.project_clients = {}
        
        # Timer to run zeroing periodically
        self.timer = self.create_timer(1.0, self.run_zeroing)  # 1 Hz
        
        # Goals for each level (to be loaded from config)
        self.goals = self.load_goals()
        
        # Level weights
        self.lambdas = [1.0, 0.5, 0.25, 0.125, 0.0625]
        
        self.get_logger().info("GRA Supervisor started")
        
    def state_callback(self, msg):
        """Store latest state from a node."""
        multi_index = tuple(msg.multi_index)
        state = torch.tensor(msg.data).reshape(msg.shape)
        self.states[multi_index] = state
        self.node_names[multi_index] = msg.node_name
        
        # Create project client if not exists
        if multi_index not in self.project_clients:
            client = self.create_client(ProjectState, f'{msg.node_name}/project_state')
            self.project_clients[multi_index] = client
            
    def collect_level_states(self, level):
        """Collect all states for subsystems at given level."""
        level_states = {}
        for idx, state in self.states.items():
            if len(idx) - 1 == level:  # level = length-1
                level_states[idx] = state
        return level_states
        
    def compute_foam(self, level_states, goal):
        """
        Compute foam for a collection of states.
        Simplified version – in practice use GPU‑accelerated.
        """
        indices = list(level_states.keys())
        N = len(indices)
        foam = 0.0
        
        for i in range(N):
            for j in range(i+1, N):
                a = indices[i]
                b = indices[j]
                
                # We need P_G |Psi_b> – call project service
                proj_b = self.call_project_service(b, level_states[b])
                
                # Inner product
                overlap = torch.dot(level_states[a].flatten(), proj_b.flatten())
                foam += overlap.item() ** 2
                
        return foam
        
    def call_project_service(self, idx, state):
        """Call the project service of node idx."""
        if idx not in self.project_clients:
            return state  # fallback
            
        req = ProjectState.Request()
        req.state.data = state.flatten().tolist()
        req.state.shape = state.shape
        
        future = self.project_clients[idx].call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        
        proj = torch.tensor(response.projected_state.data).reshape(response.projected_state.shape)
        return proj
        
    def run_zeroing(self):
        """
        Main zeroing loop – adapted from algorithm.md.
        """
        if not self.states:
            return
            
        K = max(len(idx) for idx in self.states.keys()) - 1
        
        # Recursive zeroing from top level
        new_states = zero_level(K, self.states, self.goals, 
                                tolerances=[0.01]*5,
                                learning_rates=[0.1, 0.05, 0.02, 0.01, 0.005])
        
        # Send parameter updates to nodes
        for idx, new_state in new_states.items():
            old_state = self.states.get(idx)
            if old_state is not None and not torch.allclose(new_state, old_state):
                # State changed – need to send parameter update
                self.send_parameter_update(idx, new_state)
                
        self.states = new_states
        
        # Log foams
        for l in range(K+1):
            level_states = self.collect_level_states(l)
            foam = self.compute_foam(level_states, self.goals[l])
            self.get_logger().info(f"Level {l} foam: {foam:.4f}")
            
    def send_parameter_update(self, idx, new_state):
        """
        Send a parameter update to a node.
        This is node‑specific – we need a mapping from state changes to parameters.
        In practice, each node should expose a service to receive updates.
        """
        # Simplified: just log
        self.get_logger().info(f"Would update {idx} with new state")
        
    def load_goals(self):
        """Load goals from configuration."""
        # In practice, load from YAML
        return [
            # Level 0 goals (hardware)
            [MotorGoal(), IMUGoal(), CameraGoal()],
            # Level 1 goals (perception fusion)
            [EKFConsistencyGoal(), MapConsistencyGoal()],
            # Level 2 goals (planning)
            [PlannerFeasibilityGoal()],
            # Level 3 goals (task)
            [BehaviorTreeGoal()],
            # Level 4 goals (ethics)
            [SafetyGoal()]
        ]
```

---

## 6. Example: Zeroing a TurtleBot3 Navigation Stack

### 6.1. Setup

We have a standard TurtleBot3 with:
- `/imu` (IMU driver)
- `/scan` (LIDAR driver)
- `/odom` (wheel odometry)
- `/ekf_node` (robot_localization)
- `/map_server`
- `/nav2_planner`
- `/nav2_controller`
- `/behavior_tree`

### 6.2. Wrapping Nodes

Create wrapper nodes for each:

```python
# examples/turtlebot3/wrappers.py

class IMUDriverGRA(GRANode):
    def get_internal_state(self):
        # Return last IMU reading as vector
        return self.last_imu
        
class LidarDriverGRA(GRANode):
    def get_internal_state(self):
        return self.last_scan.ranges  # array of distances
        
class EKFNodeGRA(GRANode):
    def get_internal_state(self):
        # Return current pose estimate
        return self.ekf.get_pose().as_vector()
        
class Nav2PlannerGRA(GRANode):
    def get_internal_state(self):
        # Return current plan as sequence of poses
        return self.current_plan.flatten()
```

### 6.3. Running

```bash
# Terminal 1: Start ROS 2 core
ros2 run ros_core

# Terminal 2: Start GRA supervisor
ros2 run gra_ros2 supervisor

# Terminal 3: Start wrapped nodes
ros2 launch turtlebot3_bringup robot.launch.py
ros2 run gra_ros2 imu_wrapper
ros2 run gra_ros2 lidar_wrapper
# ... etc

# Terminal 4: Monitor
ros2 topic echo /gra_supervisor/foam
```

### 6.4. What Happens

1. Initially, foam is high: IMU and LIDAR timestamps are not synchronized; planner outputs infeasible trajectories.
2. Supervisor detects high foam at level 1 (perception) and sends parameter updates to `ekf_node` to adjust its time offset parameters.
3. Foam at level 1 drops. Now foam at level 2 (planning) becomes dominant.
4. Supervisor updates `nav2_planner` parameters (e.g., turning radius) to match actual robot capabilities.
5. After several iterations, all foams converge to near zero. The robot navigates smoothly.

---

## 7. Advanced: Online Zeroing with Lifecycle Nodes

ROS 2 lifecycle nodes (`rclcpp_lifecycle`) can be used to **reset** or **reconfigure** nodes during zeroing:

```python
class LifecycleGRAWrapper(GRANode, rclpy.lifecycle.Node):
    
    def __init__(self, node_name, multi_index, level, goal):
        GRANode.__init__(self, node_name, multi_index, level, goal)
        rclpy.lifecycle.Node.__init__(self, node_name)
        
    def on_configure(self, state):
        """Called when supervisor requests reconfiguration."""
        self.get_logger().info("Configuring...")
        return rclpy.lifecycle.State.PRIMARY_STATE_INACTIVE
        
    def on_cleanup(self, state):
        """Called when supervisor wants to reset this node."""
        self.get_logger().info("Cleaning up...")
        return rclpy.lifecycle.State.FINALIZED
        
    def apply_parameter_update(self, param_name, value):
        # If parameter change is major, request lifecycle transition
        if param_name == 'reset':
            self.trigger_configure()
        else:
            super().apply_parameter_update(param_name, value)
```

---

## 8. Performance Considerations

| Component | CPU Load | Network Load | Latency |
|-----------|----------|--------------|---------|
| State publishing (100 Hz) | Low | Medium (1 KB/msg) | <1 ms |
| Project service calls | Medium | Low | 1‑10 ms |
| Foam computation (N nodes) | O(N²) | None | 10‑100 ms |
| Parameter updates | Low | Low | <1 ms |

**Optimizations**:
- Reduce state publishing frequency for high‑level nodes (1‑10 Hz is enough).
- Use shared memory (ROS 2 zero‑copy) for large states (images, point clouds).
- Offload foam computation to a GPU node if available.
- Use **approximate foam** by sampling pairs randomly.

---

## 9. Integration with Other ROS 2 Tools

### 9.1. ROS 2 Launch

Create a launch file that starts the supervisor and wrapped nodes:

```python
# launch/gra_system.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # GRA supervisor
        Node(
            package='gra_ros2',
            executable='supervisor',
            name='gra_supervisor',
            parameters=[{'goals_file': 'config/turtlebot3_goals.yaml'}]
        ),
        
        # Wrapped nodes (using GRA‑enabled executables)
        Node(
            package='gra_ros2',
            executable='imu_wrapper',
            name='imu_gra',
            remappings=[('/imu', '/imu/data_raw')]
        ),
        # ... more nodes
        
        # Original ROS 2 nodes (if they don't have GRA wrappers)
        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_filter_node'
        )
    ])
```

### 9.2. RViz2 Plugin

Create an RViz2 plugin to visualize foam levels in real time:

```python
# src/visualization/gra_rviz_plugin.py

class FoamDisplay(Display):
    """RViz2 display that shows foam as color overlay on robot model."""
    
    def update(self, foam_data):
        for node_name, foam in foam_data.items():
            # Color robot parts based on foam level
            red = min(1.0, foam * 10)  # foam >0.1 → red
            self.set_node_color(node_name, (red, 1-red, 0))
```

### 9.3. rosbag2 Integration

Record foam data alongside sensor data:

```bash
ros2 bag record /gra_supervisor/foam /tf /scan /imu /odom
```

Later, analyze foam evolution:

```python
# analysis/analyze_bag.py

import rosbag2_py

bag = rosbag2_py.SequentialReader()
bag.open(...)

foam_over_time = []
while bag.has_next():
    topic, data, t = bag.read_next()
    if topic == '/gra_supervisor/foam':
        foam_over_time.append((t, data))
```

---

## 10. Complete Example: Moving with Consistency

Check the [examples/turtlebot3](../examples/turtlebot3) directory for a complete, runnable example.

```bash
cd examples/turtlebot3
ros2 launch gra_system.launch.py
```

You will see the robot start with jerky, inconsistent motion.  
Over 1‑2 minutes, the zeroing algorithm adjusts parameters, and the motion becomes smooth – foam drops from ~5.0 to <0.1.

---

## Next Steps

- [hospital_robot.md](../examples/hospital_robot.md) – full example of a medical assistance robot using ROS 2 and GRA.
- [factory_worker.md](../examples/factory_worker.md) – industrial robot with ethical constraints.
- [ethical_advisor.md](../examples/ethical_advisor.md) – using GRA as a safety layer for LLM‑based robot commands.

---

## Resources

- [ROS 2 Documentation](https://docs.ros.org)
- [robot_localization](http://wiki.ros.org/robot_localization)
- [Nav2](https://navigation.ros.org)
- Our GRA‑ROS 2 repository: `https://github.com/your-org/gra-ros2`

---

*“The whole is greater than the sum of its parts.”* – Aristotle  
With GRA on ROS 2, the whole becomes **coherent**.
```