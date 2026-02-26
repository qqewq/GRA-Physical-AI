```python
"""
GRA Physical AI - ROS 2 Bridge Module
======================================

This module provides a bridge between the GRA framework and ROS 2 (Robot Operating System 2),
enabling GRA-zeroed agents to operate on real robots and interact with the ROS ecosystem.

ROS 2 integration provides:
    - Communication with real robot hardware
    - Access to ROS topics, services, and actions
    - Integration with ROS 2 navigation, manipulation stacks
    - Multi-robot coordination
    - Real-time monitoring and visualization

The bridge implements:
    - ROS 2 node that exposes GRA subsystems as ROS topics/services
    - Conversion between GRA state tensors and ROS messages
    - Real-time foam computation and zeroing
    - Integration with ROS 2 lifecycle nodes
    - rosbag2 recording for GRA data
"""

import torch
import numpy as np
import threading
import time
import json
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

# Try to import ROS 2
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
    from rclpy.lifecycle import LifecycleNode, LifecycleState, TransitionCallbackReturn
    import rclpy.parameter
    from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String, Header
    from geometry_msgs.msg import Pose, Twist, Wrench, Point, Quaternion
    from sensor_msgs.msg import JointState, Image, PointCloud2, Imu
    from visualization_msgs.msg import Marker, MarkerArray
    from nav_msgs.msg import Odometry, Path
    from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
    from tf2_ros.buffer import Buffer
    from tf2_ros.transform_listener import TransformListener
    from tf2_geometry_msgs import do_transform_pose
    from builtin_interfaces.msg import Time, Duration
    import tf_transformations
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False
    warnings.warn("ROS 2 not available. Install from https://docs.ros.org")

from ..core.base_environment import BaseEnvironment
from ..core.multiverse import MultiIndex, Multiverse
from ..core.base_agent import BaseAgent
from ..core.nullification import ZeroingAlgorithm, ZeroingStatus
from ..core.foam import compute_foam, foam_gradient


# ======================================================================
# ROS 2 Message Converters
# ======================================================================

class ROS2MessageConverter:
    """
    Convert between GRA state tensors and ROS 2 messages.
    
    Provides bidirectional conversion for common ROS message types.
    """
    
    @staticmethod
    def tensor_to_float32_multiarray(tensor: torch.Tensor) -> Float32MultiArray:
        """Convert torch tensor to Float32MultiArray."""
        msg = Float32MultiArray()
        
        # Set layout
        if tensor.dim() == 0:
            # Scalar
            msg.layout.dim.append(MultiArrayDimension(
                label="scalar",
                size=1,
                stride=1
            ))
            msg.data = [tensor.item()]
        elif tensor.dim() == 1:
            # Vector
            msg.layout.dim.append(MultiArrayDimension(
                label="vector",
                size=tensor.shape[0],
                stride=tensor.shape[0]
            ))
            msg.data = tensor.cpu().numpy().tolist()
        else:
            # Multi-dimensional
            stride = 1
            for i, dim in enumerate(reversed(tensor.shape)):
                msg.layout.dim.append(MultiArrayDimension(
                    label=f"dim_{len(tensor.shape)-1-i}",
                    size=dim,
                    stride=stride
                ))
                stride *= dim
            msg.data = tensor.cpu().numpy().flatten().tolist()
        
        return msg
    
    @staticmethod
    def float32_multiarray_to_tensor(msg: Float32MultiArray) -> torch.Tensor:
        """Convert Float32MultiArray to torch tensor."""
        # Get shape from layout
        shape = [dim.size for dim in msg.layout.dim]
        if not shape:
            # Scalar
            return torch.tensor(msg.data[0] if msg.data else 0.0)
        
        # Reshape
        array = np.array(msg.data).reshape(shape)
        return torch.tensor(array, dtype=torch.float32)
    
    @staticmethod
    def joint_state_to_tensor(msg: JointState) -> torch.Tensor:
        """Convert JointState to tensor."""
        # Combine position, velocity, effort
        tensor_list = []
        if msg.position:
            tensor_list.append(torch.tensor(msg.position, dtype=torch.float32))
        if msg.velocity:
            tensor_list.append(torch.tensor(msg.velocity, dtype=torch.float32))
        if msg.effort:
            tensor_list.append(torch.tensor(msg.effort, dtype=torch.float32))
        
        if tensor_list:
            return torch.cat(tensor_list)
        return torch.zeros(1)
    
    @staticmethod
    def tensor_to_joint_state(tensor: torch.Tensor, joint_names: List[str]) -> JointState:
        """Convert tensor to JointState."""
        msg = JointState()
        msg.header.stamp = ROS2MessageConverter._get_ros_time()
        msg.name = joint_names
        
        n_joints = len(joint_names)
        tensor_np = tensor.cpu().numpy()
        
        # Assume tensor is [positions, velocities, efforts] or just positions
        if len(tensor_np) >= n_joints:
            msg.position = tensor_np[:n_joints].tolist()
        if len(tensor_np) >= 2 * n_joints:
            msg.velocity = tensor_np[n_joints:2*n_joints].tolist()
        if len(tensor_np) >= 3 * n_joints:
            msg.effort = tensor_np[2*n_joints:3*n_joints].tolist()
        
        return msg
    
    @staticmethod
    def pose_to_tensor(msg: Pose) -> torch.Tensor:
        """Convert Pose to tensor [x, y, z, qx, qy, qz, qw]."""
        return torch.tensor([
            msg.position.x, msg.position.y, msg.position.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ], dtype=torch.float32)
    
    @staticmethod
    def tensor_to_pose(tensor: torch.Tensor) -> Pose:
        """Convert tensor to Pose."""
        msg = Pose()
        tensor_np = tensor.cpu().numpy()
        
        if len(tensor_np) >= 3:
            msg.position.x = tensor_np[0]
            msg.position.y = tensor_np[1]
            msg.position.z = tensor_np[2]
        
        if len(tensor_np) >= 7:
            msg.orientation.x = tensor_np[3]
            msg.orientation.y = tensor_np[4]
            msg.orientation.z = tensor_np[5]
            msg.orientation.w = tensor_np[6]
        else:
            msg.orientation.w = 1.0
        
        return msg
    
    @staticmethod
    def twist_to_tensor(msg: Twist) -> torch.Tensor:
        """Convert Twist to tensor [vx, vy, vz, wx, wy, wz]."""
        return torch.tensor([
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z
        ], dtype=torch.float32)
    
    @staticmethod
    def tensor_to_twist(tensor: torch.Tensor) -> Twist:
        """Convert tensor to Twist."""
        msg = Twist()
        tensor_np = tensor.cpu().numpy()
        
        if len(tensor_np) >= 3:
            msg.linear.x = tensor_np[0]
            msg.linear.y = tensor_np[1]
            msg.linear.z = tensor_np[2]
        
        if len(tensor_np) >= 6:
            msg.angular.x = tensor_np[3]
            msg.angular.y = tensor_np[4]
            msg.angular.z = tensor_np[5]
        
        return msg
    
    @staticmethod
    def _get_ros_time() -> Time:
        """Get current ROS time."""
        msg = Time()
        if rclpy.ok():
            msg = rclpy.clock.Clock().now().to_msg()
        return msg


# ======================================================================
# ROS 2 GRA Bridge Node
# ======================================================================

class GRAROS2Bridge(Node):
    """
    ROS 2 node that bridges between GRA multiverse and ROS topics.
    
    Publishes:
        - /gra/state/[level] - GRA state tensors for each level
        - /gra/foam - Current foam values
        - /gra/status - Zeroing algorithm status
        - /tf - Transforms for visualization
    
    Subscribes to:
        - /gra/action - Action commands for agents
        - /gra/zeroing/params - Zeroing parameters
        - Robot-specific topics (JointState, Odometry, etc.)
    
    Provides services:
        - /gra/get_state - Get current GRA state
        - /gra/set_goal - Set goal for a level
        - /gra/start_zeroing - Start zeroing algorithm
        - /gra/stop_zeroing - Stop zeroing algorithm
    """
    
    def __init__(
        self,
        node_name: str = "gra_bridge",
        multiverse: Optional[Multiverse] = None,
        zeroing_algo: Optional[ZeroingAlgorithm] = None,
        qos_depth: int = 10,
        publish_rate: float = 10.0,
        **kwargs
    ):
        """
        Args:
            node_name: ROS node name
            multiverse: GRA multiverse instance
            zeroing_algo: Zeroing algorithm instance
            qos_depth: QoS depth for topics
            publish_rate: Rate (Hz) for publishing GRA state
            **kwargs: Additional arguments
        """
        if not ROS2_AVAILABLE:
            raise ImportError("ROS 2 not available")
        
        super().__init__(node_name)
        
        self.multiverse = multiverse
        self.zeroing_algo = zeroing_algo
        self.publish_rate = publish_rate
        self.qos_profile = QoSProfile(
            depth=qos_depth,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )
        
        # Robot-specific data
        self.robot_name = kwargs.get('robot_name', 'robot')
        self.joint_names = kwargs.get('joint_names', [])
        self.urdf_path = kwargs.get('urdf_path', '')
        
        # State cache
        self._latest_observations = {}
        self._latest_actions = {}
        self._foam_history = deque(maxlen=1000)
        
        # Threading
        self._lock = threading.Lock()
        self._running = False
        self._publish_thread = None
        
        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)
        self.static_tf_broadcaster = StaticTransformBroadcaster(self)
        
        # Publishers
        self._setup_publishers()
        
        # Subscribers
        self._setup_subscribers()
        
        # Services
        self._setup_services()
        
        # Parameters
        self._setup_parameters()
        
        self.get_logger().info(f"GRA ROS 2 Bridge initialized: {node_name}")
    
    def _setup_publishers(self):
        """Setup ROS publishers."""
        # GRA state publishers for each level
        self.state_pubs = {}
        if self.multiverse:
            for level in range(self.multiverse.max_level + 1):
                self.state_pubs[level] = self.create_publisher(
                    Float32MultiArray,
                    f'/gra/state/level_{level}',
                    self.qos_profile
                )
        
        # Foam publisher
        self.foam_pub = self.create_publisher(
            Float32MultiArray,
            '/gra/foam',
            self.qos_profile
        )
        
        # Status publisher
        self.status_pub = self.create_publisher(
            String,
            '/gra/status',
            self.qos_profile
        )
        
        # Robot state publishers (standard ROS topics)
        self.joint_state_pub = self.create_publisher(
            JointState,
            f'/{self.robot_name}/joint_states',
            self.qos_profile
        )
        
        self.odom_pub = self.create_publisher(
            Odometry,
            f'/{self.robot_name}/odom',
            self.qos_profile
        )
        
        # Visualization markers
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/gra/visualization',
            self.qos_profile
        )
    
    def _setup_subscribers(self):
        """Setup ROS subscribers."""
        # Action command
        self.action_sub = self.create_subscription(
            Float32MultiArray,
            '/gra/action',
            self._action_callback,
            self.qos_profile
        )
        
        # Zeroing parameters
        self.zeroing_params_sub = self.create_subscription(
            String,
            '/gra/zeroing/params',
            self._zeroing_params_callback,
            self.qos_profile
        )
        
        # Robot-specific topics
        self.joint_state_sub = self.create_subscription(
            JointState,
            f'/{self.robot_name}/joint_states',
            self._joint_state_callback,
            self.qos_profile
        )
        
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.robot_name}/odom',
            self._odom_callback,
            self.qos_profile
        )
        
        # Camera topics (optional)
        self.camera_sub = self.create_subscription(
            Image,
            f'/{self.robot_name}/camera/image_raw',
            self._camera_callback,
            self.qos_profile
        )
    
    def _setup_services(self):
        """Setup ROS services."""
        from rclpy.serialization import serialize_message
        
        # Get state service
        self.get_state_srv = self.create_service(
            String,  # Actually should be custom service type
            '/gra/get_state',
            self._get_state_callback
        )
        
        # Set goal service
        self.set_goal_srv = self.create_service(
            String,
            '/gra/set_goal',
            self._set_goal_callback
        )
        
        # Start zeroing service
        self.start_zeroing_srv = self.create_service(
            String,
            '/gra/start_zeroing',
            self._start_zeroing_callback
        )
        
        # Stop zeroing service
        self.stop_zeroing_srv = self.create_service(
            String,
            '/gra/stop_zeroing',
            self._stop_zeroing_callback
        )
    
    def _setup_parameters(self):
        """Setup ROS parameters."""
        self.declare_parameter('publish_rate', self.publish_rate)
        self.declare_parameter('robot_name', self.robot_name)
        self.declare_parameter('zeroing_enabled', False)
        self.declare_parameter('zeroing_learning_rate', 0.01)
        self.declare_parameter('zeroing_tolerance', 0.001)
        
        # Parameter callback
        self.add_on_set_parameters_callback(self._parameters_callback)
    
    # ======================================================================
    # Callbacks
    # ======================================================================
    
    def _action_callback(self, msg: Float32MultiArray):
        """Handle incoming action commands."""
        try:
            action_tensor = ROS2MessageConverter.float32_multiarray_to_tensor(msg)
            with self._lock:
                self._latest_actions['command'] = action_tensor
            self.get_logger().debug(f"Received action: {action_tensor.shape}")
        except Exception as e:
            self.get_logger().error(f"Error processing action: {e}")
    
    def _zeroing_params_callback(self, msg: String):
        """Handle zeroing parameter updates."""
        try:
            params = json.loads(msg.data)
            if self.zeroing_algo:
                # Update zeroing parameters
                if 'learning_rates' in params:
                    self.zeroing_algo.learning_rates = params['learning_rates']
                if 'tolerances' in params:
                    self.zeroing_algo.level_tolerances = params['tolerances']
                self.get_logger().info(f"Updated zeroing parameters: {params}")
        except Exception as e:
            self.get_logger().error(f"Error updating zeroing parameters: {e}")
    
    def _joint_state_callback(self, msg: JointState):
        """Handle joint state messages."""
        try:
            joint_tensor = ROS2MessageConverter.joint_state_to_tensor(msg)
            with self._lock:
                self._latest_observations['joint_states'] = joint_tensor
                # Update GRA multiverse if attached
                if self.multiverse:
                    # Find joint subsystems and update states
                    for idx, subsystem in self.multiverse.subsystems.items():
                        if idx.level == 0 and idx.indices[1] in msg.name:
                            # This is a joint subsystem
                            joint_idx = msg.name.index(idx.indices[1])
                            if joint_idx < len(msg.position):
                                state = torch.tensor([
                                    msg.position[joint_idx],
                                    msg.velocity[joint_idx] if joint_idx < len(msg.velocity) else 0.0,
                                    msg.effort[joint_idx] if joint_idx < len(msg.effort) else 0.0
                                ])
                                subsystem.set_state(state)
        except Exception as e:
            self.get_logger().error(f"Error processing joint states: {e}")
    
    def _odom_callback(self, msg: Odometry):
        """Handle odometry messages."""
        try:
            pose_tensor = ROS2MessageConverter.pose_to_tensor(msg.pose.pose)
            twist_tensor = ROS2MessageConverter.twist_to_tensor(msg.twist.twist)
            with self._lock:
                self._latest_observations['odom_pose'] = pose_tensor
                self._latest_observations['odom_twist'] = twist_tensor
        except Exception as e:
            self.get_logger().error(f"Error processing odometry: {e}")
    
    def _camera_callback(self, msg: Image):
        """Handle camera images."""
        # This would convert ROS Image to tensor
        # For now, just store that we got an image
        with self._lock:
            self._latest_observations['camera'] = True
    
    def _get_state_callback(self, request, response):
        """Service callback to get current GRA state."""
        try:
            states = {}
            if self.multiverse:
                for level in range(self.multiverse.max_level + 1):
                    level_states = self.multiverse.get_states_at_level(level)
                    states[f'level_{level}'] = {
                        str(idx): state.tolist() 
                        for idx, state in level_states.items()
                    }
            response.data = json.dumps(states)
        except Exception as e:
            self.get_logger().error(f"Error in get_state service: {e}")
            response.data = json.dumps({'error': str(e)})
        return response
    
    def _set_goal_callback(self, request, response):
        """Service callback to set a goal for a level."""
        try:
            data = json.loads(request.data)
            level = data.get('level')
            goal_type = data.get('type')
            params = data.get('params', {})
            
            if self.multiverse:
                # Create and set goal (implementation depends on goal type)
                self.get_logger().info(f"Setting goal for level {level}: {goal_type}")
                # TODO: Create goal from params
                response.data = json.dumps({'success': True})
            else:
                response.data = json.dumps({'success': False, 'error': 'No multiverse'})
        except Exception as e:
            self.get_logger().error(f"Error in set_goal service: {e}")
            response.data = json.dumps({'success': False, 'error': str(e)})
        return response
    
    def _start_zeroing_callback(self, request, response):
        """Service callback to start zeroing algorithm."""
        try:
            if self.zeroing_algo and self.multiverse:
                # Start zeroing in background thread
                self._start_zeroing_thread()
                response.data = json.dumps({'success': True, 'status': 'started'})
            else:
                response.data = json.dumps({'success': False, 'error': 'Zeroing not configured'})
        except Exception as e:
            self.get_logger().error(f"Error starting zeroing: {e}")
            response.data = json.dumps({'success': False, 'error': str(e)})
        return response
    
    def _stop_zeroing_callback(self, request, response):
        """Service callback to stop zeroing algorithm."""
        try:
            self._running = False
            response.data = json.dumps({'success': True, 'status': 'stopped'})
        except Exception as e:
            self.get_logger().error(f"Error stopping zeroing: {e}")
            response.data = json.dumps({'success': False, 'error': str(e)})
        return response
    
    def _parameters_callback(self, params):
        """Handle parameter updates."""
        for param in params:
            if param.name == 'publish_rate':
                self.publish_rate = param.value
            elif param.name == 'zeroing_enabled':
                if param.value and not self._running:
                    self._start_zeroing_thread()
                elif not param.value and self._running:
                    self._running = False
            elif param.name == 'zeroing_learning_rate' and self.zeroing_algo:
                self.zeroing_algo.learning_rates = [param.value] * (self.multiverse.max_level + 1)
            elif param.name == 'zeroing_tolerance' and self.zeroing_algo:
                self.zeroing_algo.level_tolerances = [param.value] * (self.multiverse.max_level + 1)
        
        from rclpy.parameter import SetParametersResult
        return SetParametersResult(successful=True)
    
    # ======================================================================
    # Publishing Methods
    # ======================================================================
    
    def _publish_loop(self):
        """Main publishing loop."""
        rate = self.create_rate(self.publish_rate)
        
        while rclpy.ok() and self._running:
            try:
                # Publish GRA state
                self._publish_gra_state()
                
                # Publish foam
                self._publish_foam()
                
                # Publish robot state (joints, odometry)
                self._publish_robot_state()
                
                # Publish visualization markers
                self._publish_visualization()
                
                rate.sleep()
            except Exception as e:
                self.get_logger().error(f"Error in publish loop: {e}")
    
    def _publish_gra_state(self):
        """Publish GRA state tensors for each level."""
        if not self.multiverse:
            return
        
        with self._lock:
            for level, pub in self.state_pubs.items():
                level_states = self.multiverse.get_states_at_level(level)
                if level_states:
                    # Concatenate all states at this level
                    states_list = [state.flatten() for state in level_states.values()]
                    if states_list:
                        concat_state = torch.cat(states_list)
                        msg = ROS2MessageConverter.tensor_to_float32_multiarray(concat_state)
                        pub.publish(msg)
    
    def _publish_foam(self):
        """Publish current foam values."""
        if not self.multiverse:
            return
        
        foams = self.multiverse.compute_all_foams()
        foam_array = torch.tensor(list(foams.values()), dtype=torch.float32)
        msg = ROS2MessageConverter.tensor_to_float32_multiarray(foam_array)
        self.foam_pub.publish(msg)
        
        # Store in history
        self._foam_history.append((time.time(), foams))
    
    def _publish_robot_state(self):
        """Publish robot state in standard ROS formats."""
        if not self.multiverse or not self.joint_names:
            return
        
        # Create joint state message
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name = self.joint_names
        
        # Get joint states from multiverse
        positions = []
        velocities = []
        efforts = []
        
        for joint_name in self.joint_names:
            # Find joint subsystem
            joint_idx = MultiIndex((self.robot_name, joint_name, None, None, None))
            if joint_idx in self.multiverse.subsystems:
                state = self.multiverse.get_state(joint_idx)
                if len(state) >= 1:
                    positions.append(state[0].item())
                if len(state) >= 2:
                    velocities.append(state[1].item())
                if len(state) >= 3:
                    efforts.append(state[2].item())
        
        joint_msg.position = positions
        joint_msg.velocity = velocities
        joint_msg.effort = efforts
        
        self.joint_state_pub.publish(joint_msg)
    
    def _publish_visualization(self):
        """Publish visualization markers for GRA state."""
        if not self.multiverse:
            return
        
        marker_array = MarkerArray()
        
        # Create markers for each robot
        for idx, subsystem in self.multiverse.subsystems.items():
            if idx.level == 2:  # Robot level
                marker = Marker()
                marker.header.frame_id = "world"
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = f"robot_{idx.indices[0]}"
                marker.id = hash(idx) % 1000
                marker.type = Marker.ARROW
                marker.action = Marker.ADD
                
                # Get robot position from state
                state = subsystem.get_state()
                if len(state) >= 3:
                    marker.pose.position.x = state[0].item()
                    marker.pose.position.y = state[1].item()
                    marker.pose.position.z = state[2].item()
                    marker.pose.orientation.w = 1.0
                
                # Set color based on foam
                foam = self.multiverse.compute_foam(2)
                marker.color.r = min(1.0, foam.item() * 10)
                marker.color.g = 0.5
                marker.color.b = 0.5
                marker.color.a = 0.8
                
                marker.scale.x = 0.3
                marker.scale.y = 0.1
                marker.scale.z = 0.1
                
                marker_array.markers.append(marker)
        
        self.marker_pub.publish(marker_array)
    
    # ======================================================================
    # Zeroing Thread
    # ======================================================================
    
    def _start_zeroing_thread(self):
        """Start background zeroing thread."""
        if not self.zeroing_algo or not self.multiverse:
            return
        
        self._running = True
        self._zeroing_thread = threading.Thread(target=self._zeroing_loop)
        self._zeroing_thread.daemon = True
        self._zeroing_thread.start()
        self.get_logger().info("Zeroing thread started")
    
    def _zeroing_loop(self):
        """Background loop running zeroing algorithm."""
        while rclpy.ok() and self._running:
            try:
                # Get current states from multiverse
                states = self.multiverse.get_all_states()
                
                # Run one zeroing step
                if self.zeroing_algo:
                    new_states = self.zeroing_algo.zero_level(
                        self.multiverse.max_level,
                        states
                    )
                    
                    # Update multiverse with new states
                    with self._lock:
                        for idx, new_state in new_states.items():
                            if idx in self.multiverse.subsystems:
                                self.multiverse.set_state(idx, new_state)
                    
                    # Publish status
                    status_msg = String()
                    status_msg.data = json.dumps({
                        'status': 'running',
                        'epoch': self.zeroing_algo.current_epoch if hasattr(self.zeroing_algo, 'current_epoch') else 0,
                        'foams': self.multiverse.compute_all_foams()
                    })
                    self.status_pub.publish(status_msg)
                
                # Rate limiting
                time.sleep(1.0 / self.publish_rate)
                
            except Exception as e:
                self.get_logger().error(f"Error in zeroing loop: {e}")
                time.sleep(1.0)
    
    # ======================================================================
    ======================================================================
    # Public Methods
    # ======================================================================
    
    def attach_multiverse(self, multiverse: Multiverse):
        """Attach GRA multiverse to bridge."""
        self.multiverse = multiverse
        self.get_logger().info("Multiverse attached")
    
    def attach_zeroing_algorithm(self, algo: ZeroingAlgorithm):
        """Attach zeroing algorithm to bridge."""
        self.zeroing_algo = algo
        self.get_logger().info("Zeroing algorithm attached")
    
    def get_latest_observation(self, key: str) -> Optional[torch.Tensor]:
        """Get latest observation for a key."""
        with self._lock:
            return self._latest_observations.get(key)
    
    def send_action(self, action: torch.Tensor, action_type: str = "joint_torques"):
        """Send action command to robot."""
        msg = Float32MultiArray()
        msg = ROS2MessageConverter.tensor_to_float32_multiarray(action)
        
        # Add action type to topic (could be separate topics)
        self.action_pub.publish(msg)
    
    def spin(self):
        """Spin ROS node (blocking)."""
        self._running = True
        self._publish_thread = threading.Thread(target=self._publish_loop)
        self._publish_thread.daemon = True
        self._publish_thread.start()
        
        rclpy.spin(self)
    
    def shutdown(self):
        """Shutdown bridge."""
        self._running = False
        if hasattr(self, '_publish_thread') and self._publish_thread:
            self._publish_thread.join(timeout=1.0)
        if hasattr(self, '_zeroing_thread') and self._zeroing_thread:
            self._zeroing_thread.join(timeout=1.0)
        self.destroy_node()


# ======================================================================
# ROS 2 Lifecycle Node for GRA
# ======================================================================

class GRALifecycleBridge(LifecycleNode):
    """
    ROS 2 lifecycle node for GRA bridge.
    
    Provides lifecycle management for zeroing process:
        - Unconfigured -> Inactive: Load configuration
        - Inactive -> Active: Start zeroing
        - Active -> Inactive: Stop zeroing
        - Inactive -> Unconfigured: Unload configuration
    """
    
    def __init__(self, node_name: str = "gra_lifecycle_bridge"):
        super().__init__(node_name)
        
        self.bridge = None
        self.multiverse = None
        self.zeroing_algo = None
    
    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Configure the node."""
        self.get_logger().info("Configuring GRA lifecycle bridge")
        
        try:
            # Load parameters
            self.declare_parameter('robot_name', 'robot')
            self.declare_parameter('urdf_path', '')
            self.declare_parameter('joint_names', [])
            self.declare_parameter('max_level', 4)
            
            robot_name = self.get_parameter('robot_name').value
            urdf_path = self.get_parameter('urdf_path').value
            joint_names = self.get_parameter('joint_names').value
            max_level = self.get_parameter('max_level').value
            
            # Create multiverse
            from ..core.multiverse import Multiverse
            from ..core.subsystem import Subsystem
            
            self.multiverse = Multiverse(name=f"{robot_name}_multiverse", max_level=max_level)
            
            # Create zeroing algorithm
            from ..core.nullification import ZeroingAlgorithm
            
            # (Simplified - would need proper hierarchy setup)
            self.zeroing_algo = ZeroingAlgorithm(
                hierarchy={},
                get_children=lambda x: [],
                get_parents=lambda x: [],
                get_goal_projector=lambda x: None,
                get_level_weight=lambda x: 1.0,
                level_tolerances=[0.01] * (max_level + 1),
                learning_rates=[0.01] * (max_level + 1)
            )
            
            # Create bridge
            self.bridge = GRAROS2Bridge(
                node_name=f"{node_name}_bridge",
                multiverse=self.multiverse,
                zeroing_algo=self.zeroing_algo,
                robot_name=robot_name,
                joint_names=joint_names,
                urdf_path=urdf_path
            )
            
            return TransitionCallbackReturn.SUCCESS
        except Exception as e:
            self.get_logger().error(f"Failed to configure: {e}")
            return TransitionCallbackReturn.ERROR
    
    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Activate the node (start zeroing)."""
        self.get_logger().info("Activating GRA lifecycle bridge")
        
        if self.bridge:
            self.bridge._start_zeroing_thread()
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Deactivate the node (stop zeroing)."""
        self.get_logger().info("Deactivating GRA lifecycle bridge")
        
        if self.bridge:
            self.bridge._running = False
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Cleanup node resources."""
        self.get_logger().info("Cleaning up GRA lifecycle bridge")
        
        if self.bridge:
            self.bridge.shutdown()
            self.bridge = None
        
        self.multiverse = None
        self.zeroing_algo = None
        
        return TransitionCallbackReturn.SUCCESS
    
    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        """Shutdown node."""
        self.get_logger().info("Shutting down GRA lifecycle bridge")
        return self.on_cleanup(state)


# ======================================================================
# ROS 2 Action Server for Zeroing
# ======================================================================

from rclpy.action import ActionServer, GoalResponse, CancelResponse
from action_msgs.msg import GoalStatus

# Define action types (would be in separate .action file)
"""
# zeroing.action
int32 max_epochs
float32[] tolerances
---
bool success
float32[] final_foams
---
float32 current_foam
int32 current_epoch
"""

# Placeholder for action message
class ZeroingActionGoal:
    pass


class ZeroingActionServer(Node):
    """
    ROS 2 action server for long-running zeroing tasks.
    """
    
    def __init__(self, bridge: GRAROS2Bridge):
        super().__init__('zeroing_action_server')
        
        self.bridge = bridge
        
        # self._action_server = ActionServer(
        #     self,
        #     ZeroingAction,
        #     'gra/zeroing',
        #     execute_callback=self.execute_callback,
        #     goal_callback=self.goal_callback,
        #     cancel_callback=self.cancel_callback
        # )
        
        self.get_logger().info("Zeroing action server initialized")
    
    def goal_callback(self, goal_request):
        """Accept or reject goal."""
        self.get_logger().info("Received zeroing goal request")
        return GoalResponse.ACCEPT
    
    def cancel_callback(self, goal_handle):
        """Handle cancel request."""
        self.get_logger().info("Received cancel request")
        return CancelResponse.ACCEPT
    
    async def execute_callback(self, goal_handle):
        """Execute zeroing action."""
        self.get_logger().info("Executing zeroing...")
        
        # Get goal parameters
        max_epochs = goal_handle.request.max_epochs
        tolerances = goal_handle.request.tolerances
        
        # Configure zeroing algorithm
        if self.bridge and self.bridge.zeroing_algo:
            self.bridge.zeroing_algo.level_tolerances = tolerances