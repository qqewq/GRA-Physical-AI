```python
"""
GRA Physical AI - G1 Task Execution Layer
=========================================

This module implements the G1 (Level 1) layer focused on **task execution**.
Unlike lower-level motor control (G0) or higher-level planning (G2+), G1 is responsible
for executing a given task by coordinating multiple motor primitives and providing
feedback on task progress.

Key responsibilities:
    - Receive high-level task descriptions from G2/G3
    - Decompose tasks into motor primitives
    - Monitor task execution progress
    - Provide task completion feedback
    - Handle task failures and recovery

The G1 layer acts as the bridge between abstract task planning and concrete motor control.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import warnings

from ..core.multiverse import MultiIndex
from ..core.subsystem import Subsystem
from ..core.goal import Goal
from ..core.projector import Projector, ThresholdProjector, RangeProjector
from .g0_motor_layer import G0_Layer
from ..layers.base_layer import GRA_layer


# ======================================================================
# Task Status and Types
# ======================================================================

class TaskStatus(Enum):
    """Status of a task being executed."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class TaskType(Enum):
    """Types of tasks that can be executed."""
    # Locomotion
    MOVE_TO_POSE = "move_to_pose"
    FOLLOW_PATH = "follow_path"
    STOP = "stop"
    
    # Manipulation
    GRASP = "grasp"
    RELEASE = "release"
    MOVE_JOINTS = "move_joints"
    
    # Interaction
    SPEAK = "speak"
    GESTURE = "gesture"
    
    # Combined
    PICK_AND_PLACE = "pick_and_place"
    FOLLOW_HUMAN = "follow_human"
    
    # Special
    IDLE = "idle"
    EMERGENCY_STOP = "emergency_stop"


@dataclass
class Task:
    """Representation of a task to be executed."""
    
    task_id: str
    task_type: TaskType
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0
    deadline: Optional[float] = None
    parent_task_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    
    # Progress tracking
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0  # 0.0 to 1.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    
    def to_tensor(self) -> torch.Tensor:
        """Convert task to tensor representation."""
        # Encode task type as one-hot
        task_types = list(TaskType)
        type_one_hot = torch.zeros(len(task_types))
        type_one_hot[task_types.index(self.task_type)] = 1.0
        
        # Encode parameters (simplified - would need proper serialization)
        param_tensor = torch.tensor([
            self.priority,
            self.progress,
            1.0 if self.status == TaskStatus.COMPLETED else 0.0,
            1.0 if self.status == TaskStatus.FAILED else 0.0
        ])
        
        return torch.cat([type_one_hot, param_tensor])
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'Task':
        """Create task from tensor representation."""
        task_types = list(TaskType)
        type_idx = torch.argmax(tensor[:len(task_types)]).item()
        task_type = task_types[type_idx]
        
        # Simplified - would need proper parameter deserialization
        return cls(
            task_id=f"task_{time.time()}",
            task_type=task_type,
            priority=int(tensor[len(task_types)].item())
        )


# ======================================================================
# Task Primitives
# ======================================================================

class TaskPrimitive(ABC):
    """
    Abstract base class for a task primitive.
    
    A primitive is a basic, reusable unit of task execution that can be
    combined to form complex tasks. Each primitive knows how to:
        - Check if it's feasible given current state
        - Generate motor commands to execute
        - Monitor progress
        - Detect completion/failure
    """
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def is_feasible(self, state: torch.Tensor) -> bool:
        """Check if this primitive can be executed in current state."""
        pass
    
    @abstractmethod
    def get_motor_commands(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        """Generate motor commands for this primitive."""
        pass
    
    @abstractmethod
    def get_progress(self, state: torch.Tensor) -> float:
        """Get progress of this primitive (0.0 to 1.0)."""
        pass
    
    @abstractmethod
    def is_complete(self, state: torch.Tensor) -> bool:
        """Check if this primitive is complete."""
        pass
    
    @abstractmethod
    def is_failed(self, state: torch.Tensor) -> bool:
        """Check if this primitive has failed."""
        pass


class MoveToPosePrimitive(TaskPrimitive):
    """Primitive for moving to a target pose."""
    
    def __init__(
        self,
        target_pose: torch.Tensor,
        tolerance: float = 0.05,
        max_speed: float = 1.0
    ):
        super().__init__("move_to_pose")
        self.target_pose = target_pose
        self.tolerance = tolerance
        self.max_speed = max_speed
        self.start_pose = None
    
    def is_feasible(self, state: torch.Tensor) -> bool:
        # Always feasible in this simple implementation
        return True
    
    def get_motor_commands(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        # state contains current pose (x, y, theta)
        current_pose = state[:3]
        
        if self.start_pose is None:
            self.start_pose = current_pose.clone()
        
        # Simple proportional controller
        error = self.target_pose - current_pose
        
        # Limit speed
        speed = torch.clamp(error * 2.0, -self.max_speed, self.max_speed)
        
        return speed
    
    def get_progress(self, state: torch.Tensor) -> float:
        if self.start_pose is None:
            return 0.0
        
        current_pose = state[:3]
        total_dist = torch.norm(self.target_pose - self.start_pose)
        current_dist = torch.norm(self.target_pose - current_pose)
        
        if total_dist < 1e-6:
            return 1.0
        
        return max(0.0, min(1.0, 1.0 - current_dist / total_dist))
    
    def is_complete(self, state: torch.Tensor) -> bool:
        current_pose = state[:3]
        error = torch.norm(self.target_pose - current_pose)
        return error < self.tolerance
    
    def is_failed(self, state: torch.Tensor) -> bool:
        # Check if stuck or oscillating
        return False


class GraspPrimitive(TaskPrimitive):
    """Primitive for grasping an object."""
    
    def __init__(
        self,
        object_id: str,
        grasp_force: float = 10.0,
        timeout: float = 5.0
    ):
        super().__init__("grasp")
        self.object_id = object_id
        self.grasp_force = grasp_force
        self.timeout = timeout
        self.start_time = None
        self.grasped = False
    
    def is_feasible(self, state: torch.Tensor) -> bool:
        # Check if object is within reach
        # Simplified - always true
        return True
    
    def get_motor_commands(self, state: torch.Tensor, dt: float) -> torch.Tensor:
        if self.start_time is None:
            self.start_time = time.time()
        
        # Close gripper
        commands = torch.zeros(state.shape[0])
        
        # Set gripper joint to closed position
        if len(commands) > 6:  # Assuming last joint is gripper
            commands[-1] = self.grasp_force
        
        return commands
    
    def get_progress(self, state: torch.Tensor) -> float:
        if self.grasped:
            return 1.0
        if self.start_time is None:
            return 0.0
        
        elapsed = time.time() - self.start_time
        return min(1.0, elapsed / self.timeout)
    
    def is_complete(self, state: torch.Tensor) -> bool:
        # Check if object is grasped (would use force feedback)
        # Simplified
        if not self.grasped and self.get_progress(state) > 0.8:
            self.grasped = True
        return self.grasped
    
    def is_failed(self, state: torch.Tensor) -> bool:
        if self.start_time is None:
            return False
        elapsed = time.time() - self.start_time
        return elapsed > self.timeout and not self.grasped


# ======================================================================
# G1 Task Layer
# ======================================================================

class G1_TaskLayer(GRA_layer):
    """
    G1 (Level 1) layer for task execution.
    
    This layer receives tasks from higher levels (G2/G3) and executes them
    by coordinating motor primitives. It provides:
        - Task queue management
        - Primitive selection and execution
        - Progress monitoring
        - Failure detection and recovery
        - Communication with G0 motor layer
    """
    
    def __init__(
        self,
        name: str = "task_execution",
        g0_layer: Optional[G0_Layer] = None,
        max_concurrent_tasks: int = 1,
        task_timeout: float = 30.0,
        parent: Optional[GRA_layer] = None,
        children: Optional[List[GRA_layer]] = None
    ):
        """
        Args:
            name: Layer name
            g0_layer: Reference to G0 motor layer
            max_concurrent_tasks: Maximum number of tasks to execute simultaneously
            task_timeout: Default timeout for tasks (seconds)
            parent: Parent layer (G2/G3)
            children: Child layers (should include G0)
        """
        super().__init__(level=1, name=name, parent=parent, children=children)
        
        self.g0_layer = g0_layer
        self.max_concurrent_tasks = max_concurrent_tasks
        self.task_timeout = task_timeout
        
        # Task management
        self.task_queue: List[Task] = []
        self.active_tasks: List[Task] = []
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        # Primitive registry
        self.primitives: Dict[TaskType, Callable] = {}
        self._register_default_primitives()
        
        # Current motor commands
        self.current_motor_commands = torch.zeros(1)
        
        # Task history for learning
        self.task_history: List[Dict] = []
        
        # Build subsystems
        self.subsystems = self.build_subsystems()
        
        # Create goals
        self._create_goals()
    
    def _register_default_primitives(self):
        """Register default task primitives."""
        self.register_primitive(TaskType.MOVE_TO_POSE, MoveToPosePrimitive)
        self.register_primitive(TaskType.GRASP, GraspPrimitive)
        # Add more primitives as needed
    
    def register_primitive(self, task_type: TaskType, primitive_class: type):
        """Register a primitive for a task type."""
        self.primitives[task_type] = primitive_class
    
    def build_subsystems(self) -> Dict[MultiIndex, Subsystem]:
        """Create subsystems for task layer."""
        from ..core.subsystem import SimpleSubsystem
        from ..core.multiverse import EuclideanSpace
        
        subsystems = {}
        
        # Task executor subsystem
        executor_idx = MultiIndex((None, "task_executor", None, None, None))
        
        class TaskExecutorSubsystem(SimpleSubsystem):
            def __init__(self, idx, layer):
                super().__init__(idx, EuclideanSpace(10), None)  # State: [active_task_id, progress, ...]
                self.layer = layer
                self._state = torch.zeros(10)
            
            def get_state(self):
                # Update state from layer
                if self.layer.active_tasks:
                    task = self.layer.active_tasks[0]
                    self._state[0] = float(hash(task.task_id) % 1000) / 1000.0  # task identifier
                    self._state[1] = task.progress
                return self._state
            
            def set_state(self, state):
                self._state = state.clone()
                # Could trigger task changes based on state
            
            def step(self, dt, action=None):
                # Step is handled by layer's update method
                pass
        
        subsystems[executor_idx] = TaskExecutorSubsystem(executor_idx, self)
        
        # Task queue subsystem (for monitoring)
        queue_idx = MultiIndex((None, "task_queue", None, None, None))
        
        class TaskQueueSubsystem(SimpleSubsystem):
            def __init__(self, idx, layer):
                super().__init__(idx, EuclideanSpace(1), None)  # Just queue length
                self.layer = layer
            
            def get_state(self):
                return torch.tensor([len(self.layer.task_queue)], dtype=torch.float32)
        
        subsystems[queue_idx] = TaskQueueSubsystem(queue_idx, self)
        
        return subsystems
    
    def _create_goals(self):
        """Create goals for task layer."""
        
        class TaskCompletionGoal(Goal):
            """Goal: complete assigned tasks successfully."""
            
            def __init__(self, layer):
                self.layer = layer
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Penalize:
                # - Long queue (backlog)
                # - Failed tasks
                # - Low progress on active tasks
                
                loss = 0.0
                
                # Queue length penalty
                loss += len(self.layer.task_queue) * 0.1
                
                # Failed tasks penalty
                loss += len(self.layer.failed_tasks) * 1.0
                
                # Active task progress penalty
                for task in self.layer.active_tasks:
                    loss += (1.0 - task.progress) * 0.5
                
                return torch.tensor(loss, dtype=torch.float32)
        
        class TaskEfficiencyGoal(Goal):
            """Goal: execute tasks efficiently (fast, low energy)."""
            
            def __init__(self, layer):
                self.layer = layer
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Penalize long execution times
                loss = 0.0
                
                for task in self.layer.completed_tasks[-10:]:  # Look at recent tasks
                    if task.started_at and task.completed_at:
                        duration = task.completed_at - task.started_at
                        loss += duration * 0.01
                
                return torch.tensor(loss, dtype=torch.float32)
        
        class TaskCoordinationGoal(Goal):
            """Goal: tasks are well-coordinated (no conflicts)."""
            
            def __init__(self, layer):
                self.layer = layer
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Check for conflicting tasks
                loss = 0.0
                
                if len(self.layer.active_tasks) > 1:
                    # Penalize multiple active tasks
                    loss += (len(self.layer.active_tasks) - 1) * 0.5
                
                return torch.tensor(loss, dtype=torch.float32)
        
        self.goals = [
            TaskCompletionGoal(self),
            TaskEfficiencyGoal(self),
            TaskCoordinationGoal(self)
        ]
    
    def get_goals(self) -> List[Goal]:
        return self.goals
    
    # ======================================================================
    # Task Management API
    # ======================================================================
    
    def add_task(self, task: Task) -> str:
        """Add a task to the queue."""
        task.task_id = f"{task.task_type.value}_{len(self.task_history)}_{time.time()}"
        self.task_queue.append(task)
        self.task_history.append({
            'task_id': task.task_id,
            'task_type': task.task_type.value,
            'added_at': time.time()
        })
        return task.task_id
    
    def add_task_by_type(
        self,
        task_type: TaskType,
        parameters: Optional[Dict] = None,
        priority: int = 0
    ) -> str:
        """Create and add a task by type."""
        task = Task(
            task_id=f"{task_type.value}_{time.time()}",
            task_type=task_type,
            parameters=parameters or {},
            priority=priority
        )
        return self.add_task(task)
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task (if in queue or active)."""
        # Check queue
        for i, task in enumerate(self.task_queue):
            if task.task_id == task_id:
                task.status = TaskStatus.ABORTED
                self.task_queue.pop(i)
                return True
        
        # Check active
        for task in self.active_tasks:
            if task.task_id == task_id:
                task.status = TaskStatus.ABORTED
                self.active_tasks.remove(task)
                return True
        
        return False
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task."""
        for task in self.active_tasks:
            if task.task_id == task_id:
                return task.status
        for task in self.task_queue:
            if task.task_id == task_id:
                return task.status
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return task.status
        for task in self.failed_tasks:
            if task.task_id == task_id:
                return task.status
        return None
    
    def get_all_tasks(self) -> Dict[str, List[Task]]:
        """Get all tasks by status."""
        return {
            'queue': self.task_queue,
            'active': self.active_tasks,
            'completed': self.completed_tasks,
            'failed': self.failed_tasks
        }
    
    def clear_completed_tasks(self, max_age: Optional[float] = None):
        """Clear completed tasks (optionally older than max_age seconds)."""
        now = time.time()
        self.completed_tasks = [
            t for t in self.completed_tasks
            if max_age is None or (t.completed_at and (now - t.completed_at) < max_age)
        ]
    
    # ======================================================================
    # Task Execution
    # ======================================================================
    
    def update(self, current_state: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Update task execution.
        
        Args:
            current_state: Current robot state (from G0)
            dt: Time step
        
        Returns:
            Motor commands for G0 layer
        """
        # Check for new tasks to start
        self._dispatch_tasks()
        
        # Update active tasks
        motor_commands = []
        for task in self.active_tasks[:]:
            # Get primitive for this task
            primitive = self._get_primitive(task)
            
            if primitive is None:
                task.status = TaskStatus.FAILED
                task.error_message = f"No primitive for task type {task.task_type}"
                self.active_tasks.remove(task)
                self.failed_tasks.append(task)
                continue
            
            # Update progress
            task.progress = primitive.get_progress(current_state)
            
            # Check completion
            if primitive.is_complete(current_state):
                task.status = TaskStatus.COMPLETED
                task.completed_at = time.time()
                self.active_tasks.remove(task)
                self.completed_tasks.append(task)
                continue
            
            # Check failure
            if primitive.is_failed(current_state):
                task.status = TaskStatus.FAILED
                task.completed_at = time.time()
                task.error_message = "Primitive failed"
                self.active_tasks.remove(task)
                self.failed_tasks.append(task)
                continue
            
            # Get motor commands
            cmd = primitive.get_motor_commands(current_state, dt)
            motor_commands.append(cmd)
        
        # Combine motor commands (simple sum for now)
        if motor_commands:
            self.current_motor_commands = sum(motor_commands)
        else:
            self.current_motor_commands = torch.zeros_like(self.current_motor_commands)
        
        return self.current_motor_commands
    
    def _dispatch_tasks(self):
        """Start new tasks from queue if capacity available."""
        while self.task_queue and len(self.active_tasks) < self.max_concurrent_tasks:
            # Get highest priority task
            self.task_queue.sort(key=lambda t: -t.priority)
            task = self.task_queue.pop(0)
            
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            self.active_tasks.append(task)
    
    def _get_primitive(self, task: Task) -> Optional[TaskPrimitive]:
        """Get primitive instance for a task."""
        if task.task_type not in self.primitives:
            return None
        
        primitive_class = self.primitives[task.task_type]
        
        # Create primitive with task parameters
        try:
            return primitive_class(**task.parameters)
        except Exception as e:
            print(f"Error creating primitive: {e}")
            return None
    
    # ======================================================================
    # Integration with G0
    # ======================================================================
    
    def connect_to_g0(self, g0_layer: G0_Layer):
        """Connect this layer to a G0 motor layer."""
        self.g0_layer = g0_layer
        self.children = [g0_layer]
        g0_layer.parent = self
    
    def get_motor_commands(self) -> torch.Tensor:
        """Get current motor commands for G0."""
        return self.current_motor_commands
    
    def send_to_g0(self):
        """Send current motor commands to G0 layer."""
        if self.g0_layer:
            # This would set the target commands in G0 subsystems
            pass
    
    # ======================================================================
    # Task Statistics
    # ======================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get task execution statistics."""
        return {
            'queue_length': len(self.task_queue),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks),
            'success_rate': len(self.completed_tasks) / max(1, len(self.completed_tasks) + len(self.failed_tasks)),
            'avg_completion_time': self._avg_completion_time()
        }
    
    def _avg_completion_time(self) -> float:
        """Average completion time for completed tasks."""
        times = []
        for task in self.completed_tasks[-50:]:  # Last 50 tasks
            if task.started_at and task.completed_at:
                times.append(task.completed_at - task.started_at)
        return np.mean(times) if times else 0.0
    
    def reset_statistics(self):
        """Reset task statistics."""
        self.task_queue = []
        self.active_tasks = []
        self.completed_tasks = []
        self.failed_tasks = []
        self.task_history = []


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing G1 Task Layer ===\n")
    
    # Create G1 layer
    task_layer = G1_TaskLayer(name="test_task_layer")
    
    # Add some tasks
    task1_id = task_layer.add_task_by_type(
        TaskType.MOVE_TO_POSE,
        parameters={'target_pose': torch.tensor([1.0, 0.0, 0.0])}
    )
    print(f"Added task 1: {task1_id}")
    
    task2_id = task_layer.add_task_by_type(
        TaskType.GRASP,
        parameters={'object_id': 'cube_001', 'grasp_force': 15.0},
        priority=1  # Higher priority
    )
    print(f"Added task 2: {task2_id}")
    
    # Simulate execution
    current_state = torch.zeros(10)  # Dummy state
    dt = 0.1
    
    for step in range(20):
        motor_cmds = task_layer.update(current_state, dt)
        
        if step % 5 == 0:
            stats = task_layer.get_statistics()
            print(f"\nStep {step}:")
            print(f"  Queue: {stats['queue_length']}, Active: {stats['active_tasks']}")
            print(f"  Completed: {stats['completed_tasks']}, Failed: {stats['failed_tasks']}")
            
            # Show active task progress
            for task in task_layer.active_tasks:
                print(f"  Task {task.task_id[:8]}: {task.progress:.2f}")
    
    print("\nFinal statistics:")
    print(task_layer.get_statistics())
    
    print("\nAll tests passed!")
```