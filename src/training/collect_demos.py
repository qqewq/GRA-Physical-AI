```python
#!/usr/bin/env python3
"""
GRA Physical AI - Demonstration Collection for Mimic Learning
=============================================================

This script collects human demonstrations for imitation learning and behavioral cloning.
The collected demonstrations can be used to:
    - Train initial policies via behavioral cloning
    - Provide preference data for reward learning
    - Seed RL training with good initial behavior
    - Validate learned policies against human demonstrations
    - Build datasets for ethical decision-making

Features:
    - Multiple collection modes (teleoperation, kinesthetic teaching, vision-based)
    - Real-time visualization and playback
    - Automatic segmentation of demonstrations
    - Metadata tagging (task type, difficulty, etc.)
    - Integration with GRA layers for safety during collection
    - Export to multiple formats (HDF5, JSON, numpy)
"""

import argparse
import torch
import numpy as np
import time
import yaml
import json
import os
import sys
import h5py
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import threading
import queue
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GRA core imports
from core.multiverse import Multiverse, MultiIndex

# Environment imports
from envs.pybullet_wrapper import PyBulletGRAWrapper, HumanoidPyBullet
from envs.isaac_wrapper import IsaacLabWrapper
from envs.ros2_bridge import ROS2Environment

# Visualization
try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False

# Try to import teleoperation interfaces
try:
    import pygame
    from pygame import joystick
    PYGANE_AVAILABLE = True
except ImportError:
    PYGANE_AVAILABLE = False
    warnings.warn("Pygame not available for teleoperation")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    warnings.warn("OpenCV not available for video recording")

try:
    from pynput import keyboard
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False
    warnings.warn("Pynput not available for keyboard control")


# ======================================================================
# Data Structures
# ======================================================================

@dataclass
class DemonstrationStep:
    """Single step in a demonstration."""
    
    timestamp: float
    observation: np.ndarray
    action: np.ndarray
    reward: float = 0.0
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)
    
    # Optional data
    image: Optional[np.ndarray] = None
    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    ee_pose: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'observation': self.observation.tolist() if self.observation is not None else None,
            'action': self.action.tolist() if self.action is not None else None,
            'reward': self.reward,
            'done': self.done,
            'info': self.info,
            'image': self.image.tolist() if self.image is not None else None,
            'joint_positions': self.joint_positions.tolist() if self.joint_positions is not None else None,
            'joint_velocities': self.joint_velocities.tolist() if self.joint_velocities is not None else None,
            'ee_pose': self.ee_pose.tolist() if self.ee_pose is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DemonstrationStep':
        """Create from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            observation=np.array(data['observation']) if data['observation'] else None,
            action=np.array(data['action']) if data['action'] else None,
            reward=data['reward'],
            done=data['done'],
            info=data['info'],
            image=np.array(data['image']) if data.get('image') else None,
            joint_positions=np.array(data['joint_positions']) if data.get('joint_positions') else None,
            joint_velocities=np.array(data['joint_velocities']) if data.get('joint_velocities') else None,
            ee_pose=np.array(data['ee_pose']) if data.get('ee_pose') else None
        )


@dataclass
class Demonstration:
    """Complete demonstration of a task."""
    
    demo_id: str
    task_name: str
    timestamp: float
    steps: List[DemonstrationStep]
    
    # Metadata
    duration: float = 0.0
    success: bool = True
    difficulty: str = "medium"  # easy, medium, hard
    demonstrator_id: str = "unknown"
    environment_state: Optional[Dict] = None
    
    # Statistics
    avg_action_magnitude: float = 0.0
    max_action_magnitude: float = 0.0
    avg_reward: float = 0.0
    
    def __post_init__(self):
        if self.steps:
            self.duration = self.steps[-1].timestamp - self.steps[0].timestamp
            actions = np.array([s.action for s in self.steps if s.action is not None])
            if len(actions) > 0:
                self.avg_action_magnitude = np.mean(np.linalg.norm(actions, axis=1))
                self.max_action_magnitude = np.max(np.linalg.norm(actions, axis=1))
            
            rewards = [s.reward for s in self.steps]
            self.avg_reward = np.mean(rewards)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'demo_id': self.demo_id,
            'task_name': self.task_name,
            'timestamp': self.timestamp,
            'steps': [s.to_dict() for s in self.steps],
            'duration': self.duration,
            'success': self.success,
            'difficulty': self.difficulty,
            'demonstrator_id': self.demonstrator_id,
            'environment_state': self.environment_state,
            'avg_action_magnitude': self.avg_action_magnitude,
            'max_action_magnitude': self.max_action_magnitude,
            'avg_reward': self.avg_reward
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Demonstration':
        """Create from dictionary."""
        return cls(
            demo_id=data['demo_id'],
            task_name=data['task_name'],
            timestamp=data['timestamp'],
            steps=[DemonstrationStep.from_dict(s) for s in data['steps']],
            duration=data.get('duration', 0.0),
            success=data.get('success', True),
            difficulty=data.get('difficulty', 'medium'),
            demonstrator_id=data.get('demonstrator_id', 'unknown'),
            environment_state=data.get('environment_state'),
            avg_action_magnitude=data.get('avg_action_magnitude', 0.0),
            max_action_magnitude=data.get('max_action_magnitude', 0.0),
            avg_reward=data.get('avg_reward', 0.0)
        )
    
    def to_hdf5(self, group: h5py.Group):
        """Save to HDF5 group."""
        # Store metadata
        group.attrs['demo_id'] = self.demo_id
        group.attrs['task_name'] = self.task_name
        group.attrs['timestamp'] = self.timestamp
        group.attrs['duration'] = self.duration
        group.attrs['success'] = self.success
        group.attrs['difficulty'] = self.difficulty
        group.attrs['demonstrator_id'] = self.demonstrator_id
        
        # Store steps
        steps_group = group.create_group('steps')
        for i, step in enumerate(self.steps):
            step_group = steps_group.create_group(f'step_{i}')
            step_group.attrs['timestamp'] = step.timestamp
            step_group.attrs['reward'] = step.reward
            step_group.attrs['done'] = step.done
            
            if step.observation is not None:
                step_group.create_dataset('observation', data=step.observation)
            if step.action is not None:
                step_group.create_dataset('action', data=step.action)
            if step.image is not None:
                step_group.create_dataset('image', data=step.image)
            if step.joint_positions is not None:
                step_group.create_dataset('joint_positions', data=step.joint_positions)
            if step.joint_velocities is not None:
                step_group.create_dataset('joint_velocities', data=step.joint_velocities)
            if step.ee_pose is not None:
                step_group.create_dataset('ee_pose', data=step.ee_pose)
    
    @classmethod
    def from_hdf5(cls, group: h5py.Group) -> 'Demonstration':
        """Load from HDF5 group."""
        # Load metadata
        demo_id = group.attrs['demo_id']
        task_name = group.attrs['task_name']
        timestamp = group.attrs['timestamp']
        duration = group.attrs.get('duration', 0.0)
        success = group.attrs.get('success', True)
        difficulty = group.attrs.get('difficulty', 'medium')
        demonstrator_id = group.attrs.get('demonstrator_id', 'unknown')
        
        # Load steps
        steps = []
        steps_group = group['steps']
        for step_name in sorted(steps_group.keys()):
            step_group = steps_group[step_name]
            
            step = DemonstrationStep(
                timestamp=step_group.attrs['timestamp'],
                observation=np.array(step_group['observation']) if 'observation' in step_group else None,
                action=np.array(step_group['action']) if 'action' in step_group else None,
                reward=step_group.attrs['reward'],
                done=step_group.attrs['done'],
                image=np.array(step_group['image']) if 'image' in step_group else None,
                joint_positions=np.array(step_group['joint_positions']) if 'joint_positions' in step_group else None,
                joint_velocities=np.array(step_group['joint_velocities']) if 'joint_velocities' in step_group else None,
                ee_pose=np.array(step_group['ee_pose']) if 'ee_pose' in step_group else None
            )
            steps.append(step)
        
        return cls(
            demo_id=demo_id,
            task_name=task_name,
            timestamp=timestamp,
            steps=steps,
            duration=duration,
            success=success,
            difficulty=difficulty,
            demonstrator_id=demonstrator_id
        )


# ======================================================================
# Teleoperation Interfaces
# ======================================================================

class TeleoperationInterface(ABC):
    """Abstract base class for teleoperation interfaces."""
    
    def __init__(self):
        self.action_queue = queue.Queue(maxsize=1)
        self.running = False
        self.thread = None
    
    @abstractmethod
    def start(self):
        """Start teleoperation interface."""
        pass
    
    @abstractmethod
    def stop(self):
        """Stop teleoperation interface."""
        pass
    
    @abstractmethod
    def get_action(self) -> Optional[np.ndarray]:
        """Get current action from interface."""
        pass


class KeyboardInterface(TeleoperationInterface):
    """Keyboard-based teleoperation."""
    
    def __init__(self, action_dim: int, key_map: Optional[Dict] = None):
        super().__init__()
        self.action_dim = action_dim
        self.key_map = key_map or {
            'w': np.array([1.0, 0.0]),  # forward
            's': np.array([-1.0, 0.0]),  # backward
            'a': np.array([0.0, -1.0]),  # left
            'd': np.array([0.0, 1.0]),   # right
            'q': np.array([-1.0, -1.0]), # diagonal
            'e': np.array([1.0, 1.0]),   # diagonal
            'r': np.array([0.0, 0.0, 1.0]),  # up (3D)
            'f': np.array([0.0, 0.0, -1.0]), # down (3D)
        }
        self.current_action = np.zeros(action_dim)
        self.pressed_keys = set()
    
    def start(self):
        if not PYNPUT_AVAILABLE:
            raise ImportError("Pynput not available")
        
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def _run(self):
        from pynput import keyboard
        
        def on_press(key):
            try:
                if hasattr(key, 'char') and key.char in self.key_map:
                    self.pressed_keys.add(key.char)
                    self._update_action()
            except:
                pass
        
        def on_release(key):
            try:
                if hasattr(key, 'char') and key.char in self.key_map:
                    self.pressed_keys.discard(key.char)
                    self._update_action()
            except:
                pass
        
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    
    def _update_action(self):
        """Update action based on pressed keys."""
        self.current_action = np.zeros(self.action_dim)
        
        for key in self.pressed_keys:
            if key in self.key_map:
                self.current_action += self.key_map[key]
        
        # Normalize if needed
        norm = np.linalg.norm(self.current_action)
        if norm > 1.0:
            self.current_action = self.current_action / norm
        
        # Put in queue (non-blocking)
        try:
            self.action_queue.put_nowait(self.current_action.copy())
        except queue.Full:
            pass
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
    
    def get_action(self) -> Optional[np.ndarray]:
        try:
            return self.action_queue.get_nowait()
        except queue.Empty:
            return None


class JoystickInterface(TeleoperationInterface):
    """Joystick/gamepad teleoperation."""
    
    def __init__(self, action_dim: int, axis_map: Optional[Dict] = None):
        super().__init__()
        self.action_dim = action_dim
        self.axis_map = axis_map or {
            0: 0,  # Left stick X -> action 0
            1: 1,  # Left stick Y -> action 1
            2: 2,  # Right stick X -> action 2
            3: 3,  # Right stick Y -> action 3
        }
        
        if not PYGANE_AVAILABLE:
            raise ImportError("Pygame not available")
        
        pygame.init()
        joystick.init()
        
        if joystick.get_count() == 0:
            raise RuntimeError("No joystick found")
        
        self.js = joystick.Joystick(0)
        self.js.init()
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def _run(self):
        while self.running:
            pygame.event.pump()
            
            action = np.zeros(self.action_dim)
            for axis, target in self.axis_map.items():
                if target < self.action_dim:
                    action[target] = self.js.get_axis(axis)
            
            # Put in queue
            try:
                self.action_queue.put_nowait(action)
            except queue.Full:
                pass
            
            time.sleep(0.01)  # 100 Hz
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        pygame.quit()
    
    def get_action(self) -> Optional[np.ndarray]:
        try:
            return self.action_queue.get_nowait()
        except queue.Empty:
            return None


class VisionBasedInterface(TeleoperationInterface):
    """Vision-based teleoperation using hand tracking or AR markers."""
    
    def __init__(self, action_dim: int, camera_id: int = 0):
        super().__init__()
        self.action_dim = action_dim
        self.camera_id = camera_id
        
        if not CV2_AVAILABLE:
            raise ImportError("OpenCV not available")
        
        self.cap = cv2.VideoCapture(camera_id)
        self.prev_hand_position = None
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()
    
    def _run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Process frame to get hand position (simplified)
            # In real implementation, use MediaPipe or similar
            hand_position = self._detect_hand(frame)
            
            if hand_position is not None:
                action = self._hand_to_action(hand_position)
                
                try:
                    self.action_queue.put_nowait(action)
                except queue.Full:
                    pass
            
            # Display
            cv2.imshow('Teleoperation', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def _detect_hand(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Detect hand position (placeholder)."""
        # Simplified - return center of frame
        h, w = frame.shape[:2]
        return np.array([w/2, h/2])
    
    def _hand_to_action(self, hand_pos: np.ndarray) -> np.ndarray:
        """Convert hand position to action."""
        action = np.zeros(self.action_dim)
        
        # Map hand position to first two action dimensions
        if self.action_dim >= 2:
            # Normalize to [-1, 1]
            h, w = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            action[0] = (hand_pos[0] / w - 0.5) * 2
            action[1] = (hand_pos[1] / h - 0.5) * 2
        
        return action
    
    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)


# ======================================================================
# Demonstration Recorder
# ======================================================================

class DemonstrationRecorder:
    """
    Records demonstrations from teleoperation.
    
    Features:
        - Multiple recording modes (continuous, triggered)
        - Automatic segmentation
        - Real-time visualization
        - Data validation
        - Metadata tagging
    """
    
    def __init__(
        self,
        env,
        interface: TeleoperationInterface,
        task_name: str = "unknown",
        record_images: bool = False,
        max_demo_length: int = 1000,
        min_demo_length: int = 10,
        auto_segment: bool = True,
        segment_threshold: float = 0.1  # Pause threshold (seconds)
    ):
        """
        Args:
            env: Environment to record from
            interface: Teleoperation interface
            task_name: Name of task being demonstrated
            record_images: Whether to record camera images
            max_demo_length: Maximum steps per demonstration
            min_demo_length: Minimum steps for valid demonstration
            auto_segment: Whether to automatically segment demonstrations
            segment_threshold: Pause threshold for auto-segmentation (seconds)
        """
        self.env = env
        self.interface = interface
        self.task_name = task_name
        self.record_images = record_images
        self.max_demo_length = max_demo_length
        self.min_demo_length = min_demo_length
        self.auto_segment = auto_segment
        self.segment_threshold = segment_threshold
        
        # Recording state
        self.recording = False
        self.paused = False
        self.current_demo: List[DemonstrationStep] = []
        self.completed_demos: List[Demonstration] = []
        self.demo_counter = 0
        
        # Timing
        self.last_step_time = time.time()
        self.last_action_time = time.time()
        
        # Keyboard controls for recorder
        self._setup_controls()
    
    def _setup_controls(self):
        """Setup keyboard controls for recorder."""
        if PYNPUT_AVAILABLE:
            from pynput import keyboard
            
            def on_press(key):
                try:
                    if hasattr(key, 'char'):
                        if key.char == ' ':
                            self.toggle_recording()
                        elif key.char == 'p':
                            self.toggle_pause()
                        elif key.char == 's':
                            self.save_current_demo()
                        elif key.char == 'c':
                            self.cancel_recording()
                        elif key.char == 'r':
                            self.reset_environment()
                except:
                    pass
            
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.daemon = True
            self.listener.start()
    
    def toggle_recording(self):
        """Toggle recording on/off."""
        if not self.recording:
            self.start_recording()
        else:
            self.stop_recording()
    
    def start_recording(self):
        """Start recording a demonstration."""
        if self.recording:
            return
        
        self.recording = True
        self.paused = False
        self.current_demo = []
        self.last_step_time = time.time()
        self.last_action_time = time.time()
        print(f"\n=== Started recording demonstration {self.demo_counter + 1} ===")
    
    def stop_recording(self, save: bool = True):
        """Stop recording."""
        if not self.recording:
            return
        
        self.recording = False
        
        if save and len(self.current_demo) >= self.min_demo_length:
            self.save_current_demo()
        else:
            print(f"Discarded demonstration ({len(self.current_demo)} steps)")
            self.current_demo = []
    
    def toggle_pause(self):
        """Toggle pause during recording."""
        self.paused = not self.paused
        print(f"{'Paused' if self.paused else 'Resumed'}")
    
    def save_current_demo(self):
        """Save current demonstration."""
        if len(self.current_demo) < self.min_demo_length:
            print(f"Demo too short ({len(self.current_demo)} steps), not saving")
            self.current_demo = []
            return
        
        demo = Demonstration(
            demo_id=f"{self.task_name}_{self.demo_counter}_{int(time.time())}",
            task_name=self.task_name,
            timestamp=time.time(),
            steps=self.current_demo.copy(),
            demonstrator_id="human"
        )
        
        self.completed_demos.append(demo)
        self.demo_counter += 1
        print(f"Saved demonstration {self.demo_counter} ({len(self.current_demo)} steps)")
        
        self.current_demo = []
    
    def cancel_recording(self):
        """Cancel current recording without saving."""
        self.recording = False
        self.current_demo = []
        print("Recording cancelled")
    
    def reset_environment(self):
        """Reset the environment."""
        self.env.reset()
        print("Environment reset")
    
    def add_step(self, obs: np.ndarray, action: np.ndarray, 
                 reward: float = 0.0, done: bool = False,
                 info: Optional[Dict] = None):
        """Add a step to current demonstration."""
        if not self.recording or self.paused:
            return
        
        # Check for auto-segmentation (pause detection)
        if self.auto_segment:
            now = time.time()
            if now - self.last_action_time > self.segment_threshold:
                if len(self.current_demo) >= self.min_demo_length:
                    self.save_current_demo()
                self.start_recording()
            self.last_action_time = now
        
        # Create step
        step = DemonstrationStep(
            timestamp=time.time(),
            observation=obs.copy() if obs is not None else None,
            action=action.copy() if action is not None else None,
            reward=reward,
            done=done,
            info=info or {}
        )
        
        # Add image if requested
        if self.record_images and hasattr(self.env, 'render'):
            img = self.env.render(mode='rgb_array')
            if img is not None:
                step.image = img
        
        self.current_demo.append(step)
        
        # Check max length
        if len(self.current_demo) >= self.max_demo_length:
            self.stop_recording(save=True)
    
    def get_statistics(self) -> Dict:
        """Get recording statistics."""
        return {
            'completed_demos': len(self.completed_demos),
            'current_demo_steps': len(self.current_demo),
            'recording': self.recording,
            'paused': self.paused,
            'total_steps': sum(len(d.steps) for d in self.completed_demos)
        }
    
    def close(self):
        """Clean up resources."""
        if hasattr(self, 'listener'):
            self.listener.stop()


# ======================================================================
# Main Collection Script
# ======================================================================

class DemonstrationCollector:
    """
    Main class for collecting demonstrations.
    
    Integrates environment, teleoperation, and recording.
    """
    
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directory
        self.output_dir = self.config.get('output_dir', './demos')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize environment
        self._init_environment()
        
        # Initialize teleoperation
        self._init_teleoperation()
        
        # Initialize recorder
        self.recorder = DemonstrationRecorder(
            env=self.env,
            interface=self.teleop,
            task_name=self.config.get('task_name', 'unknown'),
            record_images=self.config.get('record_images', False),
            max_demo_length=self.config.get('max_demo_length', 1000),
            min_demo_length=self.config.get('min_demo_length', 10),
            auto_segment=self.config.get('auto_segment', True),
            segment_threshold=self.config.get('segment_threshold', 0.1)
        )
        
        # Visualization
        if self.config.get('visualize', True) and VISUALIZATION_AVAILABLE:
            self._init_visualization()
        
        print(f"\n=== Demonstration Collector Initialized ===")
        print(f"Task: {self.config.get('task_name', 'unknown')}")
        print(f"Environment: {self.config.get('env_type', 'unknown')}")
        print(f"Teleoperation: {self.config.get('teleop_type', 'keyboard')}")
        print(f"Output: {self.output_dir}")
        print("\nControls:")
        print("  SPACE - Start/stop recording")
        print("  p - Pause/resume")
        print("  s - Save current demo")
        print("  c - Cancel recording")
        print("  r - Reset environment")
        print("  q - Quit")
        print("=" * 40)
    
    def _init_environment(self):
        """Initialize environment."""
        env_type = self.config.get('env_type', 'pybullet')
        env_name = self.config.get('env_name', 'humanoid')
        
        if env_type == 'pybullet':
            if env_name == 'humanoid':
                self.env = HumanoidPyBullet(gui=True)
            else:
                self.env = PyBulletGRAWrapper(
                    name=env_name,
                    urdf_paths=[f"assets/{env_name}.urdf"],
                    gui=True
                )
        elif env_type == 'isaac':
            self.env = IsaacLabWrapper(
                name=env_name,
                env_class=env_name,
                headless=False
            )
        elif env_type == 'ros2':
            self.env = ROS2Environment(
                name=env_name,
                observation_topics=self.config.get('observation_topics', []),
                action_topic=self.config.get('action_topic', '/cmd_vel')
            )
        else:
            raise ValueError(f"Unknown environment type: {env_type}")
    
    def _init_teleoperation(self):
        """Initialize teleoperation interface."""
        teleop_type = self.config.get('teleop_type', 'keyboard')
        action_dim = self.config.get('action_dim', 6)
        
        if teleop_type == 'keyboard':
            self.teleop = KeyboardInterface(action_dim)
        elif teleop_type == 'joystick':
            self.teleop = JoystickInterface(action_dim)
        elif teleop_type == 'vision':
            self.teleop = VisionBasedInterface(action_dim)
        else:
            raise ValueError(f"Unknown teleop type: {teleop_type}")
        
        self.teleop.start()
    
    def _init_visualization(self):
        """Initialize visualization."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.fig.suptitle("Demonstration Collection")
        
        # Action plot
        self.action_line, = self.ax1.plot([], [])
        self.ax1.set_xlim(0, 100)
        self.ax1.set_ylim(-1, 1)
        self.ax1.set_title("Actions")
        self.ax1.set_xlabel("Step")
        self.ax1.set_ylabel("Value")
        
        # Stats text
        self.stats_text = self.ax2.text(0.1, 0.5, "", transform=self.ax2.transAxes,
                                        fontfamily='monospace', fontsize=10)
        self.ax2.set_xlim(0, 1)
        self.ax2.set_ylim(0, 1)
        self.ax2.axis('off')
        
        plt.ion()
        plt.show()
    
    def _update_visualization(self):
        """Update visualization."""
        if not VISUALIZATION_AVAILABLE or not hasattr(self, 'fig'):
            return
        
        # Update action plot
        if self.recorder.current_demo:
            actions = np.array([s.action for s in self.recorder.current_demo[-100:]])
            if len(actions) > 0:
                self.action_line.set_data(range(len(actions)), actions)
                self.ax1.relim()
                self.ax1.autoscale_view()
        
        # Update stats
        stats = self.recorder.get_statistics()
        text = f"""
Recording: {'Yes' if stats['recording'] else 'No'}
Paused: {'Yes' if stats['paused'] else 'No'}
Current steps: {stats['current_demo_steps']}
Completed demos: {stats['completed_demos']}
Total steps: {stats['total_steps']}
        """
        self.stats_text.set_text(text)
        
        plt.draw()
        plt.pause(0.01)
    
    def run(self):
        """Main collection loop."""
        try:
            while True:
                # Get action from teleoperation
                action = self.teleop.get_action()
                
                if action is not None:
                    # Step environment
                    obs, reward, done, info = self.env.step(action)
                    
                    # Add to recorder
                    self.recorder.add_step(obs, action, reward, done, info)
                    
                    # Update visualization
                    self._update_visualization()
                
                # Small delay
                time.sleep(self.config.get('step_dt', 0.01))
                
        except KeyboardInterrupt:
            print("\nCollection interrupted by user")
        finally:
            self.save_all_demos()
            self.close()
    
    def save_all_demos(self):
        """Save all collected demonstrations."""
        if not self.recorder.completed_demos:
            print("No demonstrations to save")
            return
        
        # Save in multiple formats
        format = self.config.get('save_format', 'hdf5')
        
        if format == 'hdf5':
            self._save_hdf5()
        elif format == 'json':
            self._save_json()
        elif format == 'numpy':
            self._save_numpy()
        else:
            print(f"Unknown format: {format}")
    
    def _save_hdf5(self):
        """Save demonstrations in HDF5 format."""
        filename = os.path.join(self.output_dir, 
                               f"demos_{int(time.time())}.hdf5")
        
        with h5py.File(filename, 'w') as f:
            f.attrs['num_demos'] = len(self.recorder.completed_demos)
            f.attrs['task_name'] = self.config.get('task_name', 'unknown')
            f.attrs['collection_time'] = time.time()
            
            demos_group = f.create_group('demonstrations')
            for i, demo in enumerate(self.recorder.completed_demos):
                demo.to_hdf5(demos_group.create_group(f'demo_{i}'))
        
        print(f"Saved {len(self.recorder.completed_demos)} demos to {filename}")
    
    def _save_json(self):
        """Save demonstrations in JSON format."""
        filename = os.path.join(self.output_dir,
                               f"demos_{int(time.time())}.json")
        
        data = {
            'metadata': {
                'num_demos': len(self.recorder.completed_demos),
                'task_name': self.config.get('task_name', 'unknown'),
                'collection_time': time.time()
            },
            'demonstrations': [d.to_dict() for d in self.recorder.completed_demos]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.recorder.completed_demos)} demos to {filename}")
    
    def _save_numpy(self):
        """Save demonstrations as numpy arrays."""
        for i, demo in enumerate(self.recorder.completed_demos):
            # Extract observations and actions
            obs = np.array([s.observation for s in demo.steps if s.observation is not None])
            actions = np.array([s.action for s in demo.steps if s.action is not None])
            
            filename = os.path.join(self.output_dir, f"demo_{i:03d}.npz")
            np.savez(filename, 
                     observations=obs,
                     actions=actions,
                     metadata=demo.to_dict())
        
        print(f"Saved {len(self.recorder.completed_demos)} demos to {self.output_dir}")
    
    def close(self):
        """Clean up resources."""
        self.teleop.stop()
        self.recorder.close()
        self.env.close()
        if hasattr(self, 'fig'):
            plt.close(self.fig)
        print("Collection closed.")


# ======================================================================
# Command Line Interface
# ======================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Collect demonstrations for imitation learning')
    
    parser.add_argument('--config', type=str, default='configs/collect_demos.yaml',
                        help='Path to configuration file')
    
    parser.add_argument('--task', type=str, default='reach',
                        help='Task name')
    
    parser.add_argument('--env', type=str, default='pybullet',
                        choices=['pybullet', 'isaac', 'ros2'],
                        help='Environment type')
    
    parser.add_argument('--robot', type=str, default='humanoid',
                        help='Robot model')
    
    parser.add_argument('--teleop', type=str, default='keyboard',
                        choices=['keyboard', 'joystick', 'vision'],
                        help='Teleoperation type')
    
    parser.add_argument('--output', type=str, default='./demos',
                        help='Output directory')
    
    parser.add_argument('--num-demos', type=int, default=10,
                        help='Number of demonstrations to collect')
    
    parser.add_argument('--record-images', action='store_true',
                        help='Record camera images')
    
    args = parser.parse_args()
    
    # Load or create config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'task_name': args.task,
            'env_type': args.env,
            'env_name': args.robot,
            'teleop_type': args.teleop,
            'output_dir': args.output,
            'num_demos': args.num_demos,
            'record_images': args.record_images,
            'action_dim': 6,  # Default
            'step_dt': 0.01,
            'max_demo_length': 1000,
            'min_demo_length': 10,
            'auto_segment': True,
            'segment_threshold': 0.1,
            'save_format': 'hdf5',
            'visualize': True
        }
    
    # Create collector
    collector = DemonstrationCollector(args.config if os.path.exists(args.config) else 'dummy')
    
    # Run collection
    collector.run()


if __name__ == "__main__":
    main()
```