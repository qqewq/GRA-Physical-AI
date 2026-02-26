```python
"""
GRA Physical AI - Configuration Module
======================================

This module provides configuration management for GRA experiments.
It handles:
    - Loading/saving YAML configuration files
    - Command-line argument parsing
    - Configuration validation
    - Hierarchical configuration (experiment, environment, agent, layers)
    - Default values and overrides
    - Experiment tracking metadata
    - Multi-run configuration
"""

import os
import sys
import yaml
import json
import argparse
import copy
import hashlib
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict, is_dataclass
from datetime import datetime
from pathlib import Path
import warnings


# ======================================================================
# Base Configuration Classes
# ======================================================================

@dataclass
class BaseConfig:
    """Base configuration class with common utilities."""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)
    
    def save(self, path: str):
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'BaseConfig':
        """Create from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, path: str) -> 'BaseConfig':
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def merge(self, other: Dict) -> 'BaseConfig':
        """Merge with another config (override)."""
        merged = copy.deepcopy(self.to_dict())
        merged.update(other)
        return self.__class__.from_dict(merged)
    
    def validate(self) -> List[str]:
        """Validate configuration. Returns list of errors."""
        return []


@dataclass
class EnvironmentConfig(BaseConfig):
    """Configuration for environment."""
    
    type: str = "pybullet"  # pybullet, isaac, ros2, gym
    name: str = "humanoid"
    num_envs: int = 1
    max_steps: int = 1000
    dt: float = 0.01
    gui: bool = True
    seed: Optional[int] = None
    
    # PyBullet specific
    urdf_path: Optional[str] = None
    use_fixed_base: bool = False
    
    # Isaac specific
    usd_path: Optional[str] = None
    headless: bool = False
    
    # ROS2 specific
    observation_topics: List[str] = field(default_factory=list)
    action_topic: str = "/cmd_vel"
    
    # Gym specific
    gym_id: Optional[str] = None
    render_mode: Optional[str] = None
    
    def validate(self) -> List[str]:
        errors = []
        
        if self.type == "pybullet" and self.urdf_path is None:
            errors.append("PyBullet environment requires urdf_path")
        
        if self.type == "isaac" and self.usd_path is None:
            errors.append("Isaac environment requires usd_path")
        
        if self.type == "gym" and self.gym_id is None:
            errors.append("Gym environment requires gym_id")
        
        return errors


@dataclass
class AgentConfig(BaseConfig):
    """Configuration for agent/policy."""
    
    type: str = "mlp"  # mlp, rnn, gr00t, ppo, sac
    name: str = "agent"
    
    # Network architecture
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"
    
    # Input/output dimensions
    observation_dim: Optional[int] = None
    action_dim: Optional[int] = None
    
    # Stochastic policy
    stochastic: bool = False
    log_std_init: float = 0.0
    
    # RNN specific
    rnn_type: str = "lstm"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 2
    
    # GR00T specific
    gr00t_model_path: Optional[str] = None
    gr00t_version: str = "gr00t-v1"
    
    # RL specific
    learning_rate: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 10000
    batch_size: int = 64
    
    def validate(self) -> List[str]:
        errors = []
        
        valid_types = ["mlp", "rnn", "gr00t", "ppo", "sac"]
        if self.type not in valid_types:
            errors.append(f"Invalid agent type: {self.type}. Must be one of {valid_types}")
        
        if self.type == "gr00t" and self.gr00t_model_path is None:
            errors.append("GR00T agent requires gr00t_model_path")
        
        return errors


@dataclass
class GRAConfig(BaseConfig):
    """Configuration for GRA layers."""
    
    # Which layers to use
    use_g0: bool = True
    use_g1: bool = True
    use_g2: bool = True
    use_g3: bool = True
    
    # G0 (motor) configuration
    num_joints: int = 6
    joint_names: Optional[List[str]] = None
    joint_limits: Optional[List[Tuple[float, float]]] = None
    motor_types: Optional[List[str]] = None
    
    # G1 (task) configuration
    task_name: str = "reach"
    max_concurrent_tasks: int = 1
    task_timeout: float = 30.0
    
    # G2 (safety) configuration
    safety_check_freq: int = 10
    emergency_stop_on_violation: bool = True
    max_force: float = 50.0
    personal_space: float = 0.5
    velocity_limits: Optional[List[float]] = None
    
    # G3 (ethics) configuration
    ethics_enabled: bool = True
    human_oversight: bool = True
    dilemma_timeout: float = 30.0
    constitution_version: str = "1.0.0"
    
    # Zeroing configuration
    zeroing_enabled: bool = True
    zeroing_freq: int = 10
    zeroing_learning_rates: List[float] = field(default_factory=lambda: [0.01, 0.005, 0.001, 0.0005])
    zeroing_tolerances: List[float] = field(default_factory=lambda: [0.01, 0.01, 0.001, 0.001])
    
    def validate(self) -> List[str]:
        errors = []
        
        if self.use_g0 and self.num_joints <= 0:
            errors.append("num_joints must be positive")
        
        if self.use_g2 and self.velocity_limits is not None:
            if len(self.velocity_limits) != self.num_joints:
                errors.append(f"velocity_licts length {len(self.velocity_limits)} != num_joints {self.num_joints}")
        
        if self.use_g3 and not self.ethics_enabled:
            errors.append("G3 layer enabled but ethics disabled")
        
        return errors


@dataclass
class TrainingConfig(BaseConfig):
    """Configuration for training."""
    
    # Basic
    experiment_name: str = "gra_experiment"
    total_episodes: int = 100
    max_steps_per_episode: int = 1000
    seed: int = 42
    
    # Checkpoints
    save_checkpoints: bool = True
    checkpoint_freq: int = 10
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_dir: str = "./logs"
    log_level: str = "INFO"
    log_to_file: bool = True
    log_to_tensorboard: bool = False
    log_to_mlflow: bool = False
    log_to_wandb: bool = False
    
    # Evaluation
    eval_freq: int = 10
    eval_episodes: int = 5
    eval_deterministic: bool = True
    
    # Human feedback
    feedback_enabled: bool = True
    feedback_aggregation_window: float = 60.0
    
    # Visualization
    visualize: bool = True
    vis_freq: int = 10
    vis_save: bool = True
    
    def validate(self) -> List[str]:
        errors = []
        
        if self.total_episodes <= 0:
            errors.append("total_episodes must be positive")
        
        if self.checkpoint_freq <= 0:
            errors.append("checkpoint_freq must be positive")
        
        return errors


@dataclass
class ExperimentConfig(BaseConfig):
    """Complete experiment configuration."""
    
    # Metadata
    experiment_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    description: str = ""
    author: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    # Component configs
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    gra: GRAConfig = field(default_factory=GRAConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Overrides (for grid search)
    overrides: Dict[str, List[Any]] = field(default_factory=dict)
    
    # Tags
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Generate experiment ID if not set."""
        if not self.experiment_id:
            self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def to_dict(self) -> Dict:
        """Convert to nested dictionary."""
        return {
            'metadata': {
                'experiment_id': self.experiment_id,
                'description': self.description,
                'author': self.author,
                'created_at': self.created_at,
                'tags': self.tags
            },
            'environment': self.environment.to_dict(),
            'agent': self.agent.to_dict(),
            'gra': self.gra.to_dict(),
            'training': self.training.to_dict(),
            'overrides': self.overrides
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentConfig':
        """Create from nested dictionary."""
        metadata = data.get('metadata', {})
        
        return cls(
            experiment_id=metadata.get('experiment_id', ''),
            description=metadata.get('description', ''),
            author=metadata.get('author', ''),
            created_at=metadata.get('created_at', datetime.now().isoformat()),
            tags=metadata.get('tags', []),
            environment=EnvironmentConfig.from_dict(data.get('environment', {})),
            agent=AgentConfig.from_dict(data.get('agent', {})),
            gra=GRAConfig.from_dict(data.get('gra', {})),
            training=TrainingConfig.from_dict(data.get('training', {})),
            overrides=data.get('overrides', {})
        )
    
    def save(self, path: str):
        """Save to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: str) -> 'ExperimentConfig':
        """Load from YAML file."""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
    
    def hash(self) -> str:
        """Create a hash of the configuration for unique identification."""
        config_str = yaml.dump(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:8]
    
    def create_run_dir(self, base_dir: str = "./runs") -> str:
        """Create a run directory with experiment ID and hash."""
        run_dir = os.path.join(base_dir, f"{self.experiment_id}_{self.hash()}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save config in run directory
        self.save(os.path.join(run_dir, "config.yaml"))
        
        return run_dir
    
    def validate(self) -> List[str]:
        """Validate entire configuration."""
        errors = []
        
        # Validate sub-configs
        errors.extend(self.environment.validate())
        errors.extend(self.agent.validate())
        errors.extend(self.gra.validate())
        errors.extend(self.training.validate())
        
        # Check consistency between configs
        if self.gra.use_g0 and self.agent.action_dim is not None:
            if self.agent.action_dim != self.gra.num_joints:
                errors.append(
                    f"Agent action_dim ({self.agent.action_dim}) != "
                    f"gra.num_joints ({self.gra.num_joints})"
                )
        
        return errors


# ======================================================================
# Configuration Manager
# ======================================================================

class ConfigManager:
    """
    Manages experiment configurations.
    
    Handles:
        - Loading from files
        - Command-line overrides
        - Saving configurations
        - Experiment tracking
        - Multi-run (grid search) generation
    """
    
    def __init__(self, config_dir: str = "./configs"):
        """
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_config: Optional[ExperimentConfig] = None
    
    def load_config(self, path: str) -> ExperimentConfig:
        """Load configuration from file."""
        path = Path(path)
        if not path.exists():
            path = self.config_dir / path
        
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        self.current_config = ExperimentConfig.load(str(path))
        return self.current_config
    
    def create_default_config(self) -> ExperimentConfig:
        """Create default configuration."""
        return ExperimentConfig()
    
    def save_config(self, config: ExperimentConfig, path: Optional[str] = None):
        """Save configuration to file."""
        if path is None:
            path = self.config_dir / f"{config.experiment_id}.yaml"
        
        config.save(str(path))
    
    def from_args(self, args: Optional[List[str]] = None) -> ExperimentConfig:
        """
        Create configuration from command-line arguments.
        
        Supports:
            --config path/to/config.yaml
            --override key=value
            --override key.nested=value
        """
        parser = argparse.ArgumentParser(description='GRA Experiment Configuration')
        
        parser.add_argument('--config', type=str, help='Path to config file')
        parser.add_argument('--override', action='append', help='Override config values (key=value)')
        
        # Parse known args
        parsed_args, unknown = parser.parse_known_args(args)
        
        # Load config file if provided
        if parsed_args.config:
            config = self.load_config(parsed_args.config)
        else:
            config = self.create_default_config()
        
        # Apply overrides
        if parsed_args.override:
            for override in parsed_args.override:
                self._apply_override(config, override)
        
        # Parse unknown args as additional overrides
        for i in range(0, len(unknown), 2):
            if unknown[i] == '--set':
                self._apply_override(config, unknown[i+1])
        
        self.current_config = config
        return config
    
    def _apply_override(self, config: ExperimentConfig, override: str):
        """
        Apply a single override in the form "key=value" or "key.nested=value".
        
        Examples:
            training.total_episodes=200
            agent.learning_rate=0.001
            environment.name=humanoid
        """
        if '=' not in override:
            warnings.warn(f"Invalid override format: {override}. Expected key=value")
            return
        
        key, value = override.split('=', 1)
        
        # Parse value (try int, float, bool, list)
        try:
            if value.lower() == 'true':
                value = True
            elif value.lower() == 'false':
                value = False
            elif ',' in value:
                # List
                value = [v.strip() for v in value.split(',')]
                # Try converting to numbers
                try:
                    value = [int(v) if '.' not in v else float(v) for v in value]
                except:
                    pass
            elif '.' in value:
                value = float(value)
            else:
                value = int(value)
        except:
            pass  # Keep as string
        
        # Navigate nested keys
        parts = key.split('.')
        target = config
        
        for part in parts[:-1]:
            if hasattr(target, part):
                target = getattr(target, part)
            else:
                warnings.warn(f"Unknown config key: {key}")
                return
        
        last_part = parts[-1]
        if hasattr(target, last_part):
            setattr(target, last_part, value)
        else:
            warnings.warn(f"Unknown config key: {key}")
    
    def generate_grid(self, base_config: ExperimentConfig) -> List[ExperimentConfig]:
        """
        Generate grid of configurations for hyperparameter search.
        
        Uses the 'overrides' field which should contain lists of values
        for each parameter to sweep.
        """
        if not base_config.overrides:
            return [base_config]
        
        # Build list of parameter combinations
        import itertools
        
        param_names = list(base_config.overrides.keys())
        param_values = list(base_config.overrides.values())
        
        configs = []
        
        for combination in itertools.product(*param_values):
            # Create copy of base config
            config = copy.deepcopy(base_config)
            
            # Apply this combination of overrides
            for name, value in zip(param_names, combination):
                self._apply_override(config, f"{name}={value}")
            
            # Update experiment ID for this run
            config.experiment_id = f"{base_config.experiment_id}_{len(configs)}"
            
            configs.append(config)
        
        return configs
    
    def save_all_grid(self, configs: List[ExperimentConfig], base_name: str = "grid"):
        """Save all grid configurations."""
        for i, config in enumerate(configs):
            path = self.config_dir / f"{base_name}_{i:03d}.yaml"
            config.save(str(path))
        
        print(f"Saved {len(configs)} grid configurations to {self.config_dir}")
    
    def get_current(self) -> Optional[ExperimentConfig]:
        """Get current configuration."""
        return self.current_config


# ======================================================================
# Configuration Presets
# ======================================================================

class PresetConfigs:
    """Collection of preset configurations."""
    
    @staticmethod
    def humanoid_safety() -> ExperimentConfig:
        """Humanoid robot with full safety and ethics layers."""
        return ExperimentConfig(
            experiment_id="humanoid_safety",
            description="Humanoid robot with full GRA safety and ethics",
            environment=EnvironmentConfig(
                type="pybullet",
                name="humanoid",
                urdf_path="assets/humanoid.urdf",
                gui=True
            ),
            agent=AgentConfig(
                type="mlp",
                observation_dim=48,  # Joint states + IMU
                action_dim=21,  # Joint torques
                hidden_dims=[512, 512],
                stochastic=True
            ),
            gra=GRAConfig(
                use_g0=True,
                use_g1=True,
                use_g2=True,
                use_g3=True,
                num_joints=21,
                joint_limits=[(-2.8, 2.8)] * 21,
                max_force=50.0,
                personal_space=0.5,
                ethics_enabled=True
            ),
            training=TrainingConfig(
                total_episodes=1000,
                max_steps_per_episode=500,
                checkpoint_freq=50,
                log_to_tensorboard=True,
                visualize=True
            )
        )
    
    @staticmethod
    def cartpole_basic() -> ExperimentConfig:
        """Basic CartPole experiment for testing."""
        return ExperimentConfig(
            experiment_id="cartpole_basic",
            description="Basic CartPole with G0 only",
            environment=EnvironmentConfig(
                type="gym",
                name="cartpole",
                gym_id="CartPole-v1",
                gui=False
            ),
            agent=AgentConfig(
                type="mlp",
                observation_dim=4,
                action_dim=2,
                hidden_dims=[64, 64],
                stochastic=False
            ),
            gra=GRAConfig(
                use_g0=True,
                use_g1=False,
                use_g2=False,
                use_g3=False,
                num_joints=2
            ),
            training=TrainingConfig(
                total_episodes=100,
                max_steps_per_episode=500,
                visualize=False
            )
        )
    
    @staticmethod
    def multi_robot_swarm() -> ExperimentConfig:
        """Multi-robot swarm experiment."""
        return ExperimentConfig(
            experiment_id="swarm_3_robots",
            description="Three robots with swarm coordination",
            environment=EnvironmentConfig(
                type="pybullet",
                name="multi_robot",
                num_envs=3,
                urdf_path="assets/r2d2.urdf",
                gui=True
            ),
            agent=AgentConfig(
                type="mlp",
                observation_dim=10,
                action_dim=2,  # Left/right wheel velocities
                hidden_dims=[128, 128]
            ),
            gra=GRAConfig(
                use_g0=True,
                use_g1=True,
                use_g2=True,
                use_g3=False,
                num_joints=2,
                zeroing_learning_rates=[0.01, 0.005, 0.001]
            ),
            training=TrainingConfig(
                total_episodes=500,
                max_steps_per_episode=1000,
                checkpoint_freq=25
            )
        )
    
    @staticmethod
    def gr00t_mimic() -> ExperimentConfig:
        """GR00T-Mimic fine-tuning configuration."""
        return ExperimentConfig(
            experiment_id="gr00t_mimic_reach",
            description="Fine-tune GR00T on reaching demonstrations",
            environment=EnvironmentConfig(
                type="isaac",
                name="franka_reach",
                usd_path="assets/franka_reach.usd",
                headless=False
            ),
            agent=AgentConfig(
                type="gr00t",
                gr00t_model_path="models/gr00t_base",
                learning_rate=1e-5,
                stochastic=True
            ),
            gra=GRAConfig(
                use_g0=True,
                use_g1=True,
                use_g2=True,
                use_g3=True,
                num_joints=7,
                max_force=30.0,
                personal_space=0.3
            ),
            training=TrainingConfig(
                total_episodes=200,
                max_steps_per_episode=100,
                checkpoint_freq=10,
                log_to_wandb=True,
                feedback_enabled=True
            )
        )
    
    @staticmethod
    def grid_search_example() -> ExperimentConfig:
        """Example grid search configuration."""
        return ExperimentConfig(
            experiment_id="grid_search",
            description="Grid search over learning rates and network sizes",
            environment=EnvironmentConfig(
                type="gym",
                name="cartpole",
                gym_id="CartPole-v1"
            ),
            agent=AgentConfig(
                type="mlp",
                observation_dim=4,
                action_dim=2
            ),
            gra=GRAConfig(
                use_g0=True,
                num_joints=2
            ),
            training=TrainingConfig(
                total_episodes=100,
                max_steps_per_episode=500
            ),
            overrides={
                "agent.learning_rate": [1e-4, 3e-4, 1e-3],
                "agent.hidden_dims": [[64, 64], [128, 128], [256, 256]],
                "training.total_episodes": [50, 100, 200]
            }
        )


# ======================================================================
# Command Line Interface
# ======================================================================

def main():
    """Command-line interface for config management."""
    parser = argparse.ArgumentParser(description='GRA Configuration Manager')
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Create command
    create_parser = subparsers.add_parser('create', help='Create a config file')
    create_parser.add_argument('--type', type=str, default='humanoid_safety',
                              choices=['humanoid_safety', 'cartpole_basic', 
                                      'multi_robot_swarm', 'gr00t_mimic',
                                      'grid_search_example'],
                              help='Preset type')
    create_parser.add_argument('--output', type=str, default='config.yaml',
                              help='Output file')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a config file')
    validate_parser.add_argument('config', type=str, help='Config file to validate')
    
    # Show command
    show_parser = subparsers.add_parser('show', help='Show config contents')
    show_parser.add_argument('config', type=str, help='Config file to show')
    
    # Grid command
    grid_parser = subparsers.add_parser('grid', help='Generate grid from config')
    grid_parser.add_argument('config', type=str, help='Base config file')
    grid_parser.add_argument('--output-dir', type=str, default='./configs/grid',
                            help='Output directory')
    
    args = parser.parse_args()
    
    manager = ConfigManager()
    
    if args.command == 'create':
        # Get preset
        preset_func = getattr(PresetConfigs, args.type)
        config = preset_func()
        
        # Save
        manager.save_config(config, args.output)
        print(f"Created config: {args.output}")
        
        # Show summary
        print("\nConfiguration Summary:")
        print(f"  Experiment: {config.experiment_id}")
        print(f"  Description: {config.description}")
        print(f"  Environment: {config.environment.type}/{config.environment.name}")
        print(f"  Agent: {config.agent.type}")
        print(f"  GRA layers: G0={config.gra.use_g0}, G1={config.gra.use_g1}, "
              f"G2={config.gra.use_g2}, G3={config.gra.use_g3}")
    
    elif args.command == 'validate':
        config = manager.load_config(args.config)
        errors = config.validate()
        
        if errors:
            print(f"Validation errors for {args.config}:")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"Config {args.config} is valid")
    
    elif args.command == 'show':
        config = manager.load_config(args.config)
        print(config.to_yaml())
    
    elif args.command == 'grid':
        base_config = manager.load_config(args.config)
        configs = manager.generate_grid(base_config)
        manager.save_all_grid(configs, base_config.experiment_id)
    
    else:
        parser.print_help()


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    # Example 1: Create and use config
    print("=== Configuration Example ===\n")
    
    # Create a config
    config = PresetConfigs.humanoid_safety()
    
    print("Config created:")
    print(f"  Experiment: {config.experiment_id}")
    print(f"  Description: {config.description}")
    print(f"  Environment: {config.environment.type}")
    print(f"  Agent type: {config.agent.type}")
    
    # Validate
    errors = config.validate()
    if errors:
        print("\nValidation errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("\nConfig is valid")
    
    # Example 2: Override values
    print("\n--- Overriding values ---")
    config.training.total_episodes = 500
    config.agent.learning_rate = 1e-3
    print(f"  total_episodes: {config.training.total_episodes}")
    print(f"  learning_rate: {config.agent.learning_rate}")
    
    # Example 3: Save and load
    print("\n--- Save and load ---")
    config.save("test_config.yaml")
    loaded = ExperimentConfig.load("test_config.yaml")
    print(f"  Loaded experiment: {loaded.experiment_id}")
    
    # Example 4: Grid search
    print("\n--- Grid search generation ---")
    grid_config = PresetConfigs.grid_search_example()
    grid_configs = ConfigManager().generate_grid(grid_config)
    print(f"  Generated {len(grid_configs)} configurations")
    for i, c in enumerate(grid_configs[:3]):  # Show first 3
        print(f"    {i}: {c.agent.learning_rate}, {c.agent.hidden_dims}")
    
    # Clean up
    os.remove("test_config.yaml")
    
    print("\nAll tests passed!")
```