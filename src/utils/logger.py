```python
"""
GRA Physical AI - Logging Module
================================

This module provides comprehensive logging for GRA training and evaluation.
It handles:
    - Training metrics (rewards, losses, foam levels)
    - Safety incidents and ethical violations
    - Human feedback and interactions
    - Policy parameters and gradients
    - System resource usage
    - Experiment metadata and configuration

Supports multiple backends:
    - Console logging (colored, formatted)
    - File logging (JSON, CSV, text)
    - TensorBoard integration
    - MLflow tracking
    - Weights & Biases
    - Custom callbacks
"""

import os
import sys
import json
import time
import yaml
import csv
import numpy as np
import torch
from typing import Dict, List, Optional, Union, Any, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict, deque
from enum import Enum
import warnings
import threading
import queue
import atexit

# Try to import optional visualization backends
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from colorama import init, Fore, Back, Style
    init()
    COLORAMA_AVAILABLE = True
except ImportError:
    COLORAMA_AVAILABLE = False


# ======================================================================
# Log Levels and Colors
# ======================================================================

class LogLevel(Enum):
    """Log levels with numeric values."""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


_LOG_LEVEL_COLORS = {
    LogLevel.DEBUG: 'CYAN' if COLORAMA_AVAILABLE else None,
    LogLevel.INFO: 'GREEN' if COLORAMA_AVAILABLE else None,
    LogLevel.WARNING: 'YELLOW' if COLORAMA_AVAILABLE else None,
    LogLevel.ERROR: 'RED' if COLORAMA_AVAILABLE else None,
    LogLevel.CRITICAL: 'MAGENTA' if COLORAMA_AVAILABLE else None,
}


def _colorize(text: str, color: Optional[str]) -> str:
    """Add ANSI color codes if available."""
    if not COLORAMA_AVAILABLE or not color:
        return text
    
    color_map = {
        'RED': Fore.RED,
        'GREEN': Fore.GREEN,
        'YELLOW': Fore.YELLOW,
        'BLUE': Fore.BLUE,
        'MAGENTA': Fore.MAGENTA,
        'CYAN': Fore.CYAN,
        'WHITE': Fore.WHITE,
    }
    
    return color_map.get(color, '') + text + Style.RESET_ALL


# ======================================================================
# Log Entry Classes
# ======================================================================

@dataclass
class LogEntry:
    """Base class for all log entries."""
    
    timestamp: float
    level: LogLevel
    source: str
    message: str
    data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'level': self.level.value,
            'source': self.source,
            'message': self.message,
            'data': self.data
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LogEntry':
        """Create from dictionary."""
        return cls(
            timestamp=data['timestamp'],
            level=LogLevel(data['level']),
            source=data['source'],
            message=data['message'],
            data=data.get('data')
        )


@dataclass
class MetricEntry(LogEntry):
    """Log entry for scalar metrics."""
    
    name: str
    value: float
    step: Optional[int] = None
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'type': 'metric',
            'name': self.name,
            'value': self.value,
            'step': self.step
        })
        return base


@dataclass
class HistogramEntry(LogEntry):
    """Log entry for histogram data."""
    
    name: str
    data: np.ndarray
    bins: int = 50
    step: Optional[int] = None
    
    def to_dict(self) -> Dict:
        hist, edges = np.histogram(self.data, bins=self.bins)
        base = super().to_dict()
        base.update({
            'type': 'histogram',
            'name': self.name,
            'histogram': hist.tolist(),
            'edges': edges.tolist(),
            'step': self.step
        })
        return base


@dataclass
class ImageEntry(LogEntry):
    """Log entry for images."""
    
    name: str
    image: np.ndarray
    step: Optional[int] = None
    format: str = 'png'
    
    def to_dict(self) -> Dict:
        # Don't store full image in dict
        base = super().to_dict()
        base.update({
            'type': 'image',
            'name': self.name,
            'shape': list(self.image.shape),
            'format': self.format,
            'step': self.step
        })
        return base


@dataclass
class SafetyEntry(LogEntry):
    """Log entry for safety incidents."""
    
    incident_id: str
    violation_type: str
    severity: float
    robot_state: Optional[Dict] = None
    intervention_taken: bool = False
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'type': 'safety',
            'incident_id': self.incident_id,
            'violation_type': self.violation_type,
            'severity': self.severity,
            'intervention_taken': self.intervention_taken,
            'robot_state': self.robot_state
        })
        return base


@dataclass
class EthicalEntry(LogEntry):
    """Log entry for ethical dilemmas and violations."""
    
    dilemma_id: Optional[str]
    principle: str
    resolution: Optional[str]
    human_involved: bool = False
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'type': 'ethical',
            'dilemma_id': self.dilemma_id,
            'principle': self.principle,
            'resolution': self.resolution,
            'human_involved': self.human_involved
        })
        return base


@dataclass
class FeedbackEntry(LogEntry):
    """Log entry for human feedback."""
    
    feedback_id: str
    feedback_type: str
    user_id: Optional[str]
    rating: Optional[float] = None
    
    def to_dict(self) -> Dict:
        base = super().to_dict()
        base.update({
            'type': 'feedback',
            'feedback_id': self.feedback_id,
            'feedback_type': self.feedback_type,
            'user_id': self.user_id,
            'rating': self.rating
        })
        return base


# ======================================================================
# Log Handlers
# ======================================================================

class LogHandler(ABC):
    """Abstract base class for log handlers."""
    
    def __init__(self, level: LogLevel = LogLevel.INFO):
        self.level = level
    
    @abstractmethod
    def handle(self, entry: LogEntry):
        """Handle a log entry."""
        pass
    
    def should_handle(self, entry: LogEntry) -> bool:
        """Check if this handler should process the entry."""
        return entry.level.value >= self.level.value


class ConsoleHandler(LogHandler):
    """Handler that prints logs to console."""
    
    def __init__(self, level: LogLevel = LogLevel.INFO, 
                 colored: bool = True,
                 show_source: bool = True,
                 show_timestamp: bool = True):
        super().__init__(level)
        self.colored = colored
        self.show_source = show_source
        self.show_timestamp = show_timestamp
    
    def handle(self, entry: LogEntry):
        if not self.should_handle(entry):
            return
        
        # Build message
        parts = []
        
        if self.show_timestamp:
            time_str = datetime.fromtimestamp(entry.timestamp).strftime('%H:%M:%S')
            parts.append(f"[{time_str}]")
        
        level_name = entry.level.name
        if self.colored:
            level_name = _colorize(level_name, _LOG_LEVEL_COLORS.get(entry.level))
        parts.append(f"[{level_name}]")
        
        if self.show_source and entry.source:
            parts.append(f"[{entry.source}]")
        
        parts.append(entry.message)
        
        # Print
        print(" ".join(parts))
        
        # Print data if present
        if entry.data:
            print(f"  Data: {entry.data}")


class FileHandler(LogHandler):
    """Handler that writes logs to files."""
    
    def __init__(self, log_dir: str, 
                 level: LogLevel = LogLevel.DEBUG,
                 format: str = 'jsonl'):  # 'jsonl', 'csv', 'txt'
        super().__init__(level)
        self.log_dir = log_dir
        self.format = format
        self.files = {}
        
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_file(self, entry_type: str):
        """Get file handle for entry type."""
        if entry_type not in self.files:
            filename = os.path.join(self.log_dir, f"{entry_type}.{self.format}")
            
            if self.format == 'jsonl':
                self.files[entry_type] = open(filename, 'a')
            elif self.format == 'csv':
                self.files[entry_type] = open(filename, 'a', newline='')
            else:
                self.files[entry_type] = open(filename, 'a')
        
        return self.files[entry_type]
    
    def handle(self, entry: LogEntry):
        if not self.should_handle(entry):
            return
        
        # Determine entry type
        if isinstance(entry, MetricEntry):
            entry_type = 'metrics'
        elif isinstance(entry, SafetyEntry):
            entry_type = 'safety'
        elif isinstance(entry, EthicalEntry):
            entry_type = 'ethical'
        elif isinstance(entry, FeedbackEntry):
            entry_type = 'feedback'
        else:
            entry_type = 'logs'
        
        f = self._get_file(entry_type)
        
        if self.format == 'jsonl':
            f.write(json.dumps(entry.to_dict()) + '\n')
            f.flush()
        
        elif self.format == 'csv':
            writer = csv.DictWriter(f, fieldnames=entry.to_dict().keys())
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(entry.to_dict())
            f.flush()
        
        else:  # txt
            f.write(f"{datetime.fromtimestamp(entry.timestamp)} - {entry.level.name} - {entry.source}: {entry.message}\n")
            if entry.data:
                f.write(f"  {entry.data}\n")
            f.flush()
    
    def close(self):
        """Close all open files."""
        for f in self.files.values():
            f.close()


class TensorBoardHandler(LogHandler):
    """Handler that logs to TensorBoard."""
    
    def __init__(self, log_dir: str, level: LogLevel = LogLevel.INFO):
        if not TENSORBOARD_AVAILABLE:
            raise ImportError("TensorBoard not available")
        
        super().__init__(level)
        self.writer = SummaryWriter(log_dir)
        self.steps = defaultdict(int)
    
    def handle(self, entry: LogEntry):
        if not self.should_handle(entry):
            return
        
        if isinstance(entry, MetricEntry):
            step = entry.step if entry.step is not None else self.steps[entry.name]
            self.writer.add_scalar(entry.name, entry.value, step)
            self.steps[entry.name] = step + 1
        
        elif isinstance(entry, HistogramEntry):
            step = entry.step if entry.step is not None else self.steps[entry.name]
            self.writer.add_histogram(entry.name, entry.data, step)
            self.steps[entry.name] = step + 1
        
        elif isinstance(entry, ImageEntry):
            step = entry.step if entry.step is not None else self.steps[entry.name]
            self.writer.add_image(entry.name, entry.image, step, dataformats='HWC')
            self.steps[entry.name] = step + 1
    
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()


class MLflowHandler(LogHandler):
    """Handler that logs to MLflow."""
    
    def __init__(self, experiment_name: str, 
                 tracking_uri: Optional[str] = None,
                 level: LogLevel = LogLevel.INFO):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available")
        
        super().__init__(level)
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        mlflow.set_experiment(experiment_name)
        self.run = mlflow.start_run()
        self.steps = defaultdict(int)
    
    def handle(self, entry: LogEntry):
        if not self.should_handle(entry):
            return
        
        with mlflow.start_run(run_id=self.run.info.run_id, nested=True):
            if isinstance(entry, MetricEntry):
                step = entry.step if entry.step is not None else self.steps[entry.name]
                mlflow.log_metric(entry.name, entry.value, step=step)
                self.steps[entry.name] = step + 1
            
            elif isinstance(entry, HistogramEntry):
                # MLflow doesn't support histograms directly
                pass
            
            elif isinstance(entry, ImageEntry):
                # Save image temporarily
                import tempfile
                from PIL import Image
                
                with tempfile.NamedTemporaryFile(suffix=f".{entry.format}", delete=False) as f:
                    img = Image.fromarray(entry.image)
                    img.save(f.name)
                    mlflow.log_artifact(f.name, artifact_path="images")
    
    def close(self):
        """End MLflow run."""
        mlflow.end_run()


class WandBHandler(LogHandler):
    """Handler that logs to Weights & Biases."""
    
    def __init__(self, project: str, 
                 config: Optional[Dict] = None,
                 level: LogLevel = LogLevel.INFO):
        if not WANDB_AVAILABLE:
            raise ImportError("wandb not available")
        
        super().__init__(level)
        
        wandb.init(project=project, config=config)
        self.steps = defaultdict(int)
    
    def handle(self, entry: LogEntry):
        if not self.should_handle(entry):
            return
        
        if isinstance(entry, MetricEntry):
            step = entry.step if entry.step is not None else self.steps[entry.name]
            wandb.log({entry.name: entry.value}, step=step)
            self.steps[entry.name] = step + 1
        
        elif isinstance(entry, HistogramEntry):
            step = entry.step if entry.step is not None else self.steps[entry.name]
            wandb.log({entry.name: wandb.Histogram(entry.data)}, step=step)
            self.steps[entry.name] = step + 1
        
        elif isinstance(entry, ImageEntry):
            step = entry.step if entry.step is not None else self.steps[entry.name]
            wandb.log({entry.name: wandb.Image(entry.image)}, step=step)
            self.steps[entry.name] = step + 1
    
    def close(self):
        """Finish wandb run."""
        wandb.finish()


# ======================================================================
# Main Logger Class
# ======================================================================

class GRA_Logger:
    """
    Main logger for GRA Physical AI.
    
    Features:
        - Multiple handlers (console, file, tensorboard, etc.)
        - Async logging (non-blocking)
        - Metrics aggregation
        - Experiment tracking
        - Checkpointing
    """
    
    def __init__(self, 
                 name: str = "gra_experiment",
                 log_dir: str = "./logs",
                 level: LogLevel = LogLevel.INFO,
                 async_logging: bool = True,
                 buffer_size: int = 1000):
        """
        Args:
            name: Experiment name
            log_dir: Directory for log files
            level: Default log level
            async_logging: Use async logging (non-blocking)
            buffer_size: Size of async queue
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        self.async_logging = async_logging
        
        os.makedirs(log_dir, exist_ok=True)
        
        # Handlers
        self.handlers: List[LogHandler] = []
        
        # Async queue
        if async_logging:
            self.queue = queue.Queue(maxsize=buffer_size)
            self.running = True
            self.thread = threading.Thread(target=self._process_queue)
            self.thread.daemon = True
            self.thread.start()
        
        # Metrics aggregation
        self.metrics = defaultdict(list)
        self.metrics_agg = defaultdict(lambda: {'sum': 0, 'count': 0, 'min': float('inf'), 'max': float('-inf')})
        
        # Global step counter
        self.global_step = 0
        
        # Register cleanup
        atexit.register(self.close)
        
        # Log experiment start
        self.info("logger", f"Logger initialized - {name}", {
            'log_dir': log_dir,
            'level': level.name,
            'async': async_logging
        })
    
    # ======================================================================
    # Handler Management
    # ======================================================================
    
    def add_handler(self, handler: LogHandler):
        """Add a log handler."""
        self.handlers.append(handler)
    
    def add_console_handler(self, level: LogLevel = LogLevel.INFO, **kwargs):
        """Add console handler."""
        handler = ConsoleHandler(level=level, **kwargs)
        self.add_handler(handler)
        return handler
    
    def add_file_handler(self, level: LogLevel = LogLevel.DEBUG, **kwargs):
        """Add file handler."""
        handler = FileHandler(self.log_dir, level=level, **kwargs)
        self.add_handler(handler)
        return handler
    
    def add_tensorboard_handler(self, level: LogLevel = LogLevel.INFO):
        """Add TensorBoard handler."""
        tb_dir = os.path.join(self.log_dir, 'tensorboard')
        handler = TensorBoardHandler(tb_dir, level=level)
        self.add_handler(handler)
        return handler
    
    def add_mlflow_handler(self, experiment_name: Optional[str] = None, **kwargs):
        """Add MLflow handler."""
        if experiment_name is None:
            experiment_name = self.name
        handler = MLflowHandler(experiment_name, **kwargs)
        self.add_handler(handler)
        return handler
    
    def add_wandb_handler(self, project: Optional[str] = None, config: Optional[Dict] = None, **kwargs):
        """Add Weights & Biases handler."""
        if project is None:
            project = self.name
        handler = WandBHandler(project, config=config, **kwargs)
        self.add_handler(handler)
        return handler
    
    # ======================================================================
    # Core Logging Methods
    # ======================================================================
    
    def log(self, entry: LogEntry):
        """Log an entry."""
        if self.async_logging:
            try:
                self.queue.put_nowait(entry)
            except queue.Full:
                warnings.warn("Log queue full, dropping message")
        else:
            self._handle_entry(entry)
    
    def _handle_entry(self, entry: LogEntry):
        """Process a log entry through all handlers."""
        for handler in self.handlers:
            try:
                handler.handle(entry)
            except Exception as e:
                print(f"Error in log handler: {e}")
    
    def _process_queue(self):
        """Process async log queue."""
        while self.running:
            try:
                entry = self.queue.get(timeout=1.0)
                self._handle_entry(entry)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in log processor: {e}")
    
    # ======================================================================
    # Convenience Logging Methods
    # ======================================================================
    
    def debug(self, source: str, message: str, data: Optional[Dict] = None):
        """Log debug message."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.DEBUG,
            source=source,
            message=message,
            data=data
        )
        self.log(entry)
    
    def info(self, source: str, message: str, data: Optional[Dict] = None):
        """Log info message."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            source=source,
            message=message,
            data=data
        )
        self.log(entry)
    
    def warning(self, source: str, message: str, data: Optional[Dict] = None):
        """Log warning message."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.WARNING,
            source=source,
            message=message,
            data=data
        )
        self.log(entry)
    
    def error(self, source: str, message: str, data: Optional[Dict] = None):
        """Log error message."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.ERROR,
            source=source,
            message=message,
            data=data
        )
        self.log(entry)
    
    def critical(self, source: str, message: str, data: Optional[Dict] = None):
        """Log critical message."""
        entry = LogEntry(
            timestamp=time.time(),
            level=LogLevel.CRITICAL,
            source=source,
            message=message,
            data=data
        )
        self.log(entry)
    
    # ======================================================================
    # Metric Logging
    # ======================================================================
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a scalar metric."""
        if step is None:
            step = self.global_step
        
        entry = MetricEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            source="metrics",
            message=f"{name}: {value:.4f}",
            name=name,
            value=value,
            step=step
        )
        self.log(entry)
        
        # Update aggregation
        self.metrics[name].append(value)
        agg = self.metrics_agg[name]
        agg['sum'] += value
        agg['count'] += 1
        agg['min'] = min(agg['min'], value)
        agg['max'] = max(agg['max'], value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        for name, value in metrics.items():
            self.log_metric(name, value, step)
    
    def log_histogram(self, name: str, data: Union[np.ndarray, torch.Tensor], 
                      bins: int = 50, step: Optional[int] = None):
        """Log histogram data."""
        if step is None:
            step = self.global_step
        
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        
        entry = HistogramEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            source="histograms",
            message=f"Histogram: {name}",
            name=name,
            data=data,
            bins=bins,
            step=step
        )
        self.log(entry)
    
    def log_image(self, name: str, image: np.ndarray, 
                  step: Optional[int] = None, format: str = 'png'):
        """Log image data."""
        if step is None:
            step = self.global_step
        
        entry = ImageEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            source="images",
            message=f"Image: {name}",
            name=name,
            image=image,
            step=step,
            format=format
        )
        self.log(entry)
    
    # ======================================================================
    # GRA-Specific Logging
    # ======================================================================
    
    def log_safety_incident(self, incident_id: str, violation_type: str,
                           severity: float, robot_state: Optional[Dict] = None,
                           intervention_taken: bool = False):
        """Log safety incident."""
        entry = SafetyEntry(
            timestamp=time.time(),
            level=LogLevel.WARNING,
            source="safety",
            message=f"Safety incident: {violation_type} (severity: {severity:.2f})",
            incident_id=incident_id,
            violation_type=violation_type,
            severity=severity,
            robot_state=robot_state,
            intervention_taken=intervention_taken
        )
        self.log(entry)
    
    def log_ethical_dilemma(self, dilemma_id: Optional[str], principle: str,
                           resolution: Optional[str] = None,
                           human_involved: bool = False):
        """Log ethical dilemma."""
        entry = EthicalEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            source="ethics",
            message=f"Ethical dilemma: {principle} - {'Resolved' if resolution else 'Pending'}",
            dilemma_id=dilemma_id,
            principle=principle,
            resolution=resolution,
            human_involved=human_involved
        )
        self.log(entry)
    
    def log_human_feedback(self, feedback_id: str, feedback_type: str,
                          user_id: Optional[str] = None, rating: Optional[float] = None):
        """Log human feedback."""
        entry = FeedbackEntry(
            timestamp=time.time(),
            level=LogLevel.INFO,
            source="feedback",
            message=f"Human feedback: {feedback_type}",
            feedback_id=feedback_id,
            feedback_type=feedback_type,
            user_id=user_id,
            rating=rating
        )
        self.log(entry)
    
    def log_foam(self, foam_dict: Dict[int, float], step: Optional[int] = None):
        """Log foam values for all levels."""
        for level, foam in foam_dict.items():
            self.log_metric(f"foam/level_{level}", foam, step)
    
    # ======================================================================
    # Aggregation and Statistics
    # ======================================================================
    
    def get_metric_stats(self, name: str, window: Optional[int] = None) -> Dict:
        """Get statistics for a metric."""
        values = self.metrics[name]
        if window:
            values = values[-window:]
        
        if not values:
            return {}
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'last': values[-1]
        }
    
    def get_all_metrics(self) -> Dict:
        """Get all metrics."""
        return dict(self.metrics)
    
    def reset_metrics(self, name: Optional[str] = None):
        """Reset metrics."""
        if name:
            self.metrics[name] = []
            self.metrics_agg[name] = {'sum': 0, 'count': 0, 'min': float('inf'), 'max': float('-inf')}
        else:
            self.metrics.clear()
            self.metrics_agg.clear()
    
    # ======================================================================
    # Experiment Tracking
    # ======================================================================
    
    def log_config(self, config: Dict):
        """Log experiment configuration."""
        config_path = os.path.join(self.log_dir, 'config.yaml')
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        self.info("config", "Configuration saved", {'path': config_path})
    
    def log_params(self, params: Dict):
        """Log model parameters."""
        # Count parameters
        total_params = 0
        trainable_params = 0
        
        for name, param in params.items():
            if hasattr(param, 'numel'):
                numel = param.numel()
                total_params += numel
                if hasattr(param, 'requires_grad') and param.requires_grad:
                    trainable_params += numel
                
                # Log histogram for each parameter
                if hasattr(param, 'cpu'):
                    self.log_histogram(f"params/{name}", param.cpu().detach().numpy())
        
        self.log_metric("params/total", total_params)
        self.log_metric("params/trainable", trainable_params)
    
    def log_gradients(self, model: torch.nn.Module, step: Optional[int] = None):
        """Log gradient norms."""
        total_norm = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                self.log_metric(f"gradients/{name}", grad_norm, step)
                total_norm += grad_norm ** 2
        
        self.log_metric("gradients/total_norm", total_norm ** 0.5, step)
    
    # ======================================================================
    # System Monitoring
    # ======================================================================
    
    def log_system_stats(self):
        """Log system resource usage."""
        import psutil
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.1)
        self.log_metric("system/cpu", cpu_percent)
        
        # Memory
        memory = psutil.virtual_memory()
        self.log_metric("system/memory_used", memory.used / 1024**3)  # GB
        self.log_metric("system/memory_percent", memory.percent)
        
        # Disk
        disk = psutil.disk_usage('/')
        self.log_metric("system/disk_used", disk.used / 1024**3)
        self.log_metric("system/disk_percent", disk.percent)
        
        # GPU if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self.log_metric(f"system/gpu_{i}_memory_allocated", 
                               torch.cuda.memory_allocated(i) / 1024**3)
                self.log_metric(f"system/gpu_{i}_memory_cached", 
                               torch.cuda.memory_reserved(i) / 1024**3)
    
    # ======================================================================
    # Checkpointing
    # ======================================================================
    
    def save_checkpoint(self, state: Dict, filename: str):
        """Save checkpoint."""
        path = os.path.join(self.log_dir, filename)
        torch.save(state, path)
        self.info("checkpoint", f"Saved checkpoint", {'path': path})
    
    def load_checkpoint(self, filename: str) -> Dict:
        """Load checkpoint."""
        path = os.path.join(self.log_dir, filename)
        if os.path.exists(path):
            self.info("checkpoint", f"Loading checkpoint", {'path': path})
            return torch.load(path)
        else:
            self.warning("checkpoint", f"Checkpoint not found", {'path': path})
            return {}
    
    # ======================================================================
    # Cleanup
    # ======================================================================
    
    def close(self):
        """Clean up logger."""
        self.running = False
        
        if self.async_logging:
            # Process remaining queue items
            while not self.queue.empty():
                try:
                    entry = self.queue.get_nowait()
                    self._handle_entry(entry)
                except queue.Empty:
                    break
            
            if hasattr(self, 'thread'):
                self.thread.join(timeout=2.0)
        
        # Close handlers
        for handler in self.handlers:
            if hasattr(handler, 'close'):
                try:
                    handler.close()
                except:
                    pass
        
        self.info("logger", "Logger closed")


# ======================================================================
# Global Logger Instance
# ======================================================================

_default_logger: Optional[GRA_Logger] = None


def get_logger(name: str = "gra_experiment", 
               log_dir: str = "./logs",
               level: LogLevel = LogLevel.INFO) -> GRA_Logger:
    """Get or create default logger."""
    global _default_logger
    if _default_logger is None:
        _default_logger = GRA_Logger(name, log_dir, level)
    return _default_logger


def set_logger(logger: GRA_Logger):
    """Set default logger."""
    global _default_logger
    _default_logger = logger


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    # Create logger
    logger = GRA_Logger("test_experiment", "./test_logs")
    
    # Add handlers
    logger.add_console_handler(colored=True)
    logger.add_file_handler(format='jsonl')
    
    if TENSORBOARD_AVAILABLE:
        logger.add_tensorboard_handler()
    
    # Log some messages
    logger.info("test", "Starting test")
    logger.debug("test", "Debug message", {'foo': 'bar'})
    
    # Log metrics
    for i in range(100):
        logger.log_metric("test_metric", np.sin(i * 0.1), step=i)
        logger.global_step += 1
    
    # Log histogram
    data = np.random.randn(1000)
    logger.log_histogram("test_hist", data)
    
    # Log safety incident
    logger.log_safety_incident(
        incident_id="test_001",
        violation_type="collision_imminent",
        severity=0.7,
        intervention_taken=True
    )
    
    # Log ethical dilemma
    logger.log_ethical_dilemma(
        dilemma_id="dilemma_001",
        principle="do_no_harm",
        resolution="stopped"
    )
    
    # Get statistics
    stats = logger.get_metric_stats("test_metric", window=50)
    print(f"Metric stats: {stats}")
    
    # Close
    logger.close()
    
    print("\nAll tests passed!")
```