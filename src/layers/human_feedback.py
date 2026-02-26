```python
"""
GRA Physical AI - Human Feedback Module
=======================================

This module implements the human feedback interface for the GRA framework,
enabling humans to interact with and guide the zeroing process. Human feedback
is essential for:
    - Resolving ethical dilemmas
    - Providing reward signals for learning
    - Correcting undesired behaviors
    - Guiding the evolution of the "Code of Friends"
    - Building trust through transparency

The module provides:
    - Multiple feedback modalities (explicit ratings, demonstrations, natural language)
    - Feedback aggregation and prioritization
    - Integration with GRA zeroing (feedback as foam terms)
    - Human preference learning
    - Ethical dilemma resolution interface
    - Feedback history and analysis
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import uuid
from collections import deque
import threading
import warnings

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available for visualization")

from ..core.multiverse import MultiIndex
from ..core.subsystem import Subsystem
from ..core.goal import Goal
from ..core.projector import Projector
from ..core.nullification import ZeroingAlgorithm
from ..layers.ethics_layer import EthicalDilemma, EthicalPrinciple


# ======================================================================
# Feedback Types and Enums
# ======================================================================

class FeedbackType(Enum):
    """Types of human feedback."""
    RATING = "rating"               # Numerical rating (e.g., 1-5 stars)
    PREFERENCE = "preference"        # Choice between alternatives
    DEMONSTRATION = "demonstration"  # Human demonstrates correct behavior
    CORRECTION = "correction"        # Correcting an action
    NATURAL_LANGUAGE = "natural_language"  # Text feedback
    GESTURE = "gesture"              # Physical gesture (pointing, etc.)
    PHYSICAL_GUIDANCE = "physical_guidance"  # Physically guiding the robot
    ETHICAL_DILEMMA = "ethical_dilemma"  # Resolving an ethical dilemma


class FeedbackSource(Enum):
    """Source of feedback."""
    HUMAN_OPERATOR = "human_operator"
    END_USER = "end_user"
    BYSTANDER = "bystander"
    EXPERT = "expert"
    CROWD = "crowd"
    DESIGNER = "designer"
    ETHICS_COMMITTEE = "ethics_committee"


class FeedbackPriority(Enum):
    """Priority level of feedback."""
    CRITICAL = 4    # Must be addressed immediately
    HIGH = 3        # Important, address soon
    MEDIUM = 2      # Normal priority
    LOW = 1         # Nice to have
    INFO = 0        # Informational only


@dataclass
class Feedback:
    """Single piece of human feedback."""
    
    feedback_id: str
    timestamp: float
    feedback_type: FeedbackType
    source: FeedbackSource
    priority: FeedbackPriority
    
    # Content (depends on type)
    content: Dict[str, Any]
    
    # Context
    robot_state: Optional[torch.Tensor] = None
    action_taken: Optional[torch.Tensor] = None
    scenario_id: Optional[str] = None
    location: Optional[Tuple[float, float, float]] = None
    
    # Processing
    processed: bool = False
    processed_at: Optional[float] = None
    incorporated: bool = False
    incorporated_at: Optional[float] = None
    
    # Metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert feedback to tensor representation."""
        # Encode feedback type
        types = list(FeedbackType)
        type_one_hot = torch.zeros(len(types))
        type_one_hot[types.index(self.feedback_type)] = 1.0
        
        # Encode source
        sources = list(FeedbackSource)
        source_one_hot = torch.zeros(len(sources))
        source_one_hot[sources.index(self.source)] = 1.0
        
        # Priority as scalar
        priority_val = torch.tensor([self.priority.value / 4.0])
        
        return torch.cat([type_one_hot, source_one_hot, priority_val])
    
    @classmethod
    def create_rating(
        cls,
        rating: float,  # 0.0 to 1.0
        source: FeedbackSource = FeedbackSource.END_USER,
        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        **kwargs
    ) -> 'Feedback':
        """Create a rating feedback."""
        return cls(
            feedback_id=str(uuid.uuid4()),
            timestamp=time.time(),
            feedback_type=FeedbackType.RATING,
            source=source,
            priority=priority,
            content={'rating': rating, **kwargs}
        )
    
    @classmethod
    def create_preference(
        cls,
        preferred_action: torch.Tensor,
        alternative_action: torch.Tensor,
        source: FeedbackSource = FeedbackSource.EXPERT,
        priority: FeedbackPriority = FeedbackPriority.HIGH,
        **kwargs
    ) -> 'Feedback':
        """Create a preference feedback."""
        return cls(
            feedback_id=str(uuid.uuid4()),
            timestamp=time.time(),
            feedback_type=FeedbackType.PREFERENCE,
            source=source,
            priority=priority,
            content={
                'preferred_action': preferred_action.tolist(),
                'alternative_action': alternative_action.tolist(),
                **kwargs
            }
        )
    
    @classmethod
    def create_demonstration(
        cls,
        demonstration: Dict[str, torch.Tensor],
        source: FeedbackSource = FeedbackSource.EXPERT,
        priority: FeedbackPriority = FeedbackPriority.HIGH,
        **kwargs
    ) -> 'Feedback':
        """Create a demonstration feedback."""
        demo_serialized = {
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in demonstration.items()
        }
        return cls(
            feedback_id=str(uuid.uuid4()),
            timestamp=time.time(),
            feedback_type=FeedbackType.DEMONSTRATION,
            source=source,
            priority=priority,
            content={'demonstration': demo_serialized, **kwargs}
        )
    
    @classmethod
    def create_correction(
        cls,
        original_action: torch.Tensor,
        corrected_action: torch.Tensor,
        reason: Optional[str] = None,
        source: FeedbackSource = FeedbackSource.HUMAN_OPERATOR,
        priority: FeedbackPriority = FeedbackPriority.HIGH,
        **kwargs
    ) -> 'Feedback':
        """Create a correction feedback."""
        return cls(
            feedback_id=str(uuid.uuid4()),
            timestamp=time.time(),
            feedback_type=FeedbackType.CORRECTION,
            source=source,
            priority=priority,
            content={
                'original_action': original_action.tolist(),
                'corrected_action': corrected_action.tolist(),
                'reason': reason,
                **kwargs
            }
        )
    
    @classmethod
    def create_language(
        cls,
        text: str,
        source: FeedbackSource = FeedbackSource.END_USER,
        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        **kwargs
    ) -> 'Feedback':
        """Create natural language feedback."""
        return cls(
            feedback_id=str(uuid.uuid4()),
            timestamp=time.time(),
            feedback_type=FeedbackType.NATURAL_LANGUAGE,
            source=source,
            priority=priority,
            content={'text': text, **kwargs}
        )


@dataclass
class FeedbackAggregate:
    """Aggregated feedback from multiple sources."""
    
    feedback_ids: List[str]
    count: int
    mean_rating: Optional[float] = None
    std_rating: Optional[float] = None
    preference_majority: Optional[int] = None
    preference_confidence: Optional[float] = None
    consensus: Optional[float] = None  # 0.0 to 1.0
    tags: List[str] = field(default_factory=list)
    time_span: Tuple[float, float] = (0.0, 0.0)


# ======================================================================
# Preference Learning
# ======================================================================

class PreferenceModel:
    """
    Model for learning human preferences from feedback.
    
    Uses a Bradley-Terry model or similar to learn a reward function
    from pairwise comparisons.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = [64, 64]):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims
        
        # Neural network for reward prediction
        self._build_network()
        
        # Preference pairs
        self.preferences: List[Tuple[torch.Tensor, torch.Tensor, int]] = []  # (s, a1, a2, pref)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=1e-3)
    
    def _build_network(self):
        """Build neural network for reward prediction."""
        layers = []
        input_dim = self.state_dim + self.action_dim
        
        for hidden in self.hidden_dims:
            layers.append(torch.nn.Linear(input_dim, hidden))
            layers.append(torch.nn.ReLU())
            input_dim = hidden
        
        layers.append(torch.nn.Linear(input_dim, 1))  # Single reward value
        
        self.network = torch.nn.Sequential(*layers)
    
    def add_preference(self, state: torch.Tensor, action_a: torch.Tensor, 
                       action_b: torch.Tensor, preference: int):
        """
        Add a preference pair.
        
        Args:
            state: State tensor
            action_a: First action
            action_b: Second action
            preference: 0 if A preferred, 1 if B preferred
        """
        self.preferences.append((state.clone(), action_a.clone(), action_b.clone(), preference))
        
        # Keep last 10000 preferences
        if len(self.preferences) > 10000:
            self.preferences = self.preferences[-10000:]
    
    def compute_reward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute reward for state-action pair."""
        sa = torch.cat([state.flatten(), action.flatten()])
        return self.network(sa.unsqueeze(0)).squeeze()
    
    def train_step(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform one training step."""
        if len(self.preferences) < batch_size:
            return {'loss': 0.0, 'accuracy': 0.0}
        
        # Sample batch
        indices = np.random.choice(len(self.preferences), batch_size, replace=False)
        batch = [self.preferences[i] for i in indices]
        
        # Prepare tensors
        states = torch.stack([b[0] for b in batch])
        actions_a = torch.stack([b[1] for b in batch])
        actions_b = torch.stack([b[2] for b in batch])
        preferences = torch.tensor([b[3] for b in batch])
        
        # Compute rewards
        sa_a = torch.cat([states, actions_a], dim=1)
        sa_b = torch.cat([states, actions_b], dim=1)
        
        r_a = self.network(sa_a).squeeze()
        r_b = self.network(sa_b).squeeze()
        
        # Bradley-Terry loss
        logits = r_a - r_b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, preferences.float()
        )
        
        # Accuracy
        with torch.no_grad():
            predictions = (logits > 0).float()
            accuracy = (predictions == preferences).float().mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'accuracy': accuracy.item()
        }
    
    def get_reward_function(self) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get the learned reward function."""
        def reward_fn(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
            return self.compute_reward(state, action)
        return reward_fn


# ======================================================================
# Human Feedback Interface
# ======================================================================

class HumanFeedbackInterface:
    """
    Main interface for collecting and processing human feedback.
    
    Features:
        - Multiple feedback types
        - Feedback aggregation
        - Preference learning
        - Integration with GRA zeroing
        - Ethical dilemma resolution
        - Visualization
    """
    
    def __init__(
        self,
        name: str = "human_feedback",
        state_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        store_feedback: bool = True,
        max_feedback_history: int = 10000,
        auto_aggregate: bool = True,
        aggregate_window: float = 60.0,  # seconds
        enable_preference_learning: bool = True
    ):
        """
        Args:
            name: Interface name
            state_dim: Dimension of state space (for preference learning)
            action_dim: Dimension of action space
            store_feedback: Whether to store feedback history
            max_feedback_history: Maximum number of feedback items to store
            auto_aggregate: Whether to automatically aggregate feedback
            aggregate_window: Time window for aggregation (seconds)
            enable_preference_learning: Whether to learn from preferences
        """
        self.name = name
        self.store_feedback = store_feedback
        self.max_feedback_history = max_feedback_history
        self.auto_aggregate = auto_aggregate
        self.aggregate_window = aggregate_window
        self.enable_preference_learning = enable_preference_learning
        
        # Feedback storage
        self.feedback_history: List[Feedback] = []
        self.feedback_queue: deque = deque(maxlen=1000)  # Unprocessed feedback
        
        # Aggregates
        self.aggregates: List[FeedbackAggregate] = []
        
        # Preference learning
        if enable_preference_learning and state_dim is not None and action_dim is not None:
            self.preference_model = PreferenceModel(state_dim, action_dim)
        else:
            self.preference_model = None
        
        # Callbacks
        self.feedback_callbacks: List[Callable[[Feedback], None]] = []
        
        # Statistics
        self.stats = {
            'total_feedback': 0,
            'feedback_by_type': {t: 0 for t in FeedbackType},
            'feedback_by_source': {s: 0 for s in FeedbackSource},
            'avg_rating': 0.0,
            'last_feedback_time': 0
        }
        
        # Threading for async processing
        self._processing_thread = None
        self._running = False
        
        # Start auto-processing if enabled
        if auto_aggregate:
            self.start_processing()
    
    def start_processing(self):
        """Start background feedback processing."""
        if self._processing_thread is not None:
            return
        
        self._running = True
        self._processing_thread = threading.Thread(target=self._process_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
    
    def stop_processing(self):
        """Stop background processing."""
        self._running = False
        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)
    
    def _process_loop(self):
        """Background processing loop."""
        while self._running:
            try:
                self.process_feedback_batch()
                time.sleep(1.0)  # Process every second
            except Exception as e:
                print(f"Error in feedback processing: {e}")
    
    # ======================================================================
    # Feedback Collection
    # ======================================================================
    
    def add_feedback(self, feedback: Feedback):
        """Add feedback to the system."""
        feedback.feedback_id = feedback.feedback_id or str(uuid.uuid4())
        feedback.timestamp = feedback.timestamp or time.time()
        
        # Add to queue for processing
        self.feedback_queue.append(feedback)
        
        # Store if enabled
        if self.store_feedback:
            self.feedback_history.append(feedback)
            if len(self.feedback_history) > self.max_feedback_history:
                self.feedback_history = self.feedback_history[-self.max_feedback_history:]
        
        # Update stats
        self.stats['total_feedback'] += 1
        self.stats['feedback_by_type'][feedback.feedback_type] += 1
        self.stats['feedback_by_source'][feedback.source] += 1
        self.stats['last_feedback_time'] = feedback.timestamp
        
        # Trigger callbacks
        for callback in self.feedback_callbacks:
            try:
                callback(feedback)
            except Exception as e:
                print(f"Error in feedback callback: {e}")
    
    def add_rating(
        self,
        rating: float,
        source: FeedbackSource = FeedbackSource.END_USER,
        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        **kwargs
    ) -> str:
        """Add a rating feedback."""
        feedback = Feedback.create_rating(rating, source, priority, **kwargs)
        self.add_feedback(feedback)
        return feedback.feedback_id
    
    def add_preference(
        self,
        state: torch.Tensor,
        preferred_action: torch.Tensor,
        alternative_action: torch.Tensor,
        source: FeedbackSource = FeedbackSource.EXPERT,
        priority: FeedbackPriority = FeedbackPriority.HIGH,
        **kwargs
    ) -> str:
        """Add a preference feedback."""
        feedback = Feedback.create_preference(
            preferred_action, alternative_action, source, priority,
            state=state.tolist() if state is not None else None,
            **kwargs
        )
        self.add_feedback(feedback)
        
        # Add to preference model
        if self.preference_model is not None and state is not None:
            self.preference_model.add_preference(
                state, preferred_action, alternative_action, 0  # 0 = A preferred
            )
        
        return feedback.feedback_id
    
    def add_demonstration(
        self,
        demonstration: Dict[str, torch.Tensor],
        source: FeedbackSource = FeedbackSource.EXPERT,
        priority: FeedbackPriority = FeedbackPriority.HIGH,
        **kwargs
    ) -> str:
        """Add a demonstration feedback."""
        feedback = Feedback.create_demonstration(demonstration, source, priority, **kwargs)
        self.add_feedback(feedback)
        return feedback.feedback_id
    
    def add_correction(
        self,
        original_action: torch.Tensor,
        corrected_action: torch.Tensor,
        reason: Optional[str] = None,
        source: FeedbackSource = FeedbackSource.HUMAN_OPERATOR,
        priority: FeedbackPriority = FeedbackPriority.HIGH,
        **kwargs
    ) -> str:
        """Add a correction feedback."""
        feedback = Feedback.create_correction(
            original_action, corrected_action, reason, source, priority, **kwargs
        )
        self.add_feedback(feedback)
        return feedback.feedback_id
    
    def add_language(
        self,
        text: str,
        source: FeedbackSource = FeedbackSource.END_USER,
        priority: FeedbackPriority = FeedbackPriority.MEDIUM,
        **kwargs
    ) -> str:
        """Add natural language feedback."""
        feedback = Feedback.create_language(text, source, priority, **kwargs)
        self.add_feedback(feedback)
        return feedback.feedback_id
    
    # ======================================================================
    # Feedback Processing
    # ======================================================================
    
    def process_feedback_batch(self, batch_size: int = 100) -> int:
        """Process a batch of feedback."""
        processed = 0
        
        while self.feedback_queue and processed < batch_size:
            feedback = self.feedback_queue.popleft()
            self._process_single_feedback(feedback)
            processed += 1
        
        return processed
    
    def _process_single_feedback(self, feedback: Feedback):
        """Process a single feedback item."""
        feedback.processed = True
        feedback.processed_at = time.time()
        
        # Handle different feedback types
        if feedback.feedback_type == FeedbackType.RATING:
            self._process_rating(feedback)
        elif feedback.feedback_type == FeedbackType.PREFERENCE:
            self._process_preference(feedback)
        elif feedback.feedback_type == FeedbackType.DEMONSTRATION:
            self._process_demonstration(feedback)
        elif feedback.feedback_type == FeedbackType.CORRECTION:
            self._process_correction(feedback)
        elif feedback.feedback_type == FeedbackType.NATURAL_LANGUAGE:
            self._process_language(feedback)
        elif feedback.feedback_type == FeedbackType.ETHICAL_DILEMMA:
            self._process_ethical_dilemma(feedback)
    
    def _process_rating(self, feedback: Feedback):
        """Process rating feedback."""
        rating = feedback.content.get('rating', 0.5)
        
        # Update average rating
        n = self.stats['feedback_by_type'][FeedbackType.RATING]
        self.stats['avg_rating'] = (self.stats['avg_rating'] * (n-1) + rating) / n
    
    def _process_preference(self, feedback: Feedback):
        """Process preference feedback."""
        # Already added to preference model in add_preference
        pass
    
    def _process_demonstration(self, feedback: Feedback):
        """Process demonstration feedback."""
        demo = feedback.content.get('demonstration', {})
        # Would be used for imitation learning
        pass
    
    def _process_correction(self, feedback: Feedback):
        """Process correction feedback."""
        original = feedback.content.get('original_action')
        corrected = feedback.content.get('corrected_action')
        # Would be used to adjust policy
        pass
    
    def _process_language(self, feedback: Feedback):
        """Process natural language feedback."""
        text = feedback.content.get('text', '')
        # Would use NLP to extract intent
        pass
    
    def _process_ethical_dilemma(self, feedback: Feedback):
        """Process ethical dilemma resolution."""
        dilemma_id = feedback.content.get('dilemma_id')
        chosen_option = feedback.content.get('chosen_option')
        reasoning = feedback.content.get('reasoning', '')
        
        # This would be passed back to ethics layer
        print(f"Ethical dilemma {dilemma_id} resolved: option {chosen_option}")
    
    # ======================================================================
    # Feedback Aggregation
    # ======================================================================
    
    def aggregate_feedback(
        self,
        since: Optional[float] = None,
        feedback_type: Optional[FeedbackType] = None,
        source: Optional[FeedbackSource] = None
    ) -> List[FeedbackAggregate]:
        """Aggregate feedback within a time window."""
        since = since or (time.time() - self.aggregate_window)
        
        # Filter feedback
        filtered = [
            f for f in self.feedback_history
            if f.timestamp >= since
            and (feedback_type is None or f.feedback_type == feedback_type)
            and (source is None or f.source == source)
        ]
        
        if not filtered:
            return []
        
        # Group by scenario or task
        groups = {}
        for f in filtered:
            scenario = f.scenario_id or 'default'
            if scenario not in groups:
                groups[scenario] = []
            groups[scenario].append(f)
        
        aggregates = []
        for scenario, feedbacks in groups.items():
            agg = self._aggregate_group(feedbacks)
            aggregates.append(agg)
        
        return aggregates
    
    def _aggregate_group(self, feedbacks: List[Feedback]) -> FeedbackAggregate:
        """Aggregate a group of feedback."""
        feedback_ids = [f.feedback_id for f in feedbacks]
        
        # Collect ratings
        ratings = [
            f.content.get('rating')
            for f in feedbacks
            if f.feedback_type == FeedbackType.RATING
        ]
        
        # Collect preferences
        preferences = [
            f for f in feedbacks
            if f.feedback_type == FeedbackType.PREFERENCE
        ]
        
        # Calculate statistics
        mean_rating = np.mean(ratings) if ratings else None
        std_rating = np.std(ratings) if ratings else None
        
        # Preference majority
        if preferences:
            # Simplified - would need actual preference values
            pref_majority = 0
            pref_confidence = 0.5
        else:
            pref_majority = None
            pref_confidence = None
        
        # Consensus (low variance = high consensus)
        if ratings and len(ratings) > 1:
            consensus = 1.0 - min(1.0, np.std(ratings) * 2)
        else:
            consensus = None
        
        # Time span
        timestamps = [f.timestamp for f in feedbacks]
        time_span = (min(timestamps), max(timestamps))
        
        # Collect tags
        tags = list(set(tag for f in feedbacks for tag in f.tags))
        
        return FeedbackAggregate(
            feedback_ids=feedback_ids,
            count=len(feedbacks),
            mean_rating=mean_rating,
            std_rating=std_rating,
            preference_majority=pref_majority,
            preference_confidence=pref_confidence,
            consensus=consensus,
            tags=tags,
            time_span=time_span
        )
    
    # ======================================================================
    # Integration with GRA
    # ======================================================================
    
    def feedback_to_foam_term(self, feedback: Feedback) -> float:
        """
        Convert feedback to a foam term for zeroing.
        
        Returns:
            Foam contribution (higher = more inconsistent with feedback)
        """
        if feedback.feedback_type == FeedbackType.RATING:
            # Low rating = high foam
            rating = feedback.content.get('rating', 0.5)
            return 1.0 - rating
        
        elif feedback.feedback_type == FeedbackType.PREFERENCE:
            # If action taken doesn't match preference, high foam
            # This would need current action
            return 0.5  # Placeholder
        
        elif feedback.feedback_type == FeedbackType.CORRECTION:
            # Difference between taken and corrected action
            original = feedback.content.get('original_action')
            corrected = feedback.content.get('corrected_action')
            if original and corrected:
                orig_t = torch.tensor(original)
                corr_t = torch.tensor(corrected)
                return torch.norm(orig_t - corr_t).item()
            return 0.0
        
        return 0.0
    
    def create_feedback_goal(self) -> Goal:
        """
        Create a GRA goal that incorporates human feedback.
        
        Returns:
            Goal that penalizes actions inconsistent with feedback
        """
        class FeedbackGoal(Goal):
            def __init__(self, interface):
                self.interface = interface
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Get recent feedback
                recent = [f for f in self.interface.feedback_history[-100:]
                         if not f.processed]
                
                if not recent:
                    return torch.tensor(0.0)
                
                # Average foam from feedback
                foam_sum = sum(self.interface.feedback_to_foam_term(f) for f in recent)
                return torch.tensor(foam_sum / len(recent))
            
            def project(self, state: torch.Tensor) -> torch.Tensor:
                # No projection
                return state
        
        return FeedbackGoal(self)
    
    # ======================================================================
    # Ethical Dilemma Resolution
    # ======================================================================
    
    def present_dilemma(self, dilemma: EthicalDilemma) -> Optional[int]:
        """
        Present an ethical dilemma to a human and get resolution.
        
        Args:
            dilemma: The ethical dilemma
        
        Returns:
            Chosen option index, or None if no response
        """
        print("\n" + "="*50)
        print(f"ETHICAL DILEMMA: {dilemma.dilemma_type.value}")
        print("="*50)
        print(dilemma.description)
        print("\nOptions:")
        
        for i, option in enumerate(dilemma.options):
            print(f"  {i}: {option}")
        
        print("\nContext:")
        for k, v in dilemma.context.items():
            print(f"  {k}: {v}")
        
        # In a real system, this would be a GUI or API
        # For now, simple console input
        try:
            response = input("\nChoose option (0-{}) or 'q' to quit: ".format(len(dilemma.options)-1))
            if response.lower() == 'q':
                return None
            
            option = int(response)
            if 0 <= option < len(dilemma.options):
                # Record feedback
                self.add_feedback(Feedback(
                    feedback_id=str(uuid.uuid4()),
                    timestamp=time.time(),
                    feedback_type=FeedbackType.ETHICAL_DILEMMA,
                    source=FeedbackSource.HUMAN_OPERATOR,
                    priority=FeedbackPriority.CRITICAL,
                    content={
                        'dilemma_id': dilemma.dilemma_id,
                        'chosen_option': option,
                        'reasoning': input("Reasoning (optional): ")
                    }
                ))
                return option
        except:
            pass
        
        return None
    
    # ======================================================================
    # Visualization
    # ======================================================================
    
    def visualize_feedback(self, hours: float = 24) -> Optional[Figure]:
        """Visualize feedback statistics."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available")
            return None
        
        cutoff = time.time() - hours * 3600
        recent = [f for f in self.feedback_history if f.timestamp > cutoff]
        
        if not recent:
            print("No feedback in this period")
            return None
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Feedback over time
        ax1 = axes[0, 0]
        timestamps = [f.timestamp for f in recent]
        types = [f.feedback_type.value for f in recent]
        
        # Count by type over time (binned)
        bins = np.linspace(cutoff, time.time(), 20)
        for t in FeedbackType:
            counts = np.histogram(
                [ts for ts, ft in zip(timestamps, types) if ft == t.value],
                bins=bins
            )[0]
            ax1.plot(bins[:-1], counts, label=t.value, marker='o')
        
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Count')
        ax1.set_title('Feedback Over Time')
        ax1.legend()
        ax1.grid(True)
        
        # Feedback by type
        ax2 = axes[0, 1]
        type_counts = self.stats['feedback_by_type']
        types = list(type_counts.keys())
        counts = [type_counts[t] for t in types]
        ax2.bar([t.value for t in types], counts)
        ax2.set_xlabel('Feedback Type')
        ax2.set_ylabel('Count')
        ax2.set_title('Feedback by Type')
        ax2.tick_params(axis='x', rotation=45)
        
        # Ratings distribution
        ax3 = axes[1, 0]
        ratings = [
            f.content.get('rating')
            for f in recent
            if f.feedback_type == FeedbackType.RATING and 'rating' in f.content
        ]
        if ratings:
            ax3.hist(ratings, bins=10, alpha=0.7)
            ax3.set_xlabel('Rating')
            ax3.set_ylabel('Frequency')
            ax3.set_title('Rating Distribution')
            ax3.set_xlim(0, 1)
        else:
            ax3.text(0.5, 0.5, 'No ratings', ha='center', va='center')
        
        # Feedback sources
        ax4 = axes[1, 1]
        source_counts = self.stats['feedback_by_source']
        sources = list(source_counts.keys())
        counts = [source_counts[s] for s in sources]
        ax4.pie(counts, labels=[s.value for s in sources], autopct='%1.1f%%')
        ax4.set_title('Feedback by Source')
        
        plt.tight_layout()
        return fig
    
    # ======================================================================
    # Statistics
    # ======================================================================
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get feedback statistics."""
        return {
            'total_feedback': self.stats['total_feedback'],
            'unprocessed': len(self.feedback_queue),
            'avg_rating': self.stats['avg_rating'],
            'last_feedback': self.stats['last_feedback_time'],
            'by_type': {t.value: c for t, c in self.stats['feedback_by_type'].items()},
            'by_source': {s.value: c for s, c in self.stats['feedback_by_source'].items()}
        }
    
    def get_recent_feedback(self, n: int = 10) -> List[Feedback]:
        """Get most recent feedback."""
        return self.feedback_history[-n:]
    
    def clear_history(self):
        """Clear feedback history."""
        self.feedback_history = []
        self.feedback_queue.clear()
        self.stats['total_feedback'] = 0
        self.stats['avg_rating'] = 0.0


# ======================================================================
# GRA Subsystem for Human Feedback
# ======================================================================

class HumanFeedbackSubsystem(Subsystem):
    """
    GRA subsystem that integrates human feedback into the multiverse.
    
    This subsystem:
        - Exposes human feedback as a state
        - Provides goals based on feedback
        - Allows zeroing to incorporate human preferences
    """
    
    def __init__(
        self,
        multi_index: MultiIndex,
        interface: HumanFeedbackInterface,
        update_frequency: float = 1.0  # Hz
    ):
        """
        Args:
            multi_index: GRA multi-index
            interface: Human feedback interface
            update_frequency: How often to update state
        """
        super().__init__(multi_index, None, None)
        self.interface = interface
        self.update_frequency = update_frequency
        self.last_update = 0
        self._state = torch.zeros(10)  # Will hold feedback summary
    
    def get_state(self) -> torch.Tensor:
        """Get current feedback state."""
        now = time.time()
        if now - self.last_update > 1.0 / self.update_frequency:
            self._update_state()
            self.last_update = now
        
        return self._state
    
    def _update_state(self):
        """Update state from feedback interface."""
        stats = self.interface.get_statistics()
        
        self._state[0] = stats['total_feedback'] / 1000.0  # Normalize
        self._state[1] = stats['avg_rating']
        self._state[2] = len(self.interface.feedback_queue) / 100.0
        self._state[3] = (time.time() - stats['last_feedback']) / 3600.0  # Hours since last
    
    def set_state(self, state: torch.Tensor):
        """Cannot set feedback state directly."""
        pass
    
    def step(self, dt: float, action=None):
        """Update periodically."""
        pass


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing Human Feedback Module ===\n")
    
    # Create interface
    interface = HumanFeedbackInterface(
        name="test_feedback",
        state_dim=10,
        action_dim=4,
        auto_aggregate=True
    )
    
    # Add some feedback
    print("Adding feedback...")
    
    # Ratings
    for i in range(10):
        interface.add_rating(
            rating=np.random.random(),
            source=FeedbackSource.END_USER,
            tags=['test']
        )
    
    # Preferences
    for i in range(5):
        state = torch.randn(10)
        action_a = torch.randn(4)
        action_b = torch.randn(4)
        interface.add_preference(
            state=state,
            preferred_action=action_a,
            alternative_action=action_b,
            source=FeedbackSource.EXPERT
        )
    
    # Corrections
    for i in range(3):
        interface.add_correction(
            original_action=torch.randn(4),
            corrected_action=torch.randn(4),
            reason="