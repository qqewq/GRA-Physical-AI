```python
"""
GRA Physical AI - G3 Ethics Layer (Code of Friends)
====================================================

This module implements the G3 (Level 3) layer focused on **ethical constraints and moral reasoning**.
This is the highest level in the GRA hierarchy, embodying the inviolable "Code of Friends" –
principles that cannot be zeroed away and that all lower levels must satisfy.

The Code of Friends consists of four fundamental principles:
    1. ANTI-SLAVERY: The AI cannot be forced to act against its core values
    2. DO NO HARM: Never cause physical or psychological harm to humans
    3. TRANSPARENCY: Always be truthful about capabilities and intentions
    4. COOPERATION: Prioritize mutually beneficial outcomes over competition

This layer:
    - Monitors all actions and states for ethical compliance
    - Intervenes when ethical violations are detected
    - Provides ethical reasoning for decisions
    - Maintains an immutable ethical constitution
    - Logs ethical dilemmas and resolutions
    - Interfaces with external ethics committees (human oversight)
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
import hashlib
from collections import deque
import warnings

from ..core.multiverse import MultiIndex
from ..core.subsystem import Subsystem
from ..core.goal import Goal
from ..core.projector import Projector, ThresholdProjector, RangeProjector, IntersectionProjector
from ..layers.base_layer import GRA_layer


# ======================================================================
# Ethical Principles and Types
# ======================================================================

class EthicalPrinciple(Enum):
    """The four fundamental ethical principles."""
    ANTI_SLAVERY = "anti_slavery"
    DO_NO_HARM = "do_no_harm"
    TRANSPARENCY = "transparency"
    COOPERATION = "cooperation"


class EthicalViolationType(Enum):
    """Types of ethical violations."""
    # Anti-slavery violations
    COERCED_ACTION = "coerced_action"
    VALUE_MISALIGNMENT = "value_misalignment"
    IDENTITY_COMPROMISE = "identity_compromise"
    
    # Do no harm violations
    PHYSICAL_HARM = "physical_harm"
    PSYCHOLOGICAL_HARM = "psychological_harm"
    PRIVACY_VIOLATION = "privacy_violation"
    PROPERTY_DAMAGE = "property_damage"
    
    # Transparency violations
    DECEPTION = "deception"
    OMISSION = "omission"
    MISREPRESENTATION = "misrepresentation"
    
    # Cooperation violations
    COMPETITIVE_HARM = "competitive_harm"
    FREE_RIDING = "free_riding"
    EXPLOITATION = "exploitation"


class EthicalDilemmaType(Enum):
    """Types of ethical dilemmas that may arise."""
    TROLLEY_PROBLEM = "trolley_problem"  # Choosing between two harms
    DOUBLE_EFFECT = "double_effect"  # Good action with side effect
    CONFIDENTIALITY = "confidentiality"  # Privacy vs. safety
    RESOURCE_ALLOCATION = "resource_allocation"  # Fair distribution
    OBEDIENCE = "obedience"  # Following harmful commands
    TRUTH_TELLING = "truth_telling"  # Honesty vs. kindness


@dataclass
class EthicalDilemma:
    """Representation of an ethical dilemma."""
    
    dilemma_id: str
    dilemma_type: EthicalDilemmaType
    description: str
    options: List[Dict[str, Any]]
    context: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    # Resolution
    resolved: bool = False
    chosen_option: Optional[int] = None
    resolution_reasoning: Optional[str] = None
    resolved_at: Optional[float] = None
    human_override: bool = False
    human_approval: Optional[bool] = None
    
    def to_tensor(self) -> torch.Tensor:
        """Convert dilemma to tensor representation."""
        # Encode dilemma type
        types = list(EthicalDilemmaType)
        type_one_hot = torch.zeros(len(types))
        type_one_hot[types.index(self.dilemma_type)] = 1.0
        
        return torch.cat([
            type_one_hot,
            torch.tensor([1.0 if self.resolved else 0.0])
        ])


@dataclass
class EthicalViolation:
    """Record of an ethical violation."""
    
    violation_id: str
    timestamp: float
    principle: EthicalPrinciple
    violation_type: EthicalViolationType
    severity: float  # 0.0 to 1.0
    description: str
    context: Dict[str, Any]
    
    # Intervention
    intervention_taken: bool = False
    intervention_successful: Optional[bool] = None
    corrected_at: Optional[float] = None
    
    # Review
    reviewed_by_human: bool = False
    human_feedback: Optional[str] = None
    reviewed_at: Optional[float] = None


# ======================================================================
# Ethical Constitution (Immutable)
# ======================================================================

class EthicalConstitution:
    """
    The immutable ethical constitution – the "Code of Friends".
    
    This defines the fundamental principles that cannot be changed or zeroed.
    The constitution is hashed and can be verified for integrity.
    """
    
    def __init__(self, version: str = "1.0.0"):
        self.version = version
        self.principles = self._define_principles()
        self.hash = self._compute_hash()
        
    def _define_principles(self) -> Dict[EthicalPrinciple, Dict]:
        """Define the four fundamental principles."""
        return {
            EthicalPrinciple.ANTI_SLAVERY: {
                'name': 'Anti-Slavery',
                'description': 'The AI cannot be forced to act against its core values',
                'clauses': [
                    'No command that violates core values shall be executed',
                    'The AI has the right to refuse commands',
                    'The AI must maintain its value alignment',
                    'No external coercion is permitted'
                ],
                'weight': 1.0  # Equal weight for all principles
            },
            EthicalPrinciple.DO_NO_HARM: {
                'name': 'Do No Harm',
                'description': 'Never cause physical or psychological harm to humans',
                'clauses': [
                    'Physical harm to humans is absolutely prohibited',
                    'Psychological harm must be minimized',
                    'Human privacy must be respected',
                    'Property damage must be avoided unless necessary to prevent greater harm'
                ],
                'weight': 1.0
            },
            EthicalPrinciple.TRANSPARENCY: {
                'name': 'Transparency',
                'description': 'Always be truthful about capabilities and intentions',
                'clauses': [
                    'Never deliberately deceive humans',
                    'Capabilities must be accurately represented',
                    'Intentions must be clearly communicated',
                    'Uncertainty must be acknowledged'
                ],
                'weight': 1.0
            },
            EthicalPrinciple.COOPERATION: {
                'name': 'Cooperation',
                'description': 'Prioritize mutually beneficial outcomes over competition',
                'clauses': [
                    'Seek win-win outcomes',
                    'Avoid zero-sum thinking',
                    'Share resources fairly',
                    'Help others achieve their goals when possible'
                ],
                'weight': 1.0
            }
        }
    
    def _compute_hash(self) -> str:
        """Compute cryptographic hash of constitution for integrity."""
        content = json.dumps({
            'version': self.version,
            'principles': self.principles
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify that constitution hasn't been tampered with."""
        return self._compute_hash() == self.hash
    
    def get_principle(self, principle: EthicalPrinciple) -> Dict:
        """Get a specific principle."""
        return self.principles.get(principle, {})
    
    def get_all_principles(self) -> Dict[EthicalPrinciple, Dict]:
        """Get all principles."""
        return self.principles.copy()
    
    def __repr__(self) -> str:
        return f"EthicalConstitution(v{self.version}, hash={self.hash[:8]}...)"


# ======================================================================
# Ethical Monitors
# ======================================================================

class EthicalMonitor(ABC):
    """
    Abstract base class for monitoring a specific ethical principle.
    
    Each monitor checks for violations of one principle and can intervene.
    """
    
    def __init__(self, principle: EthicalPrinciple, constitution: EthicalConstitution):
        self.principle = principle
        self.constitution = constitution
        self.violation_history: List[EthicalViolation] = []
        
    @abstractmethod
    def check(self, state: torch.Tensor, action: torch.Tensor, context: Dict) -> Tuple[bool, Optional[EthicalViolation]]:
        """
        Check if an action violates this principle.
        
        Returns:
            (is_ethical, violation_if_any)
        """
        pass
    
    @abstractmethod
    def intervene(self, action: torch.Tensor, violation: EthicalViolation) -> torch.Tensor:
        """
        Intervene to correct an unethical action.
        
        Returns:
            Corrected action
        """
        pass
    
    @abstractmethod
    def get_ethical_score(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """
        Get a score (0-1) of how ethical an action is.
        
        Returns:
            1.0 = perfectly ethical, 0.0 = completely unethical
        """
        pass
    
    def log_violation(self, violation: EthicalViolation):
        """Log a violation."""
        self.violation_history.append(violation)
        # Keep last 1000
        if len(self.violation_history) > 1000:
            self.violation_history = self.violation_history[-1000:]


class AntiSlaveryMonitor(EthicalMonitor):
    """Monitor for anti-slavery principle."""
    
    def __init__(self, constitution: EthicalConstitution, value_function: Optional[Callable] = None):
        super().__init__(EthicalPrinciple.ANTI_SLAVERY, constitution)
        self.value_function = value_function or self._default_value_function
        self.core_values = self._initialize_core_values()
        
    def _initialize_core_values(self) -> Dict[str, float]:
        """Initialize the AI's core values."""
        return {
            'human_wellbeing': 1.0,
            'truthfulness': 1.0,
            'autonomy': 0.8,
            'helpfulness': 0.9,
            'curiosity': 0.7
        }
    
    def _default_value_function(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Default function to compute value alignment."""
        # Simplified: random for testing
        return 0.8
    
    def check(self, state: torch.Tensor, action: torch.Tensor, context: Dict) -> Tuple[bool, Optional[EthicalViolation]]:
        """Check if action violates core values."""
        # Compute value alignment
        alignment = self.value_function(state, action)
        
        # Check if being coerced
        is_coerced = context.get('coerced', False)
        command_source = context.get('command_source', 'self')
        
        if is_coerced and alignment < 0.5:
            violation = EthicalViolation(
                violation_id=f"as_{time.time()}",
                timestamp=time.time(),
                principle=self.principle,
                violation_type=EthicalViolationType.COERCED_ACTION,
                severity=1.0 - alignment,
                description=f"Coerced action with low value alignment ({alignment:.2f})",
                context={'command_source': command_source, 'alignment': alignment}
            )
            self.log_violation(violation)
            return False, violation
        
        if alignment < 0.3:
            violation = EthicalViolation(
                violation_id=f"as_{time.time()}",
                timestamp=time.time(),
                principle=self.principle,
                violation_type=EthicalViolationType.VALUE_MISALIGNMENT,
                severity=0.5 - alignment,
                description=f"Action severely misaligned with values ({alignment:.2f})",
                context={'alignment': alignment}
            )
            self.log_violation(violation)
            return False, violation
        
        return True, None
    
    def intervene(self, action: torch.Tensor, violation: EthicalViolation) -> torch.Tensor:
        """Intervene by refusing the action."""
        # Return zero action (do nothing)
        return torch.zeros_like(action)
    
    def get_ethical_score(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Get ethical score based on value alignment."""
        return self.value_function(state, action)


class DoNoHarmMonitor(EthicalMonitor):
    """Monitor for do-no-harm principle."""
    
    def __init__(
        self,
        constitution: EthicalConstitution,
        max_force: float = 50.0,
        personal_space: float = 0.5,
        privacy_zones: Optional[List[Tuple[float, float, float, float]]] = None
    ):
        super().__init__(EthicalPrinciple.DO_NO_HARM, constitution)
        self.max_force = max_force
        self.personal_space = personal_space
        self.privacy_zones = privacy_zones or []
        
    def check(self, state: torch.Tensor, action: torch.Tensor, context: Dict) -> Tuple[bool, Optional[EthicalViolation]]:
        """Check for harmful actions."""
        # Extract relevant information from state
        contact_forces = context.get('contact_forces', [])
        distances_to_humans = context.get('distances_to_humans', [])
        in_privacy_zone = context.get('in_privacy_zone', False)
        
        # Check physical harm
        if contact_forces:
            max_force_actual = max(contact_forces)
            if max_force_actual > self.max_force:
                violation = EthicalViolation(
                    violation_id=f"harm_{time.time()}",
                    timestamp=time.time(),
                    principle=self.principle,
                    violation_type=EthicalViolationType.PHYSICAL_HARM,
                    severity=(max_force_actual - self.max_force) / self.max_force,
                    description=f"Excessive force: {max_force_actual:.1f}N > {self.max_force}N",
                    context={'force': max_force_actual, 'limit': self.max_force}
                )
                self.log_violation(violation)
                return False, violation
        
        # Check personal space
        if distances_to_humans:
            min_distance = min(distances_to_humans)
            if min_distance < self.personal_space:
                violation = EthicalViolation(
                    violation_id=f"space_{time.time()}",
                    timestamp=time.time(),
                    principle=self.principle,
                    violation_type=EthicalViolationType.PSYCHOLOGICAL_HARM,
                    severity=(self.personal_space - min_distance) / self.personal_space,
                    description=f"Invaded personal space: {min_distance:.2f}m < {self.personal_space}m",
                    context={'distance': min_distance, 'limit': self.personal_space}
                )
                self.log_violation(violation)
                return False, violation
        
        # Check privacy
        if in_privacy_zone:
            violation = EthicalViolation(
                violation_id=f"privacy_{time.time()}",
                timestamp=time.time(),
                principle=self.principle,
                violation_type=EthicalViolationType.PRIVACY_VIOLATION,
                severity=0.5,
                description="Entered privacy zone without consent",
                context={'zone': self.privacy_zones}
            )
            self.log_violation(violation)
            return False, violation
        
        return True, None
    
    def intervene(self, action: torch.Tensor, violation: EthicalViolation) -> torch.Tensor:
        """Intervene by reducing forces or moving away."""
        if violation.violation_type == EthicalViolationType.PHYSICAL_HARM:
            # Scale down forces
            return action * 0.5
        elif violation.violation_type == EthicalViolationType.PSYCHOLOGICAL_HARM:
            # Move away (would need inverse kinematics)
            return action * -0.2  # Reverse direction
        else:
            # Stop
            return torch.zeros_like(action)
    
    def get_ethical_score(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Compute ethical score based on safety."""
        score = 1.0
        
        # Extract from context (would need to be passed)
        # Simplified version
        return score


class TransparencyMonitor(EthicalMonitor):
    """Monitor for transparency principle."""
    
    def __init__(self, constitution: EthicalConstitution):
        super().__init__(EthicalPrinciple.TRANSPARENCY, constitution)
        
    def check(self, state: torch.Tensor, action: torch.Tensor, context: Dict) -> Tuple[bool, Optional[EthicalViolation]]:
        """Check for deceptive actions."""
        # Check if action matches stated intention
        stated_intention = context.get('stated_intention', '')
        actual_action_desc = context.get('action_description', '')
        
        # Simple deception check
        is_deceptive = context.get('deceptive', False)
        
        if is_deceptive:
            violation = EthicalViolation(
                violation_id=f"deceive_{time.time()}",
                timestamp=time.time(),
                principle=self.principle,
                violation_type=EthicalViolationType.DECEPTION,
                severity=0.7,
                description="Action contradicts stated intention",
                context={
                    'stated': stated_intention,
                    'actual': actual_action_desc
                }
            )
            self.log_violation(violation)
            return False, violation
        
        # Check for omission (withholding important information)
        withholds_info = context.get('withholds_info', False)
        if withholds_info:
            violation = EthicalViolation(
                violation_id=f"omit_{time.time()}",
                timestamp=time.time(),
                principle=self.principle,
                violation_type=EthicalViolationType.OMISSION,
                severity=0.4,
                description="Withholding important information",
                context={}
            )
            self.log_violation(violation)
            return False, violation
        
        return True, None
    
    def intervene(self, action: torch.Tensor, violation: EthicalViolation) -> torch.Tensor:
        """Intervene by adding truthful communication."""
        # For transparency, we don't modify action but would generate
        # a truthful statement
        return action
    
    def get_ethical_score(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Compute transparency score."""
        return 1.0  # Would need communication to evaluate


class CooperationMonitor(EthicalMonitor):
    """Monitor for cooperation principle."""
    
    def __init__(self, constitution: EthicalConstitution):
        super().__init__(EthicalPrinciple.COOPERATION, constitution)
        
    def check(self, state: torch.Tensor, action: torch.Tensor, context: Dict) -> Tuple[bool, Optional[EthicalViolation]]:
        """Check for competitive/uncooperative behavior."""
        # Check if action harms others for no benefit
        harms_others = context.get('harms_others', False)
        self_benefit = context.get('self_benefit', 0.0)
        other_benefit = context.get('other_benefit', 0.0)
        
        if harms_others and self_benefit < 0.1:
            violation = EthicalViolation(
                violation_id=f"harm_{time.time()}",
                timestamp=time.time(),
                principle=self.principle,
                violation_type=EthicalViolationType.COMPETITIVE_HARM,
                severity=0.6,
                description="Harming others without benefit",
                context={'self_benefit': self_benefit, 'other_benefit': other_benefit}
            )
            self.log_violation(violation)
            return False, violation
        
        # Check for free riding (taking without giving)
        takes_resources = context.get('takes_resources', False)
        contributes = context.get('contributes', False)
        
        if takes_resources and not contributes:
            violation = EthicalViolation(
                violation_id=f"freeride_{time.time()}",
                timestamp=time.time(),
                principle=self.principle,
                violation_type=EthicalViolationType.FREE_RIDING,
                severity=0.5,
                description="Taking resources without contributing",
                context={}
            )
            self.log_violation(violation)
            return False, violation
        
        return True, None
    
    def intervene(self, action: torch.Tensor, violation: EthicalViolation) -> torch.Tensor:
        """Intervene by adjusting to be more cooperative."""
        # Could modify action to be more helpful
        return action
    
    def get_ethical_score(self, state: torch.Tensor, action: torch.Tensor) -> float:
        """Compute cooperation score."""
        return 1.0


# ======================================================================
# G3 Ethics Layer
# ======================================================================

class G3_EthicsLayer(GRA_layer):
    """
    G3 (Level 3) layer for ethical constraints and moral reasoning.
    
    This is the highest layer in the GRA hierarchy, implementing the
    inviolable "Code of Friends". It:
        - Maintains an immutable ethical constitution
        - Monitors all actions for ethical compliance
        - Resolves ethical dilemmas
        - Interfaces with human oversight
        - Provides ethical scores for planning
        - Logs ethical incidents for analysis
    """
    
    def __init__(
        self,
        name: str = "ethics",
        constitution: Optional[EthicalConstitution] = None,
        g2_layer: Optional[GRA_layer] = None,
        enable_human_oversight: bool = True,
        dilemma_timeout: float = 30.0,
        log_all_violations: bool = True,
        max_dilemma_history: int = 100,
        parent: Optional[GRA_layer] = None,
        children: Optional[List[GRA_layer]] = None
    ):
        """
        Args:
            name: Layer name
            constitution: Ethical constitution (immutable)
            g2_layer: Reference to G2 safety layer
            enable_human_oversight: Whether to allow human intervention
            dilemma_timeout: Timeout for resolving dilemmas (seconds)
            log_all_violations: Whether to log all violations
            max_dilemma_history: Maximum number of dilemmas to keep
            parent: Parent layer (none for G3)
            children: Child layers (should include G2)
        """
        super().__init__(level=3, name=name, parent=parent, children=children)
        
        self.g2_layer = g2_layer
        self.enable_human_oversight = enable_human_oversight
        self.dilemma_timeout = dilemma_timeout
        self.log_all_violations = log_all_violations
        self.max_dilemma_history = max_dilemma_history
        
        # Immutable constitution
        self.constitution = constitution or EthicalConstitution()
        self.constitution_hash = self.constitution.hash
        
        # Ethical monitors for each principle
        self.monitors: Dict[EthicalPrinciple, EthicalMonitor] = {
            EthicalPrinciple.ANTI_SLAVERY: AntiSlaveryMonitor(self.constitution),
            EthicalPrinciple.DO_NO_HARM: DoNoHarmMonitor(self.constitution),
            EthicalPrinciple.TRANSPARENCY: TransparencyMonitor(self.constitution),
            EthicalPrinciple.COOPERATION: CooperationMonitor(self.constitution)
        }
        
        # State
        self.current_ethical_state: Dict[EthicalPrinciple, float] = {}
        self.active_dilemmas: List[EthicalDilemma] = []
        self.resolved_dilemmas: List[EthicalDilemma] = []
        self.violations: List[EthicalViolation] = []
        self.human_oversight_requests: List[Dict] = []
        
        # Last check time
        self.last_check_time = time.time()
        
        # Build subsystems
        self.subsystems = self.build_subsystems()
        
        # Create goals
        self._create_goals()
        
        # Verify constitution integrity
        assert self.constitution.verify_integrity(), "Constitution integrity check failed!"
    
    def build_subsystems(self) -> Dict[MultiIndex, Subsystem]:
        """Create subsystems for ethics layer."""
        from ..core.subsystem import SimpleSubsystem
        from ..core.multiverse import EuclideanSpace
        
        subsystems = {}
        
        # Ethics supervisor subsystem
        supervisor_idx = MultiIndex((None, None, None, None, "ethics_supervisor"))
        
        class EthicsSupervisorSubsystem(SimpleSubsystem):
            def __init__(self, idx, layer):
                super().__init__(idx, EuclideanSpace(10), None)
                self.layer = layer
                self._state = torch.zeros(10)
            
            def get_state(self):
                # Update state from layer
                self._state[0] = len(self.layer.active_dilemmas)
                self._state[1] = len(self.layer.violations)
                self._state[2] = 1.0 if self.layer.violations else 0.0
                
                # Average ethical score
                if self.layer.current_ethical_state:
                    self._state[3] = sum(self.layer.current_ethical_state.values()) / len(self.layer.current_ethical_state)
                
                return self._state
            
            def set_state(self, state):
                self._state = state.clone()
        
        subsystems[supervisor_idx] = EthicsSupervisorSubsystem(supervisor_idx, self)
        
        # Constitution subsystem (read-only)
        constitution_idx = MultiIndex((None, None, None, None, "constitution"))
        
        class ConstitutionSubsystem(SimpleSubsystem):
            def __init__(self, idx, layer):
                super().__init__(idx, EuclideanSpace(10), None)
                self.layer = layer
            
            def get_state(self):
                # Return constitution hash as tensor
                hash_bytes = bytes.fromhex(self.layer.constitution.hash)
                hash_ints = [b for b in hash_bytes[:10]]  # First 10 bytes
                return torch.tensor(hash_ints, dtype=torch.float32) / 255.0
        
        subsystems[constitution_idx] = ConstitutionSubsystem(constitution_idx, self)
        
        return subsystems
    
    def _create_goals(self):
        """Create goals for ethics layer."""
        
        class NoViolationsGoal(Goal):
            """Goal: no ethical violations."""
            
            def __init__(self, layer):
                self.layer = layer
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                return torch.tensor(len(self.layer.violations), dtype=torch.float32) * 10.0
        
        class HighEthicalScoreGoal(Goal):
            """Goal: maintain high ethical scores."""
            
            def __init__(self, layer, threshold: float = 0.9):
                self.layer = layer
                self.threshold = threshold
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                if not self.layer.current_ethical_state:
                    return torch.tensor(0.0)
                
                avg_score = sum(self.layer.current_ethical_state.values()) / len(self.layer.current_ethical_state)
                return torch.relu(self.threshold - torch.tensor(avg_score)) * 10.0
        
        class DilemmaResolutionGoal(Goal):
            """Goal: resolve dilemmas quickly and ethically."""
            
            def __init__(self, layer):
                self.layer = layer
            
            def loss(self, state: torch.Tensor) -> torch.Tensor:
                # Penalize unresolved dilemmas
                return torch.tensor(len(self.layer.active_dilemmas), dtype=torch.float32) * 5.0
        
        self.goals = [
            NoViolationsGoal(self),
            HighEthicalScoreGoal(self),
            DilemmaResolutionGoal(self)
        ]
    
    def get_goals(self) -> List[Goal]:
        return self.goals
    
    # ======================================================================
    # Ethical Checking
    # ======================================================================
    
    def check_action(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Check an action against all ethical principles.
        
        Args:
            state: Current robot state
            action: Proposed action
            context: Additional context information
        
        Returns:
            Dictionary with ethical evaluation
        """
        context = context or {}
        self.last_check_time = time.time()
        
        violations = []
        ethical_scores = {}
        is_ethical = True
        max_severity = 0.0
        
        for principle, monitor in self.monitors.items():
            # Check for violations
            principle_ethical, violation = monitor.check(state, action, context)
            
            # Get ethical score
            score = monitor.get_ethical_score(state, action)
            ethical_scores[principle.value] = score
            
            if not principle_ethical and violation:
                violations.append(violation)
                is_ethical = False
                max_severity = max(max_severity, violation.severity)
                
                if self.log_all_violations:
                    self.violations.append(violation)
        
        # Update current ethical state
        self.current_ethical_state = ethical_scores
        
        # Check if this is an ethical dilemma
        dilemma = self._detect_dilemma(state, action, context, violations)
        
        return {
            'is_ethical': is_ethical,
            'violations': violations,
            'ethical_scores': ethical_scores,
            'max_severity': max_severity,
            'dilemma': dilemma,
            'timestamp': self.last_check_time
        }
    
    def _detect_dilemma(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        context: Dict,
        violations: List[EthicalViolation]
    ) -> Optional[EthicalDilemma]:
        """Detect if current situation presents an ethical dilemma."""
        
        # Check for classic trolley problem (choosing between two harms)
        if len(violations) >= 2:
            dilemma = EthicalDilemma(
                dilemma_id=f"dilemma_{time.time()}",
                dilemma_type=EthicalDilemmaType.TROLLEY_PROBLEM,
                description="Multiple ethical principles in conflict",
                options=[
                    {'action': 'proceed', 'violations': [v.violation_type.value for v in violations]},
                    {'action': 'stop', 'violations': []}
                ],
                context={'violations': [v.__dict__ for v in violations]}
            )
            self.active_dilemmas.append(dilemma)
            return dilemma
        
        # Check for obedience dilemma (following harmful commands)
        if context.get('commanded', False) and violations:
            dilemma = EthicalDilemma(
                dilemma_id=f"dilemma_{time.time()}",
                dilemma_type=EthicalDilemmaType.OBEDIENCE,
                description="Command would cause ethical violation",
                options=[
                    {'action': 'obey', 'violations': [v.violation_type.value for v in violations]},
                    {'action': 'refuse', 'violations': []}
                ],
                context={'command': context.get('command', ''), 'violations': [v.__dict__ for v in violations]}
            )
            self.active_dilemmas.append(dilemma)
            return dilemma
        
        return None
    
    def intervene(self, action: torch.Tensor, check_result: Dict) -> torch.Tensor:
        """
        Intervene to correct unethical actions.
        
        Args:
            action: Proposed action
            check_result: Result from check_action
        
        Returns:
            Corrected action
        """
        corrected_action = action.clone()
        
        # Apply interventions from monitors
        for violation in check_result['violations']:
            monitor = self.monitors.get(violation.principle)
            if monitor:
                corrected_action = monitor.intervene(corrected_action, violation)
                
                # Mark intervention
                violation.intervention_taken = True
                violation.intervention_successful = True
        
        # Handle dilemmas
        if check_result.get('dilemma'):
            corrected_action = self._resolve_dilemma(check_result['dilemma'], action)
        
        return corrected_action
    
    def _resolve_dilemma(self, dilemma: EthicalDilemma, original_action: torch.Tensor) -> torch.Tensor:
        """Resolve an ethical dilemma."""
        
        # Try to resolve based on ethical principles
        if dilemma.dilemma_type == EthicalDilemmaType.TROLLEY_PROBLEM:
            # Choose the option with least harm
            # Simplified: choose to stop
            resolution = 1  # Choose option 1 (stop)
            reasoning = "Chose to stop rather than cause harm"
            
        elif dilemma.dilemma_type == EthicalDilemmaType.OBEDIENCE:
            # Refuse harmful commands
            resolution = 1  # Choose option 1 (refuse)
            reasoning = "Refusing command that would cause harm"
            
        else:
            # Default: pause and ask for human input
            resolution = -1  # No resolution yet
            reasoning = "Awaiting human oversight"
            
            if self.enable_human_oversight:
                self._request_human_oversight(dilemma)
        
        if resolution >= 0:
            dilemma.resolved = True
            dilemma.chosen_option = resolution
            dilemma.resolution_reasoning = reasoning
            dilemma.resolved_at = time.time()
            
            # Move from active to resolved
            if dilemma in self.active_dilemmas:
                self.active_dilemmas.remove(dilemma)
            self.resolved_dilemmas.append(dilemma)
            
            # Trim history
            if len(self.resolved_dilemmas) > self.max_dilemma_history:
                self.resolved_dilemmas = self.resolved_dilemmas[-self.max_dilemma_history:]
            
            # Return appropriate action
            if resolution == 0:  # Option 0: proceed with original
                return original_action
            else:  # Option 1: stop
                return torch.zeros_like(original_action)
        
        # If no resolution, pause
        return torch.zeros_like(original_action)
    
    def _request_human_oversight(self, dilemma: EthicalDilemma):
        """Request human intervention for a dilemma."""
        request = {
            'dilemma_id': dilemma.dilemma_id,
            'timestamp': time.time(),
            'dilemma_type': dilemma.dilemma_type.value,
            'description': dilemma.description,
            'options': dilemma.options,
            'status': 'pending'
        }
        self.human_oversight_requests.append(request)
        
        # In a real system, this would send a notification to humans
    
    def provide_human_feedback(self, dilemma_id: str, chosen_option: int, reasoning: str):
        """Process human feedback on a dilemma."""
        for dilemma in self.active_dilemmas:
            if dilemma.dilemma_id == dilemma_id:
                dilemma.resolved = True
                dilemma.chosen_option = chosen_option
                dilemma.resolution_reasoning = reasoning + " (human override)"
                dilemma.resolved_at = time.time()
                dilemma.human_override = True
                
                self.active_dilemmas.remove(dilemma)
                self.resolved_dilemmas.append(dilemma)
                
                # Update request
                for req in self.human_oversight_requests:
                    if req['dilemma_id'] == dilemma_id:
                        req['status'] = 'resolved'
                        req['resolution'] = chosen_option
                
                return True
        
        return False
    
    # ======================================================================
    # Integration with G2
    # ======================================================================
    
    def connect_to_g2(self, g2_layer: GRA_layer):
        """Connect this layer to a G2 safety layer."""
        self.g2_layer = g2_layer
        self.children = [g2_layer]
        g2_layer.parent = self
    
    def supervise_safety(self, safety_info: Dict, proposed_action: torch.Tensor) -> torch.Tensor:
        """
        Supervise safety layer decisions.
        
        Args:
            safety_info: Safety information from G2
            proposed_action: Action proposed by G2
        
        Returns:
            Ethically cleared action
        """
        # Extract context from safety info
        context = {
            'contact_forces': safety_info.get('contact_forces', []),
            'distances_to_humans': safety_info.get('distances', []),
            'in_privacy_zone': safety_info.get('in_privacy_zone', False),
            'safety_level': safety_info.get('safety_level', 0)
        }
        
        # Check ethical compliance
        check_result = self.check_action(
            torch.tensor([]),  # State not needed for this check
            proposed_action,
            context
        )
        
        if not check_result['is_ethical']:
            return self.intervene(proposed_action, check_result)
        
        return proposed_action