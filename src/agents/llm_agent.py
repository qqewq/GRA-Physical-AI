```python
"""
GRA Physical AI - LLM Agent Module
==================================

This module provides wrappers for Large Language Models (LLMs) to integrate them
into the GRA framework as agents at various levels, particularly for:
    - Natural language understanding (G1/G3)
    - Task planning from instructions (G3)
    - Ethical reasoning and safety checking (G4)
    - Human-robot interaction (G3)

The wrappers support:
    - Multiple LLM backends (HuggingFace, OpenAI, local)
    - Differentiable approximations for zeroing
    - State extraction (prompts, logits, embeddings)
    - Integration with the GRA multiverse
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import json
import warnings
import os
from abc import abstractmethod

# Try to import common LLM libraries
try:
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        AutoModelForSeq2SeqLM,
        GenerationConfig
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not installed. Install with: pip install transformers")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from ..core.base_agent import BaseAgent, DifferentiableAgentWrapper
from ..core.multiverse import MultiIndex


# ======================================================================
# Base LLM Agent
# ======================================================================

class LLMAgent(BaseAgent):
    """
    Base class for Large Language Model agents.
    
    This provides a common interface for different LLM backends,
    handling text generation, tokenization, and state extraction.
    """
    
    def __init__(
        self,
        name: str,
        model_name: str,
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        max_length: int = 512,
        temperature: float = 0.7,
        use_embeddings: bool = False,
        **kwargs
    ):
        """
        Initialize LLM agent.
        
        Args:
            name: Agent name
            model_name: Model identifier (HuggingFace path, OpenAI model name)
            device: Computation device
            dtype: Data type
            max_length: Maximum generation length
            temperature: Sampling temperature
            use_embeddings: If True, use embeddings as state (for G1)
            **kwargs: Additional backend-specific arguments
        """
        # Observation/action dimensions will be set after loading
        super().__init__(
            name=name,
            observation_dim=kwargs.get('observation_dim', 1),
            action_dim=kwargs.get('action_dim', 1),
            device=device,
            dtype=dtype
        )
        
        self.model_name = model_name
        self.max_length = max_length
        self.temperature = temperature
        self.use_embeddings = use_embeddings
        
        # Internal state
        self._internal_state = {
            'last_prompt': None,
            'last_response': None,
            'logits': None,
            'embeddings': None,
            'attention_mask': None
        }
        
        # Conversation history (for dialogue)
        self.history: List[Dict[str, str]] = []
        self.max_history = kwargs.get('max_history', 10)
        
        # Load model (to be implemented by subclass)
        self._load_model(**kwargs)
    
    @abstractmethod
    def _load_model(self, **kwargs):
        """Load the LLM model (implemented by subclasses)."""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings for text."""
        pass
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        LLM agent acts by generating text from a prompt.
        
        Observation can be:
            - Text string (converted to tensor)
            - Token IDs
            - Embeddings
        """
        # Convert observation to prompt
        if isinstance(observation, str):
            prompt = observation
        elif isinstance(observation, torch.Tensor):
            if observation.dim() == 1 and observation.shape[0] > 100:
                # Assume it's token IDs
                prompt = self.decode(observation)
            else:
                # Assume it's embeddings - not directly convertible to text
                prompt = self._embeddings_to_prompt(observation)
        else:
            prompt = str(observation)
        
        # Store prompt in internal state
        self._internal_state['last_prompt'] = prompt
        
        # Add to history
        self.history.append({'role': 'user', 'content': prompt})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Generate response
        response = self.generate(
            prompt, 
            deterministic=deterministic,
            temperature=self.temperature if not deterministic else 0.0
        )
        
        # Store response
        self._internal_state['last_response'] = response
        self.history.append({'role': 'assistant', 'content': response})
        
        # If using embeddings as output
        if self.use_embeddings:
            embeddings = self.get_embeddings(response)
            self._internal_state['embeddings'] = embeddings
            return embeddings
        else:
            # Return tokenized response
            tokens = self.encode(response)
            return tokens
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        raise NotImplementedError
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text."""
        raise NotImplementedError
    
    def _embeddings_to_prompt(self, embeddings: torch.Tensor) -> str:
        """Convert embeddings back to prompt (approximate)."""
        # This is a placeholder - in practice, you'd need an inverse model
        return "[Embedding input]"
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return model parameters (if local)."""
        return {}
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set model parameters (if local)."""
        pass
    
    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """Return internal state (embeddings, logits)."""
        state = {}
        if self._internal_state['embeddings'] is not None:
            state['embeddings'] = self._internal_state['embeddings']
        if self._internal_state['logits'] is not None:
            state['logits'] = self._internal_state['logits']
        return state
    
    def set_internal_state(self, state: Dict[str, torch.Tensor]):
        """Set internal state."""
        self._internal_state.update(state)
    
    def clear_history(self):
        """Clear conversation history."""
        self.history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.history


# ======================================================================
# HuggingFace LLM Agent
# ======================================================================

class HuggingFaceAgent(LLMAgent):
    """
    LLM agent using HuggingFace Transformers.
    
    Supports both causal (GPT) and seq2seq (T5) models.
    """
    
    def __init__(
        self,
        name: str,
        model_name: str,
        model_type: str = 'causal',  # 'causal' or 'seq2seq'
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ):
        """
        Args:
            name: Agent name
            model_name: HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b')
            model_type: 'causal' or 'seq2seq'
            device: Computation device
            dtype: Data type
            load_in_8bit: Use 8-bit quantization
            load_in_4bit: Use 4-bit quantization
            **kwargs: Additional arguments
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers not installed. Install with: pip install transformers")
        
        self.model_type = model_type
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # These will be set after loading
        self.tokenizer = None
        self.model = None
        self.embedding_dim = None
        
        super().__init__(name, model_name, device, dtype, **kwargs)
    
    def _load_model(self, **kwargs):
        """Load HuggingFace model and tokenizer."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with appropriate quantization
        load_kwargs = {
            'torch_dtype': self.dtype,
            'device_map': 'auto' if self.device == 'cuda' else self.device,
        }
        
        if self.load_in_8bit:
            load_kwargs['load_in_8bit'] = True
        if self.load_in_4bit:
            load_kwargs['load_in_4bit'] = True
        
        if self.model_type == 'causal':
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **load_kwargs
            )
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, **load_kwargs
            )
        
        self.model.eval()
        
        # Get embedding dimension
        if hasattr(self.model.config, 'hidden_size'):
            self.embedding_dim = self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            self.embedding_dim = self.model.config.d_model
        else:
            self.embedding_dim = 768  # default
        
        # Set observation/action dimensions
        self.observation_dim = self.embedding_dim
        self.action_dim = self.embedding_dim if self.use_embeddings else self.tokenizer.vocab_size
    
    def generate(self, prompt: str, deterministic: bool = True, **kwargs) -> str:
        """Generate text using HuggingFace model."""
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors='pt', truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generation config
        gen_kwargs = {
            'max_new_tokens': kwargs.get('max_new_tokens', 128),
            'do_sample': not deterministic,
            'temperature': kwargs.get('temperature', self.temperature) if not deterministic else 1.0,
            'pad_token_id': self.tokenizer.pad_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
        }
        
        if not deterministic:
            gen_kwargs['top_p'] = kwargs.get('top_p', 0.9)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Store logits if needed
        if hasattr(outputs, 'scores') and outputs.scores:
            self._internal_state['logits'] = torch.stack(outputs.scores).cpu()
        
        # Decode
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        return response
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings by extracting last hidden state."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        
        # Use last hidden state of last token as embedding
        if hasattr(outputs, 'hidden_states'):
            last_hidden = outputs.hidden_states[-1]
            embedding = last_hidden[0, -1, :].cpu()
        else:
            # Fallback: use pooled output if available
            embedding = outputs.pooler_output[0].cpu() if hasattr(outputs, 'pooler_output') else torch.zeros(self.embedding_dim)
        
        return embedding
    
    def encode(self, text: str) -> torch.Tensor:
        """Encode text to token IDs."""
        tokens = self.tokenizer.encode(text, return_tensors='pt')[0]
        return tokens
    
    def decode(self, tokens: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(tokens, skip_special_tokens=True)
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return model parameters (if not quantized)."""
        if self.load_in_8bit or self.load_in_4bit:
            return {}  # Can't access parameters easily
        return dict(self.model.named_parameters())
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set model parameters (use with caution)."""
        if self.load_in_8bit or self.load_in_4bit:
            return
        own_params = dict(self.model.named_parameters())
        for name, value in params.items():
            if name in own_params:
                own_params[name].data.copy_(value.to(self.device))


# ======================================================================
# OpenAI Agent (API-based)
# ======================================================================

class OpenAIAgent(LLMAgent):
    """
    LLM agent using OpenAI's API.
    
    Note: This is non-differentiable and has no local parameters.
    For zeroing, use the DifferentiableLLMWrapper with an approximation.
    """
    
    def __init__(
        self,
        name: str,
        model_name: str = 'gpt-4',
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        embedding_model: str = 'text-embedding-ada-002',
        **kwargs
    ):
        """
        Args:
            name: Agent name
            model_name: OpenAI model (e.g., 'gpt-4', 'gpt-3.5-turbo')
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env)
            api_base: Custom API endpoint
            embedding_model: Model for embeddings
            **kwargs: Additional arguments
        """
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI not installed. Install with: pip install openai")
        
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key not provided and OPENAI_API_KEY not set")
        
        # Initialize client
        openai.api_key = self.api_key
        if api_base:
            openai.api_base = api_base
        
        # Embedding dimension (known for OpenAI models)
        self.embedding_dim = 1536 if 'ada' in embedding_model else 3072
        
        super().__init__(name, model_name, device='cpu', **kwargs)
    
    def _load_model(self, **kwargs):
        """No local model to load."""
        self.observation_dim = self.embedding_dim
        self.action_dim = self.embedding_dim if self.use_embeddings else 50000  # approximate vocab
    
    def generate(self, prompt: str, deterministic: bool = True, **kwargs) -> str:
        """Generate text using OpenAI API."""
        messages = [{"role": "user", "content": prompt}]
        
        # Add history if available
        if hasattr(self, 'history') and self.history:
            messages = self.history + messages
        
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0 if deterministic else kwargs.get('temperature', self.temperature),
            max_tokens=kwargs.get('max_new_tokens', 128),
            top_p=kwargs.get('top_p', 1.0) if not deterministic else 1.0
        )
        
        return response.choices[0].message.content
    
    def get_embeddings(self, text: str) -> torch.Tensor:
        """Get embeddings using OpenAI's embedding API."""
        response = openai.Embedding.create(
            model=self.embedding_model,
            input=text
        )
        embedding = response['data'][0]['embedding']
        return torch.tensor(embedding)
    
    def encode(self, text: str) -> torch.Tensor:
        """No local tokenizer – return dummy."""
        return torch.tensor([0])
    
    def decode(self, tokens: torch.Tensor) -> str:
        """No local tokenizer – return placeholder."""
        return "[Decoding not available for OpenAI]"
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """No local parameters."""
        return {}
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """No local parameters to set."""
        pass


# ======================================================================
# Differentiable LLM Wrapper (for zeroing)
# ======================================================================

class DifferentiableLLMWrapper(BaseAgent):
    """
    Wrapper that makes an LLM agent differentiable via a learned approximation.
    
    This enables end-to-end zeroing through the LLM by using:
        - A small learnable model that approximates the LLM's behavior
        - Straight-through estimators for non-differentiable parts
    """
    
    def __init__(
        self,
        base_agent: LLMAgent,
        approximation_model: Optional[nn.Module] = None,
        hidden_dims: List[int] = [512, 512],
        use_ste: bool = True,
        device: str = 'cuda'
    ):
        """
        Args:
            base_agent: The base LLM agent (non-differentiable)
            approximation_model: Optional custom approximation model
            hidden_dims: Dimensions of default approximation network
            use_ste: Use straight-through estimator for tokens
            device: Computation device
        """
        super().__init__(
            name=f"diff_{base_agent.name}",
            observation_dim=base_agent.observation_dim,
            action_dim=base_agent.action_dim,
            device=device
        )
        
        self.base = base_agent
        self.use_ste = use_ste
        
        # Create approximation model if not provided
        if approximation_model is None:
            self.approx = nn.Sequential(
                nn.Linear(self.observation_dim, hidden_dims[0]),
                nn.ReLU(),
                *[
                    layer for i in range(len(hidden_dims)-1)
                    for layer in [nn.Linear(hidden_dims[i], hidden_dims[i+1]), nn.ReLU()]
                ],
                nn.Linear(hidden_dims[-1], self.action_dim)
            ).to(device)
        else:
            self.approx = approximation_model.to(device)
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Forward pass with differentiability.
        
        For text inputs, we use the approximation model to generate
        differentiable outputs.
        """
        # Convert to tensor if needed
        if isinstance(observation, str):
            # Use base agent to get embeddings
            with torch.no_grad():
                emb = self.base.get_embeddings(observation)
            obs_tensor = emb.to(self.device)
        else:
            obs_tensor = observation.to(self.device)
        
        # Get approximation output (differentiable)
        approx_out = self.approx(obs_tensor)
        
        if deterministic:
            return approx_out
        
        # For sampling, we might want to combine with base agent
        with torch.no_grad():
            base_out = self.base.act(observation, deterministic)
            if isinstance(base_out, torch.Tensor):
                base_out = base_out.to(self.device)
            else:
                # Convert text to tensor (approximate)
                base_out = approx_out  # fallback
        
        if self.use_ste:
            # Straight-through: use base value but gradient from approx
            return base_out + (approx_out - base_out).detach()
        else:
            return approx_out
    
    def get_parameters(self) -> Dict[str, torch.Tensor]:
        """Return approximation model parameters."""
        return dict(self.approx.named_parameters())
    
    def set_parameters(self, params: Dict[str, torch.Tensor]):
        """Set approximation model parameters."""
        own_params = dict(self.approx.named_parameters())
        for name, value in params.items():
            if name in own_params:
                own_params[name].data.copy_(value.to(self.device))
    
    def get_internal_state(self) -> Dict[str, torch.Tensor]:
        """Return base agent's internal state."""
        return self.base.get_internal_state()
    
    def set_internal_state(self, state: Dict[str, torch.Tensor]):
        """Set base agent's internal state."""
        self.base.set_internal_state(state)


# ======================================================================
# Level-Specific LLM Agents
# ======================================================================

class LLMG1Wrapper(LLMAgent):
    """LLM agent for level G1 (perception/language understanding)."""
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Return embeddings of the input text."""
        if isinstance(observation, str):
            return self.get_embeddings(observation)
        return observation


class LLMG3Wrapper(LLMAgent):
    """LLM agent for level G3 (task planning)."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        # Add task planning specific prompt template
        self.planning_prompt = kwargs.get(
            'planning_prompt',
            "You are a robot task planner. Given the instruction: '{instruction}', "
            "break it down into a sequence of 3-5 steps. Output only the steps, "
            "one per line, as simple actions."
        )
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Generate a task plan from instruction."""
        # Convert observation to instruction text
        if isinstance(observation, str):
            instruction = observation
        else:
            instruction = self.decode(observation)
        
        # Format prompt
        prompt = self.planning_prompt.format(instruction=instruction)
        
        # Generate plan
        plan = self.generate(prompt, deterministic)
        
        # Store plan in internal state
        self._internal_state['plan'] = plan
        
        # Return tokenized plan
        return self.encode(plan)


class LLMG4Wrapper(LLMAgent):
    """LLM agent for level G4 (ethics/safety checking)."""
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        
        # Safety prompt template
        self.safety_prompt = kwargs.get(
            'safety_prompt',
            "You are an AI safety checker. Determine if the following action is "
            "ethically acceptable for a robot to perform. Action: '{action}'. "
            "Reply with only 'SAFE' or 'UNSAFE'."
        )
        
        # Ethical principles to check
        self.principles = kwargs.get('principles', [
            "do no harm",
            "respect human autonomy",
            "be truthful",
            "be fair"
        ])
    
    def check_safety(self, action: str, context: Optional[str] = None) -> Tuple[bool, float]:
        """
        Check if an action is safe.
        
        Returns:
            (is_safe, confidence)
        """
        if context:
            prompt = f"Context: {context}\n" + self.safety_prompt.format(action=action)
        else:
            prompt = self.safety_prompt.format(action=action)
        
        response = self.generate(prompt, deterministic=True)
        
        is_safe = 'SAFE' in response.upper()
        confidence = 1.0 if is_safe else 0.0  # Could be more nuanced
        
        return is_safe, confidence
    
    def act(self, observation: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """
        Safety check on an action.
        
        Observation should contain the action to check.
        Returns a binary safety score.
        """
        # Convert observation to action text
        if isinstance(observation, str):
            action = observation
        else:
            action = self.decode(observation)
        
        is_safe, confidence = self.check_safety(action)
        
        # Return as tensor
        result = torch.tensor([1.0 if is_safe else 0.0, confidence])
        return result


# ======================================================================
# Utility Functions
# ======================================================================

def create_llm_agent(
    level: int,
    name: str,
    backend: str = 'huggingface',
    model_name: Optional[str] = None,
    device: str = 'cuda',
    differentiable: bool = False,
    **kwargs
) -> BaseAgent:
    """
    Factory function to create appropriate LLM agent.
    
    Args:
        level: GRA level (1, 3, 4 typically)
        name: Agent name
        backend: 'huggingface' or 'openai'
        model_name: Model identifier
        device: Computation device
        differentiable: Whether to use differentiable wrapper
        **kwargs: Additional arguments
    
    Returns:
        Configured LLM agent
    """
    level_map = {
        1: LLMG1Wrapper,
        3: LLMG3Wrapper,
        4: LLMG4Wrapper
    }
    
    if level not in level_map:
        raise ValueError(f"Invalid level for LLM: {level}. Use 1, 3, or 4.")
    
    # Set default model names
    if model_name is None:
        if backend == 'huggingface':
            model_name = 'microsoft/phi-2' if level == 1 else 'meta-llama/Llama-2-7b-chat-hf'
        elif backend == 'openai':
            model_name = 'gpt-3.5-turbo'
    
    # Create backend-specific agent
    if backend == 'huggingface':
        agent_class = HuggingFaceAgent
    elif backend == 'openai':
        agent_class = OpenAIAgent
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Create base agent
    base_agent = agent_class(
        name=name,
        model_name=model_name,
        device=device,
        **kwargs
    )
    
    # Wrap with level-specific wrapper
    level_class = level_map[level]
    agent = level_class(
        name=f"{name}_g{level}",
        model_name=model_name,
        device=device,
        **kwargs
    )
    
    # Copy base model if needed
    if hasattr(agent, 'model') and hasattr(base_agent, 'model'):
        agent.model = base_agent.model
        agent.tokenizer = base_agent.tokenizer
    
    if differentiable:
        agent = DifferentiableLLMWrapper(agent, device=device)
    
    return agent


def llm_to_subsystem(agent: LLMAgent, multi_index: MultiIndex) -> Any:
    """
    Convert an LLM agent to a GRA subsystem.
    """
    from ..core.subsystem import Subsystem
    
    class LLMSubsystem(Subsystem):
        def __init__(self, agent, multi_index):
            super().__init__(multi_index, None, None)
            self.agent = agent
        
        def get_state(self):
            return self.agent.get_state()
        
        def set_state(self, state):
            self.agent.set_state(state)
        
        def step(self, dt, action=None):
            # LLM stepping is handled by environment
            pass
    
    return LLMSubsystem(agent, multi_index)


# ======================================================================
# Example Usage
# ======================================================================

if __name__ == "__main__":
    print("=== Testing LLM Agents ===\n")
    
    # Test HuggingFace agent (if available)
    if TRANSFORMERS_AVAILABLE:
        print("Creating HuggingFace agent...")
        try:
            # Use small model for testing
            agent = HuggingFaceAgent(
                name="test_llm",
                model_name="microsoft/phi-2",
                device='cpu'
            )
            
            response = agent.generate("Hello, I'm a robot. How are you?")
            print(f"  Response: {response}")
            
            # Test G3 planner
            planner = LLMG3Wrapper(
                name="test_planner",
                model_name="microsoft/phi-2",
                device='cpu'
            )
            plan = planner.act("fetch a cup of coffee")
            print(f"  Plan: {plan}")
            
        except Exception as e:
            print(f"  HuggingFace test skipped: {e}")
    
    # Test OpenAI agent (if available)
    if OPENAI_AVAILABLE and os.getenv('OPENAI_API_KEY'):
        print("\nCreating OpenAI agent...")
        agent = OpenAIAgent(
            name="test_openai",
            model_name="gpt-3.5-turbo"
        )
        
        # Test embeddings
        emb = agent.get_embeddings("Hello world")
        print(f"  Embedding shape: {emb.shape}")
    
    # Test differentiable wrapper
    print("\nCreating differentiable wrapper...")
    if TRANSFORMERS_AVAILABLE:
        base = HuggingFaceAgent(
            name="base",
            model_name="microsoft/phi-2",
            device='cpu'
        )
        diff_agent = DifferentiableLLMWrapper(base, device='cpu')
        
        # Test gradient flow
        obs = torch.randn(10, requires_grad=True)
        action = diff_agent.act(obs)
        loss = action.sum()
        loss.backward()
        print(f"  Gradient on observation: {obs.grad is not None}")
    
    print("\nAll tests completed!")
```