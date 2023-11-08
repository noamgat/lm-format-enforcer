try:
    from haystack.preview import component
    from canals import Component
except ImportError:
    raise ImportError('haystack is not installed. Please install it with "pip install farm-haystack" or "pip install haystack-ai')

import enum
from typing import Any, Callable, Dict, List, Optional, cast
from lmformatenforcer import CharacterLevelParser


class _ModelType(enum.Enum):
    HUGGINGFACE = 'HuggingFaceLocalGenerator'
    # VLLM = 'vLLMLocalInvocationLayer'
    # LLAMA_CPP = 'LlamaCppLocalInvocationLayer.cpp'

@component
class LMFormatEnforcerLocalGenerator:
    
    def __init__(self, model_component: Component, character_level_parser: Optional[CharacterLevelParser] = None):
        self.model_component = model_component
        self.character_level_parser = character_level_parser
        self._model_type = self._resolve_model_type()
        self.token_enforcer_fn: Optional[Callable] = None

    @component.output_types(replies=List[str])
    def run(self, prompt: str, generation_kwargs: Optional[Dict[str, Any]] = None):
        try:
            self._inject_enforcer_into_model()
            return self.model_component.run(prompt)
        finally:
            self._release_model_injection()

    def warm_up(self):
        if hasattr(self.model_component, 'warm_up'):
            self.model_component.warm_up()
        self.token_enforcer_fn = self._prepare_token_enforcer_fn()

    def _prepare_token_enforcer_fn(self) -> Optional[Callable]:
        if not self.character_level_parser:
            return None
        if self._model_type == _ModelType.HUGGINGFACE:
            tokenizer = self.model_component.pipeline.tokenizer
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            return build_transformers_prefix_allowed_tokens_fn(tokenizer, self.character_level_parser)
        raise NotImplementedError(f"Token enforcer not implemented for model type {self._model_type.name}")

    def _resolve_model_type(self) -> _ModelType:
        invocation_layer_name = self.model_component.__class__.__name__ 
        try:
            return _ModelType(invocation_layer_name)
        except ValueError:
            supported_strings = ",".join(str(t.name) for t in _ModelType)
            raise ValueError(f"Unsupported invocation layer: {invocation_layer_name}. "
                             f"Must be one of {supported_strings}")
        
    def _inject_enforcer_into_model(self):
        if not self.token_enforcer_fn:
            return
        if self._model_type == _ModelType.HUGGINGFACE:
            self.model_component.generation_kwargs['prefix_allowed_tokens_fn'] = self.token_enforcer_fn
        
    
    def _release_model_injection(self):
        if not self.token_enforcer_fn:
            return
        if self._model_type == _ModelType.HUGGINGFACE:
            del self.model_component.generation_kwargs['prefix_allowed_tokens_fn']
