try:
    from haystack.nodes import PromptNode, PromptModel
except ImportError:
    raise ImportError('haystack is not installed. Please install it with "pip install farm-haystack"')

import enum
from typing import Callable, Optional, cast
from lmformatenforcer import CharacterLevelParser

class LMFormatEnforcerPromptNode(PromptNode):
    class ModelType(enum.Enum):
        HUGGINGFACE = 'HFLocalInvocationLayer'
        VLLM = 'vLLMLocalInvocationLayer'
        LLAMA_CPP = 'LlamaCppLocalInvocationLayer.cpp'

    def __init__(self, *args, character_level_parser: Optional[CharacterLevelParser] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.character_level_parser = character_level_parser
        self.model_type = self._resolve_model_type()
        self.token_enforcer_fn = self._prepare_token_enforcer_fn()

    #@property
    #def prompt_model(self) -> PromptModel:
    #    return cast(PromptModel, self.model_name_or_path)
    
    def _prepare_token_enforcer_fn(self) -> Optional[Callable]:
        if not self.character_level_parser:
            return None
        if self.model_type == LMFormatEnforcerPromptNode.ModelType.HUGGINGFACE:
            tokenizer = self.prompt_model.model_invocation_layer.pipe.tokenizer
            from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
            return build_transformers_prefix_allowed_tokens_fn(tokenizer, self.character_level_parser)
        raise NotImplementedError(f"Token enforcer not implemented for model type {self.model_type.name}")

    def _resolve_model_type(self) -> ModelType:
        invocation_layer_name = self.prompt_model.model_invocation_layer.__class__.__name__ 
        try:
            return LMFormatEnforcerPromptNode.ModelType(invocation_layer_name)
        except ValueError:
            supported_strings = ",".join(str(t.name) for t in LMFormatEnforcerPromptNode.ModelType)
            raise ValueError(f"Unsupported invocation layer: {invocation_layer_name}. "
                             f"Must be one of {supported_strings}")
        
    def _prepare_model_kwargs(self):
        model_kwargs = super()._prepare_model_kwargs()
        if self.token_enforcer_fn:
            if self.model_type == LMFormatEnforcerPromptNode.ModelType.HUGGINGFACE:
                if 'generation_kwargs' not in model_kwargs:
                    model_kwargs['generation_kwargs'] = {}
                model_kwargs['generation_kwargs']['prefix_allowed_tokens_fn'] = self.token_enforcer_fn
        return model_kwargs