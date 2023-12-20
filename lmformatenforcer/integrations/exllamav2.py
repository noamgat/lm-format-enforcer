from typing import List, Set, Tuple, Union
try:
    import torch
    from exllamav2 import ExLlamaV2Tokenizer
except ImportError:
    raise ImportError('exllamav2 is not installed. Please install it with "pip install exllamav2"')

from ..characterlevelparser import CharacterLevelParser
from ..tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData


def _build_regular_tokens_list(tokenizer: ExLlamaV2Tokenizer) -> List[Tuple[int, str, bool]]:
    token_0 = tokenizer.encode("0")[0]
    regular_tokens = []
    vocab_size = tokenizer.tokenizer.vocab_size()
    all_special_ids = [tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id, tokenizer.unk_token_id]
    for token_idx in range(vocab_size):
        if token_idx in all_special_ids:
            continue
        # We prepend token 0 and skip the first letter of the result to get a space if the token is a start word.
        tensor_after_0 = torch.tensor(token_0.tolist() + [token_idx], dtype=torch.long)
        decoded_after_0 = tokenizer.decode(tensor_after_0)[1:]
        decoded_regular = tokenizer.decode(token_0)
        is_word_start_token = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))
    return regular_tokens


def build_token_enforcer_tokenizer_data(tokenizer: ExLlamaV2Tokenizer) -> TokenEnforcerTokenizerData:
    regular_tokens = _build_regular_tokens_list(tokenizer)

    def _decode(tokens: List[int]) -> str:
        tensor = torch.tensor(tokens, dtype=torch.long)
        return tokenizer.decode(tensor)
    
    return TokenEnforcerTokenizerData(regular_tokens, _decode, tokenizer.eos_token_id)


class ExLlamaV2TokenEnforcerFilter:
    """ExLlamaV2Sampler.Settings.filters filter that uses the token enforcer to only allow format-complying tokens"""
    token_sequence: List[int]

    def __init__(self, 
                 character_level_parser: CharacterLevelParser, 
                 tokenizer_data: Union[ExLlamaV2Tokenizer, TokenEnforcerTokenizerData]):
        if isinstance(tokenizer_data, ExLlamaV2Tokenizer):
            tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer_data)
        self.token_enforcer = TokenEnforcer(tokenizer_data, character_level_parser)
        self.token_sequence = []
    
    def begin(self, prefix_str: str) -> None:
        self.token_sequence = []
    
    def feed(self, token: torch.Tensor) -> None:
        self.token_sequence.append(int(token[0][0]))

    def clone(self):
        return self
    
    def next(self) -> Tuple[Set[int], Set[int]]:
        allowed_tokens = self.token_enforcer.get_allowed_tokens(self.token_sequence)
        return set(allowed_tokens), set()


__all__ = ['ExLlamaV2TokenEnforcerFilter', 'build_token_enforcer_tokenizer_data']