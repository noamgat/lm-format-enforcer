import math
from typing import List, Optional, Tuple, Union
import torch
from transformers import PreTrainedTokenizerBase
from lmformatenforcer import CharacterLevelParser, FormatEnforcerAnalyzer
from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData


class TRTLLMLogitsProcessor:
    def __init__(self, token_enforcer: TokenEnforcer, eos_token_id, analyze):
        self.token_enforcer = token_enforcer
        self.analyzer = FormatEnforcerAnalyzer(token_enforcer) if analyze else None
        self.mask: Optional[torch.Tensor] = None
        self.mask_val = -math.inf
        self.eos_token_id = eos_token_id

    def _trim(self, input):
        return [x for x in input.tolist() if x not in \
                (self.eos_token_id if isinstance(self.eos_token_id, list) else [self.eos_token_id])]

    def __call__(self, step: int, batch_input_ids: List[List[int]], logits: torch.Tensor) -> torch.Tensor:
        for idx in range(len(batch_input_ids)):
            if self.analyzer:
                self.analyzer.report_raw_logits(batch_input_ids[idx], logits[idx].tolist())

            allowed_tokens = self.token_enforcer.get_allowed_tokens(self._trim(batch_input_ids[idx]))

            if self.mask is not None:
                self.mask.fill_(self.mask_val)
            else:
                # We create it here because full_like() also copies the device and dtype
                self.mask = torch.full_like(logits[idx], self.mask_val)
            self.mask[allowed_tokens] = 0
            logits[idx] = logits[idx] + self.mask

        return logits


def _build_regular_tokens_list(tokenizer) -> List[Tuple[int, str, bool]]:
    # There are many classes that can be passed here, this logic should work on all of them.
    if hasattr(tokenizer, 'get_tokenizer'):
        tokenizer = tokenizer.get_tokenizer()
    if hasattr(tokenizer, 'tokenizer'):
        tokenizer = tokenizer.tokenizer
    token_0 = [tokenizer.encode("0")[-1]]
    regular_tokens = []
    vocab_size = tokenizer.vocab_size
    for token_idx in range(vocab_size):
        if token_idx in tokenizer.all_special_ids:
            continue
        # We prepend token 0 and skip the first letter of the result to get a space if the token is a start word.
        tensor_after_0 = torch.tensor(token_0 + [token_idx], dtype=torch.long)
        decoded_after_0 = tokenizer.decode(tensor_after_0)[1:]
        decoded_regular = tokenizer.decode(token_0)
        is_word_start_token = len(decoded_after_0) > len(decoded_regular)
        regular_tokens.append((token_idx, decoded_after_0, is_word_start_token))
    return regular_tokens


def build_trtlmm_tokenizer_data(tokenizer: PreTrainedTokenizerBase) -> TokenEnforcerTokenizerData:
    """Build the TokenEnforcerTokenizerData from a tokenizer in order to cache it between instances"""
    regular_tokens = _build_regular_tokens_list(tokenizer)

    def _decode(tokens: List[int]) -> str:
        tensor = torch.tensor(tokens, dtype=torch.long)
        return tokenizer.decode(tensor)

    tokenizer_data = TokenEnforcerTokenizerData(regular_tokens, _decode, tokenizer.eos_token_id)
    return tokenizer_data


def build_trtllm_logits_processor(tokenizer: Union[PreTrainedTokenizerBase, TokenEnforcerTokenizerData],
                                  character_level_parser: CharacterLevelParser,
                                  analyze: bool = False) -> TRTLLMLogitsProcessor:
    """
    Build logits processor for feeding it into generate function (use_py_session should be True)
    """
    if isinstance(tokenizer, TokenEnforcerTokenizerData):
        tokenizer_data = tokenizer
    else:
        tokenizer_data = build_trtlmm_tokenizer_data(tokenizer)

    token_enforcer = TokenEnforcer(tokenizer_data, character_level_parser)
    return TRTLLMLogitsProcessor(token_enforcer, tokenizer.eos_token_id, analyze)
