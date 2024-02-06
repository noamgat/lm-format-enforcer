import math
from typing import List, Optional
import torch
from transformers import PreTrainedTokenizerBase
from lmformatenforcer import CharacterLevelParser, TokenEnforcer, FormatEnforcerAnalyzer
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data


class TRTLLMLogitsProcessor:
    def __init__(self, token_enforcer: TokenEnforcer, analyze):
        self.token_enforcer = token_enforcer
        self.analyzer = FormatEnforcerAnalyzer(token_enforcer) if analyze else None
        self.mask: Optional[torch.Tensor] = None
        self.mask_val = -math.inf

    def __call__(self, step: int, batch_input_ids: List[List[int]], logits: torch.Tensor) -> torch.Tensor:
        for idx in range(len(batch_input_ids)):
            if self.analyzer:
                self.analyzer.report_raw_logits(batch_input_ids[idx], logits.tolist())

            allowed_tokens = self.token_enforcer.get_allowed_tokens(batch_input_ids[idx])

            if self.mask is not None:
                self.mask.fill_(self.mask_val)
            else:
                # We create it here because full_like() also copies the device and dtype
                self.mask = torch.full_like(logits[idx], self.mask_val)
            self.mask[allowed_tokens] = 0
            logits[idx] = logits[idx] + self.mask

        return logits


def build_trtllm_logits_processor(tokenizer: PreTrainedTokenizerBase,
                                  character_level_parser: CharacterLevelParser,
                                  analyze: bool = False) -> TRTLLMLogitsProcessor:
    """
    Build logits processor for feeding it into generate function (use_py_session should be True)
    """
    tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)
    token_enforcer = TokenEnforcer(tokenizer_data, character_level_parser)
    return TRTLLMLogitsProcessor(token_enforcer, analyze)
