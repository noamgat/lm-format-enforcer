from typing import Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.exceptions import LMFormatEnforcerException
from lmformatenforcer.tokenenforcer import TokenEnforcer
from lmformatenforcer.integrations.transformers import build_regular_tokens_list


_tokenizer: Optional[PreTrainedTokenizerBase] = None


class CharacterNotAllowedException(LMFormatEnforcerException):
    pass


def assert_parser_with_string_direct(string: str, parser: CharacterLevelParser, expect_success: bool):
    for idx, character in enumerate(string):
        try:
            if character in parser.get_allowed_characters():
                parser = parser.add_character(character)
            else:
                if expect_success:
                    raise CharacterNotAllowedException(f"Parser does not allow '{character}' at index {idx}")
                else:
                    return  # Success
        except LMFormatEnforcerException:
            raise
        except Exception as e:
            raise Exception(f"Error parsing '{character}' at index {idx}: {e}", e)
    if parser.can_end() and not expect_success:
        raise ValueError("Parser succeeded when it should have failed")
    if not parser.can_end() and expect_success:
        raise ValueError("Parser did not reach end state when it should have")
    

def assert_parser_with_string_token_enforcer(string: str, parser: CharacterLevelParser, expect_success: bool):
    global _tokenizer
    if _tokenizer is None:
        model_id = 'TheBloke/Llama-2-7b-Chat-GPTQ'
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
        
    prompt = "This is my question:\n\n"
    initial_token_array = _tokenizer.encode(prompt)
    # While the LMFE allows several ways to build correct output using different token sequences, we only
    # test for the tokenizer's default way to encode the output string, as we assume that it will
    # take the shortest path, which is the most likely to be taken by the LM, and the one that challenges
    # the parser the most.
    target_token_array = _tokenizer.encode(prompt + string)
    regular_tokens = build_regular_tokens_list(_tokenizer)
    eos_token_id = _tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer does not have an EOS token")
    
    token_enforcer = TokenEnforcer(regular_tokens, parser, _tokenizer.decode, eos_token_id)
    # The token enforcer is stateful - it keeps track of the parsing state as tokens arrive on a token by token basis.
    # We simulate a language model that "chooses" the next token in the encoded sequence, and check that it is in the
    # allowed list at every timestep.
    for prefix_length in range(len(initial_token_array), len(target_token_array) + 1):
        prefix = target_token_array[:prefix_length]
        allowed_tokens = token_enforcer.get_allowed_tokens(prefix)
        if prefix_length < len(target_token_array):
            next_token = target_token_array[prefix_length]
            if next_token not in allowed_tokens:
                if expect_success:
                    decoded_before = _tokenizer.decode(prefix, skip_special_tokens=True)
                    decoded_after = _tokenizer.decode(prefix + [next_token], skip_special_tokens=True)
                    next_char = decoded_after[len(decoded_before)]
                    next_idx = len(decoded_before) - len(prompt)
                    raise CharacterNotAllowedException(f"Parser does not allow '{next_char}' at index {next_idx}")
                else:
                    return  # Test success
        else:
            # Reached the end of the sequence, check that ending state matches expected ending state
            can_end = eos_token_id in allowed_tokens
            if can_end and not expect_success:
                raise ValueError("Parser succeeded when it should have failed")
            if not can_end and expect_success:
                raise ValueError("Parser did not reach end state when it should have")
        
    
def assert_parser_with_string(string: str, parser: CharacterLevelParser, expect_success: bool):
    assert_parser_with_string_direct(string, parser, expect_success)
    assert_parser_with_string_token_enforcer(string, parser, expect_success)
