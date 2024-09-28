import cProfile
from pstats import Stats
from typing import Optional
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.exceptions import LMFormatEnforcerException
from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
import logging
            

_tokenizer: Optional[PreTrainedTokenizerBase] = None
_tokenizer_data: Optional[TokenEnforcerTokenizerData] = None


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
    

def assert_parser_with_string_token_enforcer(string: str, parser: CharacterLevelParser, expect_success: bool, profile_file_path: Optional[str]):
    global _tokenizer
    if _tokenizer is None:
        model_id = 'Qwen/Qwen2.5-72B-Instruct'
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    global _tokenizer_data

    # For testing, we make sure that all letters exist individually in the tokenizer
    encoded_0 = _tokenizer.encode("0")
    for word in set(string):
        encoded_word = _tokenizer.encode(word)
        if len(encoded_word) > len(encoded_0):
            logging.basicConfig(level=logging.INFO)
            logging.warning("Encountered out-of-tokenizer character, LMFE does not deal with this well")
    
    if _tokenizer_data is None:
        _tokenizer_data = build_token_enforcer_tokenizer_data(_tokenizer)
        
    prompt = "This is my question:\n\n"
    initial_token_array = _tokenizer.encode(prompt)
    # While the LMFE allows several ways to build correct output using different token sequences, we only
    # test for the tokenizer's default way to encode the output string, as we assume that it will
    # take the shortest path, which is the most likely to be taken by the LM, and the one that challenges
    # the parser the most.
    target_token_array = _tokenizer.encode(prompt + string)
    eos_token_id = _tokenizer.eos_token_id
    if not eos_token_id:
        raise ValueError(f"Tokenizer does not have {'an EOS token' if eos_token_id is None else 'EOS tokens'}")
    
    token_enforcer = TokenEnforcer(_tokenizer_data, parser)
    # The token enforcer is stateful - it keeps track of the parsing state as tokens arrive on a token by token basis.
    # We simulate a language model that "chooses" the next token in the encoded sequence, and check that it is in the
    # allowed list at every timestep.
    profiler: Optional[cProfile.Profile] = None
    if profile_file_path:
        profiler = start_profiling()

    for prefix_length in range(len(initial_token_array), len(target_token_array) + 1):
        prefix = target_token_array[:prefix_length]
        allowed_tokens = token_enforcer.get_allowed_tokens(prefix)
        if prefix_length < len(target_token_array):
            next_token = target_token_array[prefix_length]
            if next_token not in allowed_tokens:
                if expect_success:
                    decoded_before = _tokenizer.decode(prefix, skip_special_tokens=True)
                    decoded_after = _tokenizer.decode(prefix + [next_token], skip_special_tokens=True)
                    next_token_chars = decoded_after[len(decoded_before):]
                    next_idx = len(decoded_before) - len(prompt)
                    raise CharacterNotAllowedException(f"Parser does not allow '{next_token_chars}' at index {next_idx}")
                else:
                    return  # Test success
        else:
            # Reached the end of the sequence, check that ending state matches expected ending state
            can_end = any(token in allowed_tokens for token in (eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]))
            if can_end and not expect_success:
                raise ValueError("Parser succeeded when it should have failed")
            if not can_end and expect_success:
                raise ValueError("Parser did not reach end state when it should have")
            
    if profiler and profile_file_path:
        finish_profiling(profiler, profile_file_path)
        
    
def assert_parser_with_string(string: str, parser: CharacterLevelParser, expect_success: bool, profile_file_path: Optional[str] = None):
    assert_parser_with_string_direct(string, parser, expect_success)
    assert_parser_with_string_token_enforcer(string, parser, expect_success, profile_file_path)


def start_profiling() -> cProfile.Profile:
    profiler = cProfile.Profile()
    profiler.enable()
    return profiler

def finish_profiling(profiler: cProfile.Profile, profile_file_path: str):
    profiler.disable()
    with open(profile_file_path, 'w') as stream:
        stats = Stats(profiler, stream=stream)
        stats.strip_dirs()
        stats.sort_stats('time')
        stats.dump_stats(profile_file_path + '.prof_stats')
        stats.print_stats()
