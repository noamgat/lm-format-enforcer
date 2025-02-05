from dataclasses import dataclass, field
import sys
from typing import Callable, Dict, Hashable, List, Optional, Tuple, Union
import logging

from .exceptions import LMFormatEnforcerException
from .characterlevelparser import CharacterLevelParser, ForceStopParser, CharacterLevelParserConfig
from .tokenizerprefixtree import TokenizerPrefixTree, TokenizerPrefixTreeNode
from .regexparser import RegexParser


class TokenEnforcerTokenizerData:
    """TokenEnforcerTokenizerData contains all of the preprocessing for preparing the TokenEnforcer to work with a 
    specific tokenizer. It does some calculations, so it is recommended to reuse it for multiple TokenEnforcers"""
    def __init__(self, 
                 regular_tokens: List[Tuple[int, str, bool]], 
                 decoder: Callable[[List[int]], str],
                 eos_token_id: Union[int, List[int]]):
        """
        Create the tokenizer data that the TokenEnforcer needs. This can be reused for multiple TokenEnforcers if they work with the same tokenizer.
        :param regular_tokens: A list of tuples (token_id, token_string, is_new_word_token) for all the regular (not special) tokens in the tokenizer vocabulary.
        Note that token_string is expected to include leading / trailing whitespaces if relevant.
        :param decoder: A function that decodes a list of token ids into a string.
        :param eos_token_id: The token id(s) of the end-of-string token(s).
        """
        self.regular_tokens = regular_tokens
        self.tokenizer_tree = TokenizerPrefixTree(regular_tokens)
        self.decoder = decoder
        self.eos_token_id = eos_token_id
        self.tokenizer_alphabet = "".join(token_str for token_str in self.tokenizer_tree.root.children.keys() if len(token_str) == 1)


class TokenEnforcer:
    """TokenEnforcer provides a token filtering mechanism, given a CharacterLevelParser and some information about the tokenizer.
    It is the main entry point for extending lm-format-enforcer to new inference libraries. See __init__() and get_allowed_tokens()"""
    @dataclass
    class OutputTensorState:
        parser: CharacterLevelParser
        allowed_tokens: List[int] = field(default_factory=list)
        current_word_tokens: List[int] = field(default_factory=list)

    def __init__(self, tokenizer_data: TokenEnforcerTokenizerData, parser: CharacterLevelParser):
        """
        Create a new TokenEnforcer.
        :param tokenizer_data: Per tokenizer data that the token enforcer needs in order to operate.
        :param parser: A CharacterLevelParser that defines the allowed strings.
        """
        self.prefix_states: Dict[Tuple, TokenEnforcer.OutputTensorState] = {}
        self.root_parser = parser
        self.tokenizer_tree = tokenizer_data.tokenizer_tree
        self.decoder = tokenizer_data.decoder
        self.eos_token_id = tokenizer_data.eos_token_id
        self.regular_tokens = tokenizer_data.regular_tokens
        self.allowed_token_cache: Dict[Hashable, List[int]] = {}
        
        config = CharacterLevelParserConfig(alphabet=tokenizer_data.tokenizer_alphabet)
        parser.config = config

    def get_allowed_tokens(self, token_sequence: List[int]) -> List[int]:
        """
        Get a list of allowed tokens, given a list of tokens that were already generated.
        :param token_sequence: The tokens that were already generated, and the next token will be generated for.
        :return: A list of token ids that are allowed to be selected next.
        """
        # In order to elegantly support beam search and batching, we don't store per-batch information.
        # Instead, we store a hash of all the states (unique token tensors) we encountered so far.
        # When we encounter a new unique token tensor, we find the token tensor that led to it, and continue from there.
        sent_tuple = tuple(token_sequence)
        prev_step_tuple = sent_tuple[:-1]

        if sent_tuple in self.prefix_states:
            # We already calculated for this node, return cached list
            return self.prefix_states[sent_tuple].allowed_tokens
        elif prev_step_tuple not in self.prefix_states:
            # We have not encountered the tensor up to the before-last entry. This means that this is the first call - the instruction / prompt tensor.
            # Initialize the root node
            state = TokenEnforcer.OutputTensorState(parser=self.root_parser)
            self.prefix_states[sent_tuple] = state
            self._compute_allowed_tokens(sent_tuple, state)
            return state.allowed_tokens
        else:
            # Find the state that led to this node. We explicitly don't use the concept of "timestep" because of beam search        
            prev_step_state = self.prefix_states[prev_step_tuple]
            new_state = self._apply_new_characters(prev_step_state, token_sequence)
            self.prefix_states[sent_tuple] = new_state
            self._compute_allowed_tokens(sent_tuple, new_state)
            return new_state.allowed_tokens

    def _compute_allowed_tokens(self, state_tokens: Tuple, state: 'TokenEnforcer.OutputTensorState'):
        try:
            allowed_tokens: List[int] = []
            cache_key = state.parser.cache_key()
            if cache_key is not None and cache_key in self.allowed_token_cache:
                state.allowed_tokens = self.allowed_token_cache[cache_key]
                return
            shortcut_key = state.parser.shortcut_key()
            self._collect_allowed_tokens(state.parser, self.tokenizer_tree.root, allowed_tokens, shortcut_key)
            if state.parser.can_end():
                allowed_tokens.extend(self.eos_token_id if isinstance(self.eos_token_id, list) else [self.eos_token_id])
            if not allowed_tokens:
                raise ValueError(f"Parser reached state with no allowed tokens")
            # root_state = next(state for state in self.prefix_states.values() if state.parser == self.root_parser)
            # print(f"Allowing {len(allowed_tokens)} tokens after {state.str_so_far[len(root_state.str_so_far):]}")
            state.allowed_tokens = allowed_tokens
            if cache_key is not None:
                self.allowed_token_cache[cache_key] = allowed_tokens
        except LMFormatEnforcerException:
            # Getting an LMFormatEnforcerException means that we know what the user did wrong, 
            # and we can give a nice error message for them to fix.
            raise
        except Exception:
            # Other exceptions are potential bugs and should be reported
            logging.basicConfig(level=logging.ERROR)  # Initialize if no loggers
            prefix = self.decoder(list(state_tokens))
            logging.exception(f"Unknown LMFormatEnforcer Problem. Prefix: '{prefix}'\n"
                              "Terminating the parser. Please open an issue at \n"
                              "https://github.com/noamgat/lm-format-enforcer/issues with the prefix and "
                              "CharacterLevelParser parameters")
            state.allowed_tokens = self.eos_token_id if isinstance(self.eos_token_id, list) else [self.eos_token_id]

    def _collect_allowed_tokens(self, parser: CharacterLevelParser, tree_node: TokenizerPrefixTreeNode, allowed_tokens: List[int], shortcut_key: Optional[Hashable]):
        allowed_tokens.extend(tree_node.tokens)
        
        if isinstance(shortcut_key, tuple):
            if shortcut_key[0] == 'regex_state':
                _, pattern_hash, state = shortcut_key
                # Check state-level cache first
                if isinstance(parser, RegexParser):
                    if state in parser.context.state_token_cache:
                        allowed_tokens.extend(parser.context.state_token_cache[state])
                        return
                    # If not in state cache, collect tokens and cache them
                    state_tokens = []
                    allowed_characters = parser.get_allowed_characters()
                    relevant_characters = tree_node.children.keys()
                    characters_to_explore = set(relevant_characters).intersection(allowed_characters)
                    
                    for character in characters_to_explore:
                        next_parser = parser.add_character(character)
                        next_tree_node = tree_node.children[character]
                        self._collect_allowed_tokens(next_parser, next_tree_node, state_tokens, None)
                    
                    parser.context.state_token_cache[state] = state_tokens
                    allowed_tokens.extend(state_tokens)
                    return
            elif shortcut_key[0] == 'regex_pattern':
                # Existing regex pattern cache logic
                _, pattern_hash, cur_len = shortcut_key
                cache = self.tokenizer_tree.get_regex_pattern_cache(pattern_hash)
                if cache:
                    allowed_tokens.extend(cache.allowed_tokens)
                    return
                # If not cached yet, collect tokens and cache them
                pattern_tokens = []
                self._collect_allowed_tokens(parser, tree_node, pattern_tokens, None)
                self.tokenizer_tree.cache_regex_pattern(pattern_hash, pattern_tokens)
                allowed_tokens.extend(pattern_tokens)

        # Regular token collection logic
        allowed_characters = parser.get_allowed_characters()
        relevant_characters = tree_node.children.keys()
        characters_to_explore = set(relevant_characters).intersection(allowed_characters)
        
        for character in characters_to_explore:
            next_parser = parser.add_character(character)
            next_tree_node = tree_node.children[character]
            self._collect_allowed_tokens(next_parser, next_tree_node, allowed_tokens, None)
            
    def _apply_new_characters(self, state: 'TokenEnforcer.OutputTensorState', token_sequence: List[int]):
        new_state = TokenEnforcer.OutputTensorState(parser=state.parser)
        new_token = token_sequence[-1]
        if new_token in self.tokenizer_tree.new_word_tokens:
            new_state.current_word_tokens = [new_token]
            new_characters = self.tokenizer_tree.tokens_to_strs[new_token]
        else:
            new_state.current_word_tokens = state.current_word_tokens + [new_token]
            prev_decoded = self.decoder(state.current_word_tokens)
            new_decoded = self.decoder(new_state.current_word_tokens)
            new_characters = new_decoded[len(prev_decoded):]
        for character in new_characters:
            try:
                new_state.parser = new_state.parser.add_character(character)
            except Exception as e:
                # This can happen in beam / batch scenarios, when some of the batches finished but others are continuing.
                logging.debug(f"Received an invalid character '{character}', switching to ForceStopParser (Exception:{e})")
                new_state.parser = ForceStopParser()
        return new_state
        

