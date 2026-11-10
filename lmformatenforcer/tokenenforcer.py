from dataclasses import dataclass, field, replace
import sys
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Tuple, Union
import logging

from .exceptions import LMFormatEnforcerException
from .characterlevelparser import CharacterLevelParser, ForceStopParser, CharacterLevelParserConfig, SequenceParser, UnionParser
from .tokenizerprefixtree import TokenizerPrefixTree, TokenizerPrefixTreeNode, ByteTokenizerPrefixTree
from .utf8utils import advance_utf8
from .consts import COMPLETE_ALPHABET
from .tokenlist import TokenList


class TokenEnforcerTokenizerData:
    """TokenEnforcerTokenizerData contains all of the preprocessing for preparing the TokenEnforcer to work with a 
    specific tokenizer. It does some calculations, so it is recommended to reuse it for multiple TokenEnforcers"""
    def __init__(self, 
                 regular_tokens: List[Tuple[int, str, bool]], 
                 decoder: Callable[[List[int]], str],
                 eos_token_id: Union[int, List[int]],
                 use_bitmask: bool,
                 vocab_size: int,
                 raw_token_bytes: Optional[Mapping[int, bytes]] = None):
        """
        Create the tokenizer data that the TokenEnforcer needs. This can be reused for multiple TokenEnforcers if they work with the same tokenizer.
        :param regular_tokens: A list of tuples (token_id, token_string, is_new_word_token) for all the regular (not special) tokens in the tokenizer vocabulary.
        Note that token_string is expected to include leading / trailing whitespaces if relevant.
        :param decoder: A function that decodes a list of token ids into a string.
        :param eos_token_id: The token id(s) of the end-of-string token(s).
        :param raw_token_bytes: Optional exact byte mapping for every regular token.
            Must include all 256 single-byte tokens; incomplete metadata is rejected.
            When omitted, the legacy string-decoding behavior is unchanged.
        """
        filtered_regular_tokens = [token_tuple for token_tuple in regular_tokens if token_tuple[0] <= vocab_size]
        self.regular_tokens = filtered_regular_tokens
        self.raw_token_bytes = None
        self.byte_tree = None
        if raw_token_bytes is not None:
            ids = {token_id for token_id, _, _ in filtered_regular_tokens}
            if set(raw_token_bytes) != ids or any(not isinstance(b, bytes) or not b for b in raw_token_bytes.values()):
                raise ValueError("raw_token_bytes must contain nonempty bytes for every regular token")
            if {bytes([i]) for i in range(256)} - set(raw_token_bytes.values()):
                raise ValueError("raw_token_bytes must include every single-byte token")
            self.raw_token_bytes = dict(raw_token_bytes)
            self.byte_tree = ByteTokenizerPrefixTree(self.raw_token_bytes, use_bitmask, vocab_size)
        self.tokenizer_tree = TokenizerPrefixTree(self.regular_tokens, use_bitmask, vocab_size)
        self.decoder = decoder
        self.eos_token_id = eos_token_id
        self.tokenizer_alphabet = "".join(token_str for token_str in self.tokenizer_tree.root.children.keys() if len(token_str) == 1)
        self.vocab_size = vocab_size
        self.use_bitmask = use_bitmask


class TokenEnforcer:
    """TokenEnforcer provides a token filtering mechanism, given a CharacterLevelParser and some information about the tokenizer.
    It is the main entry point for extending lm-format-enforcer to new inference libraries. See __init__() and get_allowed_tokens()"""
    @dataclass
    class OutputTensorState:
        parser: CharacterLevelParser
        allowed_tokens: TokenList | None = field(default=None)
        current_word_tokens: List[int] = field(default_factory=list)
        pending_utf8: bytes = b''
        

    def __init__(self, tokenizer_data: TokenEnforcerTokenizerData, parser: CharacterLevelParser):
        """
        Create a new TokenEnforcer.
        :param tokenizer_data: Per tokenizer data that the token enforcer needs in order to operate.
        :param parser: A CharacterLevelParser that defines the allowed strings.
        """
        self.prefix_states: Dict[Tuple, TokenEnforcer.OutputTensorState] = {}
        self.root_parser = parser
        self.tokenizer_tree = tokenizer_data.tokenizer_tree
        self.byte_tree = tokenizer_data.byte_tree
        self.raw_token_bytes = tokenizer_data.raw_token_bytes
        self.decoder = tokenizer_data.decoder
        self.eos_token_id = tokenizer_data.eos_token_id
        self.regular_tokens = tokenizer_data.regular_tokens
        self.allowed_token_cache: Dict[Hashable, Any] = {}
        self.use_bitmask = tokenizer_data.use_bitmask
        self.vocab_size = tokenizer_data.vocab_size
        
        
        # Preserve the parser's existing config (e.g., force_json_field_order) when updating alphabet
        if self.byte_tree:
            def configure(p):
                p.config = replace(p.config, byte_mode=True, allow_unicode=p.config.alphabet == COMPLETE_ALPHABET)
                if isinstance(p, (SequenceParser, UnionParser)):
                    for child in p.parsers:
                        configure(child)
            configure(parser)
        else:
            parser.config = replace(parser.config, alphabet=tokenizer_data.tokenizer_alphabet)

    def get_allowed_tokens(self, token_sequence: List[int]) -> TokenList:
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
            allowed_tokens: TokenList = TokenList(self.use_bitmask, self.vocab_size)
            
            cache_key = state.parser.cache_key()
            if cache_key is not None and self.byte_tree:
                cache_key = (cache_key, state.pending_utf8)
            if cache_key is not None and cache_key in self.allowed_token_cache:
                state.allowed_tokens = self.allowed_token_cache[cache_key]
                return
            shortcut_key = state.parser.shortcut_key()
            if self.byte_tree:
                self._collect_byte_tokens(state.parser, state.pending_utf8, allowed_tokens, shortcut_key)
            else:
                self._collect_allowed_tokens(state.parser, self.tokenizer_tree.root, allowed_tokens, shortcut_key)
            if not state.pending_utf8 and state.parser.can_end():
                if isinstance(self.eos_token_id, list):
                    allowed_tokens.extend(self.eos_token_id)                    
                else:
                    allowed_tokens.append(self.eos_token_id)
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
            state.allowed_tokens = TokenList(self.use_bitmask, self.vocab_size)
            if state.pending_utf8:
                return
            if isinstance(self.eos_token_id, list):
                state.allowed_tokens.extend(self.eos_token_id)
            else:
                state.allowed_tokens.append(self.eos_token_id)

    def _collect_byte_tokens(self, parser, pending, allowed, shortcut_key):
        tree = self.byte_tree
        if (not pending and parser.config.allow_unicode and isinstance(shortcut_key, tuple)
                and shortcut_key[0] == 'json_freetext'):
            _, cur_len, min_len, max_len = shortcut_key
            cache = tree.json_freetext_tokens
            if cache is not None:
                min_remaining = min(cache.max_token_len, max(0, min_len - cur_len))
                max_allowed = min(cache.max_token_len, max_len - cur_len)
                allowed.extend(cache.lookup_allowed_tokens(min_remaining, max_allowed).allowed_tokens)
            root = tree.json_exceptions
        else:
            root = tree.root

        def walk(node, current_parser, partial):
            allowed.extend(node.tokens)
            for byte, child in node.children.items():
                result = advance_utf8(partial, byte)
                if result is None:
                    continue
                next_partial, character, reachable = result
                if character is not None:
                    if not current_parser.allows_character(character):
                        continue
                    # A leaf needs only acceptance, not the parser state for a
                    # nonexistent next byte. In particular, constructing the
                    # JSON escape parser for a lone backslash is expensive.
                    next_parser = current_parser.add_character(character) if child.children else current_parser
                else:
                    if not current_parser.has_allowed_in_range(*reachable):
                        continue
                    next_parser = current_parser
                walk(child, next_parser, next_partial)
        walk(root, parser, pending)

    def _collect_allowed_tokens(self, parser: CharacterLevelParser, tree_node: TokenizerPrefixTreeNode, allowed_tokens: TokenList, shortcut_key: Optional[Hashable]):
        allowed_tokens.extend(tree_node.tokens)
        allowed_characters = parser.get_allowed_characters()
        relevant_characters = tree_node.children.keys()
        # This next line is the heart of the traversal algorithm. We only explore paths that are shared by both the parser and the tokenizer.
        characters_to_explore = set(relevant_characters).intersection(allowed_characters)
        
        # Performance optimization: If we are in JSON freetext, all of the tokens that don't contain quote, or end with quote, are legal, so we take
        # their cached list. If the quote character is allowed, we only need to dynamically explore the cases where the string starts with a quote.
        # This breaks the elegance of the API, but otherwise it is a huge performance hit.
        if isinstance(shortcut_key, tuple) and shortcut_key[0] == 'json_freetext':
            assert len(shortcut_key) == 4
            _, cur_len, min_len, max_len = shortcut_key
            cache = self.tokenizer_tree.json_freetext_tokens

            min_remaining = min(cache.max_token_len, max(0, min_len - cur_len))  # no " allowed before this many chars
            max_allowed_len = min(cache.max_token_len, max_len - cur_len)  # max new characters allowed (before ")

            allowed_tokens.extend(cache.lookup_allowed_tokens(min_remaining, max_allowed_len).allowed_tokens)
            characters_to_explore = characters_to_explore.intersection(['"'])

        for character in characters_to_explore:
            next_parser = parser.add_character(character)
            next_tree_node = tree_node.children[character]
            self._collect_allowed_tokens(next_parser, next_tree_node, allowed_tokens, None)
            
    def _apply_new_characters(self, state: 'TokenEnforcer.OutputTensorState', token_sequence: List[int]):
        new_state = TokenEnforcer.OutputTensorState(parser=state.parser)
        new_token = token_sequence[-1]
        if self.byte_tree:
            new_state.pending_utf8 = state.pending_utf8
            raw = self.raw_token_bytes.get(new_token)
            if raw is None:
                new_state.parser = ForceStopParser()
                return new_state
            for byte in raw:
                result = advance_utf8(new_state.pending_utf8, byte)
                if result is None:
                    new_state.parser = ForceStopParser()
                    new_state.pending_utf8 = b''
                    return new_state
                new_state.pending_utf8, character, _ = result
                if character is not None:
                    if not new_state.parser.allows_character(character):
                        new_state.parser = ForceStopParser()
                        return new_state
                    new_state.parser = new_state.parser.add_character(character)
            return new_state
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
        

