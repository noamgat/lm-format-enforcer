from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable, List, Optional, Tuple
import logging
from .characterlevelparser import CharacterLevelParser, ForceStopParser
from .tokenizerprefixtree import TokenizerPrefixTree, TokenizerPrefixTreeNode


class TokenEnforcer:
    """TokenEnforcer provides a token filtering mechanism, given a CharacterLevelParser and some information about the tokenizer.
    It is the main entry point for extending lm-format-enforcer to new inference libraries. See __init__() and get_allowed_tokens()"""
    @dataclass
    class OutputTensorState:
        str_so_far: str
        parser: CharacterLevelParser
        allowed_tokens: List[int] = field(default_factory=list)

    def __init__(self, regular_tokens: List[Tuple[int, str]], 
                 parser: CharacterLevelParser,
                 decoder: Callable[[List[int]], str],
                 eos_token_id: int):
        """
        Create a new TokenEnforcer.
        :param regular_tokens: A list of tuples (token_id, token_string) for all the regular (not special) tokens in the tokenizer vocabulary.
        Note that token_string is expected to include leading / trailing whitespaces if relevant.
        :param parser: A CharacterLevelParser that defines the allowed strings.
        :param decoder: A function that decodes a list of token ids into a string.
        :param eos_token_id: The token id of the end-of-string token.
        """
        self.prefix_states: Dict[Hashable, TokenEnforcer.OutputTensorState] = {}
        self.root_parser = parser
        self.tokenizer_tree = TokenizerPrefixTree(regular_tokens)
        self.decoder = decoder
        self.eos_token_id = eos_token_id

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
            state = TokenEnforcer.OutputTensorState(str_so_far=self.decoder(token_sequence),
                                                    parser=self.root_parser)
            self.prefix_states[sent_tuple] = state
            self._compute_allowed_tokens(state)
            return state.allowed_tokens
        else:
            # Find the state that led to this node. We explicitly don't use the concept of "timestep" because of beam search        
            prev_step_state = self.prefix_states[prev_step_tuple]
            new_state = self._apply_new_characters(prev_step_state, token_sequence)
            self.prefix_states[sent_tuple] = new_state
            self._compute_allowed_tokens(new_state)
            return new_state.allowed_tokens

    def _compute_allowed_tokens(self, state: 'TokenEnforcer.OutputTensorState'):
        allowed_tokens: List[int] = []
        shortcut_key = state.parser.shortcut_key()
        self._collect_allowed_tokens(state.parser, self.tokenizer_tree.root, allowed_tokens, shortcut_key)
        if state.parser.can_end():
            allowed_tokens.append(self.eos_token_id)
        if not allowed_tokens:
            raise ValueError(f"Parser reached state with no allowed tokens")
        # root_state = next(state for state in self.prefix_states.values() if state.parser == self.root_parser)
        # print(f"Allowing {len(allowed_tokens)} tokens after {state.str_so_far[len(root_state.str_so_far):]}")
        state.allowed_tokens = allowed_tokens

    def _collect_allowed_tokens(self, parser: CharacterLevelParser, tree_node: TokenizerPrefixTreeNode, allowed_tokens: List[int], shortcut_key: Optional[str]):
        allowed_tokens.extend(tree_node.tokens)
        allowed_characters = parser.get_allowed_characters()
        relevant_characters = tree_node.children.keys()
        # This next line is the heart of the traversal algorithm. We only explore paths that are shared by both the parser and the tokenizer.
        characters_to_explore = set(relevant_characters).intersection(allowed_characters)
        
        # Performance optimization: If we are in JSON freetext, all of the tokens that don't contain quote, or end with quote, are legal, so we take
        # their cached list. If the quote character is allowed, we only need to dynamically explore the cases where the string starts with a quote.
        # This breaks the elegance of the API, but otherwise it is a huge performance hit.
        if shortcut_key == 'json_freetext':
            allowed_tokens.extend(self.tokenizer_tree.json_freetext_tokens)
            characters_to_explore = characters_to_explore.intersection(['"'])

        for character in characters_to_explore:
            next_parser = parser.add_character(character )
            next_tree_node = tree_node.children[character]
            self._collect_allowed_tokens(next_parser, next_tree_node, allowed_tokens, None)
            
    def _apply_new_characters(self, state: 'TokenEnforcer.OutputTensorState', token_sequence: List[int]):
        characters = self.decoder(token_sequence)
        new_state = TokenEnforcer.OutputTensorState(str_so_far=characters, parser=state.parser)
        new_characters = characters[len(state.str_so_far):]
        for character in new_characters:
            if character in new_state.parser.get_allowed_characters():
                new_state.parser = new_state.parser.add_character(character)
            else:
                # This can happen in beam / batch scenarios, when some of the batches finished but others are continuing.
                logging.debug(f"Received an invalid character '{character}', switching to ForceStopParser")
                new_state.parser = ForceStopParser()
        return new_state
        

    