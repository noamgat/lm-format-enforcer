from dataclasses import dataclass, field
from typing import Dict, Hashable, List, Optional, Union
import logging

from numpy import rec
from .characterlevelparser import CharacterLevelParser, ForceStopParser
from .jsonschemaparser import JsonSchemaParser
from transformers.tokenization_utils import PreTrainedTokenizerBase
from .external.jsonschemaobject import JsonSchemaObject

from .tokenizerprefixtree import TokenizerPrefixTree, TokenizerPrefixTreeNode


class TokenEnforcer:
    @dataclass
    class OutputTensorState:
        str_so_far: str
        parser: CharacterLevelParser
        allowed_tokens: List[int] = field(default_factory=list)

    def __init__(self, tokenizer: PreTrainedTokenizerBase, parser: CharacterLevelParser):
        self.tokenizer = tokenizer
        self.token_0 = tokenizer.encode("0")[-1]
        self.prefix_states: Dict[Hashable, TokenEnforcer.OutputTensorState] = {}
        self.root_parser = parser
        self.tokenizer_tree = TokenizerPrefixTree(tokenizer)
    
    def _decode_single_token(self, token: int) -> str:
        # We prepend token 0 and skip the first letter of the result to get a space if the token is a start word.
        decoded = self.tokenizer.decode([self.token_0, token])[1:]
        return decoded

    def filter_allowed_tokens(self, batch_id: int, sent: 'torch.Tensor') -> List[int]:
        # In order to elegantly support beam search and batching, we don't store per-batch information.
        # Instead, we store a hash of all the states (unique token tensors) we encountered so far.
        # When we encounter a new unique token tensor, we find the token tensor that led to it, and continue from there.

        sent_tuple = tuple(sent.tolist())
        prev_step_tuple = sent_tuple[:-1]

        if sent_tuple in self.prefix_states:
            # We already calculated for this node, return cached list
            return self.prefix_states[sent_tuple].allowed_tokens
        elif prev_step_tuple not in self.prefix_states:
            # We have not encountered the tensor up to the before-last entry. This means that this is the first call - the instruction / prompt tensor.
            # Initialize the root node
            state = TokenEnforcer.OutputTensorState(str_so_far=self.tokenizer.decode(sent),
                                                    parser=self.root_parser)
            self.prefix_states[sent_tuple] = state
            self._compute_allowed_tokens(state)
            return state.allowed_tokens
        else:
            # Find the state that led to this node. We explicitly don't use the concept of "timestep" because of beam search        
            prev_step_state = self.prefix_states[prev_step_tuple]
            new_state = self._apply_new_characters(prev_step_state, sent)
            self.prefix_states[sent_tuple] = new_state
            self._compute_allowed_tokens(new_state)
            return new_state.allowed_tokens

    def _compute_allowed_tokens(self, state: 'TokenEnforcer.OutputTensorState'):
        allowed_tokens: List[int] = []
        shortcut_key = state.parser.shortcut_key()
        self._collect_allowed_tokens(state.parser, self.tokenizer_tree.root, allowed_tokens, shortcut_key)
        if state.parser.can_end():
            allowed_tokens.append(self.tokenizer.eos_token_id)
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
            
    
    def _apply_new_characters(self, state: 'TokenEnforcer.OutputTensorState', sent: 'torch.Tensor'):
        characters = self.tokenizer.decode(sent)
        new_state = TokenEnforcer.OutputTensorState(str_so_far=characters, parser=state.parser)
        new_characters = characters[len(state.str_so_far):]
        for character in new_characters:
            if character in new_state.parser.get_allowed_characters():
                new_state.parser = new_state.parser.add_character(character)
            else:
                logging.warning(f"Received an invalid character '{character}', switching to ForceStopParser")
                new_state.parser = ForceStopParser()
        return new_state
        

    