from typing import Dict, List, Optional, Union

from numpy import rec
from .characterlevelparser import CharacterLevelParser
from .jsonschemaparser import JsonSchemaParser
from transformers.tokenization_utils import PreTrainedTokenizerBase
from .external.jsonschemaobject import JsonSchemaObject

from .tokenizerprefixtree import TokenizerPrefixTree, TokenizerPrefixTreeNode


class TokenEnforcer:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, parser: CharacterLevelParser):
        self.tokenizer = tokenizer
        self.token_0 = tokenizer.encode("0")[-1]
        self.str_so_far: str = None
        self.parser = parser
        self.tokenizer_tree = TokenizerPrefixTree(tokenizer)
        self.num_visited_nodes_by_timestep: List[int] = []
        self.num_tokens_allowed_by_timestep: List[int] = []
    
    def _decode_single_token(self, token: int) -> str:
        # We prepend token 0 and skip the first letter of the result to get a space if the token is a start word.
        decoded = self.tokenizer.decode([self.token_0, token])[1:]
        return decoded

    def filter_allowed_tokens(self, batch_id: int, sent: 'torch.Tensor') -> List[int]:
        self._apply_new_characters(sent)
        self.num_visited_nodes_by_timestep.append(0)
        shortcut_key = self.parser.shortcut_key()
        allowed_tokens: List[int] = []
        self._collect_allowed_tokens(self.parser, self.tokenizer_tree.root, allowed_tokens, shortcut_key)
        if self.parser.can_end():
            allowed_tokens.append(self.tokenizer.eos_token_id)
        self.num_tokens_allowed_by_timestep.append(len(allowed_tokens))
        return allowed_tokens

    def _collect_allowed_tokens(self, parser: CharacterLevelParser, tree_node: TokenizerPrefixTreeNode, allowed_tokens: List[int], shortcut_key: Optional[str]):
        self.num_visited_nodes_by_timestep[-1] += 1
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
            
    
    def _apply_new_characters(self, sent: 'torch.Tensor'):
        characters = self.tokenizer.decode(sent)
        if self.str_so_far is not None:
            new_characters = characters[len(self.str_so_far):]
            # print(f"Received new characters: '{new_characters}'")
            self.add_characters(new_characters)
        self.str_so_far = characters

    def add_characters(self, new_characters: str):
        for character in new_characters:
            self.parser = self.parser.add_character(character)
