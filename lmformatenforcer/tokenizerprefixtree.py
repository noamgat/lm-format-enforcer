from typing import Dict, List, Tuple


class TokenizerPrefixTreeNode:
    def __init__(self):
        self.tokens: List[int] = []
        self.children: Dict[str, TokenizerPrefixTreeNode] = {}


class TokenizerPrefixTree:
    def __init__(self, regular_tokens: List[Tuple[int, str]]):
        self.root = TokenizerPrefixTreeNode()
        self.json_freetext_tokens: List[int] = []
        for token_idx, decoded in regular_tokens:
            self._add_token_to_tree(decoded, token_idx, self.root)
            # Performance optimization - cache the tokens of all the strings that don't contain a quote in the middle, or a line break.
            # When we are in a JSON freetext string field, they will all be permitted and this will save a lot of tree iterations.
            has_quote_before_end = '"' in decoded[0:-1]
            has_newline = "\n" in decoded or "\r" in decoded

            if not (has_quote_before_end or has_newline):
                self.json_freetext_tokens.append(token_idx)

    def _add_token_to_tree(self, token_str: str, token_idx: int, node: TokenizerPrefixTreeNode):
        for character in token_str:
            if character not in node.children:
                node.children[character] = TokenizerPrefixTreeNode()
            node = node.children[character]
        node.tokens.append(token_idx)
