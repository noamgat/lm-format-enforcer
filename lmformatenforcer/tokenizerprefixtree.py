from collections import OrderedDict
from typing import Dict, List, Set, Tuple
import json

class TokenizerPrefixTreeNode:
    def __init__(self) -> None:
        self.tokens: List[int] = []
        self.children: Dict[str, TokenizerPrefixTreeNode] = {}


class JsonFreetextTokenCache:
    """
    JSON string can contain almost any unicode character, so creating a list of allowed tokens is very expensive.
    The list can be cached, but JSON Schema also allows 'minLength' and 'maxLength' constraint on the string,
    that make some tokens illegal depending on how long the generated string is already. This class precalculates
    a separate allowlist for all possible constraint states up to maximum token length (16 in Llama, for example).
    After deduplication, this results in about ~75 lists for the Llama tokenizer.
    """
    class _StringLengthTokenCache:
        """This is an internal data structure, that given a list of string+token pairs,
        can quickly return all token ids of strings between certain lengths"""
        def __init__(self):
            self.tokens: List[int] = []
            self.first_index_geq_than_length: List[int] = [0]
        
        def build(self, token_strs_to_idx: List[Tuple[str, int]]):
            # TODO: If this becomes a performance bottleneck, bucket sort instead.
            token_strs_to_idx = sorted(token_strs_to_idx, key=lambda p:len(p[0]))
            self.tokens = [pair[1] for pair in token_strs_to_idx]
            # self.token_strs = [pair[0] for pair in token_strs_to_idx]  # For debugging
            token_lengths = [len(pair[0]) for pair in token_strs_to_idx]
            for idx, token_length in enumerate(token_lengths):
                while len(self.first_index_geq_than_length) <= token_length:
                    self.first_index_geq_than_length.append(idx)
            self.first_index_geq_than_length.append(len(token_lengths))

        def get_indices_between_length(self, min_length=-1, max_length=-1) -> List[int]:
            if min_length >= len(self.first_index_geq_than_length):
                return []
            start_index = self.first_index_geq_than_length[min_length] if min_length > 0 else 0
            if max_length == 0:
                end_index = 0
            elif max_length + 1 < len(self.first_index_geq_than_length):
                end_index = self.first_index_geq_than_length[max_length + 1]
            else:
                end_index = len(self.tokens)
            return self.tokens[start_index:end_index]

    def __init__(self, ) -> None:
        self.token_num_to_str: Dict[int, str] = {}
        self.allowlist_cache: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        self.max_token_len = 0
        self.regular_tokens_length_cache = JsonFreetextTokenCache._StringLengthTokenCache()
        self.quote_tokens_length_cache = JsonFreetextTokenCache._StringLengthTokenCache()

    def add_token(self, token_str: str, token_int: int):
        assert not self.allowlist_cache, "Cannot add more tokens after allowlists were precalculated"

        has_non_trailing_backslash = "\\" in token_str[:-1]
        has_quote_before_end = '"' in token_str[0:-1]
        has_newline = "\n" in token_str or "\r" in token_str
        if has_non_trailing_backslash or has_quote_before_end or has_newline:
            try:
                json.loads(f'"{token_str}"')
            except json.decoder.JSONDecodeError:
                return  # Illegal inside JSON string, skip this token

        if len(token_str) == 0:
            # Tokens that don't decode to anything should be ignored, will not be allowed in json freetext fields.
            # TODO: Should we instead ALWAYS allow them?
            return

        self.token_num_to_str[token_int] = token_str

    def lookup_allowed_tokens(self, min_remaining: int, max_len: int) -> Tuple[int, ...]:
        """
        Get the list of tokens that are allowed within a JSON string, such that:
        1. all candidate tokens are at most `max_len` characters long (excluding the trailing quote), and
        2. if a token ends with a quote, it's at least `min_remaining` chars long (excluding the quote).
        """
        cache_key = (min_remaining, max_len)
        if cache_key not in self.allowlist_cache:
            tokens_with_quote = self.quote_tokens_length_cache.get_indices_between_length(min_remaining + 1, max_len + 1)
            tokens_without_quote = self.regular_tokens_length_cache.get_indices_between_length(-1, max_len)
            combined = tokens_with_quote + tokens_without_quote
            self.allowlist_cache[cache_key] = tuple(combined)
        return self.allowlist_cache[cache_key]

    def freeze(self) -> None:
        """
        Precalculate token allowlists for all valid combinations of `min_remaining` and `max_len`
        based on the tokens that were added with `add_token()`.
        """
        all_tokens: List[Tuple[str, int]] = list((s, n) for n,s in self.token_num_to_str.items())
        assert all_tokens, "Cannot precalculate allowlists for an empty token list"
        assert not any(pair[0] == '' for pair in all_tokens), "Tokenizer must not contain empty tokens"

        regular_tokens: List[Tuple[str, int]] = []
        quote_tokens: List[Tuple[str, int]] = []
        for pair in all_tokens:
            if pair[0].endswith('"'):
                quote_tokens.append(pair)
            else:
                regular_tokens.append(pair)

        self.regular_tokens_length_cache.build(regular_tokens)
        self.quote_tokens_length_cache.build(quote_tokens)
        self.max_token_len = max(len(self.regular_tokens_length_cache.first_index_geq_than_length), 
                                 len(self.quote_tokens_length_cache.first_index_geq_than_length))
        del self.token_num_to_str


class TokenizerPrefixTree:
    def __init__(self, regular_tokens: List[Tuple[int, str, bool]]):
        self.root = TokenizerPrefixTreeNode()
        self.json_freetext_tokens = JsonFreetextTokenCache()
        self.new_word_tokens: Set[int] = set()
        self.tokens_to_strs = {token_idx: token_str for token_idx, token_str, _ in regular_tokens}
        for token_idx, decoded, is_new_word in regular_tokens:
            self._add_token_to_tree(decoded, token_idx, self.root)
            self.json_freetext_tokens.add_token(decoded, token_idx)
            if is_new_word:
                self.new_word_tokens.add(token_idx)

        self.json_freetext_tokens.freeze()

    def _add_token_to_tree(self, token_str: str, token_idx: int, node: TokenizerPrefixTreeNode):
        for character in token_str:
            if character not in node.children:
                node.children[character] = TokenizerPrefixTreeNode()
            node = node.children[character]
        node.tokens.append(token_idx)
