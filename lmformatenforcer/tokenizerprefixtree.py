from collections import OrderedDict
from typing import Dict, List, Set, Tuple
import json

class TokenizerPrefixTreeNode:
    def __init__(self) -> None:
        self.tokens: List[int] = []
        self.children: Dict[str, TokenizerPrefixTreeNode] = {}


class JsonFreetextTokenCache:
    def __init__(self, ) -> None:
        self.token_str_to_num: Dict[str, int] = {}
        self.allowlist_cache: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        self.max_token_len = 0


    def add_token(self, token_str: str, token_int: int):
        assert not self.allowlist_cache, "Cannot add more tokens after allowlists were precalculated"
        self.token_str_to_num[token_str] = token_int
        self.max_token_len = max(self.max_token_len, len(token_str))


    def lookup_allowed_tokens(self, min_remaining: int, max_len: int) -> Tuple[int, ...]:
        """
        Get the list of tokens that are allowed within a JSON string, such that:
        1. all candidate tokens are at most `max_len` characters long (excluding the trailing quote), and
        2. if a token ends with a quote, it's at least `min_remaining` chars long (excluding the quote).
        """
        return self.allowlist_cache[(min_remaining, max_len)]


    def freeze(self) -> None:
        """
        Precalculate token allowlists for all valid combinations of `min_remaining` and `max_len`
        based on the tokens that were added with `add_token()`.
        """
        all_tokens: List[str] = sorted(self.token_str_to_num.keys())
        assert all_tokens, "Cannot precalculate allowlists for an empty token list"
        assert not any(t == '' for t in all_tokens), "Tokenizer must not contain empty tokens"

        def _valid_for_min_remaining(token, min_remaining):
            return not token.endswith('"') or len(token.rstrip('"')) >= min_remaining

        def _valid_for_max_len(token, max_len):
            return len(token.rstrip('"')) <= max_len

        # Make a 2D array of constrained allowlists, indexed by tuple `(min_remaining, max_len)`
        token_lists = {}
        max_token_len = max(len(token) for token in all_tokens)
        for min_remaining in range(max_token_len + 1):
            for max_len in range(max_token_len + 1):
                if max_len >= min_remaining:  # Skip combinations that are never used
                    token_lists[(min_remaining, max_len)] = tuple(sorted([
                        token for token in all_tokens
                        if _valid_for_min_remaining(token, min_remaining) and _valid_for_max_len(token, max_len)
                    ]))

        # Deduplicate the lists to save RAM as many of them will be identical
        unique_lists = set(token_lists.values())
        for key, lst in token_lists.items():
            for uniq in unique_lists:
                if len(uniq) == len(lst) and uniq == lst:
                    token_lists[key] = uniq
                    break

        # Turn token strings into token numbers
        self.allowlist_cache = {
            key: tuple(self.token_str_to_num[t] for t in lst)
            for key, lst in token_lists.items()
        }
        del self.token_str_to_num


class TokenizerPrefixTree:
    def __init__(self, regular_tokens: List[Tuple[int, str, bool]]):
        self.root = TokenizerPrefixTreeNode()
        self.json_freetext_tokens = JsonFreetextTokenCache()
        self.new_word_tokens: Set[int] = set()
        self.tokens_to_strs = {token_idx: token_str for token_idx, token_str, _ in regular_tokens}
        for token_idx, decoded, is_new_word in regular_tokens:
            self._add_token_to_tree(decoded, token_idx, self.root)

            # Performance optimization - cache the tokens of all the strings that don't contain a quote in the middle, or a line break.
            # When we are in a JSON freetext string field, they will all be permitted and this will save a lot of tree iterations.
            has_non_trailing_backslash = '\\' in decoded[:-1]
            has_quote_before_end = '"' in decoded[0:-1]
            has_newline = "\n" in decoded or "\r" in decoded
            if has_non_trailing_backslash or has_quote_before_end or has_newline:  # Could be illegal inside a JSON string, test it
                try:
                    json.loads(f'"{decoded}"')
                except json.decoder.JSONDecodeError:
                    continue

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
