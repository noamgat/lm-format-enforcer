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
    def __init__(self, ) -> None:
        self.token_num_to_str: Dict[int, str] = {}
        self.allowlist_cache: Dict[Tuple[int, int], Tuple[int, ...]] = {}
        self.max_token_len = 0
        self.max_token_int = -1
        self.max_allowed_token_len = 32

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
        self.max_token_int = max(self.max_token_int, token_int)
        self.max_token_len = min(max(self.max_token_len, len(token_str)), self.max_allowed_token_len)

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
        all_tokens: List[str] = [self.token_num_to_str.get(i, None) for i in range(self.max_token_int + 1)]
        assert all_tokens, "Cannot precalculate allowlists for an empty token list"
        assert not any(t == '' for t in all_tokens), "Tokenizer must not contain empty tokens"

        def _valid_for_min_remaining(token, min_remaining):
            return token is not None and (not token.endswith('"') or len(token.rstrip('"')) >= min_remaining)

        def _valid_for_max_len(token, max_len):
            return token is not None and len(token.rstrip('"')) <= max_len

        # Precalculate valid token sets
        valid_for_min_remaining_sets = []
        for min_remaining in range(self.max_token_len + 1):
            valid_for_min_remaining_sets.append(set([
                i for i in range(len(all_tokens))
                if _valid_for_min_remaining(all_tokens[i], min_remaining)
            ]))

        valid_for_max_len_sets = []
        for max_len in range(self.max_token_len + 1):
            valid_for_max_len_sets.append(set([
                i for i in range(len(all_tokens))
                if _valid_for_max_len(all_tokens[i], max_len)
            ]))

        # Make a 2D array of constrained allowlists, indexed by tuple `(min_remaining, max_len)`
        token_lists = {}
        for min_remaining in range(self.max_token_len + 1):
            for max_len in range(min_remaining, self.max_token_len + 1):
                ids = tuple(valid_for_min_remaining_sets[min_remaining] & valid_for_max_len_sets[max_len])
                token_lists[(min_remaining, max_len)] = ids

        # Deduplicate the lists to save RAM as many of them will be identical
        unique_lists = set(token_lists.values())
        for key, lst in token_lists.items():
            for uniq in unique_lists:
                if len(uniq) == len(lst) and uniq == lst:
                    token_lists[key] = uniq
                    break

        self.allowlist_cache = token_lists
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
