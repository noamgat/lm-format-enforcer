
from lmformatenforcer.tokenizerprefixtree import JsonFreetextTokenCache


def test_json_freetext_cache():
    token_to_str = {}
    cache = JsonFreetextTokenCache()
    test_length = 20
    cache.max_allowed_token_len = test_length
    def _register_token(token_idx: int, token_str: str):
        token_to_str[token_idx] = token_str
        cache.add_token(token_str, token_idx)
    for i in range(1, test_length):
        _register_token(i, "a" * i)
        _register_token(i + cache.max_allowed_token_len, "a" * i + '"')
    cache.freeze()
    for min_remaining in range(1, test_length):
        for max_length in range(min_remaining, test_length):
            allowed_tokens = cache.lookup_allowed_tokens(min_remaining, max_length)
            num_expected_quote_tokens = max_length - min_remaining + 1
            num_expected_regular_tokens = max_length
            assert len(allowed_tokens) == num_expected_quote_tokens + num_expected_regular_tokens
