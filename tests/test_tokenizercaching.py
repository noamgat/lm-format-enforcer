
from lmformatenforcer.tokenizerprefixtree import JsonFreetextTokenCache

def test_json_freetext_cache():
    token_to_str = {}
    cache = JsonFreetextTokenCache(False, 0)
    test_length = 500
    letters = "abcde"
    num_letters = len(letters)
    def _register_token(token_str: str):
        token_idx = len(token_to_str)
        token_to_str[token_idx] = token_str
        cache.add_token(token_str, token_idx)
    _register_token("\"")
    for i in range(1, test_length):
        for letter in letters:
            _register_token(letter * i)
            _register_token(letter * i + '"')
    cache.freeze()

    def _assert_allowed_tokens(_min_remaining, _max_length, _num_expected_tokens):
        allowed_tokens = cache.lookup_allowed_tokens(_min_remaining, _max_length).allowed_tokens
        if len(allowed_tokens) != _num_expected_tokens:
            allowed_token_strs = "|".join(token_to_str[token_idx] for token_idx in allowed_tokens)
            raise Exception(f"Min={_min_remaining}, Max={_max_length}, Expected {_num_expected_tokens}, got {len(allowed_tokens)} : {allowed_token_strs}")
        
    for min_remaining in range(0, test_length):
        for max_length in range(min_remaining, test_length):
            
            num_expected_quote_tokens = num_letters * (max_length - min_remaining + 1)
            if min_remaining == 0:
                # at 0, there is only one quoted string (")
                num_expected_quote_tokens -= (num_letters - 1)
                
            num_expected_regular_tokens = max_length * num_letters
            num_expected_tokens = num_expected_quote_tokens + num_expected_regular_tokens
            _assert_allowed_tokens(min_remaining, max_length, num_expected_tokens)
            
    _assert_allowed_tokens(0, test_length + 1, len(token_to_str))
    num_nonquote_tokens = (test_length - 1) * num_letters
    _assert_allowed_tokens(test_length + 1, test_length + 1, num_nonquote_tokens)


def test_allowed_token_cache_distinguishes_regex_parsers():
    # Regression test for https://github.com/noamgat/lm-format-enforcer/issues/193
    # Two RegexParser instances can reach numerically identical FSM states while
    # allowing different next characters. The TokenEnforcer allowed-token cache
    # must key on the regex identity, not just the numeric state.
    from lmformatenforcer import RegexParser, UnionParser
    from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData

    vocabulary = ['a', 'b', 'x', 'y']
    data = TokenEnforcerTokenizerData(
        [(i, text, True) for i, text in enumerate(vocabulary)],
        lambda ids: ''.join(vocabulary[i] for i in ids),
        eos_token_id=4,
        use_bitmask=False,
        vocab_size=5,
    )
    parser = UnionParser([RegexParser('ax'), RegexParser('by')])
    enforcer = TokenEnforcer(data, parser)

    def allowed(prefix):
        tokens = enforcer.get_allowed_tokens(prefix)
        return [text for i, text in enumerate(vocabulary)
                if tokens.is_token_allowed(i)]

    assert allowed([]) == ['a', 'b']
    assert allowed([0]) == ['x']  # after 'a', only 'x' continues 'ax'
    assert allowed([1]) == ['y']  # after 'b', only 'y' continues 'by'
