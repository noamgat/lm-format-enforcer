"""Regression tests for UTF-8 split across tokenizer tokens."""
import pytest

from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig, StringParser
from lmformatenforcer.jsonschemaparser import JsonSchemaParser
from lmformatenforcer.regexparser import RegexParser
from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData


def enforcer(parser, extra=(), use_bitmask=False):
    raw = {i: bytes([i]) for i in range(256)}
    raw.update({256 + i: b for i, b in enumerate(extra)})
    regular = [(i, b.decode('utf-8', errors='replace'), False) for i, b in raw.items()]
    decoder = lambda ids: b''.join(raw[i] for i in ids if i in raw).decode('utf-8', errors='replace')
    data = TokenEnforcerTokenizerData(regular, decoder, 500, use_bitmask, 501, raw_token_bytes=raw)
    return TokenEnforcer(data, parser)


def allowed(e, seq):
    return set(e.get_allowed_tokens(seq).allowed_tokens)


@pytest.mark.parametrize('text', ['Ł', '中', '😀', '\ufffd'])
def test_split_string(text):
    e = enforcer(StringParser(text))
    seq = [999]
    for b in text.encode('utf-8'):
        assert b in allowed(e, seq)
        seq.append(b)
    assert 500 in allowed(e, seq)


def test_json_pending_and_length():
    e = enforcer(JsonSchemaParser({'type': 'string', 'minLength': 1, 'maxLength': 1}))
    seq = [999]
    assert 34 in allowed(e, seq)
    seq += [34]
    assert 0xc5 in allowed(e, seq)
    seq += [0xc5]
    assert 0x81 in allowed(e, seq)
    assert not ({34, 500, 65, 0xc0} & allowed(e, seq))
    seq += [0x81]
    assert 34 in allowed(e, seq)
    assert 0xc5 not in allowed(e, seq)
    assert 500 in allowed(e, seq + [34])


@pytest.mark.parametrize('pattern,text', [('Ł', 'Ł'), ('.', '😀')])
def test_regex_split(pattern, text):
    e = enforcer(RegexParser(pattern))
    seq = [999]
    for b in text.encode():
        assert b in allowed(e, seq)
        seq.append(b)
    assert 500 in allowed(e, seq)


def test_invalid_prefixes_and_branches():
    e = enforcer(JsonSchemaParser({'type': 'string'}))
    start = [999, 34]
    # Prime root/prompt state before extending sequence.
    allowed(e, [999]); allowed(e, start)
    a = allowed(e, start)
    assert not ({0x80, 0xc0, 0xc1, 0xf5, 0xff} & a)
    assert 0xe0 in a and 0xed in a and 0xf4 in a
    assert 0x9f not in allowed(e, start + [0xe0])
    assert 0xa0 not in allowed(e, start + [0xed])
    assert 0x90 not in allowed(e, start + [0xf4])
    assert 0xa0 in allowed(e, start + [0xe0])
    assert 0x80 in allowed(e, start + [0xed])


def test_custom_alphabet():
    config = CharacterLevelParserConfig(alphabet='abc')
    e = enforcer(JsonSchemaParser({'type': 'string'}, config))
    allowed(e, [999])
    a = allowed(e, [999, 34])
    assert ord('a') in a
    assert 0xc5 not in a


def test_json_pattern_wildcard():
    e = enforcer(JsonSchemaParser({'type': 'string', 'pattern': '.'}))
    seq = [999]
    allowed(e, seq)
    seq.append(34)
    for b in '😀'.encode():
        assert b in allowed(e, seq)
        seq.append(b)
    assert 34 in allowed(e, seq)


def test_config_positional_compatibility():
    config = CharacterLevelParserConfig('abc', 3, True, 9)
    assert (config.alphabet, config.max_consecutive_whitespaces,
            config.force_json_field_order, config.max_json_array_length) == ('abc', 3, True, 9)
    assert not config.allow_unicode


def test_fast_bytelevel_public_api():
    tokenizers = pytest.importorskip('tokenizers')
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('torch')
    from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

    # Native ByteLevel's published reversible byte-to-Unicode mapping.
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = bs[:]
    for b in range(256):
        if b not in bs:
            bs.append(b)
            chars.append(256 + len(chars) - 188)
    byte_chars = dict(zip(bs, map(chr, chars)))
    vocab = {byte_chars[b]: b for b in range(256)}
    vocab['<eos>'] = 256
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token='<eos>'))
    backend.decoder = tokenizers.decoders.ByteLevel()
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>')
    data = build_token_enforcer_tokenizer_data(tokenizer)
    e = TokenEnforcer(data, JsonSchemaParser({'type': 'string', 'enum': ['Ł']}))
    seq = [999]
    assert 34 in allowed(e, seq)
    seq.append(34)
    assert 0xc5 in allowed(e, seq)
    seq.append(0xc5)
    assert 0x81 in allowed(e, seq)


def test_custom_alphabet_allows_json_delimiter():
    e = enforcer(JsonSchemaParser({'type': 'string'}, CharacterLevelParserConfig(alphabet='abc')))
    allowed(e, [999])
    assert 34 in allowed(e, [999, 34])
    assert ord('x') not in allowed(e, [999, 34])


def test_multi_byte_token_and_invalid_scalar_bytes():
    extra = [b'\xc5\x81', b'\xed\xa0\x80', b'\xf4\x90\x80\x80', b'\xef\xbf\xbd']
    e = enforcer(JsonSchemaParser({'type': 'string'}), extra)
    allowed(e, [999])
    a = allowed(e, [999, 34])
    assert 256 in a and 259 in a
    assert 257 not in a and 258 not in a


def test_json_pattern_rejects_unescaped_control():
    e = enforcer(JsonSchemaParser({'type': 'string', 'pattern': '.'}))
    allowed(e, [999])
    a = allowed(e, [999, 34])
    assert not ({0, 10, 13, 31} & a)


def test_custom_regex_alphabet_and_literal():
    cfg = CharacterLevelParserConfig(alphabet='abc')
    e = enforcer(RegexParser('.', cfg))
    assert ord('a') in allowed(e, [999])
    assert 0xc5 not in allowed(e, [999])
    e = enforcer(RegexParser('Ł', cfg))
    assert 0xc5 in allowed(e, [999])


def test_pending_prefix_cache_isolated():
    e = enforcer(RegexParser('[ŁĀ]'))
    root = [999]
    a = allowed(e, root)
    assert {0xc4, 0xc5} <= a
    assert 0x80 in allowed(e, root + [0xc4])
    assert 0x81 not in allowed(e, root + [0xc4])
    assert 0x81 in allowed(e, root + [0xc5])
    assert 0x80 not in allowed(e, root + [0xc5])


def test_json_zero_max_length():
    e = enforcer(JsonSchemaParser({'type': 'string', 'maxLength': 0}))
    allowed(e, [999])
    a = allowed(e, [999, 34])
    assert 34 in a
    assert ord('a') not in a and 0xc5 not in a


def test_composite_wildcard():
    from lmformatenforcer.characterlevelparser import SequenceParser
    e = enforcer(SequenceParser([StringParser(':'), RegexParser('.')]))
    allowed(e, [999])
    assert 0xf0 in allowed(e, [999, ord(':')])


def test_raw_metadata_must_cover_tokens_and_all_bytes():
    tokens = [(i, chr(i), False) for i in range(128)]
    with pytest.raises(ValueError, match='every single-byte'):
        TokenEnforcerTokenizerData(tokens, lambda _: '', 500, False, 501,
                                   raw_token_bytes={i: bytes([i]) for i in range(128)})
    tokens = [(i, chr(i), False) for i in range(256)]
    with pytest.raises(ValueError, match='every regular token'):
        TokenEnforcerTokenizerData(tokens, lambda _: '', 500, False, 501,
                                   raw_token_bytes={i: bytes([i]) for i in range(255)})


def test_partial_tail_after_complete_scalar_and_enum():
    e = enforcer(JsonSchemaParser({'type': 'string', 'enum': ['AŁ']}), [b'A\xc5'])
    allowed(e, [999])
    s = [999, 34]
    assert 256 in allowed(e, s)
    assert 0x81 in allowed(e, s + [256])
    assert 34 not in allowed(e, s + [256])
    assert 34 in allowed(e, s + [256, 0x81])


def test_partial_tail_respects_character_max_length():
    e = enforcer(JsonSchemaParser({'type': 'string', 'maxLength': 1}), [b'A\xc5'])
    allowed(e, [999])
    assert 256 not in allowed(e, [999, 34])


def test_legacy_without_raw_bytes():
    tokens = [(0, 'a', False), (1, 'b', False)]
    data = TokenEnforcerTokenizerData(tokens, lambda ids: ''.join('ab'[i] for i in ids), 2, False, 3)
    e = TokenEnforcer(data, StringParser('a'))
    assert allowed(e, []) == {0}
    assert allowed(e, [0]) == {2}


def test_bitmask_byte_filter():
    pytest.importorskip('torch')
    e = enforcer(JsonSchemaParser({'type': 'string'}), use_bitmask=True)
    e.get_allowed_tokens([999])
    first = e.get_allowed_tokens([999, 34])
    assert first.is_token_allowed(0xc5)
    pending = e.get_allowed_tokens([999, 34, 0xc5])
    assert pending.is_token_allowed(0x81)
    assert not pending.is_token_allowed(34)
    assert not pending.is_token_allowed(500)


def test_default_json_free_text_all_legal_ascii():
    e = enforcer(JsonSchemaParser({'type': 'string'}), [b'a|~\x7f'])
    allowed(e, [999])
    seq = [999, 34]
    a = allowed(e, seq)
    assert {ord('|'), ord('~'), 127, 256} <= a
    assert not set(range(32)) & a
    for char in b'|~\x7f':
        assert char in allowed(e, seq)
        seq = seq + [char]
    assert 34 in allowed(e, seq)


def test_default_regex_wildcard_all_legal_ascii():
    for byte in b'|~\x7f':
        e = enforcer(JsonSchemaParser({'type': 'string', 'pattern': '.'}))
        allowed(e, [999])
        assert byte in allowed(e, [999, 34])


def test_array_numeric_and_nested_beam_state_isolation():
    integer = {'type': 'integer'}
    inner = {'type': 'array', 'items': integer, 'minItems': 1, 'maxItems': 1}
    schema = {'type': 'array', 'items': inner, 'minItems': 1, 'maxItems': 1}
    e = enforcer(JsonSchemaParser(schema))
    assert 91 in allowed(e, [999])
    assert 91 in allowed(e, [999, 91])
    prefix = [999, 91, 91]
    a = allowed(e, prefix)
    assert {ord('0'), ord('1')} <= a
    zero = prefix + [ord('0')]
    one = prefix + [ord('1')]
    assert ord(']') in allowed(e, zero)
    assert ord(']') in allowed(e, one)
    assert ord(']') in allowed(e, zero + [ord(']')])
    assert ord(']') in allowed(e, one + [ord(']')])


def test_single_item_integer_array_after_zero():
    schema = {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 1, 'maxItems': 1}
    e = enforcer(JsonSchemaParser(schema))
    allowed(e, [999])
    allowed(e, [999, 91])
    a = allowed(e, [999, 91, 48])
    assert 93 in a
    assert 500 not in a


def test_json_cache_preclassifies_partial_and_impossible_utf8(monkeypatch):
    # Cache both scalar-boundary tokens and valid unfinished scalars once, not
    # through a per-generation-step walk of all non-ASCII byte branches.
    partials = [b'\xc5', b'\xe0\xa0', b'\xed\x9f', b'\xf0\x90', b'\xf4\x8f\xbf',
                b'A\xc5', 'Ł'.encode() + b'\xf0\x9f']
    invalid = [b'\x80', b'\xc0', b'\xc1', b'\xf5', b'\xff', b'\xe0\x9f',
               b'\xed\xa0', b'\xf4\x90', b'A\x80', b'A\x00']
    e = enforcer(JsonSchemaParser({'type': 'string'}), partials + invalid)
    cache = e.byte_tree.json_freetext_tokens
    cached = set(cache.lookup_allowed_tokens(0, 2).allowed_tokens)
    assert set(range(256, 256 + len(partials))) <= cached
    assert not set(range(256 + len(partials), 256 + len(partials + invalid))) & cached
    assert not set(range(32)) & cached
    assert not set(range(32)) & set(e.byte_tree.json_exceptions.children)

    import lmformatenforcer.tokenenforcer as module
    original = module.advance_utf8
    visited = []

    def record(prefix, byte):
        visited.append(byte)
        return original(prefix, byte)

    allowed(e, [999])
    monkeypatch.setattr(module, 'advance_utf8', record)
    assert 0xc5 in allowed(e, [999, 34])
    assert not any(byte >= 128 for byte in visited)


@pytest.mark.parametrize('max_length', [0, 1, 2])
def test_json_cached_partial_scalar_length_budget(max_length):
    extra = [b'\xc5', b'A\xc5', 'Ł'.encode() + b'\xf0\x9f', b'AB\xc5']
    e = enforcer(JsonSchemaParser({'type': 'string', 'maxLength': max_length}), extra)
    allowed(e, [999])
    seq = [999, 34]
    a = allowed(e, seq)
    assert (256 in a) == (max_length >= 1)
    assert (257 in a) == (max_length >= 2)
    assert (258 in a) == (max_length >= 2)
    assert 259 not in a
    if max_length == 2:
        assert 0x80 in allowed(e, seq + [258])
        state = e.prefix_states[tuple(seq + [258])]
        assert state.pending_utf8 == b'\xf0\x9f'
        assert state.parser.object_stack[-1].parsed_string == 'Ł'
        assert not ({34, 500, ord('A')} & allowed(e, seq + [258]))
        assert 0x80 in allowed(e, seq + [258, 0x80])
        assert 34 in allowed(e, seq + [258, 0x80, 0x80])
        assert 0xc5 not in allowed(e, seq + [258, 0x80, 0x80])


def test_json_cached_partial_min_length_quotes_and_eos():
    extras = [b'A\xc5', b'A"', b'"', 'Ł"'.encode(), b'AB"', b'\\n', b'A\x1f']
    e = enforcer(JsonSchemaParser({'type': 'string', 'minLength': 2, 'maxLength': 2}), extras)
    allowed(e, [999])
    seq = [999, 34]
    a = allowed(e, seq)
    assert {256, 260, 261} <= a
    assert not ({257, 258, 259, 262, 500} & a)
    assert not ({34, 500} & allowed(e, seq + [256]))
    assert 0x81 in allowed(e, seq + [256])
    assert 34 in allowed(e, seq + [256, 0x81])
    assert 500 in allowed(e, seq + [256, 0x81, 34])


def test_json_cache_keeps_quote_escape_and_postquote_structure_dynamic():
    schema = {'type': 'array', 'items': {'type': 'string', 'minLength': 1},
              'minItems': 1, 'maxItems': 1}
    extras = [b'\\n', b'A"]', b'"A', b'\xc5\x81"]', b'"\xc5']
    e = enforcer(JsonSchemaParser(schema), extras)
    seq = [999]
    allowed(e, seq)
    seq += [91]
    assert 260 in allowed(e, seq)
    seq += [34]
    a = allowed(e, seq)
    assert {256, 257, 259} <= a
    assert not ({258, 260, 500} & a)
    assert 500 in allowed(e, seq + [259])


def test_json_leaf_backslash_lookup_preserves_actual_escape_state():
    e = enforcer(JsonSchemaParser({'type': 'string', 'minLength': 1, 'maxLength': 3}))
    allowed(e, [999])
    seq = [999, 34]
    assert 92 in allowed(e, seq)
    seq += [92]
    a = allowed(e, seq)
    assert {ord('n'), ord('u'), 34, 92} <= a
    assert not ({ord('x'), 0xc5, 500} & a)
    # A quote here is escaped content, not the closing delimiter.
    assert 500 not in allowed(e, seq + [34])
    assert 34 in allowed(e, seq + [34])
    assert 500 in allowed(e, seq + [34, 34])


@pytest.mark.parametrize('first_alphabet,second_alphabet', [
    ('a', 'b'), ('a', None), (None, 'b'),
])
def test_json_pattern_interleaved_roots_keep_alphabets(first_alphabet, second_alphabet):
    schema = {'type': 'string', 'pattern': '.{0,5}'}

    def make(alphabet):
        config = (CharacterLevelParserConfig(alphabet=alphabet) if alphabet is not None
                  else CharacterLevelParserConfig())
        return enforcer(JsonSchemaParser(schema, config))

    first = make(first_alphabet)
    second = make(second_alphabet)
    for e, byte in [(first, ord('a')), (second, ord('b'))]:
        allowed(e, [999])
        assert byte in allowed(e, [999, 34])
        allowed(e, [999, 34, byte])

    # Continue at fresh prefixes: an allowed-token cache hit would hide a
    # shared RegexParser context being mutated by the other root's config.
    for e, alphabet, byte, other in [
        (first, first_alphabet, ord('a'), ord('b')),
        (second, second_alphabet, ord('b'), ord('a')),
    ]:
        a = allowed(e, [999, 34, byte, byte])
        assert byte in a and 34 in a
        assert (other in a) == (alphabet is None)
        assert (0xc5 in a) == (alphabet is None)


def test_json_pattern_context_cache_is_root_local_but_shared_by_clones():
    schema = {'type': 'string', 'pattern': '.{0,5}'}
    first = JsonSchemaParser(schema)
    second = JsonSchemaParser(schema)
    clone = first.add_character('"')
    assert clone.context is first.context
    assert clone.context.regex_parser_cache is first.context.regex_parser_cache
    assert first.context is not second.context
    assert first.context.regex_parser_cache is not second.context.regex_parser_cache
    assert (first.context.regex_parser_cache['.{0,5}'] is not
            second.context.regex_parser_cache['.{0,5}'])


def test_legacy_regex_keeps_direct_walk_alphabet_cache():
    parser = RegexParser('...', CharacterLevelParserConfig(alphabet='abc'))
    direct = parser
    for character in 'ab':
        assert character in direct.get_allowed_characters()
        direct = direct.add_character(character)
    assert 'Σ' not in direct.get_allowed_characters()

    tokens = [(0, 'a', False), (1, 'b', False), (2, 'Σ', False)]
    data = TokenEnforcerTokenizerData(
        tokens, lambda ids: ''.join('abΣ'[i] for i in ids), 3, False, 4)
    e = TokenEnforcer(data, parser)
    assert data.raw_token_bytes is None and e.byte_tree is None
    assert not parser.config.byte_mode and not parser.config.allow_unicode
    # Match the existing direct-then-enforcer test helper. Legacy alphabet
    # replacement must not invalidate the states primed by the direct walk.
    for sequence in ([999], [999, 0], [999, 0, 1]):
        a = allowed(e, sequence)
        assert {0, 1} <= a
        assert 2 not in a


@pytest.mark.parametrize('alphabet', [None, 'abc'])
def test_byte_mode_invalidates_primed_regex_character_cache(alphabet):
    config = (CharacterLevelParserConfig(alphabet=alphabet) if alphabet is not None
              else CharacterLevelParserConfig())
    parser = RegexParser('...', config)
    direct = parser
    for character in 'ab':
        assert character in direct.get_allowed_characters()
        direct = direct.add_character(character)
    direct.get_allowed_characters()
    assert parser.context.state_character_cache

    e = enforcer(parser)
    assert parser.config.byte_mode
    assert parser.context.state_character_cache == {}
    a = allowed(e, [999])
    assert ord('a') in a
    assert (0xc5 in a) == (alphabet is None)
