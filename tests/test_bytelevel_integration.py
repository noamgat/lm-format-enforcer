"""Public Transformers builder coverage for native ByteLevel special tokens."""
import pytest

tokenizers = pytest.importorskip('tokenizers')
transformers = pytest.importorskip('transformers')
pytest.importorskip('torch')

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    _build_regular_tokens_list,
    build_token_enforcer_tokenizer_data,
)
from lmformatenforcer.tokenenforcer import TokenEnforcer


def make_tokenizer(*, regular_added=False, decoder_sequence=False, incomplete=False):
    alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
    if incomplete:
        alphabet.pop()
    vocab = {char: i for i, char in enumerate(alphabet)}
    vocab['<eos>'] = len(vocab)
    backend = tokenizers.Tokenizer(tokenizers.models.BPE(vocab, merges=[]))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
    byte_decoder = tokenizers.decoders.ByteLevel()
    backend.decoder = (tokenizers.decoders.Sequence([byte_decoder])
                       if decoder_sequence else byte_decoder)
    # Like Llama, these backend special tokens have no named special-token map.
    backend.add_special_tokens([tokenizers.AddedToken('<backend_special>', special=True)])
    if regular_added:
        backend.add_tokens([tokenizers.AddedToken('<regular_added>', special=False)])
    return transformers.PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<eos>')


class TokenizerProxy:
    def __init__(self, tokenizer, **overrides):
        self.tokenizer = tokenizer
        self.__dict__.update(overrides)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __len__(self):
        return len(self.tokenizer)


class MissingAddedMetadata:
    def __init__(self, backend):
        self.backend = backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def get_added_tokens_decoder(self):
        return {}


def test_backend_special_tokens_enable_bytes_and_are_excluded():
    tokenizer = make_tokenizer()
    special_id = tokenizer.convert_tokens_to_ids('<backend_special>')
    assert special_id not in tokenizer.all_special_ids
    assert tokenizer.backend_tokenizer.get_added_tokens_decoder()[special_id].special
    legacy_tokens = _build_regular_tokens_list(tokenizer, len(tokenizer))
    assert special_id in {i for i, _, _ in legacy_tokens}

    data = build_token_enforcer_tokenizer_data(tokenizer)
    assert data.raw_token_bytes is not None
    regular_ids = {i for i, _, _ in data.regular_tokens}
    assert set(data.raw_token_bytes) == regular_ids
    assert special_id not in regular_ids
    assert tokenizer.eos_token_id not in regular_ids
    assert {bytes([i]) for i in range(256)} <= set(data.raw_token_bytes.values())


@pytest.mark.parametrize('use_bitmask', [False, True])
@pytest.mark.parametrize('text', ['Ł', '中', '😀'])
@pytest.mark.parametrize('kind', ['enum', 'free', 'regex'])
def test_backend_special_tokens_public_builder_unicode(kind, text, use_bitmask):
    tokenizer = make_tokenizer()
    data = build_token_enforcer_tokenizer_data(tokenizer, use_bitmask=use_bitmask)
    assert data.raw_token_bytes is not None
    schema = {'type': 'string'}
    if kind == 'enum':
        schema['enum'] = [text]
    elif kind == 'free':
        schema.update(minLength=1, maxLength=1)
    else:
        schema['pattern'] = text
    enforcer = TokenEnforcer(data, JsonSchemaParser(schema))
    encoded = tokenizer.encode('"' + text + '"', add_special_tokens=False)
    sequence = [tokenizer.eos_token_id]
    for token in encoded:
        allowed = enforcer.get_allowed_tokens(sequence)
        assert allowed.is_token_allowed(token)
        assert not allowed.is_token_allowed(tokenizer.convert_tokens_to_ids('<backend_special>'))
        sequence.append(token)
    assert enforcer.get_allowed_tokens(sequence).is_token_allowed(tokenizer.eos_token_id)
    assert tokenizer.decode(encoded) == '"' + text + '"'


@pytest.mark.parametrize('case', ['regular_added', 'decoder_sequence', 'nonfast',
                                 'incomplete', 'missing_metadata'])
def test_unsupported_bytelevel_keeps_legacy_regular_tokens(case):
    tokenizer = make_tokenizer(**{case: True} if case in {
        'regular_added', 'decoder_sequence', 'incomplete'} else {})
    if case == 'nonfast':
        tokenizer = TokenizerProxy(tokenizer, is_fast=False)
    elif case == 'missing_metadata':
        tokenizer = TokenizerProxy(
            tokenizer, backend_tokenizer=MissingAddedMetadata(tokenizer.backend_tokenizer))
    legacy_tokens = _build_regular_tokens_list(tokenizer, len(tokenizer))
    data = build_token_enforcer_tokenizer_data(tokenizer)
    assert data.raw_token_bytes is None
    assert data.regular_tokens == legacy_tokens
    assert tokenizer.convert_tokens_to_ids('<backend_special>') in {
        i for i, _, _ in data.regular_tokens}
