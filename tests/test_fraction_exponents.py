"""Number grammar checks using only synthetic text and token vocabularies."""
import json

import pytest

from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData

SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "number"}},
    "required": ["value"],
}


def accepts(text):
    parser = JsonSchemaParser(SCHEMA)
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


@pytest.mark.parametrize("number", ["1.e2", "0.e2", "-1.e2", "-0.E-2", "1.E+2"])
def test_missing_fraction_digit(number):
    document = '{"value":' + number + "}"
    with pytest.raises(json.JSONDecodeError):
        json.loads(document)
    assert not accepts(document)


@pytest.mark.parametrize("number", ["1e2", "1.0e2", "-0.5E-2", "0E+2", "-1.2", "0.0"])
def test_valid_fraction_and_exponent(number):
    document = '{"value":' + number + "}"
    json.loads(document)
    assert accepts(document)


def test_token_filter_requires_fraction_digit():
    words = ['{"value":1.', "e2}", "0e2}", '{"value":1.e2}', '{"value":1.0e2}']
    data = TokenEnforcerTokenizerData(
        [(i, word, False) for i, word in enumerate(words)],
        lambda ids: "".join(words[i] for i in ids),
        eos_token_id=5,
        use_bitmask=False,
        vocab_size=6,
    )
    enforcer = TokenEnforcer(data, JsonSchemaParser(SCHEMA))
    initial = enforcer.get_allowed_tokens([])
    assert initial.is_token_allowed(0)
    assert initial.is_token_allowed(4)
    assert not initial.is_token_allowed(3)
    after_point = enforcer.get_allowed_tokens([0])
    assert after_point.is_token_allowed(2)
    assert not after_point.is_token_allowed(1)
    assert not after_point.is_token_allowed(5)
    complete = enforcer.get_allowed_tokens([0, 2])
    assert complete.is_token_allowed(5)
