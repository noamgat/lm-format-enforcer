"""Check number syntax without downloading a tokenizer or loading a model."""

import json

import pytest

from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData

SCHEMA = {
    "type": "object",
    "properties": {"value": {"type": "number"}},
    "required": ["value"],
}


@pytest.mark.parametrize("mantissa", ["0.", "1.", "-0.", "-1."])
@pytest.mark.parametrize("exponent", ["e2", "E-2"])
def test_exponent_requires_a_fraction_digit(mantissa, exponent):
    document = '{"value":' + mantissa + exponent + "}"
    with pytest.raises(json.JSONDecodeError):
        json.loads(document)
    parser = JsonSchemaParser(SCHEMA)
    for character in document:
        if character not in parser.get_allowed_characters():
            return
        parser = parser.add_character(character)
    assert not parser.can_end()


@pytest.mark.parametrize("number", ["1e2", "-0E-2", "0.1e2", "-1.25E+2", "0.0", "-0.5"])
def test_valid_number_syntax_remains_accepted(number):
    document = '{"value":' + number + "}"
    json.loads(document)
    parser = JsonSchemaParser(SCHEMA)
    for character in document:
        assert character in parser.get_allowed_characters()
        parser = parser.add_character(character)
    assert parser.can_end()


def test_token_enforcer_rejects_exponent_immediately_after_decimal_point():
    # Use multi-character tokens so filtering traverses the production prefix tree.
    vocabulary = ['{"value":1.', "e2}", "0e2}", "2}", "<eos>"]
    data = TokenEnforcerTokenizerData(
        [(index, token, False) for index, token in enumerate(vocabulary[:-1])],
        lambda ids: "".join(vocabulary[index] for index in ids),
        4,
        False,
        len(vocabulary),
    )
    enforcer = TokenEnforcer(data, JsonSchemaParser(SCHEMA))
    assert 0 in enforcer.get_allowed_tokens([]).allowed_tokens
    allowed = enforcer.get_allowed_tokens([0]).allowed_tokens
    assert 1 not in allowed
    assert 2 in allowed
    assert 3 in allowed
