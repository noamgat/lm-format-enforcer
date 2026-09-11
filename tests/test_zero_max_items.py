"""An array schema with maxItems=0 must never offer its first item."""

import json

import pytest

from lmformatenforcer import JsonSchemaParser, TokenEnforcer, TokenEnforcerTokenizerData


@pytest.mark.parametrize("item", [1, "text", {}, [], None, True])
def test_zero_max_items_rejects_nonempty_array(item):
    parser = JsonSchemaParser({"type": "array", "maxItems": 0})
    for character in json.dumps([item]):
        if character not in parser.get_allowed_characters():
            return
        parser = parser.add_character(character)
    assert not parser.can_end()


@pytest.mark.parametrize("document", ["[]", "[ ]", "[\n\t]"])
def test_zero_max_items_accepts_empty_array(document):
    parser = JsonSchemaParser({"type": "array", "maxItems": 0})
    for character in document:
        assert character in parser.get_allowed_characters()
        parser = parser.add_character(character)
    assert parser.can_end()


def test_zero_max_items_filters_item_tokens():
    vocabulary = ["[", "]", "1]", '"x"]', "<eos>"]
    data = TokenEnforcerTokenizerData(
        [(index, token, False) for index, token in enumerate(vocabulary[:-1])],
        lambda ids: "".join(vocabulary[index] for index in ids),
        4,
        False,
        len(vocabulary),
    )
    enforcer = TokenEnforcer(data, JsonSchemaParser({"type": "array", "maxItems": 0}))
    assert 0 in enforcer.get_allowed_tokens([]).allowed_tokens
    allowed = enforcer.get_allowed_tokens([0]).allowed_tokens
    assert 1 in allowed
    assert 2 not in allowed
    assert 3 not in allowed


@pytest.mark.parametrize("limit", [1, 2])
def test_positive_max_items_still_accepts_items(limit):
    parser = JsonSchemaParser(
        {"type": "array", "items": {"type": "integer"}, "maxItems": limit}
    )
    for character in json.dumps(list(range(limit))):
        assert character in parser.get_allowed_characters()
        parser = parser.add_character(character)
    assert parser.can_end()


def test_empty_array_property_finishes_its_parent_object():
    parser = JsonSchemaParser(
        {
            "type": "object",
            "properties": {"values": {"type": "array", "maxItems": 0}},
            "required": ["values"],
        }
    )
    for character in '{"values": []}':
        assert character in parser.get_allowed_characters()
        parser = parser.add_character(character)
    assert parser.can_end()
