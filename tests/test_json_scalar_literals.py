"""Scalar schema literals must use JSON spelling rather than Python spelling."""

import json

import pytest

from lmformatenforcer import JsonSchemaParser


@pytest.mark.parametrize("value", [None, True, False, 42, -7, 1.25, "text"])
def test_scalar_const_accepts_its_json_value(value):
    parser = JsonSchemaParser({"const": value})
    for character in json.dumps(value):
        assert character in parser.get_allowed_characters()
        parser = parser.add_character(character)
    assert parser.can_end()


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("with_type", [False, True])
def test_boolean_enum_accepts_json_literal(value, with_type):
    schema = {"enum": [value]}
    if with_type:
        schema["type"] = "boolean"
    parser = JsonSchemaParser(schema)
    for character in json.dumps(value):
        assert character in parser.get_allowed_characters()
        parser = parser.add_character(character)
    assert parser.can_end()
    assert str(value)[0] not in JsonSchemaParser(schema).get_allowed_characters()
    assert (
        json.dumps(not value)[0]
        not in JsonSchemaParser(schema).get_allowed_characters()
    )


@pytest.mark.parametrize("value", [None, False, 12, -1.5])
def test_scalar_const_inside_object(value):
    schema = {
        "type": "object",
        "properties": {"value": {"const": value}},
        "required": ["value"],
    }
    parser = JsonSchemaParser(schema)
    for character in json.dumps({"value": value}):
        assert character in parser.get_allowed_characters()
        parser = parser.add_character(character)
    assert parser.can_end()


@pytest.mark.parametrize(
    "expected, other", [(42, 43), (True, False), (None, 0), (1.25, 1.2)]
)
def test_scalar_const_rejects_a_different_value(expected, other):
    parser = JsonSchemaParser({"const": expected})
    for character in json.dumps(other):
        if character not in parser.get_allowed_characters():
            return
        parser = parser.add_character(character)
    assert not parser.can_end()
