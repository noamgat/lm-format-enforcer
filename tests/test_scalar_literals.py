"""Scalar schema constraints, exercised without tokenizer/model downloads."""
import json

import pytest

from lmformatenforcer import JsonSchemaParser


def accepts(schema, text):
    parser = JsonSchemaParser(schema)
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


@pytest.mark.parametrize(
    "value,bad",
    [
        (42, 43),
        (-3, -2),
        (1.5, 1.6),
        (True, False),
        (False, True),
        (None, 0),
        ("hello", "different"),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_scalar_const(value, bad, nested):
    schema = {"const": value}
    good = value
    if nested:
        schema = {
            "type": "object",
            "properties": {"value": schema},
            "required": ["value"],
        }
        good = {"value": good}
        bad = {"value": bad}
    assert accepts(schema, json.dumps(good))
    assert not accepts(schema, json.dumps(bad))


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("typed", [False, True])
def test_boolean_enum(value, typed):
    schema = {"enum": [value]}
    if typed:
        schema["type"] = "boolean"
    assert accepts(schema, json.dumps(value))
    assert not accepts(schema, json.dumps(not value))
    assert not accepts(schema, str(value))


@pytest.mark.parametrize("value", [42, -3, 1.5, "hello"])
def test_existing_enum(value):
    assert accepts({"enum": [value]}, json.dumps(value))
    assert not accepts({"enum": [value]}, json.dumps("different"))


def test_mixed_boolean_numeric_enum():
    schema = {"enum": [True, 2]}
    for text in ["true", "2"]:
        assert accepts(schema, text)
    for text in ["false", "1", "3", "True"]:
        assert not accepts(schema, text)


def test_scalar_const_delimiters():
    schema = {
        "type": "object",
        "properties": {"value": {"const": True}, "next": {"const": None}},
        "required": ["value", "next"],
    }
    assert accepts(schema, ' { "value" : true , "next" : null } ')
    assert not accepts(schema, '{"value":false,"next":null}')
    assert not accepts(schema, '{"value":true,"next":0}')
