import json
from typing import List

import pytest

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
from lmformatenforcer.consts import COMPLETE_ALPHABET
from lmformatenforcer.tokenenforcer import TokenEnforcer, TokenEnforcerTokenizerData
from lmformatenforcer.tokenlist import _HAS_TORCH


STRINGS = [
    '""',
    '"a"',
    '" a"',
    '"  "',
    r'"\n"',
    r'"\u0061"',
    r'"a\n"',
    r'"\u0061b"',
    r'"\ud83d\ude00"',
    r'"\uD83D\uDE00a"',
    r'"\ud83d\ude00\ud83d\ude00"',
    r'"\ud800a\udc00"',
    r'"\ud800\ud800\udc00"',
    '"é"',
    '"e\u0301"',
    '"😀"',
]


def _parser(schema: dict) -> JsonSchemaParser:
    return JsonSchemaParser(
        schema,
        CharacterLevelParserConfig(alphabet=COMPLETE_ALPHABET + "é\u0301😀"),
    )


def _accepts(text: str, schema: dict) -> bool:
    parser = _parser(schema)
    for character in text:
        if character not in parser.get_allowed_characters():
            return False
        parser = parser.add_character(character)
    return parser.can_end()


@pytest.mark.parametrize("text", STRINGS)
def test_direct_length_uses_decoded_unicode_code_points(text: str) -> None:
    length = len(json.loads(text))
    for minimum in range(4):
        for maximum in range(minimum, 4):
            schema = {"type": "string", "minLength": minimum, "maxLength": maximum}
            assert _accepts(text, schema) == (minimum <= length <= maximum)


@pytest.mark.parametrize("text", STRINGS)
@pytest.mark.parametrize(
    "use_bitmask",
    [
        False,
        pytest.param(
            True, marks=pytest.mark.skipif(not _HAS_TORCH, reason="requires torch")
        ),
    ],
)
def test_token_enforcer_length_is_independent_of_token_boundaries(
    text: str, use_bitmask: bool
) -> None:
    pieces = sorted(
        {
            text[start:end]
            for start in range(len(text))
            for end in range(start + 1, len(text) + 1)
        }
        | set(COMPLETE_ALPHABET)
    )
    token_ids = {piece: index for index, piece in enumerate(pieces)}

    def decode(ids: List[int]) -> str:
        return "".join(pieces[index] for index in ids)

    data = TokenEnforcerTokenizerData(
        [(index, piece, False) for index, piece in enumerate(pieces)],
        decode,
        len(pieces),
        use_bitmask,
        len(pieces) + 1,
    )
    splits = [[text[:index], text[index:]] for index in range(1, len(text))]
    splits.extend([[text], list(text)])
    length = len(json.loads(text))
    for minimum in range(4):
        for maximum in range(minimum, 4):
            schema = {"type": "string", "minLength": minimum, "maxLength": maximum}
            for split in splits:
                enforcer = TokenEnforcer(data, _parser(schema))
                prefix: List[int] = []
                accepted = True
                for piece in split:
                    allowed = enforcer.get_allowed_tokens(prefix)
                    token_id = token_ids[piece]
                    if not allowed.is_token_allowed(token_id):
                        accepted = False
                        break
                    prefix.append(token_id)
                if accepted:
                    accepted = bool(
                        enforcer.get_allowed_tokens(prefix).is_token_allowed(
                            len(pieces)
                        )
                    )
                assert accepted == (minimum <= length <= maximum), (schema, split)


def test_partial_escape_and_surrogate_continuations_are_immutable() -> None:
    parser = _parser({"type": "string", "maxLength": 1})
    for character in r'"\ud83d':
        parser = parser.add_character(character)
    assert parser.get_allowed_characters() == '"\\'
    escape = parser.add_character("\\")
    assert escape.get_allowed_characters() == "u"
    assert parser.get_allowed_characters() == '"\\'
    for character in "uDE00":
        escape = escape.add_character(character)
    assert escape.get_allowed_characters() == '"'
    assert parser.add_character('"').can_end()
    assert escape.add_character('"').can_end()


@pytest.mark.parametrize("bad_escape", [r"\q", r"\u001", r"\u00xz"])
def test_malformed_escapes_cannot_finish(bad_escape: str) -> None:
    assert not _accepts('"' + bad_escape + '"', {"type": "string"})


def test_nested_strings_keep_independent_length_state() -> None:
    schema = {
        "type": "object",
        "properties": {
            "one": {"type": "string", "minLength": 1, "maxLength": 1},
            "two": {"type": "string", "minLength": 2, "maxLength": 2},
        },
        "required": ["one", "two"],
    }
    assert _accepts(r'{"one":"\ud83d\ude00","two":"\n\u0061"}', schema)
    assert not _accepts(r'{"one":"\ud83d\ude00","two":"\ud83d\ude00"}', schema)


def test_existing_escaped_keys_enum_and_pattern_behavior_is_preserved() -> None:
    schema = {
        "type": "object",
        "properties": {"a\\b": {"type": "string", "enum": ["a\\b"]}},
        "required": ["a\\b"],
    }
    assert _accepts(json.dumps({"a\\b": "a\\b"}), schema)
    assert not _accepts(json.dumps({"a\\b": "different"}), schema)
    assert _accepts('"aa"', {"type": "string", "pattern": "^a+$"})
    assert not _accepts('"ab"', {"type": "string", "pattern": "^a+$"})


def test_configured_non_ascii_alphabet_is_preserved() -> None:
    parser = JsonSchemaParser(
        {"type": "string", "minLength": 1, "maxLength": 1},
        CharacterLevelParserConfig(alphabet=COMPLETE_ALPHABET + "界"),
    )
    for character in '"界"':
        assert character in parser.get_allowed_characters()
        parser = parser.add_character(character)
    assert parser.can_end()


def test_vocabulary_with_only_complex_json_tokens() -> None:
    text = r'"a\n"'
    data = TokenEnforcerTokenizerData(
        [(0, text, False)], lambda ids: text * len(ids), 1, False, 2
    )
    enforcer = TokenEnforcer(
        data, _parser({"type": "string", "minLength": 2, "maxLength": 2})
    )
    assert enforcer.get_allowed_tokens([]).is_token_allowed(0)
    assert enforcer.get_allowed_tokens([0]).is_token_allowed(1)


@pytest.mark.parametrize("control", [chr(index) for index in range(0x20)])
def test_complex_token_traversal_does_not_allow_raw_json_controls(control: str) -> None:
    pieces = ['"', "a", control, "a" + control + '"', json.dumps(control)[1:]]
    data = TokenEnforcerTokenizerData(
        [(index, piece, False) for index, piece in enumerate(pieces)],
        lambda ids: "".join(pieces[index] for index in ids),
        len(pieces),
        False,
        len(pieces) + 1,
    )
    enforcer = TokenEnforcer(data, _parser({"type": "string", "maxLength": 3}))
    enforcer.get_allowed_tokens([])
    allowed = enforcer.get_allowed_tokens([0])
    assert not allowed.is_token_allowed(2)
    assert not allowed.is_token_allowed(3)
    assert allowed.is_token_allowed(4)
