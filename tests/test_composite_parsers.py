from lmformatenforcer import UnionParser, SequenceParser, StringParser
from .common import assert_parser_with_string
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser


def test_string_choice():
    parser = UnionParser([StringParser('aa'), StringParser('bb')])
    assert_parser_with_string('aa', parser, True)
    assert_parser_with_string('bb', parser, True)
    assert_parser_with_string('ab', parser, False)
    assert_parser_with_string('aabb', parser, False)


def test_string_sequence():
    parser  = SequenceParser([StringParser('aa'), StringParser('bb')])
    assert_parser_with_string('aa', parser, False)
    assert_parser_with_string('bb', parser, False)
    assert_parser_with_string('ab', parser, False)
    assert_parser_with_string('aabb', parser, True)
    assert_parser_with_string('bbaa', parser, False)


def test_json_markdown_sequence():
    class TestModel(BaseModel):
        a: str
    json_parser = JsonSchemaParser(TestModel.schema())
    parser = SequenceParser([StringParser("```json\n"), json_parser, StringParser('\n```')])
    assert_parser_with_string('```json\n{"a": "b"}\n```', parser, True)
    assert_parser_with_string('{"a": "b"}', parser, False)


def test_string_sequence_vocabulary():
    parser = SequenceParser([StringParser('aa'), StringParser('bb')])

    if "a" not in parser.get_allowed_characters():
        raise ValueError(f"Expect parser vocabulary '{parser.get_allowed_characters()}' to contain 'a'")

    if "b" in parser.get_allowed_characters():
        raise ValueError(f"Expect parser vocabulary '{parser.get_allowed_characters()}' to not contain 'b'")
