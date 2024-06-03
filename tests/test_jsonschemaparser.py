import json
import os
from contextlib import contextmanager
from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field
from lmformatenforcer import JsonSchemaParser
from enum import Enum
import pytest
from lmformatenforcer.consts import BACKSLASH, BACKSLASH_ESCAPING_CHARACTERS, CONFIG_ENV_VAR_STRICT_JSON_FIELD_ORDER, CONFIG_ENV_VAR_MAX_CONSECUTIVE_WHITESPACES

from .common import assert_parser_with_string, CharacterNotAllowedException


def _test_json_schema_parsing_with_string(string: str, schema_dict: Optional[dict], expect_success: bool, profile_file_path: Optional[str] = None):
    parser = JsonSchemaParser(schema_dict)
    assert_parser_with_string(string, parser, expect_success, profile_file_path)
    if expect_success:
        # If expecting success, also check minified and pretty-printed
        minified = json.dumps(json.loads(string), separators=(',', ':'))
        assert_parser_with_string(minified, parser, expect_success)
        pretty_printed = json.dumps(json.loads(string), indent=2)
        assert_parser_with_string(pretty_printed, parser, expect_success)


class InnerModel(BaseModel):
    list_of_ints: List[int]


class IntegerEnum(Enum):
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4


class StringEnum(Enum):
    ONE = "One"
    TWO = "Two"
    THREE = "Three"
    FOUR = "Four"


class SampleModel(BaseModel):
    num: int
    dec: Optional[float] = None
    message: Optional[str] = None
    list_of_strings: Optional[List[str]] = Field(None, min_length=2, max_length=3)
    inner_dict: Optional[Dict[str, InnerModel]] = None
    simple_dict: Optional[Dict[str, int]] = None
    list_of_models: Optional[List[InnerModel]] = None
    enum: Optional[IntegerEnum] = None
    enum_dict: Optional[Dict[str, StringEnum]] = None
    true_or_false: Optional[bool] = None


def test_minimal():
    test_string = '{"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)

def test_parsing_test_model():
    test_string = '{"num":1,"dec":1.1,"message":"ok","list_of_strings":["a","b","c"],"inner_dict":{"a":{"list_of_ints":[1,2,3]}}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)


def test_invalid_key_in_json_string():
    test_string = '{"numa":1,"dec":1.1,"message":"ok","list_of_strings":["a","b","c"],"inner_dict":{"a":{"list_of_ints":[1,2,3]}}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), False)


def test_incomplete_json():
    # Intentionally missing closing }
    test_string = '{"num":1,"dec":1.1,"message":"ok","list_of_strings":["a","b","c"],"inner_dict":{"a":{"list_of_ints":[1,2,3]}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), False)


def test_invalid_value_type_in_json_string():
    test_string = '{"num":"1","dec":1.1,"message":"ok","list_of_strings":["a","b","c"],"inner_dict":{"a":{"list_of_ints":[1,2,3]}}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), False)


def test_list_of_objects():
    test_string = '{"list_of_models":[{"list_of_ints":[1,2,3]},{"list_of_ints":[4,5,6]}],"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)
    test_string = '{"list_of_models": [{"list_of_ints":[1, 2, 3]} , {"list_of_ints":[4,5,6]}],"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)


def test_simple_dict():
    test_string = '{"simple_dict":{"a":1,"b":2,"c":3},"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)


def test_int_enum():
    test_string = '{"enum":4,"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)


def test_invalid_int_enum_value():
    test_string = '{"enum":5,"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), False)


def test_str_enum():
    test_string = '{"enum_dict":{"a":"One","b":"Two","c":"Three","d":"Four"},"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)


def test_invalid_str_enum_value():
    test_string = '{"enum_dict":{"a":"Onee"},"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), False)


def test_whitespaces():
    test_string = '{ "message": "","num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)


def test_whitespace_before_number():
    test_string = '{"num": 1, "dec": 1.1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)


def test_whitespace_before_close():
    test_string = '{"num":1 }'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)    


def test_required_field():
    test_string = '{"dec": 1.1}'  # num is a required field, doesn't exist, should fail.
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), False)


def test_boolean_field():
    _test_json_schema_parsing_with_string('{"num":1,"true_or_false":false}', SampleModel.model_json_schema(), True)
    _test_json_schema_parsing_with_string('{"num":1,"true_or_false":true}', SampleModel.model_json_schema(), True)
    _test_json_schema_parsing_with_string('{"num":1,"true_or_false": true}', SampleModel.model_json_schema(), True)
    _test_json_schema_parsing_with_string('{"num":1,"true_or_false":falsy}', SampleModel.model_json_schema(), False)

def test_unspecified_dict():
    class DictModel(BaseModel):
        num: int
        d: dict

    _test_json_schema_parsing_with_string('{"num":1,"d":{"k":"v"}}', DictModel.model_json_schema(), True)


def test_unspecified_list():
    class DictModel(BaseModel):
        num: int
        l: list

    _test_json_schema_parsing_with_string('{"num":1,"l":[1,2,3,"b"]}', DictModel.model_json_schema(), True)


def test_list_length_limitations():
    # list_of_strings is defined as having a min length of 2 and a max length of 3
    no_strings = '{"num":1,"list_of_strings":[]}'
    _test_json_schema_parsing_with_string(no_strings, SampleModel.model_json_schema(), False)
    one_string = '{"num":1,"list_of_strings":["a"]}'
    _test_json_schema_parsing_with_string(one_string, SampleModel.model_json_schema(), False)
    two_strings = '{"num":1,"list_of_strings":["a", "b"]}'
    _test_json_schema_parsing_with_string(two_strings, SampleModel.model_json_schema(), True)
    three_strings = '{"num":1,"list_of_strings":["a","b","c"]}'
    _test_json_schema_parsing_with_string(three_strings, SampleModel.model_json_schema(), True)
    four_strings = '{"num":1,"list_of_strings":["a","b","c","d"]}'
    _test_json_schema_parsing_with_string(four_strings, SampleModel.model_json_schema(), False)

    class EmptyListOKModel(BaseModel):
        num: int
        list_of_strings: Optional[List[str]] = Field(None, min_length=0, max_length=1)
    _test_json_schema_parsing_with_string(no_strings, EmptyListOKModel.model_json_schema(), True)
    _test_json_schema_parsing_with_string(one_string, EmptyListOKModel.model_json_schema(), True)
    _test_json_schema_parsing_with_string(two_strings, EmptyListOKModel.model_json_schema(), False)

    class ListOfExactlyOneModel(BaseModel):
        num: int
        list_of_strings: Optional[List[str]] = Field(None, min_length=1, max_length=1)
    _test_json_schema_parsing_with_string(no_strings, ListOfExactlyOneModel.model_json_schema(), False)
    _test_json_schema_parsing_with_string(one_string, ListOfExactlyOneModel.model_json_schema(), True)
    _test_json_schema_parsing_with_string(two_strings, ListOfExactlyOneModel.model_json_schema(), False)

    class ListOfNoMinLengthModel(BaseModel):
        num: int
        list_of_strings: Optional[List[str]] = Field(None, max_length=1)
    _test_json_schema_parsing_with_string(no_strings, ListOfNoMinLengthModel.model_json_schema(), True)
    _test_json_schema_parsing_with_string(one_string, ListOfNoMinLengthModel.model_json_schema(), True)
    _test_json_schema_parsing_with_string(two_strings, ListOfNoMinLengthModel.model_json_schema(), False)


def test_string_escaping():
    for escaping_character in BACKSLASH_ESCAPING_CHARACTERS:
        test_string = f'{{"num":1,"message":"hello {BACKSLASH}{escaping_character} world"}}'
        _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)
    for non_escaping_character in 'a1?':
        test_string = f'{{"num":1,"message":"hello {BACKSLASH}{non_escaping_character} world"}}'
        _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), False)

    # Unicode
    test_string = f'{{"num":1,"message":"hello {BACKSLASH}uf9f0 world"}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), True)

    # Not enough unicode digits
    test_string = f'{{"num":1,"message":"hello {BACKSLASH}uf9f world"}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), False)

    # Unicode digit outside of hex range
    test_string = f'{{"num":1,"message":"hello {BACKSLASH}uf9fP world"}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.model_json_schema(), False)


def test_comma_after_all_object_keys_fails():
    class SomeSchema(BaseModel):
        key: str

    test_string = '{"key": "val",'
    with pytest.raises(CharacterNotAllowedException):
        _test_json_schema_parsing_with_string(test_string, SomeSchema.model_json_schema(), True)


def test_single_quote_must_not_be_escaped():
    class SomeSchema(BaseModel):
        key: str

    test_string = '{"key": "I\\\'m a string"}'
    with pytest.raises(CharacterNotAllowedException):
        _test_json_schema_parsing_with_string(test_string, SomeSchema.model_json_schema(), True)


def test_string_length_limitation():
    class SomeSchema(BaseModel):
        # This is the elegant way to do it, but requires python >=3.9, we want to support 3.8
        # key: Annotated[str, StringConstraints(min_length=2, max_length=3)]
        key: str = Field(..., min_length=2, max_length=3)

    for str_length in range(10):
        test_string = f'{{"key": "{str_length * "a"}"}}'
        expect_sucess = 2 <= str_length <= 3
        _test_json_schema_parsing_with_string(test_string, SomeSchema.model_json_schema(), expect_sucess)


def test_any_json_object():
    _test_json_schema_parsing_with_string("{}", None, True)
    _test_json_schema_parsing_with_string('{"a": 1, "b": 2.2, "c": "c", "d": [1,2,3, null], "e": {"ee": 2}}', None, True)
    _test_json_schema_parsing_with_string("true", None, True)
    _test_json_schema_parsing_with_string('"str"', None, True)
    
    
def test_allof():
    # Define a schema that includes allOf
    allof_schema = {
        "type": "object",
        "allOf": [
            {
                "type": "object",
                "properties": {
                    "num": {
                        "type": "number"
                    }
                },
                "required": ["num"]
            },
            {
                "type": "object",
                "properties": {
                    "str": {
                        "type": "string"
                    }
                },
                "required": ["str"]
            }
        ]
    }

    # Valid cases
    valid_test_strings = [
        '{"num": 123, "str": "test"}',
        '{"num": 0, "str": ""}'
    ]

    # Invalid cases
    invalid_test_strings = [
        '{"num": 123}',  # Missing 'str'
        '{"str": "test"}',  # Missing 'num'
        '{"num": "123", "str": "test"}',  # Invalid type for 'num'
        '{"num": 123, "str": 456}'  # Invalid type for 'str'
    ]

    for test_string in valid_test_strings:
        _test_json_schema_parsing_with_string(test_string, allof_schema, True)

    for test_string in invalid_test_strings:
        _test_json_schema_parsing_with_string(test_string, allof_schema, False)


def test_leading_comma():
    array_of_objects_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string"
                }
            },
            "required": ["key"]
        }
    }

    _test_json_schema_parsing_with_string('[{"key": "val"}, {"key": "val2"}]', array_of_objects_schema, True) 
    _test_json_schema_parsing_with_string('[,{"key": "val"}]', array_of_objects_schema, False)


def test_long_json_object():
    from urllib.request import urlopen
    import json
    json_url = 'https://microsoftedge.github.io/Demos/json-dummy-data/64KB.json'
    json_text = urlopen(json_url).read().decode('utf-8')
    # These are several "hacks" on top of the json file in order to bypass some shortcomings of the unit testing method.
    json_text = ''.join(c for c in json_text if 0 < ord(c) < 127)
    json_text = json_text.replace('.",', '",')
    json_text = json_text.replace(' ",', '",')
    json_text = json_text.replace('.",', '",')
    json_text = json.dumps(json.loads(json_text)[:20])

    profile_file_path = None  # '64KB.prof' 
    _test_json_schema_parsing_with_string(json_text, None, True, profile_file_path=profile_file_path)


def test_union():
    class SchemaWithUnion(BaseModel):
        key: Union[int, str]

    _test_json_schema_parsing_with_string('{"key": 1}', SchemaWithUnion.model_json_schema(), True)
    _test_json_schema_parsing_with_string('{"key": "a"}', SchemaWithUnion.model_json_schema(), True)
    _test_json_schema_parsing_with_string('{"key": 1.2}', SchemaWithUnion.model_json_schema(), False)
    _test_json_schema_parsing_with_string('{"key": false}', SchemaWithUnion.model_json_schema(), False)


class StringConstraints(BaseModel):
    min_5: Optional[str] = Field(None, min_length=5)
    max_8: Optional[str] = Field(None, max_length=8)
    max_16: Optional[str] = Field(None, max_length=16)
    min_8_max_8: Optional[str] = Field(None, min_length=8, max_length=8)
    min_4_max_6: Optional[str] = Field(None, min_length=4, max_length=6)


def test_more_string_constraints():
    for str_length in range(20):
        test_string = f'{{"min_4_max_6": "{str_length * "#"}"}}'
        expect_sucess = 4 <= str_length <= 6
        print(test_string, expect_sucess)
        _test_json_schema_parsing_with_string(test_string, StringConstraints.model_json_schema(), expect_sucess)

    for k,v in {
        'min_5': ['test5', 'test567'],
        'max_8': ['test5678', 'test56'],
        'max_16': ['123test??0123456', r'1\n\"'],
        'min_8_max_8': ['12t, t78', r'##\\n####'],
        'min_4_max_6': ['12_4', '12_4:5']
    }.items():
        for val in v:
            print(val)
            _test_json_schema_parsing_with_string(f'{{"{k}": "{val}"}}', StringConstraints.model_json_schema(), True)

    for k,v in {
        'min_5': 'test',
        'max_8': 'te\nst',
        'max_16': '123test89-1 34567',
        'min_8_max_8': '12test7"',
        'min_4_max_6': '12_'
    }.items():
        _test_json_schema_parsing_with_string(f'{{"{k}": "{v}"}}', StringConstraints.model_json_schema(), False)


def test_string_pattern_requirement():
    class SchemaWithPattern(BaseModel):
        str_field: str = Field(pattern=r"[ab]+")

    _test_json_schema_parsing_with_string('{"str_field": "ababab"}', SchemaWithPattern.model_json_schema(), True)
    _test_json_schema_parsing_with_string('{"str_field": "abc"}', SchemaWithPattern.model_json_schema(), False)


def test_phone_number_in_string():
    class ContactInfo(BaseModel):
        name: str
        # phone: str 
        phone: str = Field(pattern=r"\([0-9]{3}\)[0-9]{3}-[0-9]{4}")
    _test_json_schema_parsing_with_string('{"name": "John", "phone": "(312)011-2444"}', ContactInfo.model_json_schema(), True)
    _test_json_schema_parsing_with_string('{"name": "John", "phone": "312-011-2444"}', ContactInfo.model_json_schema(), False)


def test_union_typed_arrays():
    class AppleSchema(BaseModel):
        apple_type: int

    class BananaSchema(BaseModel):
        is_ripe: bool

    class FruitsSchema(BaseModel):
        fruits: List[Union[AppleSchema, BananaSchema]]

    _test_json_schema_parsing_with_string('{"fruits": [{"apple_type": 1}, {"apple_type": 2}] }', FruitsSchema.model_json_schema(), True)
    _test_json_schema_parsing_with_string('{"fruits": [{"apple_type": 1}, {"is_ripe": true}] }', FruitsSchema.model_json_schema(), True)
    _test_json_schema_parsing_with_string('{"fruits": [{"apple_type": 1, "is_ripe": true}] }', FruitsSchema.model_json_schema(), False)


def test_empty_list_with_newline():
    class EmptyListOKModel(BaseModel):
        num: int
        list_of_strings: Optional[List[str]] = Field(None, min_length=0, max_length=1)
    
    no_strings = '{"num":1,"list_of_strings":[\n]}'
    _test_json_schema_parsing_with_string(no_strings, EmptyListOKModel.model_json_schema(), True)


def test_comma_cannot_start_list():
    class FlightRoute(BaseModel):
        airports: List[str]
    output_notok = """ { "airports": [,"name"] } """
    output_ok = """ { "airports": ["name"] } """
    
    _test_json_schema_parsing_with_string(output_ok, FlightRoute.model_json_schema(), True)
    _test_json_schema_parsing_with_string(output_notok, FlightRoute.model_json_schema(), False)


def test_comma_cannot_start_list_2():
    # This also stresses the whitespace handling + max consecutive whitespace concept,
    # bug reported in https://github.com/noamgat/lm-format-enforcer/issues/80
    output_notok = """
    {
        "airports": [
           ,"Hamad",
           ",Doha",
           ",Bahrain",
           ",Dammam"
        ]
    }"""
    class FlightRoute(BaseModel):
        airports: List[str]
    _test_json_schema_parsing_with_string(output_notok, FlightRoute.model_json_schema(), False)


def test_multi_function_schema():
    # https://github.com/noamgat/lm-format-enforcer/issues/95
    _multi_function_schema = {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": [
                    "sums",
                    "concat"
                ]
            }
        },
        "oneOf": [
            {
            "properties": {
                "name": {
                "const": "sums"
                },
                "arguments": {
                "properties": {
                    "a": {
                    "type": "integer"
                    },
                    "b": {
                    "default": 1,
                    "type": "integer"
                    }
                },
                "required": [
                    "a"
                ],
                "type": "object"
                }
            }
            },
            {
            "properties": {
                "name": {
                "const": "concat"
                },
                "arguments": {
                "properties": {
                    "c": {
                    "type": "string"
                    },
                    "d": {
                    "default": 1,
                    "type": "string"
                    }
                },
                "required": [
                    "c"
                ],
                "type": "object"
                }
            }
            }
        ],
        "required": [
            "name",
            "arguments"
        ]
    }
    valid_examples = [
        """{"name": "concat", "arguments": {"c": "hello", "d": "world"}}""",
        """{"name": "sums", "arguments": {"a": 1}}""",
    ]
    invalid_examples = [
        """{"name": "concat", "arguments": {"b": 1}}""",
        """{"name": "concat", "arguments": {"a": 1}}""",
        """{"name": "concat"}""",
        """{"name": "badname", "arguments": {"c": "hello", "b": "world"}}""",
    ]
    for example in valid_examples:
        _test_json_schema_parsing_with_string(example, _multi_function_schema, True)
    for example in invalid_examples:
        _test_json_schema_parsing_with_string(example, _multi_function_schema, False)


def test_top_level_array_object():
    test_schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "arguments": {
                    "type": "object"
                },
                "name": {
                    "type": "string"
                }
            },
            "required": [
                "name",
                "arguments"
            ]
        },
        "minItems": 1
    }
    valid_result = """[
  {
    "name": "sums",
    "arguments": {
      "a": 5,
      "b": 6
    }
  },
  {
    "name": "sums",
    "arguments": {
      "a": 2,
      "b": 7
    }
  },
  {
    "name": "subtraction",
    "arguments": {
      "c": 3,
      "d": 3
    }
  }]"""
    invalid_result = valid_result[:-1]
    _test_json_schema_parsing_with_string(valid_result, test_schema, True)
    _test_json_schema_parsing_with_string(invalid_result, test_schema, False)


def test_arrays_with_multiple_enums():
    schema = {
        "properties": {
            "array_of_numbers": {
                "default": [],
                "description": "An array of numbers",
                "items": {
                    "type": "integer",
                    "enum": [1, 2, 3, 4, 5],
                },
                "title": "Items",
                "type": "array",
                "maxItems": 2
            }
        },
        "required": ["array_of_numbers"],
        "type": "object",
    }
    
    _test_json_schema_parsing_with_string('{"array_of_numbers":[4]}', schema, True)
    _test_json_schema_parsing_with_string('{"array_of_numbers":[4, 1]}', schema, True)
    _test_json_schema_parsing_with_string('{"array_of_numbers":[4, 4]}', schema, True)
    _test_json_schema_parsing_with_string('{"array_of_numbers":[1, 2, 3]}', schema, False)
    _test_json_schema_parsing_with_string('{"array_of_numbers":[6]}', schema, False)
    _test_json_schema_parsing_with_string('{"array_of_numbers":[1, 6]}', schema, False)

@contextmanager
def _temp_replace_env_var(env_var_name, temp_value):
    try:
        prev_env_var = os.environ.get(env_var_name, None)
        if prev_env_var is not None:
            os.environ.pop(env_var_name)
        if temp_value is not None:
            os.environ[env_var_name] = str(temp_value)
        yield None
    finally:
        if prev_env_var is None:
            if temp_value is not None:
                os.environ.pop(env_var_name)
        else:
            os.environ[env_var_name] = prev_env_var


def test_control_json_force_field_order_via_env_var():
    class TwoRequiredModel(BaseModel):
        a: int
        b: str
        c: int = 1
    schema = TwoRequiredModel.model_json_schema()
    env_var_name = CONFIG_ENV_VAR_STRICT_JSON_FIELD_ORDER
    with _temp_replace_env_var(env_var_name, None):
        # Check that the default is false
        _test_json_schema_parsing_with_string('{"b": "X", "a": 1}', schema, True)
    with _temp_replace_env_var(env_var_name, 'True'):
        # Check that setting to true behaves correctly
        _test_json_schema_parsing_with_string('{"b": "X", "a": 1}', schema, False)
        _test_json_schema_parsing_with_string('{"a": 1, "b": "X"}', schema, True)


def test_max_whitespaces_via_env_var():
    env_var_name = CONFIG_ENV_VAR_MAX_CONSECUTIVE_WHITESPACES
    schema = SampleModel.model_json_schema()
    base_answer = '{"num":$1}'
    with _temp_replace_env_var(env_var_name, '8'):
        for num_spaces in range(12):
            expect_success = num_spaces <= 8
            _test_json_schema_parsing_with_string(base_answer.replace("$", " " * num_spaces), schema, expect_success)
