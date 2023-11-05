import json
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from lmformatenforcer import JsonSchemaParser
from enum import Enum
import pytest

from lmformatenforcer.exceptions import LMFormatEnforcerException

from .common import assert_parser_with_string


def _test_json_schema_parsing_with_string(string: str, schema_dict: dict, expect_success: bool):
    parser = JsonSchemaParser(schema_dict)
    assert_parser_with_string(string, parser, expect_success)
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
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)

def test_parsing_test_model():
    test_string = '{"num":1,"dec":1.1,"message":"ok","list_of_strings":["a","b","c"],"inner_dict":{"a":{"list_of_ints":[1,2,3]}}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)


def test_invalid_key_in_json_string():
    test_string = '{"numa":1,"dec":1.1,"message":"ok","list_of_strings":["a","b","c"],"inner_dict":{"a":{"list_of_ints":[1,2,3]}}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), False)


def test_incomplete_json():
    # Intentionally missing closing }
    test_string = '{"num":1,"dec":1.1,"message":"ok","list_of_strings":["a","b","c"],"inner_dict":{"a":{"list_of_ints":[1,2,3]}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), False)


def test_invalid_value_type_in_json_string():
    test_string = '{"num:"1","dec":1.1,"message":"ok","list_of_strings":["a","b","c"],"inner_dict":{"a":{"list_of_ints":[1,2,3]}}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), False)


def test_list_of_objects():
    test_string = '{"list_of_models":[{"list_of_ints":[1,2,3]},{"list_of_ints":[4,5,6]}],"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)
    test_string = '{"list_of_models": [{"list_of_ints":[1, 2, 3]} , {"list_of_ints":[4,5,6]}],"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)


def test_simple_dict():
    test_string = '{"simple_dict":{"a":1,"b":2,"c":3},"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)


def test_int_enum():
    test_string = '{"enum":4,"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)


def test_invalid_int_enum_value():
    test_string = '{"enum":5,"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), False)


def test_str_enum():
    test_string = '{"enum_dict":{"a":"One","b":"Two","c":"Three","d":"Four"},"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)


def test_invalid_str_enum_value():
    test_string = '{"enum_dict":{"a":"Onee"},"num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), False)


def test_whitespaces():
    test_string = '{ "message": "","num":1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)


def test_whitespace_before_number():
    test_string = '{"num": 1, "dec": 1.1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)


def test_whitespace_before_close():
    test_string = '{"num":1 }'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)    


def test_required_field():
    test_string = '{"dec": 1.1}'  # num is a required field, doesn't exist, should fail.
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), False)


def test_boolean_field():
    _test_json_schema_parsing_with_string('{"num":1,"true_or_false":false}', SampleModel.schema(), True)
    _test_json_schema_parsing_with_string('{"num":1,"true_or_false":true}', SampleModel.schema(), True)
    _test_json_schema_parsing_with_string('{"num":1,"true_or_false": true}', SampleModel.schema(), True)
    _test_json_schema_parsing_with_string('{"num":1,"true_or_false":falsy}', SampleModel.schema(), False)

def test_unspecified_dict_fails():
    class DictModel(BaseModel):
        num: int
        d: dict

    # dict is not valid because we don't know what the value type is, expect our exception
    with pytest.raises(LMFormatEnforcerException):
        _test_json_schema_parsing_with_string('{"num":1,"d":{"k":"v"}}', DictModel.schema(), False)


def test_unspecified_list_fails():
    class DictModel(BaseModel):
        num: int
        l: list

    # list is not valid because we don't know what the member type is, expect our exception
    with pytest.raises(LMFormatEnforcerException):
        _test_json_schema_parsing_with_string('{"num":1,"l":[1,2,3]}', DictModel.schema(), False)


def test_list_length_limitations():
    # list_of_strings is defined as having a min length of 2 and a max length of 3
    no_strings = '{"num":1,"list_of_strings":[]}'
    _test_json_schema_parsing_with_string(no_strings, SampleModel.schema(), False)
    one_string = '{"num":1,"list_of_strings":["a"]}'
    _test_json_schema_parsing_with_string(one_string, SampleModel.schema(), False)
    two_strings = '{"num":1,"list_of_strings":["a", "b"]}'
    _test_json_schema_parsing_with_string(two_strings, SampleModel.schema(), True)
    three_strings = '{"num":1,"list_of_strings":["a","b","c"]}'
    _test_json_schema_parsing_with_string(three_strings, SampleModel.schema(), True)
    four_strings = '{"num":1,"list_of_strings":["a","b","c","d"]}'
    _test_json_schema_parsing_with_string(four_strings, SampleModel.schema(), False)

    class EmptyListOKModel(BaseModel):
        num: int
        list_of_strings: Optional[List[str]] = Field(None, min_length=0, max_length=1)
    _test_json_schema_parsing_with_string(no_strings, EmptyListOKModel.schema(), True)
    _test_json_schema_parsing_with_string(one_string, EmptyListOKModel.schema(), True)
    _test_json_schema_parsing_with_string(two_strings, EmptyListOKModel.schema(), False)

    class ListOfExactlyOneModel(BaseModel):
        num: int
        list_of_strings: Optional[List[str]] = Field(None, min_length=1, max_length=1)
    _test_json_schema_parsing_with_string(no_strings, ListOfExactlyOneModel.schema(), False)
    _test_json_schema_parsing_with_string(one_string, ListOfExactlyOneModel.schema(), True)
    _test_json_schema_parsing_with_string(two_strings, ListOfExactlyOneModel.schema(), False)

    class ListOfNoMinLengthModel(BaseModel):
        num: int
        list_of_strings: Optional[List[str]] = Field(None, max_length=1)
    _test_json_schema_parsing_with_string(no_strings, ListOfNoMinLengthModel.schema(), True)
    _test_json_schema_parsing_with_string(one_string, ListOfNoMinLengthModel.schema(), True)
    _test_json_schema_parsing_with_string(two_strings, ListOfNoMinLengthModel.schema(), False)
