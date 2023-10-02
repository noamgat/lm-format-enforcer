from typing import Dict, List
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser
from enum import Enum
 
from .common import assert_parser_with_string


def _test_json_schema_parsing_with_string(string: str, schema_dict: dict, expect_success: bool):
    parser = JsonSchemaParser(schema_dict)
    assert_parser_with_string(string, parser, expect_success)

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
    dec: float 
    message: str
    list_of_strings: List[str]
    inner_dict: Dict[str, InnerModel]
    simple_dict: Dict[str, int]
    list_of_models: List[InnerModel]
    enum: IntegerEnum
    enum_dict: Dict[str, StringEnum]


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
    test_string = '{"list_of_models":[{"list_of_ints":[1,2,3]},{"list_of_ints":[4,5,6]}]}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)

def test_simple_dict():
    test_string = '{"simple_dict":{"a":1,"b":2,"c":3}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)

def test_int_enum():
    test_string = '{"enum":4}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)

def test_invalid_int_enum_value():
    test_string = '{"enum":5}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), False)

def test_str_enum():
    test_string = '{"enum_dict":{"a":"One","b":"Two","c":"Three","d":"Four"}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)

def test_invalid_str_enum_value():
    test_string = '{"enum_dict":{"a":"Onee"}}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), False)

def test_whitespaces():
    test_string = '{ "message": ""}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)

def test_whitespace_before_number():
    test_string = '{"num": 1, "dec": 1.1}'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)

def test_whitespace_before_close():
    test_string = '{"num":1 }'
    _test_json_schema_parsing_with_string(test_string, SampleModel.schema(), True)    


    
