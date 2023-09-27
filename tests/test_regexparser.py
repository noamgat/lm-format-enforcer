from lmformatenforcer import RegexParser
from .common import assert_parser_with_string

def _test_regex_parsing_with_string(string: str, regex: str, expect_success: bool):
    parser = RegexParser(regex)
    assert_parser_with_string(string, parser, expect_success)


def test_parsing_exact_string():
    _test_regex_parsing_with_string(
        string='abc123', 
        regex='abc123', 
        expect_success=True)

def test_parsing_exact_string_failure():
    _test_regex_parsing_with_string(
        string='abc124', 
        regex='abc123', 
        expect_success=False)
    
def test_parsing_exact_string_not_reaching_end():
    _test_regex_parsing_with_string(
        string='abc123', 
        regex='abc1234', 
        expect_success=False)


def test_parsing_letter_options():
    for letter in 'cdefghif':
        expect_success = letter in 'cdef'
        _test_regex_parsing_with_string(
            string=f'ab{letter}123', 
            regex='ab(c|d|e|f)123', 
            expect_success=expect_success)
        

def test_parsing_digits():
    for letter in '0123abcd':
        expect_success = letter.isnumeric()
        _test_regex_parsing_with_string(
            string=f'ab{letter}123', 
            regex='ab\d123', 
            expect_success=expect_success)
        

def test_parsing_repeat():
    for num_repeats in range(20):
        expect_success = num_repeats > 0
        _test_regex_parsing_with_string(
            string=f'ab{"c" * num_repeats}123', 
            regex='abc+123', 
            expect_success=expect_success)