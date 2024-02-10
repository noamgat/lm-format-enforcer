from typing import Optional
from lmformatenforcer import RegexParser, CharacterLevelParserConfig
from lmformatenforcer.consts import COMPLETE_ALPHABET
from .common import assert_parser_with_string

def _test_regex_parsing_with_string(string: str, regex: str, expect_success: bool, custom_alphabet: Optional[str] = None, profile_file_path: Optional[str] = None):
    parser = RegexParser(regex)
    if custom_alphabet:
        parser.config = CharacterLevelParserConfig(alphabet=custom_alphabet)
    assert_parser_with_string(string, parser, expect_success, profile_file_path=profile_file_path)


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
        

def test_any_character():
    for num_repeats, character in enumerate('0123456789abcdefghij'):
        expect_success = num_repeats > 0
        _test_regex_parsing_with_string(
            string=f'ab{character * num_repeats}123', 
            regex='ab.+123', 
            expect_success=expect_success,
            #profile_file_path=f'RegexAny{num_repeats}.prof')
            profile_file_path=None)


def test_dates():
    # https://stackoverflow.com/q/15491894 , removed the ^ and $ because interegular doesn't support them
    date_regex = r'(0?[1-9]|[12][0-9]|3[01])[\/\-](0?[1-9]|1[012])[\/\-]\d{4}'
    _test_regex_parsing_with_string('01/01/2020', date_regex, True)
    _test_regex_parsing_with_string('29/04/1986', date_regex, True)
    _test_regex_parsing_with_string('001/01/2020', date_regex, False)


def test_string_choice():
    choice_regex = r'abc|def|ghi'
    _test_regex_parsing_with_string('abc', choice_regex, True)
    _test_regex_parsing_with_string('def', choice_regex, True)
    _test_regex_parsing_with_string('ghi', choice_regex, True)
    _test_regex_parsing_with_string('aei', choice_regex, False)


def test_increasing_alphabet():
    any_regex = '...'
    _test_regex_parsing_with_string('abc', any_regex, True)
    _test_regex_parsing_with_string('abΣ', any_regex, False)
    custom_alphabet = COMPLETE_ALPHABET + 'Σ'
    _test_regex_parsing_with_string('abΣ', any_regex, True, custom_alphabet=custom_alphabet)

def test_phone_number():
    phone_regex = r"\([0-9]{3}\)[0-9]{3}-[0-9]{4}"
    _test_regex_parsing_with_string('(312)011-2444', phone_regex, True)
    _test_regex_parsing_with_string('312-011-2444', phone_regex, False)

def test_negative_matching():
    # https://github.com/noamgat/lm-format-enforcer/issues/70
    pattern = r'- Keywords: [^;:,/\n\r]+; [^;:,/\n\r]+; [^;:,/\n\r]+; [^;:,/\n\r]+; [^;:,/\n\r]+'
    text = '- Keywords: intranasal vaccine, long-lasting immunity, mucosal antibody response, T cells, adjuvants'
    _test_regex_parsing_with_string(text, pattern, False)
    correct_text = text.replace(',', ';')
    _test_regex_parsing_with_string(correct_text, pattern, True)
