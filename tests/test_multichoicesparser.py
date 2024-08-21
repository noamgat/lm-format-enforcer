from typing import Optional
from lmformatenforcer import MultiChoicesParser, CharacterLevelParserConfig
from lmformatenforcer.consts import COMPLETE_ALPHABET
from .common import assert_parser_with_string

def _test_mcp_parsing_with_string(string: str, list_of_choices: list[list[str]], expect_success: bool, custom_alphabet: Optional[str] = None, profile_file_path: Optional[str] = None):
    parser = MultiChoicesParser(list_of_choices)
    if custom_alphabet:
        parser.config = CharacterLevelParserConfig(alphabet=custom_alphabet)
    assert_parser_with_string(string, parser, expect_success, profile_file_path=profile_file_path)


def test_parsing_exact_string():
    _test_mcp_parsing_with_string(
        string='abc123', 
        list_of_choices=[['abc123']], 
        expect_success=True)

def test_parsing_exact_string_failure():
    _test_mcp_parsing_with_string(
        string='abc124', 
        list_of_choices=[['abc123']], 
        expect_success=False)
    
def test_parsing_exact_string_not_reaching_end():
    _test_mcp_parsing_with_string(
        string='abc123', 
        list_of_choices=[['abc1234']], 
        expect_success=False)


def test_parsing_letter_options():
    for letter in 'cdefghif':
        expect_success = letter in 'cdef'
        _test_mcp_parsing_with_string(
            string=f'ab{letter}123', 
            list_of_choices=[
                ['ab'],
                list('cdef'),
                ['123']
            ], 
            expect_success=expect_success)
        

def test_parsing_digits():
    for letter in '0123abcd':
        expect_success = letter.isnumeric()
        _test_mcp_parsing_with_string(
            string=f'ab{letter}123', 
            list_of_choices=[
                ['ab'],
                list('0123456789'),
                ['123']
            ],  
            expect_success=expect_success)
        

def test_parsing_repeat():
    for num_repeats in range(20):
        expect_success = num_repeats > 0
        _test_mcp_parsing_with_string(
            string=f'ab{"c" * num_repeats}123', 
            list_of_choices=[
                ['ab'],
                *([['c']] + [["c", ""]]*(num_repeats-1)),
                ['123']
            ],  
            expect_success=expect_success)
        

def test_any_character():
    chars = list('0123456789abcdefghij') + ['']
    for num_repeats, character in enumerate(chars[:-1]):
        expect_success = num_repeats > 0
        _test_mcp_parsing_with_string(
            string=f'ab{character * num_repeats}123', 
            list_of_choices=[
                ['ab'],
                *([chars[:-1]] + [chars]*(num_repeats-1)),
                ['123']
            ],  
            expect_success=expect_success,
            #profile_file_path=f'RegexAny{num_repeats}.prof')
            profile_file_path=None)


def test_dates():
    # https://stackoverflow.com/q/15491894 , removed the ^ and $ because interegular doesn't support them
    date_lcs = [
        [str(x).zfill(2) for x in range(1,32)],
        ["/"],
        [str(x).zfill(2) for x in range(1,13)],
        ["/"],
        [str(x).zfill(4) for x in range(3000)],
    ]
    _test_mcp_parsing_with_string('01/01/2020', date_lcs, True)
    _test_mcp_parsing_with_string('29/04/1986', date_lcs, True)
    _test_mcp_parsing_with_string('001/01/2020', date_lcs, False)


def test_string_choice():
    lcs = [
        ['abc', 'def', 'ghi']
    ]
    _test_mcp_parsing_with_string('abc', lcs, True)
    _test_mcp_parsing_with_string('def', lcs, True)
    _test_mcp_parsing_with_string('ghi', lcs, True)
    _test_mcp_parsing_with_string('aei', lcs, False)


def test_increasing_alphabet():
    alph = COMPLETE_ALPHABET
    any_3chars = [alph, alph, alph]
    _test_mcp_parsing_with_string('abc', any_3chars, True)
    _test_mcp_parsing_with_string('abΣ', any_3chars, False)
    custom_alphabet = COMPLETE_ALPHABET + 'Σ'
    any_3chars = [custom_alphabet, custom_alphabet, custom_alphabet]
    _test_mcp_parsing_with_string('abΣ', any_3chars, True, custom_alphabet=custom_alphabet)

def test_phone_number():
    phone_lcs = [
        ["("],
        [str(x).zfill(3) for x in range(1000)],
        [")"],
        [str(x).zfill(3) for x in range(1000)],
        ['-'],
        [str(x).zfill(4) for x in range(10000)]
    ]
    _test_mcp_parsing_with_string('(312)011-2444', phone_lcs, True)
    _test_mcp_parsing_with_string('312-011-2444', phone_lcs, False)

# def test_negative_matching():
#     # https://github.com/noamgat/lm-format-enforcer/issues/70
#     pattern = r'- Keywords: [^;:,/\n\r]+; [^;:,/\n\r]+; [^;:,/\n\r]+; [^;:,/\n\r]+; [^;:,/\n\r]+'
#     text = '- Keywords: intranasal vaccine, long-lasting immunity, mucosal antibody response, T cells, adjuvants'
#     _test_regex_parsing_with_string(text, pattern, False)
#     correct_text = text.replace(',', ';')
#     _test_regex_parsing_with_string(correct_text, pattern, True)
