from lmformatenforcer import CharacterLevelParser
from lmformatenforcer.exceptions import LMFormatEnforcerException


class CharacterNotAllowedException(LMFormatEnforcerException):
    pass


def assert_parser_with_string(string: str, parser: CharacterLevelParser, expect_success: bool):
    for idx, character in enumerate(string):
        try:
            if character in parser.get_allowed_characters():
                parser = parser.add_character(character)
            else:
                if expect_success:
                    raise CharacterNotAllowedException(f"Parser does not allow '{character}' at index {idx}")
                else:
                    return  # Success
        except LMFormatEnforcerException:
            raise
        except Exception as e:
            raise Exception(f"Error parsing '{character}' at index {idx}: {e}", e)
    if parser.can_end() and not expect_success:
        raise ValueError("Parser succeeded when it should have failed")
    if not parser.can_end() and expect_success:
        raise ValueError("Parser did not reach end state when it should have")