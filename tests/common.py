from characterlevelparser import CharacterLevelParser


def assert_parser_with_string(string: str, parser: CharacterLevelParser, expect_success: bool):
    for character in string:
        if character in parser.get_allowed_characters():
            parser = parser.add_character(character)
        else:
            if expect_success:
                raise ValueError(f"Parser failed to parse '{character}'")
            else:
                return  # Success
    if parser.can_end() and not expect_success:
        raise ValueError("Parser succeeded when it should have failed")
    if not parser.can_end() and expect_success:
        raise ValueError("Parser did not reach end state when it should have")