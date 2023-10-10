import abc
from typing import Optional


class CharacterLevelParser(abc.ABC):
    """CharacterLevelParser is an interface for classes that can parse strings one character at a time, and determine which characters are allowed at any specific time"""
    @abc.abstractmethod
    def add_character(self, new_character: str) -> 'CharacterLevelParser':
        """Add a character to the parser, and return a new parser that represents the state of the parser after the character has been added. This has to be
        an immutable operation - the original CharacterLevelParser (self) must not be modified."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_allowed_characters(self) ->str:
        """Return a string containing all characters that are allowed at the current point in the parsing process."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def can_end(self) -> bool:
        """Return True if the parser is in a state where it can end (potentially finished parsing the desired structure), and False otherwise."""
        raise NotImplementedError()
    
    def shortcut_key(self) -> Optional[str]:
        """Optional. Return a string that denotes that this state is a repeating state, full tree traversal should be avoided."""
        return None


class StringParser(CharacterLevelParser):
    """RegexParser is an example CharacterLevelParser that only allows an exact string. It is a debugging / learning tool
    to show how CharacterLevelParser works together with TokenizerPrefixTree to filter the allowed tokens (some of whom may contain multiple characters)"""
    def __init__(self, string: str):
        self.target_str = string

    def add_character(self, new_character: str) -> CharacterLevelParser:
        if self.target_str.startswith(new_character):
            return StringParser(self.target_str[len(new_character):])
        else:
            raise ValueError(f"Expected '{self.target_str[0]}' but got '{new_character}'")

    def get_allowed_characters(self) -> str:
        return self.target_str[0] if self.target_str else ""

    def can_end(self) -> bool:
        return not self.target_str
    

class ForceStopParser(CharacterLevelParser):
    """A simple parser that forbids any characters except the stop token. Used to force stop LM operation"""
    def add_character(self, new_character: str) -> CharacterLevelParser:
        return self
    def get_allowed_characters(self) -> str:
        return ""
    def can_end(self) -> bool:
        return True
