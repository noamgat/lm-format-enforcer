import abc
from dataclasses import dataclass
from typing import Hashable, List, Optional
from .consts import COMPLETE_ALPHABET


@dataclass
class CharacterLevelParserConfig:
        alphabet: str = COMPLETE_ALPHABET


class CharacterLevelParser(abc.ABC):
    """CharacterLevelParser is an interface for classes that can parse strings one character at a time, and determine which characters are allowed at any specific time"""

    def __init__(self, config: Optional[CharacterLevelParserConfig] = None):
        self._config = config or CharacterLevelParserConfig()
    
    @abc.abstractmethod
    def add_character(self, new_character: str) -> 'CharacterLevelParser':
        """Add a character to the parser, and return a new parser that represents the state of the parser after the character has been added. This has to be
        an immutable operation - the original CharacterLevelParser (self) must not be modified."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_allowed_characters(self) -> str:
        """Return a string containing all characters that are allowed at the current point in the parsing process."""
        raise NotImplementedError()
    
    @abc.abstractmethod
    def can_end(self) -> bool:
        """Return True if the parser is in a state where it can end (potentially finished parsing the desired structure), and False otherwise."""
        raise NotImplementedError()
    
    def shortcut_key(self) -> Optional[str]:
        """Optional. Return a string that denotes that this state is a repeating state, full tree traversal should be avoided."""
        return None
    
    def cache_key(self) -> Optional[Hashable]:
        """Optional. Return a string that denotes that this state is a repeating state, and if it is visited again, results can be cached."""
        return None
    
    @property
    def config(self) -> CharacterLevelParserConfig:
        return self._config
    
    @config.setter
    def config(self, new_config: CharacterLevelParserConfig):
        self._config = new_config
        return self


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
    

class UnionParser(CharacterLevelParser):
    """A parser that allows a string that would be allowed by any of several different parsers"""
    def __init__(self, parsers: List[CharacterLevelParser]):
        self.parsers = parsers

    def add_character(self, new_character: str) -> CharacterLevelParser:
        # This is a bit of a performance hit, as it means get_allowed_characters() is called twice.
        relevant_parsers = [parser for parser in self.parsers if new_character in parser.get_allowed_characters()]
        next_parsers = [parser.add_character(new_character) for parser in relevant_parsers]
        if len(next_parsers) == 1:
            return next_parsers[0]
        return UnionParser(next_parsers)
    
    def get_allowed_characters(self) -> str:
        allowed = "".join([parser.get_allowed_characters() for parser in self.parsers])
        return "".join(set(allowed))
    
    def can_end(self) -> bool:
        return any([parser.can_end() for parser in self.parsers])
    
    def shortcut_key(self) -> Optional[str]:
        return self.parsers[0].shortcut_key() if len(self.parsers) == 1 else None
    
    def cache_key(self) -> Optional[Hashable]:
        all_cache_keys = tuple(parser.cache_key() for parser in self.parsers)
        if all(key is not None for key in all_cache_keys):
            return ('union', all_cache_keys)
        return None


class SequenceParser(CharacterLevelParser):
    """A parser that is a sequence of multiple parsers."""
    def __init__(self, parsers: List[CharacterLevelParser]):
        self.parsers = parsers

    def add_character(self, new_character: str) -> CharacterLevelParser:
        legal_parsers = []
        # Tricky edge case: if the first parser can both end and accept the character,
        # and the second parser can also accept, we don't know which scenario we are dealing
        # with, so we need to return a UnionParser.
        for idx, parser in enumerate(self.parsers):
            if new_character in parser.get_allowed_characters():
                updated_parser = parser.add_character(new_character)
                next_parsers = [updated_parser] + self.parsers[idx+1:]
                legal_parsers.append(SequenceParser(next_parsers))
            if not parser.can_end():
                break
        if len(legal_parsers) == 1:
            return legal_parsers[0]
        return UnionParser(legal_parsers)
    
    def get_allowed_characters(self) -> str:
        allowed_characters = set()
        for parser in self.parsers:
            allowed_characters.update(parser.get_allowed_characters())
            if not parser.can_end():
                break
        return "".join(allowed_characters)
    
    def can_end(self) -> bool:
        return all([parser.can_end() for parser in self.parsers])
    
    def shortcut_key(self) -> Optional[str]:
        return self.parsers[0].shortcut_key() if len(self.parsers) == 1 else None
    
    def cache_key(self) -> Optional[Hashable]:
        all_cache_keys = tuple(parser.cache_key() for parser in self.parsers)
        if all(key is not None for key in all_cache_keys):
            return ('sequence', all_cache_keys)
        return None


