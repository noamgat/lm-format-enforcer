from typing import Optional, Set
from .characterlevelparser import CharacterLevelParser
from .external import regex

class SequentialPattern:
    REPLACEMENT_DICT = {
        "\\d" : "(0|1|2|3|4|5|6|7|8|9)"
    }
    def __init__(self, pattern: str, compiled_pattern = None, current_states: Optional[set] = None):
        if not compiled_pattern:
            for src, tgt in SequentialPattern.REPLACEMENT_DICT.items():
                pattern = pattern.replace(src, tgt)
        self.compiled_pattern = compiled_pattern or regex.compile(pattern)
        self.current_states = current_states
        if current_states is None:
            self.current_states = set()
            self.compiled_pattern.addstate(self.compiled_pattern.start, self.current_states)

    def add_characters(self, s: str):
        current_states = self.current_states
        for c in s:
            next_states = set()
            for state in current_states:
                if c in state.transitions.keys():
                    trans_state = state.transitions[c]
                    self.compiled_pattern.addstate(trans_state, next_states)
           
            current_states = next_states
            if not current_states:
                break
        return SequentialPattern(None, self.compiled_pattern, current_states)
    
    @property
    def can_end(self) -> bool:
        for s in self.current_states:
            if s.is_end:
                return True
        return False
    
    @property
    def is_dead_end(self) -> bool:
         return not self.current_states
    
    @property
    def potential_next_characters(self) -> Set[str]:
        next_characters = set()
        for state in self.current_states:
            next_characters.update(state.transitions.keys())
        return next_characters


class RegexParser(CharacterLevelParser):
    """RegexParser is an example CharacterLevelParser that only allows strings that match a given regular expression.
    Due to the get_allowed_characters() requirement, we use a custom regex implementation which does not support all of the regex capabilites."""
    sequential_pattern: SequentialPattern

    def __init__(self, regex: str, existing_pattern: SequentialPattern = None) -> None:
        self.sequential_pattern: SequentialPattern = existing_pattern or SequentialPattern(regex)

    def add_character(self, new_character: str) -> 'RegexParser':
        return RegexParser(None, self.sequential_pattern.add_characters(new_character))

    def get_allowed_characters(self) -> str:
        return "".join(self.sequential_pattern.potential_next_characters)

    def can_end(self) -> bool:
        return self.sequential_pattern.can_end
