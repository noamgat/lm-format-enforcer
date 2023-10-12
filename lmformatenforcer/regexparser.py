from typing import Dict, Union
import interegular
from interegular.fsm import anything_else

from .characterlevelparser import CharacterLevelParser
from .consts import COMPLETE_ALPHABET

class RegexParser(CharacterLevelParser):
    """RegexParser is an example CharacterLevelParser that only allows strings that match a given regular expression."""

    UNINITIALIZED_STATE = -1
    INVALID_STATE = -2

    class _Context:
        pattern: interegular.FSM
        anything_else_characters: str
        state_character_cache: Dict[int, str]
    
    context: _Context
    current_state: int

    def __init__(self, pattern: Union[str, _Context], current_state: int = UNINITIALIZED_STATE):
        if isinstance(pattern, str):
            self.context = RegexParser._Context()
            self.context.pattern = interegular.parse_pattern(pattern).to_fsm()
            self.context.state_character_cache = {}
            not_anything_else_characters = set([c for c in self.context.pattern.alphabet.keys() if c != anything_else])
            self.context.anything_else_characters = "".join([c for c in COMPLETE_ALPHABET if c not in not_anything_else_characters])
        else:
            self.context = pattern
        self.current_state: int = self.context.pattern.initial if current_state == RegexParser.UNINITIALIZED_STATE else current_state

    def add_character(self, new_character: str) -> 'RegexParser':
        if self.current_state == RegexParser.INVALID_STATE:
            return self
        
        state = self.current_state
        fsm = self.context.pattern
        # Mostly taken from FSM.accept()
        symbol = new_character
        if anything_else in fsm.alphabet and not symbol in fsm.alphabet:
            symbol = anything_else
        transition = fsm.alphabet[symbol]

        # Missing transition = transition to dead state
        if not (state in fsm.map and transition in fsm.map[state]):
            return RegexParser(self.context, RegexParser.INVALID_STATE)

        state = fsm.map[state][transition]

        return RegexParser(self.context, state)
    
    def can_end(self) -> bool:
        return self.current_state in self.context.pattern.finals
    
    def get_allowed_characters(self) -> str:
        if self.current_state not in self.context.pattern.map:
            return ''
        if self.current_state not in self.context.state_character_cache:
            allowed_characters = []
            state_map = self.context.pattern.map[self.current_state]
            for symbol, symbol_idx in self.context.pattern.alphabet.items():
                if symbol_idx in state_map:
                    if symbol == anything_else:
                        allowed_characters.append(self.context.anything_else_characters)
                    else:
                        allowed_characters.append(symbol)
            self.context.state_character_cache[self.current_state] = "".join(allowed_characters)
        return self.context.state_character_cache[self.current_state]

