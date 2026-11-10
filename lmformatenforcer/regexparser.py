from typing import Dict, Hashable, Optional, Union, List
import interegular
from interegular.fsm import anything_else

from .characterlevelparser import CharacterLevelParser, CharacterLevelParserConfig

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

    def __init__(self, pattern: Union[str, _Context], config: Optional[CharacterLevelParserConfig] = None, current_state: int = UNINITIALIZED_STATE):
        super().__init__(config)
        if isinstance(pattern, str):
            self.context = RegexParser._Context()
            self.context.pattern = interegular.parse_pattern(pattern).to_fsm()
            self.context.state_character_cache = {}
            self._update_alphabet(self.config.alphabet)
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

        try:
            # Prefer try-catch to checking if transition exists to avoid double lookup perf hit in valid case
            state = fsm.map[state][transition]  # type: ignore
            return RegexParser(self.context, self.config, state)
        except KeyError:
            # Missing transition = transition to dead state
            return RegexParser(self.context, self.config, RegexParser.INVALID_STATE)
    
    def can_end(self) -> bool:
        return self.current_state in self.context.pattern.finals or self.current_state == RegexParser.INVALID_STATE
    
    def get_allowed_characters(self) -> str:
        if self.current_state not in self.context.pattern.map:
            return ''
        if self.current_state not in self.context.state_character_cache:
            allowed_characters = []
            state_map = self.context.pattern.map[self.current_state]
            for symbol_idx in state_map:
                symbols: List[str] = self.context.pattern.alphabet.by_transition[symbol_idx]
                for symbol in symbols:
                    if symbol == anything_else:
                        allowed_characters.append(self.context.anything_else_characters)
                    else:
                        allowed_characters.append(symbol)
            self.context.state_character_cache[self.current_state] = "".join(allowed_characters)
        return self.context.state_character_cache[self.current_state]
    
    def allows_character(self, character: str) -> bool:
        fsm = self.context.pattern
        if self.current_state not in fsm.map:
            return False
        if character in fsm.alphabet:
            symbol = character
        elif anything_else in fsm.alphabet and (self.config.allow_unicode or character in self.config.alphabet):
            symbol = anything_else
        else:
            return False
        return fsm.alphabet[symbol] in fsm.map[self.current_state]

    def has_allowed_in_range(self, lo: int, hi: int) -> bool:
        fsm = self.context.pattern
        if self.current_state not in fsm.map:
            return False
        transitions = fsm.map[self.current_state]
        if any(lo <= ord(c) <= hi and self.allows_character(c)
               for c in fsm.alphabet if c != anything_else):
            return True
        if anything_else not in fsm.alphabet or fsm.alphabet[anything_else] not in transitions:
            return False
        if not self.config.allow_unicode:
            return any(lo <= ord(c) <= hi and c not in fsm.alphabet for c in self.config.alphabet)
        excluded = sum(1 for c in fsm.alphabet if c != anything_else and lo <= ord(c) <= hi)
        return hi - lo + 1 > excluded

    def cache_key(self) -> Optional[Hashable]:
        # If we are in the same regex fsm state, the allowed next tokens are the same ones
        return self.current_state

    def _update_alphabet(self, new_alphabet: str):
        if self.context:
            # Byte-aware configuration needs fresh character caches; retain the
            # legacy direct-then-enforcer cache behavior for text tokenizers.
            if self.config.byte_mode:
                self.context.state_character_cache.clear()
            not_anything_else_characters = set([c for c in self.context.pattern.alphabet.keys() if c != anything_else])
            self.context.anything_else_characters = "".join([c for c in new_alphabet if c not in not_anything_else_characters])    
    
    @CharacterLevelParser.config.setter
    def config(self, new_config: CharacterLevelParserConfig):
        CharacterLevelParser.config.fset(self, new_config)  # Original set
        self._update_alphabet(new_config.alphabet)


