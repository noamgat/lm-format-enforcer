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
        state_token_cache: Dict[int, List[int]] = {}  # Cache allowed tokens per FSM state
    
    context: _Context
    current_state: int
    pattern: Union[str, None]
    pattern_hash: Union[int, None]
    parsed_string: str

    def __init__(self, pattern: Union[str, _Context], config: Optional[CharacterLevelParserConfig] = None, current_state: int = UNINITIALIZED_STATE):
        super().__init__(config)
        if isinstance(pattern, str):
            self.context = RegexParser._Context()
            self.context.pattern = interegular.parse_pattern(pattern).to_fsm()
            self.context.state_character_cache = {}
            self._update_alphabet(self.config.alphabet)
            self.pattern = pattern
            self.pattern_hash = hash(pattern)
            self.parsed_string = ""
        else:
            self.context = pattern
            self.pattern = pattern.pattern if hasattr(pattern, 'pattern') else None
            self.pattern_hash = hash(self.pattern) if self.pattern else None
            self.parsed_string = ""
        self.current_state = self.context.pattern.initial if current_state == RegexParser.UNINITIALIZED_STATE else current_state

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
            new_parser = RegexParser(self.context, self.config, state)
            new_parser.pattern = self.pattern
            new_parser.pattern_hash = self.pattern_hash
            new_parser.parsed_string = self.parsed_string + new_character
            return new_parser
        except KeyError:
            # Missing transition = transition to dead state
            new_parser = RegexParser(self.context, self.config, RegexParser.INVALID_STATE)
            new_parser.pattern = self.pattern
            new_parser.pattern_hash = self.pattern_hash
            new_parser.parsed_string = self.parsed_string
            return new_parser
    
    def can_end(self) -> bool:
        return self.current_state in self.context.pattern.finals or self.current_state == RegexParser.INVALID_STATE
    
    def get_allowed_characters(self) -> str:
        # Only compute allowed characters if we don't have a token cache for this state
        if self.current_state in self.context.state_token_cache:
            return ''  # Skip character exploration if we have cached tokens
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
    
    def cache_key(self) -> Optional[Hashable]:
        # If we are in the same regex fsm state, the allowed next tokens are the same ones
        if self.current_state == RegexParser.INVALID_STATE:
            return None
        return ('regex_state', self.pattern_hash, self.current_state)

    def _update_alphabet(self, new_alphabet: str):
        if self.context:
            not_anything_else_characters = set([c for c in self.context.pattern.alphabet.keys() if c != anything_else])
            self.context.anything_else_characters = "".join([c for c in new_alphabet if c not in not_anything_else_characters])    
    
    @CharacterLevelParser.config.setter
    def config(self, new_config: CharacterLevelParserConfig):
        CharacterLevelParser.config.fset(self, new_config)  # Original set
        self._update_alphabet(new_config.alphabet)

    def shortcut_key(self) -> Optional[Hashable]:
        if self.pattern is None:
            return None
        return ('regex_pattern', self.pattern_hash, len(self.parsed_string))


