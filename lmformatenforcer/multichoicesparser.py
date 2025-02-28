from __future__ import annotations
from .characterlevelparser import CharacterLevelParser, CharacterLevelParserConfig
from multi_choices_parser import MultiChoicesParser as MCP

class MultiChoicesParser(CharacterLevelParser):
    def __init__(self, list_of_choices : list[list[str]] | None):
        """
        NOTE: This parser is based on the multi-choices-parser package: https://github.com/HichemAK/multi-choices-parser.

        A efficient incremental parser for multi-choice grammars. They are defined as grammars of the form:

        start: list1 list2 ... listn

        list1: choice1_1 | choice1_2 | ... | choice1_k1

        list2: choice2_1 | choice2_2 | ... | choice2_k2

        ...
        
        listn: choicen_1 | choicen_2 | ... | choicen_km

        where choicex_y is a sequence of integers and can possibly be empty

        Example:
        start: det noun
        
        det: "the " | "an " | "a " | ""

        noun: "orange" | "apple" | "banana"
        
        This was particularly optimized when the size of the lists of choices is 
        very large (up to order of millions), which can be helpful
        to represent entities preceeded (or not) by a determinent. 
        For example, in Wikipedia, there are around 7 million entities.
        """
        if list_of_choices is not None:
            self.parser = MCP(list_of_choices)
            config = CharacterLevelParserConfig()
        else:
            self.parser = None
            config = None
        
        super().__init__(config)
    
    def add_character(self, new_character: str) -> MultiChoicesParser:
        copy = self.copy()
        copy.parser.step(new_character)
        return copy

    def get_allowed_characters(self) -> str:
        return ''.join(x for x in self.parser.next() if x is not self.parser.end_symb)

    def can_end(self) -> bool:
        return self.parser.end_symb in self.parser.next() or self.parser.finished
    
    def cache_key(self):
        return hash((id(self.parser.current_state), self.parser.finished, self.parser.success))
    
    def copy(self) -> MultiChoicesParser:
        copy = MultiChoicesParser(None)
        copy.parser = self.parser.copy()
        copy.config = self.config
        return copy