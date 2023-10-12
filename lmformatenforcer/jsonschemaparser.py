from copy import deepcopy
from typing import Any, List, Optional, Union

from lmformatenforcer.consts import COMPLETE_ALPHABET

from .external.jsonschemaobject import JsonSchemaObject

from .characterlevelparser import CharacterLevelParser
from .consts import COMPLETE_ALPHABET

class JsonSchemaParser(CharacterLevelParser):
    def __init__(self, json_schema: Union[dict, JsonSchemaObject], existing_stack: List[Any] = None):
        self.model_class = json_schema if isinstance(json_schema, JsonSchemaObject) else JsonSchemaObject(**json_schema)
        self.object_stack: 'List[BaseParsingState]' = existing_stack or [ObjectParsingState(self.model_class, self)]

    def __deepcopy__(self, memo):
        # Avoid cloning the model class, since it is immutable
        for parser in self.object_stack:
            parser.root = None
        clone = JsonSchemaParser(self.model_class, deepcopy(self.object_stack, memo))
        for parser in self.object_stack:
            parser.root = self
        for parser in clone.object_stack:
            parser.root = clone 
        return clone
    
    def add_character(self, new_character: str) -> CharacterLevelParser:
        # The add_character contract requires immutability, therefore we clone before modifying.
        clone = deepcopy(self)
        clone.object_stack[-1].add_character(new_character)
        return clone

    def get_allowed_characters(self) -> str:
        return self.object_stack[-1].get_allowed_characters() if self.object_stack else ""

    def can_end(self) -> bool:
        return not self.object_stack
    
    def finish_parser(self, character_to_pass_to_parent: str = ""):
        self.object_stack.pop()
        if self.object_stack and character_to_pass_to_parent:
            self.object_stack[-1].add_character(character_to_pass_to_parent)

    def shortcut_key(self) -> Optional[str]:
        if self.object_stack:
            current_parser = self.object_stack[-1]
            if isinstance(current_parser, StringParsingState):
                if not current_parser.allowed_strings and current_parser.seen_opening_quote and not current_parser.seen_closing_quote:
                    # Performance optimization: When we are parsing a string that is not from a list of allowed strings, most tokens
                    # are legal. The exploration can be more costly than the LM itself for large tokenizers (because this is pure python),
                    # so we signal that we are in a "freetext" mode, and reuse the allowed token list throughout the run.
                    return 'json_freetext'
        return None


class BaseParsingState:
    def __init__(self, root: JsonSchemaParser):
        self.root = root

    def add_character(self, new_character: str):
        raise NotImplementedError()

    def get_allowed_characters(self) ->str:
        raise NotImplementedError()
    
    def can_end(self) -> bool:
        raise NotImplementedError()


class ObjectParsingStage:
    START_OBJECT = "StartObject"
    PARSING_KEY_OR_END = "ParsingKey"
    PARSING_VALUE = "ParsingValue"
    PARSING_SEPARATOR_OR_END = "ParsingSeparatorOrEnd"
    END_OBJECT = "EndObject"


def get_parser(
    parsing_state: JsonSchemaParser,
    value_schema: JsonSchemaObject,
    ending_characters: str,
) -> BaseParsingState:
    # Sometimes the schema is a union of a type and null, so we need to get the first type
    if value_schema.anyOf and len(value_schema.anyOf) == 2 and value_schema.anyOf[1].type == 'null':
        value_schema = value_schema.anyOf[0]
    if value_schema.type == "string":
        return StringParsingState(
            parsing_state,
            ending_characters,
            value_schema.enum,
            require_opening_quote=True,
        )
    elif value_schema.type == "object":
        return ObjectParsingState(value_schema, parsing_state)
    elif value_schema.type == None and value_schema.ref:
        value_class_name = value_schema.ref.split('/')[-1]
        extras = parsing_state.model_class.extras
        # Pydantic V1 and V2 have different names for the definitions field
        if 'definitions' in extras:
            definitions = extras['definitions']
        elif '$defs' in extras:
            definitions = extras['$defs']
        else:
            raise ValueError("No definitions found in schema")
        class_dict = definitions[value_class_name]
        value_schema = JsonSchemaObject(**class_dict)
        return get_parser(parsing_state, value_schema, ending_characters)
    elif value_schema.enum:
        is_numeric = all(isinstance(i, (int, float)) for i in value_schema.enum)
        is_string = all(isinstance(i, (str)) for i in value_schema.enum)
        if is_string:
            return StringParsingState(
            parsing_state,
            ending_characters,
            value_schema.enum,
            require_opening_quote=True,
        )
        elif is_numeric:
            return StringParsingState(
                parsing_state,
                ending_characters,
                [str(i) for i in value_schema.enum],
                require_opening_quote=False,
                require_closing_quote=False,
            )
        else:
            raise Exception("Unsupported enum type " + str(value_schema.enum))
    elif value_schema.type == "integer":
        return NumberParsingState(parsing_state, ending_characters, False)
    elif value_schema.type == "boolean":
        return StringParsingState(
            parsing_state,
            ending_characters,
            ["true", "false"],
            require_opening_quote=False,
            require_closing_quote=False,
        )
    elif value_schema.type == "number":
        return NumberParsingState(parsing_state, ending_characters, True)
    elif value_schema.type == "array":
        return ListParsingState(parsing_state, ending_characters, value_schema.items)
    else:
        raise Exception("Unsupported type " + str(value_schema.type))



class ObjectParsingState(BaseParsingState):
    schema_object: JsonSchemaObject
    current_stage: ObjectParsingStage
    existing_keys: List[str]
    current_key: str
    current_key_parser: Any  #  type: StringParsingState
    is_dictionary: bool

    def __init__(self, schema_object: JsonSchemaObject, root: JsonSchemaParser):
        super().__init__(root)
        self.schema_object = schema_object
        self.current_stage = ObjectParsingStage.START_OBJECT
        self.root = root
        self.existing_keys = []
        self.current_key = None
        self.current_key_parser = None
        # Javascript objects represent both classes and dictionaries, so we need to know which one we are parsing
        self.is_dictionary = self.schema_object.properties is None

    def add_character(self, new_character: str):
        if new_character.strip() == "":
            # In object scope, whitespaces can be ignored
            return
        if (
            self.current_stage == ObjectParsingStage.START_OBJECT
            and new_character == "{"
        ):
            self.current_stage = ObjectParsingStage.PARSING_KEY_OR_END
        elif self.current_stage == ObjectParsingStage.PARSING_KEY_OR_END:
            if new_character == "}":
                self.current_stage = ObjectParsingStage.END_OBJECT
                self.root.finish_parser()
            if new_character == '"':
                self.current_stage = ObjectParsingStage.PARSING_KEY_OR_END
                possible_keys = None
                if not self.is_dictionary:
                    possible_keys = list(self.schema_object.properties.keys())
                    possible_keys = list(
                        set(possible_keys).difference(self.existing_keys)
                    )
                ending_characters = ': '
                # We send require_opening_quote=True and then add_character('"') instead of require_opening_quote=False
                # Because there is a difference between "don't need a quote" and "received it before creating the parser"
                key_parser = StringParsingState(
                    self.root, ending_characters, possible_keys, require_opening_quote=True, require_closing_quote=True
                )
                key_parser.add_character('"')
                self.root.object_stack.append(key_parser)
                self.current_key_parser = key_parser
            if new_character == ":":
                self.current_stage = ObjectParsingStage.PARSING_VALUE
                self.current_key = self.current_key_parser.parsed_string
                self.existing_keys.append(self.current_key)
                if self.is_dictionary:
                    value_schema = self.schema_object.additionalProperties
                    can_continue = True
                    can_end = True
                else:
                    possible_keys = list(self.schema_object.properties.keys())
                    possible_keys = list(
                        set(possible_keys).difference(self.existing_keys)
                    )
                    value_schema = self.schema_object.properties[self.current_key]
                    can_continue = len(possible_keys) > 0
                    required_keys = self.schema_object.required or []
                    can_end = set(self.existing_keys).issuperset(required_keys)
                ending_characters = " "
                if can_continue:
                    ending_characters += ","
                if can_end:
                    ending_characters += "}"
                self.current_key_parser = get_parser(
                    self.root, value_schema, ending_characters
                )
                self.root.object_stack.append(self.current_key_parser)
                self.current_key_parser = None
        elif self.current_stage == ObjectParsingStage.PARSING_VALUE:
            # If we recieve a character during parsing value, it means that its the finishing character
            # of the value parser
            if new_character == '"':
                self.current_stage = ObjectParsingStage.PARSING_SEPARATOR_OR_END
            elif new_character == ",":
                self.current_stage = ObjectParsingStage.PARSING_KEY_OR_END
            elif new_character == "}":
                self.current_stage = ObjectParsingStage.END_OBJECT
                self.root.finish_parser()
        elif self.current_stage == ObjectParsingStage.PARSING_SEPARATOR_OR_END:
            if new_character == ",":
                self.current_stage = ObjectParsingStage.PARSING_KEY_OR_END
            elif new_character == "}":
                self.current_stage = ObjectParsingStage.END_OBJECT
                self.root.finish_parser()

    def get_allowed_characters(self) -> str:
        possible_keys = (
            list(self.schema_object.properties.keys())
            if not self.is_dictionary
            else None
        )
        required_keys = self.schema_object.required or []
        can_end = set(self.existing_keys).issuperset(required_keys)
        can_parse_key = self.is_dictionary or set(possible_keys).issuperset(
            self.existing_keys
        )

        possible_characters = [' ']
        if self.current_stage == ObjectParsingStage.START_OBJECT:
            possible_characters.append('{')
        elif self.current_stage == ObjectParsingStage.PARSING_KEY_OR_END:
            if can_end:
                possible_characters.append('}')
            if can_parse_key:
                possible_characters.append('"')
        elif self.current_stage == ObjectParsingStage.PARSING_VALUE:
            # Sometimes the value parser considers finishing, so it needs to know which continuations are possible
            if can_end:
                possible_characters.append('}')
            if can_parse_key:
                possible_characters.append(',')
        elif self.current_stage == ObjectParsingStage.PARSING_SEPARATOR_OR_END:
            if can_end:
                possible_characters.append('}')
            if can_parse_key:
                possible_characters.append(',')
        return "".join(possible_characters)


class StringParsingStage:
    START_TOKEN = "StartToken"
    PARSING_STRING = "ParsingString"
    END_TOKEN = "EndToken"


class PrimitiveParsingState(BaseParsingState):
    def __init__(self, root: JsonSchemaParser, ending_characters: str):
        super().__init__(root)
        self.stage = StringParsingStage.START_TOKEN
        self.parsed_string = ""
        self.ending_characters = ending_characters

    def add_character(self, new_character: str):
        if self.can_end() and new_character in self.ending_characters:
            self.root.finish_parser(new_character)
        else:
            self.parsed_string += new_character

    def get_allowed_characters(self) -> str:
        allowed_characters = self._get_allowed_primitive_characters()
        if self.can_end():
            allowed_characters += self.ending_characters
        return allowed_characters

    def _get_allowed_primitive_characters(self) -> str:
        return ''

    def can_end(self) -> bool:
        return True


class NumberParsingState(PrimitiveParsingState):
    def __init__(
        self,
        root: JsonSchemaParser,
        ending_characters: str,
        allow_floating_point: bool,
    ):
        super().__init__(root, ending_characters)
        self.allow_floating_point = allow_floating_point
        self.seen_decimal_point = False

    def add_character(self, new_character: str):
        if new_character == " ":
            return
        super().add_character(new_character)
        if new_character == ".":
            self.seen_decimal_point = True

    def _get_allowed_primitive_characters(self) -> str:
        allowed_characters = "0123456789"
        if not self.parsed_string:
            allowed_characters += "- "
        if self.allow_floating_point and not self.seen_decimal_point:
            allowed_characters += "."
        return allowed_characters

    def can_end(self) -> bool:
        return bool(self.parsed_string) and self.parsed_string[-1] in "0123456789"


class StringParsingState(PrimitiveParsingState):
    allowed_strings: List[str]
    parsed_string: str
    seen_closing_quote: bool
    seen_opening_quote: bool

    def __init__(
        self,
        root: JsonSchemaParser,
        ending_characters: str,
        allowed_strings: List[str],
        require_opening_quote: bool,
        require_closing_quote: bool = True,
    ):
        super().__init__(root, ending_characters)
        self.allowed_strings = allowed_strings
        self.seen_closing_quote = False
        self.seen_opening_quote = not require_opening_quote
        self.require_closing_quote = require_closing_quote
        self.require_opening_quote = require_opening_quote

    def add_character(self, new_character: str):
        if not self.parsed_string and new_character == ' ':
            return
        super().add_character(new_character)
        if new_character == '"':
            if not self.seen_opening_quote:
                self.seen_opening_quote = True
                self.parsed_string = ""
            else:
                self.seen_closing_quote = True
                self.parsed_string = self.parsed_string[:-1]

    def _get_allowed_primitive_characters(self) -> str:
        if not self.seen_opening_quote:
            return '" '
        if self.seen_closing_quote:
            return ''
        if self.allowed_strings:
            allowed_continuations = [
                s[len(self.parsed_string) :]
                for s in self.allowed_strings
                if s.startswith(self.parsed_string)
            ]
            allowed_next_characters = [allowed_continuation[0] for allowed_continuation in allowed_continuations if len(allowed_continuation) > 0]
            allowed_next_characters = list(set(allowed_next_characters))
            if self.parsed_string in self.allowed_strings and self.require_closing_quote:
                allowed_next_characters.append('"')
            if (not self.parsed_string) and (not self.seen_opening_quote or not self.require_opening_quote):
                allowed_next_characters.append(' ')
            return "".join(allowed_next_characters)
        else:
            return COMPLETE_ALPHABET

    def can_end(self) -> bool:
        if self.require_closing_quote:
            return self.seen_closing_quote
        else:
            if self.allowed_strings:
                return self.parsed_string in self.allowed_strings
            else:
                return bool(self.parsed_string)


class ListParsingState(PrimitiveParsingState):
    list_member_type: JsonSchemaObject
    seen_list_opener: bool = False
    seen_list_closer: bool = False

    def __init__(
        self,
        root: JsonSchemaParser,
        ending_characters: str,
        list_member_type: JsonSchemaObject,
    ):
        super().__init__(root, ending_characters)
        self.list_member_type = list_member_type

    def add_character(self, new_character: str):
        if self.seen_list_closer:
            super().add_character(new_character)
        if new_character == "[":
            self.seen_list_opener = True
            self.root.object_stack.append(
                get_parser(
                    self.root,
                    self.list_member_type,
                    "],",
                )
            )
        elif new_character == "]":
            self.seen_list_closer = True
        elif new_character == ",":
            if not self.seen_list_closer:
                self.root.object_stack.append(
                    get_parser(
                        self.root,
                        self.list_member_type,
                        "],",
                    )
                )

    def _get_allowed_primitive_characters(self) -> str:
        if not self.seen_list_opener:
            return "[ "
        elif not self.seen_list_closer:
            return "], "
        else:
            # The parent function will take care of allowing the ending tokens.
            return ""

    def can_end(self) -> bool:
        return self.seen_list_closer
