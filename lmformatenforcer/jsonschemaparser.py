from copy import deepcopy
import enum
import sys
from typing import Dict, Hashable, List, Optional, Union, cast


from .external.jsonschemaobject import JsonSchemaObject, json_schema_data_formats
from .exceptions import LMFormatEnforcerException
from .characterlevelparser import CharacterLevelParser, CharacterLevelParserConfig, ForceStopParser, SequenceParser, StringParser, UnionParser
from .consts import BACKSLASH, BACKSLASH_ESCAPING_CHARACTERS, WHITESPACE_CHARACTERS
from .regexparser import RegexParser

# No need to include the 'integer' option in the anyOf, as it is a subset of 'number'
_ANY_JSON_SCHEMA_DICT = {'anyOf': [{'type': type} for type in json_schema_data_formats.keys() if type != 'integer']}

class JsonSchemaParser(CharacterLevelParser):
    ANY_JSON_OBJECT_SCHEMA: JsonSchemaObject = JsonSchemaObject(**_ANY_JSON_SCHEMA_DICT)
    class _Context:
        model_class: JsonSchemaObject
        # We store the active parser in the context, so that if a node adds to the stack, it knows
        # to which parser's stack to add.
        active_parser: "JsonSchemaParser"
        alphabet_without_quotes: str
        regex_parser_cache: Dict[str, RegexParser] = {}

    object_stack: List[CharacterLevelParser]
    context: _Context
    num_consecutive_whitespaces: int
    last_parsed_string: str  # Slight hack to allow communicating the parsed key to the object parser
    last_non_whitespace_character: str  # Slight hack to allow list parser to know if there is an item on top

    def __init__(self, 
                 json_schema: Union[dict, _Context, None], 
                 config: Optional[CharacterLevelParserConfig] = None, 
                 existing_stack: Optional[List[CharacterLevelParser]] = None, 
                 num_consecutive_whitespaces: int = 0):
        """Create a CharacterLevelParser for parsing JSON.
        :param json_schema: The json schema to parse. Can be a dict of a JSON schema, or None if any json output is allowed."""
        super().__init__(config)
        if isinstance(json_schema, JsonSchemaParser._Context):
            self.context = json_schema
        else:
            self.context = JsonSchemaParser._Context()
            json_schema = json_schema or _ANY_JSON_SCHEMA_DICT
            self.context.model_class = JsonSchemaObject(**json_schema)
            self.context.active_parser = self
            self.context.alphabet_without_quotes = self.config.alphabet.replace('"', '')
        
        self.num_consecutive_whitespaces = num_consecutive_whitespaces
        if existing_stack is None:
            self.object_stack = [get_parser(self, self.context.model_class)]
        else:
            self.object_stack = existing_stack
        self.last_parsed_string = ""
        self.last_non_whitespace_character = ""
    
    def add_character(self, new_character: str) -> CharacterLevelParser:
        self.context.active_parser = self
        # Assumption: The top-most parser that can accept the character is the one that should accept it.
        # This is different from the SequenceParser, in which we need to split (union) into all options.
        receiving_idx = len(self.object_stack) - 1
        last_parsed_string = self.last_parsed_string
        while receiving_idx >= 0 and new_character not in self.object_stack[receiving_idx].get_allowed_characters():
            finished_receiver = self.object_stack[receiving_idx]
            if isinstance(finished_receiver, StringParsingState):
                last_parsed_string = finished_receiver.parsed_string
            receiving_idx -= 1
        
        updated_stack = self.object_stack[:receiving_idx + 1]
        updated_parser = JsonSchemaParser(self.context, self.config, updated_stack, self.num_consecutive_whitespaces)
        updated_parser.context.active_parser = updated_parser
        updated_parser.last_parsed_string = last_parsed_string
        if receiving_idx >= 0:
            updated_parser.object_stack[receiving_idx] = updated_parser.object_stack[receiving_idx].add_character(new_character)
        if new_character in WHITESPACE_CHARACTERS:
            updated_parser.num_consecutive_whitespaces += 1
            updated_parser.last_non_whitespace_character = self.last_non_whitespace_character
        else:
            updated_parser.num_consecutive_whitespaces = 0
            updated_parser.last_non_whitespace_character = new_character

        if updated_parser.object_stack and isinstance(updated_parser.object_stack[-1], UnionParser) and \
            any(isinstance(parser, (ObjectParsingState, ListParsingState)) for parser in updated_parser.object_stack[-1].parsers):
            # If the top parser is a union parser with "advanced" (=parsers that modify the object stack) parsers inside, 
            # we need to split the top level parser into the different options,
            # As each "fork" can live with a different object stack, and we need to make sure they have their own ones.
            option_json_schema_parsers = []
            for option_parser in updated_parser.object_stack[-1].parsers:
                option_stack = updated_parser.object_stack[:-1] + [option_parser]
                option_parser = JsonSchemaParser(self.context, self.config, option_stack, updated_parser.num_consecutive_whitespaces)
                option_parser.context.active_parser = option_parser
                option_parser.last_parsed_string = last_parsed_string
                option_parser.last_non_whitespace_character = updated_parser.last_non_whitespace_character
                option_json_schema_parsers.append(option_parser)
            return UnionParser(option_json_schema_parsers)

        # For some performance optimizations to work, we want to make sure we don't leave irrelevant
        # objects at the top of the stack, which we know will be passed over next timestep
        new_object_stack = updated_parser.object_stack
        while new_object_stack and new_object_stack[-1].can_end() and new_object_stack[-1].get_allowed_characters() == '':
            finished_receiver = new_object_stack[-1]
            if isinstance(finished_receiver, StringParsingState):
                updated_parser.last_parsed_string = finished_receiver.parsed_string
            del new_object_stack[-1]
            if new_object_stack:
                new_top_parser = new_object_stack[-1]
                if isinstance(new_top_parser, ListParsingState):
                    new_top_parser = new_top_parser._clone()
                    new_top_parser.num_items_seen += 1
                    new_object_stack[-1] = new_top_parser
                    

        return updated_parser

    def get_allowed_characters(self) -> str:
        self.context.active_parser = self
        
        allowed_character_strs = []
        for parser in reversed(self.object_stack):
            # Similar to SequenceParser, if the top object can end, we need to know to accept the next character of parser below, etc.
            allowed_character_strs.append(parser.get_allowed_characters())
            if not parser.can_end():
                break
        if len(allowed_character_strs) > 0:
            allowed_characters =  "".join(allowed_character_strs)
        else:
            # In certain cases, beam search / sample crashes when there are less legal 
            # continuation tokens than there are beams. Therefore, we allow whitespace 
            # characters when the object stack is empty (= we are done parsing)
            allowed_characters = WHITESPACE_CHARACTERS
        
        if self.num_consecutive_whitespaces >= self.config.max_consecutive_whitespaces:
            # print("Filtering whitespace characters")
            allowed_characters = "".join(c for c in allowed_characters if c not in WHITESPACE_CHARACTERS)
        return allowed_characters

    def can_end(self) -> bool:
        return all(parser.can_end() for parser in self.object_stack)

    def shortcut_key(self) -> Optional[Hashable]:
        if self.object_stack:
            current_parser = self.object_stack[-1]
            if isinstance(current_parser, StringParsingState):
                if not current_parser.allowed_strings and current_parser.seen_opening_quote and not current_parser.seen_closing_quote and not current_parser.regex_parser:
                    # Performance optimization: When we are parsing a string that is not from a list of allowed strings, most tokens
                    # are legal. The exploration can be more costly than the LM itself for large tokenizers (because this is pure python),
                    # so we signal that we are in a "freetext" mode, and reuse the allowed token list throughout the run.
                    cur_len = len(current_parser.parsed_string)
                    min_len = current_parser.min_length or 0
                    max_len = current_parser.max_length or sys.maxsize
                    assert min_len <= max_len, "Invalid schema for str: min length is larger than max length"
                    if cur_len < max_len:
                        return ('json_freetext', cur_len, min_len, max_len)
        return None


class BaseParsingState(CharacterLevelParser):
    def __init__(self, root: JsonSchemaParser):
        self.root = root


def _merge_object_schemas(base_schema: JsonSchemaObject, option_schema: JsonSchemaObject) -> JsonSchemaObject:
    base_schema_properties = base_schema.properties or {}
    for property_name, property_value in base_schema_properties.items():
        # We assume that if a property exists in both base and option, the option version will be
        # more specific, therefore we only take missing entries
        if property_name not in option_schema.properties:
            option_schema.properties[property_name] = property_value
    for required_property in base_schema.required:
        if required_property not in option_schema.required:
            option_schema.required.append(required_property)
    return option_schema


def get_parser(
    parsing_state: JsonSchemaParser,
    value_schema: JsonSchemaObject
) -> CharacterLevelParser:
    if value_schema is None:
        raise Exception("JsonSchemaParser: Value schema is None")
    if value_schema.anyOf:
        parsers = [get_parser(parsing_state, schema) for schema in value_schema.anyOf]
        return UnionParser(parsers)
    if value_schema.allOf:
        merged_schema = value_schema.allOf[0]
        for schema in value_schema.allOf[1:]:
            merged_schema = _merge_object_schemas(merged_schema, schema)
        return get_parser(parsing_state, merged_schema)
    if value_schema.extras and 'const' in value_schema.extras:
        allowed_value = value_schema.extras['const']
        is_string = type(allowed_value) == str
        return StringParsingState(parsing_state, 
                                  [allowed_value], 
                                  require_opening_quote=is_string, 
                                  require_closing_quote=is_string)
    if value_schema.type == "string":
        return StringParsingState(
            parsing_state,
            value_schema.enum,
            require_opening_quote=True,
            min_length=value_schema.minLength,
            max_length=value_schema.maxLength,
            pattern=value_schema.pattern,
        )
    if value_schema.oneOf:
        # We create a combined object schema for each option that includes the information from the parent
        # And then create a UnionParser based on the combined options
        merged_schemas = [_merge_object_schemas(value_schema, option_schema) for option_schema in value_schema.oneOf]
        object_parsing_options = [ObjectParsingState(merged_schema, parsing_state) for merged_schema in merged_schemas]
        return UnionParser(object_parsing_options)
    elif value_schema.type == "object":
        return ObjectParsingState(value_schema, parsing_state)
    elif value_schema.type == None and value_schema.ref:
        value_class_name = value_schema.ref.split('/')[-1]
        extras = parsing_state.context.model_class.extras
        # Pydantic V1 and V2 have different names for the definitions field
        if 'definitions' in extras:
            definitions = extras['definitions']
        elif '$defs' in extras:
            definitions = extras['$defs']
        else:
            raise ValueError("No definitions found in schema")
        class_dict = definitions[value_class_name]
        value_schema = JsonSchemaObject(**class_dict)
        return get_parser(parsing_state, value_schema)
    elif value_schema.enum:
        is_numeric = all(isinstance(i, (int, float)) for i in value_schema.enum)
        is_string = all(isinstance(i, (str)) for i in value_schema.enum)
        if is_string:
            return StringParsingState(
            parsing_state,
            value_schema.enum,
            require_opening_quote=True,
        )
        elif is_numeric:
            return StringParsingState(
                parsing_state,
                [str(i) for i in value_schema.enum],
                require_opening_quote=False,
                require_closing_quote=False,
            )
        else:
            raise Exception("Unsupported enum type " + str(value_schema.enum))
    elif value_schema.type == "integer":
        return NumberParsingState(parsing_state, False)
    elif value_schema.type == "boolean":
        return StringParsingState(
            parsing_state,
            ["true", "false"],
            require_opening_quote=False,
            require_closing_quote=False,
        )
    elif value_schema.type == "null":
        return StringParsingState(
            parsing_state,
            ["null"],
            require_opening_quote=False,
            require_closing_quote=False,
        )
    elif value_schema.type == "number":
        return NumberParsingState(parsing_state, True)
    elif value_schema.type == "array":
        item_schema = value_schema.items or JsonSchemaParser.ANY_JSON_OBJECT_SCHEMA
        return ListParsingState(parsing_state, item_schema, value_schema.minItems, value_schema.maxItems)
    elif isinstance(value_schema.type, list):
        parsers = [get_parser(parsing_state, JsonSchemaObject(type=schema)) for schema in value_schema.type]
        return UnionParser(parsers)
    else:
        raise Exception("Unsupported type " + str(value_schema.type))


class ObjectParsingStage(enum.Enum):
    START_OBJECT = "StartObject"
    PARSING_KEY_OR_END = "ParsingKey"
    PARSING_KEY_VALUE_SEPARATOR = "ParsingKeyValueSeparator"
    PARSING_VALUE = "ParsingValue"
    PARSING_SEPARATOR_OR_END = "ParsingSeparatorOrEnd"
    END_OBJECT = "EndObject"


class ObjectParsingState(BaseParsingState):
    schema_object: JsonSchemaObject
    current_stage: ObjectParsingStage
    existing_keys: List[str]
    current_key: Optional[str]
    is_dictionary: bool

    def __init__(self, schema_object: JsonSchemaObject, root: JsonSchemaParser):
        super().__init__(root)
        self.schema_object = schema_object
        self.current_stage = ObjectParsingStage.START_OBJECT
        self.root = root
        self.existing_keys = []
        self.current_key = None
        # Javascript objects represent both classes and dictionaries, so we need to know which one we are parsing
        self.is_dictionary = self.schema_object.properties is None

    def clone(self) -> 'ObjectParsingState':
        clone = ObjectParsingState(self.schema_object, self.root)
        clone.current_stage = self.current_stage
        clone.existing_keys = self.existing_keys[:]
        clone.current_key = self.current_key
        clone.is_dictionary = self.is_dictionary
        return clone

    def add_character(self, new_character: str) -> CharacterLevelParser:
        if new_character.strip() == "":
            # In object scope, whitespaces can be ignored
            return self
        self = self.clone()  # Immutability requirement
        if (
            self.current_stage == ObjectParsingStage.START_OBJECT
            and new_character == "{"
        ):
            self.current_stage = ObjectParsingStage.PARSING_KEY_OR_END
        elif self.current_stage == ObjectParsingStage.PARSING_KEY_OR_END:
            if new_character == "}":
                self.current_stage = ObjectParsingStage.END_OBJECT
            if new_character == '"':
                possible_keys = None
                if not self.is_dictionary:
                    required_keys = self.schema_object.required or []
                    next_required_key = next((key for key in required_keys if key not in self.existing_keys), None)
                    if self.root.config.force_json_field_order and next_required_key:
                        possible_keys = [next_required_key]
                    else:
                        possible_keys = list(self.schema_object.properties.keys())
                        possible_keys = list(
                            set(possible_keys).difference(self.existing_keys)
                        )
                # We send require_opening_quote=True and then add_character('"') instead of require_opening_quote=False
                # Because there is a difference between "don't need a quote" and "received it before creating the parser"
                key_parser = StringParsingState(
                    self.root, possible_keys, require_opening_quote=True, require_closing_quote=True
                )
                key_parser = key_parser.add_character('"')
                self.root.context.active_parser.object_stack.append(key_parser)
                self.current_stage = ObjectParsingStage.PARSING_KEY_VALUE_SEPARATOR
        elif self.current_stage == ObjectParsingStage.PARSING_KEY_VALUE_SEPARATOR:
            if new_character == ":":
                self.current_stage = ObjectParsingStage.PARSING_VALUE
                self.current_key = self.root.context.active_parser.last_parsed_string
                self.existing_keys.append(self.current_key)
                if self.is_dictionary:
                    if self.schema_object.additionalProperties:
                        value_schema = self.schema_object.additionalProperties
                    else:
                        value_schema = JsonSchemaParser.ANY_JSON_OBJECT_SCHEMA
                else:
                    value_schema = self.schema_object.properties[self.current_key]
                self.current_key_parser = get_parser(
                    self.root, value_schema
                )
                self.root.context.active_parser.object_stack.append(self.current_key_parser)
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
        elif self.current_stage == ObjectParsingStage.PARSING_SEPARATOR_OR_END:
            if new_character == ",":
                self.current_stage = ObjectParsingStage.PARSING_KEY_OR_END
            elif new_character == "}":
                self.current_stage = ObjectParsingStage.END_OBJECT
        return self

    def get_allowed_characters(self) -> str:
        possible_keys = (
            list(self.schema_object.properties.keys())
            if not self.is_dictionary
            else None
        )
        required_keys = self.schema_object.required or []
        can_end = set(self.existing_keys).issuperset(required_keys)
        can_parse_key = self.is_dictionary or set(possible_keys).difference(
            self.existing_keys
        )

        possible_characters = [c for c in WHITESPACE_CHARACTERS]
        if self.current_stage == ObjectParsingStage.START_OBJECT:
            possible_characters.append('{')
        elif self.current_stage == ObjectParsingStage.PARSING_KEY_OR_END:
            if can_end:
                possible_characters.append('}')
            if can_parse_key:
                possible_characters.append('"')
        elif self.current_stage == ObjectParsingStage.PARSING_KEY_VALUE_SEPARATOR:
            possible_characters.append(':')
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
    
    def can_end(self) -> bool:
        return self.current_stage == ObjectParsingStage.END_OBJECT


class StringParsingStage:
    START_TOKEN = "StartToken"
    PARSING_STRING = "ParsingString"
    END_TOKEN = "EndToken"


class PrimitiveParsingState(BaseParsingState):
    def __init__(self, root: JsonSchemaParser):
        super().__init__(root)
        self.stage = StringParsingStage.START_TOKEN
        self.parsed_string = ""

    def _clone(self) -> "PrimitiveParsingState":
        raise NotImplementedError()
    
    def add_character(self, new_character: str) -> "PrimitiveParsingState":
        new = self._clone()
        new.parsed_string += new_character
        return new

    def can_end(self) -> bool:
        return True


class NumberParsingState(PrimitiveParsingState):
    def __init__(
        self,
        root: JsonSchemaParser,
        allow_floating_point: bool,
    ):
        super().__init__(root)
        self.allow_floating_point = allow_floating_point
        self.seen_decimal_point = False
        self.seen_whitespace_after_digits = False
        self.seen_exponent = False
        self.seen_digit = False

    def _clone(self) -> "NumberParsingState":
        clone = NumberParsingState(self.root, self.allow_floating_point)
        clone.parsed_string = self.parsed_string
        clone.seen_decimal_point = self.seen_decimal_point
        clone.seen_whitespace_after_digits = self.seen_whitespace_after_digits
        clone.seen_exponent = self.seen_exponent
        clone.seen_digit = self.seen_digit
        return clone
    
    def add_character(self, new_character: str) -> CharacterLevelParser:
        if not self.parsed_string and new_character in WHITESPACE_CHARACTERS:
            return self
        self = cast(NumberParsingState, super().add_character(new_character))
        if new_character in WHITESPACE_CHARACTERS:
            if self.parsed_string:
                self.seen_whitespace_after_digits = True
            return self
        if new_character == ".":
            if not self.parsed_string or len(self.parsed_string) == 1:
                raise LMFormatEnforcerException("Numbers cannot start with a decimal point.")
            if self.seen_decimal_point:
                raise LMFormatEnforcerException("Numbers cannot contain more than two decimal points.")
            self.seen_decimal_point = True
        elif new_character in "eE":
            if self.seen_exponent or not self.seen_digit:
                raise LMFormatEnforcerException("Invalid number format")
            self.seen_exponent = True
        elif new_character.isdigit():
            self.seen_digit = True
        return self

    def get_allowed_characters(self) -> str:
        if self.seen_whitespace_after_digits:
            return WHITESPACE_CHARACTERS
        allowed_characters = "0123456789"
        if not self.parsed_string:
            allowed_characters += "-" + WHITESPACE_CHARACTERS
        if self.parsed_string and len(self.parsed_string) == 1 and self.parsed_string[0] == "0":
            allowed_characters = WHITESPACE_CHARACTERS
        if self.parsed_string and len(self.parsed_string) == 2 and self.parsed_string == "-0":
            allowed_characters = "." + WHITESPACE_CHARACTERS
        if self.parsed_string and self.parsed_string[-1] in "eE":
            allowed_characters += "-+"
        if self.seen_digit and not self.seen_exponent:
            allowed_characters += "eE"
        if self.allow_floating_point and not self.seen_decimal_point and self.seen_digit and not self.seen_exponent:
            allowed_characters += "."
        if self.parsed_string and self.parsed_string[-1].isdigit():
            allowed_characters += WHITESPACE_CHARACTERS
        return allowed_characters

    def can_end(self) -> bool:
        if self.seen_exponent and self.parsed_string[-1] in "eE+-":
            return False
        return bool(self.parsed_string) and (self.parsed_string[-1].isdigit() or self.seen_whitespace_after_digits)


class StringParsingState(PrimitiveParsingState):
    allowed_strings: List[str]
    parsed_string: str
    seen_closing_quote: bool
    seen_opening_quote: bool
    min_length: Optional[int]
    max_length: Optional[int]
    pattern: Optional[str]
    regex_parser: Optional[RegexParser]

    def __init__(
        self,
        root: JsonSchemaParser,
        allowed_strings: List[str],
        require_opening_quote: bool,
        require_closing_quote: bool = True,
        min_length: Optional[int]=None,
        max_length: Optional[int]=None,
        pattern: Optional[str]=None,
        regex_parser: Optional[RegexParser]=None,
    ):
        super().__init__(root)
        self.allowed_strings = allowed_strings
        self.seen_closing_quote = False
        self.seen_opening_quote = not require_opening_quote
        self.require_closing_quote = require_closing_quote
        self.require_opening_quote = require_opening_quote
        self.min_length = min_length
        self.max_length = max_length
        self.pattern = pattern
        if self.pattern and (self.min_length or self.max_length):
            raise LMFormatEnforcerException("String schema contains both a pattern and a min/max length, which is not currently supported")
        self.regex_parser = regex_parser
        if self.pattern and not regex_parser:
            if self.pattern not in self.root.context.regex_parser_cache:
                self.root.context.regex_parser_cache[self.pattern] = RegexParser(self.pattern, self.root.config)
            self.regex_parser = self.root.context.regex_parser_cache[self.pattern]


    def _clone(self) -> "StringParsingState":
        clone = StringParsingState(
            self.root,
            self.allowed_strings,
            self.require_opening_quote,
            self.require_closing_quote,
            self.min_length,
            self.max_length,
            self.pattern,
            self.regex_parser
        )
        clone.parsed_string = self.parsed_string
        clone.seen_closing_quote = self.seen_closing_quote
        clone.seen_opening_quote = self.seen_opening_quote
        return clone

    def add_character(self, new_character: str):
        if (not self.parsed_string or self.seen_closing_quote) and new_character in WHITESPACE_CHARACTERS:
            return self
        self = cast(StringParsingState, super().add_character(new_character))
        if new_character == '"':
            if not self.seen_opening_quote:
                self.seen_opening_quote = True
                self.parsed_string = ""
            else:
                self.seen_closing_quote = True
                self.parsed_string = self.parsed_string[:-1]
        if self.regex_parser and new_character != '"' and self.seen_opening_quote and not self.seen_closing_quote:
            self.regex_parser = self.regex_parser.add_character(new_character)
        if new_character == BACKSLASH:
            # After a backslack we immediately have the escaping character, and if its 'u', we have 4 hex digits
            escaping_character_parsers: List[CharacterLevelParser] = [StringParser(c) for c in BACKSLASH_ESCAPING_CHARACTERS]
            hex_digit_parser: CharacterLevelParser = UnionParser([StringParser(c) for c in "0123456789abcdefABCDEF"])
            unicode_components: List[CharacterLevelParser] = list([StringParser("u")] + [hex_digit_parser] * 4)
            unicode_escape_parser: CharacterLevelParser = SequenceParser(unicode_components)
            json_escaping_parser = UnionParser(escaping_character_parsers + [unicode_escape_parser])
            self.root.context.active_parser.object_stack.append(json_escaping_parser)
        return self

    def get_allowed_characters(self) -> str:
        if not self.seen_opening_quote:
            return '"' + WHITESPACE_CHARACTERS
        if self.seen_closing_quote:
            return WHITESPACE_CHARACTERS
        if self.regex_parser:
            regex_chars = self.regex_parser.get_allowed_characters()
            # We don't currently support regexes with quotes or escaping backslashes, so we remove them from the allowed characters
            regex_chars = regex_chars.replace('"', '').replace(BACKSLASH, '')
            if self.regex_parser.can_end():
                regex_chars += '"'
            return regex_chars
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
                allowed_next_characters.extend(WHITESPACE_CHARACTERS)
            return "".join(allowed_next_characters)
        else:
            if self.min_length is not None and len(self.parsed_string) < self.min_length:
                return self.root.context.alphabet_without_quotes + BACKSLASH
            if self.max_length is not None and len(self.parsed_string) >= self.max_length:
                return '"'
            return self.root.config.alphabet + BACKSLASH

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
    num_items_seen: int = 0

    def __init__(
        self,
        root: JsonSchemaParser,
        list_member_type: JsonSchemaObject,
        min_items: Optional[int],
        max_items: Optional[int],
    ):
        super().__init__(root)
        self.list_member_type = list_member_type
        self.min_items = min_items
        self.max_items = max_items
        default_max = root.config.max_json_array_length
        if self.max_items is None and default_max > 0 and (min_items is None or min_items < default_max):
            self.max_items = default_max

    def _clone(self) -> PrimitiveParsingState:
        new = ListParsingState(self.root, self.list_member_type, self.min_items, self.max_items)
        new.parsed_string = self.parsed_string
        new.num_items_seen = self.num_items_seen
        new.seen_list_opener = self.seen_list_opener
        new.seen_list_closer = self.seen_list_closer
        return new

    def add_character(self, new_character: str) -> "ListParsingState":
        self = cast(ListParsingState, super().add_character(new_character))
        if new_character == "[":
            self.seen_list_opener = True
            item_parser = get_parser(self.root, self.list_member_type)
            requires_items = self.min_items is not None and self.min_items > 0
            if requires_items:
                parser_to_push = item_parser
            else:
                # If we don't require items, we can also end immediately, the Union + ForceStopParser combination achieves this
                empty_list_parser = ForceStopParser(allow_whitespace=True)
                if isinstance(item_parser, UnionParser):
                    item_parser.parsers.append(empty_list_parser)
                    parser_to_push = item_parser
                else:
                    parser_to_push = UnionParser([item_parser, empty_list_parser])
            self.root.context.active_parser.object_stack.append(parser_to_push)
        elif new_character == "]":
            self.seen_list_closer = True
        elif new_character == ",":
            if not self.seen_list_closer:
                self.num_items_seen += 1
                
                self.root.context.active_parser.object_stack.append(
                    get_parser(
                        self.root,
                        self.list_member_type,
                    )
                )
        return self

    def get_allowed_characters(self) -> str:
        if not self.seen_list_opener:
            return "[" + WHITESPACE_CHARACTERS
        elif not self.seen_list_closer:
            return self.get_allowed_control_characters() + WHITESPACE_CHARACTERS
        else:
            return ""

    def can_end(self) -> bool:
        return self.seen_list_closer
    
    def get_allowed_control_characters(self):
        num_items = self.num_items_seen
        top_parser = self.root.context.active_parser.object_stack[-1]
        is_on_top = top_parser == self or isinstance(top_parser, UnionParser) and self in top_parser.parsers
        if (not is_on_top) and self.root.context.active_parser.last_non_whitespace_character != "[":
            # If there is an active parser above us, and the last character is not [, 
            # there is an active item parser on the stack that we did not count yet.
            num_items += 1
        control_characters = ""
        has_enough_items = self.min_items is None or num_items >= self.min_items
        can_add_another_item = self.max_items is None or num_items < self.max_items

        if num_items > 0 and can_add_another_item:
            control_characters += ","
        if has_enough_items:
            control_characters += "]"
        return control_characters
    
