COMPLETE_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{};:,./<>? `'\""
DEFAULT_MAX_CONSECUTIVE_WHITESPACES = 12
DEFAULT_FORCE_JSON_FIELD_ORDER = False
DEFAULT_MAX_JSON_ARRAY_LENGTH = 20
WHITESPACE_CHARACTERS = " \t\n\r"
BACKSLASH = "\\"
BACKSLASH_ESCAPING_CHARACTERS = '"\\/bfnrt'  # Characters allowed after an escaping backslash, except unicode
BACKSLACH_UNICODE_ESCAPE = "u"

CONFIG_ENV_VAR_MAX_CONSECUTIVE_WHITESPACES = 'LMFE_MAX_CONSECUTIVE_WHITESPACES'
"""Environment variable for externally controlling how many consective whitespaces the 
JsonSchemaParser will allow. Default: 12"""

CONFIG_ENV_VAR_STRICT_JSON_FIELD_ORDER = 'LMFE_STRICT_JSON_FIELD_ORDER'
"""Environment variable for externally controlling whether the JsonSchemaParser will force 
fields to appear in the order of the 'required' field in the schema. Default: false"""

CONFIG_ENV_VAR_MAX_JSON_ARRAY_LENGTH = 'LMFE_MAX_JSON_ARRAY_LENGTH'
"""Environment variable for externally controlling what is the maximal JSON array length,
if not specified by the schema. Default: 20"""

CONFIG_ENV_VAR_DEFAULT_ALPHABET = "LMFE_DEFAULT_ALPHABET"
"""Environment variable for overriding the default alphabet used for 'any character'.
If you override, it should probably be a superset of consts.COMPLETE_ALPHABET.
Default: consts.COMPLETE_ALPHABET."""
