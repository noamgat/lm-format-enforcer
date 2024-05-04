COMPLETE_ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()_+-=[]{};:,./<>? `'\""
DEFAULT_MAX_CONSECUTIVE_WHITESPACES = 12
DEFAULT_FORCE_JSON_FIELD_ORDER = False
WHITESPACE_CHARACTERS = " \t\n\r"
BACKSLASH = "\\"
BACKSLASH_ESCAPING_CHARACTERS = '"\\/bfnrt'  # Characters allowed after an escaping backslash, except unicode
BACKSLACH_UNICODE_ESCAPE = "u"

CONFIG_ENV_VAR_MAX_CONSECUTIVE_WHITESPACES = 'LMFE_MAX_CONSECUTIVE_WHITESPACES'
"""Environment variable for externally controlling how many consective whitespaces the 
JsonSchemaParser will allow. Default: 12"""

CONFIG_ENV_VAR_LMFE_FORCE_JSON_FIELD_ORDER = 'LMFE_FORCE_JSON_FIELD_ORDER'
"""Environment variable for externally controlling whether the JsonSchemaParser will force 
fields to appear in the order of the 'required' field in the schema. Default: false"""