# https://github.com/koxudaxi/datamodel-code-generator/blob/master/datamodel_code_generator/parser/jsonschema.py
# MIT License

# Copyright (c) 2019 Koudai Aono

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
from __future__ import annotations

import enum as _enum
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Optional,
    Set,
    Union,
)
from warnings import warn


from pydantic import (
    Field,
)

from .jsonschemaobjectutil import (
    PYDANTIC_V2,
    BaseModel,
    cached_property,
    field_validator,
    model_validator,
    is_url,
    Types,
    UnionIntFloat
)

if PYDANTIC_V2:
    from pydantic import ConfigDict


def get_model_by_path(
    schema: Union[Dict[str, Any], List[Any]], keys: Union[List[str], List[int]]
) -> Dict[Any, Any]:
    model: Union[Dict[Any, Any], List[Any]]
    if not keys:
        model = schema
    elif len(keys) == 1:
        if isinstance(schema, dict):
            model = schema.get(keys[0], {})  # type: ignore
        else:  # pragma: no cover
            model = schema[int(keys[0])]
    elif isinstance(schema, dict):
        model = get_model_by_path(schema[keys[0]], keys[1:])  # type: ignore
    else:
        model = get_model_by_path(schema[int(keys[0])], keys[1:])
    if isinstance(model, dict):
        return model
    raise NotImplementedError(  # pragma: no cover
        f'Does not support json pointer to array. schema={schema}, key={keys}'
    )


json_schema_data_formats: Dict[str, Dict[str, Types]] = {
    'integer': {
        'int32': Types.int32,
        'int64': Types.int64,
        'default': Types.integer,
        'date-time': Types.date_time,
        'unix-time': Types.int64,
    },
    'number': {
        'float': Types.float,
        'double': Types.double,
        'decimal': Types.decimal,
        'date-time': Types.date_time,
        'time': Types.time,
        'default': Types.number,
    },
    'string': {
        'default': Types.string,
        'byte': Types.byte,  # base64 encoded string
        'binary': Types.binary,
        'date': Types.date,
        'date-time': Types.date_time,
        'time': Types.time,
        'password': Types.password,
        'email': Types.email,
        'idn-email': Types.email,
        'uuid': Types.uuid,
        'uuid1': Types.uuid1,
        'uuid2': Types.uuid2,
        'uuid3': Types.uuid3,
        'uuid4': Types.uuid4,
        'uuid5': Types.uuid5,
        'uri': Types.uri,
        'uri-reference': Types.string,
        'hostname': Types.hostname,
        'ipv4': Types.ipv4,
        'ipv4-network': Types.ipv4_network,
        'ipv6': Types.ipv6,
        'ipv6-network': Types.ipv6_network,
        'decimal': Types.decimal,
        'integer': Types.integer,
    },
    'boolean': {'default': Types.boolean},
    'object': {'default': Types.object},
    'null': {'default': Types.null},
    'array': {'default': Types.array},
}


class JSONReference(_enum.Enum):
    LOCAL = 'LOCAL'
    REMOTE = 'REMOTE'
    URL = 'URL'


class Discriminator(BaseModel):
    propertyName: str
    mapping: Optional[Dict[str, str]] = None


class JsonSchemaObject(BaseModel):
    if not TYPE_CHECKING:
        if PYDANTIC_V2:

            @classmethod
            def get_fields(cls) -> Dict[str, Any]:
                return cls.model_fields

        else:

            @classmethod
            def get_fields(cls) -> Dict[str, Any]:
                return cls.__fields__

            @classmethod
            def model_rebuild(cls) -> None:
                cls.update_forward_refs()

    __constraint_fields__: Set[str] = {
        'exclusiveMinimum',
        'minimum',
        'exclusiveMaximum',
        'maximum',
        'multipleOf',
        'minItems',
        'maxItems',
        'minLength',
        'maxLength',
        'pattern',
        'uniqueItems',
    }
    __extra_key__: str = 'extras'

    @model_validator(mode='before')
    def validate_exclusive_maximum_and_exclusive_minimum(
        cls, values: Dict[str, Any]
    ) -> Any:
        exclusive_maximum: Union[float, bool, None] = values.get('exclusiveMaximum')
        exclusive_minimum: Union[float, bool, None] = values.get('exclusiveMinimum')

        if exclusive_maximum is True:
            values['exclusiveMaximum'] = values['maximum']
            del values['maximum']
        elif exclusive_maximum is False:
            del values['exclusiveMaximum']
        if exclusive_minimum is True:
            values['exclusiveMinimum'] = values['minimum']
            del values['minimum']
        elif exclusive_minimum is False:
            del values['exclusiveMinimum']
        return values

    @field_validator('ref')
    def validate_ref(cls, value: Any) -> Any:
        if isinstance(value, str) and '#' in value:
            if value.endswith('#/'):
                return value[:-1]
            elif '#/' in value or value[0] == '#' or value[-1] == '#':
                return value
            return value.replace('#', '#/')
        return value

    items: Union[List[JsonSchemaObject], JsonSchemaObject, bool, None] = None
    uniqueItems: Optional[bool] = None
    type: Union[str, List[str], None] = None
    format: Optional[str] = None
    pattern: Optional[str] = None
    minLength: Optional[int] = None
    maxLength: Optional[int] = None
    minimum: Optional[UnionIntFloat] = None
    maximum: Optional[UnionIntFloat] = None
    minItems: Optional[int] = None
    maxItems: Optional[int] = None
    multipleOf: Optional[float] = None
    exclusiveMaximum: Union[float, bool, None] = None
    exclusiveMinimum: Union[float, bool, None] = None
    additionalProperties: Union[JsonSchemaObject, bool, None] = None
    patternProperties: Optional[Dict[str, JsonSchemaObject]] = None
    oneOf: List[JsonSchemaObject] = []
    anyOf: List[JsonSchemaObject] = []
    allOf: List[JsonSchemaObject] = []
    enum: List[Any] = []
    writeOnly: Optional[bool] = None
    properties: Optional[Dict[str, Union[JsonSchemaObject, bool]]] = None
    required: List[str] = []
    ref: Optional[str] = Field(default=None, alias='$ref')
    nullable: Optional[bool] = False
    x_enum_varnames: List[str] = Field(default=[], alias='x-enum-varnames')
    description: Optional[str] = None
    title: Optional[str] = None
    example: Any = None
    examples: Any = None
    default: Any = None
    id: Optional[str] = Field(default=None, alias='$id')
    custom_type_path: Optional[str] = Field(default=None, alias='customTypePath')
    custom_base_path: Optional[str] = Field(default=None, alias='customBasePath')
    extras: Dict[str, Any] = Field(alias=__extra_key__, default_factory=dict)
    discriminator: Union[Discriminator, str, None] = None
    if PYDANTIC_V2:
        model_config = ConfigDict(
            arbitrary_types_allowed=True,
            ignored_types=(cached_property,),
        )
    else:

        class Config:
            arbitrary_types_allowed = True
            keep_untouched = (cached_property,)
            smart_casts = True

    if not TYPE_CHECKING:

        def __init__(self, **data: Any) -> None:
            super().__init__(**data)
            self.extras = {k: v for k, v in data.items() if k not in EXCLUDE_FIELD_KEYS}

    @cached_property
    def is_object(self) -> bool:
        return (
            self.properties is not None
            or self.type == 'object'
            and not self.allOf
            and not self.oneOf
            and not self.anyOf
            and not self.ref
        )

    @cached_property
    def is_array(self) -> bool:
        return self.items is not None or self.type == 'array'

    @cached_property
    def ref_object_name(self) -> str:  # pragma: no cover
        return self.ref.rsplit('/', 1)[-1]  # type: ignore

    @field_validator('items', mode='before')
    def validate_items(cls, values: Any) -> Any:
        # this condition expects empty dict
        return values or None

    @cached_property
    def has_default(self) -> bool:
        return 'default' in self.__fields_set__ or 'default_factory' in self.extras

    @cached_property
    def has_constraint(self) -> bool:
        return bool(self.__constraint_fields__ & self.__fields_set__)

    @cached_property
    def ref_type(self) -> Optional[JSONReference]:
        if self.ref:
            return get_ref_type(self.ref)
        return None  # pragma: no cover

    @cached_property
    def type_has_null(self) -> bool:
        return isinstance(self.type, list) and 'null' in self.type


@lru_cache()
def get_ref_type(ref: str) -> JSONReference:
    if ref[0] == '#':
        return JSONReference.LOCAL
    elif is_url(ref):
        return JSONReference.URL
    return JSONReference.REMOTE


def _get_type(type_: str, format__: Optional[str] = None) -> Types:
    if type_ not in json_schema_data_formats:
        return Types.any
    data_formats: Optional[Types] = json_schema_data_formats[type_].get(
        'default' if format__ is None else format__
    )
    if data_formats is not None:
        return data_formats

    warn(
        'format of {!r} not understood for {!r} - using default'
        ''.format(format__, type_)
    )
    return json_schema_data_formats[type_]['default']


JsonSchemaObject.model_rebuild()

DEFAULT_FIELD_KEYS: Set[str] = {
    'example',
    'examples',
    'description',
    'discriminator',
    'title',
    'const',
    'default_factory',
}

EXCLUDE_FIELD_KEYS = (set(JsonSchemaObject.get_fields()) - DEFAULT_FIELD_KEYS) | {
    '$id',
    '$ref',
    JsonSchemaObject.__extra_key__,
}
