# https://github.com/koxudaxi/datamodel-code-generator/blob/master/datamodel_code_generator/util.py
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

from typing import TYPE_CHECKING, Any, Callable, TypeVar
from enum import Enum, auto

import pydantic
from packaging import version
from pydantic import BaseModel as _BaseModel

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    TypeVar,
    Union,
)

PYDANTIC_VERSION = version.parse(
    pydantic.VERSION if isinstance(pydantic.VERSION, str) else str(pydantic.VERSION)
)

PYDANTIC_V2: bool = PYDANTIC_VERSION >= version.parse('2.0b3')

if PYDANTIC_V2:
    from pydantic import GetCoreSchemaHandler
    from pydantic_core import core_schema

if TYPE_CHECKING:
    cached_property = property
    from yaml import SafeLoader

    Protocol = object
    runtime_checkable: Callable[..., Any]

    from typing_extensions import Literal
else:
    try:
        from typing import Protocol
    except ImportError:
        from typing_extensions import Protocol  # noqa
    try:
        from typing import runtime_checkable
    except ImportError:
        from typing_extensions import runtime_checkable  # noqa
    try:
        from yaml import CSafeLoader as SafeLoader
    except ImportError:  # pragma: no cover
        from yaml import SafeLoader

    try:
        from functools import cached_property
    except ImportError:
        _NOT_FOUND = object()

        class cached_property:
            def __init__(self, func: Callable) -> None:
                self.func: Callable = func
                self.__doc__: Any = func.__doc__

            def __get__(self, instance: Any, owner: Any = None) -> Any:
                value = instance.__dict__.get(self.func.__name__, _NOT_FOUND)
                if value is _NOT_FOUND:  # pragma: no cover
                    value = instance.__dict__[self.func.__name__] = self.func(instance)
                return value


SafeLoader.yaml_constructors[
    'tag:yaml.org,2002:timestamp'
] = SafeLoader.yaml_constructors['tag:yaml.org,2002:str']


Model = TypeVar('Model', bound=_BaseModel)


def model_validator(
    mode: Literal['before', 'after'] = 'after',
) -> Callable[[Callable[[Model, Any], Any]], Callable[[Model, Any], Any]]:
    def inner(method: Callable[[Model, Any], Any]) -> Callable[[Model, Any], Any]:
        if PYDANTIC_V2:
            from pydantic import model_validator as model_validator_v2

            return model_validator_v2(mode=mode)(method)  # type: ignore
        else:
            from pydantic import root_validator

            return root_validator(method, pre=mode == 'before')  # type: ignore

    return inner


def field_validator(
    field_name: str,
    *fields: str,
    mode: Literal['before', 'after'] = 'after',
) -> Callable[[Any], Callable[[Model, Any], Any]]:
    def inner(method: Callable[[Model, Any], Any]) -> Callable[[Model, Any], Any]:
        if PYDANTIC_V2:
            from pydantic import field_validator as field_validator_v2

            return field_validator_v2(field_name, *fields, mode=mode)(method)  # type: ignore
        else:
            from pydantic import validator

            return validator(field_name, *fields, pre=mode == 'before')(method)  # type: ignore

    return inner


if PYDANTIC_V2:
    from pydantic import ConfigDict as ConfigDict
else:
    ConfigDict = dict  # type: ignore


class BaseModel(_BaseModel):
    if PYDANTIC_V2:
        model_config = ConfigDict(strict=False)


def is_url(ref: str) -> bool:
    return ref.startswith(('https://', 'http://'))


class Types(Enum):
    integer = auto()
    int32 = auto()
    int64 = auto()
    number = auto()
    float = auto()
    double = auto()
    decimal = auto()
    time = auto()
    string = auto()
    byte = auto()
    binary = auto()
    date = auto()
    date_time = auto()
    password = auto()
    email = auto()
    uuid = auto()
    uuid1 = auto()
    uuid2 = auto()
    uuid3 = auto()
    uuid4 = auto()
    uuid5 = auto()
    uri = auto()
    hostname = auto()
    ipv4 = auto()
    ipv4_network = auto()
    ipv6 = auto()
    ipv6_network = auto()
    boolean = auto()
    object = auto()
    null = auto()
    array = auto()
    any = auto()

class UnionIntFloat:
    def __init__(self, value: Union[int, float]) -> None:
        self.value: Union[int, float] = value

    def __int__(self) -> int:
        return int(self.value)

    def __float__(self) -> float:
        return float(self.value)

    def __str__(self) -> str:
        return str(self.value)

    @classmethod
    def __get_validators__(cls) -> Iterator[Callable[[Any], Any]]:
        yield cls.validate

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        from_int_schema = core_schema.chain_schema(
            [
                core_schema.union_schema(
                    [core_schema.int_schema(), core_schema.float_schema()]
                ),
                core_schema.no_info_plain_validator_function(cls.validate),
            ]
        )

        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_plain_validator_function(cls.validate),
            python_schema=core_schema.union_schema(
                [
                    # check if it's an instance first before doing any further work
                    core_schema.is_instance_schema(UnionIntFloat),
                    from_int_schema,
                ]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda instance: instance.value
            ),
        )

    @classmethod
    def validate(cls, v: Any) -> UnionIntFloat:
        if isinstance(v, UnionIntFloat):
            return v
        elif not isinstance(v, (int, float)):  # pragma: no cover
            raise TypeError(f'{v} is not int or float')
        return cls(v)
