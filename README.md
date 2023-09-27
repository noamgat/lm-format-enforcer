# lm-format-enforcer
Enforce the format of the output of a language model
Language models are able to generate text, but when requiring a precise output format, they do not always perform as instructed.
Various prompt engineering techniques have been introduced to improve the robustness of the generated text, but they are not always sufficient.
This project solves the issues by filtering the tokens that the language model is allowed to generate at every timestep, thus ensuring that the output format is respected, while minimizing the limitations on the language model.

## Installation
```pip install lm-format-enforcer```

## Simple example
```python
from pydantic import BaseModel
from lmformatenforcer.jsonschemaparser import JsonSchemaParser
from lmformatenforcer.transformerenforcer import generate_enforced

question = 'Please give me information about Michael Jordan. You MUST answer using the following json schema: '


class AnswerFormat(BaseModel):
    first_name: str
    last_name: str
    year_of_birth: int
    num_seasons_in_nba: int


question_with_schema = f'{question}{AnswerFormat.schema_json()}'
inputs = tokenizer([question_with_schema], return_tensors='pt', add_special_tokens=False, return_token_type_ids=False).to(device)

parser = JsonSchemaParser(AnswerFormat.schema())

# What we would usually call is this:
# result = model.generate(inputs=inputs, ...)
# Instead, call this: 
result = generate_enforced(model, tokenizer, parser, inputs=inputs)
print(result)
# {'first_name': 'Michael', 'last_name': 'Jordan', 'year_of_birth': 1963, 'num_seasons_in_nba': 15}
```

## How does it work?

The library works by combining a character level parser and a tokenizer prefix tree into a smart token filtering mechanism.

### Character Level Parser

A character level parser is a tree-like interface that parses a certain format character by character, and can also return the allowed next characters at any given moment.