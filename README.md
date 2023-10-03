# lm-format-enforcer
**Enforce the output format (JSON Schema, Regex etc) of a language model**

<a target="_blank" href="https://colab.research.google.com/github/noamgat/lm-format-enforcer/blob/main/samples/colab_llama2_enforcer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


![Solution at a glance](https://raw.githubusercontent.com/noamgat/lm-format-enforcer/main/docs/Intro.drawio.svg?sanitize=true)


Language models are able to generate text, but when requiring a precise output format, they do not always perform as instructed.
Various prompt engineering techniques have been introduced to improve the robustness of the generated text, but they are not always sufficient.
This project solves the issues by filtering the tokens that the language model is allowed to generate at every timestep, thus ensuring that the output format is respected, while minimizing the limitations on the language model.

## Installation
```pip install lm-format-enforcer```

## Simple example
```python
from pydantic import BaseModel
from lmformatenforcer import JsonSchemaParser, generate_enforced

class AnswerFormat(BaseModel):
    first_name: str
    last_name: str
    year_of_birth: int
    num_seasons_in_nba: int

question = f'Please give me information about Michael Jordan. You MUST answer using the following json schema: {AnswerFormat.schema_json()}'
parser = JsonSchemaParser(AnswerFormat.schema())

# Call generate_enforced(model, tokenizer, parser, ...) instead of model.generate(...):
inputs = tokenizer([question], return_tensors='pt', add_special_tokens=False, return_token_type_ids=False).to(device)
result = generate_enforced(model, tokenizer, parser, inputs=inputs)
print(result)
# {'first_name': 'Michael', 'last_name': 'Jordan', 'year_of_birth': 1963, 'num_seasons_in_nba': 15}
```
## Capabilities / Advantages

- Works with any language model and tokenizer (currently works with transformers, can be adapted into any python language model framework)
- Supports both JSON Schema (strong) and Regular Expression (limited) formats
- Supports both required and optional fields in JSON schemas
- Supports nested fields, arrays and dictionaries in JSON schemas
- Gives the language model freedom to control whitespacing and field ordering in JSON schemas, reducing hallucinations

## Detailed example

We created a Google Colab Notebook which contains a full example of how to use this library to enforce the output format of llama2, including interpreting the intermediate results. The notebook can run on a free GPU-backed runtime in Colab.

<a target="_blank" href="https://colab.research.google.com/github/noamgat/lm-format-enforcer/blob/main/samples/colab_llama2_enforcer.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

You can also [view the notebook in GitHub](https://github.com/noamgat/lm-format-enforcer/blob/main/samples/colab_llama2_enforcer.ipynb).
## How does it work?

The library works by combining a character level parser and a tokenizer prefix tree into a smart token filtering mechanism.

![An example of the character level parser and tokenizer prefix tree in a certain timestep](https://raw.githubusercontent.com/noamgat/lm-format-enforcer/main/docs/Trees.drawio.svg?sanitize=true)

### Character Level Parser

Parsing a string into any kind of formatter can be looked at as an implicit tree structure - at any moment in the parsing process, there is a set of allowed next characters, and if any of them are selected, there is a new set of allowed next characters, and so on.

```CharacterLevelParser``` is an interface for parsing according to this implicit structure. ```add_character()``` and ```get_allowed_characters()``` can be seen as tree traversal methods.

There are several implementations of this interface:
- ```JsonSchemaParser``` - parses according to a json schema. 
- ```StringParser``` - forces an exact string (used mainly for diagnostics)
- ```RegexParser``` - parses according to a regular expression. Note that this cannot use the built in python regex and uses a manually implemented one (https://github.com/xysun/regex), so it has very limited capabilities.
### Tokenizer Prefix Tree

Given a tokenizer used by a certain language model, we can build a prefix tree of all the tokens that the language model can generate. This is done by generating all possible sequences of tokens, and adding them to the tree.
See ```TokenizerPrefixTree```

### Combining the two

Given a character level parser and a tokenizer prefix tree, we can elegantly and efficiently filter the tokens that the language model is allowed to generate at the next timestep:
We only traverse the characters that are in BOTH the character level parsing node and the tokenizer prefix tree node. This allows us to find all of the tokens (including complex subword tokens such as ```","``` which are critical in JSON parsing).
We do this recursively on both trees and return all of the allowed tokens. When the language model generates a token, we advance the character level parser according to the new characters, ready to filter the next timestep.


## Diagnostics - Will I always get good results?

Using this library guarantees that the output will match the format, but it does not guarantee that the output will be semantically correct. Forcing the language model to conform to a certain output may lead to increased hallucinations. Guiding the model via prompt engineering is still likely to improve results.

In order to help you understand the aggressiveness caused by the format enforcement, if you pass ```output_scores=True``` and ```return_dict_in_generate=True``` in the ```kwargs``` to ```generate_enforced()``` (these are existing optional parameters in the ```transformers``` library), you will also get a token-by-token dataframe showing which token was selected, its score, and what was the token that would have been chosen if the format enforcement was not applied. If you see that the format enforcer forced the language model to select tokens with very low weights, it is a likely contributor to the poor results. Try modifying the prompt to guide the language model to not force the format enforcer to be so aggressive.

Example using the regular expression format ```Michael Jordan was Born in (\d)+.```
```
   generated_token  generated_token_idx  generated_score leading_token  leading_token_idx  leading_score
0                ▁                29871         1.000000             ▁              29871       1.000000
1          Michael                24083         0.000027         ▁Sure              18585       0.959473
2          ▁Jordan                18284         1.000000       ▁Jordan              18284       1.000000
3             ▁was                  471         1.000000          ▁was                471       1.000000
4            ▁Born                19298         0.000008         ▁born               6345       1.000000
5              ▁in                  297         0.994629           ▁in                297       0.994629
6                ▁                29871         0.982422             ▁              29871       0.982422
7                1                29896         1.000000             1              29896       1.000000
8                9                29929         1.000000             9              29929       1.000000
9                6                29953         1.000000             6              29953       1.000000
10               3                29941         1.000000             3              29941       1.000000
11               .                29889         0.999512             .              29889       0.999512
12            </s>                    2         0.981445          </s>                  2       0.981445
```

You can see that the model "wanted" to start the answer using ```Sure```, but the format enforcer forced it to use ```Michael``` - there was a big gap in token 1. Afterwards, the leading scores are all within the allowed token set, meaning the model likely did not hallucinate due to the token forcing.
