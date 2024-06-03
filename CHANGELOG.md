# LM Format Enforcer Changelog

## v0.10.2
- [#100] JsonSchemaParser: Added allOf support
- [#99] JsonSchemaParser: Fixed edge case that would allow leading comma in JSON Array 
- [#102] JsonSchemaParser: Fixed Array of Enums not producing multiple values

## v0.10.1
- Allowing control of LM Format Enforcer's heuristics via env var / configuration objects. See the 'Configuration options' section of the README.

## v0.9.10
- [#95] Added anyOf support to JsonSchemaParser, making function calls possible.

## v0.9.9
- Updated README with vLLM OpenAI Server Inference integration

## v0.9.8
- [#80] JSONSchemaParser List would allow opening comma before first element if there was a whitespace before it


## v0.9.7
- [#93] Improved JSONSchemaParser performance, unit tests run twice as fast! Joint effort with [Ari Weinstein](https://github.com/AriX). Thanks! 


## v0.9.6
- [#88] ExllamaV2 optimizations
- Bugfix in ExllamaV2 sample notebook that generated garbage data after the response.

## v0.9.5
- [#87] Allow regex to exit to forcestopparser when receiving a pad/eos token after having been in a final state. Thanks Josh C!

## v0.9.4
- [#27] Improving vLLM class support (AsyncLLMEngine etc)

## v0.9.3
- [#83](https://github.com/noamgat/lm-format-enforcer/pull/83) - Supporting added model tokens
- Improved support for out-of-token-vocabulary characters

## v0.9.2
- [#80](https://github.com/noamgat/lm-format-enforcer/issues/80) - Fixed bug where comma could start json list
- [#34](https://github.com/noamgat/lm-format-enforcer/issues/34) - Fixed llama-cpp-python low max_tokens default in sample

## v0.9.1
- Fixed build errors in certain edge cases

## v0.9.0
- [#68](https://github.com/noamgat/lm-format-enforcer/pull/68) Added NVIDIA TensorRT-LLM Support, NVIDIA's contribution by [Ahmet Erdem](https://github.com/aerdem4). Thanks!
- Much faster TokenizerData initialization, new JSON freetext token caching algorithm.
- More robust error reporting.

## v0.8.3
- [#67](https://github.com/noamgat/lm-format-enforcer/issues/67) Updating vLLM integration to support v0.3.0
- [#63](https://github.com/noamgat/lm-format-enforcer/issues/63) JSONSchemaParser: Empty list cannot be closed after a newline

## v0.8.2
Several `JsonSchemaParser` improvements:
- [#32](https://github.com/noamgat/lm-format-enforcer/issues/32) Added limited support for regex-constrained string in JSON using the `pattern` field. See `test_phone_number_in_string()`.
- [#54](https://github.com/noamgat/lm-format-enforcer/issues/54) Fixed regression bug caused by limited-length JSON string caching.
- [#53](https://github.com/noamgat/lm-format-enforcer/issues/53) Fixed problems with arrays of union types. 

## v0.8.1
- Performannce improvement: Limited-length JSON strings will also enjoy caching. Thanks [Jarno Elonen](https://github.com/elonen) for the contribution!

## v0.8.0
 - Performance improvement: Introduced `TokenEnforcerTokenizerData` that allows reusing the tokenizer preprocessing data between different `TokenEnforcer` instances. The sample notebooks have been updated to take advantage of this option.
 - Performance improvement: Long sequences will see up to 5x `TokenEnforcer` runtime footprint reduction.

## v0.7.3
- Bug fixes

## v0.7.2
- vLLM performance improvements
- Sample notebooks don't require huggingface account anymore

## v0.7.1
- Added [ExLlamaV2 integration](https://github.com/noamgat/lm-format-enforcer/blob/main/samples/colab_exllamav2_integration.ipynb)

## v0.7.0
- JSON Schema: Added support for union types. In pydantic, both `key: int | str` and `key: Union[int, str]` formats are supported
- JSON Schema: Added support for schemaless JSON mode. `JsonSchemaParser(None)` will now create a parser that accepts any valid JSON.

## v0.6.5
- Added official vLLM integration that doesn't require monkey patching.

## v0.6.4
- JSON Schema : Supports string min/max length limitation

## v0.6.3
- Community PR: Fixed SequenceParser bug

## v0.6.2
- Added haystack integration

## v0.6.1
- Fixed llama.cpp integration to be able to generate unicode characters in json freetext fields
- Fixed unescaped newlines being allowed in json freetext fields
  
## v0.6.0
- RegexParser and JsonSchemaParser can now output all of the characters that exist in the tokenzier
- Added "Known issues and limitations" section to the README
- Fixed a bug in JsonSchemaParser where sometimes illegal commas were allowed

## v0.5.2
- JSON Schema : Supports empty arrays and escape characters in strings
- Regex : Performance improvement in some cases
- Added `UnionParser` and `SequenceParser` to allow combining parsers

## v0.5.1
- Made it easier to report bugs in the library

## v0.5.0
- Introduced FormatEnforcerAnalyzer to allow all inference engines to be analyzed in a unified way. (Was previously only available for transformers)
- Added support for the analyser in llama.cpp, updated example notebook
- JsonSchemaParser now take list min/max items into consideration

## v0.4.3
- Improved JsonSchemaParser whitespace support
- Improved RegexParser performance, especially in regular expressions with `.+` and `.*` sections.

## v0.4.2
- Modified example in main README to be able to run end to end in Google Colab

## v0.4.1
- Added integration with the `LlamaIndex` library (huggingface and llama.cpp backends) via sample notebook.

## v0.4.0
- Introduced ```lmformatenforcer.integrations``` module which will have the integrations with inference engines.
- Added llama-cpp-python integration to the library in ```lmformatenforcer.integrations.llamacpp```
- Breaking API Change: Moved ```'build_transformers_prefix_allowed_tokens_fn', 
    'generate_enforced'``` from ```lmformatenforcer``` to ```lmformatenforcer.integrations.transformers```.
