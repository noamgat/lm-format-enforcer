# LM Format Enforcer Changelog

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
