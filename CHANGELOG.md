# LM Format Enforcer Changelog

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
