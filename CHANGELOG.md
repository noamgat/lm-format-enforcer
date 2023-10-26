# LM Format Enforcer Changelog

## v0.4.0
- Introduced ```lmformatenforcer.integrations``` module which will have the integrations with inference engines.
- Added llama-cpp-python integration to the library in ```lmformatenforcer.integrations.llamacpp```
- Breaking API Change: Moved ```'build_transformers_prefix_allowed_tokens_fn', 
    'generate_enforced'``` from ```lmformatenforcer``` to ```lmformatenforcer.integrations.transformers```.
