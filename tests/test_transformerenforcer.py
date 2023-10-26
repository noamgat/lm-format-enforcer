from lmformatenforcer import RegexParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn, generate_enforced
from transformers import pipeline

def _build_pipeline():
    return pipeline('text-generation', model='hf-internal-testing/tiny-random-GPTNeoModel')

def _build_parser():
    return RegexParser('abc123')

def test_transfomers_pipelines_forward_params_integration():
    hf_pipeline = _build_pipeline()
    parser = _build_parser()
    prefix_function = build_transformers_prefix_allowed_tokens_fn(hf_pipeline.tokenizer, parser)
    hf_pipeline._forward_params['prefix_allowed_tokens_fn'] = prefix_function
    prompt = 'Generate a string'
    output_dict = hf_pipeline('Generate a string')
    output_text = output_dict[0]['generated_text'][len(prompt):]
    assert output_text == 'abc123'


def test_transfomers_pipelines_call_kwargs_integration():
    hf_pipeline = _build_pipeline()
    parser = _build_parser()
    prefix_function = build_transformers_prefix_allowed_tokens_fn(hf_pipeline.tokenizer, parser)
    prompt = 'Generate a string'
    output_dict = hf_pipeline('Generate a string', prefix_allowed_tokens_fn=prefix_function)
    output_text = output_dict[0]['generated_text'][len(prompt):]
    assert output_text == 'abc123'


def test_transfomers_generate_enforced_integration():
    hf_pipeline = _build_pipeline()
    parser = _build_parser()
    prompts = ['Generate a string', 'Generate a strang']
    inputs = hf_pipeline.tokenizer(prompts, return_tensors='pt')
    outputs = generate_enforced(hf_pipeline.model, hf_pipeline.tokenizer, parser, **inputs)
    for idx in range(len(prompts)):
        output_text = hf_pipeline.tokenizer.decode(outputs[idx], skip_special_tokens=True)[len(prompts[idx]):]
        assert output_text == 'abc123'


def test_transfomers_generate_function_integration():
    hf_pipeline = _build_pipeline()
    parser = _build_parser()
    prompts = ['Generate a string', 'Generate a strang']
    inputs = hf_pipeline.tokenizer(prompts, return_tensors='pt')
    prefix_function = build_transformers_prefix_allowed_tokens_fn(hf_pipeline.tokenizer, parser)
    outputs = hf_pipeline.model.generate(**inputs, prefix_allowed_tokens_fn=prefix_function)
    for idx in range(len(prompts)):
        output_text = hf_pipeline.tokenizer.decode(outputs[idx], skip_special_tokens=True)[len(prompts[idx]):]
        assert output_text == 'abc123'
