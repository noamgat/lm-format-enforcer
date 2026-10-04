import json

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.characterlevelparser import CharacterLevelParserConfig
from lmformatenforcer.consts import COMPLETE_ALPHABET
from lmformatenforcer.integrations.transformers import (
    build_token_enforcer_tokenizer_data,
    build_transformers_prefix_allowed_tokens_fn,
)
from lmformatenforcer.tokenenforcer import TokenEnforcer


VALUES = ["", "a", " a", "  ", "\n", "a\n", "é", "e\u0301", "😀", "😀a", "😀😀"]


@pytest.fixture(scope="module")
def tokenizer() -> PreTrainedTokenizerFast:
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=512,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<unk>", "<eos>"],
    )
    samples = [
        json.dumps(value, ensure_ascii=ascii_only)
        for value in VALUES
        for ascii_only in (False, True)
    ]
    backend.train_from_iterator(samples * 20, trainer)
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        clean_up_tokenization_spaces=False,
    )


@pytest.mark.parametrize("use_bitmask", [False, True])
def test_local_bytelevel_bpe_adapter_uses_decoded_string_length(
    tokenizer: PreTrainedTokenizerFast, use_bitmask: bool
) -> None:
    data = build_token_enforcer_tokenizer_data(tokenizer, use_bitmask=use_bitmask)
    for value in VALUES:
        for ascii_only in (False, True):
            text = json.dumps(value, ensure_ascii=ascii_only)
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert tokenizer.decode(ids) == text
            length = len(json.loads(text))
            for minimum in range(4):
                for maximum in range(minimum, 4):
                    parser = JsonSchemaParser(
                        {"type": "string", "minLength": minimum, "maxLength": maximum},
                        CharacterLevelParserConfig(
                            alphabet=COMPLETE_ALPHABET + "é\u0301😀"
                        ),
                    )
                    enforcer = TokenEnforcer(data, parser)
                    prefix = []
                    accepted = True
                    for token_id in ids:
                        if not enforcer.get_allowed_tokens(prefix).is_token_allowed(
                            token_id
                        ):
                            accepted = False
                            break
                        prefix.append(token_id)
                    if accepted:
                        accepted = bool(
                            enforcer.get_allowed_tokens(prefix).is_token_allowed(
                                tokenizer.eos_token_id
                            )
                        )
                    assert accepted == (minimum <= length <= maximum), (
                        text,
                        minimum,
                        maximum,
                    )


@pytest.mark.parametrize("length", [0, 1, 2])
def test_tiny_untrained_local_model_uses_production_prefix_adapter(length: int) -> None:
    # This exercises generation plumbing, not pretrained model accuracy.
    vocabulary = ["<unk>", "<eos>"] + [chr(index) for index in range(0x20, 0x7F)]
    backend = Tokenizer(
        models.BPE(
            vocab={piece: index for index, piece in enumerate(vocabulary)},
            merges=[],
            unk_token="<unk>",
        )
    )
    backend.decoder = decoders.Fuse()
    local_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        clean_up_tokenization_spaces=False,
    )
    schema = {
        "type": "object",
        "properties": {
            "value": {"type": "string", "minLength": length, "maxLength": length}
        },
        "required": ["value"],
        "additionalProperties": False,
    }
    parser = JsonSchemaParser(
        schema, CharacterLevelParserConfig(max_consecutive_whitespaces=1)
    )
    prefix_adapter = build_transformers_prefix_allowed_tokens_fn(
        local_tokenizer, parser
    )
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(2)
            config = GPT2Config(
                vocab_size=len(local_tokenizer),
                n_positions=128,
                n_embd=8,
                n_layer=1,
                n_head=1,
                bos_token_id=local_tokenizer.eos_token_id,
                eos_token_id=local_tokenizer.eos_token_id,
                pad_token_id=local_tokenizer.eos_token_id,
            )
            model = GPT2LMHeadModel(config).eval()
            inputs = local_tokenizer("0", return_tensors="pt")
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    prefix_allowed_tokens_fn=prefix_adapter,
                    max_new_tokens=64,
                    do_sample=False,
                )
            assert output[0, -1].item() == local_tokenizer.eos_token_id
            text = local_tokenizer.decode(
                output[0, inputs.input_ids.shape[-1] :], skip_special_tokens=True
            )
            value = json.loads(text)
            assert set(value) == {"value"}
            assert isinstance(value["value"], str)
            assert len(value["value"]) == length
    finally:
        torch.set_num_threads(old_threads)
