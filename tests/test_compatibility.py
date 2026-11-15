"""Offline HF save/load/forward contracts, with real library modules and tiny random weights.

EVALUATOR_TEST_DEVICE=npu:0 runs the identical matrix on Ascend. Not a quality benchmark.
"""
from __future__ import annotations

import os

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

from llm_training_evaluator.config import AnalysisConfig, ModelConfig
from llm_training_evaluator.runner import Evaluator
from llm_training_evaluator.schemas import Sample

COMMON = dict(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
              num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64,
              bos_token_id=1, eos_token_id=2, pad_token_id=0)
FAMILIES = {
    "llama": {}, "qwen2": {}, "qwen3": {"head_dim": 8},
    "mistral": {"head_dim": 8}, "mixtral": {"num_local_experts": 4, "num_experts_per_tok": 2},
    "qwen2_moe": {"num_experts": 4, "num_experts_per_tok": 2,
                   "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16},
    "qwen3_moe": {"num_experts": 4, "num_experts_per_tok": 2,
                   "moe_intermediate_size": 16, "head_dim": 8},
    "gemma": {"head_dim": 8}, "gemma2": {"head_dim": 8, "query_pre_attn_scalar": 8,
                                           "sliding_window": 32, "final_logit_softcapping": 2.0},
    "phi": {"partial_rotary_factor": .5},
    "phi3": {"original_max_position_embeddings": 64},
    "olmo": {}, "olmo2": {}, "starcoder2": {}, "stablelm": {}, "granite": {},
    "cohere": {"logit_scale": .3},
    "gpt2": {"n_embd": 16, "n_layer": 2, "n_head": 2, "n_positions": 64},
    "gptj": {"n_embd": 16, "n_layer": 2, "n_head": 2, "rotary_dim": 8, "n_positions": 64},
    "gpt_neox": {"rotary_pct": .5},
    "opt": {"word_embed_proj_dim": 8, "ffn_dim": 32},
    "bloom": {"n_head": 2, "n_layer": 2},
    "falcon": {"num_kv_heads": 2, "new_decoder_architecture": True},
    "mpt": {"d_model": 16, "n_heads": 2, "n_layers": 2, "max_seq_len": 64,
            "expansion_ratio": 2},
}


def save_tiny_checkpoint(path, family="llama"):
    torch.manual_seed(17)
    config = AutoConfig.for_model(family, **(COMMON | FAMILIES[family]))
    config._attn_implementation = "eager"
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(path)
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2,
                                  "[UNK]": 3, "Hello": 4, "world": 5, "test": 6},
                                 unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]",
                                       bos_token="[BOS]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.save_pretrained(path)
    return model, tokenizer


@pytest.mark.parametrize("family", FAMILIES)
def test_hf_architecture_roundtrip(tmp_path, family):
    save_tiny_checkpoint(tmp_path, family)
    device = os.environ.get("EVALUATOR_TEST_DEVICE", "cpu")
    result = Evaluator(ModelConfig(str(tmp_path), device=device, dtype=os.environ.get("EVALUATOR_TEST_DTYPE", "float32"),
                                  attn_implementation="eager"),
                       AnalysisConfig(top_k=3, similarity_top_k=2, position_mode="all",
                                      max_length=64, scoring_stride=16,
                                      projection_chunk_size=1)).run(
                           [Sample("s", "Hello world", target=" test", tags=["tiny"])])
    sample = result["samples"][0]
    assert sample["final_logit_lens_validation"]["passed"]
    assert sample["target_metrics"]["token_count"] == 1
    assert result["evaluation"]["sample_count"] == 1
    assert len(sample["layer_predictions"]) == 4
    js = sample["layer_predictions"][-1]["adjacent_js_divergence"]
    assert js is not None and -.000001 <= js <= .694


def test_native_scoring_matches_independent_teacher_forcing_and_native_only(tmp_path):
    model, tokenizer = save_tiny_checkpoint(tmp_path)
    result = Evaluator(ModelConfig(str(tmp_path), device='cpu', layers_path='does.not.exist'),
                       AnalysisConfig(native_only=True, max_length=64, scoring_stride=32)).run(
                           [Sample('s', 'Hello world', target=' test')])
    score = result['samples'][0]['target_metrics']
    model.eval()
    ids = torch.tensor([score['input_ids']])
    with torch.inference_mode():
        output = model(input_ids=ids)
        loss = torch.nn.functional.cross_entropy(output.logits[0, -2:-1].float(), ids[0, -1:],
                                                reduction='sum')
    assert score['nll_sum'] == pytest.approx(loss.item(), abs=1e-6)
    assert result['samples'][0]['layer_predictions'] == []
    assert result['samples'][0]['final_logit_lens_validation']['passed'] is None
