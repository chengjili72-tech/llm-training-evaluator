from __future__ import annotations

import torch
from torch import nn

from llm_training_evaluator.adapters import GenericDecoderAdapter
from llm_training_evaluator.config import AnalysisConfig
from llm_training_evaluator.probes import LogitLensProbe, MoeRoutingProbe, TokenSimilarityProbe


class TinyTokenizer:
    all_special_ids = [3]

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return ["zero", "one", "two", "<special>"][token_id]

    def decode(self, token_ids: list[int], **_kwargs: object) -> str:
        return self.convert_ids_to_tokens(token_ids[0])


class TinyBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(4, 2)
        self.layers = nn.ModuleList([nn.Identity(), nn.Identity()])
        self.norm = nn.LayerNorm(2)


class TinyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = TinyBody()
        self.lm_head = nn.Linear(2, 4, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.model.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


def test_logit_lens_matches_native_final_logits() -> None:
    torch.manual_seed(7)
    model = TinyLM()
    adapter = GenericDecoderAdapter(model)
    probe = LogitLensProbe(adapter, TinyTokenizer(), AnalysisConfig(top_k=2))
    hidden = torch.randn(1, 3, 2)
    input_ids = torch.tensor([[0, 1, 2]])
    mask = torch.ones_like(input_ids)

    result = probe.analyze(hidden, 1, input_ids, mask, is_last_layer=True)
    native = model.lm_head(model.model.norm(hidden))
    validation = probe.validate_final_logits(native)

    assert len(result) == 1
    assert result[0].position == 2
    assert validation["max_abs"] <= 1e-6


def test_token_similarity_excludes_self_and_special_tokens() -> None:
    model = TinyLM()
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(
            torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [1.0, 1.0]])
        )
    probe = TokenSimilarityProbe(GenericDecoderAdapter(model), TinyTokenizer(), chunk_size=2)

    result = probe.analyze(torch.tensor([[0]]), top_k=2)

    assert result[0]["neighbors"][0]["token_id"] == 1
    assert all(item["token_id"] not in {0, 3} for item in result[0]["neighbors"])


def test_moe_probe_reports_selected_experts_and_load() -> None:
    config = AnalysisConfig(position_mode="last")
    probe = MoeRoutingProbe(TinyTokenizer(), config, top_k=1)
    logits = torch.tensor([[[5.0, 1.0, 0.0], [0.0, 1.0, 4.0]]])
    input_ids = torch.tensor([[0, 1]])
    mask = torch.ones_like(input_ids)

    result = probe.analyze(logits, 0, "model.layers.0.mlp.gate", input_ids, mask)

    assert result is not None
    assert result.expert_load == {0: 1, 2: 1}
    assert result.selected_routes[0]["experts"][0]["expert_id"] == 2
