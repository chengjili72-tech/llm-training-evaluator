from __future__ import annotations

import pytest

from llm_training_evaluator.compare import compare_analyses


def _analysis(model: str, top_token: int) -> dict:
    prediction = {
        "layer": 0,
        "position": 1,
        "entropy": 0.5,
        "hidden_state": [1.0, 0.0],
        "top_tokens": [
            {
                "token_id": top_token,
                "token": str(top_token),
                "text": str(top_token),
                "probability": 0.8,
                "rank": 1,
            },
            {
                "token_id": 2,
                "token": "2",
                "text": "2",
                "probability": 0.1,
                "rank": 2,
            },
        ],
    }
    return {
        "metadata": {
            "model": model,
            "model_type": "tiny",
            "num_layers": 1,
            "tokenizer_hash": "same",
        },
        "samples": [
            {
                "id": "s1",
                "prompt": "test",
                "tokens": [{"token_id": 1}],
                "layer_predictions": [prediction],
                "moe_routing": [],
            }
        ],
    }


def test_compare_detects_top1_change() -> None:
    result = compare_analyses(_analysis("a", 1), _analysis("b", 3))

    row = result["samples"][0]["layer_comparisons"][0]
    assert row["top1_agreement"] is False
    assert result["summary"]["top1_agreement"] == 0.0


def test_compare_rejects_different_tokenizers() -> None:
    left = _analysis("a", 1)
    right = _analysis("b", 1)
    right["metadata"]["tokenizer_hash"] = "different"

    with pytest.raises(ValueError, match="tokenizer_hash"):
        compare_analyses(left, right)

