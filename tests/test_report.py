from __future__ import annotations

from llm_training_evaluator.report import write_html_report


def _top_token(token_id: int, text: str, probability: float, rank: int) -> dict:
    return {
        "token_id": token_id,
        "text": text,
        "probability": probability,
        "rank": rank,
    }


def test_single_model_report_prioritizes_core_metrics(tmp_path) -> None:
    payload = {
        "metadata": {
            "model": "demo/model-a",
            "model_type": "llama",
            "num_layers": 2,
        },
        "samples": [
            {
                "id": "critical-case",
                "prompt": "请分析 <unsafe> 输入",
                "tags": ["critical", "regression"],
                "layer_predictions": [
                    {
                        "layer": 0,
                        "position": 3,
                        "entropy": 2.3,
                        "top_tokens": [
                            _top_token(10, "答案", 0.42, 1),
                            _top_token(11, "分析", 0.25, 2),
                        ],
                    },
                    {
                        "layer": 1,
                        "position": 3,
                        "entropy": 1.7,
                        "top_tokens": [
                            _top_token(11, "分析", 0.61, 1),
                            _top_token(10, "答案", 0.20, 2),
                        ],
                    },
                ],
                "token_similarity": [
                    {
                        "token_id": 11,
                        "text": "分析",
                        "neighbors": [
                            {
                                "token_id": 12,
                                "text": "推理",
                                "similarity": 0.91,
                                "rank": 1,
                            }
                        ],
                    }
                ],
                "moe_routing": [
                    {
                        "layer": 1,
                        "module": "model.layers.1.mlp.gate",
                        "routing_entropy": 0.38,
                        "expert_load": {0: 2, 1: 1},
                        "selected_routes": [
                            {
                                "token_position": 3,
                                "experts": [0, 1],
                                "weights": [0.75, 0.25],
                            }
                        ],
                    }
                ],
                "final_logit_lens_validation": {"max_abs": 1e-8},
                "warnings": [],
            }
        ],
    }

    output = write_html_report(payload, tmp_path / "single.html")
    report = output.read_text(encoding="utf-8")

    assert "模型逐层训练效果分析" in report
    assert "核心指标总览" in report
    assert "逐层 Top-1 概率" in report
    assert "完整 Top-K" in report
    assert "Token Embedding 相似度" in report
    assert "MoE 专家路由" in report
    assert "chart-svg" in report
    assert "&lt;unsafe&gt;" in report
    assert "<unsafe>" not in report
    assert "https://" not in report


def test_comparison_report_surfaces_hotspots_and_drift(tmp_path) -> None:
    payload = {
        "metadata": {"model_a": "demo/model-a", "model_b": "demo/model-b"},
        "summary": {
            "top1_agreement": 0.5,
            "mean_topk_overlap": 0.62,
            "mean_probability_l1": 0.14,
            "mean_expert_load_tv": 0.08,
        },
        "samples": [
            {
                "id": "sample-1",
                "prompt": "比较两个模型",
                "layer_comparisons": [
                    {
                        "layer": 0,
                        "top1_a": "答案",
                        "top1_b": "答案",
                        "top1_same": True,
                        "topk_overlap": 0.8,
                        "probability_l1": 0.04,
                    },
                    {
                        "layer": 1,
                        "top1_a": "答案",
                        "top1_b": "拒绝",
                        "top1_same": False,
                        "topk_overlap": 0.3,
                        "probability_l1": 0.24,
                    },
                ],
                "moe_comparisons": [
                    {
                        "layer": 1,
                        "module": "model.layers.1.mlp.gate",
                        "expert_load_tv": 0.08,
                        "routing_entropy_delta": -0.03,
                    }
                ],
            }
        ],
    }

    output = write_html_report(payload, tmp_path / "comparison.html")
    report = output.read_text(encoding="utf-8")

    assert "双权重逐层对比" in report
    assert "差异热点" in report
    assert "Top-K 重叠率" in report
    assert "概率 L1 漂移" in report
    assert "MoE 路由差异" in report
    assert "demo/model-a" in report
    assert "demo/model-b" in report
    assert "变化" in report
    assert "chart-svg" in report
    assert "https://" not in report
