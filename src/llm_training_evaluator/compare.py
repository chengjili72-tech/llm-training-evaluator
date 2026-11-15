from __future__ import annotations

import math
from statistics import fmean
from typing import Any

from .metrics import paired_bootstrap_nll
from .representation import linear_cka


def _hidden_cosine(left: list[float] | None, right: list[float] | None) -> float | None:
    if left is None or right is None or len(left) != len(right):
        return None
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return None
    return dot / (left_norm * right_norm)


def _layer_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_tokens = {item["token_id"]: item["probability"] for item in left["top_tokens"]}
    right_tokens = {item["token_id"]: item["probability"] for item in right["top_tokens"]}
    left_ids = set(left_tokens)
    right_ids = set(right_tokens)
    union = left_ids | right_ids
    intersection = left_ids & right_ids
    return {
        "layer": left["layer"],
        "position": left["position"],
        "top1_agreement": left["top_tokens"][0]["token_id"]
        == right["top_tokens"][0]["token_id"],
        "model_a_top1": left["top_tokens"][0],
        "model_b_top1": right["top_tokens"][0],
        "top_k_overlap": len(intersection) / max(min(len(left_ids), len(right_ids)), 1),
        "top_k_jaccard": len(intersection) / max(len(union), 1),
        "top_k_probability_l1": sum(
            abs(left_tokens.get(token_id, 0.0) - right_tokens.get(token_id, 0.0))
            for token_id in union
        ),
        "entropy_delta": right["entropy"] - left["entropy"],
        "hidden_cosine": _hidden_cosine(
            left.get("hidden_state"), right.get("hidden_state")
        ),
    }


def _load_distribution(result: dict[str, Any]) -> dict[int, float]:
    counts = {int(key): float(value) for key, value in result.get("expert_load", {}).items()}
    total = sum(counts.values())
    if total == 0:
        return counts
    return {key: value / total for key, value in counts.items()}


def _moe_comparison(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_load = _load_distribution(left)
    right_load = _load_distribution(right)
    experts = set(left_load) | set(right_load)
    left_routes = {
        route["position"]: {expert["expert_id"] for expert in route["experts"]}
        for route in left.get("selected_routes", [])
    }
    right_routes = {
        route["position"]: {expert["expert_id"] for expert in route["experts"]}
        for route in right.get("selected_routes", [])
    }
    common_positions = set(left_routes) & set(right_routes)
    overlaps = [
        len(left_routes[position] & right_routes[position])
        / max(min(len(left_routes[position]), len(right_routes[position])), 1)
        for position in common_positions
    ]
    return {
        "layer": left["layer"],
        "model_a_module": left["module"],
        "model_b_module": right["module"],
        "routing_entropy_delta": right["routing_entropy"] - left["routing_entropy"],
        "expert_load_total_variation": 0.5
        * sum(abs(left_load.get(expert, 0.0) - right_load.get(expert, 0.0)) for expert in experts),
        "selected_expert_overlap": fmean(overlaps) if overlaps else None,
    }


def compare_analyses(model_a: dict[str, Any], model_b: dict[str, Any]) -> dict[str, Any]:
    meta_a = model_a["metadata"]
    meta_b = model_b["metadata"]
    errors = []
    for key in ("model_type", "num_layers", "tokenizer_hash"):
        if meta_a.get(key) != meta_b.get(key):
            errors.append(f"{key}: {meta_a.get(key)!r} != {meta_b.get(key)!r}")
    if errors:
        raise ValueError("Models are not directly comparable: " + "; ".join(errors))

    for name, payload in (("A", model_a), ("B", model_b)):
        ids = [s["id"] for s in payload["samples"]]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f"Model {name} has empty or duplicate sample IDs")
    if {s["id"] for s in model_a["samples"]} != {s["id"] for s in model_b["samples"]}:
        raise ValueError("Sample ID sets differ")
    for key in ("tokenizer_config_hash", "scoring_protocol"):
        if meta_a.get(key) != meta_b.get(key):
            raise ValueError(f"Evaluation protocol differs: {key}")
    samples_b = {sample["id"]: sample for sample in model_b["samples"]}
    nll_pairs = []
    pairs_by_tag: dict[str, list] = {}
    sample_comparisons = []
    all_layers = []
    all_moe = []
    representations: dict[int, tuple[list, list]] = {}
    for sample_a in model_a["samples"]:
        sample_b = samples_b.get(sample_a["id"])
        if sample_b is None:
            raise ValueError(f"Sample {sample_a['id']!r} is missing from model B results")
        if (sample_a.get("prompt"), sample_a.get("target"), sample_a.get("tags", [])) != (
            sample_b.get("prompt"), sample_b.get("target"), sample_b.get("tags", [])
        ):
            raise ValueError(f"Prompt, target or tags differ for sample {sample_a['id']!r}")
        score_a, score_b = sample_a.get("target_metrics"), sample_b.get("target_metrics")
        target_comparison = None
        if bool(score_a) != bool(score_b):
            raise ValueError("Target scoring coverage differs")
        if score_a and score_b:
            if score_a.get("status") != score_b.get("status"):
                raise ValueError("Target scoring status differs")
            if score_a.get("status") == "scored":
                for key in ("input_ids", "target_start", "context_window", "stride", "boundary_policy"):
                    if score_a[key] != score_b[key]:
                        raise ValueError(f"Target scoring protocol differs: {key}")
                if score_a["token_count"] != score_b["token_count"]:
                    raise ValueError("Target token counts differ")
                pair = (score_a["nll_sum"], score_b["nll_sum"], score_a["token_count"])
                nll_pairs.append(pair)
                for tag in sample_a.get("tags", []):
                    pairs_by_tag.setdefault(tag, []).append(pair)
                target_comparison = {"mean_nll_delta": score_b["mean_nll"] - score_a["mean_nll"],
                                     "accuracy_delta": score_b["accuracy"] - score_a["accuracy"],
                                     "token_count": score_a["token_count"]}
        if [item["token_id"] for item in sample_a["tokens"]] != [
            item["token_id"] for item in sample_b["tokens"]
        ]:
            raise ValueError(f"Tokenization differs for sample {sample_a['id']!r}")

        for name, sample in (("A", sample_a), ("B", sample_b)):
            keys = [(r["layer"], r["position"]) for r in sample["layer_predictions"]]
            if len(set(keys)) != len(keys):
                raise ValueError(f"Model {name} contains duplicate layer/position rows")
        layers_b = {
            (item["layer"], item["position"]): item
            for item in sample_b["layer_predictions"]
        }
        if {(r["layer"], r["position"]) for r in sample_a["layer_predictions"]} != set(layers_b):
            raise ValueError("Layer/position sets differ")
        layer_rows = []
        for layer_a in sample_a["layer_predictions"]:
            key = (layer_a["layer"], layer_a["position"])
            if key not in layers_b:
                raise ValueError(f"Model B is missing layer/position {key}")
            hidden_a, hidden_b = layer_a.get("hidden_state"), layers_b[key].get("hidden_state")
            if hidden_a is not None and hidden_b is not None:
                left, right = representations.setdefault(layer_a["layer"], ([], []))
                left.append(hidden_a)
                right.append(hidden_b)
            row = _layer_comparison(layer_a, layers_b[key])
            layer_rows.append(row)
            all_layers.append(row)

        moe_a_by_layer = {item["layer"]: item for item in sample_a.get("moe_routing", [])}
        moe_b_by_layer = {item["layer"]: item for item in sample_b.get("moe_routing", [])}
        moe_rows = []
        for layer in sorted(set(moe_a_by_layer) & set(moe_b_by_layer)):
            row = _moe_comparison(moe_a_by_layer[layer], moe_b_by_layer[layer])
            moe_rows.append(row)
            all_moe.append(row)

        sample_comparisons.append(
            {
                "id": sample_a["id"],
                "prompt": sample_a["prompt"],
                "layer_comparisons": layer_rows,
                "moe_comparisons": moe_rows,
                "target_comparison": target_comparison,
            }
        )

    return {
        "metadata": {
            "schema_version": "1.1",
            "model_a": meta_a["model"],
            "source_metadata_a": meta_a,
            "source_metadata_b": meta_b,
            "model_b": meta_b["model"],
            "model_type": meta_a.get("model_type"),
            "num_layers": meta_a["num_layers"],
        },
        "summary": {
            "top1_agreement": fmean(float(row["top1_agreement"]) for row in all_layers) if all_layers else None,
            "mean_top_k_overlap": fmean(row["top_k_overlap"] for row in all_layers) if all_layers else None,
            "mean_top_k_jaccard": fmean(row["top_k_jaccard"] for row in all_layers) if all_layers else None,
            "mean_top_k_probability_l1": fmean(
                row["top_k_probability_l1"] for row in all_layers
            ) if all_layers else None,
            "mean_expert_load_total_variation": (
                fmean(row["expert_load_total_variation"] for row in all_moe)
                if all_moe
                else None
            ),
        },
        "representation_similarity": [{"layer": layer, **linear_cka(left, right)}
                                      for layer, (left, right) in sorted(representations.items())],
        "target_evaluation": {
            "paired_nll": paired_bootstrap_nll(nll_pairs) if nll_pairs else None,
            "by_tag": {tag: paired_bootstrap_nll(pairs) for tag, pairs in pairs_by_tag.items()},
            "interpretation": "Negative NLL B-A favors B on these targets. Percentile intervals "
                              "resample whole samples; no multiple-comparison correction."
        },
        "samples": sample_comparisons,
    }

