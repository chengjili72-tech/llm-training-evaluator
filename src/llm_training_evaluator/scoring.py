"""Teacher-forced conditional scoring with explicit target boundaries and context windows."""
from __future__ import annotations

from typing import Any

import torch

from .metrics import summarize_tokens, token_metrics
from .schemas import Sample


def encode_continuation(tokenizer: Any, prompt: str, target: str) -> tuple[list[int], int, str]:
    # Joint encoding preserves real BPE/SentencePiece boundaries when offsets are available.
    try:
        encoded = tokenizer(prompt + target, add_special_tokens=False,
                            return_offsets_mapping=True)
        ids = list(encoded["input_ids"])
        offsets = encoded["offset_mapping"]
        if any(begin < len(prompt) < end for begin, end in offsets):
            # Scoring a token containing prompt bytes would leak prompt into the denominator.
            prefix = tokenizer.encode(prompt, add_special_tokens=False)
            ids = prefix + tokenizer.encode(target, add_special_tokens=False)
            start = len(prefix)
            policy = "separate_encoding_boundary_merge"
        else:
            start = next((i for i, (_, end) in enumerate(offsets) if end > len(prompt)), len(ids))
            policy = "joint_offsets"
    except (NotImplementedError, TypeError, ValueError, KeyError):
        prefix = tokenizer.encode(prompt, add_special_tokens=False)
        suffix = tokenizer.encode(target, add_special_tokens=False)
        ids, start = prefix + suffix, len(prefix)
        policy = "separate_encoding_no_offsets"
    if start == 0 and ids:
        bos = tokenizer.bos_token_id
        if bos is None:
            raise ValueError("Target has no preceding context; provide a prompt or tokenizer BOS")
        ids.insert(0, bos)
        start = 1
    return ids, start, policy


def score_sample(model: Any, tokenizer: Any, sample: Sample, device: Any, *,
                 max_length: int, stride: int, bins: int) -> dict[str, Any] | None:
    if sample.target is None:
        return None
    if not sample.target:
        return {"status": "empty_target", **summarize_tokens([], bins), "tokens": []}
    ids, target_start, policy = encode_continuation(tokenizer, sample.prompt, sample.target)
    if target_start == len(ids):
        raise ValueError(f"Target of sample {sample.id!r} has no scoreable tokens")
    tokens = []
    # Disjoint groups of target labels, each with a right-aligned context window.
    # Label at global j is predicted from logits at j-1; no label is counted twice.
    for first in range(target_start, len(ids), stride):
        end = min(first + stride, len(ids))
        begin = max(0, end - max_length)
        if begin >= first:
            raise ValueError("Context window does not leave a predecessor for every label")
        input_ids = torch.tensor([ids[begin:end]], dtype=torch.long, device=device)
        with torch.inference_mode():
            output = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                           use_cache=False, return_dict=True)
            selected = output.logits[0, first - begin - 1:end - begin - 1]
            labels = torch.tensor(ids[first:end], device=selected.device)
            scored = token_metrics(selected, labels)
        for j, item in enumerate(scored, first):
            item.update(position=j, text=tokenizer.decode([item["token_id"]],
                                                         clean_up_tokenization_spaces=False))
        tokens.extend(scored)
    unknown = getattr(tokenizer, "unk_token_id", None)
    unknown_count = sum(t["token_id"] == unknown for t in tokens)
    summary = summarize_tokens(tokens, bins)
    return {"status": "scored", "boundary_policy": policy,
            "target_start": target_start, "input_ids": ids,
            "context_window": max_length, "stride": stride,
            "unknown_target_tokens": unknown_count,
            "utf8_target_bytes": len(sample.target.encode("utf-8")),
            **summary, "tokens": tokens}


def aggregate_scoring(results: list[dict[str, Any]], bins: int) -> dict[str, Any]:
    scored = [r for r in results if (r.get("target_metrics") or {}).get("status") == "scored"]
    def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"sample_count": len(items), **summarize_tokens(
            [t for r in items for t in r["target_metrics"]["tokens"]], bins)}
    overall = aggregate(scored)
    overall["by_tag"] = {tag: aggregate([r for r in scored if tag in r["tags"]])
                         for tag in sorted({t for r in scored for t in r["tags"]})}
    overall["interpretation"] = ("Teacher-forced conditional target likelihood; token accuracy and "
                                 "calibration do not measure free-generation task correctness.")
    return overall
