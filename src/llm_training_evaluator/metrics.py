"""Native next-token metrics; never score truncated Top-K distributions."""
from __future__ import annotations

import math
import random
from typing import Any

import torch
from torch import Tensor


def finite_exp(value: float) -> float | None:
    return math.exp(value) if value < 709 else None


def token_metrics(logits: Tensor, labels: Tensor) -> list[dict[str, Any]]:
    """Optimistic rank assigns equal logits the same rank (1 + strictly greater)."""
    logits = logits.detach().float()
    labels = labels.to(logits.device)
    if logits.ndim != 2 or labels.shape != logits.shape[:1]:
        raise ValueError("Expected logits [tokens,vocabulary] and labels [tokens]")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Native logits contain NaN/Inf; metrics are undefined")
    logp = logits.log_softmax(-1)
    p = logp.exp()
    target_logp = logp.gather(-1, labels[:, None]).squeeze(-1)
    target_logits = logits.gather(-1, labels[:, None])
    ranks = (logits > target_logits).sum(-1) + 1
    confidence, predicted = p.max(-1)
    # Multiclass Brier score without allocating a one-hot vocabulary matrix.
    brier = (p.square().sum(-1) - 2 * target_logp.exp() + 1).clamp_min(0)
    columns = torch.stack((
        -target_logp, target_logp.exp(), ranks.float(), confidence,
        predicted.eq(labels).float(), -(p * logp).sum(-1), brier,
    ), dim=-1).cpu().tolist()
    ids = labels.cpu().tolist()
    return [dict(token_id=int(t), nll=float(a), probability=float(b), rank=int(c),
                 confidence=float(d), correct=bool(e), entropy=float(f), brier=float(g))
            for t, (a, b, c, d, e, f, g) in zip(ids, columns)]


def summarize_tokens(tokens: list[dict[str, Any]], bins: int = 15) -> dict[str, Any]:
    if bins < 1:
        raise ValueError("calibration_bins must be positive")
    if not tokens:
        return {"token_count": 0, "nll_sum": 0.0, "mean_nll": None, "perplexity": None,
                "accuracy": None, "ece": None, "calibration_bins": []}
    n = len(tokens)
    nll = math.fsum(t["nll"] for t in tokens)
    reliability = []
    ece = 0.0
    for i in range(bins):
        members = [t for t in tokens if min(int(t["confidence"] * bins), bins - 1) == i]
        count = len(members)
        conf = math.fsum(t["confidence"] for t in members) / count if count else None
        acc = sum(t["correct"] for t in members) / count if count else None
        if count:
            ece += count / n * abs(conf - acc)
        reliability.append(dict(lower=i / bins, upper=(i + 1) / bins, count=count,
                                confidence=conf, accuracy=acc))
    return dict(token_count=n, nll_sum=nll, mean_nll=nll / n,
                perplexity=finite_exp(nll / n), accuracy=sum(t["correct"] for t in tokens) / n,
                top5_accuracy=sum(t["rank"] <= 5 for t in tokens) / n,
                mrr=math.fsum(1 / t["rank"] for t in tokens) / n,
                mean_entropy=math.fsum(t["entropy"] for t in tokens) / n,
                brier=math.fsum(t["brier"] for t in tokens) / n,
                ece=ece, calibration_bins=reliability)


def paired_bootstrap_nll(pairs: list[tuple[float, float, int]], *,
                         replicates: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Resample independent samples, preserving tokens within each sample and A/B pairing.

    Each tuple is (sum NLL A, sum NLL B, shared scored-token count).
    """
    if not pairs or any(n <= 0 for _, _, n in pairs):
        raise ValueError("Bootstrap requires nonempty, positive-token pairs")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    point = math.fsum(b - a for a, b, _ in pairs) / sum(n for _, _, n in pairs)
    result = dict(estimate=point, unit="nats/token B-A", sample_count=len(pairs),
                  replicates=replicates, seed=seed, resampling_unit="sample", ci95=None)
    if len(pairs) < 2:
        result["warning"] = "At least two independent samples are needed for an interval."
        return result
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        chosen = rng.choices(pairs, k=len(pairs))
        draws.append(math.fsum(b - a for a, b, _ in chosen) / sum(n for _, _, n in chosen))
    draws.sort()
    def quantile(q: float) -> float:
        index = q * (len(draws) - 1)
        lo = int(index)
        return draws[lo] + (draws[min(lo + 1, len(draws) - 1)] - draws[lo]) * (index - lo)
    result["ci95"] = [quantile(.025), quantile(.975)]
    return result
