"""Descriptive linear CKA for aligned token representations (Kornblith et al., 2019)."""
from __future__ import annotations

import torch


def linear_cka(left: list[list[float]], right: list[list[float]],
               max_observations: int = 512) -> dict:
    if max_observations < 4:
        raise ValueError('max_observations must be at least four')
    if len(left) != len(right):
        raise ValueError('CKA observations must be aligned')
    n = len(left)
    result = {'observation_count': n, 'used_observations': min(n, max_observations),
              'value': None, 'estimator': 'biased_centered_linear_cka',
              'selection': 'evenly_spaced_aligned_observations'}
    if n < 4:
        result['warning'] = 'At least four aligned representations required; small-n CKA is biased.'
        return result
    indices = torch.linspace(0, n - 1, min(n, max_observations)).long().tolist()
    x = torch.tensor([left[i] for i in indices], dtype=torch.float64)
    y = torch.tensor([right[i] for i in indices], dtype=torch.float64)
    if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
        raise ValueError('Nonfinite representations cannot be compared')
    x, y = x - x.mean(0), y - y.mean(0)
    gx, gy = x @ x.T, y @ y.T
    denominator = torch.linalg.vector_norm(gx) * torch.linalg.vector_norm(gy)
    if denominator.item() == 0:
        result['warning'] = 'Constant representations have undefined CKA.'
    else:
        result['value'] = float(((gx * gy).sum() / denominator).clamp(0, 1))
    return result
