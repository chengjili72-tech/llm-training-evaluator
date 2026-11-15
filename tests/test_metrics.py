from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from llm_training_evaluator.metrics import paired_bootstrap_nll, summarize_tokens, token_metrics
from llm_training_evaluator.scoring import encode_continuation, score_sample
from llm_training_evaluator.schemas import Sample


def test_uniform_distribution_and_tie_rank():
    tokens = token_metrics(torch.zeros(3, 4), torch.tensor([0, 1, 2]))
    summary = summarize_tokens(tokens, bins=4)
    assert summary['perplexity'] == pytest.approx(4)
    assert summary['brier'] == pytest.approx(.75)
    assert summary['accuracy'] == pytest.approx(1 / 3)
    assert summary['ece'] == pytest.approx(abs(.25 - 1 / 3))
    assert summary['mrr'] == 1  # Explicit optimistic tie convention, not token accuracy.
    assert sum(b['count'] for b in summary['calibration_bins']) == 3


def test_probabilities_ranks_match_independent_scalar_reference():
    logits = torch.tensor([[1., 2., -1.], [-1., 0., 3.]])
    rows = token_metrics(logits, torch.tensor([0, 2]))
    for scores, label, row in zip(logits.tolist(), [0, 2], rows):
        p = [math.exp(x) / sum(math.exp(y) for y in scores) for x in scores]
        assert row['nll'] == pytest.approx(-math.log(p[label]))
        assert row['brier'] == pytest.approx(sum((q - int(i == label)) ** 2
                                               for i, q in enumerate(p)), abs=1e-7)
        assert row['rank'] == 1 + sum(x > scores[label] for x in scores)


def test_nonfinite_logits_rejected():
    with pytest.raises(ValueError, match='NaN/Inf'):
        token_metrics(torch.tensor([[float('nan'), 0.]]), torch.tensor([0]))


def test_bootstrap_identity_sign_and_reproducibility():
    pairs = [(3., 2., 2), (4., 2., 3), (2., 1., 1)]
    result = paired_bootstrap_nll(pairs, seed=7)
    assert result == paired_bootstrap_nll(pairs, seed=7)
    assert result['estimate'] == pytest.approx(-4 / 6)
    assert result['ci95'][1] < 0
    assert paired_bootstrap_nll([(2., 2., 3), (5., 5., 2)])['ci95'] == [0, 0]
    assert paired_bootstrap_nll([(2., 1., 1)])['ci95'] is None


class CharTokenizer:
    bos_token_id = 0
    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) % 16 for c in text],
                'offset_mapping': [(i, i + 1) for i in range(len(text))]}
    def encode(self, text, **kwargs):
        return [ord(c) % 16 for c in text]
    def decode(self, ids, **kwargs):
        return str(ids[0])


class UniformModel:
    def __init__(self):
        self.lengths = []
    def __call__(self, input_ids, **kwargs):
        self.lengths.append(input_ids.shape[-1])
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 16))


@pytest.mark.parametrize('length,stride', [(2, 1), (5, 1), (5, 4), (8, 3)])
def test_windows_score_every_target_once_and_mask_prompt(length, stride):
    model = UniformModel()
    result = score_sample(model, CharTokenizer(), Sample('s', 'abcdef', 'ghijklmnop'),
                          'cpu', max_length=length, stride=stride, bins=5)
    assert result['token_count'] == 10
    assert [t['position'] for t in result['tokens']] == list(range(6, 16))
    assert result['perplexity'] == pytest.approx(16)
    assert max(model.lengths) <= length


def test_joint_boundary_spanning_token_and_empty_prompt():
    class BoundaryTokenizer(CharTokenizer):
        def __call__(self, text, **kwargs):
            return {'input_ids': [2, 3], 'offset_mapping': [(0, 3), (3, 4)]}
    ids, start, policy = encode_continuation(BoundaryTokenizer(), 'ab', 'cd')
    assert ids == [1, 2, 3, 4] and start == 2
    assert policy == 'separate_encoding_boundary_merge'
    ids, start, _ = encode_continuation(CharTokenizer(), '', 'ab')
    assert ids[0] == 0 and start == 1


def test_empty_target_not_counted_and_overflow_is_json_null():
    result = score_sample(UniformModel(), CharTokenizer(), Sample('s', 'abc', ''),
                          'cpu', max_length=8, stride=2, bins=5)
    assert result['token_count'] == 0 and result['perplexity'] is None
    rows = token_metrics(torch.tensor([[0., -1000.]]), torch.tensor([1]))
    assert summarize_tokens(rows)['perplexity'] is None


def test_window_logits_are_shifted_before_labels():
    class NextCharModel:
        def __call__(self, input_ids, **kwargs):
            logits = torch.full((*input_ids.shape, 16), -5.)
            logits.scatter_(-1, ((input_ids + 1) % 16).unsqueeze(-1), 5.)
            return SimpleNamespace(logits=logits)
    for stride in (1, 2):
        result = score_sample(NextCharModel(), CharTokenizer(), Sample('s', 'ab', 'cdefgh'),
                              'cpu', max_length=3, stride=stride, bins=5)
        assert result['accuracy'] == 1
        assert result['mean_nll'] < .001
