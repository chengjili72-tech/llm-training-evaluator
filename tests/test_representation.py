import pytest

from llm_training_evaluator.representation import linear_cka


def test_cka_invariant_to_rotation_scale_and_translation():
    x = [[1., 2.], [3., 1.], [-1., 4.], [2., -3.], [-2., -1.]]
    y = [[-3 * b + 7, 3 * a - 4] for a, b in x]
    assert linear_cka(x, y)['value'] == pytest.approx(1)


def test_cka_rejects_unpaired_and_flags_degeneracy():
    assert linear_cka([[1.]] * 3, [[1.]] * 3)['value'] is None
    assert linear_cka([[1.]] * 4, [[2.]] * 4)['value'] is None
    with pytest.raises(ValueError):
        linear_cka([[1.]], [[1.], [2.]])


def test_cka_observation_budget_is_bounded():
    x = [[float(i), float(i % 7)] for i in range(800)]
    result = linear_cka(x, x)
    assert result['observation_count'] == 800
    assert result['used_observations'] == 512
    assert result['value'] == pytest.approx(1)
