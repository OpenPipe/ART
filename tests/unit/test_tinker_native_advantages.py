import math
import warnings

import pytest

from art.tinker_native.data import compute_advantages


@pytest.mark.parametrize("rewards", [[0.0, 1.0], [0.0, 1.0, 2.0, 3.0]])
def test_advantages_use_population_standard_deviation(rewards):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((reward - mean) ** 2 for reward in rewards) / len(rewards))
    expected = [(reward - mean) / std for reward in rewards]
    advantages = compute_advantages(rewards)
    assert advantages == pytest.approx(expected)
    assert sum(advantage**2 for advantage in advantages) / len(rewards) == (
        pytest.approx(1.0)
    )


@pytest.mark.parametrize(
    "rewards",
    [[], [0.9], [0.9] * 7, [0.1] * 7, [0.0, 1e-12], [0.0, 1e-8], [0.0, 2e-8]],
)
def test_degenerate_advantages_are_zero_without_warnings(rewards):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert compute_advantages(rewards) == [0.0] * len(rewards)


def test_advantages_above_std_guard_are_normalized():
    assert compute_advantages([0.0, 4e-8]) == pytest.approx([-1.0, 1.0])


def test_advantage_normalization_can_be_disabled():
    assert compute_advantages([0.0, 1.0, 2.0, 3.0], False) == [-1.5, -0.5, 0.5, 1.5]
