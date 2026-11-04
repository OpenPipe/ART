"""CPU checks of the WY storage-rounding analysis, independent of FLA/CUDA."""

import pytest
import torch
import torch.nn.functional as F

from tests.integration.megatron.model_support.gdn_storage_reference import (
    chunk,
    recurrent,
)


@pytest.mark.parametrize("length", (1, 4, 5, 63, 64))
def test_wy_algebra_and_gradients_match_fp64_recurrence(length):
    torch.manual_seed(41)
    q, k, v = [torch.randn(length, 2, 7, dtype=torch.float64) for _ in range(3)]
    q, k = [F.normalize(x, dim=-1) for x in (q, k)]
    g = -torch.rand(length, 2, dtype=torch.float64)
    beta = torch.rand(length, 2, dtype=torch.float64)
    xs = (q, k, v, g, beta)
    for x in xs:
        x.requires_grad_()
    actual = chunk(q, k, v, g, beta, torch.float64)
    expected = recurrent(*xs)
    dy = torch.randn_like(expected)
    ga = torch.autograd.grad(actual, xs, dy, retain_graph=True)
    ge = torch.autograd.grad(expected, xs, dy)
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-12)
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-13, rtol=1e-11)


def test_intermediate_rounding_is_separate_from_final_output_rounding():
    torch.manual_seed(3)
    q, k, v = [torch.randn(63, 2, 8) for _ in range(3)]
    q, k = [F.normalize(x, dim=-1).bfloat16() for x in (q, k)]
    xs = (q, k, v.bfloat16(), -torch.rand(63, 2), torch.rand(63, 2).bfloat16())
    reference = recurrent(*xs)
    low = chunk(*xs, torch.bfloat16)
    high = chunk(*xs, torch.float32).bfloat16().double()
    # This is an oracle precision check, not a replacement for the <1% GPU gate.
    assert (high - reference).norm() < (low - reference).norm()
    assert torch.equal(low, low.bfloat16().double())
