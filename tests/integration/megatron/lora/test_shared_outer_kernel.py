"""Numerical qualification for the compact shared-outer QuACK path."""

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rank", [4, 8, 12, 16, 32])
@pytest.mark.parametrize("shared_factor", ["A", "B"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_shared_outer_matches_expanded_grouped(rank, shared_factor, dtype):
    from art.megatron.kernels.shared_outer_lora import shared_outer_lora

    torch.manual_seed(19)
    counts = torch.tensor([7, 0, 13], dtype=torch.int64)
    x = torch.randn(20, 128, device="cuda", dtype=dtype, requires_grad=True)
    a = (
        torch.randn(
            1 if shared_factor == "A" else 3, 128, rank, device="cuda", dtype=dtype
        )
        * 0.05
    ).requires_grad_()
    b = (
        torch.randn(
            1 if shared_factor == "B" else 3, rank, 128, device="cuda", dtype=dtype
        )
        * 0.05
    ).requires_grad_()
    reference = [value.detach().clone().requires_grad_() for value in (x, a, b)]
    xx, aa, bb = reference
    # Per-expert Torch reference: QuACK itself rejects ranks like 12.
    rows = torch.split(xx, counts.tolist())
    expected = torch.cat(
        [
            value
            @ aa[0 if shared_factor == "A" else expert]
            @ bb[0 if shared_factor == "B" else expert]
            * 0.75
            for expert, value in enumerate(rows)
        ]
    )
    actual = shared_outer_lora(x, a, b, counts, scale=0.75, shared_factor=shared_factor)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    expected.backward(grad)
    # Compact shared gradients accumulate before the final low-precision cast;
    # the expanded reference rounds each expert's gradient before summing.
    tolerance = 0.025 if dtype == torch.bfloat16 else 0.004
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    for value, ref in zip((x, a, b), reference, strict=True):
        torch.testing.assert_close(value.grad, ref.grad, atol=tolerance, rtol=tolerance)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shared_factor", ["A", "B"])
def test_shared_outer_empty_rank_retains_zero_gradients(shared_factor):
    from art.megatron.kernels.shared_outer_lora import shared_outer_lora

    x = torch.empty(0, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    a = torch.ones(
        1 if shared_factor == "A" else 3,
        128,
        12,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    b = torch.ones(
        1 if shared_factor == "B" else 3,
        12,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    shared_outer_lora(
        x, a, b, torch.tensor([0, 0, 0]), shared_factor=shared_factor
    ).sum().backward()
    for value in (x, a, b):
        assert value.grad is not None
        assert torch.count_nonzero(value.grad).item() == 0
