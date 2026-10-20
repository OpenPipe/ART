"""Offline checks of rank/scale geometry and independent gradient ownership."""

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from art.megatron.segmented_lora import (
    _CURRENT_LORA_SEGMENTS,
    LoRASegments,
    segmented_lora_factors,
    use_lora_segments,
)


@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("adapters", [1, 4, 16, 64])
def test_outputs_losses_and_original_parameter_gradients(adapters, grouped):
    torch.manual_seed(913)
    counts = tuple(0 if i % 7 == 1 else 3 + i % 5 for i in range(adapters)) + (5,)
    refs = tuple(cast(Any, f"run-{i}") for i in range(adapters)) + (None,)
    segments = LoRASegments(refs, counts, grouped=grouped)
    x = torch.randn(sum(counts), 11, requires_grad=True)
    factors = [
        (
            torch.nn.Parameter(torch.randn(11, i + 1) * 0.02),
            torch.nn.Parameter(torch.randn(i + 1, 13) * 0.02),
            1.375 / (i + 1),
        )
        for i in range(adapters)
    ]
    chunks = x.split(counts)
    expected = torch.cat(
        [(chunks[i] @ a) @ b * scale for i, (a, b, scale) in enumerate(factors)]
        + [chunks[-1].new_zeros(counts[-1], 13)]
    )

    def losses(output):
        # Unequal run lengths, separate request normalization. Empty requests
        # still produce connected zero gradients for their installed slots.
        return torch.stack(
            [
                chunk.square().sum() / max(n, 1)
                for chunk, n in zip(output.split(counts), counts, strict=True)
            ]
        )

    losses(expected).sum().backward()
    assert x.grad is not None
    dx = x.grad.clone()
    grads = [
        (cast(torch.Tensor, a.grad).clone(), cast(torch.Tensor, b.grad).clone())
        for a, b, _ in factors
    ]
    x.grad = None
    for a, b, _ in factors:
        a.grad = b.grad = None
    actual = segmented_lora_factors(
        x,
        factors + [None],
        segments,
        template_a=factors[0][0],
        template_b=factors[0][1],
    )
    losses(actual).sum().backward()
    atol, rtol = (1e-6, 1e-5) if grouped else (0, 0)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    torch.testing.assert_close(losses(actual), losses(expected), atol=atol, rtol=rtol)
    torch.testing.assert_close(x.grad, dx, atol=atol, rtol=rtol)
    for (a, b, _), (ga, gb) in zip(factors, grads, strict=True):
        torch.testing.assert_close(a.grad, ga, atol=atol, rtol=rtol)
        torch.testing.assert_close(b.grad, gb, atol=atol, rtol=rtol)


@pytest.mark.parametrize("grouped", [False, True])
def test_masked_loss_has_no_other_adapter_gradients(grouped):
    a = [torch.randn(7, r, requires_grad=True) for r in (1, 8)]
    b = [torch.randn(r, 5, requires_grad=True) for r in (1, 8)]
    segments = LoRASegments(cast(Any, ("a", "b")), (3, 9), grouped)
    output = segmented_lora_factors(
        torch.randn(12, 7),
        list(zip(a, b, (1.0, 0.5), strict=True)),
        segments,
        template_a=a[0],
        template_b=b[0],
    )
    output[:3].square().sum().backward()
    for p in (a[0], b[0]):
        assert p.grad is not None and p.grad.any()
    for p in (a[1], b[1]):
        assert p.grad is not None and not p.grad.any()


def test_layout_validation_empty_gradients_and_context_restoration():
    refs = cast(Any, ("first", "second"))
    table = LoRASegments(refs, (2, 5))
    assert LoRASegments.from_adapter_indices(refs, table.adapter_indices()) == table
    with pytest.raises(ValueError, match="sorted"):
        LoRASegments.from_adapter_indices(refs, table.adapter_indices().flip(0))
    for counts in ((True, 1), (-1, 2)):
        with pytest.raises(ValueError, match="nonnegative"):
            LoRASegments(refs, counts)
    with pytest.raises(ValueError, match="one sorted"):
        LoRASegments(cast(Any, ("a", "a")), (1, 1))
    a, b = torch.randn(7, 3, requires_grad=True), torch.randn(3, 5, requires_grad=True)
    for grouped in (False, True):
        a.grad = b.grad = None
        empty = replace(table, counts=(0, 0), grouped=grouped)
        result = segmented_lora_factors(
            torch.empty(0, 7),
            [(a, b, 1.0), None],
            empty,
            template_a=a,
            template_b=b,
        )
        result.sum().backward()
        assert a.grad is not None and not a.grad.any()
        assert b.grad is not None and not b.grad.any()
    with pytest.raises(ValueError, match="rows"):
        segmented_lora_factors(
            torch.empty(1, 7), [(a, b, 1.0), None], table, template_a=a, template_b=b
        )
    with pytest.raises(RuntimeError), use_lora_segments(table):
        assert _CURRENT_LORA_SEGMENTS.get() is table
        raise RuntimeError("forward failed")
    assert _CURRENT_LORA_SEGMENTS.get() is None


def test_canonical_root_slices_keep_original_adapter_ownership():
    from art.megatron.segmented_lora import slice_lora_segments

    table = LoRASegments(cast(Any, ("a", "b")), (256, 128))
    with use_lora_segments(table):
        with slice_lora_segments(128, 256):
            assert _CURRENT_LORA_SEGMENTS.get() == LoRASegments(table.refs, (128, 0))
        with slice_lora_segments(256, 384):
            assert _CURRENT_LORA_SEGMENTS.get() == LoRASegments(table.refs, (0, 128))
        assert _CURRENT_LORA_SEGMENTS.get() is table
    with pytest.raises(ValueError, match="within"):
        table.slice(0, 385)


def test_segment_metadata_owns_constructor_lists():
    counts = [2, 5]
    refs = ["a", "b"]
    table = LoRASegments(cast(Any, refs), cast(Any, counts))
    counts[0] = 100
    refs.reverse()
    assert table.counts == (2, 5) and table.refs == ("a", "b")
