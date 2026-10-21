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


@pytest.mark.parametrize("reduction", ["sum", "strided"])
def test_grouped_backward_accepts_noncontiguous_cotangents(monkeypatch, reduction):
    pytest.importorskip("quack")
    from art.megatron.kernels import cute_grouped_lora_quack as kernel

    # Exercise the actual custom autograd boundary on CPU. Only the CUDA GEMMs
    # are replaced, retaining their last-dimension stride contract.
    def gemm(values, weights, *, expert_offsets, alpha=1.0, **kwargs):
        assert values.stride(-1) == 1
        counts = expert_offsets.diff().tolist()
        return (
            torch.cat(
                [chunk @ weights[i].T for i, chunk in enumerate(values.split(counts))]
            )
            * alpha
        )

    def gemm_k(left, right, *, expert_offsets, alpha=1.0, **kwargs):
        return (
            torch.stack(
                [
                    left[:, start:stop] @ right[:, start:stop].T
                    for start, stop in zip(
                        expert_offsets[:-1].tolist(),
                        expert_offsets[1:].tolist(),
                        strict=True,
                    )
                ]
            )
            * alpha
        )

    monkeypatch.setattr(kernel, "_varlen_quack_gemm", gemm)
    monkeypatch.setattr(kernel, "_varlen_quack_gemm_k", gemm_k)
    torch.manual_seed(74)
    counts = torch.tensor([2, 0, 3])
    x = torch.randn(5, 4, requires_grad=True)
    a = torch.randn(3, 4, 3, requires_grad=True)
    b = torch.randn(3, 3, 6, requires_grad=True)
    expected = (
        torch.cat(
            [(chunk @ a[i]) @ b[i] for i, chunk in enumerate(x.split(counts.tolist()))]
        )
        * 1.375
    )
    cotangent = (
        torch.ones(1).expand_as(expected) if reduction == "sum" else torch.randn(6, 5).T
    )
    assert cotangent.stride(-1) != 1
    expected.backward(cotangent)
    grads = [cast(torch.Tensor, p.grad).clone() for p in (x, a, b)]
    for p in (x, a, b):
        p.grad = None
    actual = kernel._QuackGroupedLoraFn.apply(x, a, b, counts, 1.375, None)
    actual.backward(cotangent)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=1e-5)
    for p, grad in zip((x, a, b), grads, strict=True):
        torch.testing.assert_close(p.grad, grad, atol=2e-5, rtol=1e-5)


def test_expert_views_reject_missing_dispatch_ownership(monkeypatch):
    pytest.importorskip("megatron.core")
    from art.megatron import lora as lm
    from art.megatron.lora import LoRA

    for name in (
        "get_expert_model_parallel_rank",
        "get_expert_data_parallel_rank",
        "get_data_parallel_rank",
    ):
        monkeypatch.setattr(lm.ps, name, lambda **_: 0)
    monkeypatch.setattr(lm.ps, "get_tensor_model_parallel_world_size", lambda: 1)
    lora = LoRA(
        "experts.{expert}",
        3,
        4,
        2,
        3.5,
        torch.float32,
        torch.device("cpu"),
        num_local_experts=3,
    )
    with use_lora_segments(LoRASegments((None,), (3,))):
        with pytest.raises(NotImplementedError, match="not supported"):
            lora.active_lora_tensors()
        with pytest.raises(NotImplementedError, match="not supported"):
            lora(torch.randn(3, 3))
