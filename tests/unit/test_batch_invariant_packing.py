"""CPU regressions at actual attention metadata and GDN dispatch boundaries."""

from types import SimpleNamespace
from typing import Any

import pytest
import torch

from art.megatron.batch_invariant.packing import root_local_gdn


@pytest.mark.parametrize("length", (63, 64, 65, 129))
@pytest.mark.parametrize("core", (False, True))
def test_packed_attention_uses_standalone_root_geometry(monkeypatch, length, core):
    attention = pytest.importorskip("art.megatron.flex_attn.attention")
    from art.megatron.context_parallel.core_attention import (
        ArtContextParallelCoreAttention,
    )
    from art.megatron.prefix_tree_state import create_prefix_tree_state

    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("CPU test touched CUDA")
    )
    calls = []

    def eager(q, k, v, **kw):
        calls.append((q.shape, q.stride(), k.shape, k.stride()))
        assert all(t.is_contiguous() for t in (q, k, v))
        # Flex has no CPU backward. Use native differentiable SDPA with the
        # production mask_mod while retaining the actual ART dispatch boundary.
        i = torch.arange(q.shape[2])[:, None]
        j = torch.arange(k.shape[2])[None, :]
        mask = kw["block_mask"].mask_mod(0, 0, i, j)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=kw["scale"], enable_gqa=kw["enable_gqa"]
        )

    monkeypatch.setattr(
        attention, "get_dense_compiled_flex_attention", lambda **_: eager
    )
    config: Any = SimpleNamespace(
        tensor_model_parallel_size=1,
        kv_channels=8,
        num_attention_heads=4,
        num_query_groups=2,
        softmax_type="vanilla",
    )
    cls = ArtContextParallelCoreAttention if core else attention.FlexDotProductAttention
    from megatron.core.transformer.enums import AttnMaskType

    module = cls(config, 1, AttnMaskType.causal, "self")
    torch.manual_seed(808)
    q = torch.randn(length, 1, 4, 8, requires_grad=True)
    k, v = [torch.randn(length, 1, 2, 8, requires_grad=True) for _ in range(2)]
    other = [torch.randn(65, 1, t.shape[2], 8) for t in (q, k, v)]

    def run(streams, lengths):
        groups = torch.cat(
            [
                torch.full((1, n), i + 1, dtype=torch.long)
                for i, n in enumerate(lengths)
            ],
            1,
        )
        state = create_prefix_tree_state(
            groups, groups, attention_head_dim=8, attention_value_head_dim=8
        )
        return module(*streams, torch.empty(0), attention_bias=state)

    alone = run((q, k, v), [length])
    geometry = calls[-1]
    packed = run(
        [torch.cat((o, t)) for o, t in zip(other, (q, k, v), strict=True)], [65, length]
    )
    assert calls[-1] == geometry  # same tile origin, lengths AND strides
    assert torch.equal(alone, packed[65:])
    dy = torch.randn_like(alone)
    ga = torch.autograd.grad(alone, (q, k, v), dy)
    gp = torch.autograd.grad(
        packed, (q, k, v), torch.cat((torch.zeros_like(packed[:65]), dy))
    )
    assert all(torch.equal(a, b) for a, b in zip(ga, gp, strict=True))
    changed = run(
        [torch.cat((o * 100, t)) for o, t in zip(other, (q, k, v), strict=True)],
        [65, length],
    )
    assert torch.equal(alone, changed[65:])


@pytest.mark.parametrize("length", (63, 64, 65, 129))
@pytest.mark.parametrize("packed", (False, True))
def test_gdn_dense_and_varlen_use_same_recurrence_and_gradients(length, packed):
    from tests.integration.megatron.model_support.gdn_storage_reference import recurrent

    torch.manual_seed(808)
    q, k, v = [
        torch.randn(1, length, 2, 3, dtype=torch.float64).mul(0.1).requires_grad_()
        for _ in range(3)
    ]
    g = (-torch.rand(1, length, 2, dtype=torch.float64)).requires_grad_()
    beta = torch.rand_like(g, requires_grad=True)
    xs = (q, k, v, g, beta)
    state = torch.randn(1, 2, 3, 3, dtype=torch.float64)
    seen = []

    def kernel(q, k, v, *, g, beta, initial_state, **kw):
        # Actual CPU recurrence; transport must choose the standalone FLA key.
        assert q.shape[0] == 1 and "cu_seqlens" not in kw
        assert all(t.is_contiguous() for t in (q, k, v, g, beta))
        seen.append((q.shape, q.stride(), kw))
        return recurrent(q[0], k[0], v[0], g[0], beta[0]).unsqueeze(
            0
        ), initial_state + v.sum()

    alone, final = root_local_gdn(kernel, q, k, v, g=g, beta=beta, initial_state=state)
    geometry = seen[-1]
    other_length = 65 if packed else length
    other = [
        torch.randn(1, other_length, *t.shape[2:], dtype=t.dtype) * 0.1 for t in xs
    ]
    dim = 1 if packed else 0
    combined = [torch.cat((o, t), dim=dim) for o, t in zip(other, xs, strict=True)]
    cu = torch.tensor([0, other_length, other_length + length]) if packed else None
    actual, actual_final = root_local_gdn(
        kernel,
        *combined[:3],
        g=combined[3],
        beta=combined[4],
        initial_state=torch.cat((state * 2, state)),
        cu_seqlens=cu,
        cu_seqlens_cpu=cu,
    )
    actual = actual[:, other_length:] if packed else actual[1:]
    assert seen[-1] == geometry and torch.equal(alone, actual)
    assert torch.equal(final, actual_final[1:])
    dy = torch.randn_like(alone)
    ga = torch.autograd.grad(alone, xs, dy)
    gb = torch.autograd.grad(actual, xs, dy)
    assert all(torch.equal(a, b) for a, b in zip(ga, gb, strict=True))
