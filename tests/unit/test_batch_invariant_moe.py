"""CPU isolation of routing, variable expert occupancy and weighted combination."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from art.megatron.batch_invariant.modules import _te_grouped
from art.megatron.batch_invariant.ops import grouped_lora, matmul


def test_low_precision_cpu_oracle_accumulates_across_k_in_fp32():
    # The CUDA kernel keeps its accumulator in FP32 across every K tile.
    a = torch.ones(1, 96, dtype=torch.bfloat16)
    b = torch.cat((torch.full((32, 1), 8.0), torch.full((64, 1), 1 / 32))).bfloat16()
    expected = (a.float() @ b.float()).bfloat16()
    assert expected.item() == 258
    assert torch.equal(matmul(a, b), expected)


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16))
@pytest.mark.parametrize("shared", (None, "A", "B"))
def test_native_routing_expert_tiles_and_combine_ignore_occupancy(dtype, shared):
    check_native_moe_occupancy(dtype, shared)


def check_native_moe_occupancy(dtype, shared, *, device="cpu", fused=False):
    utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
    torch.manual_seed(63)
    x = torch.randn(1000, 35, dtype=dtype, device=device)
    logits = torch.randn(1000, 4, dtype=torch.float32, device=device)
    logits[13] = 0  # Preserve the same exact top-k tie in every composition.
    weights = [torch.randn(17, 35, dtype=dtype, device=device) * 0.1 for _ in range(4)]
    module = SimpleNamespace(
        num_gemms=4,
        use_bias=False,
        apply_bias=False,
        return_bias=False,
        te_return_bias=False,
        **{f"weight{i}": w for i, w in enumerate(weights)},
    )
    a = torch.randn(1 if shared == "A" else 4, 35, 8, dtype=dtype, device=device) * 0.1
    b = torch.randn(1 if shared == "B" else 4, 8, 17, dtype=dtype, device=device) * 0.1

    def forward(xx, scores):
        probs, routes = utils.topk_routing_with_score_function(scores, 2)
        permuted, pp, indices, _, _ = utils.permute(
            xx, routes, probs=probs, num_out_tokens=2 * len(xx), fused=fused
        )
        counts = routes.sum(0).tolist()
        base, _ = _te_grouped(module, permuted, counts)
        values = base + grouped_lora(permuted, a, b, counts)
        # Both native weighting locations preserve expert-ID order. Exercise
        # the unpermute weighting path directly, including its autograd gather.
        out = utils.unpermute(
            values, indices, (len(xx), 17), probs=probs, routing_map=routes, fused=fused
        )
        assert pp is not None
        return out, routes, counts

    reference, routes, full_counts = forward(x, logits)
    for length in (1, 4, 5, 63, 64, 65, 129, 1000):
        out, prefix_routes, counts = forward(x[:length], logits[:length])
        assert torch.equal(prefix_routes, routes[:length])
        assert torch.equal(out, reference[:length])
        assert sum(counts) == 2 * length
        if length == 63:
            assert counts != full_counts  # The expert GEMM M dimensions changed.
        moved = forward(
            torch.cat((x[-65:] * 100, x[:length])),
            torch.cat((logits[-65:], logits[:length])),
        )[0]
        assert torch.equal(out, moved[65:])


def test_expert_tile_backward_and_weighted_unpermute_match_fp64():
    utils = pytest.importorskip("megatron.core.transformer.moe.moe_utils")
    torch.manual_seed(13)
    x = torch.randn(64, 35, dtype=torch.float64, requires_grad=True)
    w = torch.randn(4, 17, 35, dtype=torch.float64, requires_grad=True)
    probs = torch.rand(64, 4, dtype=torch.float64, requires_grad=True)
    routes = torch.zeros(64, 4, dtype=torch.bool)
    routes[:, :2] = True  # Two empty experts must keep zero parameter gradients.
    permuted, _, indices, _, _ = utils.permute(
        x, routes, num_out_tokens=128, fused=False
    )
    module = SimpleNamespace(
        num_gemms=4,
        use_bias=False,
        apply_bias=False,
        return_bias=False,
        te_return_bias=False,
        **{f"weight{i}": v for i, v in enumerate(w)},
    )
    actual = utils.unpermute(
        _te_grouped(module, permuted, routes.sum(0).tolist())[0],
        indices,
        (64, 17),
        probs=probs,
        routing_map=routes,
        fused=False,
    )
    expected = sum(
        (F.linear(x, w[e]) * probs[:, e : e + 1] for e in range(2)),
        start=x.new_zeros((64, 17)),
    )
    dy = torch.randn_like(actual)
    ga = torch.autograd.grad(actual, (x, w, probs), dy, retain_graph=True)
    ge = torch.autograd.grad(expected, (x, w, probs), dy)
    torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-13)
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e, atol=1e-13, rtol=1e-13)
    assert ga[1][2:].count_nonzero() == 0
