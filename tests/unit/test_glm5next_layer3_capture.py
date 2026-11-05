from __future__ import annotations

import torch

from tests.integration.megatron.glm5next_layer3_capture import (
    compiled_norm,
    indexer_rows,
    router_rows,
    sampler_fp8,
    sampler_hadamard128,
    sampler_pool_scores,
)


def test_indexer_classification_ignores_layout_and_detects_real_pool_loss() -> None:
    # A short row can differ in width, order, and duplicate sentinels yet select
    # the identical causal set. A missing complete pool must still be visible.
    native = torch.tensor(
        [[0, -1, -1, -1], [1, 0, -1, -1], [0, 1, 2, -1], [-1, -1, -1, -1]]
    )
    hf = torch.tensor(
        [[0, -1, -1, -1, -1], [0, 1, -1, -1, -1], [0, 1, 2, -1, -1], [0, 1, 2, 3, -1]]
    )
    report = indexer_rows(native, hf)
    assert report["counts"]["same_set_different_order"] == 1
    assert report["counts"]["all_candidates.set_disagreements"] == 1
    assert report["disagreements"] == [
        {"position": 3, "native_only_pools": [], "hf_only_pools": [0]}
    ]


def test_compiled_norm_distinguishes_the_saved_gpu_operation_order() -> None:
    torch.manual_seed(403)
    x = torch.randn(8, 4096).bfloat16()
    w = torch.randn(4096).bfloat16()
    normalized = x.float() * (x.float().square().mean(-1, keepdim=True) + 1e-5).rsqrt()
    eager = w * normalized.bfloat16()
    fused = (w.float() * normalized).bfloat16()
    assert (eager != fused).any()
    torch.testing.assert_close(compiled_norm(x, w), fused, rtol=0, atol=0)


def test_router_input_drift_is_not_an_exact_score_tie() -> None:
    # Deliberate, nonzero top-8/top-9 gaps on both sides, with one route swap.
    weight = torch.eye(9)
    bias = torch.zeros(9)
    hf = torch.tensor([[3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 0.001, -0.001]])
    native = hf.clone()
    native[0, 7:] *= -1
    hids = torch.sigmoid(hf).topk(8).indices
    nids = torch.sigmoid(native).topk(8).indices
    result = router_rows(native, hf, weight, bias, nids, hids)
    assert result["set_disagreements"] == 1
    assert result["native_capture_cpu_disagreements"] == 0
    assert result["hf_capture_cpu_disagreements"] == 0
    assert result["disagreement_margin_le_1e6"] == {"native": 0, "hf": 0}
    assert result["disagreement_margin_le_twice_input_score_drift"] == {
        "native": 1,
        "hf": 1,
    }


def test_sampler_rotation_and_fp8_match_independent_matrix_reference() -> None:
    torch.manual_seed(528)
    x = torch.randn(5, 128).bfloat16()
    h = torch.ones(1, 1)
    for _ in range(7):
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    reference = (x.float() @ h / 128**0.5).bfloat16()
    torch.testing.assert_close(sampler_hadamard128(x), reference, rtol=0, atol=0)
    quantized, scale = sampler_fp8(x)
    expected_scale = torch.exp2(
        torch.ceil(
            torch.log2(
                reference.float().abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
            )
        )
    )
    torch.testing.assert_close(scale, expected_scale, rtol=0, atol=0)
    torch.testing.assert_close(
        quantized.float(),
        (reference.float() / expected_scale).to(torch.float8_e4m3fn).float(),
        rtol=0,
        atol=0,
    )
    zero, zero_scale = sampler_fp8(torch.zeros(1, 128).bfloat16())
    assert not zero.float().any()
    assert zero_scale.item() > 0


def test_sampler_policy_can_change_selection_on_identical_inputs() -> None:
    torch.manual_seed(42)
    q = torch.randn(12, 32, 128).bfloat16()
    keys = torch.randn(80, 128).bfloat16()
    gates = torch.randn_like(keys)
    ape = torch.randn(4, 128)
    weights = torch.randn(12, 32)
    from art.megatron.glm5next.kpool_indexer import _pool_scores, compress_keys

    native = _pool_scores(
        q, compress_keys(keys, gates, ape, torch.arange(20) * 4), weights * 32**-0.5
    )
    sampler = sampler_pool_scores(q, keys, weights, gates, ape)
    assert torch.isfinite(sampler).all()
    assert (native - sampler).abs().max() > 0.01
    assert (
        native.topk(8).indices.sort().values != sampler.topk(8).indices.sort().values
    ).any()
