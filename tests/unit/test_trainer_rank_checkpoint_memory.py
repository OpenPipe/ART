"""Conditional checkpoint accounting; scalar/CPU evidence, not a CUDA bound."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import ForwardInput, TrainerRank
from art.trainer_rank._impl import (
    _TE_CUBLAS_WORKSPACE_BYTES,
    Unset,
    _ForwardRefusal,
    _MemoryProfile,
)


def rank():
    from megatron.core.transformer.transformer_block import TransformerBlock

    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(
        hidden_size=2048,
        num_layers=40,
        padded_vocab_size=32,
        params_dtype=torch.bfloat16,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        distribute_saved_activations=False,
        sequence_parallel=False,
        fp32_residual_connection=False,
        cpu_offloading=False,
        cuda_graph_impl="none",
        fp8=None,
        fp4=None,
    )
    block.layers = torch.nn.ModuleList(
        [torch.nn.Linear(1, 1).bfloat16() for _ in range(40)]
    )
    block.num_layers_per_pipeline_rank = 40
    model: Any = torch.nn.Module()
    model.config = block.config
    model.decoder = block
    model._preprocess = lambda: None
    result = TrainerRank(
        cast(
            Any,
            SimpleNamespace(
                model=[model],
                optimizer=None,
                provider=SimpleNamespace(hidden_size=2048, num_layers=40),
                model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
            ),
        )
    )
    result._moe_output_bytes_per_token = 188416
    result._moe_checkpoint_grad_bytes_per_token = 188416
    return result


def requests(grad=1024, reference=15360):
    return [
        ForwardInput(
            input_tokens=torch.arange(grad), hidden_states=True, no_grad=False
        ),
        ForwardInput(
            input_tokens=torch.arange(reference), hidden_states=True, no_grad=True
        ),
    ]


def price(r, values):
    n, out, signature, groups, head_workspace_bytes = values
    return r._subforward_cost(
        packed_tokens=n,
        output_bytes=out,
        signature=signature,
        logical_tokens=n,
        group_rows=groups,
        head_workspace_bytes=head_workspace_bytes,
    )


def test_same_old_signature_different_gradient_rows():
    r = rank()
    a = r._estimate_flat_forward(requests())
    b = r._estimate_flat_forward(requests(15360, 1024))
    assert a[:3] == b[:3]
    assert a[3] == ((1024, True), (15360, False))
    assert b[3] == ((15360, True), (1024, False))
    assert (
        r._checkpoint_memory_floor(a[3])[0] * 15 == r._checkpoint_memory_floor(b[3])[0]
    )
    assert price(r, b).required > price(r, a).required


def test_required_and_learned_retained_use_max_not_sum():
    r = rank()
    values = r._estimate_flat_forward(requests())
    n, out, sig, groups, head_workspace_bytes = values
    retained, work = r._checkpoint_memory_floor(groups)
    r._memory_profiles[sig] = _MemoryProfile(
        bytes_per_token=1,
        packed_tokens=n,
        logical_per_packed=1,
        retained_compute_bytes_per_token=1,
    )
    cost = price(r, values)
    old = max(n * 2048 * 2 * 14, n * 188416)
    gradient = r._checkpoint_input_gradient_bytes(groups)
    assert gradient == 1024 * 2048 * 2  # The incoming gradient only.
    assert cost.required == int((out + max(old, retained + gradient + work)) * 1.1)
    assert cost.retained == int((out + retained) * 1.1)
    r._memory_profiles[sig] = replace(
        r._memory_profiles[sig],
        bytes_per_token=1_000_000,
        retained_compute_bytes_per_token=500_000,
    )
    cost = price(r, values)
    assert cost.required == int((out + n * 1_000_000) * 1.1)
    assert cost.retained == int((out + n * 500_000) * 1.1)


def test_materialized_and_lower_bound_keep_group_association():
    r = rank()
    req = requests(17, 19)
    exact = r._estimate_flat_forward(req, exact=True)
    plan = r._plan_flat_forward(req)
    assert exact[3] == r._plan_group_rows(plan)
    assert r._memory_check(plan).estimated_required_bytes == price(r, exact).required
    rows = tuple(x.input_tokens for x in req)
    assert r._split_chunk_lower_cost(req, rows, checkpoint=Unset) == price(r, exact)


def test_per_group_padding_precedes_gradient_filter():
    r = rank()
    r._physical_tokens = lambda n: n + (-n % 8)
    values = r._estimate_flat_forward(requests(9, 17))
    assert values[0] == 40 and values[3] == ((16, True), (24, False))
    assert r._checkpoint_memory_floor(values[3]) == (
        16 * 40 * 4096,
        24 * (188416 + 4 * 2048 * 2) + _TE_CUBLAS_WORKSPACE_BYTES,
    )


def test_no_grad_enclosure_empty_and_unsupported():
    r = rank()
    assert r._checkpoint_memory_floor(()) == (0, 0)
    assert r._checkpoint_memory_floor(((8192, False),)) == (
        0,
        8192 * (188416 + 4 * 2048 * 2),
    )
    values = r._estimate_flat_forward(requests())
    baseline = price(r, values).required
    r.runtime.model[0].decoder.eval()
    n, out, sig, groups, head_workspace_bytes = values
    old = r._estimate_required_memory_bytes_from_values(
        packed_tokens=n, output_bytes=out, signature=sig
    )
    assert price(r, values).required == old <= baseline


@pytest.mark.parametrize(
    "field,value",
    [
        ("recompute_granularity", "selective"),
        ("recompute_method", "block"),
        ("recompute_num_layers", 2),
        ("distribute_saved_activations", True),
        ("sequence_parallel", True),
        ("fp32_residual_connection", True),
        ("cpu_offloading", True),
        ("cuda_graph_impl", "local"),
        ("params_dtype", torch.float32),
        ("fp8", "hybrid"),
        ("fp4", True),
        ("num_layers", 39),
        ("hidden_size", 1024),
    ],
)
def test_actual_config_revalidated(field, value):
    r = rank()
    assert r._checkpoint_memory_floor(((10, True),))[0] > 0
    setattr(r.runtime.model[0].decoder.config, field, value)
    assert r._checkpoint_memory_floor(((10, True),)) == (0, 0)


@pytest.mark.parametrize("axis", [1, 3])
def test_topology_revalidated(axis):
    r = rank()
    topology = [1, 1, 1, 1]
    topology[axis] = 2
    r._topology_key = lambda: tuple(topology)
    assert r._checkpoint_memory_floor(((10, True),)) == (0, 0)


@pytest.mark.parametrize("rows", [(10, True), (11, False)])
@pytest.mark.parametrize("cp", [2, 4])
def test_cp_floor_prices_rank_rows(cp, rows):
    # Callers pass rows on the most loaded CP rank; the per-row floor matches
    # CP1, except that CP attention also keeps its stage buffers while a
    # gradient group recomputes the layer.
    r = rank()
    single = r._checkpoint_memory_floor((rows,))
    r._topology_key = lambda: (1, 1, cp, 1)
    retained, workspace = r._checkpoint_memory_floor((rows,))
    assert retained == single[0] and single != (0, 0)
    count, grad = rows
    # No attention geometry in this stub: Q and KV widths fall back to hidden.
    stage = count * 2 * (3 * 2048 + 2 * 2048) if grad else 0
    assert workspace == single[1] + stage


@pytest.mark.parametrize("share", [lambda n: -(-n // 2), lambda n: n * 3 // 4])
def test_cp_probe_defers_and_lower_bound_stays_below_exact(monkeypatch, share):
    r = rank()
    monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 2, 1))
    monkeypatch.setattr(r, "_topology", lambda: SimpleNamespace(cp=2, tp=1))
    # Even (tight) and uneven ownership of each group by the busiest CP rank.
    monkeypatch.setattr(
        r, "_max_rank_model_tokens", lambda batch, **_: share(batch.tokens.numel())
    )
    reqs = requests()
    # Global counts would price the per-rank floors about cp times too high.
    assert r._estimate_flat_forward(reqs) is None
    plan = r._plan_flat_forward(reqs)
    assert r._checkpoint_memory_floor(r._plan_group_rows(plan))[0] > 0
    exact = r._plan_cost(plan)
    lower = r._split_chunk_lower_cost(
        reqs, [q.input_tokens for q in reqs], checkpoint=Unset
    )
    assert lower.required <= exact.required and lower.retained <= exact.retained


def test_cp_width_search_retries_full_sharing_inside_the_trust_window(monkeypatch):
    r = rank()
    r._dp_rank_and_size = lambda: (0, 1)
    monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 2, 1))
    monkeypatch.setattr(r, "_topology", lambda: SimpleNamespace(cp=2, tp=1))
    monkeypatch.setattr(
        r, "_max_rank_model_tokens", lambda batch, **_: batch.tokens.numel() * 3 // 4
    )
    r._available_memory_bytes = lambda: 1 << 50
    # Cost-optimal layouts stay unshared; memory-minimal layouts share fully.
    monkeypatch.setattr(
        r,
        "_layout_anchor",
        lambda *, memory_minimal=False: (
            "full_sharing" if memory_minimal else "no_sharing"
        ),
    )
    items = [
        ForwardInput(
            input_tokens=torch.tensor([7, 100 + i]), hidden_states=True, no_grad=True
        )
        for i in range(16)
    ]
    signature = r._plan_flat_forward(items[:1]).signature
    r._memory_profiles[signature] = _MemoryProfile(bytes_per_token=1, packed_tokens=2)
    # Unshared layouts leave the 8x window above width 8; full sharing
    # (width + 1 packed rows) stays trusted through width 15.
    selected = r._search_next_micro_batch(items, 0)
    assert not isinstance(selected, _ForwardRefusal)
    assert selected.stats_global_count == 15
    # The minimum wave retries too: one item's unshared layout (32 rows) is
    # outside a 3-row profile's window; full sharing (17 rows) is inside it.
    r._memory_profiles[signature] = _MemoryProfile(bytes_per_token=1, packed_tokens=3)
    r._last_global_micro_batch_size = None
    selected = r._search_next_micro_batch([items], 0)
    assert not isinstance(selected, _ForwardRefusal)
    assert selected.plan.packed_tokens == 17 and not selected.cold_start


def test_dp_empty_and_local_count():
    r = rank()
    r._topology_key = lambda: (3, 1, 1, 1)
    assert r._checkpoint_memory_floor(()) == (0, 0)
    block = r.runtime.model[0].decoder
    block.layers = torch.nn.ModuleList(list(block.layers[:4]))
    block.num_layers_per_pipeline_rank = 4
    block.config.num_layers = 4
    assert r._checkpoint_memory_floor(((10, True),))[0] == 10 * 4 * 2048 * 2
    block.num_layers_per_pipeline_rank = 3
    assert r._checkpoint_memory_floor(((10, True),)) == (0, 0)


def test_prior_live_graphs_are_not_an_added_term():
    r = rank()
    values = r._estimate_flat_forward(requests())
    cost = price(r, values)
    r._available_memory_bytes = lambda: cost.required - 1
    assert not r._memory_check_required(cost.required).fits
    # New-call cost is unchanged; live memory affects only existing availability.
    assert price(r, values) == cost


def test_existing_api_rejects_budget_below_conditional_checkpoint_term():
    # This same test reaches the original materialized-plan API on the base.
    r = rank()
    req = [
        ForwardInput(
            input_tokens=torch.arange(128),
            target_tokens=torch.arange(128),
            no_grad=False,
        )
    ]
    plan = r._plan_flat_forward(req)
    old_component = int((plan.output_bytes + 128 * 188416) * 1.1)
    r._available_memory_bytes = lambda: old_component + 1
    assert not r._memory_check(plan).fits


def test_split_keeps_complete_order_and_checks_each_new_subforward():
    from art.trainer_rank._impl import _SplitForwardPlan

    r = rank()
    req = [
        ForwardInput(
            input_tokens=torch.arange(128) + i * 1000,
            target_tokens=torch.arange(128),
            no_grad=False,
        )
        for i in range(4)
    ]
    flat = r._plan_flat_forward(req)
    r._memory_profiles[flat.signature] = _MemoryProfile(
        bytes_per_token=1,
        packed_tokens=512,
        logical_per_packed=1,
        retained_compute_bytes_per_token=1,
    )
    # Just below the unsplit plan's requirement: two subforwards must fit.
    limit = r._plan_cost(flat).required - 1
    used = 0
    r._available_memory_bytes = lambda: limit - used
    result = r._find_admissible_forward(req, checkpoint=Unset, refusal_prefix="test")
    assert isinstance(result, tuple)
    plan, check = result
    assert (
        isinstance(plan, _SplitForwardPlan)
        and len(plan.subforwards) == 2
        and check.fits
    )
    assert sorted(i for group in plan.request_indices for i in group) == list(range(4))
    restored = [None] * 4
    for sub, indices in zip(plan.subforwards, plan.request_indices, strict=True):
        assert r._memory_check(sub).fits
        for group in sub.groups:
            for local, item in zip(group.request_indices, group.items, strict=True):
                restored[indices[local]] = item.request
        used += r._plan_cost(sub).retained
    assert all(a is b for a, b in zip(restored, req, strict=True))


def test_optimistic_split_profile_cliff_preserves_checkpoint_floor():
    r = rank()
    req = [
        ForwardInput(
            input_tokens=torch.arange(128),
            target_tokens=torch.arange(128),
            no_grad=False,
        )
        for _ in range(16)
    ]
    full = r._plan_flat_forward(req, memory_minimal=True)
    r._memory_profiles[full.signature] = _MemoryProfile(
        bytes_per_token=1,
        packed_tokens=256,
        logical_per_packed=1,
        retained_compute_bytes_per_token=1,
    )
    cost = r._split_chunk_lower_cost(
        req, tuple(x.input_tokens for x in req), checkpoint=Unset
    )
    retained = 128 * 40 * 2048 * 2
    assert cost.retained == int((full.output_bytes + retained) * 1.1)


@pytest.mark.parametrize(
    "field,value", [("recompute_num_layers", True), ("cpu_offloading", 0)]
)
def test_malformed_flag_types_do_not_claim_supported_schedule(field, value):
    r = rank()
    setattr(r.runtime.model[0].decoder.config, field, value)
    assert r._checkpoint_memory_floor(((10, True),)) == (0, 0)


@pytest.mark.parametrize("profile_rate", [None, 1, 1_000_000])
def test_no_grad_enclosure_exact_lower_and_profile(profile_rate):
    r = rank()
    req = [
        ForwardInput(input_tokens=torch.arange(17), hidden_states=True, no_grad=True)
    ]
    values = r._estimate_flat_forward(req, exact=True)
    n, out, sig, groups, _ = values
    if profile_rate is not None:
        r._memory_profiles[sig] = _MemoryProfile(
            bytes_per_token=profile_rate,
            packed_tokens=n,
            logical_per_packed=1,
        )
    expected = int(
        (out + max(n * (188416 + 4 * 2048 * 2), n * (profile_rate or 0))) * 1.1
    )
    plan = r._plan_flat_forward(req)
    assert groups == r._plan_group_rows(plan) == ((n, False),)
    assert r._checkpoint_memory_floor(groups) == (0, n * (188416 + 4 * 2048 * 2))
    assert (
        price(r, values).required
        == r._memory_check(plan).estimated_required_bytes
        == expected
    )
    assert (
        r._split_chunk_lower_cost(
            req, tuple(x.input_tokens for x in req), checkpoint=Unset
        ).required
        == expected
    )
    r._available_memory_bytes = lambda: expected - 1
    assert not r._memory_check(plan).fits
    r._available_memory_bytes = lambda: expected
    assert r._memory_check(plan).fits


def test_no_grad_enclosure_uses_max_group_and_affine_stage():
    r = rank()
    r._moe_forward_stages = ((1, 1_000_000),)
    groups = ((3, False), (11, False))
    assert r._checkpoint_memory_floor(groups) == (
        0,
        max(
            max(rows * 188416, rows + 1_000_000) + 4 * rows * 2048 * 2
            for rows, _ in groups
        ),
    )
    mixed = ((3, True), (11, False))
    assert r._checkpoint_memory_floor(mixed) == (
        3 * 40 * 2048 * 2,
        max(3 * 188416, max(11 * 188416, 11 + 1_000_000) + 4 * 11 * 2048 * 2)
        + _TE_CUBLAS_WORKSPACE_BYTES,
    )


@pytest.mark.parametrize("gradient_first", [False, True])
def test_mixed_plan_keeps_reference_enclosure(gradient_first):
    r = rank()
    gradient, reference = requests(1, 10_000)
    req = [gradient, reference] if gradient_first else [reference, gradient]
    reference_plan = r._plan_flat_forward([reference])
    mixed = r._plan_flat_forward(req)
    reference_cost = r._plan_cost(reference_plan).required
    mixed_cost = r._plan_cost(mixed).required
    assert mixed_cost >= reference_cost
    # The exact plan and cheap split bound must retain the same group charge.
    assert r._memory_check(mixed).estimated_required_bytes == mixed_cost
    assert (
        r._split_chunk_lower_cost(
            req, tuple(x.input_tokens for x in req), checkpoint=Unset
        ).required
        == mixed_cost
    )
    assert not r._memory_profiles


@pytest.mark.parametrize("fits", [False, True])
def test_reference_prefix_search_agrees_with_mixed_demand(fits):
    r = rank()
    # This uninitialized CPU fixture declares DP1; all pricing/search stays real.
    r._dp_rank_and_size = lambda: (0, 1)
    gradient, reference = requests(1, 10_000)
    req = [reference, gradient]
    reference_plan = r._plan_flat_forward([reference])
    mixed = r._plan_flat_forward(req)
    # Retain the review witness's budget between its old nonmonotone costs.
    budget = r._plan_cost(mixed).required if fits else 2_207_849_881
    r._available_memory_bytes = lambda: budget
    assert r._memory_check(reference_plan).fits is fits
    assert r._memory_check(mixed).fits is fits
    selected = r._search_next_micro_batch(req, 0)
    if fits:
        assert not isinstance(selected, _ForwardRefusal)
        assert selected.check.fits and selected.cold_start
        assert selected.stats_global_count == 1
        assert (
            selected.check.estimated_required_bytes
            >= r._plan_cost(reference_plan).required
        )
    else:
        assert isinstance(selected, _ForwardRefusal)
        assert not selected.check.fits
        assert (
            selected.check.estimated_required_bytes
            == r._plan_cost(reference_plan).required
        )
    assert not r._memory_profiles


@pytest.mark.parametrize(
    "field,value",
    [
        ("recompute_granularity", None),
        ("recompute_granularity", "selective"),
        ("recompute_method", "block"),
        ("recompute_num_layers", True),
        ("cpu_offloading", True),
        ("params_dtype", torch.float32),
    ],
)
def test_no_grad_enclosure_config_guard(field, value):
    r = rank()
    setattr(r.runtime.model[0].decoder.config, field, value)
    assert r._checkpoint_memory_floor(((11, False),)) == (0, 0)


def test_routed_rows_move_only_the_routed_moe_part():
    # A CP2/EP2 real-data trace: the busiest CP rank held 52,480 rows, while
    # HybridEP dispatched 8 x 96,794 pairs per layer across both ranks, so a
    # balanced rank receives 48,397 rows' pairs. Boundaries, the mixer and the
    # shared expert stay on the local rows.
    r = rank()
    r._topology_key = lambda: (1, 1, 2, 1)
    r._moe_gradient_shared_bytes = 8192
    local, routed = 52480, 48397
    retained, workspace = r._checkpoint_memory_floor(((local, True),))
    assert r._checkpoint_memory_floor(((local, True),), None, (local,)) == (
        retained,
        workspace,
    )
    fewer = r._checkpoint_memory_floor(((local, True),), None, (routed,))
    assert fewer[0] == retained
    assert workspace - fewer[1] == (local - routed) * (188416 - 8192)
    # Never more routed rows than local ones.
    assert r._moe_workspace_bytes(
        10, routed_rows=20, checkpoint_grad=True
    ) == r._moe_workspace_bytes(10, checkpoint_grad=True)
    r._moe_gradient_shared_bytes = 188417
    with pytest.raises(ValueError, match="shared-expert"):
        r._moe_workspace_bytes(10, checkpoint_grad=True)


def qwen36_attention(r):
    # Qwen3.6-35B-A3B attention: 16 heads and 2 query groups of 256, gated.
    r._geometry = replace(
        r._geometry, num_attention_heads=16, num_query_groups=2, kv_channels=256
    )
    r._attention_output_gate = True
    return r


@pytest.mark.parametrize(
    "cp,expected",
    # Traces measured 66 KB per token at CP1 and 94 KB at CP2. CP4 reuses the
    # CP2 stage allowance; it is not measured.
    [(1, 2 * (2 * 2048 + 7 * 4096 + 3 * 512)), (2, 95232), (4, 95232)],
)
def test_recomputed_attention_is_priced_beside_the_moe_stage(cp, expected):
    r = qwen36_attention(rank())
    r._topology_key = lambda: (1, 1, cp, 1)
    assert r._recomputed_mixer_bytes_per_token() == expected
    moe = r._moe_workspace_bytes(10, checkpoint_grad=True)
    # Beside the mixer: the recomputed layer's residual and pre-MLP norm rows,
    # and Transformer Engine's cuBLAS workspaces.
    assert (
        r._checkpoint_memory_floor(((10, True),))[1]
        == moe + 10 * (expected + 2 * 2048 * 2) + _TE_CUBLAS_WORKSPACE_BYTES
    )


@pytest.mark.parametrize(
    "cp,gdn_layers,expected",
    [
        # Hybrid: GDN (82 KB) is larger at CP1, CP attention (95 KB) at CP2.
        (1, 30, 81920),
        (2, 30, 95232),
        # GDN only: CP exchanges add a hidden-width and a value-width row.
        (1, 40, 81920),
        (2, 40, 94208),
    ],
)
def test_larger_recomputed_mixer_is_priced(cp, gdn_layers, expected):
    r = qwen36_attention(rank())
    r._geometry = replace(
        r._geometry,
        gdn_key_heads=16,
        gdn_key_head_dim=128,
        gdn_value_heads=32,
        gdn_value_head_dim=128,
    )
    r._gdn_layers = gdn_layers
    r._topology_key = lambda: (1, 1, cp, 1)
    assert r._recomputed_mixer_bytes_per_token() == expected


def test_no_grad_groups_do_not_recompute_a_mixer():
    r = qwen36_attention(rank())
    moe = r._moe_workspace_bytes(10)
    assert r._checkpoint_memory_floor(((10, False),)) == (0, moe + 4 * 10 * 2048 * 2)


def test_ungated_attention_prices_fewer_projections():
    r = qwen36_attention(rank())
    r._attention_output_gate = False
    assert r._recomputed_mixer_bytes_per_token() == 2 * (2 * 2048 + 5 * 4096 + 3 * 512)
    r._topology_key = lambda: (1, 1, 2, 1)
    assert r._recomputed_mixer_bytes_per_token() == 2 * (
        2 * 2048 + 5 * 4096 + 3 * 512 + 3 * 4096 + 2 * 512
    )


@pytest.mark.parametrize("cp", [1, 2])
def test_gdn_width_follows_hidden_key_and_value_separately(cp):
    # Hidden differs from the key width, as in Qwen3.5-27B: CP exchanges
    # carry hidden-width inputs and value-width outputs.
    r = rank()
    r._hidden_size = r.runtime.model[0].decoder.config.hidden_size = 5120
    r._geometry = replace(
        r._geometry,
        gdn_key_heads=16,
        gdn_key_head_dim=128,
        gdn_value_heads=48,
        gdn_value_head_dim=128,
    )
    r._gdn_layers = r._num_layers
    r._topology_key = lambda: (1, 1, cp, 1)
    key, value = 16 * 128, 48 * 128
    # q and k are l2-normalized after expansion to the 48 value heads.
    width = 5120 + 2 * key + 2 * 48 * 128 + 6 * value + 64 * 48
    if cp > 1:
        width += 5120 + value
    assert r._recomputed_mixer_bytes_per_token() == 2 * width
    moe = r._moe_workspace_bytes(7, checkpoint_grad=True)
    assert (
        r._checkpoint_memory_floor(((7, True),))[1]
        == moe + 7 * 2 * (width + 2 * 5120) + _TE_CUBLAS_WORKSPACE_BYTES
    )


@pytest.mark.parametrize("ep,routing", [(1, 3332), (2, 3600)])
def test_moe_state_beside_the_recomputed_stage(ep, routing):
    # Qwen3.6-35B-A3B: 256 experts, 512-wide shared expert. Traces measured
    # 3,332 (EP1) and 3,593 (EP2) bytes of routing state per local token.
    r = rank()
    r._geometry = replace(r._geometry, moe_experts=256, moe_shared_expert_ffn=512)
    r._parallel_shape = replace(r._parallel_shape, ep=ep)
    assert r._moe_checkpoint_state_bytes_per_token() == routing + 3 * 512 * 2
    r._geometry = replace(r._geometry, moe_experts=0)
    assert r._moe_checkpoint_state_bytes_per_token() == 0


def test_te_workspaces_are_growth_until_allocated(monkeypatch):
    gemm = pytest.importorskip("transformer_engine.pytorch.cpp_extensions.gemm")
    r = rank()
    entries = [0]

    class Cached:
        def cache_info(self):
            return SimpleNamespace(currsize=entries[0])

    monkeypatch.setattr(gemm, "get_cublas_workspace", Cached())
    assert r._te_workspace_growth_bytes() == _TE_CUBLAS_WORKSPACE_BYTES
    entries[0] = 1  # Plain GEMM only: the grouped streams are still to come.
    assert r._te_workspace_growth_bytes() == _TE_CUBLAS_WORKSPACE_BYTES
    entries[0] = 2
    assert r._te_workspace_growth_bytes() == 0
