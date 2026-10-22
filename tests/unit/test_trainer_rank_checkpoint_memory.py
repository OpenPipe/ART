"""Conditional checkpoint accounting; scalar/CPU evidence, not a CUDA bound."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import ForwardInput, TrainerRank
from art.trainer_rank._impl import Unset, _ForwardRefusal, _MemoryProfile


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
    assert cost.required == int((out + max(old, 2 * retained + work)) * 1.1)
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
        24 * (188416 + 4 * 2048 * 2),
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
    # Callers pass rows on the most loaded CP rank; the per-row floor matches CP1.
    r = rank()
    single = r._checkpoint_memory_floor((rows,))
    r._topology_key = lambda: (1, 1, cp, 1)
    assert r._checkpoint_memory_floor((rows,)) == single != (0, 0)


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
    limit = 250_000_000
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
        max(3 * 188416, max(11 * 188416, 11 + 1_000_000) + 4 * 11 * 2048 * 2),
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


def dense_rank():
    """Qwen3.8-27B's dense shape: H 5120, F 17408, 64 layers, fused SwiGLU."""
    r = rank()
    config = r.runtime.model[0].config
    config.hidden_size, config.num_layers = 5120, 64
    config.ffn_hidden_size, config.bias_activation_fusion = 17408, True
    decoder = r.runtime.model[0].decoder
    decoder.layers = torch.nn.ModuleList(
        [torch.nn.Linear(1, 1).bfloat16().requires_grad_(False) for _ in range(64)]
    )
    decoder.num_layers_per_pipeline_rank = 64
    r.runtime.provider = SimpleNamespace(
        hidden_size=5120, num_layers=64, ffn_hidden_size=17408
    )
    dense = TrainerRank(r.runtime)
    assert dense._geometry.ffn_hidden_size == 17408
    assert not dense._geometry.moe_experts and dense._mlp_activation_factor == 3
    # These stand-in layers have no MLP; the traced model wraps every FC1.
    assert dense._dense_fc1_adapted is False
    dense._dense_fc1_adapted = True
    return dense


@pytest.mark.parametrize("cp", [2, 4, 8])
@pytest.mark.parametrize("adapter_rank", [8, 768])
@pytest.mark.parametrize("subclass", [False, True])
def test_dense_cp_prices_loaded_lora_products_without_opt_in(
    cp, adapter_rank, subclass
):
    from test_trainer_rank_slot_memory import load_slot

    from art.megatron.lora import LoRA
    from art.trainer_rank._impl import _lora_modules_per_layer

    class DerivedLoRA(LoRA):
        pass

    r = dense_rank()
    r._hidden_size = r.runtime.model[0].config.hidden_size = 32
    r._geometry = replace(
        r._geometry,
        hidden_size=32,
        ffn_hidden_size=64,
        num_attention_heads=1,
        num_query_groups=1,
        kv_channels=32,
    )
    r._mixer_top_gaps = (0, None)
    r._topology_key = lambda: (1, 1, cp, 1)
    r._parallel_shape = replace(r._parallel_shape, cp=cp)
    # q/k/v/output/gate/up/down: the selected rank differs from the template.
    loras = torch.nn.ModuleList(
        [
            (DerivedLoRA if subclass else LoRA)(
                f"target.{i}", 32, 32, 4, 4, torch.bfloat16, torch.device("cpu")
            )
            for i in range(7)
        ]
    )
    r.runtime.model[0].decoder.layers[0].loras = loras
    r._lora_modules_per_layer = _lora_modules_per_layer(r.runtime.model[0])
    if subclass:
        # Use the real loader for inherited owners too; the shared test helper
        # deliberately handles only its original exact-class fixtures.
        from art.megatron.lora import LoRASlotRef

        ref = LoRASlotRef("checkpoint", "selected")
        for owner in loras:
            assert isinstance(owner, LoRA)
            prefix = owner.adapter_model_prefix
            assert owner.load_lora_slot(
                ref,
                {
                    prefix + ".lora_A.weight": torch.zeros(adapter_rank, 32).bfloat16(),
                    prefix + ".lora_B.weight": torch.zeros(32, adapter_rank).bfloat16(),
                },
                requires_grad=True,
            )
    else:
        ref = load_slot(r, "selected", adapter_rank)
    assert r._lora_modules_per_layer == 7
    rows = 65_536
    request = ForwardInput(
        input_tokens=torch.zeros(2 * rows, dtype=torch.long), hidden_states=True
    )
    signature = r._memory_signature_from_requests(
        [request], slot_group_count=1, grad_modes=(True,), slot_groups=((ref, True),)
    )
    assert len(signature.slot_shapes[0][1]) == 7
    assert {shape[2] for shape in signature.slot_shapes[0][1]} == {adapter_rank}
    assert not getattr(signature, "bounded_forward_peak", False)
    groups = ((rows, True),)
    retained, workspace = r._checkpoint_memory_floor(groups)
    # All targets' saved products plus backward factor-product gradients.
    lora_bytes = 2 * rows * 7 * adapter_rank * 2
    assert r._sequence_parallel_lora_floor(
        (retained, workspace), groups, signature
    ) == (retained, workspace + lora_bytes)
    required = r._estimate_required_memory_bytes_from_values(
        packed_tokens=2 * rows,
        logical_tokens=2 * rows,
        output_bytes=2 * rows * 32 * 2,
        signature=signature,
        group_rows=groups,
    )
    assert (required <= 550_000_000) is (adapter_rank == 8)
    # Even the six products measured live by FC1 in Sol's CPU witness exceed
    # that cap at rank 768. The full charge also covers the seventh target.
    assert required >= rows * 6 * adapter_rank * 2
    # The literal original probe omitted slot ranks despite declaring seven
    # targets. That must preserve the old allowance, not imply rank zero.
    missing = replace(signature, slot_shapes=())
    assert r._sequence_parallel_lora_floor((retained, workspace), groups, missing) == (
        retained,
        workspace + retained,
    )
    assert (
        r._estimate_required_memory_bytes_from_values(
            packed_tokens=2 * rows,
            logical_tokens=2 * rows,
            output_bytes=2 * rows * 32 * 2,
            signature=missing,
            group_rows=groups,
        )
        > 550_000_000
    )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize(
    "kind", ["foreign", "custom forward", "malformed factors", "unmarked"]
)
def test_dense_cp_unknown_adapter_inventory_keeps_legacy_allowance(mixed, kind):
    from art.megatron.lora import LoRA
    from art.trainer_rank._impl import _lora_modules_per_layer

    class CustomLoRA(LoRA):
        def forward(self, x, tokens_per_expert=None):
            return super().forward(x, tokens_per_expert)

    r = dense_rank()
    model = r.runtime.model[0]
    model.requires_grad_(False)
    owner = (
        CustomLoRA("custom", 32, 32, 8, 8, torch.bfloat16, torch.device("cpu"))
        if kind == "custom forward"
        else torch.nn.Module()
        if kind in ("foreign", "unmarked")
        else LoRA("bad", 32, 32, 8, 8, torch.bfloat16, torch.device("cpu"))
    )
    if kind == "foreign":
        owner.A_T = torch.nn.Parameter(torch.zeros(32, 512).bfloat16())
        owner.B_T = torch.nn.Parameter(torch.zeros(512, 32).bfloat16())
    elif kind == "malformed factors":
        owner.B_T = torch.nn.Parameter(torch.zeros(9, 32).bfloat16())
    elif kind == "unmarked":
        owner.weight = torch.nn.Parameter(torch.zeros(32, 512).bfloat16())
    model.decoder.layers[0].unknown = owner
    if mixed:
        model.decoder.layers[0].known = LoRA(
            "known", 32, 32, 8, 8, torch.bfloat16, torch.device("cpu")
        )
    r._lora_modules_per_layer = _lora_modules_per_layer(model)
    r._topology_key = lambda: (1, 1, 2, 1)
    r._parallel_shape = replace(r._parallel_shape, cp=2)
    req = ForwardInput(input_tokens=torch.arange(128), hidden_states=True)
    sig = r._memory_signature_from_requests(
        [req], slot_group_count=1, grad_modes=(True,)
    )
    groups = ((64, True),)
    retained, workspace = r._checkpoint_memory_floor(groups)
    # Unknown inventory must restore the old per-layer allowance even when
    # there are zero recognized owners, or other owners have usable ranks.
    assert r._sequence_parallel_lora_floor((retained, workspace), groups, sig)[1] >= (
        workspace + retained
    )
    args = dict(
        packed_tokens=128,
        logical_tokens=128,
        output_bytes=128 * 5120 * 2,
        signature=sig,
        group_rows=groups,
    )
    required = r._estimate_required_memory_bytes_from_values(**args)
    r._dense_fc1_adapted = False
    assert required >= r._estimate_required_memory_bytes_from_values(**args)
    assert not torch.cuda.is_initialized()


def test_glm52_lora_subclass_witness_refuses_1150mb(monkeypatch):
    from art.megatron.glm52.lora import Glm52LoRA
    from art.megatron.lora import LoRA, LoRASlotRef, SharedExpertsLinearFC1LoRA
    from art.trainer_rank._impl import _lora_modules_per_layer

    r = dense_rank()
    model = r.runtime.model[0]
    model.config.hidden_size = 32
    model.config.ffn_hidden_size = 64
    model.config.num_layers = model.decoder.num_layers_per_pipeline_rank = 128
    model.config.num_attention_heads = model.config.num_query_groups = 1
    model.config.kv_channels = 256
    model.config.num_moe_experts = 1
    r.runtime.model_support_handler.is_moe = True
    layers = []
    for index in range(128):

        def lora(label, inputs=32, outputs=32):
            return Glm52LoRA(
                f"{index}.{label}",
                inputs,
                outputs,
                4,
                4,
                torch.bfloat16,
                torch.device("cpu"),
            )

        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        fc1 = SharedExpertsLinearFC1LoRA.__new__(SharedExpertsLinearFC1LoRA)
        torch.nn.Module.__init__(fc1)
        fc1.non_gated = False
        fc1.gate_lora, fc1.up_lora = lora("gate", outputs=64), lora("up", outputs=64)
        layer.mlp.linear_fc1 = fc1
        layer.other_loras = torch.nn.ModuleList(
            [
                lora("q_a", 32, 512),
                lora("kv_a", 32, 576),
                lora("q_b", 512, 576),
                lora("o", 256, 32),
                lora("down", 64, 32),
            ]
        )
        layers.append(layer)
    model.decoder.layers = torch.nn.ModuleList(layers)
    model.embedding = torch.nn.Linear(1, 1).bfloat16().requires_grad_(False)
    r = TrainerRank(r.runtime)
    assert r._dense_fc1_adapted and not r._moe_layers
    monkeypatch.setattr(r, "_topology_key", lambda: (1, 1, 2, 1))
    r._parallel_shape = replace(r._parallel_shape, cp=2)
    selected = LoRASlotRef("checkpoint", "selected")
    for owner in model.modules():
        if isinstance(owner, LoRA):
            prefix = owner.adapter_model_prefix
            assert owner.load_lora_slot(
                selected,
                {
                    prefix + ".lora_A.weight": torch.zeros(
                        512, owner.in_features
                    ).bfloat16(),
                    prefix + ".lora_B.weight": torch.zeros(
                        owner.out_features, 512
                    ).bfloat16(),
                },
                alpha=512,
                requires_grad=True,
            )
    rows = 65_536
    req = ForwardInput(input_tokens=torch.arange(2 * rows), hidden_states=True)
    sig = r._memory_signature_from_requests(
        [req],
        slot_group_count=1,
        grad_modes=(True,),
        slot_groups=((selected, True),),
    )
    required = r._estimate_required_memory_bytes_from_values(
        packed_tokens=2 * rows,
        logical_tokens=2 * rows,
        output_bytes=2 * rows * 32 * 2,
        signature=sig,
        group_rows=((rows, True),),
        slot_refs=(selected,),
    )
    monkeypatch.setattr(r, "_available_memory_bytes", lambda: 1_150_000_000)
    assert not r._memory_check_required(required).fits
    assert _lora_modules_per_layer(model) == 7
    assert (
        sum(len(shape) == 5 and shape[2] == 512 for shape in sig.slot_shapes[0][1])
        == 896
    )
    # Sol's CPU autograd storage witness already exceeds this cap before
    # other activations, outputs, cold scratch and the safety factor.
    assert required >= 1_174_405_120
    assert not torch.cuda.is_initialized()


# Frozen rank-8/all-target Qwen3.8-27B complete lifecycle receipts:
# workstreams/cp-dense/gpu/evidence/timed-v2-cp{2,4,8}/result.rank*.json;
# lc-qual recovered-r2/results/dense-n65536/capacity-cp{2,4}/result.rank*.json.
# Rows are the CPU planner's worst attention/GDN extent, not tokens / CP.
# Admission prices additional allocations; compare resident + demand to the
# measured allocated peak. Reserved peaks are allocator cache, not live bytes.
@pytest.mark.parametrize(
    "cp,tokens,rows,resident,measured,usable",
    [
        (2, 32768, 19456, 55332260352, 74358620160, 59837713613),
        (4, 131072, 54528, 55332869632, 104346683392, 84904971469),
        (8, 131072, 33280, 55332021760, 84065598464, 47617608909),
        (2, 65536, 41472, 55333083648, 95180360704, 63154038375),
        (4, 65536, 24576, 55332046336, 77951084544, 53037246055),
    ],
)
def test_dense_cp_rank8_floor_covers_measured_lifecycles(
    cp, tokens, rows, resident, measured, usable
):
    r = dense_rank()
    r._geometry = replace(
        r._geometry,
        num_attention_heads=24,
        num_query_groups=4,
        kv_channels=256,
        gdn_key_heads=16,
        gdn_value_heads=48,
        gdn_key_head_dim=128,
        gdn_value_head_dim=128,
        gdn_conv_kernel=4,
    )
    r._gdn_layers = 48
    r._attention_output_gate = True
    r._mixer_top_gaps = (0, 1)
    r._lora_modules_per_layer = 8
    r._topology_key = lambda: (1, 1, cp, 1)
    r._parallel_shape = replace(r._parallel_shape, cp=cp)
    req = ForwardInput(input_tokens=torch.arange(tokens), hidden_states=True)
    sig = r._memory_signature_from_requests(
        [req], slot_group_count=1, grad_modes=(True,)
    )
    sig = replace(sig, slot_shapes=((True, ((2, 5120, 8, 8, 5120),) * 8),))
    # A small learned slope must never erase the analytic charge.
    r._memory_profiles[sig] = _MemoryProfile(1, tokens)
    required = r._estimate_required_memory_bytes_from_values(
        packed_tokens=tokens,
        logical_tokens=tokens,
        output_bytes=tokens * 5120 * 2,
        signature=sig,
        group_rows=((rows, True),),
        gdn_segments=1,
    )
    assert resident + required >= measured
    assert required <= usable


# Qwen3.8-27B TP1/CP1 no-grad target-only waves (random init, ART a68fa500,
# 2026-09-29): a cold first wave peaked at 260,789 B per row, warm waves at
# 250,181-250,507; the floor they share is (6F + 4H) x 2 = 249,856.
_DENSE_NO_GRAD_ROW = (6 * 17408 + 4 * 5120) * 2
_COLD_NO_GRAD_ROW = 260_789


def no_grad_wave(r, n):
    return r._estimate_flat_forward(
        [
            ForwardInput(
                input_tokens=torch.arange(n),
                target_tokens=torch.arange(n),
                no_grad=True,
            )
        ],
        checkpoint=None,
    )


def estimate(r, signature, groups, packed, logical=None):
    return r._estimate_required_memory_bytes_from_values(
        packed_tokens=packed,
        output_bytes=0,
        signature=signature,
        logical_tokens=logical,
        group_rows=groups,
    )


@pytest.mark.parametrize("n", [7268, 144228])
def test_cp1_dense_no_grad_floor_covers_each_row(n):
    r = dense_rank()
    values = no_grad_wave(r, n)
    _, _, signature, groups, _ = values
    assert not signature.grad_enabled and signature.topology == (1, 1, 1, 1)
    assert groups == ((n, False),)
    # The 16H per packed token floor admitted 180 KB per row, below the cold
    # peak; the per-row floor, itself below the warm peaks, admits it with the
    # usual 1.1 factor.
    assert estimate(r, signature, groups, n) == int(n * _DENSE_NO_GRAD_ROW * 1.1)
    assert price(r, values).required >= n * _COLD_NO_GRAD_ROW


def test_cp1_dense_no_grad_floor_bounds_profiles():
    r = dense_rank()
    _, _, signature, groups, _ = no_grad_wave(r, 7268)
    floor = estimate(r, signature, groups, 7268)
    r._memory_profiles[signature] = _MemoryProfile(100_000, 7268)
    assert estimate(r, signature, groups, 7268) == floor
    r._memory_profiles[signature] = _MemoryProfile(480_869, 7268)
    assert estimate(r, signature, groups, 7268) > floor


def test_cp1_dense_no_grad_floor_prices_physical_rows_of_the_largest_group():
    r = dense_rank()
    _, _, signature, _, _ = no_grad_wave(r, 64)
    # Shared prefixes add logical rows, not physical ones: beside a low
    # profile, which does grow with logical rows, the floor stays put.
    r._memory_profiles[signature] = _MemoryProfile(1_000, 144228)
    single = estimate(r, signature, ((144228, False),), 144228)
    assert single == int(144228 * _DENSE_NO_GRAD_ROW * 1.1)
    assert estimate(r, signature, ((144228, False),), 144228, 1_500_000) == single
    del r._memory_profiles[signature]
    # Groups run one after another: the largest group's rows bound the stage,
    # and the per-packed-token floor still applies to their sum.
    groups = ((100_000, False), (44_228, False))
    assert estimate(r, signature, groups, 144228) == int(
        max(144228 * 5120 * 2 * 16, 100_000 * _DENSE_NO_GRAD_ROW) * 1.1
    )


@pytest.mark.parametrize("case", ["cp2", "tp2", "gradient", "moe", "unwrapped_fc1"])
def test_dense_no_grad_floor_leaves_other_pricing(case):
    r = dense_rank()
    _, _, signature, _, _ = no_grad_wave(r, 64)
    groups = ((144228, False),)
    if case == "cp2":
        signature = replace(signature, topology=(1, 1, 2, 1))
    elif case == "tp2":
        signature = replace(signature, topology=(1, 2, 1, 1))
    elif case == "gradient":
        signature = replace(signature, grad_enabled=True, grad_modes=(True,))
        groups = ((144228, True),)
    elif case == "moe":
        r._geometry = replace(r._geometry, moe_experts=8)
    else:
        # Attention-only adapters leave FC1 unwrapped: no adapter output or sum.
        r._dense_fc1_adapted = False
    priced = estimate(r, signature, groups, 144228)
    r._geometry = replace(r._geometry, ffn_hidden_size=0)
    assert estimate(r, signature, groups, 144228) == priced
    if case in ("cp2", "unwrapped_fc1"):
        # CP2 ranks hold part of the packed rows and are left to the existing
        # floors (and #986's traced CP2 stage), as is an unwrapped FC1.
        assert priced == int(144228 * 5120 * 2 * 16 * 1.1)


@pytest.mark.parametrize("factor,stage", [(3, 6), (5, 6), (7, 7)])
def test_cp1_dense_no_grad_stage_takes_the_wider_swiglu_live_set(factor, stage):
    r = dense_rank()
    r._mlp_activation_factor = factor
    _, _, signature, groups, _ = no_grad_wave(r, 4096)
    assert estimate(r, signature, groups, 4096) == int(
        4096 * (stage * 17408 + 4 * 5120) * 2 * 1.1
    )


@pytest.mark.parametrize("gradient_first", [False, True])
def test_dense_mixed_plan_keeps_the_no_grad_stage(gradient_first):
    r = dense_rank()
    gradient, reference = requests(1, 10_000)
    req = [gradient, reference] if gradient_first else [reference, gradient]
    reference_cost = r._plan_cost(r._plan_flat_forward([reference])).required
    assert reference_cost >= int(10_000 * _DENSE_NO_GRAD_ROW * 1.1)
    mixed = r._plan_flat_forward(req)
    mixed_cost = r._plan_cost(mixed).required
    # One gradient row cannot make the no-grad group's stage cheaper.
    assert mixed_cost >= reference_cost
    assert r._memory_check(mixed).estimated_required_bytes == mixed_cost
    assert (
        r._split_chunk_lower_cost(
            req, tuple(x.input_tokens for x in req), checkpoint=Unset
        ).required
        == mixed_cost
    )


def test_dense_fc1_adapted_needs_every_layer_wrapped():
    from art.megatron.lora import SharedExpertsLinearFC1LoRA
    from art.trainer_rank._impl import _dense_fc1_adapted

    wrapped = SharedExpertsLinearFC1LoRA.__new__(SharedExpertsLinearFC1LoRA)
    wrapped.__dict__["non_gated"] = False
    non_gated = SharedExpertsLinearFC1LoRA.__new__(SharedExpertsLinearFC1LoRA)
    non_gated.__dict__["non_gated"] = True

    def model(*fc1s: object) -> Any:
        layers = [SimpleNamespace(mlp=SimpleNamespace(linear_fc1=f)) for f in fc1s]
        return SimpleNamespace(
            _preprocess=lambda: None, decoder=SimpleNamespace(layers=layers)
        )

    assert _dense_fc1_adapted(model(wrapped, wrapped)) is True
    assert _dense_fc1_adapted(model(wrapped, torch.nn.Linear(1, 1))) is False
    # A non-gated FC1 outputs F, not 2F: not the traced stage.
    assert _dense_fc1_adapted(model(non_gated, non_gated)) is False
    assert _dense_fc1_adapted(model()) is False
    assert _dense_fc1_adapted(cast(Any, SimpleNamespace())) is False


@pytest.mark.parametrize("pending_per_layer", [20_000, 2_000_000])
@pytest.mark.parametrize("profiled", [False, True])
def test_dense_no_grad_stage_and_pending_backward_are_separate_peaks(
    monkeypatch, pending_per_layer, profiled
):
    from art.megatron.lora import LoRASlotRef
    from art.trainer_rank._impl import _COLD_RECOMPUTE_TRANSIENT_BYTES as COLD

    r = dense_rank()
    plan = r._plan_flat_forward(requests(1, 400))
    if profiled:
        r._memory_profiles[plan.signature] = _MemoryProfile(1, 401)
    monkeypatch.setattr(
        r,
        "_pending_adapter_gradient_bytes",
        lambda refs: (pending_per_layer,) * 64 + (0,) if tuple(refs) else (),
    )
    # The first-layer backward holds all pending gradients and one boundary;
    # the existing floor already holds all 64 boundaries. A larger external
    # retained floor must remain beside whichever of the two stages peaks.
    retained = 64 * 5120 * 2
    adapter_extra = 64 * pending_per_layer - 63 * 5120 * 2
    backward = retained + adapter_extra + (0 if profiled else COLD)
    no_grad_stage = 400 * _DENSE_NO_GRAD_ROW
    actual = r._estimate_required_memory_bytes_from_values(
        packed_tokens=401,
        output_bytes=0,
        signature=plan.signature,
        group_rows=((1, True), (400, False)),
        slot_refs=(LoRASlotRef("checkpoint", "policy"), None),
        checkpoint_memory=(retained, 0),
        checkpoint_floor=(1_000_000, 0),
    )
    assert actual == int((1_000_000 + max(backward, no_grad_stage)) * 1.1)
    assert (backward > no_grad_stage) == (pending_per_layer == 2_000_000)
    assert actual < int((1_000_000 + backward + no_grad_stage) * 1.1)


@pytest.mark.parametrize("degree", [2, 4, 8])
def test_checkpoint_storage_partition_has_separate_profile_and_full_workspace(
    monkeypatch, degree
):
    from art.megatron.context_parallel.checkpoint import (
        current_checkpoint_partition,
        shard_saved_activations,
    )

    r = rank()
    monkeypatch.setattr(r, "_topology_key", lambda: (degree, 1, 1, 1))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: degree)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    req = [
        ForwardInput(input_tokens=torch.arange(1024), hidden_states=True, no_grad=False)
    ]
    plain = r._estimate_flat_forward(req)
    original, _ = r._checkpoint_memory_floor(plain[3])
    with shard_saved_activations(object(), shape=(1024, 1, 2048), key="root"):
        shared = r._estimate_flat_forward(req)
        saved, workspace = r._checkpoint_memory_floor(shared[3])
        assert saved * degree == original
        assert workspace >= r._sequence_parallel_workspace_bytes(1024) + 1024 * 2048 * 2
        assert price(r, shared).checkpoint_input_gradient == 1024 * 2048 * 2
        assert shared[2] != plain[2]
        assert shared[2].checkpoint_partition == (degree, (1024, 1, 2048))
        r._memory_profiles[shared[2]] = _MemoryProfile(
            bytes_per_token=1, packed_tokens=1024, logical_per_packed=1
        )
        assert plain[2] not in r._memory_profiles
    assert current_checkpoint_partition() is None
    assert r._checkpoint_memory_floor(plain[3])[0] == original


@pytest.mark.parametrize("unsupported", ["length", "multiple", "recompute"])
def test_checkpoint_storage_partition_rejects_unmatched_roots(monkeypatch, unsupported):
    from art.megatron.context_parallel.checkpoint import (
        current_checkpoint_partition,
        shard_saved_activations,
    )

    r = rank()
    monkeypatch.setattr(r, "_topology_key", lambda: (2, 1, 1, 1))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    req = [
        ForwardInput(
            input_tokens=torch.arange(1025 if unsupported == "length" else 1024),
            hidden_states=True,
            no_grad=False,
        )
    ]
    if unsupported == "multiple":
        req *= 2
    if unsupported == "recompute":
        r.runtime.model[0].decoder.config.recompute_num_layers = 2
    with pytest.raises(ValueError, match="one CP1 uniform-recompute root"):
        with shard_saved_activations(object(), shape=(1024, 1, 2048), key="root"):
            r._estimate_flat_forward(req)
    assert current_checkpoint_partition() is None


def test_checkpoint_storage_partition_rejects_moe_models(monkeypatch):
    from art.megatron.context_parallel.checkpoint import (
        current_checkpoint_partition,
        shard_saved_activations,
    )

    r = rank()
    monkeypatch.setattr(r, "_topology_key", lambda: (2, 1, 1, 1))
    monkeypatch.setattr(r, "_moe_layers", (0,))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    req = [
        ForwardInput(input_tokens=torch.arange(1024), hidden_states=True, no_grad=False)
    ]
    with pytest.raises(ValueError, match="dense models only"):
        with shard_saved_activations(object(), shape=(1024, 1, 2048), key="root"):
            r._estimate_flat_forward(req)
    assert current_checkpoint_partition() is None


@pytest.mark.parametrize("element_bytes", [2, 4, 8])
def test_checkpoint_partition_uses_activation_element_size(monkeypatch, element_bytes):
    from art.megatron.context_parallel.checkpoint import shard_saved_activations
    from art.trainer_rank import _memory

    r = rank()
    r._param_dtype_size = element_bytes
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 4)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    shape = (3, 1, 7)  # Deliberately uneven, with a known padded gather size.
    monkeypatch.setattr(r, "_sequence_parallel_workspace_bytes", lambda rows: 0)
    monkeypatch.setattr(r, "_moe_workspace_bytes", lambda *a, **k: 0)
    with shard_saved_activations(object(), shape=shape, key="dtype"):
        retained, workspace = _memory._checkpoint_floor_from_facts(
            r, ((3, True),), None, 0, 5
        )
        assert retained == 6 * element_bytes * 5
        assert workspace == 6 * 4 * element_bytes
        assert (
            r._checkpoint_input_gradient_bytes(((3, True),), retained)
            == 21 * element_bytes
        )


def test_split_memory_key_separates_checkpoint_partitions(monkeypatch):
    from art.megatron.context_parallel.checkpoint import shard_saved_activations
    from art.trainer_rank._impl import _SplitForwardPlan

    r = rank()
    monkeypatch.setattr(r, "_topology_key", lambda: (2, 1, 1, 1))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    inputs = [
        ForwardInput(input_tokens=torch.arange(17), hidden_states=True, no_grad=False)
    ]
    plain = r._plan_flat_forward(inputs)
    with shard_saved_activations(object(), shape=(17, 1, 2048), key="root"):
        sharded = r._plan_flat_forward(inputs)
    # Keep all plan fields fixed, isolating the signature's partition identity.
    a = _SplitForwardPlan((plain,), ((0,),), 1)
    b = replace(a, subforwards=(replace(plain, signature=sharded.signature),))
    assert r._split_memory_key(a) != r._split_memory_key(b)
    changed = replace(sharded.signature, checkpoint_partition=(4, (17, 1, 2048)))
    c = replace(a, subforwards=(replace(plain, signature=changed),))
    assert r._split_memory_key(b) != r._split_memory_key(c)


def test_megatron_checkpoint_shards_one_boundary_per_layer(monkeypatch):
    from contextlib import nullcontext

    from megatron.core.tensor_parallel import random

    from art.megatron.context_parallel.checkpoint import shard_saved_activations

    # Exercise Megatron's actual checkpoint autograd function on CPU. Only its
    # CUDA RNG helpers are replaced; tensor saves and checkpoint calls are real.
    monkeypatch.setattr(random, "_get_all_rng_states", lambda: ())
    monkeypatch.setattr(random, "_set_all_rng_states", lambda *a: None)
    monkeypatch.setattr(random, "_fork_rng", nullcontext)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    model = torch.nn.ModuleList([torch.nn.Linear(7, 7) for _ in range(3)])
    x = torch.randn(5, 1, 7, requires_grad=True)
    with shard_saved_activations(object(), shape=tuple(x.shape), key="mcore") as stats:
        value = x.sin()  # Ordinary saves of this shape must stay local.
        for layer in model:
            value = random.checkpoint(layer, False, value)
    assert value.shape == x.shape
    assert stats.tensors == len(model)
    assert stats.original_bytes == len(model) * x.numel() * x.element_size()
