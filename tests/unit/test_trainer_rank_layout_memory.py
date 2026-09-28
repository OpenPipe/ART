"""CP2 layout-aware recompute pricing: per-rank ledgers and gates, CPU only."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_trainer_rank_checkpoint_memory import rank
import torch

from art.trainer_rank import ForwardInput
from art.trainer_rank._impl import _TE_CUBLAS_WORKSPACE_BYTES, Unset, _GroupLayout

H = 2048 * 2


def qwen36(r):
    """Qwen3.6-35B-A3B's 3:1 GDN/attention pattern and mixer geometry at CP2."""
    r._topology_key = lambda: (1, 1, 2, 1)
    r._geometry = replace(
        r._geometry,
        num_attention_heads=16,
        num_query_groups=2,
        kv_channels=256,
        gdn_key_heads=16,
        gdn_key_head_dim=128,
        gdn_value_heads=32,
        gdn_value_head_dim=128,
    )
    r._attention_output_gate = True
    r._gdn_layers = 30
    for index, layer in enumerate(r.runtime.model[0].decoder.layers):
        is_gdn = index % 4 != 3
        after_gdn = index > 0 and (index - 1) % 4 != 3
        layer._art_gdn_island_boundary = SimpleNamespace(
            is_gdn=is_gdn, input_layout="gdn" if is_gdn and after_gdn else "attention"
        )
    return r


def test_boundaries_follow_each_layer_inputs_layout():
    # Traced real-data CP2 ranks: 20 inputs in each layout, and boundaries of
    # 8.248 and 7.612 GB. The busiest attention rank is not the busiest overall.
    r = qwen36(rank())
    layers = r.runtime.model[0].decoder.layers
    layout = _GroupLayout((52480, 44314), (48194, 48600), (0, 0))
    retained, _ = r._layout_checkpoint_floor(layers, (None,), (48397,), (layout,))
    assert retained == H * (20 * 52480 + 20 * 48194)
    assert H * (20 * 44314 + 20 * 48600) < retained < 40 * 52480 * H


def test_largest_rank_total_not_a_sum_of_rank_maxima():
    r = qwen36(rank())
    layers = r.runtime.model[0].decoder.layers
    widths = r._recomputed_mixer_widths(stage_buffers=False)
    assert widths == {"attention": 2 * (2 * 2048 + 7 * 4096 + 3 * 512), "gdn": 94208}
    # Rank 1's two full-query stages keep 3.08 GB against rank 0's 0.97 GB.
    layout = _GroupLayout((52480, 44314), (48194, 48600), (970_720_000, 3_079_000_000))
    retained, workspace = r._layout_checkpoint_floor(
        layers, (None,), (48397,), (layout,)
    )

    def total(rank_index):
        attention = layout.attention_rows[rank_index]
        assert layout.gdn_rows is not None
        gdn = layout.gdn_rows[rank_index]
        ledger = H * (20 * attention + 20 * gdn)
        stages = (
            attention * (widths["attention"] + 2 * H)
            + layout.attention_retained[rank_index]
            + r._moe_workspace_bytes(
                attention, routed_rows=48397, checkpoint_grad=True
            ),
            gdn * (widths["gdn"] + 2 * H)
            + r._moe_workspace_bytes(gdn, routed_rows=48397, checkpoint_grad=True),
        )
        return ledger + max(stages)

    assert retained + workspace == max(total(0), total(1)) + _TE_CUBLAS_WORKSPACE_BYTES
    # Rank 1 has fewer rows but is the largest total: its attention runs two
    # full-query stages. Pricing it on its own rows is about neutral against
    # the busiest-rank floor, which over-counts boundaries and GDN instead.
    assert total(1) > total(0)
    busiest = sum(
        r._checkpoint_memory_floor(((52480, True),), None, routed_rows=(48397,))
    )
    assert abs(retained + workspace - busiest) < 0.01 * busiest


def test_routed_rows_above_a_ranks_own_rows_are_all_priced():
    r = qwen36(rank())
    layers = r.runtime.model[0].decoder.layers
    few = _GroupLayout((100, 100), (100, 100), (0, 0))
    many = _GroupLayout((100, 100), (100, 100), (0, 0))
    low = sum(r._layout_checkpoint_floor(layers, (None,), (100,), (few,)))
    high = sum(r._layout_checkpoint_floor(layers, (None,), (400,), (many,)))
    assert high - low == 300 * 188416


def test_no_grad_groups_keep_busiest_rank_pricing():
    r = qwen36(rank())
    layout = _GroupLayout((10, 8), (9, 9), (0, 0))
    groups = ((10, True), (12, False))
    assert r._checkpoint_memory_floor(
        groups, None, routed_rows=(9, 12), layouts=(layout, layout)
    ) == r._checkpoint_memory_floor(groups, None, routed_rows=(9, 12))


def _plan(r, *, no_grad=False):
    plan = r._plan_flat_forward(
        [
            ForwardInput(
                input_tokens=torch.arange(64), hidden_states=True, no_grad=no_grad
            )
        ]
    )
    return replace(plan, signature=replace(plan.signature, topology=(1, 1, 2, 1)))


@pytest.mark.parametrize(
    "topology", [(1, 1, 1, 1), (1, 2, 2, 1), (1, 1, 4, 1), (1, 1, 2, 2)]
)
def test_layout_pricing_is_cp2_tp1_pp1_only(topology):
    r = qwen36(rank())
    plan = _plan(r)
    plan = replace(plan, signature=replace(plan.signature, topology=topology))
    assert r._plan_group_layouts(plan) is None


def test_layout_pricing_needs_gradient_groups_and_art_cp_attention():
    r = qwen36(rank())
    assert r._plan_group_layouts(_plan(r, no_grad=True)) is None
    # The stub decoder's attention layers carry no ART CP core attention.
    assert r._plan_group_layouts(_plan(r)) is None
    # A GDN model without island boundaries is not modeled either.
    for layer in r.runtime.model[0].decoder.layers:
        del layer._art_gdn_island_boundary
    assert r._plan_group_layouts(_plan(r)) is None


def test_plan_cost_and_admission_use_the_same_layouts(monkeypatch):
    r = qwen36(rank())
    plan = _plan(r)
    layout = _GroupLayout((40, 24), (32, 32), (1_000_000, 3_000_000))
    monkeypatch.setattr(r, "_plan_group_layouts", lambda plan: (layout,))
    monkeypatch.setattr(r, "_plan_group_rows", lambda plan: ((40, True),))
    monkeypatch.setattr(r, "_plan_group_routed_rows", lambda plan: (32,))
    monkeypatch.setattr(r, "_plan_hybridep_growth_bytes", lambda plan: 0)
    monkeypatch.setattr(r, "_plan_retained_tokens", lambda plan: 32)
    cost = r._plan_cost(plan)
    # Admission adds CP output coexistence the plan cost leaves out (as it did
    # before layouts); the layouts themselves enter both the same way.
    gap = r._memory_check(plan).estimated_required_bytes - cost.required
    monkeypatch.setattr(r, "_plan_group_layouts", lambda plan: None)
    assert gap == r._memory_check(plan).estimated_required_bytes - (
        r._plan_cost(plan).required
    )
    monkeypatch.setattr(r, "_plan_group_layouts", lambda plan: (layout,))
    retained, _ = r._layout_checkpoint_floor(
        r.runtime.model[0].decoder.layers,
        (None,),
        r._plan_group_routed_rows(plan),
        (layout,),
    )
    assert cost.checkpoint_retained == plan.output_bytes + retained


def art_cp(r, monkeypatch):
    """Qwen3.6 at CP2 with ART's CP core attention and a CPU planning config."""
    from art.megatron.context_parallel.core_attention import (
        ArtContextParallelCoreAttention,
    )
    from art.megatron.context_parallel.types import ParallelTopology

    r = qwen36(r)
    for index, layer in enumerate(r.runtime.model[0].decoder.layers):
        if index % 4 == 3:
            core = ArtContextParallelCoreAttention.__new__(
                ArtContextParallelCoreAttention
            )
            torch.nn.Module.__init__(core)
            core.softmax_offset = None
            layer.self_attention = torch.nn.Module()
            layer.self_attention.core_attention = core
    provider = r.runtime.provider
    provider.kv_channels = 256
    provider.ffn_hidden_size = 8192
    provider.linear_num_key_heads = 16
    provider.linear_num_value_heads = 32
    provider.linear_key_head_dim = 128
    provider.linear_value_head_dim = 128
    provider.params_dtype = torch.bfloat16
    r.runtime.model_support_handler = SimpleNamespace(
        build_gdn_execution_spec=True,
        context_parallel_workload_profile=lambda provider: None,
    )
    monkeypatch.setattr(r, "_topology", lambda: ParallelTopology(tp=1, cp=2))
    return r


def _requests(lengths=(900, 700, 500)):
    start, requests = 0, []
    for n in lengths:
        requests.append(
            ForwardInput(
                input_tokens=torch.arange(start, start + n), hidden_states=True
            )
        )
        start += n
    return requests


def test_rank_layouts_are_the_executors_plans(monkeypatch):
    from art.megatron.context_parallel.runtime import context_parallel_rank_layouts
    from art.megatron.context_parallel.types import (
        ContextParallelConfig,
        ParallelTopology,
    )
    from art.megatron.prefix_tree_packing import prefix_tree_pack

    packed = prefix_tree_pack([r.input_tokens for r in _requests()], max_depth=1)
    attention, gdn, plans = context_parallel_rank_layouts(
        group_ids=packed.group_ids,
        parent_ids=packed.parent_ids,
        topology=ParallelTopology(tp=1, cp=2),
        config=ContextParallelConfig(),
        original_seq_len=int(packed.tokens.shape[1]),
        build_gdn_execution_spec=True,
    )
    total = int(packed.tokens.numel())
    assert sum(attention) == total and gdn is not None and sum(gdn) == total
    # The ledger's attention rows and the mirror's own rows are the same count.
    assert attention == tuple(plan.local_valid_lengths[0] for plan in plans)


def test_gated_plans_price_every_rank_from_its_stage_plan(monkeypatch):
    from art.megatron.context_parallel.executor import retained_stage_record_bytes
    from art.megatron.context_parallel.runtime import context_parallel_rank_layouts
    from art.megatron.context_parallel.types import ParallelTopology
    from art.megatron.training.microbatches import (
        _context_parallel_config_for_provider,
    )

    r = art_cp(rank(), monkeypatch)
    plan = _plan_with(r, _requests())
    (layout,) = r._plan_group_layouts(plan)
    assert sum(layout.attention_rows) == plan.packed_tokens
    assert layout.gdn_rows is not None and sum(layout.gdn_rows) == plan.packed_tokens
    # Each rank's retention is the executor mirror over that rank's own plan.
    (group,) = plan.groups
    _, _, rank_plans = context_parallel_rank_layouts(
        group_ids=group.packed.group_ids,
        parent_ids=group.packed.parent_ids,
        topology=ParallelTopology(tp=1, cp=2),
        config=_context_parallel_config_for_provider(
            r.runtime.provider, r.device, r.runtime.model_support_handler
        ),
        original_seq_len=int(group.packed.tokens.shape[1]),
        build_gdn_execution_spec=True,
    )
    assert layout.attention_retained == tuple(
        retained_stage_record_bytes(
            rank_plan,
            q_heads=16,
            kv_heads=2,
            head_dim=256,
            value_head_dim=256,
            element_size=2,
            block_size=(128, 128),  # the CPU device's flex block
        )
        for rank_plan in rank_plans
    )
    assert all(retained > 0 for retained in layout.attention_retained)


def test_softmax_offset_leaves_the_busiest_rank_floor(monkeypatch):
    r = art_cp(rank(), monkeypatch)
    plan = _plan_with(r, _requests())
    assert r._plan_group_layouts(plan) is not None
    for layer in r.runtime.model[0].decoder.layers:
        core = getattr(getattr(layer, "self_attention", None), "core_attention", None)
        if core is not None:
            core.softmax_offset = torch.zeros(16)
    assert r._plan_group_layouts(plan) is None


@pytest.mark.parametrize(
    "lengths",
    [(2048, 1536, 1024, 512), (4099, 3, 5, 7), (1, 2, 3, 4, 5, 6, 7), (8191,)],
)
def test_split_lower_bound_stays_below_the_layout_cost(monkeypatch, lengths):
    # Even and skewed CP splits, odd row counts, and a single long sequence.
    r = art_cp(rank(), monkeypatch)
    requests = _requests(lengths)
    plan = _plan_with(r, requests)
    lower = r._split_chunk_lower_cost(
        requests, tuple(item.input_tokens for item in requests), checkpoint=Unset
    )
    assert lower.required <= r._plan_cost(plan).required
    # It prices even-share layouts at the least attention state.
    assert r._layout_pricing_supported((1, 1, 2, 1), gradient_groups=True)


def _plan_with(r, requests):
    return r._plan_flat_forward(requests)


def _pending_adapter_rank(monkeypatch, layout, rows, per_layer=300 * H):
    from art.megatron.lora import LoRASlotRef

    r = qwen36(rank())
    monkeypatch.setattr(r, "_plan_group_layouts", lambda plan: (layout,))
    monkeypatch.setattr(r, "_plan_group_rows", lambda plan: ((rows, True),))
    monkeypatch.setattr(r, "_plan_group_routed_rows", lambda plan: (rows,))
    monkeypatch.setattr(r, "_plan_hybridep_growth_bytes", lambda plan: 0)
    monkeypatch.setattr(r, "_plan_retained_tokens", lambda plan: rows)
    # The plan's gradient group trains the policy slot.
    policy = LoRASlotRef("checkpoint", "policy")
    groups = r._checkpoint_gradient_groups
    monkeypatch.setattr(
        r,
        "_checkpoint_gradient_groups",
        lambda group_rows, slot_refs: tuple(
            (policy, boundaries) for _, boundaries in groups(group_rows, slot_refs)
        ),
    )
    pending = (per_layer,) * 40 + (0,)
    monkeypatch.setattr(
        r,
        "_pending_adapter_gradient_bytes",
        lambda refs: pending if tuple(refs) else (),
    )
    layers = r.runtime.model[0].decoder.layers
    gdn_inputs = [
        layer._art_gdn_island_boundary.input_layout == "gdn" for layer in layers
    ]

    def boundaries(rank_index):
        assert layout.gdn_rows is not None
        return [
            H
            * max(1, (layout.gdn_rows if is_gdn else layout.attention_rows)[rank_index])
            for is_gdn in gdn_inputs
        ]

    def extra(saved):
        return max(
            0,
            *(
                sum(pending[i:40]) + pending[40] - sum(saved[i + 1 :])
                for i in range(40)
            ),
        )

    floor = r._checkpoint_memory_floor(
        ((rows, True),), None, routed_rows=(rows,), layouts=(layout,)
    )
    floors = r._layout_checkpoint_rank_floors(layers, (None,), (rows,), (layout,))
    return r, boundaries, extra, floor, floors


def test_adapter_gradients_meet_each_ranks_own_layer_boundaries(monkeypatch):
    layout = _GroupLayout((400, 240), (160, 320), (0, 0))
    r, boundaries, extra, (retained, workspace), floors = _pending_adapter_rank(
        monkeypatch, layout, 400
    )
    assert r._layout_layer_boundaries((layout,)) == (
        (tuple(boundaries(0)),),
        (tuple(boundaries(1)),),
    )
    # Every rank's boundaries sum to its own floor's.
    assert [sum(boundaries(i)) for i in (0, 1)] == [floor[0] for floor in floors]
    assert retained == max(floor[0] for floor in floors)
    # Each rank releases its own boundaries in layer order; each extra sits on
    # that rank's own floor, bounded by the floor's workspace.
    expected = (
        max(
            floor[0] + max(floor[1], workspace) + extra(boundaries(i))
            for i, floor in enumerate(floors)
        )
        - retained
        - workspace
    )
    assert r._plan_cost(_plan(r)).checkpoint_adapter_gradient == expected > 0
    # An even share of the busiest rank's boundaries would misplace the peak.
    assert extra([retained // 40] * 40) != expected


def test_a_rank_without_gdn_rows_pairs_its_extra_with_its_own_floor(monkeypatch):
    # A single short sequence at CP2: every GDN row on rank 0 (Qwen3.6 2,047
    # tokens), with about 24 MB of expert LoRA gradients per layer.
    layout = _GroupLayout((1536, 511), (2047, 0), (0, 0))
    r, boundaries, extra, (retained, workspace), floors = _pending_adapter_rank(
        monkeypatch, layout, 2047, per_layer=6000 * H
    )
    extras = [extra(boundaries(i)) for i in (0, 1)]
    # Rank 1 releases almost nothing, so its extra is almost every gradient...
    assert extras[1] > extras[0] and extras[1] > 6000 * H * 38
    cost = r._plan_cost(_plan(r)).checkpoint_adapter_gradient
    # ...but it sits on rank 1's much smaller floor, not on rank 0's.
    assert cost < extras[1]
    # Every rank's own floor plus its own extra stays within the price.
    for (rank_retained, rank_workspace), rank_extra in zip(floors, extras):
        assert rank_retained + rank_workspace + rank_extra <= (
            retained + workspace + cost
        )
    assert cost >= extras[0]


def test_layout_gradient_groups_run_one_after_another_on_each_rank(monkeypatch):
    from test_trainer_rank_adapter_gradient_memory import sequential_oracle

    from art.megatron.lora import LoRASlotRef

    r = qwen36(rank())
    # A short policy sequence beside a longer sequence of another slot.
    layouts = (
        _GroupLayout((1536, 511), (2047, 0), (0, 0)),
        _GroupLayout((400, 240), (160, 320), (0, 0)),
    )
    group_rows = ((2047, True), (640, True))
    routed = (2047, 640)
    slots = (LoRASlotRef("checkpoint", "policy"), LoRASlotRef("checkpoint", "other"))
    groups = r._checkpoint_gradient_groups
    monkeypatch.setattr(
        r,
        "_checkpoint_gradient_groups",
        lambda group_rows, slot_refs: tuple(
            (slot, boundaries)
            for slot, (_, boundaries) in zip(slots, groups(group_rows, slot_refs))
        ),
    )
    pending = {
        (slots[0],): (6000 * H,) * 40 + (0,),
        (slots[1],): (100 * H,) * 40 + (5 * H,),
    }
    monkeypatch.setattr(
        r, "_pending_adapter_gradient_bytes", lambda refs: pending.get(tuple(refs), ())
    )
    floor = r._checkpoint_memory_floor(
        group_rows, None, routed_rows=routed, layouts=layouts
    )
    layers = r.runtime.model[0].decoder.layers
    floors = r._layout_checkpoint_rank_floors(layers, (None, None), routed, layouts)
    per_rank = r._layout_layer_boundaries(layouts)
    # Every rank's groups sum to its own floor's boundaries.
    assert [sum(map(sum, rank_groups)) for rank_groups in per_rank] == [
        rank_floor[0] for rank_floor in floors
    ]
    # On each rank, one group's backward runs after the other's, in the worst
    # order; each rank's extra sits on its own floor.
    extras = [
        sequential_oracle(
            [(pending[(slot,)], list(b)) for slot, b in zip(slots, rank_groups)]
        )
        for rank_groups in per_rank
    ]
    expected = max(
        0,
        max(
            rank_retained + max(rank_workspace, floor[1]) + extra
            for (rank_retained, rank_workspace), extra in zip(floors, extras)
        )
        - sum(floor),
    )
    assert (
        r._checkpoint_adapter_gradient_extra(floor, group_rows, None, routed, layouts)
        == expected
        > 0
    )
