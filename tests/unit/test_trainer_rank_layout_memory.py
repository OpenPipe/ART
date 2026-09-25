"""CP2 layout-aware recompute pricing: per-rank ledgers and gates, CPU only."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_trainer_rank_checkpoint_memory import rank
import torch

from art.trainer_rank import ForwardInput
from art.trainer_rank._impl import _TE_CUBLAS_WORKSPACE_BYTES, _GroupLayout

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
    busiest = sum(r._checkpoint_memory_floor(((52480, True),), None, (48397,)))
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
        groups, None, (9, 12), (layout, layout)
    ) == r._checkpoint_memory_floor(groups, None, (9, 12))


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
