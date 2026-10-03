"""The explicit full-recompute checkpoint floor under TP2/TP4 sequence parallelism.

CPU admission math, not a bound. Allocator traces of dense Qwen3.8-27B (64
layers, sequence parallel) at TP4 and TP2, cold and warm, from 20,816 to 66,284
rows and one to four GDN segments, put every peak at each rank's boundary
shards plus the recomputed layer's workspace, term by term; the floor prices
exactly those terms. Other shapes keep today's pricing.
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import TrainerRank
from art.trainer_rank._impl import _COLD_RECOMPUTE_TRANSIENT_BYTES as COLD
from art.trainer_rank._impl import _MemoryProfile, _MemorySignature

H, F, LAYERS = 5120, 17408, 64
TP4 = (1, 4, 1, 1)
# The traced 062 wave: one request, 25,727 tokens padded to 25,728 rows.
ROWS, OUTPUT = 25_728, 102_908
# One segment's initial and final fp32 states over 12 local value heads, plus
# conv history over 3 taps, in the recomputed GDN layer.
SEGMENT = (4 * 12 * 128 * 128 + 2 * (2 * 4 * 128 + 12 * 128) * 3) * 2
TP2 = (1, 2, 1, 1)
# Every traced TP x SP call with a comparable observation: topology, rows,
# logical rows, GDN segments, output bytes, whether the signature was
# profiled, and the peak above the call's baseline (identical on every rank).
TRACED = (
    pytest.param(TP4, 25_728, 25_727, 1, 102_908, False, 7_129_740_288, id="tp4-cold"),
    pytest.param(TP4, 25_728, 25_727, 1, 102_908, True, 6_894_956_544, id="tp4-warm"),
    pytest.param(
        TP2, 23_878, 23_878, 1, 95_512, False, 12_645_404_160, id="tp2-23878-cold"
    ),
    pytest.param(
        TP2, 23_878, 23_878, 1, 95_512, False, 12_544_440_832, id="tp2-23878-prod"
    ),
    pytest.param(
        TP2, 23_878, 23_878, 1, 95_512, True, 12_298_095_616, id="tp2-23878-warm"
    ),
    pytest.param(
        TP2, 20_816, 20_816, 1, 83_264, False, 10_935_935_488, id="tp2-20816-prod"
    ),
    pytest.param(
        TP2, 20_816, 20_816, 1, 83_264, True, 10_721_021_440, id="tp2-20816-warm"
    ),
    pytest.param(
        TP2, 40_000, 40_000, 2, 160_000, True, 20_601_088_000, id="tp2-40000-warm"
    ),
    pytest.param(
        TP2, 66_284, 68_158, 4, 272_632, False, 34_932_728_320, id="tp2-66284-cold"
    ),
    pytest.param(
        TP2, 66_284, 68_158, 4, 272_632, True, 34_141_831_680, id="tp2-66284-warm"
    ),
    pytest.param(
        TP2, 66_284, 68_158, 4, 272_632, True, 34_616_171_520, id="tp2-66284-prod"
    ),
)


def workspace(rows, tp, ffn=F, mixer=4 * 16 * 128 + 8 * 48 * 128):
    """The traced recompute workspace: gathered 2H + 6F/TP + mixer/TP, then
    3.25H of norms per sharded row."""
    gathered = rows * (2 * H + -(-6 * ffn // tp) + -(-mixer // tp))
    return (gathered + -(-rows // tp) * 13 * H // 4) * 2


def tp_rank(layers=LAYERS, *, ffn=F, topology=TP4, sequence_parallel=True, **config):
    from megatron.core.transformer.transformer_block import TransformerBlock

    block = TransformerBlock.__new__(TransformerBlock)
    torch.nn.Module.__init__(block)
    block.config = SimpleNamespace(
        hidden_size=H,
        num_layers=layers,
        padded_vocab_size=32,
        params_dtype=torch.bfloat16,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=1,
        distribute_saved_activations=False,
        sequence_parallel=sequence_parallel,
        fp32_residual_connection=False,
        cpu_offloading=False,
        cuda_graph_impl="none",
        fp8=None,
        fp4=None,
        **config,
    )
    block.layers = torch.nn.ModuleList(
        [torch.nn.Linear(1, 1).bfloat16() for _ in range(layers)]
    )
    block.num_layers_per_pipeline_rank = layers
    model: Any = torch.nn.Module()
    model.config = block.config
    model.decoder = block
    model._preprocess = lambda: None
    r: Any = TrainerRank(
        cast(
            Any,
            SimpleNamespace(
                model=[model],
                optimizer=None,
                provider=SimpleNamespace(hidden_size=H, num_layers=layers),
                model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
            ),
        )
    )
    # Qwen3.8-27B: gated attention every fourth layer, GDN otherwise.
    r._geometry = replace(
        r._geometry,
        hidden_size=H,
        ffn_hidden_size=ffn,
        num_attention_heads=24,
        num_query_groups=4,
        kv_channels=256,
        gdn_key_heads=16,
        gdn_key_head_dim=128,
        gdn_value_heads=48,
        gdn_value_head_dim=128,
        gdn_conv_kernel=4,
    )
    r._attention_output_gate = True
    r._gdn_layers = layers * 3 // 4
    r._topology_key = lambda: topology
    return r


def _required(
    r,
    group_rows=((ROWS, True),),
    topology=TP4,
    gdn_segments=1,
    output=OUTPUT,
    logical=None,
):
    signature = _MemorySignature(
        topology,
        (1, None),
        len(group_rows),
        (),
        any(grad for _, grad in group_rows),
        tuple(grad for _, grad in group_rows),
    )
    return r._subforward_cost(
        packed_tokens=sum(rows for rows, _ in group_rows),
        output_bytes=output,
        signature=signature,
        logical_tokens=sum(rows for rows, _ in group_rows) - 1
        if logical is None
        else logical,
        gdn_segments=gdn_segments,
        group_rows=group_rows,
    )


@pytest.mark.parametrize(
    ("topology", "rows", "logical", "segments", "output", "warm", "observed"), TRACED
)
def test_the_floor_prices_every_traced_peak_within_five_percent(
    topology, rows, logical, segments, output, warm, observed
):
    r = tp_rank(topology=topology)
    if warm:
        # A profile below the floor: only the floor prices the wave.
        signature = _MemorySignature(topology, (1, None), 1, (), True, (True,))
        r._memory_profiles[signature] = _MemoryProfile(1.0, 1)
    cost = _required(r, ((rows, True),), topology, segments, output, logical)
    raw = cost.required / 1.1
    # Brad's standard: at or above the observed peak, by at most 5%.
    assert observed <= raw <= 1.05 * observed


@pytest.mark.parametrize(("topology", "rows"), [(TP4, ROWS), (TP2, 23_878)])
def test_the_floor_is_the_shards_the_explicit_workspace_and_one_gradient_shard(
    topology, rows
):
    tp = topology[1]
    r = tp_rank(topology=topology)
    retained, floor_workspace = r._checkpoint_memory_floor(((rows, True),))
    shard = -(-rows // tp)
    # Each rank saves its sequence shard of every boundary.
    assert retained == shard * LAYERS * H * 2
    # One GDN layer's states for each of the TP - 1 padding roots.
    roots = (tp - 1) * SEGMENT * 4 // tp
    assert floor_workspace == workspace(rows, tp) + roots
    cost = _required(r, ((rows, True),), topology)
    # One input-gradient shard, not a repeat of every boundary.
    assert cost.checkpoint_input_gradient == shard * H * 2
    state = tp * SEGMENT * 4 // tp
    assert cost.checkpoint_workspace == workspace(rows, tp) + state + COLD
    assert cost.required == int(
        (OUTPUT + retained + workspace(rows, tp) + state + COLD + shard * H * 2) * 1.1
    )


def test_a_tp2_wide_ffn_is_priced_by_its_fc1_stage():
    rows = 23_878
    narrow = tp_rank(topology=TP2)._checkpoint_memory_floor(((rows, True),))
    wide = tp_rank(topology=TP2, ffn=4 * F)._checkpoint_memory_floor(((rows, True),))
    # The same boundaries; the FC1 stage grows by 6 x 3F/TP per gathered row.
    assert wide[0] == narrow[0] > 0
    assert wide[1] - narrow[1] == rows * 6 * 3 * F // 2 * 2


def test_shallow_models_keep_the_workspace_and_fewer_boundaries():
    rows = 23_878
    deep = tp_rank(topology=TP2)._checkpoint_memory_floor(((rows, True),))
    shallow = tp_rank(layers=24, topology=TP2)._checkpoint_memory_floor(((rows, True),))
    assert shallow[0] == rows // 2 * 24 * H * 2
    assert shallow[1] == deep[1]


def test_a_split_prices_below_the_whole_wave_it_splits():
    """The production 66,284-row wave and its three requests as split children,
    under the profile the TP2 trace learned for that wave (warm)."""
    from art.trainer_rank._memory import _split_required_memory

    signature = _MemorySignature(TP2, (1, None), 1, (), True, (True,))
    profile = _MemoryProfile(
        bytes_per_token=527_011.88,
        packed_tokens=66_284,
        logical_per_packed=1.0283,
        retained_fraction=0.7674,
        retained_compute_bytes_per_token=344_514.67,
        caller_plans=1,
    )

    def warm():
        r = tp_rank(topology=TP2)
        r._memory_profiles[signature] = profile
        return r

    whole = _required(warm(), ((66_284, True),), TP2, 4, 272_632, 68_158)
    children = [
        _required(warm(), ((rows, True),), TP2, 1, rows * 4, rows)
        for rows in (24_641, 22_962, 20_555)
    ]
    # Every child's boundaries stay live, but only one child's recompute
    # workspace peaks at a time, so splitting lowers the price.
    assert _split_required_memory(children) < whole.required


def test_a_long_single_tp2_item_still_fits_the_production_budget():
    # 61,000 rows at an observed ~522 KB per row peak near 32 GB.
    cost = _required(tp_rank(topology=TP2), ((61_000, True),), TP2, 1, 244_000)
    assert cost.required < 43_835_395_277


def test_rows_are_sharded_with_ceiling_and_only_gradient_groups_save_them():
    r = tp_rank()
    # Physical rows are padded to TP, but a stray remainder still rounds up.
    assert r._checkpoint_memory_floor(((ROWS - 1, True),))[0] == (
        -(-(ROWS - 1) // 4) * LAYERS * H * 2
    )
    mixed = r._checkpoint_memory_floor(((1024, True), (4096, False)))
    assert mixed[0] == 256 * LAYERS * H * 2
    # No-grad groups keep today's four full rows (the static floor covers
    # them), here wider than the gradient group's recompute workspace; the
    # gradient group adds its TP-padding roots' states.
    assert workspace(1024, 4) < 4 * 4096 * H * 2
    assert mixed[1] == 4 * 4096 * H * 2 + 3 * SEGMENT


@pytest.mark.parametrize(
    "case",
    [
        "tp8",
        "cp2",
        "tp2_cp2",
        "pp2",
        "no_sequence_parallel",
        "sequence_parallel_at_tp1",
        "selective_recompute",
        "moe",
        "tp2_moe",
        "moe_geometry",
        "replicated_qkv",
        "tp2_replicated_qkv",
        "missing_attention_geometry",
        "missing_conv_kernel",
        "tp2_no_sequence_parallel",
    ],
)
def test_unproven_shapes_keep_todays_pricing(case):
    shapes = {
        "tp8": dict(topology=(1, 8, 1, 1)),
        "cp2": dict(topology=(1, 4, 2, 1)),
        "tp2_cp2": dict(topology=(1, 2, 2, 1)),
        "pp2": dict(topology=(1, 4, 1, 2)),
        "no_sequence_parallel": dict(sequence_parallel=False),
        "sequence_parallel_at_tp1": dict(topology=(1, 1, 1, 1)),
        "selective_recompute": dict(),
        "moe": dict(),
        "tp2_moe": dict(topology=TP2),
        "moe_geometry": dict(),
        "replicated_qkv": dict(),
        "tp2_replicated_qkv": dict(topology=TP2),
        "missing_attention_geometry": dict(),
        "missing_conv_kernel": dict(),
        "tp2_no_sequence_parallel": dict(topology=TP2, sequence_parallel=False),
    }
    r = tp_rank(**shapes[case])
    if case == "selective_recompute":
        r.runtime.model[0].decoder.config.recompute_granularity = "selective"
    if case in ("moe", "tp2_moe"):
        r._moe_layers = 1
    # Geometry-only MoE, fewer KV groups than TP (a replicated global QKV) and
    # unreadable attention widths are all outside the traced shape.
    edits = {
        "moe_geometry": dict(moe_experts=8),
        "replicated_qkv": dict(num_query_groups=2),
        "tp2_replicated_qkv": dict(num_query_groups=1),
        "missing_attention_geometry": dict(kv_channels=0),
        # Unread conv history would leave each segment's price short.
        "missing_conv_kernel": dict(gdn_conv_kernel=0),
    }
    if case in edits:
        r._geometry = replace(r._geometry, **edits[case])
    assert r._checkpoint_memory_floor(((ROWS, True),)) == (0, 0)


def test_an_ungated_attention_only_model_is_bounded_by_its_attention_width():
    r = tp_rank()
    r._gdn_layers = 0
    r._attention_output_gate = False
    attention = 5 * 24 * 256 + 3 * 4 * 256
    assert r._checkpoint_memory_floor(((ROWS, True),)) == (
        ROWS // 4 * LAYERS * H * 2,
        workspace(ROWS, 4, mixer=attention),
    )


def test_gdn_segment_states_are_priced_with_the_segments():
    """4,096 two-token requests: states grow with segments, not rows."""
    r = tp_rank()
    rows = 8192
    cost = _required(r, group_rows=((rows, True),), gdn_segments=4096)
    states = (4096 + 3) * SEGMENT
    assert cost.checkpoint_workspace == workspace(rows, 4) + states + COLD
    retained = rows // 4 * LAYERS * H * 2
    assert cost.required == int(
        (OUTPUT + retained + workspace(rows, 4) + states + COLD + rows // 4 * H * 2)
        * 1.1
    )


def test_tp_padding_roots_carry_their_own_states():
    """One one-token request: padding makes four roots, each with its states."""
    r = tp_rank()
    cost = _required(r, group_rows=((4, True),), gdn_segments=1)
    # Four roots' initial states alone: 4 x 12 value heads x 128 x 128 x fp32.
    assert (
        cost.checkpoint_workspace - COLD - workspace(4, 4)
        == 4 * SEGMENT
        > 4 * 12 * 128 * 128 * 4
    )
    assert cost.required > 4 * SEGMENT


def test_no_grad_waves_keep_their_pricing_whatever_the_segments():
    r = tp_rank()
    no_grad = r._checkpoint_memory_floor(((8192, False),), None, 8192)
    assert no_grad == (0, 4 * 8192 * H * 2)


def test_width_probes_count_only_gradient_segments():
    from test_trainer_rank_checkpoint_memory import rank, requests

    r = rank()
    # One gradient request and one no-grad reference.
    cheap: list[int] = []
    assert r._estimate_flat_forward(requests(), gdn_segments=cheap) is not None
    assert cheap == [2]  # A radix tree has fewer than twice its requests.
    # The full-sharing estimate rejects widths, so it takes a lower bound.
    minimal: list[int] = []
    assert r._estimate_flat_forward(
        requests(), memory_minimal=True, gdn_segments=minimal
    )
    assert minimal == [1]
    exact: list[int] = []
    assert r._estimate_flat_forward(requests(), exact=True, gdn_segments=exact)
    assert exact == [1]  # The selected layout's actual segments.
