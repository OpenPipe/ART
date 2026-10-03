"""The full-recompute checkpoint floor under TP2 and TP4 sequence parallelism.

CPU admission math, not a bound. A four-H200 TP4 trace and a two-H200 TP2
trace of dense Qwen3.8-27B (64 layers, sequence parallel) put the cold peak at
each rank's boundary shards plus a recompute workspace that the floor's
repeated shards cover; other shapes keep today's pricing.
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import TrainerRank
from art.trainer_rank._impl import _COLD_RECOMPUTE_TRANSIENT_BYTES as COLD
from art.trainer_rank._impl import _MemorySignature

H, F, LAYERS = 5120, 17408, 64
TP4 = (1, 4, 1, 1)
# The traced 062 wave: one request, 25,727 tokens padded to 25,728 rows.
ROWS, OUTPUT = 25_728, 102_908
# One segment's initial and final fp32 states over 12 local value heads, plus
# conv history over 3 taps, in the recomputed GDN layer.
SEGMENT = (4 * 12 * 128 * 128 + 2 * (2 * 4 * 128 + 12 * 128) * 3) * 2
TP2 = (1, 2, 1, 1)
# The traced kr28/kr29 first gradient waves: rows, output bytes, the cold peak
# measured on both TP2 ranks and the highest production report of the wave.
TP2_WAVES = (
    (23_878, 95_512, 12_645_404_160, 14_627_176_448),
    (20_816, 83_264, 10_689_269_248, 10_935_935_488),
)


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
    r, group_rows=((ROWS, True),), topology=TP4, gdn_segments=1, output=OUTPUT
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
        logical_tokens=sum(rows for rows, _ in group_rows) - 1,
        gdn_segments=gdn_segments,
        group_rows=group_rows,
    )


def test_the_traced_tp4_wave_prices_its_boundary_shards_and_their_repeat():
    r = tp_rank()
    retained, workspace = r._checkpoint_memory_floor(((ROWS, True),))
    # Each rank saves a quarter of every boundary: the traced 4.215 GB.
    assert retained == ROWS // 4 * LAYERS * H * 2 == 4_215_275_520
    # Without a segment count, only the TP-padding roots' states.
    assert workspace == 3 * SEGMENT
    cost = _required(r)
    assert cost.checkpoint_input_gradient == retained
    # One segment plus up to three TP-padding roots, each with its states,
    # beside the unprofiled first execution's transients.
    state = cost.checkpoint_workspace - COLD
    assert state == 4 * SEGMENT
    assert cost.required == int((OUTPUT + 2 * retained + state + COLD) * 1.1)
    # Measured cold on all four ranks: 7.130 GB (7.060 GB in production), all
    # but the boundaries a transient recompute workspace; this raw floor
    # (8.43 GB) covers it. Today's cold admission was 4.637 GB.
    assert cost.required / 1.1 > 7.130e9


@pytest.mark.parametrize(("rows", "output", "measured", "production"), TP2_WAVES)
def test_the_traced_tp2_waves_price_their_boundary_shards_and_their_repeat(
    rows, output, measured, production
):
    r = tp_rank(topology=TP2)
    retained, workspace = r._checkpoint_memory_floor(((rows, True),))
    # Each rank saves half of every boundary: exactly what both ranks held.
    assert retained == rows // 2 * LAYERS * H * 2
    # Half the value heads per rank: each root's states are twice TP4's.
    assert workspace == 2 * SEGMENT
    cost = _required(r, ((rows, True),), TP2, output=output)
    assert cost.checkpoint_input_gradient == retained
    # One segment plus one TP-padding root, beside the cold transients.
    assert cost.checkpoint_workspace - COLD == 2 * 2 * SEGMENT
    assert cost.required == int((output + 2 * retained + 4 * SEGMENT + COLD) * 1.1)
    # Today's cold admission was 1.1 x 16H per token (4.30 / 3.75 GB). The raw
    # floor covers the traced peak and the wave's highest production report.
    assert cost.required / 1.1 > max(measured, production)


def test_rows_are_sharded_with_ceiling_and_only_gradient_groups_save_them():
    r = tp_rank()
    # Physical rows are padded to TP, but a stray remainder still rounds up.
    assert r._checkpoint_memory_floor(((ROWS - 1, True),))[0] == (
        -(-(ROWS - 1) // 4) * LAYERS * H * 2
    )
    mixed = r._checkpoint_memory_floor(((1024, True), (4096, False)))
    assert mixed[0] == 256 * LAYERS * H * 2
    # No-grad groups keep today's four full rows (the static floor covers
    # them); the gradient group adds its TP-padding roots' states.
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
        "shallow",
        "tp2_shallow",
        "tp2_no_sequence_parallel",
        "wide_ffn",
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
        "shallow": dict(layers=48),
        "tp2_shallow": dict(layers=42, topology=TP2),
        "tp2_no_sequence_parallel": dict(topology=TP2, sequence_parallel=False),
        "wide_ffn": dict(ffn=4 * F),
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


def test_the_depth_bound_is_the_traced_workspace_at_these_widths():
    # Per gathered row, the repeated shards (layers x H / 4) must cover the
    # SP gather (2H), the FC1 stage (6F/4), the wider mixer (GDN here), the
    # norms (H), other workspace (H) and one gradient shard (H/4): about 49
    # layers at these widths.
    assert tp_rank(layers=49)._checkpoint_memory_floor(((ROWS, True),))[0] > 0
    assert tp_rank(layers=48)._checkpoint_memory_floor(((ROWS, True),)) == (0, 0)
    # At TP2 the norms are a measured 3.2H per sharded row (4H/TP per gathered
    # row, the TP4 term), so the shallowest covered model has 43 layers.
    assert (
        tp_rank(layers=43, topology=TP2)._checkpoint_memory_floor(((ROWS, True),))[0]
        > 0
    )
    assert tp_rank(layers=42, topology=TP2)._checkpoint_memory_floor(
        ((ROWS, True),)
    ) == (0, 0)


def test_an_ungated_attention_only_model_is_bounded_by_its_attention_width():
    r = tp_rank()
    r._gdn_layers = 0
    r._attention_output_gate = False
    assert r._checkpoint_memory_floor(((ROWS, True),)) == (
        ROWS // 4 * LAYERS * H * 2,
        0,
    )


def test_gdn_segment_states_are_priced_with_the_segments():
    """4,096 two-token requests: states grow with segments, not rows."""
    r = tp_rank()
    rows = 8192
    cost = _required(r, group_rows=((rows, True),), gdn_segments=4096)
    assert cost.checkpoint_workspace == (4096 + 3) * SEGMENT + COLD
    assert cost.required == int(
        (OUTPUT + 2 * rows // 4 * LAYERS * H * 2 + (4096 + 3) * SEGMENT + COLD) * 1.1
    )


def test_tp_padding_roots_carry_their_own_states():
    """One one-token request: padding makes four roots, each with its states."""
    r = tp_rank()
    cost = _required(r, group_rows=((4, True),), gdn_segments=1)
    # Four roots' initial states alone: 4 x 12 value heads x 128 x 128 x fp32.
    assert cost.checkpoint_workspace - COLD == 4 * SEGMENT > 4 * 12 * 128 * 128 * 4
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
