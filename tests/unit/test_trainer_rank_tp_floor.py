"""The full-recompute checkpoint floor under TP4 sequence parallelism.

CPU admission math, not a bound. One four-H200 trace of dense Qwen3.8-27B
(64 layers, TP4, sequence parallel) put the cold peak at each rank's boundary
shards plus a recompute workspace that the floor's repeated shards cover;
other shapes keep today's pricing.
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import TrainerRank
from art.trainer_rank._impl import _MemorySignature

H, F, LAYERS = 5120, 17408, 64
TP4 = (1, 4, 1, 1)
# The traced 062 wave: one request, 25,727 tokens padded to 25,728 rows.
ROWS, OUTPUT = 25_728, 102_908


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
    )
    r._attention_output_gate = True
    r._gdn_layers = layers * 3 // 4
    r._topology_key = lambda: topology
    return r


def _required(r, group_rows=((ROWS, True),), topology=TP4, gdn_segments=1):
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
        output_bytes=OUTPUT,
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
    assert workspace == 0
    cost = _required(r)
    assert cost.checkpoint_input_gradient == retained
    # One segment's recurrent states in the recomputed GDN layer.
    state = cost.checkpoint_workspace
    assert state == -(-r._gdn_segment_layer_bytes() // 1) < 2 * 2**20
    assert cost.required == int((OUTPUT + 2 * retained + state) * 1.1)
    # Measured cold on all four ranks: 7.130 GB (7.060 GB in production), all
    # but the boundaries a transient recompute workspace; this raw floor
    # (8.43 GB) covers it. Today's cold admission was 4.637 GB.
    assert cost.required / 1.1 > 7.130e9


def test_rows_are_sharded_with_ceiling_and_only_gradient_groups_save_them():
    r = tp_rank()
    # Physical rows are padded to TP, but a stray remainder still rounds up.
    assert r._checkpoint_memory_floor(((ROWS - 1, True),))[0] == (
        -(-(ROWS - 1) // 4) * LAYERS * H * 2
    )
    mixed = r._checkpoint_memory_floor(((1024, True), (4096, False)))
    assert mixed[0] == 256 * LAYERS * H * 2
    # No-grad groups keep today's four full rows; the static floor covers them.
    assert mixed[1] == 4 * 4096 * H * 2


@pytest.mark.parametrize(
    "case",
    [
        "tp2",
        "tp8",
        "cp2",
        "pp2",
        "no_sequence_parallel",
        "sequence_parallel_at_tp1",
        "selective_recompute",
        "moe",
        "moe_geometry",
        "replicated_qkv",
        "missing_attention_geometry",
        "shallow",
        "wide_ffn",
    ],
)
def test_unproven_shapes_keep_todays_pricing(case):
    shapes = {
        "tp2": dict(topology=(1, 2, 1, 1)),
        "tp8": dict(topology=(1, 8, 1, 1)),
        "cp2": dict(topology=(1, 4, 2, 1)),
        "pp2": dict(topology=(1, 4, 1, 2)),
        "no_sequence_parallel": dict(sequence_parallel=False),
        "sequence_parallel_at_tp1": dict(topology=(1, 1, 1, 1)),
        "selective_recompute": dict(),
        "moe": dict(),
        "moe_geometry": dict(),
        "replicated_qkv": dict(),
        "missing_attention_geometry": dict(),
        "shallow": dict(layers=48),
        "wide_ffn": dict(ffn=4 * F),
    }
    r = tp_rank(**shapes[case])
    if case == "selective_recompute":
        r.runtime.model[0].decoder.config.recompute_granularity = "selective"
    if case == "moe":
        r._moe_layers = 1
    # Geometry-only MoE, fewer KV groups than TP (a replicated global QKV) and
    # unreadable attention widths are all outside the traced shape.
    edits = {
        "moe_geometry": dict(moe_experts=8),
        "replicated_qkv": dict(num_query_groups=2),
        "missing_attention_geometry": dict(kv_channels=0),
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
    states = -(-4096 * r._gdn_segment_layer_bytes() // 1)
    assert cost.checkpoint_workspace == states
    # The recomputed layer's initial states alone: 4,096 x 12 local value
    # heads x 128 x 128 x fp32.
    assert states > 4096 * 12 * 128 * 128 * 4
    assert cost.required >= int(
        (OUTPUT + 2 * rows // 4 * LAYERS * H * 2 + states) * 1.1
    )
