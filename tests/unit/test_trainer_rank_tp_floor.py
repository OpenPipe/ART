"""The explicit full-recompute checkpoint floor under TP2/TP4 sequence parallelism.

CPU admission math, not a bound. Allocator traces of dense Qwen3.8-27B (64
layers, sequence parallel, LoRA rank 4 at TP2 and 8 at TP4) at TP4 and TP2,
cold and warm, from 20,816 to 66,284 rows and one to four GDN segments, put
every peak at each rank's boundary shards plus the recomputed layer's
workspace; the floor covers those terms explicitly. Shallower, wider,
attention-only and higher-rank dense shapes are priced by the same terms
(untraced); TP8, CP, MoE, replicated QKV and TP without sequence parallelism
keep today's pricing.
"""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from trainer_rank_test_support import fake_rank, recompute_model

from art.trainer_rank import TrainerRank
from art.trainer_rank._impl import _SEQUENCE_PARALLEL_COLD_TRANSIENT_BYTES as COLD
from art.trainer_rank._impl import _MemoryProfile, _MemorySignature

H, F, LAYERS = 5120, 17408, 64
TP4 = (1, 4, 1, 1)
# The traced 062 wave: one request, 25,727 tokens padded to 25,728 rows.
ROWS, OUTPUT = 25_728, 102_908
# One segment's initial and final fp32 states over 12 local value heads, plus
# conv history over 3 taps, in the recomputed GDN layer.
SEGMENT = (4 * 12 * 128 * 128 + 2 * (2 * 4 * 128 + 12 * 128) * 3) * 2
TP2 = (1, 2, 1, 1)
# The comparable traced TP x SP calls: topology, rows, logical rows, GDN
# segments, output bytes, whether the signature was profiled, and the peak
# above the call's baseline (identical on every rank). Excluded, see
# EXPLICIT-FLOOR.md: the 40,000-row planner-cold call (above the band: 0.69 GB
# of earlier tensors freed during it depress its harness peak) and production
# rank 0 at 23,878 rows (14.63 GB: an unexplained +1.98 GB over rank 1 and
# the trace, accepted and not covered).
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
        TP2, 20_816, 20_816, 1, 83_264, False, 10_689_269_248, id="tp2-20816-cold"
    ),
    pytest.param(
        TP2, 20_816, 20_816, 1, 83_264, True, 10_721_021_440, id="tp2-20816-warm"
    ),
    pytest.param(
        TP2, 40_000, 40_000, 2, 160_000, True, 20_601_088_000, id="tp2-40000-warm"
    ),
    pytest.param(
        TP2, 23_878, 23_878, 1, 95_512, True, 12_129_966_592, id="tp2-23878-warm-2"
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


GDN_WIDTH = 4 * 16 * 128 + 8 * 48 * 128
GATED_ATTENTION_WIDTH = 7 * 24 * 256 + 3 * 4 * 256


def workspace(
    rows, tp, ffn=F, stage=6, mixers=((GATED_ATTENTION_WIDTH, 0), (GDN_WIDTH, 1))
):
    """The traced recompute workspace (bytes): gathered 2H + stage F/TP, 3.25H
    of norms per sharded row, the fp32 rotary cache, and for each mixer kind
    (width, layers above its highest layer) its width/TP plus the boundary
    shards beyond L live at that layer's recompute."""
    shard = -(-rows // tp) * H
    peak = max(rows * -(-width // tp) + (1 - gap) * shard for width, gap in mixers)
    gathered = rows * (2 * H + -(-stage * ffn // tp))
    rotary = 2 * rows * 256 if any(width != GDN_WIDTH for width, _ in mixers) else 0
    return (gathered + -(-rows // tp) * -(-13 * H // 4) + peak + rotary) * 2


def signature(topology, rank=0, grad=True):
    """A gradient signature whose slot carries LoRA modules of ``rank``."""
    shapes = ((True, ((2, H, rank, rank, H),)),) if rank else ()
    return _MemorySignature(
        topology, (1, None), 1, (), grad, (grad,), slot_shapes=shapes
    )


def tp_rank(
    layers=LAYERS,
    *,
    ffn=F,
    topology=TP4,
    sequence_parallel=True,
    decoder_layers=None,
    **config,
):
    from megatron.core.transformer.transformer_block import TransformerBlock

    model = recompute_model(
        TransformerBlock,
        H,
        layers,
        sequence_parallel,
        layers=decoder_layers or (),
        **config,
    )
    if decoder_layers is not None:
        model.embedding = torch.nn.Linear(1, 1).bfloat16()  # a BF16 parameter
    r: Any = fake_rank(TrainerRank, [model], hidden_size=H, num_layers=layers)
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
    if decoder_layers is None:
        # Layers 3, 7, ..., 63 are attention: the top GDN layer is one below.
        r._mixer_top_gaps = (0, 1)
        # Six adapted modules per GDN layer, seven per attention layer.
        r._lora_modules_per_layer = 7
    return r


POLICY_RANK = 8192


def adapted_layers(per_layer, *, rank=POLICY_RANK):
    """Decoder layers of real LoRA modules holding a ``policy`` slot of ``rank``."""
    from art.megatron.lora import LoRA, LoRASlotRef

    ref = LoRASlotRef("checkpoint", "policy")

    def module():
        lora = LoRA.__new__(LoRA)
        torch.nn.Module.__init__(lora)
        lora._slot_keys = {ref: "policy"}
        lora._slot_modules = {
            "policy": SimpleNamespace(
                A_T=torch.empty(H, rank, device="meta"),
                B_T=torch.empty(rank, H, device="meta"),
            )
        }
        return lora

    return ref, [
        torch.nn.ModuleList([module() for _ in range(count)]) for count in per_layer
    ]


def produced_signature(r, ref, rows):
    """The memory signature the planner builds for one gradient request."""
    from art.trainer_rank import ForwardInput

    tokens = torch.zeros(rows, dtype=torch.long)
    return r._memory_signature_from_requests(
        [ForwardInput(input_tokens=tokens, target_tokens=tokens)],
        slot_group_count=1,
        grad_modes=(True,),
        slot_groups=((ref, True),),
    )


def lora_workspace(r, ref, rows):
    """The floor's workspace for one gradient group under ``ref``'s signature."""
    return r._subforward_cost(
        packed_tokens=rows,
        output_bytes=rows * 4,
        signature=produced_signature(r, ref, rows),
        logical_tokens=rows,
        gdn_segments=1,
        group_rows=((rows, True),),
    ).checkpoint_workspace


def _required(
    r,
    group_rows=((ROWS, True),),
    topology=TP4,
    gdn_segments=1,
    output=OUTPUT,
    logical=None,
    rank=0,
):
    shapes = ((True, ((2, H, rank, rank, H),)),) if rank else ()
    signature = _MemorySignature(
        topology,
        (1, None),
        len(group_rows),
        (),
        any(grad for _, grad in group_rows),
        tuple(grad for _, grad in group_rows),
        slot_shapes=shapes,
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


def _warm(r, topology, rank=0, profile=None):
    """Profile the signature (by default below the floor, so the floor prices)."""
    r._memory_profiles[signature(topology, rank)] = profile or _MemoryProfile(1.0, 1)
    return r


@pytest.mark.parametrize(
    ("topology", "rows", "logical", "segments", "output", "warm", "observed"), TRACED
)
def test_the_floor_covers_every_comparable_traced_peak_within_five_percent(
    topology, rows, logical, segments, output, warm, observed
):
    rank = 4 if topology == TP2 else 8  # the traced adapters
    r = tp_rank(topology=topology)
    if warm:
        _warm(r, topology, rank)
    cost = _required(r, ((rows, True),), topology, segments, output, logical, rank)
    raw = cost.required / 1.1
    # Brad's standard: at or above the observed peak, by at most 5%.
    assert observed <= raw <= 1.05 * observed


@pytest.mark.parametrize(
    ("topology", "rows"), [(TP4, ROWS), (TP2, 23_878), (TP2, 23_877)]
)
def test_the_floor_is_the_shards_the_explicit_workspace_and_one_gradient_shard(
    topology, rows
):
    tp = topology[1]
    r = tp_rank(topology=topology)
    retained, floor_workspace = r._checkpoint_memory_floor(((rows, True),))
    shard = -(-rows // tp)  # an odd row count still rounds up per rank
    # Each rank saves its sequence shard of every boundary.
    assert retained == shard * LAYERS * H * 2
    # One GDN layer's states for each of the TP - 1 padding roots.
    roots = (tp - 1) * SEGMENT * 4 // tp
    assert floor_workspace == workspace(rows, tp) + roots
    cost = _required(r, ((rows, True),), topology, rank=4)
    # One input-gradient shard, not a repeat of every boundary.
    assert cost.checkpoint_input_gradient == shard * H * 2
    state = tp * SEGMENT * 4 // tp
    # Seven adapted modules' rank-4 ``x @ A`` outputs over the gathered rows.
    lora = rows * 7 * 4 * 2
    assert cost.checkpoint_workspace == workspace(rows, tp) + lora + state + COLD
    assert cost.required == int(
        (OUTPUT + retained + workspace(rows, tp) + lora + state + COLD + shard * H * 2)
        * 1.1
    )


def test_the_memory_check_path_prices_the_same_floor():
    """_memory_check's estimator (include_checkpoint_input_gradient=True)."""
    for profiled in (False, True):
        r = tp_rank(topology=TP2)
        if profiled:
            _warm(r, TP2, 4)
        cost = _required(r, ((23_879, True),), TP2, rank=4)
        estimate = r._estimate_required_memory_bytes_from_values(
            packed_tokens=23_879,
            output_bytes=OUTPUT,
            signature=signature(TP2, 4),
            logical_tokens=23_878,
            gdn_segments=1,
            group_rows=((23_879, True),),
        )
        # The repeated boundaries would add 63 shards (about 7.7 GB).
        assert estimate == cost.required


def test_lora_rank_moves_the_price_by_its_intermediates():
    rows = 25_728
    low, high = (
        _required(tp_rank(), ((rows, True),), TP4, rank=rank).checkpoint_workspace
        for rank in (128, 8192)
    )
    # Seven modules' rows x rank BF16 outputs: the FC1 gate/up pair alone
    # adds 829,882,368 bytes.
    assert high - low == rows * 7 * (8192 - 128) * 2


def test_attention_only_slots_price_their_lora_intermediates():
    """An attention-only TP2 model: ranks come from the real slot-shape and
    module-count producers, not a hand-built signature."""
    rows = 23_878
    ref, layers = adapted_layers([7] * LAYERS)
    r = tp_rank(topology=TP2, decoder_layers=layers)
    r._gdn_layers = 0
    assert r._mixer_top_gaps == (0, None)  # read from the decoder
    assert r._lora_modules_per_layer == 7
    shapes = produced_signature(r, ref, rows).slot_shapes
    assert shapes == ((True, ((2, H, POLICY_RANK, POLICY_RANK, H),) * 7 * LAYERS),)
    base = lora_workspace(r, None, rows)
    # Seven modules' rows x 8,192 BF16 intermediates: 2,738,520,064 bytes.
    assert lora_workspace(r, ref, rows) - base == rows * 7 * POLICY_RANK * 2


def test_an_unknown_per_layer_count_charges_every_adapted_module():
    """Seven modules concentrated in one of 64 layers: the mean (two) would
    omit 1,956,085,760 bytes; a report without the count charges all seven."""
    rows = 23_878
    ref, layers = adapted_layers([7] + [0] * (LAYERS - 1))
    r = tp_rank(topology=TP2, decoder_layers=layers)
    r._mixer_top_gaps = (0, 1)
    assert r._lora_modules_per_layer == 7
    live = lora_workspace(r, ref, rows) - lora_workspace(r, None, rows)
    assert live == rows * 7 * POLICY_RANK * 2
    r._lora_modules_per_layer = None  # as replayed from an older report
    assert lora_workspace(r, ref, rows) - lora_workspace(r, None, rows) == live


def test_two_gradient_groups_charge_one_group_workspace_and_gradient_shard():
    r = tp_rank(topology=TP2)
    groups = ((20_000, True), (12_000, True))
    retained, floor_workspace = r._checkpoint_memory_floor(groups, None, 2)
    # Both groups' boundaries stay live; their backwards run one at a time.
    assert retained == (10_000 + 6_000) * LAYERS * H * 2
    # Two segments plus one padding root per group (states remain summed).
    assert floor_workspace == workspace(20_000, 2) + 4 * SEGMENT * 4 // 2
    cost = _required(r, groups, TP2, 2)
    assert cost.checkpoint_input_gradient == 10_000 * H * 2


def test_a_clamped_swiglu_prices_its_seven_f_stage():
    rows = 23_878
    r = tp_rank(topology=TP2)
    r._mlp_activation_factor = 7
    assert (
        r._checkpoint_memory_floor(((rows, True),))[1]
        == workspace(rows, 2, stage=7) + SEGMENT * 4 // 2
    )


@pytest.mark.parametrize("order", ["traced", "gdn_top", "unknown"])
def test_the_peak_layer_sets_the_boundary_shards(order):
    rows = 23_878
    r = tp_rank(topology=TP2)
    gaps = {"traced": (0, 1), "gdn_top": (1, 0), "unknown": None}[order]
    r._mixer_top_gaps = gaps
    gdn_gap = 1 if order == "traced" else 0
    attention_gap = 1 if order == "gdn_top" else 0
    expected = workspace(
        rows,
        2,
        mixers=((GATED_ATTENTION_WIDTH, attention_gap), (GDN_WIDTH, gdn_gap)),
    )
    assert r._checkpoint_memory_floor(((rows, True),))[1] == expected + SEGMENT * 2
    if order != "traced":
        # A top-layer GDN peak (or an unknown order) holds L + 1 shards.
        assert expected - workspace(rows, 2) == rows // 2 * H * 2


def test_attention_only_tp2_is_priced_at_its_attention_width():
    rows = 23_877
    r = tp_rank(topology=TP2)
    r._gdn_layers = 0
    r._attention_output_gate = False
    r._mixer_top_gaps = (0, None)
    attention = 5 * 24 * 256 + 3 * 4 * 256
    assert r._checkpoint_memory_floor(((rows, True),)) == (
        -(-rows // 2) * LAYERS * H * 2,
        workspace(rows, 2, mixers=((attention, 0),)),
    )


def _with_head(r, vocabulary, tp):
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    head = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(head)
    head.weight = torch.nn.Parameter(
        torch.empty(vocabulary // tp, H, dtype=torch.bfloat16, device="meta")
    )
    head.output_size_per_partition = vocabulary // tp
    head.output_size = vocabulary
    head.input_size = H
    r.runtime.model[0].output_layer = head
    r.runtime.model[0].share_embeddings_and_output_weights = False
    r._padded_vocab_size = vocabulary
    return r


def test_the_tp2_head_stage_is_priced_with_the_floor():
    from art.trainer_rank import _memory

    vocabulary = 248_320
    r = _with_head(tp_rank(topology=TP2), vocabulary, 2)
    # Each rank projects its vocabulary shard, 512 rows at a time.
    assert _memory._head_vocabulary(r) == vocabulary // 2
    # Without the floor's eligibility (no SP) TP2 keeps today's zero.
    no_sp = _with_head(tp_rank(topology=TP2, sequence_parallel=False), vocabulary, 2)
    assert _memory._head_vocabulary(no_sp) == 0
    # A short wave's head stage (logits plus both target-backward buffers)
    # outweighs its recompute workspace and prices the floor.
    head = 3 * r._head_workspace_bytes(512)
    signature = _MemorySignature(TP2, (1, None), 1, (), True, (True,))
    cost = r._subforward_cost(
        packed_tokens=512,
        output_bytes=2_048,
        signature=signature,
        logical_tokens=512,
        gdn_segments=1,
        group_rows=((512, True),),
        head_workspace_bytes=head,
    )
    assert head > workspace(512, 2) + SEGMENT * 2
    assert cost.checkpoint_workspace == head + COLD


# The bounded eager statistics' increment for a 512-row, 124,160-entry chunk:
# one 8-row FP32 sub-chunk plus per-row statistics (3,981,312 bytes).
BOUNDED = 4 * 124_160 * 8 + 16 * 512


def _labelled(rows):
    """One CPU request whose ``rows`` positions are all labelled targets."""
    tokens = torch.zeros(rows, dtype=torch.long)
    return [
        SimpleNamespace(
            input_tokens=tokens, target_tokens=tokens, logits=False, top_k=None
        )
    ]


def _head_stage(r, requests):
    """The head charge as planned: the capped projection, then each chunk."""
    rows = r._head_projection_rows(requests)
    return r._group_head_workspace_bytes(rows, requests, grad_enabled=True)


def test_the_tp2_head_stage_follows_the_statistics_path(monkeypatch):
    from art.trainer_rank import _impl, _memory

    r = _with_head(tp_rank(topology=TP2), 248_320, 2)
    monkeypatch.setattr(_memory, "_head_target_backward", lambda rank: True)
    dense = r._head_workspace_bytes(512)
    # The same predicate as _try_triton_stats: a CPU device, the
    # ART_TRAINER_RANK_TRITON_TOPK switch and short chunks take the eager
    # FP32 fallback, about seven BF16 buffers.
    assert _head_stage(r, _labelled(512)) == 7 * dense
    monkeypatch.setattr(_impl, "_triton_head_stats", lambda rank, rows, vocab: True)
    # The kernel path, plus the bounded fallback's increment should it fail.
    assert _head_stage(r, _labelled(512)) == 3 * dense + BOUNDED
    monkeypatch.setenv("ART_TRAINER_RANK_TRITON_TOPK", "0")
    assert not _impl._triton_stats_enabled(True, 512)
    monkeypatch.delenv("ART_TRAINER_RANK_TRITON_TOPK")
    assert _impl._triton_stats_enabled(True, 64)
    assert not _impl._triton_stats_enabled(True, 63)
    assert not _impl._triton_stats_enabled(False, 512)


@pytest.mark.parametrize(
    ("rows", "minimum", "expected", "eager"),
    [
        # Sol: a 511-row eager tail behind a 512-row Triton chunk.
        (1_023, "512", 7 * 511 * 124_160 * 2, True),  # 888,240,640 bytes
        # Schulman: an 812-row wave's 300-row eager tail.
        (812, "512", 7 * 300 * 124_160 * 2, True),  # 521,472,000 bytes
        # The default threshold (64): every chunk runs Triton (381,419,520
        # bytes), or the bounded statistics should a kernel fail.
        (1_023, None, 3 * 512 * 124_160 * 2 + BOUNDED, False),
        # A 40-row default tail falls back, below the full chunk's charge.
        (552, None, 3 * 512 * 124_160 * 2 + BOUNDED, True),
    ],
)
def test_every_head_chunk_is_priced_for_its_own_statistics_path(
    monkeypatch, rows, minimum, expected, eager
):
    from art.trainer_rank import _impl, _memory

    r = _with_head(tp_rank(topology=TP2), 248_320, 2)
    r.device = torch.device("cuda", 0)  # a CUDA rank whose kernels import
    monkeypatch.setattr(_impl, "_triton_stats_importable", lambda: True)
    monkeypatch.setattr(_memory, "_head_target_backward", lambda rank: True)
    if minimum is None:
        monkeypatch.delenv("ART_TRAINER_RANK_TRITON_MIN_ROWS", raising=False)
    else:
        monkeypatch.setenv("ART_TRAINER_RANK_TRITON_MIN_ROWS", minimum)
    # The planner's projection is capped at one chunk; the tail is not.
    assert r._head_projection_rows(_labelled(rows)) == 512
    assert _head_stage(r, _labelled(rows)) == expected
    # Replay capture refuses exactly the waves with an eager chunk.
    from art.trainer_rank._planner_replay import head_statistics_fallback

    assert head_statistics_fallback(r, _labelled(rows), None) == eager


def _cuda_head(monkeypatch, vocabulary=248_320):
    """A TP2 rank with a head on a CUDA device whose kernels import."""
    from art.trainer_rank import _impl, _memory

    r = _with_head(tp_rank(topology=TP2), vocabulary, 2)
    r.device = torch.device("cuda", 0)
    monkeypatch.setattr(_impl, "_triton_stats_importable", lambda: True)
    monkeypatch.setattr(_memory, "_head_target_backward", lambda rank: True)
    return r


def _positioned(*positions):
    """Labelled requests and their packed positions."""
    requests = [_labelled(len(row))[0] for row in positions]
    return requests, [torch.tensor(row) for row in positions]


@pytest.mark.parametrize(
    ("first", "second", "exact"),
    [
        # Sol: two 812-row requests sharing 600 positions, a 1,024-row union
        # of two kernel chunks; one request alone has a 300-row eager tail.
        (
            range(812),
            [*range(600), *range(812, 1024)],
            3 * 512 * 124_160 * 2 + BOUNDED,
        ),
        # Two 512-row requests sharing 212 positions: their no-sharing sum is
        # two kernel chunks, but the 812-row union has a 300-row eager tail.
        (range(512), [*range(212), *range(512, 812)], 7 * 300 * 124_160 * 2),
    ],
)
def test_head_bounds_envelope_every_possible_projected_union(
    monkeypatch, first, second, exact
):
    monkeypatch.setenv("ART_TRAINER_RANK_TRITON_MIN_ROWS", "512")
    r = _cuda_head(monkeypatch)
    requests, positions = _positioned(list(first), second)

    def charge(**bounds):
        rows = r._head_projection_rows(
            requests,
            positions=bounds.get("positions"),
            lower_bound=bounds.get("lower_bound", False),
        )
        return r._group_head_workspace_bytes(
            rows, requests, grad_enabled=True, **bounds
        )

    # The selected layout's packed union prices every chunk exactly.
    assert charge(positions=positions) == exact
    # Width search: the rejection bound never exceeds it, the acceptance
    # bound never falls below it.
    assert charge(lower_bound=True) <= exact <= charge()


def test_a_kernel_failure_is_priced_only_at_its_chunk_shape(monkeypatch):
    from art.trainer_rank import _impl, _memory

    r = _cuda_head(monkeypatch, vocabulary=16)  # an 8-entry vocabulary shard

    def statistics(rows, *, top_k, logsumexp, max_top_k=0):
        """Run one head chunk's statistics with the given kernel outcomes."""
        logits = torch.zeros(rows, 8)
        stats = (torch.zeros(rows), torch.ones(rows))
        with monkeypatch.context() as patch:
            patch.setattr(
                TrainerRank, "_local_logits_from_hidden_rows", lambda *a, **k: logits
            )
            # The chunk is attempted (as on CUDA) and the kernels return these.
            patch.setattr(_impl, "_triton_stats_enabled", lambda cuda, rows: True)
            patch.setattr(
                _impl,
                "_try_triton_local_topk_stats",
                lambda *a, **k: stats if top_k else None,
            )
            patch.setattr(
                _impl, "_try_triton_stats", lambda *a, **k: stats if logsumexp else None
            )
            patch.setattr(_impl, "_all_reduce_tensor_parallel_max", lambda t: t)
            patch.setattr(_impl, "_all_reduce_tensor_parallel_sum", lambda t: t)
            patch.setattr(_impl, "_vocab_parallel_log_z", lambda t: t[:, 0].float())
            r._local_head_stats(
                None, logits, output_weight=None, need_log_z=True, max_top_k=max_top_k
            )

    def priced(rows):
        """The head charge in chunk buffers, less the bounded increment."""
        charge = _head_stage(r, _labelled(rows))
        if charge >= 7 * r._head_workspace_bytes(rows):
            return 7
        bounded = _memory._eager_stats_extra_bytes(8, rows)
        return (charge - bounded) // r._head_workspace_bytes(rows)

    # A fused top-k failure recovered by the Triton logsumexp: no fallback.
    statistics(512, top_k=False, logsumexp=True, max_top_k=4)
    assert priced(512) == 3
    # A 64-row kernel failure takes the eager fallback (that wave was priced
    # for the kernel); 64-row chunks are priced for the fallback from then on.
    statistics(64, top_k=False, logsumexp=False)
    assert priced(64) == 7
    # A 512-row chunk whose kernel succeeds stays priced for the kernel.
    statistics(512, top_k=False, logsumexp=True)
    assert priced(512) == 3
    # A later 64-row success clears that shape.
    statistics(64, top_k=False, logsumexp=True)
    assert priced(64) == 3


class _LiveBytes:
    """Peak bytes of new CPU tensor storage created inside the context."""

    def __enter__(self):
        import weakref

        from torch.utils._python_dispatch import TorchDispatchMode
        from torch.utils._pytree import tree_leaves

        owner = self
        self.current = self.peak = 0
        live: set[int] = set()

        def free(pointer, size):
            live.discard(pointer)
            owner.current -= size

        class Mode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                out = func(*args, **(kwargs or {}))
                inputs = {
                    t.untyped_storage().data_ptr()
                    for t in tree_leaves((args, kwargs))
                    if isinstance(t, torch.Tensor)
                }
                for t in tree_leaves(out):
                    if not isinstance(t, torch.Tensor):
                        continue
                    pointer = t.untyped_storage().data_ptr()
                    size = t.untyped_storage().nbytes()
                    # Views and in-place results reuse an input's storage.
                    if not size or pointer in inputs or pointer in live:
                        continue
                    live.add(pointer)
                    owner.current += size
                    owner.peak = max(owner.peak, owner.current)
                    weakref.finalize(t, free, pointer, size)
                return out

        self._mode = Mode()
        self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        self._mode.__exit__(*exc)


def test_a_failed_kernel_falls_back_within_the_kernel_buffers(monkeypatch):
    """Admission priced the statistics kernel (three BF16 chunk buffers at the
    head stage: logits, their gradient and the target gather's dense
    gradient). The bounded eager statistics add one FP32 row sub-chunk to
    the logits in forward, and to the logits and their gradient in backward;
    the unchunked fallback adds six buffers in forward."""
    from art.trainer_rank import _impl

    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_max", lambda t: t)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_sum", lambda t: t)
    torch.manual_seed(0)
    logits = torch.randn(512, 1_000, dtype=torch.bfloat16, requires_grad=True)
    buffer = 512 * 1_000 * 2  # one BF16 chunk buffer (D)
    with _LiveBytes() as forward:
        local_max, local_sum = _impl._eager_local_logsumexp_stats(logits)
        log_z = local_max.detach() + torch.log(local_sum)
    assert forward.peak <= buffer / 16
    weights = torch.randn(512)
    with _LiveBytes() as backward:
        (gradient,) = torch.autograd.grad((log_z * weights).sum(), logits)
    assert backward.peak <= buffer + buffer / 16
    # The priced TP2 head charge for this chunk shape (1,000-entry shard)
    # covers the measured live set on both paths: forward, the logits plus
    # the forward increment; backward, the logits, the target gather's dense
    # gradient and the backward increment.
    r = _cuda_head(monkeypatch, vocabulary=2_000)
    targets = _labelled(512)
    trained = r._group_head_workspace_bytes(512, targets, grad_enabled=True)
    assert trained >= buffer + buffer + backward.peak
    inference = r._group_head_workspace_bytes(512, targets, grad_enabled=False)
    assert inference >= buffer + forward.peak
    # The unchunked FP32 fallback, for contrast: about six buffers in forward.
    with _LiveBytes() as unchunked:
        reference = _impl._vocab_parallel_log_z(logits)
    assert unchunked.peak >= 4 * buffer
    # Rows are independent: the same values and gradients.
    (expected,) = torch.autograd.grad((reference * weights).sum(), logits)
    assert torch.equal(log_z, reference)
    assert torch.allclose(gradient.float(), expected.float(), rtol=1e-2, atol=1e-6)


def test_the_bounded_statistics_never_write_fp32_logits(monkeypatch):
    """An FP32 logits chunk: ``.float()`` would alias it, so every sub-chunk
    goes through an owned FP32 work buffer."""
    from art.trainer_rank import _impl

    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_max", lambda t: t)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_sum", lambda t: t)
    torch.manual_seed(2)
    logits = torch.randn(512, 1_000, dtype=torch.float32, requires_grad=True)
    original = logits.detach().clone()
    local_max, local_sum = _impl._eager_local_logsumexp_stats(logits)
    log_z = local_max.detach() + torch.log(local_sum)
    assert torch.equal(logits.detach(), original)
    weights = torch.randn(512)
    (gradient,) = torch.autograd.grad((log_z * weights).sum(), logits)
    assert torch.equal(logits.detach(), original)
    reference = _impl._vocab_parallel_log_z(logits)
    (expected,) = torch.autograd.grad((reference * weights).sum(), logits)
    assert torch.allclose(log_z, reference, rtol=1e-6, atol=1e-6)
    assert torch.allclose(gradient, expected, rtol=1e-5, atol=1e-7)


def test_the_head_statistics_fall_back_to_the_bounded_path_after_a_kernel_failure(
    monkeypatch,
):
    from art.trainer_rank import _impl

    torch.manual_seed(1)
    r = _cuda_head(monkeypatch, vocabulary=2_000)
    logits = torch.randn(512, 1_000, dtype=torch.bfloat16)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_max", lambda t: t)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_sum", lambda t: t)
    reference = _impl._vocab_parallel_log_z(logits)
    monkeypatch.setattr(
        TrainerRank, "_local_logits_from_hidden_rows", lambda *a, **k: logits
    )
    # An attempted kernel (as on CUDA) that fails, fused top-k and logsumexp.
    monkeypatch.setattr(_impl, "_triton_stats_enabled", lambda cuda, rows: True)
    monkeypatch.setattr(_impl, "_try_triton_local_topk_stats", lambda *a, **k: None)
    monkeypatch.setattr(_impl, "_try_triton_stats", lambda *a, **k: None)

    def unchunked(_logits):
        raise AssertionError("the unchunked FP32 fallback ran after an attempt")

    monkeypatch.setattr(_impl, "_vocab_parallel_log_z", unchunked)
    _, log_z, top_k = r._local_head_stats(
        None, logits, output_weight=None, need_log_z=True, max_top_k=4
    )
    assert torch.equal(log_z, reference)
    assert top_k is not None
    assert torch.equal(top_k[0], torch.topk(logits.float(), 4, dim=-1).values)
    # The shape is recorded for later admissions and replay.
    assert (512, 1_000) in r._triton_head_stats_failures


def _requests(rows, *, targets=True, top_k=None, logits=False):
    tokens = torch.zeros(rows, dtype=torch.long)
    return [
        SimpleNamespace(
            input_tokens=tokens,
            target_tokens=tokens if targets else None,
            logits=logits,
            top_k=top_k,
        )
    ]


def test_replay_prices_the_tp2_head_stage_as_live_admission(monkeypatch):
    from art.trainer_rank import _memory

    r = _cuda_head(monkeypatch)  # kernel path: capture declines the fallback
    dense = r._head_workspace_bytes(512)
    cases = (
        # rows, gradient, requests, target rows (as captured), kernel buffers
        (512, True, _requests(512), 512, 3),  # logits, statistics, target gather
        (512, False, _requests(512), 0, 1),
        (300, True, _requests(300), 300, 3),
        # Top-k only: logits, statistics gradient and top-k's scatter.
        (512, True, _requests(512, targets=False, top_k=4), 0, 3),
        (512, False, _requests(512, targets=False, top_k=4), 0, 1),
    )
    for rows, grad, requests, target_rows, buffers in cases:
        live = r._group_head_workspace_bytes(rows, requests, grad_enabled=grad)
        bounded = _memory._eager_stats_extra_bytes(124_160, rows)
        assert live == buffers * r._head_workspace_bytes(rows) + bounded
        frozen = _memory._frozen_head_bytes(
            124_160,
            rows,
            target_rows,
            target_backward=True,
            statistics=True,
            grad=grad,
            tp=2,
        )
        assert frozen == live
    # Requested logits: the local logits, their indexed copy, the gather
    # buffer and its concatenation (2 + 2 TP buffers), live and replayed.
    logits = _requests(512, targets=False, logits=True)
    assert r._group_head_workspace_bytes(512, logits, grad_enabled=False) == 6 * dense
    for tp, units in ((2, 6), (4, 10)):
        assert (
            _memory._frozen_head_bytes(
                124_160,
                512,
                0,
                target_backward=True,
                statistics=False,
                grad=False,
                tp=tp,
            )
            == units * dense
        )
    # TP1 replay is #1068's capacity charge (its gather is the identity).
    for statistics, units in ((True, 7), (False, 3)):
        assert (
            _memory._frozen_head_bytes(
                1_000,
                512,
                0,
                target_backward=True,
                statistics=statistics,
                grad=True,
                tp=1,
            )
            == units * 512 * 1_000 * 2
        )


class _ShapeChangingKernel(torch.autograd.Function):
    """A mocked statistics kernel saving a different tensor set than the eager
    paths (as the real kernel's top-k tokens do)."""

    @staticmethod
    def forward(ctx, logits):
        local_max = logits.max(dim=-1).values.float()
        local_sum = torch.exp(logits.float() - local_max[:, None]).sum(dim=-1)
        ctx.save_for_backward(logits, local_max, torch.empty(logits.shape[0], 0))
        return local_max, local_sum

    @staticmethod
    def backward(ctx, *grad_outputs):
        _grad_max, grad_sum = grad_outputs
        logits, local_max, _ = ctx.saved_tensors
        return (torch.exp(logits.float() - local_max[:, None]) * grad_sum[:, None]).to(
            logits.dtype
        )


@pytest.mark.parametrize("forward_fails", [True, False])
def test_a_checkpoint_recompute_replays_the_forward_statistics_path(
    monkeypatch, forward_fails
):
    """Schulman: a kernel that fails in the forward and succeeds in the
    recompute (or the reverse) changed the saved tensors, a CheckpointError
    under non-reentrant checkpointing."""
    from art.trainer_rank import _impl

    r = _cuda_head(monkeypatch, vocabulary=64)
    weight = torch.randn(16, 32)
    monkeypatch.setattr(
        TrainerRank,
        "_local_logits_from_hidden_rows",
        lambda self, model, hidden, output_weight: hidden @ weight,
    )
    monkeypatch.setattr(_impl, "_triton_stats_enabled", lambda cuda, rows: True)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_max", lambda t: t)
    monkeypatch.setattr(_impl, "_all_reduce_tensor_parallel_sum", lambda t: t)
    attempts = []

    def kernel(name, logits, **kwargs):
        # The first attempt (the forward) fails or succeeds; later ones flip.
        attempts.append(name)
        succeeds = (len(attempts) > 1) == forward_fails
        return _ShapeChangingKernel.apply(logits) if succeeds else None

    monkeypatch.setattr(_impl, "_try_triton_stats", kernel)
    hidden = torch.randn(64, 16, requires_grad=True)
    _, log_z, _ = r._checkpointed_head_stats(
        None, hidden, output_weight=None, need_log_z=True, max_top_k=0
    )
    if forward_fails:
        # The recompute replays the bounded eager statistics: no kernel retry.
        log_z.sum().backward()
        assert attempts == ["local_logsumexp_stats"]
        assert hidden.grad is not None
    else:
        # A kernel that fails only in the recompute cannot reproduce the
        # forward's saved tensors: a clear error, not a CheckpointError.
        with pytest.raises(RuntimeError, match="failed in a checkpoint recompute"):
            log_z.sum().backward()


def test_replay_declines_tp_sp_adapter_estimates_without_ranks():
    from art.trainer_rank._planner_misses import _adapter_ranks_unavailable

    facts = {"checkpoint_layers": 64, "groups": [{"grad": True, "adapter": {}}]}
    bare = signature(TP2)
    ranked = signature(TP2, rank=4)
    assert _adapter_ranks_unavailable(TP2, facts, bare)
    assert not _adapter_ranks_unavailable(TP2, facts, ranked)
    # Gradient entries without a usable (2-D) rank shape decline as well.
    for shapes in (((True, ()),), ((True, ((),)),), ((False, ((2, H, 4, 4, H),)),)):
        partial = replace(bare, slot_shapes=shapes)
        assert _adapter_ranks_unavailable(TP2, facts, partial)
    assert not _adapter_ranks_unavailable((1, 1, 1, 1), facts, bare)
    base = {"checkpoint_layers": 64, "groups": [{"grad": True, "adapter": None}]}
    assert not _adapter_ranks_unavailable(TP2, base, bare)


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
    shallow = tp_rank(layers=24, topology=TP2)
    shallow._mixer_top_gaps = (0, 1)
    assert shallow._checkpoint_memory_floor(((rows, True),)) == (
        rows // 2 * 24 * H * 2,
        deep[1],
    )


def _trace_profile():
    """The profile the TP2 trace learned for the 66,284-row wave."""
    return _MemoryProfile(
        bytes_per_token=527_011.88,
        packed_tokens=66_284,
        logical_per_packed=1.0283,
        retained_fraction=0.7674,
        retained_compute_bytes_per_token=344_514.67,
        caller_plans=1,
    )


def _checkpoint_split(children):
    """_split_required_memory's checkpoint expression (no adapters or growth)."""
    return int(
        (
            sum(c.checkpoint_retained + c.checkpoint_input_gradient for c in children)
            + max(c.checkpoint_workspace for c in children)
        )
        * 1.1
    )


def test_a_warm_split_prices_below_the_whole_wave_it_splits():
    """The production 66,284-row wave and its three requests as split children."""
    from art.trainer_rank._memory import _split_required_memory

    def warm():
        return _warm(tp_rank(topology=TP2), TP2, 4, _trace_profile())

    whole = _required(warm(), ((66_284, True),), TP2, 4, 272_632, 68_158, 4)
    children = [
        _required(warm(), ((rows, True),), TP2, 1, rows * 4, rows, 4)
        for rows in (24_641, 22_962, 20_555)
    ]
    split = _split_required_memory(children)
    # Every child's boundaries stay live; one child's workspace peaks at a time.
    assert split < whole.required
    # The split still pays for that workspace (about 4.9 GB).
    assert split >= _checkpoint_split(children)
    assert max(c.checkpoint_workspace for c in children) > 4.5e9


def test_a_split_whose_checkpoint_expression_binds():
    """A profile retaining less than the boundary shards: the checkpoint
    branch, not the generic retained-plus-ephemeral sum, prices the split."""
    from art.trainer_rank._memory import _split_required_memory

    profile = _MemoryProfile(
        bytes_per_token=300_000.0,
        packed_tokens=24_641,
        retained_compute_bytes_per_token=1.0,
        retained_fraction=0.01,
        caller_plans=1,
    )
    children = [
        _required(
            _warm(tp_rank(topology=TP2), TP2, 4, profile),
            ((rows, True),),
            TP2,
            1,
            rows * 4,
            rows,
            4,
        )
        for rows in (24_641, 22_962, 20_555)
    ]
    assert _split_required_memory(children) == _checkpoint_split(children)


def test_the_largest_admissible_single_tp2_item_matches_the_measured_rate():
    """At the 43.8 GB production budget, a cold single item fits up to the rows
    the measured ~522-527 KB/row peak plus the 1.1 margin allow."""
    budget = 43_835_395_277

    def required(rows):
        return _required(
            tp_rank(topology=TP2), ((rows, True),), TP2, 1, rows * 4, rows, 4
        ).required

    assert required(61_000) < budget  # about 32 GB at the measured rate
    low, high = 1, 200_000
    while low < high:
        middle = (low + high + 1) // 2
        low, high = (middle, high) if required(middle) <= budget else (low, middle - 1)
    # Dropping the explicit workspace would admit about 115,000 rows.
    assert 73_000 < low < 76_000


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
    "case,kwargs",
    [
        ("tp8", dict(topology=(1, 8, 1, 1))),
        ("cp2", dict(topology=(1, 4, 2, 1))),
        ("tp2_cp2", dict(topology=(1, 2, 2, 1))),
        ("pp2", dict(topology=(1, 4, 1, 2))),
        ("no_sequence_parallel", dict(sequence_parallel=False)),
        ("sequence_parallel_at_tp1", dict(topology=(1, 1, 1, 1))),
        ("selective_recompute", dict()),
        ("moe", dict()),
        ("tp2_moe", dict(topology=TP2)),
        ("moe_geometry", dict()),
        ("replicated_qkv", dict()),
        ("tp2_replicated_qkv", dict(topology=TP2)),
        ("missing_attention_geometry", dict()),
        ("missing_conv_kernel", dict()),
        ("tp2_no_sequence_parallel", dict(topology=TP2, sequence_parallel=False)),
    ],
)
def test_unproven_shapes_keep_todays_pricing(case, kwargs):
    r = tp_rank(**kwargs)
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
    r._mixer_top_gaps = (0, None)
    attention = 5 * 24 * 256 + 3 * 4 * 256
    assert r._checkpoint_memory_floor(((ROWS, True),)) == (
        ROWS // 4 * LAYERS * H * 2,
        workspace(ROWS, 4, mixers=((attention, 0),)),
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
