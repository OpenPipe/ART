"""Cold admission checks; these CPU estimates are not native GPU measurements."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from art.trainer_rank import ForwardInput, TrainerRank, TrainerRankMemoryError
from art.trainer_rank._impl import _MemoryProfile


def _rank(
    granularity: str | None = "full",
    *,
    dtype: torch.dtype = torch.bfloat16,
    **geometry: Any,
) -> TrainerRank:
    provider = SimpleNamespace(
        **{
            "hidden_size": 5120,
            "ffn_hidden_size": 17408,
            "num_layers": 64,
            "num_attention_heads": 24,
            "num_query_groups": 4,
            "kv_channels": 256,
            "recompute_granularity": granularity,
            **geometry,
        }
    )
    return TrainerRank(
        cast(
            Any,
            SimpleNamespace(
                model=[torch.nn.Linear(1, 1, dtype=dtype)],
                provider=provider,
                optimizer=None,
                model_support_handler=SimpleNamespace(
                    build_gdn_execution_spec=bool(
                        geometry.get("linear_num_value_heads")
                    ),
                    is_moe=bool(geometry.get("num_moe_experts")),
                ),
            ),
        )
    )


def _plan(rank: TrainerRank, tokens: int = 32710, *, no_grad: bool = False):
    request = ForwardInput(
        input_tokens=torch.tensor([1, 2]), hidden_states=True, no_grad=no_grad
    )
    return replace(
        rank._plan_flat_forward([request]),
        packed_tokens=tokens,
        logical_tokens=tokens,
        output_bytes=tokens * rank.hidden_size * rank._param_dtype_size,
    )


@pytest.mark.parametrize("tp", (2, 4))
@pytest.mark.parametrize("granularity", (None, "selective"))
def test_reported_cold_request_is_refused_before_execution(
    monkeypatch: pytest.MonkeyPatch, tp: int, granularity: str | None
) -> None:
    rank = _rank(granularity)
    monkeypatch.setattr(rank, "_topology_key", lambda: (1, tp, 1, 1))
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: int(119.289 * 2**30))
    monkeypatch.setattr(
        rank, "_execute_flat_plan", lambda _: pytest.fail("unsafe forward admitted")
    )
    assert not rank._memory_check(_plan(rank)).fits
    assert rank._memory_check(_plan(rank, tokens=1024)).fits
    with pytest.raises(TrainerRankMemoryError):
        rank.dp_rank_forward(
            [ForwardInput(input_tokens=torch.arange(32710), hidden_states=True)]
        )


def test_full_recompute_keeps_existing_estimate() -> None:
    rank = _rank()
    plan = _plan(rank)
    assert rank._memory_check(plan).estimated_required_bytes == int(
        (plan.output_bytes + 32710 * 5120 * 2 * 16) * 1.1
    )


@pytest.mark.parametrize("granularity", (None, "selective"))
def test_no_grad_does_not_pay_for_retained_layers(granularity: str | None) -> None:
    rank, full = _rank(granularity), _rank()
    assert rank._memory_check(_plan(rank, no_grad=True)) == full._memory_check(
        _plan(full, no_grad=True)
    )
    assert rank._memory_check(_plan(rank)).estimated_required_bytes > (
        rank._memory_check(_plan(rank, no_grad=True)).estimated_required_bytes
    )


@pytest.mark.parametrize(
    "geometry",
    [
        {"num_layers": 128},
        {"ffn_hidden_size": 34816},
        {"num_attention_heads": 48},
        {"ffn_hidden_size": 0},
        {
            "linear_num_key_heads": 16,
            "linear_key_head_dim": 128,
            "linear_num_value_heads": 48,
            "linear_value_head_dim": 128,
        },
        {"num_moe_experts": 64, "moe_router_topk": 8, "moe_ffn_hidden_size": 8192},
    ],
)
def test_retained_estimate_tracks_model_geometry(geometry: dict[str, Any]) -> None:
    rank, base = _rank("selective", **geometry), _rank("selective")
    assert rank._memory_check(_plan(rank)).estimated_required_bytes > (
        base._memory_check(_plan(base)).estimated_required_bytes
    )


def test_profile_cannot_erase_recompute_floor() -> None:
    rank = _rank("selective")
    plan = _plan(rank)
    cold = rank._memory_check(plan).estimated_required_bytes
    rank._memory_profiles[plan.signature] = _MemoryProfile(0.0, plan.packed_tokens)
    assert rank._memory_check(plan).estimated_required_bytes == cold
    rank._memory_profiles[plan.signature] = _MemoryProfile(
        2 * cold / plan.packed_tokens, plan.packed_tokens
    )
    assert rank._memory_check(plan).estimated_required_bytes > cold


def _hybrid_rank(monkeypatch: pytest.MonkeyPatch, tp: int) -> TrainerRank:
    rank = _rank(
        "selective",
        sequence_parallel=True,
        bias_activation_fusion=True,
        attention_output_gate=True,
        linear_num_key_heads=16,
        linear_key_head_dim=128,
        linear_num_value_heads=48,
        linear_value_head_dim=128,
    )
    rank._gdn_layers = 48
    monkeypatch.setattr(rank, "_topology_key", lambda: (1, tp, 1, 1))
    return rank


@pytest.mark.parametrize(
    "tp,peak", [(1, 33758228992), (2, 19541553664), (4, 11132523008)]
)
def test_sharded_floor_covers_recorded_native_gdn_peaks(monkeypatch, tp, peak):
    # H200, 64-layer Qwen3.8-27B, LoRA r1, SP, cold/warm max, 2,048 tokens.
    # First unsharded campaign, linked from dev/trainer_rank_recompute_memory.md.
    rank = _hybrid_rank(monkeypatch, tp)
    estimate = rank._memory_check(_plan(rank, tokens=2048)).estimated_required_bytes
    assert peak <= estimate <= 1.2 * peak


@pytest.mark.parametrize(
    "tp,mlp,packed,logical,peak_gib",
    [
        (4, False, 4096, 4096, 20.613),
        (4, False, 6964, 8192, 35.090),
        (4, False, 13928, 16384, 69.924),
        (8, False, 4096, 4096, 14.529),
        (8, False, 6968, 8192, 24.654),
        (8, False, 13928, 16384, 48.950),
        (4, True, 4096, 4096, 11.213),
        (4, True, 6964, 8192, 19.006),
        (4, True, 13928, 16384, 37.881),
    ],
)
def test_hybrid_estimate_is_close_to_recorded_peaks(
    monkeypatch, tp, mlp, packed, logical, peak_gib
):
    # Native H200 cold/warm witnesses in the calibration report. Coverage alone
    # would allow the former 60%-high estimate; also check useful admission.
    rank = _hybrid_rank(monkeypatch, tp)
    rank._recompute_modules = frozenset(("core_attn", "mlp") if mlp else ("core_attn",))
    plan = replace(_plan(rank, tokens=packed), output_bytes=logical * 5120 * 2)
    estimate = rank._memory_check(plan).estimated_required_bytes / 2**30
    assert peak_gib <= estimate <= 1.15 * peak_gib


def test_native_fusion_discount_is_independent_of_compilation(monkeypatch):
    rank = _hybrid_rank(monkeypatch, 4)
    plan = _plan(rank, tokens=4096)
    fused = rank._memory_check(plan).estimated_required_bytes
    rank.runtime.transformer_layers_compiled = False
    assert rank._memory_check(plan).estimated_required_bytes == fused
    rank._mlp_activation_factor = 5
    assert rank._memory_check(plan).estimated_required_bytes > fused


def test_mlp_discount_requires_selective_and_keeps_one_live_workspace(monkeypatch):
    rank = _hybrid_rank(monkeypatch, 4)
    rank._recompute_granularity = None
    plan = _plan(rank)
    unrecomputed = rank._memory_check(plan).estimated_required_bytes
    rank._recompute_modules = frozenset(("mlp",))
    assert rank._memory_check(plan).estimated_required_bytes == unrecomputed
    rank._recompute_granularity = "selective"
    assert rank._memory_check(plan).estimated_required_bytes < unrecomputed
    rank._num_layers = 1
    checkpointed = rank._memory_check(plan).estimated_required_bytes
    rank._recompute_modules = frozenset()
    assert rank._memory_check(plan).estimated_required_bytes == checkpointed


def test_tp4_admits_eight_k_sibling_pair_and_prices_actual_layer_mix(monkeypatch):
    rank = _hybrid_rank(monkeypatch, 4)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 119 * 2**30)
    plan = replace(
        _plan(rank, tokens=13928), logical_tokens=16384, output_bytes=16384 * 5120 * 2
    )
    check = rank._memory_check(plan)
    assert check.fits
    rank._gdn_layers = 64
    assert (
        rank._memory_check(plan).estimated_required_bytes
        > check.estimated_required_bytes
    )
    rank._gdn_layers = 0
    assert (
        rank._memory_check(plan).estimated_required_bytes
        < check.estimated_required_bytes
    )


def test_sp_discount_excludes_gathered_lora_inputs(monkeypatch):
    rank = _hybrid_rank(monkeypatch, 4)
    plan = _plan(rank, tokens=2048)
    tp4 = rank._memory_check(plan).estimated_required_bytes
    rank._sequence_parallel = False
    assert rank._memory_check(plan).estimated_required_bytes > tp4
    monkeypatch.setattr(rank, "_topology_key", lambda: (1, 1, 1, 1))
    assert tp4 > rank._memory_check(plan).estimated_required_bytes / 4


@pytest.mark.parametrize(
    "packed,logical,peak",
    [
        (436, 512, 829413888),  # Cold 256-token pair before adding workspace.
        (1742, 2048, 3043651584),
        (6964, 8192, 11941280768),
        (13928, 16384, 23861987840),
        (27854, 32768, 47563188736),
    ],
)
def test_gathered_inputs_cover_attention_only_cold_peak(
    monkeypatch, packed, logical, peak
):
    # Native eager Qwen3-1.7B TP2 paired requests at d31f9423, max across ranks
    # and cold/warm repetitions. Four FFN widths missed all four peaks.
    rank = _rank(
        "selective",
        hidden_size=2048,
        ffn_hidden_size=6144,
        num_layers=28,
        num_attention_heads=16,
        num_query_groups=8,
        kv_channels=128,
        sequence_parallel=True,
    )
    monkeypatch.setattr(rank, "_topology_key", lambda: (1, 2, 1, 1))
    plan = replace(
        _plan(rank, tokens=packed),
        logical_tokens=logical,
        output_bytes=logical * 2048 * 2,
    )
    assert rank._memory_check(plan).estimated_required_bytes >= peak


def test_non_full_estimate_covers_retained_gated_mlp_tensors() -> None:
    # Selective core-attention recompute leaves this MLP graph live. Count
    # distinct saved activation storage, excluding model parameters/views.
    layers = [
        torch.nn.Sequential(
            torch.nn.LayerNorm(16),
            torch.nn.Linear(16, 256),
            torch.nn.GLU(),
            torch.nn.Linear(128, 16),
        )
        for _ in range(8)
    ]
    parameters = {
        parameter.untyped_storage().data_ptr()
        for layer in layers
        for parameter in layer.parameters()
    }

    def retained(full: bool) -> int:
        saved: dict[int, int] = {}

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            storage = tensor.untyped_storage()
            if storage.data_ptr() not in parameters:
                saved[storage.data_ptr()] = storage.nbytes()
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            value = torch.zeros(32, 16, requires_grad=True)
            for layer in layers:
                value = (
                    checkpoint(layer, value, use_reentrant=False)
                    if full
                    else layer(value)
                )
        return sum(saved.values())

    rank = _rank(
        "selective",
        dtype=torch.float32,
        hidden_size=16,
        ffn_hidden_size=128,
        num_layers=8,
        num_attention_heads=2,
        kv_channels=8,
    )
    assert (
        retained(True)
        < retained(False)
        <= rank._memory_check(_plan(rank, tokens=32)).estimated_required_bytes
    )


def test_moe_discount_uses_effective_native_checkpoint_count(monkeypatch):
    rank = _rank(
        "selective",
        sequence_parallel=True,
        num_moe_experts=256,
        moe_router_topk=8,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        recompute_modules=["moe"],
    )
    monkeypatch.setattr(rank, "_topology_key", lambda: (1, 4, 1, 1))
    plan = _plan(rank, tokens=4096)
    # A module list alone cannot guarantee the native MoE checkpoint is active.
    undiscounted = rank._memory_check(plan).estimated_required_bytes
    rank._checkpointed_moe_layers = rank._num_layers
    assert rank._memory_check(plan).estimated_required_bytes < undiscounted
    rank._recompute_granularity = None
    assert rank._memory_check(plan).estimated_required_bytes == undiscounted


def test_short_hybrid_pairs_pay_for_recurrent_states(monkeypatch):
    rank = _hybrid_rank(monkeypatch, 4)
    requests = [
        ForwardInput(input_tokens=torch.arange(64) + offset, hidden_states=True)
        for offset in (0, 100)
    ]
    plan = rank._plan_flat_forward(requests)
    assert plan.grad_segment_count == 2
    estimate = rank._memory_check(plan).estimated_required_bytes
    # Cold eager TP4 at 45297a4af missed this peak without segment states.
    assert 892974592 <= estimate <= 1.2 * 892974592
    assert rank._plan_cost(plan).required == estimate
    rank._memory_profiles[plan.signature] = _MemoryProfile(0, plan.packed_tokens)
    assert rank._memory_check(plan).estimated_required_bytes == estimate
    inactive = rank._plan_flat_forward(
        requests + [ForwardInput(input_tokens=torch.arange(1000))]
    )
    assert inactive.grad_segment_count == 2
    assert rank._memory_check(inactive).estimated_required_bytes == estimate


@pytest.mark.parametrize("granularity", (None, "selective"))
def test_cp_prices_uneven_local_tokens_and_preserves_segment_states(
    monkeypatch, granularity
):
    rank = _hybrid_rank(monkeypatch, 2)
    rank._recompute_granularity = granularity
    monkeypatch.setattr(rank, "_topology_key", lambda: (1, 2, 4, 1))
    monkeypatch.setattr(rank, "_topology", lambda: SimpleNamespace(cp=4, tp=2))
    # An uneven plan puts 3/4 of the rows on one rank, including TP padding.
    monkeypatch.setattr(
        rank, "_max_rank_model_tokens", lambda batch, **_: batch.tokens.numel() * 3 // 4
    )
    requests = [ForwardInput(input_tokens=torch.arange(4097), hidden_states=True)]
    plan = rank._plan_flat_forward(requests)
    assert rank._plan_retained_tokens(plan) == 3074
    estimate = rank._memory_check(plan).estimated_required_bytes
    assert rank._plan_cost(plan).required == estimate

    def price(tokens=None, segments=plan.grad_segment_count):
        return rank._estimate_required_memory_bytes_from_values(
            packed_tokens=plan.packed_tokens,
            output_bytes=plan.output_bytes,
            signature=plan.signature,
            gdn_segments=segments,
            retained_tokens=tokens,
        )

    assert price(1026) < estimate < price()
    state_costs = [
        price(n) - price(n, segments=0) for n in (1026, 3074, plan.packed_tokens)
    ]
    assert max(state_costs) - min(state_costs) <= 1
    assert min(state_costs) > 0
    rank._memory_profiles[plan.signature] = _MemoryProfile(0, plan.packed_tokens)
    assert rank._memory_check(plan).estimated_required_bytes == estimate
    # Width selection must reach exact CP pricing even when the old global
    # token bound would refuse. Both ordinary and memory-minimal probes defer.
    for exact in (False, True):
        assert rank._estimate_flat_forward(requests, exact=exact) is None
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: estimate)
    assert rank._search_next_micro_batch(requests, 0).check.fits
    from art.trainer_rank._impl import Unset

    lower = rank._split_chunk_lower_cost(
        requests, [requests[0].input_tokens], checkpoint=Unset
    )
    assert lower.required <= estimate


@pytest.mark.parametrize("kind", ("full", "no_grad", "moe"))
def test_cp_keeps_existing_full_no_grad_and_moe_costs(monkeypatch, kind):
    rank = _rank(
        "full" if kind == "full" else "selective",
        **({"num_moe_experts": 64} if kind == "moe" else {}),
    )

    monkeypatch.setattr(rank, "_topology_key", lambda: (1, 1, 2, 1))
    monkeypatch.setattr(rank, "_topology", lambda: SimpleNamespace(cp=2, tp=1))
    monkeypatch.setattr(
        rank, "_max_rank_model_tokens", lambda batch, **_: batch.tokens.numel() - 1
    )
    plan = _plan(rank, no_grad=kind == "no_grad")
    # Retention stays global for these kinds; floors use the most loaded rank.
    assert rank._plan_retained_tokens(plan) == plan.packed_tokens
    assert rank._plan_group_rows(plan) == tuple(
        (int(group.packed.tokens.numel()) - 1, group.grad_enabled)
        for group in plan.groups
    )


def test_cp_floor_covers_recorded_uneven_27b_pair(monkeypatch):
    # Native H200 CP2, 4k siblings: rank 0 peaks at 62.918 GiB with 4,096
    # local tokens. Dividing the 6,964 global packed tokens by CP misses it.
    rank = _hybrid_rank(monkeypatch, 1)
    signature = replace(_plan(rank).signature, topology=(1, 1, 2, 1))
    values: dict[str, Any] = dict(
        packed_tokens=6964,
        output_bytes=8192 * 5120 * 2,
        signature=signature,
        gdn_segments=3,
    )
    peak = 62.918 * 2**30
    estimate = rank._estimate_required_memory_bytes_from_values(
        **values, retained_tokens=4096
    )
    assert peak <= estimate <= 1.12 * peak
    assert (
        rank._estimate_required_memory_bytes_from_values(**values, retained_tokens=3482)
        < peak
    )
