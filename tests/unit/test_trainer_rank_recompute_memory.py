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
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: int(119.289e9))
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
    # Source evidence: dev/trainer_rank_recompute_memory.csv at 22f628d6.
    rank = _hybrid_rank(monkeypatch, tp)
    assert rank._memory_check(_plan(rank, tokens=2048)).estimated_required_bytes >= peak


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
