"""CPU admission contracts; injected observations are not GPU peak measurements."""

import builtins
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from art.trainer_rank import (
    ForwardInput,
    ForwardOutput,
    TrainerRank,
    TrainerRankMemoryError,
    Unset,
    _impl,
)
from art.trainer_rank._impl import (
    _PACKED_PRICED_LOGICAL_ROW_BYTES,
    _packed_priced,
    _request_mix_key,
)


@pytest.fixture(autouse=True)
def _price_short_requests(monkeypatch):
    # These fixtures use short requests; the short-request gate has its own tests.
    monkeypatch.setattr(_impl, "_PACKED_PRICED_MIN_REQUEST_TOKENS", 1)


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.config = SimpleNamespace(hidden_size=8, num_layers=4, padded_vocab_size=32)
        self.decoder = object()

    def _preprocess(self, *args, **kwargs):
        return None


def _rank():
    return TrainerRank(
        cast(
            Any,
            SimpleNamespace(
                model=[_Model()],
                optimizer=None,
                provider=SimpleNamespace(
                    hidden_size=8,
                    num_layers=4,
                    recompute_granularity="full",
                    recompute_method="uniform",
                    recompute_num_layers=1,
                ),
                model_support_handler=SimpleNamespace(build_gdn_execution_spec=False),
            ),
        )
    )


def _requests(output="hidden_states", inactive_length=1):
    tokens = torch.arange(8)
    active = {
        "hidden_states": ForwardInput(input_tokens=tokens, hidden_states=True),
        "target_tokens": ForwardInput(input_tokens=tokens, target_tokens=tokens),
        "logits": ForwardInput(input_tokens=tokens, logits=True),
        "top_k": ForwardInput(input_tokens=tokens, top_k=2),
    }[output]
    return [
        active,
        ForwardInput(input_tokens=torch.arange(inactive_length)),
    ]


@pytest.mark.parametrize(
    "output", ["hidden_states", "target_tokens", "logits", "top_k"]
)
@pytest.mark.parametrize("no_grad", [False, True])
def test_inactive_length_preserves_warm_cost_and_profile(monkeypatch, output, no_grad):
    rank = _rank()
    short = [replace(r, no_grad=no_grad) for r in _requests(output)]
    long = [replace(r, no_grad=no_grad) for r in _requests(output, 8001)]
    plans = [rank._plan_flat_forward(requests) for requests in (short, long)]
    first, second = plans
    assert first.signature == second.signature
    assert first.packed_tokens == second.packed_tokens == 8
    assert first.output_bytes == second.output_bytes
    assert first.output_metadata == second.output_metadata
    assert (first.logical_tokens, second.logical_tokens) == (9, 8009)
    assert len(first.groups) == len(second.groups) == 1
    a, b = first.groups[0], second.groups[0]
    assert a.request_indices == b.request_indices == (0,)
    for field in ("tokens", "group_ids", "parent_ids", "position_ids"):
        assert torch.equal(getattr(a.packed, field), getattr(b.packed, field))
    rank._update_memory_profile(first, 10_000, retained_bytes=1000)
    profile = rank._memory_profiles[first.signature]
    budget = rank._memory_check(first).estimated_required_bytes
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: budget)
    assert rank._memory_check(first).fits
    assert rank._memory_check(second) == rank._memory_check(first)
    assert rank._plan_cost(first) == rank._plan_cost(second)
    for requests, plan in zip((short, long), plans, strict=True):
        lower = rank._split_chunk_lower_cost(
            requests, [r.input_tokens for r in requests], checkpoint=Unset
        )
        assert lower == rank._plan_cost(plan)
    rank._update_memory_profile(second, 10_000, retained_bytes=1000)
    assert rank._memory_profiles[first.signature] == profile


def test_public_pair_avoids_inactive_only_split_and_keeps_total_telemetry(monkeypatch):
    rank = _rank()
    monkeypatch.setattr(rank, "_dp_rank_and_size", lambda: (0, 1))
    observed = rank._plan_flat_forward(_requests())
    rank._update_memory_profile(observed, 10_000, retained_bytes=1000)
    budget = rank._memory_check(observed).estimated_required_bytes
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: budget)
    executed = []

    def run(plan, **kwargs):
        executed.append(plan)
        # Only model execution is replaced. Admission, split search, trust and
        # public iterator reconstruction all run their production paths.
        return [
            ForwardOutput(None, None, None, None) for _ in range(plan.request_count)
        ], None

    monkeypatch.setattr(rank, "_run_flat_plan_with_memory_tracking", run)
    batches = list(rank.forward_micro_batches([_requests(inactive_length=8001)]))
    assert len(batches) == 1
    batch = batches[0]
    assert batch.indices == (0,)
    assert batch.stats.global_count == batch.stats.local_count == 1
    assert batch.stats.logical_tokens == 8009
    assert batch.stats.packed_tokens == 8
    assert len(batch.outputs) == 1 and len(batch.outputs[0]) == 2
    assert batch.stats.subforward_count == len(executed) == 1
    assert batch.stats.estimated_required_bytes == budget
    assert not batch.stats.cold_start


def test_active_length_still_increases_warm_cost():
    rank = _rank()
    short = _requests()
    observed = rank._plan_flat_forward(short)
    rank._update_memory_profile(observed, 10_000, retained_bytes=1000)
    longer = rank._plan_flat_forward(
        [replace(short[0], input_tokens=torch.arange(16)), short[1]]
    )
    assert observed.signature == longer.signature
    assert rank._memory_check(longer).estimated_required_bytes == 22_000
    assert rank._plan_cost(longer).retained > rank._plan_cost(observed).retained


def test_inactive_observation_cannot_discount_later_shared_active_work(monkeypatch):
    ranks = [_rank(), _rank()]
    checks = []
    for rank, inactive_length in zip(ranks, (1, 8001), strict=True):
        requests = _requests(inactive_length=inactive_length)
        observed = rank._plan_flat_forward(requests, memory_minimal=True)
        rank._update_memory_profile(observed, 10_000, retained_bytes=1000)
        candidate = rank._plan_flat_forward(
            [requests[0]] * 8 + _requests()[1:], memory_minimal=True
        )
        assert candidate.signature == observed.signature
        assert candidate.packed_tokens == observed.packed_tokens == 8
        assert candidate.logical_tokens == 65
        monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 20_000)
        checks.append(rank._memory_check(candidate))
    # Identical observed GPU work must produce identical future admission.
    # Total-input ratios formerly discounted the second estimate to 11,985,
    # admitting a plan that the equivalent short calibration refused.
    assert checks[0] == checks[1]
    assert checks[0].estimated_required_bytes == 88_000
    assert not checks[0].fits


def test_warm_admission_rechecks_current_residency(monkeypatch):
    rank = _rank()
    plan = rank._plan_flat_forward(_requests())
    rank._update_memory_profile(plan, 10_000, retained_bytes=1000)
    required = rank._memory_check(plan).estimated_required_bytes
    profile = rank._memory_profiles[plan.signature]
    state = {"allocated": 10_000, "reserved": 10_000}
    total = 100_000
    monkeypatch.setattr(rank, "device", torch.device("cuda"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda _: (total - state["reserved"], total)
    )
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: state["allocated"])
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: state["reserved"])
    monkeypatch.setattr(torch.cuda, "get_allocator_backend", lambda: "native")
    monkeypatch.setattr(
        torch.cuda,
        "memory_stats",
        lambda _: {
            "allocated_bytes.all.current": state["allocated"],
            "active_bytes.all.current": state["allocated"],
            "reserved_bytes.all.current": state["reserved"],
        },
    )
    monkeypatch.delenv("ART_TRAINER_RANK_TEST_HOOKS", raising=False)
    # Fresh memory accounting observes newly resident state without discarding
    # a valid incremental profile; cache alone is not physical availability.
    for allocated, reserved, fits in [
        (10_000, 10_000, True),
        (90_000, 90_000, False),
        (10_000, 90_000, False),
    ]:
        state.update(allocated=allocated, reserved=reserved)
        check = rank._memory_check(plan)
        assert check.estimated_required_bytes == required
        assert check.available_bytes == total - reserved - 3000
        assert check.fits == fits
        assert rank._memory_profiles[plan.signature] == profile


def test_distinct_output_and_pure_grad_modes_require_their_own_profile():
    rank = _rank()
    requests = _requests()
    observed = rank._plan_flat_forward(requests)
    rank._update_memory_profile(observed, 10_000, retained_bytes=1000)
    for changed in (
        [replace(r, no_grad=True) for r in requests],
        _requests("logits"),
        _requests("target_tokens"),
        _requests("top_k"),
    ):
        plan = rank._plan_flat_forward(changed)
        assert plan.signature != observed.signature
        assert not rank._all_ranks_have_memory_profile(
            packed_tokens=plan.packed_tokens, signature=plan.signature
        )


@pytest.mark.parametrize("no_grad", [False, True])
def test_direct_forward_does_not_drop_observed_peak_outside_trust(monkeypatch, no_grad):
    rank = _rank()

    def request(length):
        tokens = torch.arange(length)
        return ForwardInput(input_tokens=tokens, target_tokens=tokens, no_grad=no_grad)

    observed = rank._plan_flat_forward([request(10)])
    rank._update_memory_profile(observed, 10_000, retained_bytes=1000)
    monkeypatch.setattr(rank, "_available_memory_bytes", lambda: 8000)

    def unexpected_execution(*args, **kwargs):
        raise AssertionError("An unsafe single request reached model execution")

    monkeypatch.setattr(
        rank, "_run_flat_plan_with_memory_tracking", unexpected_execution
    )
    for length in (80, 81):
        candidate = rank._plan_flat_forward([request(length)])
        assert rank._all_ranks_have_memory_profile(
            packed_tokens=length, signature=candidate.signature
        ) == (length == 80)
        # Calibration trust still ends at 8x. Crossing it must not discard a
        # peak already observed at a smaller size and admit the larger request.
        with pytest.raises(
            TrainerRankMemoryError, match="single request cannot be split"
        ):
            rank.dp_rank_forward([request(length)])
        assert rank.last_forward_telemetry()["predicted_peak_bytes"] >= 10_000


@pytest.mark.parametrize("logical_ratio", [1, 2, 10])
def test_empirical_estimate_survives_packed_trust_boundary(logical_ratio):
    rank = _rank()
    observed = rank._plan_flat_forward(_requests("target_tokens"))
    rank._update_memory_profile(observed, 100_000, retained_bytes=1000)
    estimate = rank._estimate_required_memory_bytes_from_values
    values = [
        estimate(
            packed_tokens=count,
            logical_tokens=count * logical_ratio,
            output_bytes=count * 4,
            signature=observed.signature,
        )
        for count in (8, 63, 64, 65, 800)
    ]
    assert values == sorted(values)
    profile = rank._memory_profiles[observed.signature]
    logical = 800 * logical_ratio
    packed = max(800, logical / profile.logical_per_packed / 8)
    rows = _PACKED_PRICED_LOGICAL_ROW_BYTES * logical
    assert values[-1] == int((800 * 4 + profile.bytes_per_token * packed + rows) * 1.1)
    assert not rank._all_ranks_have_memory_profile(
        packed_tokens=800, signature=observed.signature
    )


def test_packed_pricing_covers_allocator_blocks_of_fully_shared_requests():
    # A duplicate one-token request adds no packed row but still allocates its
    # head buffers, each rounded up to a 512 B block: nine at the measured peak.
    # (The short-request gate keeps such requests out; the charge covers them.)
    rank = _rank()
    observed = rank._plan_flat_forward(_requests("target_tokens"))
    rank._update_memory_profile(observed, 100_000, retained_bytes=1000)
    profile = rank._memory_profiles[observed.signature]
    assert _packed_priced(observed.signature, rank._one_layer_recompute())
    duplicates = 100_000

    def estimate(logical_tokens: int) -> int:
        return rank._estimate_required_memory_bytes_from_values(
            packed_tokens=1,
            logical_tokens=logical_tokens,
            output_bytes=0,
            signature=observed.signature,
        )

    base = profile.logical_per_packed
    assert estimate(base + duplicates) - estimate(base) >= duplicates * 9 * 512


@pytest.mark.parametrize("shape", ["flat", "row"])
def test_short_single_target_requests_keep_logical_pricing(monkeypatch, shape):
    monkeypatch.setattr(_impl, "_PACKED_PRICED_MIN_REQUEST_TOKENS", 64)

    def request(length: int) -> ForwardInput:
        tokens = torch.arange(length)
        if shape == "row":
            tokens = tokens[None]
        return ForwardInput(input_tokens=tokens, target_tokens=tokens + 1)

    assert [_impl._short_request(request(n)) for n in (63, 64, 65)] == [
        True,
        False,
        False,
    ]
    assert not _impl._short_request(ForwardInput(input_tokens=torch.arange(3)))
    rank = _rank()
    recompute = rank._one_layer_recompute()
    # Calibrate on a 64-token request, then price a short one: the short batch
    # shares the calibrated profile (no cold start) but keeps main's pricing.
    long_plan = rank._plan_flat_forward([request(64)])
    rank._update_memory_profile(long_plan, 64 * 100_000, retained_bytes=None)
    for requests in ([request(63)], [request(64), request(63)]):
        short = rank._plan_flat_forward(requests)
        assert short.signature.short_requests and short.signature == long_plan.signature
        assert _packed_priced(long_plan.signature, recompute)
        assert not _packed_priced(short.signature, recompute)
        profile = rank._memory_profiles[short.signature]
        cost = rank._plan_cost(short)
        extrapolated = profile.bytes_per_token * max(
            short.packed_tokens,
            short.active_logical_tokens / profile.logical_per_packed,
        )
        assert cost.required >= int((short.output_bytes + extrapolated) * 1.1)


def test_packed_sharing_clamp_is_monotone_and_learning_sharing_never_cheapens():
    rank = _rank()
    observed = rank._plan_flat_forward(_requests("target_tokens"))
    rank._update_memory_profile(observed, 100_000, retained_bytes=1000)
    single = observed.signature
    rank._memory_profiles[single] = replace(
        rank._memory_profiles[single], bytes_per_token=50_000, logical_per_packed=1
    )

    def cost(packed: int, logical: int = 8_000):
        return rank._subforward_cost(
            packed_tokens=packed,
            logical_tokens=logical,
            output_bytes=0,
            signature=single,
        ).required

    # Fixed logical rows; packed rows sweep across L / 8r = 1000.
    sweep = [cost(packed) for packed in (1, 500, 999, 1000, 1001, 4000, 8000)]
    assert sweep == sorted(sweep)
    rows = _PACKED_PRICED_LOGICAL_ROW_BYTES * 8_000
    assert sweep[0] == sweep[3] == int((50_000 * 1000 + rows) * 1.1)
    assert sweep[4] == int((50_000 * 1001 + rows) * 1.1)
    # Learning more sharing at a lower rate max-merges both; plans at
    # or below the older ratio never get cheaper.
    before = [cost(packed, logical) for packed, logical in ((100, 100), (50, 100))]
    wider = replace(observed, packed_tokens=1, logical_tokens=8)
    rank._update_memory_profile(wider, 1_000, retained_bytes=None)
    profile = rank._memory_profiles[single]
    assert profile.logical_per_packed > 1 and profile.bytes_per_token == 50_000
    after = [cost(packed, logical) for packed, logical in ((100, 100), (50, 100))]
    assert all(b >= a for a, b in zip(before, after, strict=True))


@pytest.mark.parametrize(
    "method,argument,fallback",
    [
        ("_head_workspace_bytes", 8, 0),
        ("_checkpoint_memory_floor", ((8, True),), (0, 0)),
    ],
)
@pytest.mark.parametrize(
    "error,unavailable",
    [
        (ModuleNotFoundError("absent package", name="megatron"), True),
        (ModuleNotFoundError("missing dependency", name="transformer_engine"), False),
        (ModuleNotFoundError("partial installation", name="megatron.core"), False),
        (ModuleNotFoundError("unspecified missing module"), False),
        (ImportError("missing imported class"), False),
        (RuntimeError("module initialization failed"), False),
    ],
    ids=["absent", "transitive", "partial", "unspecified", "class", "runtime"],
)
def test_optional_megatron_memory_guards(
    monkeypatch, method, argument, fallback, error, unavailable
):
    rank = _rank()
    original_import = builtins.__import__

    def importing(name, *args, **kwargs):
        if name.partition(".")[0] == "megatron":
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", importing)
    if unavailable:
        assert getattr(rank, method)(argument) == fallback
    else:
        with pytest.raises(type(error)) as caught:
            getattr(rank, method)(argument)
        assert caught.value is error


def test_packed_pricing_is_limited_to_grad_single_target_mixes():
    rank = _rank()
    observed = rank._plan_flat_forward(_requests("target_tokens"))
    rank._update_memory_profile(observed, 10_000, retained_bytes=1000)
    single = observed.signature
    assert single.grad_modes == (True,)
    profile = replace(
        rank._memory_profiles[single],
        bytes_per_token=100_000,
        retained_compute_bytes_per_token=50_000,
        logical_per_packed=2,
    )
    signatures = {
        "single": single,
        "multi_grad": replace(single, grad_modes=(True, True)),
        "no_grad": replace(single, grad_modes=(False,)),
        "mixed_grad": replace(single, grad_modes=(False, True)),
        "hidden": replace(single, request_mix=("hidden",)),
        "wide": replace(single, request_mix=("target:(2,)",)),
    }
    for signature in signatures.values():
        rank._memory_profiles[signature] = profile

    def estimate(name):
        return rank._estimate_required_memory_bytes_from_values(
            packed_tokens=8,
            logical_tokens=64,
            output_bytes=0,
            signature=signatures[name],
        )

    # Packed rows plus head and caller memory for every logical row.
    rows = _PACKED_PRICED_LOGICAL_ROW_BYTES * 64
    assert (
        estimate("single") == estimate("multi_grad") == int((100_000 * 8 + rows) * 1.1)
    )
    retained = rank._retained_memory_bytes(
        single, packed_tokens=8, logical_tokens=64, output_bytes=0, required=1 << 40
    )
    assert retained == int((50_000 * 8 + rows) * 1.1)
    # Others keep the logical/packed extrapolation: 64 / 2 profiled rows.
    for name in ("no_grad", "mixed_grad", "hidden", "wide"):
        assert estimate(name) == int(100_000 * 32 * 1.1)
    # Beyond 8x the profile's sharing, packed rows are priced as if sharing
    # were 8x: 64 logical rows at ratio 2 x 8 = 16 is 4 rows, not 1.
    clamped = rank._estimate_required_memory_bytes_from_values(
        packed_tokens=1, logical_tokens=64, output_bytes=0, signature=single
    )
    assert clamped == int((100_000 * 4 + rows) * 1.1)
    # The logical charge does not depend on layout, so at any rate (none gates
    # eligibility) cost never falls as packed rows grow.
    for rate in (1, 2 * _PACKED_PRICED_LOGICAL_ROW_BYTES * 2 - 1, 100_000):
        rank._memory_profiles[single] = replace(
            profile, bytes_per_token=rate, retained_compute_bytes_per_token=rate
        )
        costs = [
            rank._subforward_cost(
                packed_tokens=packed,
                logical_tokens=64,
                output_bytes=0,
                signature=single,
            )
            for packed in range(1, 65)
        ]
        assert all(a.required <= b.required for a, b in zip(costs, costs[1:]))
        assert all(a.retained <= b.retained for a, b in zip(costs, costs[1:]))
    # GDN branch states grow with segments, not packed rows.
    rank._memory_profiles[single] = profile
    rank._geometry = replace(
        rank._geometry,
        gdn_key_heads=1,
        gdn_key_head_dim=4,
        gdn_value_heads=2,
        gdn_value_head_dim=4,
        gdn_conv_kernel=4,
    )
    # 2 states x 4 B x Hv=2 x dk=4 x dv=4, plus 2 B conv history of 16 rows x 3.
    assert rank._gdn_segment_layer_bytes() == 2 * (4 * 2 * 4 * 4 + 2 * 16 * 3)
    live = 1

    def segments(count):
        return rank._estimate_required_memory_bytes_from_values(
            packed_tokens=8,
            logical_tokens=64,
            output_bytes=0,
            signature=single,
            gdn_segments=count,
        )

    assert segments(3) - segments(0) == pytest.approx(
        3 * live * rank._gdn_segment_layer_bytes() * 1.1, abs=1
    )
    # Other recompute modes, and eval mode (which skips recompute), keep more
    # live per segment and row: extrapolate.
    for setting in (
        ("selective", "uniform", 1),
        (None, None, None),
        ("full", "block", 1),
        ("full", "uniform", 2),
        ("full", None, None),
    ):
        (
            rank._recompute_granularity,
            rank._recompute_method,
            rank._recompute_num_layers,
        ) = setting
        assert not rank._one_layer_recompute()
        assert estimate("single") == estimate("hidden")
    rank._recompute_granularity, rank._recompute_method = "full", "uniform"
    rank._recompute_num_layers = 1
    assert rank._one_layer_recompute()
    rank.runtime.model[0].eval()
    assert not rank._one_layer_recompute()
    assert estimate("single") == estimate("hidden")
    rank.runtime.model[0].train()
    # Megatron checks the decoder's own mode, which can diverge from the chunk.
    rank.runtime.model[0].decoder = torch.nn.Module()
    rank.runtime.model[0].decoder.eval()
    assert rank.runtime.model[0].training and not rank._one_layer_recompute()
    assert estimate("single") == estimate("hidden")
    rank.runtime.model[0].decoder.train()
    assert rank._one_layer_recompute()
    # With a decoder config, its live settings and the decoder's mode decide.
    decoder = rank.runtime.model[0].decoder
    decoder.config = SimpleNamespace(
        recompute_granularity="full", recompute_method="uniform", recompute_num_layers=1
    )
    rank.runtime.model[0].eval()
    decoder.train()
    assert rank._one_layer_recompute()
    rank.runtime.model[0].train()
    decoder.config.recompute_num_layers = 2
    assert not rank._one_layer_recompute()
    decoder.config.recompute_num_layers = 1
    assert rank._one_layer_recompute()
    # Replay trusts the recorded mode instead of a live model.
    packed = estimate("single")
    rank._recorded_one_layer_recompute = False
    assert estimate("single") == estimate("hidden") != packed
    rank._recorded_one_layer_recompute = True
    assert estimate("single") == packed
    del rank._recorded_one_layer_recompute
    # Flattened-axis wide labels are not single-target.
    tokens = torch.arange(4).reshape(1, 4)
    wide = ForwardInput(input_tokens=tokens, target_tokens=torch.zeros(4, 3).long())
    assert _request_mix_key(wide) == "target:(3,)"
