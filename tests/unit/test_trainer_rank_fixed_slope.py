"""CPU regressions for fixed workspace spread over tiny waves."""

from collections import defaultdict
from dataclasses import asdict, replace
import itertools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from test_trainer_rank_active_memory import _rank
import torch

from art.trainer_rank import ForwardInput, TrainerRankMemoryError, _impl, _memory
from art.trainer_rank import _memory_model as model
from art.trainer_rank._memory import fixed_slope_memory_profiles


@pytest.fixture(autouse=True)
def prototype():
    with fixed_slope_memory_profiles():
        yield


def test_tiny_forward_does_not_poison_large_forward():
    rank = _rank()
    seed = rank._plan_flat_forward(
        [ForwardInput(input_tokens=torch.arange(8), hidden_states=True, no_grad=True)]
    )
    # Synthetic eight-token workspace regression with a fixed long-wave
    # reference; the large rate deliberately stresses per-token extrapolation.
    poisoned_rate = (568_361_327_001 / 1.1 - 8192 * 4096) / 8192
    large = replace(seed, packed_tokens=81920, logical_tokens=81920, output_bytes=0)
    tiny = replace(seed, output_bytes=0)
    # Seed a long reference before adding the fixed workspace observation.
    rank._update_memory_profile(large, 8 * 2**30, retained_bytes=None)
    rank._update_memory_profile(
        tiny, round(poisoned_rate * tiny.packed_tokens), retained_bytes=None
    )
    required = rank._estimate_required_memory_bytes_from_values(
        packed_tokens=8192, output_bytes=8192 * 4096, signature=seed.signature
    )
    assert required < 61 * 2**30, f"tiny workspace poisoned prediction: {required}"


FIXTURES = Path(__file__).parent / "fixtures/memory_fixed_slope"
SWEEPS = json.loads((FIXTURES / "sweeps.json").read_text())
GROUPS = defaultdict(list)
for row in SWEEPS:
    GROUPS[(row["model"], row["operation"], row["lora_rank"])].append(row)


def signature(key=(0, "forward", 8)):
    _, op, lora = key
    return _impl._MemorySignature(
        topology=(1, 1, 8, 1),
        planner_coefficients=(1, None),
        slot_group_count=1,
        request_mix=("target:single",),
        grad_enabled=op == "forward_backward",
        grad_modes=(op == "forward_backward",),
        slot_shapes=((op == "forward_backward", ((2, 2048, lora, lora, 2048),)),),
    )


def observe(rank, sig, tokens, compute, *, caller=False):
    plan = cast(
        Any,
        SimpleNamespace(
            signature=sig,
            packed_tokens=tokens,
            active_logical_tokens=tokens,
            output_bytes=0,
        ),
    )
    _memory._update_memory_profile(
        rank, plan, compute, retained_bytes=None, caller_phase=caller
    )


@pytest.mark.parametrize("key,rows", list(GROUPS.items()))
def test_all_60_sweep_peaks_covered_in_every_observation_order(key, rows):
    # This tests the learned component and real scalar admission consumer;
    # injected peaks are supplied GPU evidence, not simulated GPU execution.
    expected = None
    r = _rank()
    sig = signature(key)
    for order in itertools.permutations(rows):
        r._memory_profiles.clear()
        for row in order:
            observe(r, sig, row["length"], row["measured_increment_bytes"])
        p = r._memory_profiles[sig]
        state = p.compute_observations, p.fixed_compute_bytes, p.compute_slope
        if expected is None:
            expected = state
        assert state == expected
        for row in rows:
            tokens = row["length"]
            raw = _memory._profile_compute_peak(p, tokens)
            assert raw >= row["measured_increment_bytes"]
            effective = r._estimate_required_memory_bytes_from_values(
                packed_tokens=tokens,
                output_bytes=0,
                signature=sig,
            )
            assert effective >= row["measured_increment_bytes"]
            assert all(effective >= peak for peak in row["rank_peak_bytes"])


def test_tiny_points_only_change_fixed_and_largest_anchor_is_deterministic():
    r = _rank()
    sig = signature()
    observe(r, sig, 8192, 2**30)
    slope = r._memory_profiles[sig].compute_slope
    observe(r, sig, 1, 2**30)
    assert r._memory_profiles[sig].compute_slope == slope
    assert r._memory_profiles[sig].fixed_compute_bytes > 0
    r = _rank()
    for n in (1, 128, 2048, 8191):
        observe(r, sig, n, 2**30)
        assert r._memory_profiles[sig].compute_slope == (0, 1)


def test_hull_is_associative_idempotent_and_preserves_every_constraint():
    points = (
        (1, 1000),
        (2, 1),
        (1024, 2000),
        (8192, 8000),
        (16384, 1000),
        (8192, 9000),
    )
    expected = model.merge_points(points)
    for order in itertools.permutations(points):
        hull = ()
        for point in order:
            hull = model.merge_points(hull, (point,))
        assert hull == expected
    assert model.merge_points(expected, expected) == expected
    for split in range(len(points)):
        assert (
            model.merge_points(
                model.merge_points(points[:split]), model.merge_points(points[split:])
            )
            == expected
        )
    for numerator in (0, 1, 100, 100000):
        assert max(y - numerator * x for x, y in expected) == max(
            y - numerator * x for x, y in points
        )
    fixed, slope = model.fit_points(expected)
    assert all(model.predict(fixed, slope, n) >= peak for n, peak in points)


@pytest.mark.parametrize(
    "profile_type",
    [_impl._MemoryProfile, _memory._CallerMemoryProfile, _memory._BoundedMemoryProfile],
)
def test_monotone_through_threshold_trust_extent_and_logical_rows(profile_type):
    r = _rank()
    sig = signature()
    for tokens, peak in ((1, 2**29), (8192, 2**30), (32768, 2**31)):
        observe(r, sig, tokens, peak)
    values = {**asdict(profile_type(1, 1)), **asdict(r._memory_profiles[sig])}
    if profile_type is _memory._CallerMemoryProfile:
        # Caller trust requires completed intervals in addition to token extents.
        values.update(
            caller_compute_peaks=((1, 2**29), (8192, 2**30), (32768, 2**31)),
            compute_peak_floor=2**31,
            caller_plans=3,
        )
    p = _memory._memory_profile_from_values(values)
    r._memory_profiles[sig] = p
    ns = [1, 2, 128, 8191, 8192, 8193, 32767, 32768, 32769, 8 * 32768, 8 * 32768 + 1]
    for sharing in (1, 2, 32):
        demands = [
            r._estimate_required_memory_bytes_from_values(
                packed_tokens=n,
                logical_tokens=n * sharing,
                output_bytes=n * 4,
                signature=sig,
                checkpoint_floor=(3 * 2**30, 4 * 2**30),
            )
            for n in ns
        ]
        assert demands == sorted(demands)
        assert min(demands) >= int(7 * 2**30 * 1.1)
    r._all_ranks_true = lambda x: x
    assert r._all_ranks_have_memory_profile(packed_tokens=8 * 32768, signature=sig)
    assert not r._all_ranks_have_memory_profile(
        packed_tokens=8 * 32768 + 1, signature=sig
    )


def test_v1_floor_is_only_inherited_by_explicit_legacy_merge():
    r = _rank()
    sig = signature()
    old = _impl._MemoryProfile(60_000_000, 8192)
    # A real v1 record has none of the new fields; defaults restore its model.
    assert (
        _impl._MemoryProfile(
            **json.loads(
                json.dumps(
                    {
                        "bytes_per_token": old.bytes_per_token,
                        "packed_tokens": old.packed_tokens,
                    }
                )
            )
        )
        == old
    )
    r._memory_profiles[sig] = old
    assert _memory._profile_compute_peak(old, 32768) == 60_000_000 * 32768
    observe(r, sig, 32768, 2**30)
    p = r._memory_profiles[sig]
    assert p.peak_model_version == 2 and p.legacy_bytes_per_token == 0
    assert p.bytes_per_token < old.bytes_per_token
    restored = replace(p, **_memory._merge_memory_profile_model_fields(old, p))
    assert restored.legacy_bytes_per_token == old.bytes_per_token
    assert _memory._profile_compute_peak(restored, 32768) >= old.bytes_per_token * 32768


def test_distributed_refusal_keeps_max_then_min_and_local_fit_predicate(monkeypatch):
    r = _rank()
    r.device = torch.device("cpu")
    r._forward_memory_group = lambda: None
    r._available_memory_bytes = lambda: 100
    calls = []
    monkeypatch.setattr(_impl.dist, "is_available", lambda: True)
    monkeypatch.setattr(_impl.dist, "is_initialized", lambda: True)

    def reduce(tensor, *, op, group):
        calls.append(op)
        if op == _impl.dist.ReduceOp.MAX:
            tensor.fill_(200)
        else:
            tensor.copy_(torch.tensor([80.0, 0.0], dtype=torch.float64))

    monkeypatch.setattr(_impl.dist, "all_reduce", reduce)
    check = r._memory_check_required(90)
    assert calls == [_impl.dist.ReduceOp.MAX, _impl.dist.ReduceOp.MIN]
    assert not check.fits
    assert (check.estimated_required_bytes, check.available_bytes) == (200, 80)
    assert (check.local_required_bytes, check.local_available_bytes) == (90, 100)


def test_v2_refuses_below_observed_peak_before_execution(monkeypatch):
    r = _rank()
    request = ForwardInput(
        input_tokens=torch.arange(8192), hidden_states=True, no_grad=True
    )
    plan = r._plan_flat_forward([request])
    observe(r, plan.signature, plan.packed_tokens, 2**30)
    monkeypatch.setattr(r, "_available_memory_bytes", lambda: 2**29)
    monkeypatch.setattr(
        r, "_execute_flat_plan", lambda _: pytest.fail("refused forward executed")
    )
    with pytest.raises(TrainerRankMemoryError):
        r.dp_rank_forward([request])


def test_restore_merge_unions_evidence_instead_of_maxing_coefficients():
    profiles = []
    for points in (((1, 1000), (8192, 2000)), ((32768, 3000), (128, 1200))):
        r = _rank()
        sig = signature()
        for n, peak in points:
            observe(r, sig, n, peak)
        profiles.append(r._memory_profiles[sig])
    fields = _memory._merge_memory_profile_model_fields(*profiles)
    reverse = _memory._merge_memory_profile_model_fields(*reversed(profiles))
    assert fields == reverse
    merged = replace(profiles[0], packed_tokens=32768, **fields)
    _memory._validate_memory_profile_model(merged)
    for p in profiles:
        for n, peak in p.compute_observations:
            assert _memory._profile_compute_peak(merged, n) >= peak
    assert merged.compute_slope == (3000, 32768)


@pytest.mark.parametrize(
    "mutation",
    [
        {"peak_model_version": 3},
        {"fixed_compute_bytes": -1},
        {"compute_slope": (1, 0)},
        {"compute_observations": ((8192, -1),)},
        {"compute_observations": ((0, 1),)},
        {"legacy_bytes_per_token": float("nan")},
    ],
)
def test_invalid_v2_persistence_is_rejected(mutation):
    r = _rank()
    sig = signature()
    observe(r, sig, 8192, 1000)
    with pytest.raises(ValueError):
        _memory._validate_memory_profile_model(
            replace(r._memory_profiles[sig], **mutation)
        )


@pytest.mark.parametrize("api", ["dp_rank_forward", "forward_micro_batches"])
@pytest.mark.parametrize("ep", [False, True])
@pytest.mark.parametrize("peer_veto", [False, True])
def test_collective_exhausted_reject_site_preserves_peer_scope(
    monkeypatch, api, ep, peer_veto
):
    from test_trainer_rank_ep_admission_scope import _exercise

    refused, executed, calls, local_group = _exercise(
        monkeypatch,
        api=api,
        ep=ep,
        allow=True,
        peer_veto=peer_veto,
        count=2,
    )
    local_ep = ep and api == "dp_rank_forward"
    accepted = not peer_veto and not local_ep
    assert refused is not accepted
    assert bool(executed) is accepted
    if local_ep:
        assert all(group is local_group for group, _, _ in calls)
    else:
        votes = [
            group
            for group, values, op in calls
            if len(values) == 3 and op == _impl.dist.ReduceOp.MIN
        ]
        assert votes == [None if api == "forward_micro_batches" else local_group]


@pytest.mark.parametrize(
    "row",
    SWEEPS,
    ids=lambda r: f"m{r['model']}-{r['operation']}-r{r['lora_rank']}-{r['length']}",
)
def test_unanchored_guard_covers_all_recorded_peaks_including_seven_old_under(row):
    import math

    r = _rank()
    sig = signature((row["model"], row["operation"], row["lora_rank"]))
    observe(r, sig, 128, 1)
    old = row["ART_predicted_increment_bytes"]
    n = row["length"]
    out = n * 4
    # The fixture does not serialize native floor inputs. Reconstruct its
    # recorded OLD demand as a floor, including CP's second output charge.
    floor = math.ceil(old / 1.1) - 2 * out
    args = dict(
        packed_tokens=n, output_bytes=out, signature=sig, checkpoint_floor=(floor, 0)
    )
    effective = r._estimate_required_memory_bytes_from_values(**args)
    assert effective >= math.ceil(old * 1.75)
    assert effective >= row["measured_increment_bytes"]
    # No profile at all must also take the first-unmeasured-shape cold guard.
    r._memory_profiles.clear()
    assert (
        r._estimate_required_memory_bytes_from_values(**args)
        >= row["measured_increment_bytes"]
    )


def test_unanchored_range_boundary_and_anchor_transition():
    r = _rank()
    sig = signature()
    observe(r, sig, 128, 2**30)
    p = r._memory_profiles[sig]
    assert _memory._cold_extrapolation_factor(p, 128) == 1
    assert _memory._cold_extrapolation_factor(p, 129) == 1.75
    ns = [1, 64, 127, 128, 129, 1024, 8191, 8192, 65536]
    prices = [
        r._estimate_required_memory_bytes_from_values(
            packed_tokens=n, output_bytes=0, signature=sig
        )
        for n in ns
    ]
    assert prices == sorted(prices)
    observe(r, sig, 8192, 2 * 2**30)
    anchored = r._memory_profiles[sig]
    assert _memory._cold_extrapolation_factor(anchored, 8192) == 1
    assert _memory._cold_extrapolation_factor(anchored, 8193) == 1.75
    assert _memory._cold_extrapolation_factor(anchored, 65536) == 1.75
    assert _memory._profile_compute_peak(anchored, 128) >= 2**30
    assert _memory._profile_compute_peak(anchored, 8192) >= 2 * 2**30


@pytest.mark.parametrize("anchor_sizes", [(8192,), (8192, 32768)])
@pytest.mark.parametrize("key,rows", list(GROUPS.items()))
def test_boot_anchors_cover_held_out_sweep_cells(anchor_sizes, key, rows):
    r = _rank()
    sig = signature(key)
    anchors = [row for row in rows if row["length"] in anchor_sizes]
    expected = None
    for order in itertools.permutations(anchors):
        r._memory_profiles.clear()
        for row in order:
            observe(r, sig, row["length"], row["measured_increment_bytes"])
        prices = []
        for row in rows:
            n = row["length"]
            effective = r._estimate_required_memory_bytes_from_values(
                packed_tokens=n, output_bytes=0, signature=sig
            )
            assert effective >= row["measured_increment_bytes"]
            prices.append(effective)
        assert prices == sorted(prices)
        if expected is None:
            expected = prices
        assert prices == expected


def test_runtime_growth_only_learns_within_trust_until_anchor(monkeypatch):
    r = _rank()
    sig = signature()
    observe(r, sig, 128, 2**20)
    plan = cast(
        Any,
        SimpleNamespace(
            signature=sig,
            packed_tokens=8192,
            active_logical_tokens=8192,
            output_bytes=0,
        ),
    )
    monkeypatch.setattr(_impl.torch.cuda, "max_memory_allocated", lambda _: 2**30)
    _memory._update_peak_memory_profile(r, plan, 0)
    profile = r._memory_profiles[sig]
    # An out-of-trust forced observation retains its absolute peak, but cannot
    # establish a rate, an anchor or a larger trust window.
    assert profile.absolute_compute_floor == 2**30
    assert profile.rate_observations == ()
    assert profile.compute_slope == (0, 1)
    assert not r._all_ranks_have_memory_profile(signature=sig, packed_tokens=8192)
    for n in [1024, 8192]:
        plan.packed_tokens = plan.active_logical_tokens = n
        _memory._update_peak_memory_profile(r, plan, 0)
    assert r._memory_profiles[sig].compute_slope == (2**30, 8192)


def test_unanchored_coefficient_and_prediction_order_independence():
    points = ((1, 1000), (128, 2000), (2048, 5000), (4096, 6000))
    expected = None
    r = _rank()
    sig = signature()
    for order in itertools.permutations(points):
        r._memory_profiles.clear()
        for n, peak in order:
            observe(r, sig, n, peak, caller=True)
        p = r._memory_profiles[sig]
        state = (
            p.compute_observations,
            p.compute_slope,
            p.fixed_compute_bytes,
            tuple(
                r._estimate_required_memory_bytes_from_values(
                    packed_tokens=n, output_bytes=0, signature=sig
                )
                for n in (1, 4096, 4097, 8192)
            ),
        )
        if expected is None:
            expected = state
        assert state == expected
